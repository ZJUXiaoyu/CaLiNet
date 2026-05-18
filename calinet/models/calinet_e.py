"""CaLiNet-E: per-patient calibration + non-linear residual with FiLM conditioning.

Forward:
    W_i, b_i        = batched ridge fit on (Xc, Yc) with prior (W_global, b_global)
    rho_i           = calibration_quality(...)['rho']                  (CPU)
    W_eff           = rho * W_i + (1 - rho) * W_global
    b_eff           = rho * b_i + (1 - rho) * b_global
    e_i             = backbone.embed( [W_eff - W_global ; b_eff - b_global] )
    Y_pred          = Xt @ W_eff + b_eff + R_theta(Xt, e_i)

At init (R_theta head zero-init, FiLM gamma=beta=0):
    R_theta(...) ≡ 0
    Y_pred = Xt @ W_eff + b_eff
If rho is forced to 1.0 (sanity mode), this is exactly the PCM baseline.

Device strategy:
  - Ridge fit:  GPU (calibrate_patient_batch_torch)
  - rho:        CPU per-sample (cond + R-peak detection are awkward on GPU)
  - everything else: GPU
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .backbone import ResidualUNet
from .calibration import calibrate_patient_batch_torch
from .rho import RhoConfig, calibration_quality


class CaLiNetE(nn.Module):
    def __init__(
        self,
        n_in: int = 3,
        n_out: int = 12,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        embedding_dim: int = 128,
        pad_to_multiple: int = 16,
        W_global: np.ndarray | None = None,
        b_global: np.ndarray | None = None,
        rho_cfg: RhoConfig | None = None,
        sampling_rate: int = 100,
        rpeak_lead_idx_in_Xc: int = 1,
        lam_W: float = 1.0,
        lam_b: float = 0.1,
        force_rho_one: bool = False,
    ):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.lam_W = float(lam_W)
        self.lam_b = float(lam_b)
        self.sampling_rate = int(sampling_rate)
        self.rpeak_lead_idx_in_Xc = int(rpeak_lead_idx_in_Xc)
        self.rho_cfg = rho_cfg or RhoConfig()
        self.force_rho_one = bool(force_rho_one)

        if W_global is None:
            W_global = np.zeros((n_in, n_out), dtype=np.float32)
        if b_global is None:
            b_global = np.zeros((n_out,), dtype=np.float32)
        self.register_buffer(
            "W_global", torch.from_numpy(W_global.astype(np.float32))
        )
        self.register_buffer(
            "b_global", torch.from_numpy(b_global.astype(np.float32))
        )

        # Shared backbone — FiLM ENABLED (the only structural difference vs 1D U-Net w/ anchor)
        self.backbone = ResidualUNet(
            n_in=n_in,
            n_out=n_out,
            channels=channels,
            embedding_dim=embedding_dim,
            use_film_conditioning=True,
            pad_to_multiple=pad_to_multiple,
        )

        # Marker for shared validate() — CaLiNet-E needs the full batch dict.
        self._takes_full_batch = True

    # ------------------------------------------------------------------
    def _compute_rho_batch(
        self,
        Xc_t: torch.Tensor,           # (B, Lc, n_in)
        Yc_t: torch.Tensor,           # (B, Lc, n_out)
        W_i: torch.Tensor,            # (B, n_in, n_out)
        b_i: torch.Tensor,            # (B, n_out)
    ) -> torch.Tensor:
        """Compute rho per sample on CPU; return (B,) tensor on input device."""
        Xc_np = Xc_t.detach().cpu().numpy()
        Yc_np = Yc_t.detach().cpu().numpy()
        W_np  = W_i.detach().cpu().numpy()
        b_np  = b_i.detach().cpu().numpy()
        rhos = np.empty(Xc_np.shape[0], dtype=np.float32)
        for k in range(Xc_np.shape[0]):
            q = calibration_quality(
                Xc_np[k], Yc_np[k], W_np[k], b_np[k],
                fs=self.sampling_rate,
                cfg=self.rho_cfg,
                rpeak_lead_idx_in_Xc=self.rpeak_lead_idx_in_Xc,
            )
            rhos[k] = q["rho"]
        return torch.from_numpy(rhos).to(device=Xc_t.device, dtype=Xc_t.dtype)

    # ------------------------------------------------------------------
    def forward(self, batch: dict) -> torch.Tensor:
        """batch keys: x_calib (B, n_in, Lc), y_calib (B, n_out, Lc),
        x_test (B, n_in, Lt). Returns Y_pred (B, Lt, n_out).
        """
        Xc = batch["x_calib"]
        Yc = batch["y_calib"]
        Xt = batch["x_test"]

        Xc_t = Xc.transpose(1, 2).float()        # (B, Lc, n_in)
        Yc_t = Yc.transpose(1, 2).float()        # (B, Lc, n_out)
        Xt_t = Xt.transpose(1, 2).float()        # (B, Lt, n_in)

        # 1. Batched ridge on GPU
        with torch.no_grad():
            W_i, b_i = calibrate_patient_batch_torch(
                Xc_t, Yc_t, self.W_global, self.b_global,
                lam_W=self.lam_W, lam_b=self.lam_b,
            )
            # 2. rho on CPU
            if self.force_rho_one:
                rho = torch.ones(
                    Xc_t.shape[0], device=Xc_t.device, dtype=Xc_t.dtype,
                )
            else:
                rho = self._compute_rho_batch(Xc_t, Yc_t, W_i, b_i)

            # 3. Soft fallback (GPU)
            r3 = rho.view(-1, 1, 1)
            r2 = rho.view(-1, 1)
            W_eff = r3 * W_i + (1.0 - r3) * self.W_global       # (B, n_in, n_out)
            b_eff = r2 * b_i + (1.0 - r2) * self.b_global       # (B, n_out)

        # 4. Linear branch (batched matmul)
        Y_lin = torch.bmm(Xt_t, W_eff) + b_eff.unsqueeze(1)     # (B, Lt, n_out)

        # 5. FiLM embedding from delta to global
        delta_W = (W_eff - self.W_global).flatten(1)             # (B, n_in*n_out)
        delta_b = b_eff - self.b_global                          # (B, n_out)
        e_i = self.backbone.embed(
            torch.cat([delta_W, delta_b], dim=1)
        )                                                        # (B, embedding_dim)

        # 6. Residual branch
        Y_res = self.backbone(Xt.float(), e_i=e_i)               # (B, n_out, Lt)
        Y_res = Y_res.transpose(1, 2)                            # (B, Lt, n_out)

        return Y_lin + Y_res

    # ------------------------------------------------------------------
    @classmethod
    def from_artifacts(
        cls,
        artifact_dir: str | Path,
        n_in: int = 3,
        n_out: int = 12,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        embedding_dim: int = 128,
        pad_to_multiple: int = 16,
        sampling_rate: int = 100,
        rpeak_lead_idx_in_Xc: int = 1,
        lam_W: float = 1.0,
        lam_b: float = 0.1,
        force_rho_one: bool = False,
    ) -> "CaLiNetE":
        """Load W_global / b_global and rho_config from artifact_dir."""
        artifact_dir = Path(artifact_dir)
        gw = np.load(artifact_dir / "global_W.npz")
        rho_cfg = RhoConfig()
        rc_path = artifact_dir / "rho_config.npz"
        if rc_path.exists():
            z = np.load(rc_path)
            rho_cfg = RhoConfig(
                w_cond=float(z["w_cond"]),
                w_fit=float(z["w_fit"]),
                w_beat=float(z["w_beat"]),
                cond_center=float(z["cond_center"]),
                cond_scale=float(z["cond_scale"]),
                fit_center=float(z["fit_center"]),
                fit_scale=float(z["fit_scale"]),
                expected_hr_bpm=float(z["expected_hr_bpm"]),
                ectopic_threshold=float(z["ectopic_threshold"]),
            )
        return cls(
            n_in=n_in, n_out=n_out, channels=channels,
            embedding_dim=embedding_dim, pad_to_multiple=pad_to_multiple,
            W_global=gw["W_global"], b_global=gw["b_global"],
            rho_cfg=rho_cfg,
            sampling_rate=sampling_rate,
            rpeak_lead_idx_in_Xc=rpeak_lead_idx_in_Xc,
            lam_W=lam_W, lam_b=lam_b,
            force_rho_one=force_rho_one,
        )

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
