"""Epoch-0 sanity for CaLiNet-E.

Contract:
  force_rho_one=True   -> Y_pred = Xt @ W_i + b_i + 0  ==  PCM
                          val_score should equal measured PCM (0.6023) within 0.01.
  force_rho_one=False  -> Y_pred = Xt @ W_eff + b_eff + 0  ~ PCM but slightly lower
                          (because clean rho p50 ~ 0.96, fallback drags toward GL).

If BOTH match expectations, the wrapper is wired correctly and we can train.
If force_rho_one=True deviates > 0.01 from PCM, there is a bug in:
  - calibrate_patient_batch_torch (already sanity-checked, should not be it)
  - W_eff / b_eff arithmetic
  - linear branch matmul orientation
  - backbone zero-init not effective (R_theta != 0 at init)

Usage:
    python scripts/sanity_calinet_e_epoch0.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calinet.config import CaLiNetConfig
from calinet.data.dataset import EpisodicPTBXL
from calinet.data.episodes import resolve_lead_indices
from calinet.data.normalizer import GlobalNormalizer
from calinet.eval.metrics import per_lead_pcc
from calinet.eval.morphology import classify_leads, compute_morphology
from calinet.models.calinet_e import CaLiNetE


def load_cfg(path):
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    for k in ("input_leads", "target_leads", "train_gap_seconds",
              "unet_channels", "film_layers", "aug_noise_snr_db",
              "aug_drift_hz", "aug_amp_scale", "train_folds",
              "ridge_lambda_W_grid", "ridge_lambda_b_grid"):
        if k in d and isinstance(d[k], list):
            d[k] = tuple(d[k])
    return CaLiNetConfig(**d)


@torch.no_grad()
def validate_calinet_e(
    model, loader, normalizer, cfg, recon_idx, target_idx,
    tau_n, tau_m, device, desc="val",
) -> dict:
    model.eval()
    mu    = normalizer.mu[target_idx]
    sigma = normalizer.sigma[target_idx]

    pccs, nrmses, morphs = [], [], []
    r_amp_vals, st_vals, t_amp_vals = [], [], []

    for batch in tqdm(loader, desc=desc, leave=False):
        for k in ("x_calib", "y_calib", "x_test", "y_test"):
            batch[k] = batch[k].to(device).float()
        Yt = batch["y_test"].transpose(1, 2)            # (B, T, n_out)
        Y_pred = model(batch)                            # (B, T, n_out)

        Yt_np = Yt.cpu().numpy()
        Yp_np = Y_pred.cpu().numpy()
        for b in range(Yt_np.shape[0]):
            pcc = per_lead_pcc(Yt_np[b], Yp_np[b])
            pccs.append(float(np.mean(pcc[recon_idx])))
            diff = Yt_np[b, :, recon_idx] - Yp_np[b, :, recon_idx]
            num = float(np.sqrt((diff ** 2).mean()))
            den = float(np.sqrt((Yt_np[b, :, recon_idx] ** 2).mean()))
            nrmses.append(num / max(den, 1e-8))
            Yt_mv = Yt_np[b] * sigma + mu
            Yp_mv = Yp_np[b] * sigma + mu
            rep = compute_morphology(
                Yt_mv, Yp_mv,
                cfg.target_leads, cfg.input_leads, cfg.sampling_rate,
            )
            if rep is None:
                continue
            morph = (rep.r_amp_err_main / 1.0
                     + rep.st_j60_anterior_main / 0.1
                     + rep.t_amp_err_main / 0.3)
            morphs.append(morph)
            r_amp_vals.append(rep.r_amp_err_main)
            st_vals.append(rep.st_j60_anterior_main)
            t_amp_vals.append(rep.t_amp_err_main)

    pcc_m   = float(np.nanmean(pccs))
    nrmse_m = float(np.nanmean(nrmses))
    morph_m = float(np.nanmean(morphs)) if morphs else float("nan")
    val_score = (
        0.3 * pcc_m
        + 0.4 * float(np.exp(-nrmse_m / tau_n))
        + 0.3 * float(np.exp(-morph_m / tau_m))
    )
    return dict(
        val_score=val_score, pcc=pcc_m, nrmse=nrmse_m, morph=morph_m,
        r_amp=float(np.nanmean(r_amp_vals)) if r_amp_vals else float("nan"),
        st60_ant=float(np.nanmean(st_vals)) if st_vals else float("nan"),
        t_amp=float(np.nanmean(t_amp_vals)) if t_amp_vals else float("nan"),
    )


def main():
    cfg = load_cfg("configs/default.yaml")
    artifact_dir = Path(cfg.artifact_dir)

    tz = np.load(artifact_dir / "val_score_tau.npz")
    tau_n = float(tz["tau_n"])
    tau_m = float(tz["tau_m"])
    pcm_score = float(tz["pcm_val_score"])
    gl_score  = float(tz["gl_val_score"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalizer = GlobalNormalizer.load(artifact_dir / "normalizer.npz")

    val_ds = EpisodicPTBXL(
        cfg, split="val", normalizer=normalizer, mode="eval",
        gap_seconds=0.0, seed=cfg.seed,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0,
    )
    _, target_idx = resolve_lead_indices(cfg.input_leads, cfg.target_leads)
    groups = classify_leads(cfg.target_leads, cfg.input_leads)
    recon_idx = [cfg.target_leads.index(n) for n in groups["reconstructed"]]

    print("=" * 78)
    print("CaLiNet-E epoch-0 sanity")
    print("=" * 78)
    print(f"  PCM val_score (target for force_rho=1): {pcm_score:.4f}")
    print(f"  GL  val_score:                          {gl_score:.4f}")
    print()

    # --------------- Test 1: force_rho_one=True == PCM -----------------
    model = CaLiNetE.from_artifacts(
        artifact_dir,
        n_in=len(cfg.input_leads),
        n_out=len(cfg.target_leads),
        channels=cfg.unet_channels,
        embedding_dim=cfg.embedding_dim,
        pad_to_multiple=cfg.pad_to_multiple,
        sampling_rate=cfg.sampling_rate,
        lam_W=cfg.ridge_lambda_W,
        lam_b=cfg.ridge_lambda_b,
        force_rho_one=True,
    ).to(device)
    print(f"  CaLiNet-E params: {model.n_params():,}")

    m1 = validate_calinet_e(
        model, val_loader, normalizer, cfg, recon_idx, target_idx,
        tau_n, tau_m, device, desc="rho=1",
    )
    delta1 = m1["val_score"] - pcm_score
    print("\nTest 1 (force_rho=1, expect ≡ PCM):")
    print(f"  val_score = {m1['val_score']:.4f}  (PCM = {pcm_score:.4f})")
    print(f"  PCC={m1['pcc']:.4f}  nrmse={m1['nrmse']:.4f}  morph={m1['morph']:.4f}")
    print(f"  ST60_ant={m1['st60_ant']:.4f} mV  R_amp={m1['r_amp']:.4f}  T_amp={m1['t_amp']:.4f}")
    print(f"  delta = {delta1:+.4f}  (tol 0.01)  -> "
          f"{'PASS' if abs(delta1) <= 0.01 else 'FAIL'}")

    # --------------- Test 2: real rho ----------------------------------
    model.force_rho_one = False
    m2 = validate_calinet_e(
        model, val_loader, normalizer, cfg, recon_idx, target_idx,
        tau_n, tau_m, device, desc="real rho",
    )
    delta2 = m2["val_score"] - pcm_score
    print("\nTest 2 (real rho, expect <= PCM by < 0.03):")
    print(f"  val_score = {m2['val_score']:.4f}  (PCM = {pcm_score:.4f})")
    print(f"  PCC={m2['pcc']:.4f}  nrmse={m2['nrmse']:.4f}  morph={m2['morph']:.4f}")
    print(f"  ST60_ant={m2['st60_ant']:.4f} mV  R_amp={m2['r_amp']:.4f}  T_amp={m2['t_amp']:.4f}")
    print(f"  delta = {delta2:+.4f}")

    # Pass criteria summary
    print()
    print("-" * 78)
    if abs(delta1) <= 0.01 and -0.05 <= delta2 <= 0.01:
        print("  [PASS] CaLiNet-E wiring verified. Safe to train.")
    else:
        print("  [FAIL] check wiring before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
