"""1D U-Net with global-linear anchor (paper: '1D U-Net w/ global-linear anchor').

Locked design (v1.2):
  Y_pred = X · W_global + b_global + R_θ(X)

where R_θ is the shared ResidualUNet backbone with FiLM disabled.
This form means:
  - At init (R_θ ≈ 0 due to zero-init head), model degrades to GL
    (the population-level linear baseline). Epoch 0 val_score ≈ GL.
  - Training only has to learn the NON-LINEAR residual on top of GL,
    not the entire 12-lead signal from scratch.
  - The ablation '1D U-Net w/ anchor = CaLiNet-E without per-patient calibration'
    holds exactly: CaLiNet-E replaces W_global with W_i and adds FiLM,
    nothing else changes.

Usage:
    model = UNetWithAnchor.from_artifacts(cfg, artifact_dir)
    model.train()
    Y_pred = model(Xt)            # (B, T, 12)

Input shapes:
    Xt : (B, T, n_in)   — time-major to match Episode dataclass
Output shapes:
    Y  : (B, T, n_out)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .backbone import ResidualUNet


class UNetWithAnchor(nn.Module):
    def __init__(
        self,
        n_in: int = 3,
        n_out: int = 12,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        embedding_dim: int = 128,
        pad_to_multiple: int = 16,
        W_global: np.ndarray | None = None,
        b_global: np.ndarray | None = None,
    ):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out

        # Linear baseline (frozen buffer)
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

        # Shared backbone (FiLM disabled for 1D U-Net w/ anchor)
        self.backbone = ResidualUNet(
            n_in=n_in,
            n_out=n_out,
            channels=channels,
            embedding_dim=embedding_dim,
            use_film_conditioning=False,
            pad_to_multiple=pad_to_multiple,
        )

    def forward(self, Xt: torch.Tensor) -> torch.Tensor:
        """Xt: (B, T, n_in) → Y_pred: (B, T, n_out)."""
        # Linear branch
        Y_lin = Xt @ self.W_global + self.b_global       # (B, T, 12)

        # Residual branch (Conv1d wants channel-first)
        x = Xt.transpose(1, 2)                            # (B, n_in, T)
        y = self.backbone(x)                              # (B, n_out, T)
        Y_res = y.transpose(1, 2)                         # (B, T, 12)

        return Y_lin + Y_res

    @classmethod
    def from_artifacts(
        cls,
        artifact_dir: str | Path,
        n_in: int = 3,
        n_out: int = 12,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        embedding_dim: int = 128,
        pad_to_multiple: int = 16,
    ) -> "UNetWithAnchor":
        """Load W_global / b_global from artifact_dir/global_W.npz."""
        gw = np.load(Path(artifact_dir) / "global_W.npz")
        return cls(
            n_in=n_in, n_out=n_out, channels=channels,
            embedding_dim=embedding_dim, pad_to_multiple=pad_to_multiple,
            W_global=gw["W_global"], b_global=gw["b_global"],
        )

    def n_params(self) -> int:
        """Trainable parameter count (excludes W_global / b_global buffers)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
