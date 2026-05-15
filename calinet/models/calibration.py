"""Calibration module: ridge-regularized least squares with prior on global.

Core equations (v1.0 spec, Section 3):

  Global fit (one-time, on training set):
      min_W,b   ||Y - X W - 1 b^T||_F^2
              + lambda_W ||W||_F^2
              + lambda_b ||b||_2^2

  Per-patient fit (run online for each ECG):
      min_W,b   ||Yc - Xc W - 1 b^T||_F^2
              + lambda_W ||W - W_global||_F^2
              + lambda_b ||b - b_global||_2^2

  Augmented form (with bias as 4th row):
      X_aug = [Xc | 1]                shape (Lc, 4)
      W_aug = [W; b^T]                shape (4, n_target_leads)
      Reg   = diag(lambda_W, lambda_W, lambda_W, lambda_b)
      Prior = [lambda_W * W_global ; lambda_b * b_global^T]

      W_aug = solve( X_aug^T X_aug + Reg ,  X_aug^T Yc + Prior )

Operates entirely in numpy (np.float64 for the linear-algebra solve, then
cast back to float32). All inputs are NORMALIZED ECG.

Inputs / outputs:
  Xc : (Lc, n_in)        normalized 3-lead calibration ECG
  Yc : (Lc, n_target=12) normalized 12-lead calibration target
  W  : (n_in, 12)
  b  : (12,)
"""
from __future__ import annotations

import numpy as np


def _build_reg_matrices(
    n_in: int,
    n_target: int,
    W_prior: np.ndarray,
    b_prior: np.ndarray,
    lam_W: float,
    lam_b: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (Reg, Prior_aug) used by the augmented normal equations."""
    diag = np.array([lam_W] * n_in + [lam_b], dtype=np.float64)   # (n_in+1,)
    Reg = np.diag(diag)                                           # (n_in+1, n_in+1)

    Prior_aug = np.empty((n_in + 1, n_target), dtype=np.float64)
    Prior_aug[:n_in] = lam_W * W_prior.astype(np.float64)
    Prior_aug[n_in:] = lam_b * b_prior.astype(np.float64)         # row vector
    return Reg, Prior_aug


def _solve_augmented(
    Xc: np.ndarray,
    Yc: np.ndarray,
    Reg: np.ndarray,
    Prior_aug: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the augmented ridge system. Returns (W, b)."""
    Lc, n_in = Xc.shape
    Xc_aug = np.empty((Lc, n_in + 1), dtype=np.float64)
    Xc_aug[:, :n_in] = Xc
    Xc_aug[:, n_in] = 1.0

    A = Xc_aug.T @ Xc_aug + Reg                # (n_in+1, n_in+1)
    B = Xc_aug.T @ Yc.astype(np.float64) + Prior_aug   # (n_in+1, 12)
    W_aug = np.linalg.solve(A, B)

    W = W_aug[:n_in].astype(np.float32)        # (n_in, 12)
    b = W_aug[n_in].astype(np.float32)         # (12,)
    return W, b


def fit_global_transform(
    X_all: np.ndarray,
    Y_all: np.ndarray,
    lam_W: float = 1.0,
    lam_b: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit (W_global, b_global) on all training segments stacked together.

    Solved as ridge regression with zero prior:
      min ||Y - XW - 1 b^T||^2 + lam_W ||W||^2 + lam_b ||b||^2

    Parameters
    ----------
    X_all : (N, n_in)        stacked 3-lead samples (normalized)
    Y_all : (N, n_target=12) stacked 12-lead samples (normalized)
    """
    n_in = X_all.shape[1]
    n_target = Y_all.shape[1]
    W_prior = np.zeros((n_in, n_target), dtype=np.float64)
    b_prior = np.zeros((n_target,),      dtype=np.float64)
    Reg, Prior_aug = _build_reg_matrices(
        n_in, n_target, W_prior, b_prior, lam_W, lam_b
    )
    return _solve_augmented(X_all, Y_all, Reg, Prior_aug)


def calibrate_patient(
    Xc: np.ndarray,
    Yc: np.ndarray,
    W_global: np.ndarray,
    b_global: np.ndarray,
    lam_W: float = 1.0,
    lam_b: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-patient ridge fit with prior on (W_global, b_global).

    Parameters
    ----------
    Xc : (Lc, n_in)        normalized calibration input
    Yc : (Lc, n_target=12) normalized calibration target
    W_global, b_global : population-level prior
    lam_W, lam_b : decoupled regularization strengths

    Returns
    -------
    (W_i, b_i) of shape ((n_in, 12), (12,)).
    """
    n_in = Xc.shape[1]
    n_target = Yc.shape[1]
    Reg, Prior_aug = _build_reg_matrices(
        n_in, n_target, W_global, b_global, lam_W, lam_b
    )
    return _solve_augmented(Xc, Yc, Reg, Prior_aug)


def apply_transform(
    X: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Linear forward: Y = X @ W + b. Returns (T, 12) float32."""
    return (X.astype(np.float32) @ W + b).astype(np.float32)


# ---------------------------------------------------------------------------
# Torch GPU-vectorized ridge (used by CaLiNet-E)
# ---------------------------------------------------------------------------
try:
    import torch
except ImportError:                                 # pragma: no cover
    torch = None


def calibrate_patient_batch_torch(
    Xc: "torch.Tensor",
    Yc: "torch.Tensor",
    W_global: "torch.Tensor",
    b_global: "torch.Tensor",
    lam_W: float = 1.0,
    lam_b: float = 0.1,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Batched per-patient ridge fit, fully on the input device.

    Mirrors :func:`calibrate_patient` numerically — solves the same augmented
    normal equations in float64 and casts back to the input dtype. Use this
    inside CaLiNet-E.forward to avoid CPU<->GPU round-trips.

    Parameters
    ----------
    Xc : (B, Lc, n_in)        normalized calibration input
    Yc : (B, Lc, n_target)    normalized calibration target
    W_global : (n_in, n_target)
    b_global : (n_target,)

    Returns
    -------
    W_i : (B, n_in, n_target)
    b_i : (B, n_target)
    """
    if torch is None:
        raise RuntimeError("torch not available")
    B, Lc, n_in = Xc.shape
    n_target = Yc.shape[-1]
    device = Xc.device
    out_dtype = Xc.dtype

    Xc64 = Xc.to(torch.float64)
    Yc64 = Yc.to(torch.float64)
    Wg64 = W_global.to(torch.float64)
    bg64 = b_global.to(torch.float64)

    ones = torch.ones(B, Lc, 1, device=device, dtype=torch.float64)
    Xc_aug = torch.cat([Xc64, ones], dim=-1)                  # (B, Lc, n_in+1)

    diag = torch.empty(n_in + 1, device=device, dtype=torch.float64)
    diag[:n_in] = lam_W
    diag[n_in]  = lam_b
    Reg = torch.diag(diag)                                    # (n_in+1, n_in+1)

    Prior_aug = torch.empty(
        n_in + 1, n_target, device=device, dtype=torch.float64
    )
    Prior_aug[:n_in] = lam_W * Wg64
    Prior_aug[n_in:] = lam_b * bg64                           # broadcasts to (1, n_target)

    A  = Xc_aug.transpose(1, 2) @ Xc_aug + Reg                # (B, n_in+1, n_in+1)
    Bm = Xc_aug.transpose(1, 2) @ Yc64  + Prior_aug           # (B, n_in+1, n_target)

    W_aug = torch.linalg.solve(A, Bm)                         # (B, n_in+1, n_target)
    W_i = W_aug[:, :n_in].to(out_dtype)
    b_i = W_aug[:,  n_in].to(out_dtype)
    return W_i, b_i
