"""Reconstruction metrics — numpy implementations, shape-agnostic.

All functions accept (T, n_leads) or (N, T, n_leads) arrays and return
either scalars or per-lead vectors.

Primary metric throughout the project: per-lead Pearson correlation,
reported separately for ALL 12 leads, the 9 withheld leads, and the
6 precordial leads (V1..V6).
"""
from __future__ import annotations

import numpy as np

LEAD_NAMES = (
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
)
PRECORDIAL = ("V1", "V2", "V3", "V4", "V5", "V6")


def _pcc_1d(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.clip((a @ b) / denom, -1.0, 1.0))


def per_lead_pcc(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-lead PCC for one or many samples.

    Parameters
    ----------
    y_true, y_pred : (T, n_leads) or (N, T, n_leads)

    Returns
    -------
    (n_leads,) if 2D input; (N, n_leads) if 3D input.
    """
    if y_true.ndim == 2:
        return np.array([_pcc_1d(y_true[:, l], y_pred[:, l])
                         for l in range(y_true.shape[1])], dtype=np.float32)
    out = np.empty((y_true.shape[0], y_true.shape[2]), dtype=np.float32)
    for n in range(y_true.shape[0]):
        for l in range(y_true.shape[2]):
            out[n, l] = _pcc_1d(y_true[n, :, l], y_pred[n, :, l])
    return out


def per_lead_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-lead RMSE."""
    err = y_true - y_pred
    if y_true.ndim == 2:
        return np.sqrt((err ** 2).mean(axis=0)).astype(np.float32)
    return np.sqrt((err ** 2).mean(axis=1)).astype(np.float32)


def summarize(
    pcc_per_lead: np.ndarray,
    rmse_per_lead: np.ndarray,
    input_lead_idx: list[int],
) -> dict:
    """Aggregate per-lead metrics into the standard report groups.

    Report groups:
      all         - all 12 leads (includes pass-through inputs)
      withheld    - 9 leads not in input (includes Einthoven-trivial limb leads)
      precordial  - V1..V6 (6 leads; may include input V-lead)
      reconstructed - NON-trivial output leads only. Excludes:
                      (a) input leads (pass-through)
                      (b) Einthoven-derived limb leads (III, aVR, aVL, aVF)
                          when BOTH I AND II are inputs.
                      This is the group that actually reflects model quality.

    pcc_per_lead, rmse_per_lead : (N, 12) or (12,)
    input_lead_idx : indices of input leads
    """
    if pcc_per_lead.ndim == 1:
        pcc_per_lead = pcc_per_lead[None]
        rmse_per_lead = rmse_per_lead[None]

    withheld_idx = [i for i in range(12) if i not in input_lead_idx]
    precordial_idx = [LEAD_NAMES.index(l) for l in PRECORDIAL]

    # Einthoven-trivial set: III(2), aVR(3), aVL(4), aVF(5)
    # These are exactly determined when both I(0) and II(1) are available.
    einthoven_trivial = {2, 3, 4, 5} if (0 in input_lead_idx and 1 in input_lead_idx) else set()
    reconstructed_idx = [
        i for i in range(12)
        if i not in input_lead_idx and i not in einthoven_trivial
    ]

    def _agg(arr: np.ndarray, idx: list[int]) -> tuple[float, float]:
        if len(idx) == 0:
            return float("nan"), float("nan")
        vals = arr[:, idx].mean(axis=1)
        return float(vals.mean()), float(vals.std())

    pcc_all_m,           pcc_all_s           = _agg(pcc_per_lead,  list(range(12)))
    pcc_withheld_m,      pcc_withheld_s      = _agg(pcc_per_lead,  withheld_idx)
    pcc_precordial_m,    pcc_precordial_s    = _agg(pcc_per_lead,  precordial_idx)
    pcc_recon_m,         pcc_recon_s         = _agg(pcc_per_lead,  reconstructed_idx)
    rmse_all_m,          rmse_all_s          = _agg(rmse_per_lead, list(range(12)))
    rmse_withheld_m,     rmse_withheld_s     = _agg(rmse_per_lead, withheld_idx)
    rmse_precordial_m,   rmse_precordial_s   = _agg(rmse_per_lead, precordial_idx)
    rmse_recon_m,        rmse_recon_s        = _agg(rmse_per_lead, reconstructed_idx)

    return {
        "pcc_all_mean": pcc_all_m, "pcc_all_std": pcc_all_s,
        "pcc_withheld_mean": pcc_withheld_m, "pcc_withheld_std": pcc_withheld_s,
        "pcc_precordial_mean": pcc_precordial_m, "pcc_precordial_std": pcc_precordial_s,
        "pcc_reconstructed_mean": pcc_recon_m, "pcc_reconstructed_std": pcc_recon_s,
        "rmse_all_mean": rmse_all_m, "rmse_all_std": rmse_all_s,
        "rmse_withheld_mean": rmse_withheld_m, "rmse_withheld_std": rmse_withheld_s,
        "rmse_precordial_mean": rmse_precordial_m, "rmse_precordial_std": rmse_precordial_s,
        "rmse_reconstructed_mean": rmse_recon_m, "rmse_reconstructed_std": rmse_recon_s,
        "reconstructed_lead_names": [LEAD_NAMES[i] for i in reconstructed_idx],
        "pcc_per_lead_mean":  pcc_per_lead.mean(axis=0).tolist(),
        "rmse_per_lead_mean": rmse_per_lead.mean(axis=0).tolist(),
    }
