"""Calibration quality score rho_i and soft fallback.

rho_i ∈ [0, 1] quantifies how trustworthy the per-patient ridge fit is.

rho_i = w_cond · s_cond  +  w_fit · s_fit  +  w_beat · s_beat

where the sub-scores use a LOGISTIC shape (v1.2 update — replaced the
earlier exponential form because it decayed too fast on clean data and
pulled rho down below 0.5 in the baseline case):

  s_cond = σ((cond_center - log10(cond)) / cond_scale)
  s_fit  = σ((fit_center  - nrmse)       / fit_scale)
  s_beat = valid_ratio · (1 - ectopic_ratio)    # unchanged

`cond_center, cond_scale, fit_center, fit_scale` are calibrated from a
sample of clean records via calibrate_rho_normalization() in
scripts/05_diagnose_rho.py.

Soft fallback:
  W_eff = rho_i * W_i + (1 - rho_i) * W_global
  b_eff = rho_i * b_i + (1 - rho_i) * b_global

SANITY requirements (v1.2):
  clean     : rho median ≥ 0.90 and 95-percentile ≥ 0.85
  PVC/5dB   : rho median ≤ 0.50
Without these, fallback either degrades clean predictions or fails to
rescue poorly-calibrated ones.

All inputs in NORMALIZED space (same as calibrate_patient).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..eval.rpeak import detect_rpeaks


@dataclass
class RhoConfig:
    """Hyperparameters for calibration quality score.

    cond_center / cond_scale operate in log10(cond_number) space.
    fit_center  / fit_scale  operate on raw nrmse.

    Defaults are spec placeholders; the real values come from
    calibrate_rho_normalization() on clean training/val data.

    Weights (v1.2 update):
      w_cond=0.3, w_fit=0.6, w_beat=0.1 — reduced beat weight because
      2s calibration windows don't support reliable ectopic-beat
      estimation (insufficient RR-interval samples). s_beat is
      restricted to valid_beat_ratio only.
    """
    w_cond: float = 0.3
    w_fit:  float = 0.6
    w_beat: float = 0.1

    # Logistic shape params (calibrated from clean-data percentiles)
    cond_center: float = 3.5       # log10(cond) at which s_cond = 0.5
    cond_scale:  float = 0.5       # log10 units
    fit_center:  float = 0.25      # nrmse at which s_fit = 0.5
    fit_scale:   float = 0.05      # nrmse units

    # Beat quality
    expected_hr_bpm:   float = 75.0
    ectopic_threshold: float = 0.15   # kept for diagnostic report, NOT used in s_beat


def _sigmoid(x: float) -> float:
    # numerically safe
    if x >= 0:
        z = np.exp(-x)
        return float(1.0 / (1.0 + z))
    z = np.exp(x)
    return float(z / (1.0 + z))


# ---------------------------------------------------------------
# Raw measurements (unnormalized)
# ---------------------------------------------------------------
def compute_cond_number(Xc: np.ndarray) -> float:
    """Condition number of X_c^T X_c + tiny ridge for stability."""
    G = Xc.astype(np.float64).T @ Xc.astype(np.float64)
    G = G + 1e-8 * np.eye(G.shape[0])
    cond = float(np.linalg.cond(G))
    return cond if np.isfinite(cond) and cond > 1 else 1.0


def compute_nrmse(
    Xc: np.ndarray, Yc: np.ndarray,
    W_i: np.ndarray, b_i: np.ndarray,
) -> float:
    """Normalized ridge-fit residual on calibration: ||Y - XW - b|| / ||Y||."""
    resid = Yc - (Xc @ W_i + b_i)
    num = float(np.linalg.norm(resid))
    den = float(np.linalg.norm(Yc))
    return num / den if den >= 1e-8 else float("inf")


def compute_beat_quality(
    Xc: np.ndarray, fs: int,
    expected_hr_bpm: float,
    ectopic_threshold: float,
    rpeak_lead: int = 1,
) -> tuple[float, float]:
    """Return (valid_ratio, ectopic_ratio) from R-peaks on Xc[:, rpeak_lead].

    NOTE (v1.2): ectopic_ratio is COMPUTED but NOT used in s_beat.
    With 2s calibration windows there are only 2-3 R-peaks, so the
    ectopic_ratio estimate has no statistical support and spuriously
    saturates at 1.0 for a large fraction of clean records. We keep the
    number in the diagnostic output for transparency, but s_beat in
    the score is reduced to `valid_ratio` (below).
    """
    if rpeak_lead >= Xc.shape[1]:
        rpeak_lead = 0
    peaks = detect_rpeaks(Xc[:, rpeak_lead], fs=fs)
    if len(peaks) < 2:
        return 0.0, 1.0
    duration_s = Xc.shape[0] / fs
    expected = duration_s * expected_hr_bpm / 60.0
    valid_ratio = float(min(1.0, len(peaks) / max(1.0, expected)))
    rr = np.diff(peaks)
    med = float(np.median(rr))
    if med <= 0:
        return valid_ratio, 1.0
    ectopic = float(np.mean(np.abs(rr - med) > ectopic_threshold * med))
    return valid_ratio, ectopic


# ---------------------------------------------------------------
# Sub-scores (logistic normalization)
# ---------------------------------------------------------------
def _s_cond(cond_number: float, cfg: RhoConfig) -> float:
    log_c = np.log10(max(cond_number, 1.0))
    return _sigmoid((cfg.cond_center - log_c) / cfg.cond_scale)


def _s_fit(nrmse: float, cfg: RhoConfig) -> float:
    return _sigmoid((cfg.fit_center - nrmse) / cfg.fit_scale)


def _s_beat(valid_ratio: float, ectopic_ratio: float) -> float:
    """s_beat = valid_ratio  (v1.2: dropped ectopic term).

    Rationale: 2s calibration windows have 2-3 R-peaks -> at most 2
    RR intervals -> the ectopic_ratio estimate has no statistical
    support and spuriously triggers on clean records. Keeping only
    valid_ratio preserves the "signal acquisition failed" failure mode
    (electrodes unplugged, flat signal, saturation -> 0 R-peaks) while
    not punishing normal beat-to-beat RR variation.
    """
    return float(np.clip(valid_ratio, 0.0, 1.0))


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------
def calibration_quality(
    Xc:  np.ndarray,
    Yc:  np.ndarray,
    W_i: np.ndarray,
    b_i: np.ndarray,
    fs:  int,
    cfg: RhoConfig | None = None,
    rpeak_lead_idx_in_Xc: int = 1,
) -> dict:
    """Compute rho_i and all sub-components for one record."""
    cfg = cfg or RhoConfig()

    cond = compute_cond_number(Xc)
    nrmse = compute_nrmse(Xc, Yc, W_i, b_i)
    valid_ratio, ectopic_ratio = compute_beat_quality(
        Xc, fs, cfg.expected_hr_bpm, cfg.ectopic_threshold,
        rpeak_lead=rpeak_lead_idx_in_Xc,
    )

    s_cond = _s_cond(cond, cfg)
    s_fit  = _s_fit(nrmse, cfg)
    s_beat = _s_beat(valid_ratio, ectopic_ratio)
    rho = cfg.w_cond * s_cond + cfg.w_fit * s_fit + cfg.w_beat * s_beat

    return {
        "rho":    float(np.clip(rho, 0.0, 1.0)),
        "s_cond": s_cond,
        "s_fit":  s_fit,
        "s_beat": s_beat,
        # raw diagnostic quantities
        "cond_number":      float(cond),
        "nrmse":            float(nrmse),
        "valid_beat_ratio": float(valid_ratio),
        "ectopic_ratio":    float(ectopic_ratio),
    }


def soft_fallback(
    W_i:      np.ndarray,
    b_i:      np.ndarray,
    W_global: np.ndarray,
    b_global: np.ndarray,
    rho:      float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate between patient-specific and population-level transforms."""
    rho = float(np.clip(rho, 0.0, 1.0))
    W_eff = (rho * W_i        + (1.0 - rho) * W_global).astype(np.float32)
    b_eff = (rho * b_i.ravel() + (1.0 - rho) * b_global.ravel()).astype(np.float32)
    return W_eff, b_eff


# ---------------------------------------------------------------
# Normalization calibration (from clean-data percentiles)
# ---------------------------------------------------------------
def calibrate_rho_normalization(
    clean_records: list[dict],
    fit_p50_target: float = 0.95,
    fit_p95_target: float = 0.70,
    default_cfg: "RhoConfig | None" = None,
) -> dict:
    """Auto-fit logistic (center, scale) for s_cond and s_fit from clean data.

    v1.2 update — spread guard:
      If a quantity's empirical p95 - p50 spread is too small to
      meaningfully parameterize a logistic (distribution is saturated),
      we keep the default center/scale instead of compressing an already
      well-behaved distribution. The guard thresholds are:
        nrmse spread       < 0.03          -> keep default fit params
        log10(cond) spread < 0.5 decades   -> keep default cond params

    Targets (v1.2 relaxed):
      s_fit(p50) = fit_p50_target   (default 0.95 — clean should mostly saturate)
      s_fit(p95) = fit_p95_target   (default 0.70 — clean tail still trusted)

    Parameters
    ----------
    clean_records    : list of dicts with keys 'nrmse', 'cond_number'
    fit_p50_target   : target s-score at the 50th percentile of clean nrmse
    fit_p95_target   : target s-score at the 95th percentile of clean nrmse
    default_cfg      : fallback RhoConfig to use when spread is too small
    """
    def _logit(p: float) -> float:
        p = float(np.clip(p, 1e-6, 1 - 1e-6))
        return float(np.log(p / (1 - p)))

    default_cfg = default_cfg or RhoConfig()

    nrmses = np.array([r["nrmse"] for r in clean_records], dtype=np.float64)
    conds  = np.array([r["cond_number"] for r in clean_records], dtype=np.float64)
    log_conds = np.log10(np.clip(conds, 1.0, None))

    nrmse_p50 = float(np.percentile(nrmses, 50))
    nrmse_p95 = float(np.percentile(nrmses, 95))
    lc_p50    = float(np.percentile(log_conds, 50))
    lc_p95    = float(np.percentile(log_conds, 95))

    lg_p50 = _logit(fit_p50_target)
    lg_p95 = _logit(fit_p95_target)
    denom = lg_p50 - lg_p95

    # Spread guard for nrmse → s_fit
    nrmse_spread = nrmse_p95 - nrmse_p50
    if nrmse_spread > 0.03 and abs(denom) > 1e-6:
        fit_scale  = max((nrmse_p95 - nrmse_p50) / denom, 1e-4)
        fit_center = nrmse_p50 + fit_scale * lg_p50
        fit_calibrated = True
    else:
        fit_center = default_cfg.fit_center
        fit_scale  = default_cfg.fit_scale
        fit_calibrated = False

    # Spread guard for log10(cond) → s_cond
    log_cond_spread = lc_p95 - lc_p50
    if log_cond_spread > 0.5 and abs(denom) > 1e-6:
        cond_scale  = max((lc_p95 - lc_p50) / denom, 1e-4)
        cond_center = lc_p50 + cond_scale * lg_p50
        cond_calibrated = True
    else:
        cond_center = default_cfg.cond_center
        cond_scale  = default_cfg.cond_scale
        cond_calibrated = False

    return {
        "fit_center":  float(fit_center),
        "fit_scale":   float(fit_scale),
        "cond_center": float(cond_center),
        "cond_scale":  float(cond_scale),
        "fit_calibrated":  fit_calibrated,
        "cond_calibrated": cond_calibrated,
        # diagnostics
        "nrmse_p50":    nrmse_p50,
        "nrmse_p95":    nrmse_p95,
        "log_cond_p50": lc_p50,
        "log_cond_p95": lc_p95,
        "n_clean":      len(clean_records),
    }
