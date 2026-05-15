"""Morphology metrics for ECG reconstruction.

Locked design rules (v1.2):
  1. Fiducials (Q onset, J point, T peak, T offset) are detected on
     Y_TRUE lead II only. Same indices used to window Y_pred.
  2. All metrics reported in PHYSICAL mV. Caller passes mV-space signals.
  3. **Lead three-way classification** (locked):
        INPUT_LEADS         = pass-through, error should be ~0 trivially
        DERIVABLE_LEADS     = Einthoven / Goldberger linear combinations
                              of inputs (III, aVR, aVL, aVF when both
                              I and II are inputs). Error should be
                              small-but-nonzero if W_global fit is sane
                              (< ~0.05 mV); used as SANITY check.
        RECONSTRUCTED_LEADS = genuinely unknown leads. This is what the
                              paper's main table reports.
  4. Main table: RECONSTRUCTED_LEADS only.
     Sanity table: DERIVABLE_LEADS only.
     Supplementary table: all 12 leads.
  5. Anatomical regions intersected with reconstructed set:
        anterior_recon = V1, V3, V4   (V2 is input  → excluded)
        lateral_recon  = V5, V6       (I is input, aVL is derivable → excluded)
        inferior_recon = {}           (II input, III/aVF derivable)
     → We do NOT report 'ST_inferior' on reconstructed leads.
  6. STEMI diagnostic threshold = 0.1 mV ST elevation. Any ST error
     ≥ ~0.05 mV is not clinically safe.
  7. Abnormal beats are NOT excluded from the main table.

All inputs are (T, 12) numpy arrays in mV. target_leads tuple fixes order.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fiducials import BeatFiducials, detect_all_fiducials
from .rpeak import detect_rpeaks

# Full anatomical regions (used when intersected with a lead subset)
_ANTERIOR_FULL = ("V1", "V2", "V3", "V4")
_LATERAL_FULL  = ("I",  "aVL", "V5", "V6")
_INFERIOR_FULL = ("II", "III", "aVF")


# ──────────────────────────────────────────────────────────────────────
# Lead classification
# ──────────────────────────────────────────────────────────────────────
def classify_leads(
    target_leads: tuple[str, ...],
    input_leads:  tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    """Split target_leads into input / derivable / reconstructed.

    Derivable := {III, aVR, aVL, aVF} iff BOTH I and II are inputs.
    Otherwise only the precordial duplicates (V-leads that are inputs)
    count as non-reconstructed.
    """
    inputs_set = set(input_leads)
    derivable_set: set[str] = set()
    if "I" in inputs_set and "II" in inputs_set:
        derivable_set = {"III", "aVR", "aVL", "aVF"} & set(target_leads)

    input_leads_in_target = tuple(n for n in target_leads if n in inputs_set)
    derivable_leads = tuple(n for n in target_leads if n in derivable_set)
    reconstructed_leads = tuple(
        n for n in target_leads
        if n not in inputs_set and n not in derivable_set
    )
    return {
        "input":         input_leads_in_target,
        "derivable":     derivable_leads,
        "reconstructed": reconstructed_leads,
    }


def _lead_idx(target_leads: tuple[str, ...], names) -> list[int]:
    return [target_leads.index(n) for n in names if n in target_leads]


# ──────────────────────────────────────────────────────────────────────
# Composite morphology score (PCM-anchored)
# ──────────────────────────────────────────────────────────────────────
# Single source of truth shared by anchor, training, and evaluation
# scripts — prevents drift between the definition used to compute
# tau_m and the one tracked during training.
#
#   morph_err_norm = R_amp/1.0 + ST60_anterior/0.1 + T_amp/0.3
#
# All terms in mV. Divisors reflect clinically meaningful scales:
#   R amplitude  ~ 1.0 mV  (typical precordial range)
#   ST deviation ~ 0.1 mV  (STEMI threshold)
#   T amplitude  ~ 0.3 mV  (typical T-wave height)
def morph_err_norm(report) -> float:
    """Composite PCM-anchored morphology score (dimensionless).

    Accepts a MorphologyReport or None.  Returns NaN when report is None
    or any required field is missing.
    """
    if report is None:
        return float("nan")
    try:
        return (report.r_amp_err_main / 1.0
                + report.st_j60_anterior_main / 0.1
                + report.t_amp_err_main / 0.3)
    except AttributeError:
        return float("nan")


# ──────────────────────────────────────────────────────────────────────
# Core amplitude / deviation helpers
# ──────────────────────────────────────────────────────────────────────
def _baseline_mv(
    ecg_mv: np.ndarray,
    fid: BeatFiducials,
    lead_idx: int,
    window_ms: float = 40.0,
    fs: int = 500,
) -> float:
    """Baseline = mean of a short isoelectric segment just before Q onset."""
    w = int(window_ms * 1e-3 * fs)
    lo = max(0, fid.q_on - w)
    hi = fid.q_on
    if hi <= lo:
        return 0.0
    return float(ecg_mv[lo:hi, lead_idx].mean())


def _r_amp_err(y_t, y_p, fids, leads, fs):
    errs = []
    for f in fids:
        for l in leads:
            bt = _baseline_mv(y_t, f, l, fs=fs)
            bp = _baseline_mv(y_p, f, l, fs=fs)
            errs.append(abs((y_t[f.r, l] - bt) - (y_p[f.r, l] - bp)))
    return float(np.mean(errs)) if errs else float("nan")


def _t_amp_err(y_t, y_p, fids, leads, fs):
    errs = []
    for f in fids:
        for l in leads:
            bt = _baseline_mv(y_t, f, l, fs=fs)
            bp = _baseline_mv(y_p, f, l, fs=fs)
            errs.append(abs((y_t[f.t_peak, l] - bt) - (y_p[f.t_peak, l] - bp)))
    return float(np.mean(errs)) if errs else float("nan")


def _st_err(y_t, y_p, fids, leads, offset_ms, fs):
    off = int(offset_ms * 1e-3 * fs)
    errs = []
    for f in fids:
        s = min(f.j + off, y_t.shape[0] - 1)
        for l in leads:
            bt = _baseline_mv(y_t, f, l, fs=fs)
            bp = _baseline_mv(y_p, f, l, fs=fs)
            errs.append(abs((y_t[s, l] - bt) - (y_p[s, l] - bp)))
    return float(np.mean(errs)) if errs else float("nan")


# ──────────────────────────────────────────────────────────────────────
# Dataclass — three-way report
# ──────────────────────────────────────────────────────────────────────
@dataclass
class MorphologyReport:
    """Per-record morphology metrics in mV (errors) / ms (durations).

    Fields are named by LEAD GROUP first so main/sanity/supp separation
    stays explicit:

      *_main      = reconstructed leads only (paper main table)
      *_sanity    = derivable leads only (Einthoven check on W_global)
      *_all12     = all 12 leads (supplementary)
    """
    # --- Main: reconstructed leads ---------------------------------
    r_amp_err_main:   float
    t_amp_err_main:   float
    st_j60_anterior_main: float   # V1, V3, V4
    st_j60_lateral_main:  float   # V5, V6
    st_j80_anterior_main: float
    st_j80_lateral_main:  float
    # Main inferior is explicitly NaN (no reconstructed leads in that region).
    st_j60_inferior_main: float
    st_j80_inferior_main: float

    # --- Sanity: derivable leads (III, aVR, aVL, aVF) -------------
    r_amp_err_sanity:  float
    t_amp_err_sanity:  float
    st_j60_sanity:     float
    st_j80_sanity:     float

    # --- Supp: all 12 leads ---------------------------------------
    r_amp_err_all12:  float
    t_amp_err_all12:  float
    st_j60_all12:     float
    st_j80_all12:     float

    # --- Group B: reference timing from Y_true fiducials ----------
    qrs_duration_true_ms: float
    qt_interval_true_ms:  float
    n_beats: int


# ──────────────────────────────────────────────────────────────────────
# Public compute + aggregate
# ──────────────────────────────────────────────────────────────────────
def compute_morphology(
    y_true_mv:    np.ndarray,
    y_pred_mv:    np.ndarray,
    target_leads: tuple[str, ...],
    input_leads:  tuple[str, ...],
    fs:           int,
) -> MorphologyReport | None:
    """Compute morphology errors for one record.

    Three report groups are filled: main (reconstructed), sanity
    (derivable), all12 (everything). Fiducials from Y_true lead II only.
    """
    if "II" not in target_leads:
        raise ValueError("target_leads must contain 'II'.")
    lead_II = target_leads.index("II")

    rpeaks = detect_rpeaks(y_true_mv[:, lead_II], fs=fs)
    if len(rpeaks) == 0:
        return None
    fids = detect_all_fiducials(y_true_mv[:, lead_II], rpeaks, fs=fs)
    if len(fids) == 0:
        return None

    groups = classify_leads(target_leads, input_leads)
    recon = groups["reconstructed"]
    deriv = groups["derivable"]

    recon_set = set(recon)
    ant_recon = [target_leads.index(n) for n in _ANTERIOR_FULL if n in recon_set]
    lat_recon = [target_leads.index(n) for n in _LATERAL_FULL  if n in recon_set]
    inf_recon = [target_leads.index(n) for n in _INFERIOR_FULL if n in recon_set]

    recon_idx  = _lead_idx(target_leads, recon)
    deriv_idx  = _lead_idx(target_leads, deriv)
    all_idx    = list(range(y_true_mv.shape[1]))

    # --- Main: reconstructed ----------------------------------------
    r_main  = _r_amp_err(y_true_mv, y_pred_mv, fids, recon_idx, fs) if recon_idx else float("nan")
    t_main  = _t_amp_err(y_true_mv, y_pred_mv, fids, recon_idx, fs) if recon_idx else float("nan")
    st60_ant = _st_err(y_true_mv, y_pred_mv, fids, ant_recon, 60.0, fs) if ant_recon else float("nan")
    st60_lat = _st_err(y_true_mv, y_pred_mv, fids, lat_recon, 60.0, fs) if lat_recon else float("nan")
    st60_inf = _st_err(y_true_mv, y_pred_mv, fids, inf_recon, 60.0, fs) if inf_recon else float("nan")
    st80_ant = _st_err(y_true_mv, y_pred_mv, fids, ant_recon, 80.0, fs) if ant_recon else float("nan")
    st80_lat = _st_err(y_true_mv, y_pred_mv, fids, lat_recon, 80.0, fs) if lat_recon else float("nan")
    st80_inf = _st_err(y_true_mv, y_pred_mv, fids, inf_recon, 80.0, fs) if inf_recon else float("nan")

    # --- Sanity: derivable ----------------------------------------
    if deriv_idx:
        r_san  = _r_amp_err(y_true_mv, y_pred_mv, fids, deriv_idx, fs)
        t_san  = _t_amp_err(y_true_mv, y_pred_mv, fids, deriv_idx, fs)
        s60_s  = _st_err(y_true_mv, y_pred_mv, fids, deriv_idx, 60.0, fs)
        s80_s  = _st_err(y_true_mv, y_pred_mv, fids, deriv_idx, 80.0, fs)
    else:
        r_san = t_san = s60_s = s80_s = float("nan")

    # --- Supp: all 12 ---------------------------------------------
    r_all   = _r_amp_err(y_true_mv, y_pred_mv, fids, all_idx, fs)
    t_all   = _t_amp_err(y_true_mv, y_pred_mv, fids, all_idx, fs)
    s60_a   = _st_err(y_true_mv, y_pred_mv, fids, all_idx, 60.0, fs)
    s80_a   = _st_err(y_true_mv, y_pred_mv, fids, all_idx, 80.0, fs)

    # --- Group B ---------------------------------------------------
    qrs_ms = float(np.mean([(f.s - f.q_on) * 1000.0 / fs for f in fids]))
    qt_ms  = float(np.mean([(f.t_off - f.q_on) * 1000.0 / fs for f in fids]))

    return MorphologyReport(
        r_amp_err_main=r_main, t_amp_err_main=t_main,
        st_j60_anterior_main=st60_ant, st_j60_lateral_main=st60_lat,
        st_j80_anterior_main=st80_ant, st_j80_lateral_main=st80_lat,
        st_j60_inferior_main=st60_inf, st_j80_inferior_main=st80_inf,
        r_amp_err_sanity=r_san, t_amp_err_sanity=t_san,
        st_j60_sanity=s60_s, st_j80_sanity=s80_s,
        r_amp_err_all12=r_all, t_amp_err_all12=t_all,
        st_j60_all12=s60_a, st_j80_all12=s80_a,
        qrs_duration_true_ms=qrs_ms, qt_interval_true_ms=qt_ms,
        n_beats=len(fids),
    )


def aggregate_morphology(reports: list[MorphologyReport]) -> dict:
    """Mean / std each field across records, NaN-safe."""
    if not reports:
        return {}
    all_fields = [f.name for f in MorphologyReport.__dataclass_fields__.values()]
    scalar_fields = [f for f in all_fields if f != "n_beats"]
    out: dict = {}
    for k in scalar_fields:
        vals = np.array([getattr(r, k) for r in reports], dtype=np.float64)
        vals = vals[~np.isnan(vals)]
        out[f"{k}_mean"] = float(vals.mean()) if vals.size else float("nan")
        out[f"{k}_std"]  = float(vals.std())  if vals.size else float("nan")
    out["n_records"] = len(reports)
    out["total_beats"] = int(sum(r.n_beats for r in reports))
    return out
