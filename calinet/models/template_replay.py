"""TemplateReplay baseline.

Algorithm (corrected):
  1. Detect R-peaks in the calibration segment Yc (use lead II).
  2. Segment 12-lead beats around each R-peak with a fixed window.
  3. Compute the median beat template across beats (robust to outliers).
  4. Detect R-peaks in the test segment's input lead.
  5. Place the FIXED-LENGTH template at each test R-peak WITHOUT warping;
     QRS duration must stay ~80-120 ms regardless of RR (any warping of
     QRS is non-physiological).
  6. Where consecutive templates OVERLAP, use Hann-window overlap-add so
     boundaries are smooth.
  7. Where consecutive templates leave a GAP (TP segment), linearly
     interpolate between the tail of one template and the head of the
     next — this is physiologically reasonable (isoelectric TP segment
     with mild drift toward the next P wave).

This baseline has no notion of lead geometry — it just replays the
calibration beat template at every R-peak detected in the test segment.
It is the floor CaLiNet must beat.

Output is in the same (normalized) space as Yc.
"""
from __future__ import annotations

import numpy as np

from ..eval.rpeak import detect_rpeaks


def _segment_beats(
    sig: np.ndarray,
    rpeaks: np.ndarray,
    before: int,
    after: int,
) -> np.ndarray | None:
    """Extract fixed-length beats around each R-peak.

    Returns (n_beats, T_beat, n_leads) or None if no valid beat.
    """
    T = sig.shape[0]
    beats = []
    for r in rpeaks:
        lo, hi = r - before, r + after
        if lo < 0 or hi > T:
            continue
        beats.append(sig[lo:hi])
    if not beats:
        return None
    return np.stack(beats, axis=0)


def _hann_weights(n: int) -> np.ndarray:
    """Hann window used to taper template edges for overlap-add."""
    if n <= 1:
        return np.ones(n, dtype=np.float64)
    # Cosine-tapered (Hann) window, symmetric
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))


def template_replay(
    Xc_target: np.ndarray,          # (Lc, n_target) full 12-lead calibration
    Xt_input:  np.ndarray,          # (Lt, n_in)    input leads in test
    Yt_shape:  tuple[int, int],     # (Lt, n_target) output shape
    fs: int,
    input_leads: tuple[str, ...],
    target_leads: tuple[str, ...],
    beat_before_ms: float = 300.0,
    beat_after_ms:  float = 500.0,
    default_hr_bpm: float = 75.0,
) -> np.ndarray:
    """Reconstruct the test segment by replaying a fixed-length beat template.

    Parameters
    ----------
    Xc_target : full 12-lead calibration segment (used to build the template)
    Xt_input  : 3-lead test segment (used only for R-peak detection)
    Yt_shape  : desired output shape matching the true 12-lead test segment
    fs        : sampling rate
    input_leads, target_leads : lead name tuples
    beat_before_ms, beat_after_ms : template window around each R-peak
    default_hr_bpm : fallback HR when no R-peaks can be detected
    """
    Lt, n_target = Yt_shape
    before = int(beat_before_ms * 1e-3 * fs)
    after  = int(beat_after_ms  * 1e-3 * fs)
    T_beat = before + after

    # --- 1. R-peak detection leads ---------------------------------------
    calib_rpeak_lead = target_leads.index("II") if "II" in target_leads else 0
    if "II" in input_leads:
        test_rpeak_lead = input_leads.index("II")
    elif "I" in input_leads:
        test_rpeak_lead = input_leads.index("I")
    else:
        test_rpeak_lead = 0

    # --- 2. Build template from calibration ------------------------------
    rpeaks_c = detect_rpeaks(Xc_target[:, calib_rpeak_lead], fs=fs)
    beats = _segment_beats(Xc_target, rpeaks_c, before, after)
    if beats is None or len(beats) == 0:
        # No usable calibration beats: return flatline at calibration DC
        dc = Xc_target.mean(axis=0, keepdims=True)
        return np.broadcast_to(dc, Yt_shape).astype(np.float32).copy()

    template = np.median(beats, axis=0).astype(np.float64)  # (T_beat, n_target)

    # --- 3. Detect test R-peaks ------------------------------------------
    rpeaks_t = detect_rpeaks(Xt_input[:, test_rpeak_lead], fs=fs)
    default_rr = int(fs * 60.0 / default_hr_bpm)
    if len(rpeaks_t) == 0:
        # No R-peaks — tile template at default RR, centered on before-offset
        rpeaks_t = np.arange(before, Lt, default_rr, dtype=np.int64)

    # --- 4. Overlap-add placement with Hann weights ----------------------
    # Hann weights only on the edges so the QRS (center) is preserved.
    # We taper only the outer 25% of the template to avoid flattening QRS.
    taper_frac = 0.25
    taper_n = int(taper_frac * T_beat)
    edge_w = _hann_weights(2 * taper_n + 1)        # symmetric, length 2*taper_n+1
    # Build full-length weight vector: 1 in the middle, tapered at ends
    w = np.ones(T_beat, dtype=np.float64)
    w[:taper_n] = edge_w[:taper_n]                 # rising edge
    w[-taper_n:] = edge_w[-taper_n:]               # falling edge

    numer = np.zeros((Lt, n_target), dtype=np.float64)
    denom = np.zeros((Lt, 1),        dtype=np.float64)

    for r in rpeaks_t:
        lo = int(r - before)
        hi = int(lo + T_beat)
        a_lo = max(0, lo)
        a_hi = min(Lt, hi)
        if a_hi <= a_lo:
            continue
        t_lo = a_lo - lo
        t_hi = t_lo + (a_hi - a_lo)
        numer[a_lo:a_hi] += template[t_lo:t_hi] * w[t_lo:t_hi, None]
        denom[a_lo:a_hi] += w[t_lo:t_hi, None]

    # Where templates cover the signal, divide by weight sum
    covered = denom.squeeze(-1) > 1e-8
    out = np.zeros_like(numer)
    out[covered] = numer[covered] / denom[covered]

    # --- 5. Fill gaps (TP segments) by linear interpolation --------------
    # Find runs of uncovered samples and interpolate lead-wise between
    # the last-covered value and the next-covered value.
    if (~covered).any():
        idx = np.arange(Lt)
        xp = idx[covered]
        if len(xp) == 0:
            # Nothing covered at all — fall back to template DC
            out[:] = template.mean(axis=0, keepdims=True)
        else:
            for l in range(n_target):
                out[~covered, l] = np.interp(
                    idx[~covered], xp, out[covered, l],
                )

    return out.astype(np.float32)
