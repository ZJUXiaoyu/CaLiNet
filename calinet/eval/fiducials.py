"""Fiducial point detection for morphology evaluation.

Design rule (locked): fiducials are detected on Y_TRUE only, then the
same indices are used to window both Y_true and Y_pred. This removes
detector-error contamination from morphology metrics.

Detection is deliberately simple (not clinical-grade) but consistent:
  - R peak     : from Pan-Tompkins (calinet.eval.rpeak)
  - Q onset    : local zero-crossing of first derivative before R, within
                 60 ms window
  - S / J point: local zero-crossing of first derivative after R, within
                 100 ms window; J point is taken as the local minimum
                 after S returning toward baseline (fixed 40 ms after S)
  - T peak     : largest absolute deflection in 100-400 ms after R
  - T offset   : return-to-baseline after T peak (within 200 ms after
                 T peak); defined as the point where |derivative| first
                 falls below 10% of the peak derivative magnitude

All functions operate per beat and return arrays indexed by beat number.
Sample indices are relative to the full signal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BeatFiducials:
    """Sample indices (ints) for one beat, all relative to the full signal."""
    r:     int
    q_on:  int
    s:     int
    j:     int
    t_peak:   int
    t_off:    int


def _safe_slice(n: int, lo: int, hi: int) -> tuple[int, int]:
    return max(0, lo), min(n, hi)


def _derivative(x: np.ndarray) -> np.ndarray:
    """Central-difference derivative, same length as x."""
    d = np.zeros_like(x)
    d[1:-1] = (x[2:] - x[:-2]) / 2.0
    d[0]    = x[1]  - x[0]
    d[-1]   = x[-1] - x[-2]
    return d


def detect_beat_fiducials(
    lead_ii: np.ndarray,
    r_peak:  int,
    fs:      int,
    qrs_pre_ms:   float = 60.0,
    qrs_post_ms:  float = 100.0,
    j_offset_ms:  float = 40.0,
    t_search_ms:  tuple[float, float] = (100.0, 400.0),
    t_off_max_ms: float = 200.0,
) -> BeatFiducials | None:
    """Detect Q onset, S, J, T peak, T offset for one beat on lead II.

    Parameters
    ----------
    lead_ii : 1-D lead II signal (the whole record)
    r_peak  : index of this beat's R peak
    fs      : sampling rate

    Returns
    -------
    BeatFiducials, or None if the beat is too close to signal edge.
    """
    n = len(lead_ii)
    pre  = int(qrs_pre_ms  * 1e-3 * fs)
    post = int(qrs_post_ms * 1e-3 * fs)
    j_off = int(j_offset_ms * 1e-3 * fs)
    t_lo = int(t_search_ms[0] * 1e-3 * fs)
    t_hi = int(t_search_ms[1] * 1e-3 * fs)
    t_off_lim = int(t_off_max_ms * 1e-3 * fs)

    if r_peak - pre < 0 or r_peak + post + j_off + t_hi + t_off_lim >= n:
        return None

    # Derivative of lead II
    dsig = _derivative(lead_ii)

    # Q onset: closest zero-crossing of dsig going upward within [R-pre, R]
    # Approximation: find local minimum of lead_ii before R within the pre
    # window (Q wave is the negative deflection just before R).
    pre_lo, pre_hi = _safe_slice(n, r_peak - pre, r_peak)
    if pre_hi <= pre_lo:
        return None
    q_on = pre_lo + int(np.argmin(lead_ii[pre_lo:pre_hi]))

    # S: local minimum of lead_ii after R within post window
    post_lo, post_hi = _safe_slice(n, r_peak + 1, r_peak + 1 + post)
    if post_hi <= post_lo:
        return None
    s = post_lo + int(np.argmin(lead_ii[post_lo:post_hi]))

    # J point: fixed offset after S (typical electrocardiography convention)
    j = min(s + j_off, n - 1)

    # T peak: max absolute deflection in (R + t_lo, R + t_hi)
    tp_lo, tp_hi = _safe_slice(n, r_peak + t_lo, r_peak + t_hi)
    if tp_hi <= tp_lo:
        return None
    # Use signed value — we take whichever sign dominates
    seg = lead_ii[tp_lo:tp_hi]
    t_peak = tp_lo + int(np.argmax(np.abs(seg - seg.mean())))

    # T offset: point after T peak where |derivative| first drops below
    # 10% of the peak derivative magnitude within this search window.
    to_lo, to_hi = _safe_slice(n, t_peak + 1, t_peak + 1 + t_off_lim)
    if to_hi <= to_lo:
        t_off = min(t_peak + t_off_lim, n - 1)
    else:
        dseg = np.abs(dsig[to_lo:to_hi])
        if dseg.size == 0 or dseg.max() < 1e-8:
            t_off = min(t_peak + t_off_lim, n - 1)
        else:
            thr = 0.1 * dseg.max()
            # First index where derivative falls below threshold
            below = np.where(dseg < thr)[0]
            t_off = to_lo + int(below[0]) if len(below) > 0 else to_hi - 1

    return BeatFiducials(r=r_peak, q_on=q_on, s=s, j=j,
                         t_peak=t_peak, t_off=t_off)


def detect_all_fiducials(
    lead_ii: np.ndarray,
    rpeaks:  np.ndarray,
    fs:      int,
) -> list[BeatFiducials]:
    """Run fiducial detection for every R peak; drop beats too close to edges."""
    out = []
    for r in rpeaks:
        f = detect_beat_fiducials(lead_ii, int(r), fs)
        if f is not None:
            out.append(f)
    return out
