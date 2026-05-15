"""R-peak detection using Pan-Tompkins-style processing.

Lightweight pure-numpy/scipy implementation. If neurokit2 is installed it
can optionally be used, but we keep the dependency optional.

Inputs are signals in either raw mV or normalized units — the algorithm
is amplitude-independent after the adaptive thresholding step.

Reference: Pan, J. & Tompkins, W. (1985). "A real-time QRS detection
algorithm." IEEE Trans Biomed Eng.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks


def _pantompkins_preprocess(ecg_1d: np.ndarray, fs: int) -> np.ndarray:
    """Bandpass 5-15 Hz -> derivative -> squaring -> moving average integration."""
    nyq = 0.5 * fs
    sos = butter(2, [5.0 / nyq, 15.0 / nyq], btype="bandpass", output="sos")
    x = sosfiltfilt(sos, ecg_1d)

    # 5-point derivative (fs/8 units)
    x = np.convolve(x, np.array([1, 2, 0, -2, -1]) / 8.0, mode="same")
    x = x ** 2

    # moving-window integration, window ~150 ms
    win = max(int(0.150 * fs), 1)
    kernel = np.ones(win) / win
    x = np.convolve(x, kernel, mode="same")
    return x


def detect_rpeaks(
    ecg_1d: np.ndarray,
    fs: int,
    min_rr_ms: float = 300.0,
    min_prominence_ratio: float = 0.3,
) -> np.ndarray:
    """Detect R-peak sample indices in a 1-D ECG channel.

    Parameters
    ----------
    ecg_1d : (T,) signal (lead II or any prominent lead)
    fs : sampling rate
    min_rr_ms : minimum allowed RR interval in milliseconds (~200 bpm cap)
    min_prominence_ratio : peak prominence threshold, as fraction of
        median of the Pan-Tompkins envelope

    Returns
    -------
    (n_peaks,) int array of R-peak sample indices in the ORIGINAL signal.
    """
    if ecg_1d.ndim != 1:
        raise ValueError("detect_rpeaks expects 1D input")

    env = _pantompkins_preprocess(ecg_1d, fs)
    distance = int(min_rr_ms * 1e-3 * fs)
    prominence = float(min_prominence_ratio * np.median(np.abs(env)) + 1e-8)

    peaks_env, _ = find_peaks(env, distance=distance, prominence=prominence)

    # Refine: shift each peak to the local max of |ecg| within a short window
    refined = []
    radius = max(int(0.040 * fs), 1)   # 40 ms search radius
    for p in peaks_env:
        lo = max(0, p - radius)
        hi = min(len(ecg_1d), p + radius + 1)
        local = ecg_1d[lo:hi]
        if local.size == 0:
            continue
        refined.append(lo + int(np.argmax(np.abs(local))))

    return np.array(refined, dtype=np.int64)


def rpeak_rates(rpeaks: np.ndarray, fs: int) -> dict:
    """Compute summary stats from an R-peak index array.

    Returns dict with mean RR (samples), mean HR (bpm), RR std, ectopic
    fraction (|RR_i - median| > 15%).
    """
    if len(rpeaks) < 2:
        return {"n": len(rpeaks), "mean_rr": np.nan, "mean_hr": np.nan,
                "rr_std": np.nan, "ectopic_frac": np.nan}
    rr = np.diff(rpeaks).astype(np.float64)
    med = np.median(rr)
    return {
        "n": len(rpeaks),
        "mean_rr": float(rr.mean()),
        "mean_hr": float(60.0 * fs / rr.mean()),
        "rr_std":  float(rr.std()),
        "ectopic_frac": float(np.mean(np.abs(rr - med) > 0.15 * med)),
    }
