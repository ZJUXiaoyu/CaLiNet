"""E12: Calibration perturbation suite for PCM failure-mode study.

Each perturbation takes the normalized 12-lead calibration segment Yc
(shape: (Lc, 12)) and returns a perturbed copy. Perturbations can also
be applied to the 3-lead input Xc (for the 'shared' mode).

Perturbation catalog:
  - gaussian_noise  : additive Gaussian at specified SNR (dB)
  - baseline_drift  : low-frequency sinusoidal drift at specified amplitude
  - inject_pvc      : replace k beats with amplitude-distorted beats
  - electrode_scale : per-lead random multiplicative scale (mimics electrode
                      impedance / contact variation)

All operate in NORMALIZED space. Caller passes normalizer.invert() /
apply() for operations that need mV units (e.g. drift with physical amp).

Reproducible — pass a np.random.Generator for each perturbation call.
"""
from __future__ import annotations

import numpy as np

from ..eval.rpeak import detect_rpeaks


# ----------------------------------------------------------------------
# Perturbations on Yc (and optionally Xc via same transformation)
# ----------------------------------------------------------------------
def gaussian_noise(
    sig: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Additive Gaussian noise at specified SNR (dB)."""
    if not np.isfinite(snr_db) or snr_db > 100:
        return sig.copy()
    # Per-lead power
    p_signal = sig.var(axis=0, keepdims=True)  # (1, n_leads)
    p_noise = p_signal / (10 ** (snr_db / 10.0))
    sigma = np.sqrt(np.clip(p_noise, 1e-12, None))
    noise = rng.standard_normal(sig.shape) * sigma
    return (sig + noise).astype(np.float32)


def baseline_drift(
    sig: np.ndarray,
    fs: int,
    amp_norm: float,
    freq_hz: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add low-frequency sinusoidal drift. amp_norm in normalized units
    (≈ amp_mV / mean sigma across leads). Phase randomized per lead.
    """
    T = sig.shape[0]
    t = np.arange(T) / fs
    out = sig.copy()
    for l in range(sig.shape[1]):
        phase = rng.uniform(0, 2 * np.pi)
        out[:, l] += amp_norm * np.sin(2 * np.pi * freq_hz * t + phase)
    return out.astype(np.float32)


def inject_pvc(
    sig: np.ndarray,
    fs: int,
    n_pvc: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Replace n_pvc beats with wide, amplitude-inverted PVC-like morphology.

    Crude model: find R-peaks on lead II-like column (index 1 if shape allows),
    and for n_pvc randomly chosen beats, replace the surrounding window
    (pre=150ms, post=350ms) by a negated, broadened template.
    """
    if n_pvc <= 0 or sig.shape[0] < fs:
        return sig.copy()
    rpeak_lead = 1 if sig.shape[1] > 1 else 0
    peaks = detect_rpeaks(sig[:, rpeak_lead], fs=fs)
    if len(peaks) <= n_pvc:
        return sig.copy()

    chosen = rng.choice(len(peaks), size=n_pvc, replace=False)
    pre = int(0.15 * fs)
    post = int(0.35 * fs)
    out = sig.copy()
    for idx in chosen:
        r = int(peaks[idx])
        lo = max(0, r - pre)
        hi = min(sig.shape[0], r + post)
        # Build crude PVC: negate, widen via low-pass, mild amplitude boost
        beat = out[lo:hi].copy()
        # low-pass via boxcar (≈40 ms window)
        w = max(int(0.04 * fs), 1)
        kernel = np.ones(w, dtype=np.float32) / w
        for l in range(beat.shape[1]):
            beat[:, l] = np.convolve(beat[:, l], kernel, mode="same")
        # negate and amplify
        beat = -1.5 * beat
        out[lo:hi] = beat.astype(np.float32)
    return out


def electrode_scale(
    sig: np.ndarray,
    max_scale_offset: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Per-lead multiplicative scale ~ Uniform(1 - d, 1 + d)."""
    if max_scale_offset <= 0:
        return sig.copy()
    scales = rng.uniform(
        1.0 - max_scale_offset, 1.0 + max_scale_offset,
        size=(1, sig.shape[1]),
    ).astype(np.float32)
    return (sig * scales).astype(np.float32)


# ----------------------------------------------------------------------
# Catalog: (name, level_list, applier)
# ----------------------------------------------------------------------
def perturbation_catalog(fs: int) -> dict:
    """Return a dict describing each perturbation family with its level grid.

    Every entry is:
        {
          "levels": [human-readable level values],
          "key":    str label for table rows,
          "apply":  fn(sig, level, rng) -> perturbed sig,
        }
    """
    return {
        "clean": {
            "levels": [0],
            "key":    "clean",
            "apply":  lambda sig, lvl, rng: sig.copy(),
        },
        "gaussian_noise": {
            "levels": [40, 30, 20, 15, 10, 5],    # dB SNR (lower = worse)
            "key":    "SNR_dB",
            "apply":  lambda sig, lvl, rng: gaussian_noise(sig, lvl, rng),
        },
        "baseline_drift": {
            "levels": [0.05, 0.1, 0.2, 0.3, 0.5],  # amplitude in normalized units
            "key":    "drift_amp",
            "apply":  lambda sig, lvl, rng: baseline_drift(
                sig, fs=fs, amp_norm=lvl, freq_hz=0.3, rng=rng),
        },
        "inject_pvc": {
            "levels": [0, 1, 2],
            "key":    "n_pvc",
            "apply":  lambda sig, lvl, rng: inject_pvc(sig, fs=fs, n_pvc=lvl, rng=rng),
        },
        "electrode_scale": {
            "levels": [0.0, 0.1, 0.2, 0.3],
            "key":    "scale_offset",
            "apply":  lambda sig, lvl, rng: electrode_scale(sig, lvl, rng),
        },
    }
