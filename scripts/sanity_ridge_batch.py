"""Sanity check: calibrate_patient_batch_torch vs numpy calibrate_patient.

Must pass before writing CaLiNet-E forward — a hidden bug in the batched
ridge would silently corrupt every downstream metric.

Checks:
  1. Synthetic random (Xc, Yc): batch result matches per-sample numpy result
     to ~1e-5 max-abs (float32 round-trip tolerance).
  2. Real PTB-XL val episodes: same comparison, on actual normalized data.
  3. Both CPU and CUDA paths (if available).

Usage:
    python scripts/sanity_ridge_batch.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calinet.config import CaLiNetConfig
from calinet.data.episodes import extract_episode, resolve_lead_indices
from calinet.data.normalizer import GlobalNormalizer
from calinet.data.ptbxl import get_split_metadata, iter_records, load_metadata
from calinet.models.calibration import (
    calibrate_patient,
    calibrate_patient_batch_torch,
)


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


def _compare(
    Xc_np: np.ndarray, Yc_np: np.ndarray,
    W_global: np.ndarray, b_global: np.ndarray,
    lam_W: float, lam_b: float,
    device: torch.device,
    label: str,
) -> tuple[float, float]:
    """Return (max_abs_diff_W, max_abs_diff_b)."""
    B = Xc_np.shape[0]
    # numpy reference (per-sample)
    W_ref, b_ref = [], []
    for i in range(B):
        Wi, bi = calibrate_patient(Xc_np[i], Yc_np[i], W_global, b_global,
                                   lam_W=lam_W, lam_b=lam_b)
        W_ref.append(Wi); b_ref.append(bi)
    W_ref = np.stack(W_ref)
    b_ref = np.stack(b_ref)

    # torch batched
    Xc_t = torch.from_numpy(Xc_np).to(device).float()
    Yc_t = torch.from_numpy(Yc_np).to(device).float()
    Wg_t = torch.from_numpy(W_global).to(device).float()
    bg_t = torch.from_numpy(b_global).to(device).float()
    W_b, b_b = calibrate_patient_batch_torch(
        Xc_t, Yc_t, Wg_t, bg_t, lam_W=lam_W, lam_b=lam_b,
    )
    W_b_np = W_b.detach().cpu().numpy()
    b_b_np = b_b.detach().cpu().numpy()

    dW = float(np.max(np.abs(W_ref - W_b_np)))
    db = float(np.max(np.abs(b_ref - b_b_np)))
    print(f"  [{label:18s}]  device={str(device):5s}  B={B:3d}  "
          f"max|dW|={dW:.2e}  max|db|={db:.2e}")
    return dW, db


def synthetic_test(device: torch.device, lam_W: float, lam_b: float):
    """Random Xc, Yc with planted W_global."""
    rng = np.random.default_rng(42)
    B, Lc, n_in, n_out = 16, 200, 3, 12
    W_global = rng.standard_normal((n_in, n_out)).astype(np.float32) * 0.3
    b_global = rng.standard_normal(n_out).astype(np.float32) * 0.05
    Xc = rng.standard_normal((B, Lc, n_in)).astype(np.float32)
    Yc = (Xc @ W_global + b_global
          + rng.standard_normal((B, Lc, n_out)).astype(np.float32) * 0.2)
    return _compare(Xc, Yc, W_global, b_global, lam_W, lam_b, device, "synthetic")


def real_ptbxl_test(
    cfg: CaLiNetConfig, device: torch.device,
    lam_W: float, lam_b: float, n_records: int = 16,
):
    """Stack real (Xc, Yc) episodes from val fold."""
    artifact_dir = Path(cfg.artifact_dir)
    normalizer = GlobalNormalizer.load(artifact_dir / "normalizer.npz")
    gw = np.load(artifact_dir / "global_W.npz")
    W_global = gw["W_global"].astype(np.float32)
    b_global = gw["b_global"].astype(np.float32)

    meta = load_metadata(cfg.ptbxl_path)
    split_meta = get_split_metadata(meta, cfg.val_fold).iloc[:n_records]
    in_idx, target_idx = resolve_lead_indices(cfg.input_leads, cfg.target_leads)
    filter_kwargs = {
        "low_hz": cfg.bandpass_low_hz, "high_hz": cfg.bandpass_high_hz,
        "fs": cfg.sampling_rate, "order": cfg.bandpass_order,
    } if cfg.use_bandpass else None

    Xc_list, Yc_list = [], []
    for _id, sig, _row in iter_records(
        cfg.ptbxl_path, split_meta, cfg.sampling_rate, filter_kwargs,
    ):
        ep = extract_episode(
            sig, in_idx, target_idx,
            calib_samples=cfg.calib_samples,
            target_samples=cfg.target_samples,
            gap_samples=0, calib_start=0, normalizer=normalizer,
        )
        if ep is None:
            continue
        Xc_list.append(ep.Xc.astype(np.float32))
        Yc_list.append(ep.Yc.astype(np.float32))

    Xc = np.stack(Xc_list)
    Yc = np.stack(Yc_list)
    return _compare(Xc, Yc, W_global, b_global, lam_W, lam_b, device, "ptbxl_val")


def main():
    cfg = load_cfg("configs/default.yaml")
    lam_W, lam_b = cfg.ridge_lambda_W, cfg.ridge_lambda_b

    print("=" * 78)
    print("Sanity: calibrate_patient_batch_torch vs numpy calibrate_patient")
    print(f"  lam_W={lam_W}  lam_b={lam_b}")
    print("=" * 78)

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    TOL = 1e-4   # float32 round-trip + matmul order differences
    all_pass = True
    for dev in devices:
        dW, db = synthetic_test(dev, lam_W, lam_b)
        if dW > TOL or db > TOL:
            all_pass = False
        dW, db = real_ptbxl_test(cfg, dev, lam_W, lam_b)
        if dW > TOL or db > TOL:
            all_pass = False

    print("-" * 78)
    if all_pass:
        print(f"  [PASS]  all comparisons within tol={TOL:.0e}")
    else:
        print(f"  [FAIL]  some diff exceeds tol={TOL:.0e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
