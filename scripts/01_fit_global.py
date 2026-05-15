"""Step 1 of v1.0 implementation order:

  1. Load PTB-XL training folds.
  2. Filter (high-pass 0.5 Hz baseline removal).
  3. Fit GlobalNormalizer (per-lead mu, sigma) on training set.
  4. Fit (W_global, b_global) via ridge regression on normalized signals.
  5. Evaluate W_global on the validation fold and report PCC / RMSE
     grouped by all / withheld / precordial leads.

Expected baseline (from MGNet runs on same lead set): PCC_precordial in
[0.55, 0.70]. If significantly outside this range, something is wrong
with preprocessing or normalization (M2).

Usage:
    python scripts/01_fit_global.py
    python scripts/01_fit_global.py --config configs/default.yaml
    python scripts/01_fit_global.py --max_train_records 2000   # smoke test

Artifacts written to artifact_dir:
    normalizer.npz   GlobalNormalizer state (mu, sigma)
    global_W.npz     {W_global, b_global, lam_W, lam_b}
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calinet.config import CaLiNetConfig
from calinet.data.normalizer import GlobalNormalizer, fit_global_normalizer
from calinet.data.ptbxl import (
    PTBXL_LEAD_ORDER,
    get_split_metadata,
    iter_records,
    load_metadata,
)
from calinet.eval.metrics import per_lead_pcc, per_lead_rmse, summarize
from calinet.models.calibration import apply_transform, fit_global_transform


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--max_train_records", type=int, default=None,
                   help="Cap training records (smoke test). None = all.")
    p.add_argument("--max_val_records", type=int, default=None,
                   help="Cap validation records.")
    return p.parse_args()


def load_cfg(path: str) -> CaLiNetConfig:
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    # Tuple-ify list fields that the dataclass declares as Tuple
    for k in (
        "input_leads", "target_leads", "train_gap_seconds",
        "unet_channels", "film_layers", "aug_noise_snr_db",
        "aug_drift_hz", "aug_amp_scale", "train_folds",
        "ridge_lambda_W_grid", "ridge_lambda_b_grid",
    ):
        if k in d and isinstance(d[k], list):
            d[k] = tuple(d[k])
    return CaLiNetConfig(**d)


def collect_signals(
    ptbxl_path: str,
    metadata: pd.DataFrame,
    cfg: CaLiNetConfig,
    desc: str,
    cap: int | None = None,
) -> list[np.ndarray]:
    """Iterate metadata, return list of (T, 12) filtered signals in mV."""
    filter_kwargs = {
        "low_hz": cfg.bandpass_low_hz,
        "high_hz": cfg.bandpass_high_hz,
        "fs": cfg.sampling_rate,
        "order": cfg.bandpass_order,
    } if cfg.use_bandpass else None

    signals = []
    iterator = iter_records(
        ptbxl_path, metadata,
        sampling_rate=cfg.sampling_rate,
        filter_kwargs=filter_kwargs,
    )
    for i, (_ecg_id, sig, _row) in enumerate(
        tqdm(iterator, total=len(metadata), desc=desc, leave=False)
    ):
        signals.append(sig)
        if cap is not None and len(signals) >= cap:
            break
    return signals


def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    np.random.seed(cfg.seed)

    artifact_dir = Path(cfg.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Step 1 - Fit normalizer + global ridge transform")
    print("=" * 72)
    print(f"PTB-XL path:  {cfg.ptbxl_path}")
    print(f"Input leads:  {cfg.input_leads}")
    print(f"Target leads: {cfg.target_leads}")
    print(f"Sampling:     {cfg.sampling_rate} Hz")
    print(f"Bandpass:     low={cfg.bandpass_low_hz}, high={cfg.bandpass_high_hz}")
    print(f"Ridge:        lam_W={cfg.ridge_lambda_W}, lam_b={cfg.ridge_lambda_b}")
    print()

    # 1. Metadata --------------------------------------------------------
    t0 = time.time()
    print("[1/5] loading metadata...")
    meta = load_metadata(cfg.ptbxl_path)
    train_meta = get_split_metadata(meta, cfg.train_folds)
    val_meta   = get_split_metadata(meta, cfg.val_fold)
    if args.max_train_records:
        train_meta = train_meta.iloc[: args.max_train_records]
    if args.max_val_records:
        val_meta = val_meta.iloc[: args.max_val_records]
    print(f"  train records: {len(train_meta)}")
    print(f"  val records:   {len(val_meta)}")

    # 2. Train signals + normalizer --------------------------------------
    print("[2/5] loading + filtering training signals...")
    train_signals = collect_signals(
        cfg.ptbxl_path, train_meta, cfg, desc="train",
        cap=args.max_train_records,
    )
    print(f"  loaded {len(train_signals)} train records ({time.time()-t0:.1f}s)")

    print("[3/5] fitting GlobalNormalizer (per-lead mu, sigma)...")
    normalizer = fit_global_normalizer(train_signals)
    np_path = artifact_dir / "normalizer.npz"
    normalizer.save(np_path)
    print(f"  mu:    {np.round(normalizer.mu, 4)}")
    print(f"  sigma: {np.round(normalizer.sigma, 4)}")
    print(f"  saved → {np_path}")

    # 3. Build (X_all, Y_all) for global ridge ---------------------------
    print("[4/5] fitting global ridge (W_global, b_global)...")
    in_idx = [PTBXL_LEAD_ORDER.index(l) for l in cfg.input_leads]
    target_idx = [PTBXL_LEAD_ORDER.index(l) for l in cfg.target_leads]

    # Stack & normalize in one pass to bound memory
    X_chunks, Y_chunks = [], []
    for sig in tqdm(train_signals, desc="stack", leave=False):
        sig_n = normalizer(sig)                    # (T, 12)
        X_chunks.append(sig_n[:, in_idx])
        Y_chunks.append(sig_n[:, target_idx])
    X_all = np.concatenate(X_chunks, axis=0)
    Y_all = np.concatenate(Y_chunks, axis=0)
    print(f"  stacked  X_all={X_all.shape}, Y_all={Y_all.shape}")

    W_global, b_global = fit_global_transform(
        X_all, Y_all,
        lam_W=cfg.ridge_lambda_W,
        lam_b=cfg.ridge_lambda_b,
    )
    print(f"  W_global shape={W_global.shape}, ‖W‖_F={np.linalg.norm(W_global):.4f}")
    print(f"  b_global={np.round(b_global, 4)}")

    w_path = artifact_dir / "global_W.npz"
    np.savez(
        w_path,
        W_global=W_global, b_global=b_global,
        lam_W=cfg.ridge_lambda_W, lam_b=cfg.ridge_lambda_b,
        input_leads=np.array(cfg.input_leads),
        target_leads=np.array(cfg.target_leads),
    )
    print(f"  saved → {w_path}")

    # Free memory before validation
    del X_chunks, Y_chunks, X_all, Y_all, train_signals

    # 4. Validation evaluation -------------------------------------------
    print("[5/5] evaluating W_global on validation fold...")
    val_signals = collect_signals(
        cfg.ptbxl_path, val_meta, cfg, desc="val",
        cap=args.max_val_records,
    )

    pccs, rmses = [], []
    for sig in tqdm(val_signals, desc="eval ", leave=False):
        sig_n = normalizer(sig)
        X = sig_n[:, in_idx]
        Y = sig_n[:, target_idx]
        Y_hat = apply_transform(X, W_global, b_global)
        pccs.append(per_lead_pcc(Y, Y_hat))
        rmses.append(per_lead_rmse(Y, Y_hat))
    pcc_per_lead  = np.stack(pccs)               # (N_val, 12)
    rmse_per_lead = np.stack(rmses)              # (N_val, 12)

    summary = summarize(
        pcc_per_lead, rmse_per_lead,
        input_lead_idx=cfg.input_lead_idx,
    )

    print()
    print("=" * 72)
    print("Validation results - global ridge baseline")
    print("=" * 72)
    print(f"  PCC  all 12        : {summary['pcc_all_mean']:.4f} +/- {summary['pcc_all_std']:.4f}")
    print(f"  PCC  withheld (9)  : {summary['pcc_withheld_mean']:.4f} +/- {summary['pcc_withheld_std']:.4f}")
    print(f"  PCC  precordial (6): {summary['pcc_precordial_mean']:.4f} +/- {summary['pcc_precordial_std']:.4f}")
    print(f"  PCC  reconstructed : {summary['pcc_reconstructed_mean']:.4f} +/- {summary['pcc_reconstructed_std']:.4f}  (PRIMARY — non-trivial leads: {summary['reconstructed_lead_names']})")
    print(f"  RMSE all 12        : {summary['rmse_all_mean']:.4f} +/- {summary['rmse_all_std']:.4f}")
    print(f"  RMSE withheld (9)  : {summary['rmse_withheld_mean']:.4f} +/- {summary['rmse_withheld_std']:.4f}")
    print(f"  RMSE precordial (6): {summary['rmse_precordial_mean']:.4f} +/- {summary['rmse_precordial_std']:.4f}")
    print(f"  RMSE reconstructed : {summary['rmse_reconstructed_mean']:.4f} +/- {summary['rmse_reconstructed_std']:.4f}")

    print("\n  per-lead PCC  :", [f"{n}:{v:.3f}" for n, v in zip(
        ("I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"),
        summary["pcc_per_lead_mean"])])
    print(  "  per-lead RMSE :", [f"{n}:{v:.3f}" for n, v in zip(
        ("I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"),
        summary["rmse_per_lead_mean"])])

    # Sanity check -----------------------------------------------------
    pcc_recon = summary["pcc_reconstructed_mean"]
    if 0.55 <= pcc_recon <= 0.95:
        print("\n  [OK] reconstruction baseline in expected range [0.55, 0.95]")
    else:
        print(f"\n  [!!] unexpected baseline PCC_reconstructed={pcc_recon:.3f} "
              "- check preprocessing / normalization (M2).")

    print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
