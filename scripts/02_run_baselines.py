"""Step 2 of v1.0 implementation order:

Evaluate three non-neural baselines on validation / test folds:

  1. GlobalLinear      : y = W_global @ x + b_global  (no calibration)
  2. PCM               : per-patient ridge fit on (Xc, Yc), then y = W_i @ x + b_i
  3. TemplateReplay    : beat template from Yc replayed at R-peaks in Xt

All three use the same episode layout (calib_seconds + gap + target_seconds).
This produces the v1.0 spec's "floor" — CaLiNet must beat these numbers.

Framing decision (M6): we run TemplateReplay FIRST because its performance
determines our story. Use --gap to sweep; main table uses gap=0 but E4
later sweeps gap ∈ {0, 5, 10, 20, 30 s}.

Usage:
    python scripts/02_run_baselines.py
    python scripts/02_run_baselines.py --split val --max_records 500
    python scripts/02_run_baselines.py --split test --gap 0

Writes:
    results/baselines_<split>_gap<g>.csv   per-record per-method rows
    prints aggregated comparison table
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calinet.config import CaLiNetConfig
from calinet.data.episodes import extract_episode, resolve_lead_indices
from calinet.data.normalizer import GlobalNormalizer
from calinet.data.ptbxl import (
    PTBXL_LEAD_ORDER,
    get_split_metadata,
    iter_records,
    load_metadata,
)
from calinet.eval.metrics import per_lead_pcc, per_lead_rmse, summarize
from calinet.models.calibration import apply_transform, calibrate_patient
from calinet.models.template_replay import template_replay


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--gap", type=float, default=0.0,
                   help="Seconds between calibration end and test start.")
    p.add_argument("--max_records", type=int, default=None,
                   help="Cap for quick debugging.")
    return p.parse_args()


def load_cfg(path: str) -> CaLiNetConfig:
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    for k in (
        "input_leads", "target_leads", "train_gap_seconds",
        "unet_channels", "film_layers", "aug_noise_snr_db",
        "aug_drift_hz", "aug_amp_scale", "train_folds",
        "ridge_lambda_W_grid", "ridge_lambda_b_grid",
    ):
        if k in d and isinstance(d[k], list):
            d[k] = tuple(d[k])
    return CaLiNetConfig(**d)


def build_filter_kwargs(cfg: CaLiNetConfig) -> dict | None:
    if not cfg.use_bandpass:
        return None
    return {
        "low_hz":  cfg.bandpass_low_hz,
        "high_hz": cfg.bandpass_high_hz,
        "fs":      cfg.sampling_rate,
        "order":   cfg.bandpass_order,
    }


def print_table(df: pd.DataFrame) -> None:
    """Pretty-print method comparison on summary metrics."""
    groups = [
        "pcc_all", "pcc_withheld", "pcc_precordial", "pcc_reconstructed",
        "rmse_reconstructed",
    ]
    cols = {g: f"{g}_mean" for g in groups}

    order = ["GlobalLinear", "PCM", "TemplateReplay"]
    df = df.set_index("method").reindex(order).reset_index()

    print()
    print("=" * 96)
    hdr = f"{'Method':<16}"
    for g in groups:
        hdr += f"{g:>18}"
    print(hdr)
    print("-" * 96)
    for _, row in df.iterrows():
        line = f"{row['method']:<16}"
        for g in groups:
            mean = row[f"{g}_mean"]
            std  = row[f"{g}_std"]
            line += f"   {mean:.4f}±{std:.4f}"
        print(line)
    print("=" * 96)


def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    np.random.seed(cfg.seed)

    artifact_dir = Path(cfg.artifact_dir)
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load normalizer + global W ------------------------------------
    print("=" * 72)
    print(f"Step 2 - Baselines on {args.split.upper()} fold (gap={args.gap}s)")
    print("=" * 72)

    normalizer = GlobalNormalizer.load(artifact_dir / "normalizer.npz")
    gw = np.load(artifact_dir / "global_W.npz")
    W_global = gw["W_global"].astype(np.float32)
    b_global = gw["b_global"].astype(np.float32)
    print(f"  normalizer:  mu/sigma loaded")
    print(f"  W_global:    shape={W_global.shape} ||W||={np.linalg.norm(W_global):.4f}")

    # Metadata --------------------------------------------------------
    meta = load_metadata(cfg.ptbxl_path)
    fold = cfg.val_fold if args.split == "val" else cfg.test_fold
    split_meta = get_split_metadata(meta, fold)
    if args.max_records:
        split_meta = split_meta.iloc[: args.max_records]
    print(f"  records:     {len(split_meta)}")

    in_idx, target_idx = resolve_lead_indices(cfg.input_leads, cfg.target_leads)

    calib_samples  = cfg.calib_samples
    target_samples = cfg.target_samples
    gap_samples    = int(args.gap * cfg.sampling_rate)
    total_needed   = calib_samples + gap_samples + target_samples
    record_total   = 10 * cfg.sampling_rate   # PTB-XL is 10s
    print(f"  layout:      Lc={calib_samples} + gap={gap_samples} + Lt={target_samples} "
          f"= {total_needed} / {record_total} samples")

    if total_needed > record_total:
        print(f"  [!!] layout exceeds record length ({record_total}). aborting.")
        return

    # Storage for per-record metrics ---------------------------------
    methods = ("GlobalLinear", "PCM", "TemplateReplay")
    pcc_store  = {m: [] for m in methods}
    rmse_store = {m: [] for m in methods}

    filter_kwargs = build_filter_kwargs(cfg)

    t0 = time.time()
    n_processed = 0
    for ecg_id, sig, _row in tqdm(
        iter_records(cfg.ptbxl_path, split_meta,
                     sampling_rate=cfg.sampling_rate,
                     filter_kwargs=filter_kwargs),
        total=len(split_meta), desc="records", leave=False,
    ):
        ep = extract_episode(
            sig, in_idx, target_idx,
            calib_samples=calib_samples,
            target_samples=target_samples,
            gap_samples=gap_samples,
            calib_start=0,
            normalizer=normalizer,
        )
        if ep is None:
            continue

        # 1. GlobalLinear -----------------------------------------
        Yt_pred = apply_transform(ep.Xt, W_global, b_global)
        pcc_store["GlobalLinear"].append(per_lead_pcc(ep.Yt, Yt_pred))
        rmse_store["GlobalLinear"].append(per_lead_rmse(ep.Yt, Yt_pred))

        # 2. PCM --------------------------------------------------
        W_i, b_i = calibrate_patient(
            ep.Xc, ep.Yc, W_global, b_global,
            lam_W=cfg.ridge_lambda_W, lam_b=cfg.ridge_lambda_b,
        )
        Yt_pred = apply_transform(ep.Xt, W_i, b_i)
        pcc_store["PCM"].append(per_lead_pcc(ep.Yt, Yt_pred))
        rmse_store["PCM"].append(per_lead_rmse(ep.Yt, Yt_pred))

        # 3. TemplateReplay --------------------------------------
        Yt_pred = template_replay(
            Xc_target=ep.Yc, Xt_input=ep.Xt,
            Yt_shape=ep.Yt.shape,
            fs=cfg.sampling_rate,
            input_leads=cfg.input_leads,
            target_leads=cfg.target_leads,
        )
        pcc_store["TemplateReplay"].append(per_lead_pcc(ep.Yt, Yt_pred))
        rmse_store["TemplateReplay"].append(per_lead_rmse(ep.Yt, Yt_pred))

        n_processed += 1

    print(f"  processed {n_processed} / {len(split_meta)} records in "
          f"{(time.time()-t0)/60:.1f} min")

    # Aggregate and print ---------------------------------------------
    rows = []
    for m in methods:
        pcc_pl  = np.stack(pcc_store[m])      # (N, 12)
        rmse_pl = np.stack(rmse_store[m])
        s = summarize(pcc_pl, rmse_pl, input_lead_idx=cfg.input_lead_idx)
        row = {"method": m, "n_records": len(pcc_pl)}
        for k, v in s.items():
            if isinstance(v, (int, float)):
                row[k] = v
        rows.append(row)
    df = pd.DataFrame(rows)

    out_path = results_dir / f"baselines_{args.split}_gap{args.gap:g}.csv"
    df.to_csv(out_path, index=False)
    print(f"  saved to {out_path}")

    print_table(df)

    # Per-lead breakdown for the winner (by pcc_reconstructed_mean) ---
    best_idx = df["pcc_reconstructed_mean"].idxmax()
    best_method = df.loc[best_idx, "method"]
    print(f"\n  Best on reconstructed leads: {best_method}")
    print(f"  per-lead PCC (best):")
    pcc_pl = np.stack(pcc_store[best_method]).mean(axis=0)
    for name, v in zip(PTBXL_LEAD_ORDER, pcc_pl):
        marker = "  (input)" if name in cfg.input_leads else ""
        print(f"    {name:<4} {v:.4f}{marker}")

    # M6 framing decision aid -----------------------------------------
    pcm_recon = df[df["method"] == "PCM"]["pcc_reconstructed_mean"].iloc[0]
    tr_recon  = df[df["method"] == "TemplateReplay"]["pcc_reconstructed_mean"].iloc[0]
    gl_recon  = df[df["method"] == "GlobalLinear"]["pcc_reconstructed_mean"].iloc[0]

    print()
    print("=" * 72)
    print("Framing decision aid (M6)")
    print("=" * 72)
    print(f"  GlobalLinear    PCC_reconstructed = {gl_recon:.4f}")
    print(f"  PCM             PCC_reconstructed = {pcm_recon:.4f}  "
          f"(delta vs global = {pcm_recon - gl_recon:+.4f})")
    print(f"  TemplateReplay  PCC_reconstructed = {tr_recon:.4f}  "
          f"(delta vs PCM    = {tr_recon - pcm_recon:+.4f})")

    delta_tr_pcm = tr_recon - pcm_recon
    if delta_tr_pcm < -0.02:
        print("  -> TemplateReplay clearly weaker. Safe to claim "
              "'CaLiNet dominates in the resting short-gap regime'.")
    elif abs(delta_tr_pcm) < 0.02:
        print("  -> TemplateReplay ~ PCM. Reframe main story toward "
              "OOD / long-gap / morphology (E2, E4, E7).")
    else:
        print("  -> TemplateReplay BEATS PCM on resting short-gap. "
              "Reframe: main results move to CPSC2018 OOD (E2).")


if __name__ == "__main__":
    main()
