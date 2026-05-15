"""Step 3b.1-diagnostic: Distribution of rho_i components on sample records.

Purpose (per v1.2 spec):
  Before committing to rho_i normalization parameters, collect the
  full per-record distribution of:
    - raw cond_number, nrmse, valid_beat_ratio, ectopic_ratio
    - sub-scores s_cond / s_fit / s_beat (under default params)
    - composite rho under default params

  Then call calibrate_rho_normalization() on the clean subset to
  auto-fit (fit_center, fit_scale, cond_center, cond_scale) such that
    s_fit(nrmse_p50_clean) ≈ 0.90
    s_fit(nrmse_p95_clean) ≈ 0.50
  (analogous for s_cond in log10 space).

  Finally re-evaluate rho under the calibrated params and report the
  sanity check:
    clean:    median rho >= 0.90, p95 >= 0.85
    dirty:    median rho <= 0.50

Outputs:
  results/rho_diagnostic_<split>.csv       per-record diagnostics
  checkpoints/rho_config.npz               calibrated (fit/cond) params
  prints distribution summaries + sanity verdict

Usage:
    python scripts/05_diagnose_rho.py --split val --max_clean 100 --max_per_pert 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calinet.config import CaLiNetConfig
from calinet.data.episodes import extract_episode, resolve_lead_indices
from calinet.data.normalizer import GlobalNormalizer
from calinet.data.ptbxl import get_split_metadata, iter_records, load_metadata
from calinet.eval.perturbations import perturbation_catalog
from calinet.models.calibration import calibrate_patient
from calinet.models.rho import (
    RhoConfig, calibrate_rho_normalization, calibration_quality,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--max_clean", type=int, default=100)
    p.add_argument("--max_per_pert", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


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


PERTURBATION_SUBSET = [
    ("clean",           0),
    ("gaussian_noise",  20),
    ("gaussian_noise",  5),
    ("baseline_drift",  0.3),
    ("inject_pvc",      1),
    ("inject_pvc",      2),
    ("electrode_scale", 0.3),
]


def _percentile_summary(vals: np.ndarray, label: str) -> str:
    if len(vals) == 0 or np.all(np.isnan(vals)):
        return f"  {label:<22} no data"
    p = np.nanpercentile(vals, [5, 25, 50, 75, 95])
    return (f"  {label:<22} n={len(vals):>4}  "
            f"p05={p[0]:>7.4f}  p25={p[1]:>7.4f}  "
            f"p50={p[2]:>7.4f}  p75={p[3]:>7.4f}  p95={p[4]:>7.4f}")


def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    rng = np.random.default_rng(args.seed)

    artifact_dir = Path(cfg.artifact_dir)
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    normalizer = GlobalNormalizer.load(artifact_dir / "normalizer.npz")
    gw = np.load(artifact_dir / "global_W.npz")
    W_global = gw["W_global"].astype(np.float32)
    b_global = gw["b_global"].astype(np.float32)

    meta = load_metadata(cfg.ptbxl_path)
    fold = cfg.val_fold if args.split == "val" else cfg.test_fold
    split_meta = get_split_metadata(meta, fold)
    cap = max(args.max_clean, args.max_per_pert * 2)
    split_meta = split_meta.iloc[:cap]

    in_idx, target_idx = resolve_lead_indices(cfg.input_leads, cfg.target_leads)
    if "II" in cfg.input_leads:
        rpeak_lead_Xc = cfg.input_leads.index("II")
    elif "I" in cfg.input_leads:
        rpeak_lead_Xc = cfg.input_leads.index("I")
    else:
        rpeak_lead_Xc = 0

    filter_kwargs = {
        "low_hz": cfg.bandpass_low_hz, "high_hz": cfg.bandpass_high_hz,
        "fs": cfg.sampling_rate, "order": cfg.bandpass_order,
    } if cfg.use_bandpass else None

    # Load episodes once
    episodes = []
    for _id, sig, _row in tqdm(
        iter_records(cfg.ptbxl_path, split_meta, cfg.sampling_rate, filter_kwargs),
        total=len(split_meta), desc="load", leave=False,
    ):
        ep = extract_episode(
            sig, in_idx, target_idx,
            calib_samples=cfg.calib_samples,
            target_samples=cfg.target_samples,
            gap_samples=0, calib_start=0,
            normalizer=normalizer,
        )
        if ep is not None:
            episodes.append(ep)

    print(f"[diag] loaded {len(episodes)} episodes")

    # ---------------------------------------------------------------
    # Pass 1: raw diagnostics under DEFAULT RhoConfig
    # ---------------------------------------------------------------
    catalog = perturbation_catalog(cfg.sampling_rate)
    default_cfg = RhoConfig()       # default centers/scales

    rows = []
    for pert_name, level in PERTURBATION_SUBSET:
        cap_n = args.max_clean if pert_name == "clean" else args.max_per_pert
        spec = catalog[pert_name]
        for ep in episodes[:cap_n]:
            Yc_p = spec["apply"](ep.Yc, level, rng)
            Xc_p = ep.Xc
            W_i, b_i = calibrate_patient(
                Xc_p, Yc_p, W_global, b_global,
                lam_W=cfg.ridge_lambda_W, lam_b=cfg.ridge_lambda_b,
            )
            q = calibration_quality(
                Xc_p, Yc_p, W_i, b_i,
                fs=cfg.sampling_rate, cfg=default_cfg,
                rpeak_lead_idx_in_Xc=rpeak_lead_Xc,
            )
            rows.append({
                "perturbation": pert_name, "level": float(level), **q,
            })

    df = pd.DataFrame(rows)
    raw_path = results_dir / f"rho_diagnostic_{args.split}_raw.csv"
    df.to_csv(raw_path, index=False)
    print(f"[diag] raw diagnostics saved to {raw_path}")

    # ---------------------------------------------------------------
    # Pass 2: report distributions of raw quantities
    # ---------------------------------------------------------------
    print()
    print("=" * 88)
    print(f"Raw measurement distributions (perturbation = clean, n={len(df[df.perturbation=='clean'])})")
    print("-" * 88)
    clean = df[df.perturbation == "clean"]
    print(_percentile_summary(clean["cond_number"].values,  "cond_number"))
    print(_percentile_summary(np.log10(clean["cond_number"].values.clip(min=1)), "log10(cond_number)"))
    print(_percentile_summary(clean["nrmse"].values,        "nrmse"))
    print(_percentile_summary(clean["valid_beat_ratio"].values, "valid_beat_ratio"))
    print(_percentile_summary(clean["ectopic_ratio"].values,     "ectopic_ratio"))
    print()
    print(f"Sub-scores under DEFAULT RhoConfig  "
          f"(fit_center={default_cfg.fit_center}, fit_scale={default_cfg.fit_scale}, "
          f"cond_center={default_cfg.cond_center}, cond_scale={default_cfg.cond_scale})")
    print("-" * 88)
    print(_percentile_summary(clean["s_cond"].values, "s_cond"))
    print(_percentile_summary(clean["s_fit"].values,  "s_fit"))
    print(_percentile_summary(clean["s_beat"].values, "s_beat"))
    print(_percentile_summary(clean["rho"].values,    "rho"))
    print("=" * 88)

    # ---------------------------------------------------------------
    # Pass 3: auto-calibrate logistic params from clean subset
    # ---------------------------------------------------------------
    clean_records = df[df.perturbation == "clean"].to_dict("records")
    params = calibrate_rho_normalization(
        clean_records,
        fit_p50_target=0.95,      # v1.2 relaxed from 0.90
        fit_p95_target=0.70,      # v1.2 relaxed from 0.50
        default_cfg=default_cfg,
    )
    print()
    print("=" * 88)
    print("Auto-calibrated RhoConfig (v1.2 targets: p50 -> 0.95, p95 -> 0.70)")
    print("-" * 88)
    fit_status = "calibrated" if params["fit_calibrated"] else "DEFAULT (spread too small)"
    cond_status = "calibrated" if params["cond_calibrated"] else "DEFAULT (spread too small)"
    print(f"  fit_center   = {params['fit_center']:.4f}  [{fit_status}]  "
          f"(nrmse p50={params['nrmse_p50']:.4f}, p95={params['nrmse_p95']:.4f}, "
          f"spread={params['nrmse_p95']-params['nrmse_p50']:.4f})")
    print(f"  fit_scale    = {params['fit_scale']:.4f}")
    print(f"  cond_center  = {params['cond_center']:.4f}  [{cond_status}]  "
          f"(log10_cond p50={params['log_cond_p50']:.2f}, p95={params['log_cond_p95']:.2f}, "
          f"spread={params['log_cond_p95']-params['log_cond_p50']:.2f})")
    print(f"  cond_scale   = {params['cond_scale']:.4f}")
    print("=" * 88)

    # Save calibrated cfg
    rho_cfg_path = artifact_dir / "rho_config.npz"
    np.savez(rho_cfg_path,
             fit_center=params["fit_center"],
             fit_scale=params["fit_scale"],
             cond_center=params["cond_center"],
             cond_scale=params["cond_scale"],
             w_cond=default_cfg.w_cond,
             w_fit=default_cfg.w_fit,
             w_beat=default_cfg.w_beat,
             expected_hr_bpm=default_cfg.expected_hr_bpm,
             ectopic_threshold=default_cfg.ectopic_threshold)
    print(f"[diag] saved calibrated rho_config to {rho_cfg_path}")

    # ---------------------------------------------------------------
    # Pass 4: re-evaluate rho under calibrated params and report sanity
    # ---------------------------------------------------------------
    cal_cfg = RhoConfig(
        w_cond=default_cfg.w_cond, w_fit=default_cfg.w_fit, w_beat=default_cfg.w_beat,
        cond_center=params["cond_center"], cond_scale=params["cond_scale"],
        fit_center=params["fit_center"],   fit_scale=params["fit_scale"],
    )

    new_rows = []
    for _, r in df.iterrows():
        from calinet.models.rho import _s_cond, _s_fit   # private but fine here
        s_cond = _s_cond(r["cond_number"], cal_cfg)
        s_fit  = _s_fit(r["nrmse"],        cal_cfg)
        s_beat = r["s_beat"]   # unchanged
        rho = cal_cfg.w_cond * s_cond + cal_cfg.w_fit * s_fit + cal_cfg.w_beat * s_beat
        new_rows.append({
            "perturbation": r["perturbation"], "level": r["level"],
            "s_cond_cal": s_cond, "s_fit_cal": s_fit, "s_beat_cal": s_beat,
            "rho_cal": float(np.clip(rho, 0, 1)),
        })
    df2 = pd.DataFrame(new_rows)
    cal_path = results_dir / f"rho_diagnostic_{args.split}_calibrated.csv"
    df2.to_csv(cal_path, index=False)

    print()
    print("=" * 88)
    print("Re-evaluated rho under calibrated params")
    print("-" * 88)
    for pert_name, level in PERTURBATION_SUBSET:
        sub = df2[(df2.perturbation == pert_name) & (df2.level == float(level))]
        rhos = sub["rho_cal"].values
        p = np.nanpercentile(rhos, [5, 50, 95])
        print(f"  {pert_name:<17}@{level:<7}  n={len(rhos):>3}  "
              f"p05={p[0]:.3f}  p50={p[1]:.3f}  p95={p[2]:.3f}")
    print("=" * 88)

    # Sanity check
    clean_rhos = df2[df2.perturbation == "clean"]["rho_cal"].values
    dirty_mask = (
        ((df2.perturbation == "inject_pvc") & (df2.level == 2)) |
        ((df2.perturbation == "gaussian_noise") & (df2.level == 5))
    )
    dirty_rhos = df2[dirty_mask]["rho_cal"].values

    clean_med = float(np.median(clean_rhos))
    clean_p95 = float(np.percentile(clean_rhos, 95))
    dirty_med = float(np.median(dirty_rhos)) if len(dirty_rhos) > 0 else float("nan")

    print()
    print(f"Sanity check:")
    print(f"  clean median rho      = {clean_med:.3f}   (need >= 0.90)")
    print(f"  clean p95 rho         = {clean_p95:.3f}   (need >= 0.85)")
    print(f"  dirty median rho      = {dirty_med:.3f}   (need <= 0.50)")

    clean_ok = clean_med >= 0.90 and clean_p95 >= 0.85
    dirty_ok = dirty_med <= 0.50
    if clean_ok and dirty_ok:
        print("  [OK] sanity passed. Proceed to full E12.")
    else:
        print("  [!!] sanity FAILED. Recalibrate or investigate.")
        if not clean_ok:
            print("        clean rho too low — check s_beat distribution or weight mix.")
        if not dirty_ok:
            print("        dirty rho too high — logistic scale may be too wide.")


if __name__ == "__main__":
    main()
