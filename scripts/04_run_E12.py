"""Step 3b: E12 — PCM calibration perturbation study.

Purpose (per v1.2 spec):
  Validate that rho_i drops under realistic calibration perturbations,
  and that soft fallback (PCM + rho blend with W_global) keeps the
  reconstructed-lead ST error closer to the STEMI-safe threshold than
  naked PCM does.

For each (perturbation_type, level):
  - Apply perturbation to Yc (and optionally Xc for 'shared' mode).
  - Fit per-patient (W_i, b_i) on perturbed calibration.
  - Compute rho_i on same perturbed calibration.
  - Evaluate two predictions on CLEAN test segment:
      PCM          : Y_hat = Xt @ W_i + b_i
      PCM+fallback : Y_hat = Xt @ W_eff + b_eff
                     where (W_eff, b_eff) = rho_i * (W_i,b_i) + (1-rho_i)*(W_g,b_g)
  - Report on reconstructed leads (V1, V3, V4, V5, V6):
      PCC, RMSE_mV, ST60_anterior_mV, R_amp_mV, and mean rho_i.

Perturbation mode:
  --mode=yc     : perturbation applied to Yc only (12-lead device faults)
  --mode=xcyc   : same perturbation applied to both Xc and Yc (ambient noise)
  Default: yc   (the more diagnostic case — directly corrupts W_i fit)

Usage:
    python scripts/04_run_E12.py --split val
    python scripts/04_run_E12.py --split val --max_records 300 --mode xcyc
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
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
from calinet.eval.metrics import per_lead_pcc, per_lead_rmse
from calinet.eval.morphology import classify_leads, compute_morphology
from calinet.eval.perturbations import perturbation_catalog
from calinet.models.calibration import apply_transform, calibrate_patient
from calinet.models.rho import RhoConfig, calibration_quality, soft_fallback


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--gap", type=float, default=0.0)
    p.add_argument("--max_records", type=int, default=None)
    p.add_argument("--mode", choices=["yc", "xcyc"], default="yc")
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


@dataclass
class AggRow:
    pert_type: str
    level:     float
    mean_rho:     float
    std_rho:      float
    pcc_pcm:            float
    pcc_fallback:       float
    rmse_pcm:           float
    rmse_fallback:      float
    st60_ant_pcm:       float
    st60_ant_fallback:  float
    r_amp_pcm:          float
    r_amp_fallback:     float
    n_records:          int


def _sigma_mu_for_target(normalizer: GlobalNormalizer, target_idx):
    mu = normalizer.mu[target_idx]
    sigma = normalizer.sigma[target_idx]
    return mu, sigma


def _metrics_on_recon(
    ep_Yt: np.ndarray,
    Yt_pred: np.ndarray,
    mu, sigma,
    cfg: CaLiNetConfig,
    recon_idx_local: list[int],
) -> tuple[float, float, float, float]:
    """Return PCC, RMSE_mV, ST60_ant_mV, R_amp_mV on the reconstructed leads
    using a single test-segment reconstruction (normalized space).
    """
    # PCC / RMSE (normalized space is fine for PCC; RMSE we want in mV)
    pcc  = per_lead_pcc(ep_Yt, Yt_pred)[recon_idx_local].mean()

    # Convert to mV for amplitude metrics
    Yt_mv      = ep_Yt * (sigma + 1e-8) + mu
    Yt_pred_mv = Yt_pred * (sigma + 1e-8) + mu
    rmse_recon_mv = float(
        np.sqrt(((Yt_mv[:, recon_idx_local] - Yt_pred_mv[:, recon_idx_local]) ** 2).mean())
    )

    morph = compute_morphology(
        Yt_mv, Yt_pred_mv,
        cfg.target_leads, cfg.input_leads, cfg.sampling_rate,
    )
    if morph is None:
        return float(pcc), rmse_recon_mv, float("nan"), float("nan")
    return (
        float(pcc),
        rmse_recon_mv,
        float(morph.st_j60_anterior_main),
        float(morph.r_amp_err_main),
    )


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

    # Load calibrated rho normalization params (from scripts/05_diagnose_rho.py)
    rho_cfg_path = artifact_dir / "rho_config.npz"
    if rho_cfg_path.exists():
        z = np.load(rho_cfg_path)
        rho_cfg = RhoConfig(
            w_cond=float(z["w_cond"]),
            w_fit=float(z["w_fit"]),
            w_beat=float(z["w_beat"]),
            cond_center=float(z["cond_center"]),
            cond_scale=float(z["cond_scale"]),
            fit_center=float(z["fit_center"]),
            fit_scale=float(z["fit_scale"]),
            expected_hr_bpm=float(z["expected_hr_bpm"]),
            ectopic_threshold=float(z["ectopic_threshold"]),
        )
        print(f"  rho_config:    loaded calibrated from {rho_cfg_path}")
    else:
        rho_cfg = RhoConfig()
        print(f"  rho_config:    DEFAULT (run 05_diagnose_rho.py first for "
              f"calibrated values)")

    meta = load_metadata(cfg.ptbxl_path)
    fold = cfg.val_fold if args.split == "val" else cfg.test_fold
    split_meta = get_split_metadata(meta, fold)
    if args.max_records:
        split_meta = split_meta.iloc[: args.max_records]

    in_idx, target_idx = resolve_lead_indices(cfg.input_leads, cfg.target_leads)
    mu, sigma = _sigma_mu_for_target(normalizer, target_idx)

    groups = classify_leads(cfg.target_leads, cfg.input_leads)
    recon_idx_local = [cfg.target_leads.index(n) for n in groups["reconstructed"]]

    catalog = perturbation_catalog(cfg.sampling_rate)
    # rho_cfg already loaded above (calibrated from rho_config.npz if available)

    # Identify rpeak-lead index inside Xc (3 input leads): pick "II" if present
    if "II" in cfg.input_leads:
        rpeak_lead_Xc = cfg.input_leads.index("II")
    elif "I" in cfg.input_leads:
        rpeak_lead_Xc = cfg.input_leads.index("I")
    else:
        rpeak_lead_Xc = 0

    print("=" * 72)
    print(f"Step 3b E12 - perturbation mode={args.mode} on {args.split} fold")
    print("=" * 72)
    print(f"  records:        {len(split_meta)}")
    print(f"  reconstructed:  {groups['reconstructed']}")
    print(f"  rho weights:    w_cond={rho_cfg.w_cond}, "
          f"w_fit={rho_cfg.w_fit}, w_beat={rho_cfg.w_beat}")
    print(f"  fit_center={rho_cfg.fit_center:.4f}, fit_scale={rho_cfg.fit_scale:.4f}")
    print(f"  cond_center={rho_cfg.cond_center:.4f}, cond_scale={rho_cfg.cond_scale:.4f}")

    # Pre-load all episodes once (raw, normalized). Re-used across perturbations.
    filter_kwargs = {
        "low_hz": cfg.bandpass_low_hz, "high_hz": cfg.bandpass_high_hz,
        "fs": cfg.sampling_rate, "order": cfg.bandpass_order,
    } if cfg.use_bandpass else None

    episodes = []
    for _id, sig, _row in tqdm(
        iter_records(cfg.ptbxl_path, split_meta, cfg.sampling_rate, filter_kwargs),
        total=len(split_meta), desc="load ", leave=False,
    ):
        ep = extract_episode(
            sig, in_idx, target_idx,
            calib_samples=cfg.calib_samples,
            target_samples=cfg.target_samples,
            gap_samples=int(args.gap * cfg.sampling_rate),
            calib_start=0,
            normalizer=normalizer,
        )
        if ep is not None:
            episodes.append(ep)

    print(f"  loaded {len(episodes)} episodes")

    # Iterate perturbations × levels --------------------------------
    rows: list[AggRow] = []
    for pert_type, spec in catalog.items():
        for level in spec["levels"]:
            rhos, pccs_pcm, pccs_fb = [], [], []
            rmses_pcm, rmses_fb = [], []
            st60_pcm, st60_fb = [], []
            r_pcm, r_fb = [], []

            desc = f"{pert_type}@{level}"
            for ep in tqdm(episodes, desc=desc, leave=False):
                # Apply perturbation -----------------------------
                Yc_pert = spec["apply"](ep.Yc, level, rng)
                if args.mode == "xcyc" and pert_type != "clean":
                    # Same family, same level, fresh rng draw for Xc
                    Xc_pert = spec["apply"](ep.Xc, level, rng)
                else:
                    Xc_pert = ep.Xc

                # Fit ridge with perturbed calibration ------------
                W_i, b_i = calibrate_patient(
                    Xc_pert, Yc_pert, W_global, b_global,
                    lam_W=cfg.ridge_lambda_W, lam_b=cfg.ridge_lambda_b,
                )

                # rho_i on perturbed calibration -----------------
                q = calibration_quality(
                    Xc_pert, Yc_pert, W_i, b_i,
                    fs=cfg.sampling_rate, cfg=rho_cfg,
                    rpeak_lead_idx_in_Xc=rpeak_lead_Xc,
                )
                rho = q["rho"]
                rhos.append(rho)

                # Predictions on CLEAN Xt -----------------------
                Yt_pcm = apply_transform(ep.Xt, W_i, b_i)
                W_eff, b_eff = soft_fallback(W_i, b_i, W_global, b_global, rho)
                Yt_fb  = apply_transform(ep.Xt, W_eff, b_eff)

                # Metrics on CLEAN Yt ----------------------------
                pcc_p, rmse_p, st_p, r_p = _metrics_on_recon(
                    ep.Yt, Yt_pcm, mu, sigma, cfg, recon_idx_local,
                )
                pcc_f, rmse_f, st_f, r_f = _metrics_on_recon(
                    ep.Yt, Yt_fb, mu, sigma, cfg, recon_idx_local,
                )
                pccs_pcm.append(pcc_p); pccs_fb.append(pcc_f)
                rmses_pcm.append(rmse_p); rmses_fb.append(rmse_f)
                st60_pcm.append(st_p);  st60_fb.append(st_f)
                r_pcm.append(r_p);      r_fb.append(r_f)

            rows.append(AggRow(
                pert_type=pert_type,
                level=float(level),
                mean_rho=float(np.nanmean(rhos)),
                std_rho=float(np.nanstd(rhos)),
                pcc_pcm=float(np.nanmean(pccs_pcm)),
                pcc_fallback=float(np.nanmean(pccs_fb)),
                rmse_pcm=float(np.nanmean(rmses_pcm)),
                rmse_fallback=float(np.nanmean(rmses_fb)),
                st60_ant_pcm=float(np.nanmean(st60_pcm)),
                st60_ant_fallback=float(np.nanmean(st60_fb)),
                r_amp_pcm=float(np.nanmean(r_pcm)),
                r_amp_fallback=float(np.nanmean(r_fb)),
                n_records=len(rhos),
            ))

            print(f"  {pert_type:<17} lvl={level:<7}  rho={rows[-1].mean_rho:.3f}  "
                  f"ST60_ant PCM={rows[-1].st60_ant_pcm:.4f}  "
                  f"+fb={rows[-1].st60_ant_fallback:.4f}  "
                  f"(n={rows[-1].n_records})")

    # Save CSV --------------------------------------------------
    df = pd.DataFrame([r.__dict__ for r in rows])
    out_csv = results_dir / f"E12_{args.split}_{args.mode}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n  saved to {out_csv}")

    # Clinical threshold table -----------------------------------
    print()
    print("=" * 96)
    print("CLINICAL THRESHOLD TABLE  (STEMI threshold = 0.10 mV ST elevation)")
    print("-" * 96)
    print(f"{'Perturbation':<22}  {'rho_i':>7}  "
          f"{'PCM ST60_ant':>13}  {'+Fallback ST60_ant':>19}  "
          f"{'PCM':>12}  {'+Fallback':>12}")
    print(f"{'':<22}  {'':>7}  {'(mV)':>13}  {'(mV)':>19}  "
          f"{'status':>12}  {'status':>12}")
    print("-" * 96)

    def _status(v):
        if not np.isfinite(v):
            return "n/a"
        if v < 0.02:
            return "excellent"
        if v < 0.05:
            return "safe"
        if v < 0.10:
            return "marginal"
        return "NOT SAFE"

    for r in rows:
        label = f"{r.pert_type}@{r.level}"
        print(f"{label:<22}  {r.mean_rho:>7.3f}  "
              f"{r.st60_ant_pcm:>13.4f}  {r.st60_ant_fallback:>19.4f}  "
              f"{_status(r.st60_ant_pcm):>12}  {_status(r.st60_ant_fallback):>12}")
    print("=" * 96)

    # rho_i sanity check (trigger #1: drops under perturbation?) -----
    print()
    clean_rho = next((r.mean_rho for r in rows if r.pert_type == "clean"), float("nan"))
    dirty_rhos = [
        (f"{r.pert_type}@{r.level}", r.mean_rho)
        for r in rows
        if r.pert_type in ("gaussian_noise", "inject_pvc")
        and r.level in (5, 10, 2)
    ]
    print("rho_i sanity:")
    print(f"  clean                  rho = {clean_rho:.3f}   (expect ~0.9-1.0)")
    for name, v in dirty_rhos:
        drop = clean_rho - v
        print(f"  {name:<22} rho = {v:.3f}   (drop {drop:+.3f} vs clean)")


if __name__ == "__main__":
    main()
