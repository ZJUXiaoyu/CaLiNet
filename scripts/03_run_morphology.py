"""Step 3a: Run morphology evaluation on the three baselines.

Reports three tables per the v1.2 spec:
  MAIN    — reconstructed leads only (V1, V3, V4, V5, V6 for I/II/V2 input)
            This is what the paper's main morphology table uses.
  SANITY  — derivable leads only (III, aVR, aVL, aVF)
            Errors should be near-zero; verifies W_global fit is sane.
  SUPP    — all 12 leads (for reviewers who ask; not used for decisions)

Additionally reports PCC_reconstructed on the same reconstructed leads,
so the full 5-number story (PCC + R_amp + T_amp + ST60 + ST80) is all
in the same coordinate system.

Scenario diagnosis is made from PCM vs GlobalLinear on MAIN morphology.

Usage:
    python scripts/03_run_morphology.py --split val
    python scripts/03_run_morphology.py --split val --max_records 300
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
from calinet.data.ptbxl import get_split_metadata, iter_records, load_metadata
from calinet.eval.metrics import per_lead_pcc, per_lead_rmse
from calinet.eval.morphology import (
    MorphologyReport, aggregate_morphology, classify_leads, compute_morphology,
)
from calinet.models.calibration import apply_transform, calibrate_patient
from calinet.models.template_replay import template_replay


# ----------------------------------------------------------------------
# CLI / config
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--gap", type=float, default=0.0)
    p.add_argument("--max_records", type=int, default=None)
    return p.parse_args()


def load_cfg(path):
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


# ----------------------------------------------------------------------
# Printing
# ----------------------------------------------------------------------
def print_main_table(results: dict[str, dict], pcc_by_method: dict[str, dict]) -> None:
    """Morphology on reconstructed leads + PCC on reconstructed leads."""
    print("=" * 96)
    print("MAIN TABLE - reconstructed leads only (V1, V3, V4, V5, V6)")
    print("                                                   STEMI threshold = 0.10 mV")
    print("-" * 96)
    print(f"{'Method':<16}  {'PCC':>7}  {'R_amp':>8}  {'T_amp':>8}  "
          f"{'ST60_ant':>10}  {'ST60_lat':>10}  {'ST80_ant':>10}  {'ST80_lat':>10}")
    print(f"{'':<16}  {'':>7}  {'(mV)':>8}  {'(mV)':>8}  "
          f"{'(mV)':>10}  {'(mV)':>10}  {'(mV)':>10}  {'(mV)':>10}")
    print("-" * 96)
    for m, r in results.items():
        pcc_m = pcc_by_method[m]["pcc_recon"]
        print(f"{m:<16}  "
              f"{pcc_m:>7.4f}  "
              f"{r['r_amp_err_main_mean']:>8.4f}  "
              f"{r['t_amp_err_main_mean']:>8.4f}  "
              f"{r['st_j60_anterior_main_mean']:>10.4f}  "
              f"{r['st_j60_lateral_main_mean']:>10.4f}  "
              f"{r['st_j80_anterior_main_mean']:>10.4f}  "
              f"{r['st_j80_lateral_main_mean']:>10.4f}")
    print("=" * 96)


def print_sanity_table(results: dict[str, dict]) -> None:
    """Derivable leads (III, aVR, aVL, aVF) — errors should be very small."""
    print()
    print("=" * 80)
    print("SANITY TABLE - derivable leads only (III, aVR, aVL, aVF)")
    print("               Expected: errors near zero (< ~0.05 mV) if W_global fit sane")
    print("-" * 80)
    print(f"{'Method':<16}  {'R_amp':>8}  {'T_amp':>8}  {'ST60':>8}  {'ST80':>8}")
    print(f"{'':<16}  {'(mV)':>8}  {'(mV)':>8}  {'(mV)':>8}  {'(mV)':>8}")
    print("-" * 80)
    for m, r in results.items():
        print(f"{m:<16}  "
              f"{r['r_amp_err_sanity_mean']:>8.4f}  "
              f"{r['t_amp_err_sanity_mean']:>8.4f}  "
              f"{r['st_j60_sanity_mean']:>8.4f}  "
              f"{r['st_j80_sanity_mean']:>8.4f}")
    print("=" * 80)


def print_supp_table(results: dict[str, dict]) -> None:
    """All 12 leads — polluted by pass-through + derivable; supplementary only."""
    print()
    print("=" * 80)
    print("SUPPLEMENTARY - all 12 leads (mixes pass-through + derivable + recon)")
    print("-" * 80)
    print(f"{'Method':<16}  {'R_amp':>8}  {'T_amp':>8}  {'ST60':>8}  {'ST80':>8}")
    print("-" * 80)
    for m, r in results.items():
        print(f"{m:<16}  "
              f"{r['r_amp_err_all12_mean']:>8.4f}  "
              f"{r['t_amp_err_all12_mean']:>8.4f}  "
              f"{r['st_j60_all12_mean']:>8.4f}  "
              f"{r['st_j80_all12_mean']:>8.4f}")
    print("=" * 80)


def scenario_diagnosis(gl: dict, pcm: dict) -> None:
    """PCM vs GL on MAIN (reconstructed) morphology → A/B/C."""
    print()
    print("=" * 72)
    print("Scenario diagnosis (PCM vs GlobalLinear on reconstructed leads)")
    print("=" * 72)
    keys = [
        ("R_amp",    "r_amp_err_main_mean"),
        ("T_amp",    "t_amp_err_main_mean"),
        ("ST60_ant", "st_j60_anterior_main_mean"),
        ("ST60_lat", "st_j60_lateral_main_mean"),
        ("ST80_ant", "st_j80_anterior_main_mean"),
        ("ST80_lat", "st_j80_lateral_main_mean"),
    ]
    rel_deltas = []
    for name, k in keys:
        v_gl = gl[k]
        v_pcm = pcm[k]
        if np.isnan(v_gl) or np.isnan(v_pcm):
            continue
        abs_delta = v_gl - v_pcm                        # +ve = PCM better
        rel_delta = abs_delta / max(v_gl, 1e-9)
        rel_deltas.append(rel_delta)
        verdict = "PCM better" if abs_delta > 0 else "GL better "
        print(f"  {name:<10} GL={v_gl:.4f} mV   PCM={v_pcm:.4f} mV   "
              f"Δ={abs_delta:+.4f} mV ({rel_delta*100:+.1f}%)  {verdict}")

    if not rel_deltas:
        print("  no valid comparisons.")
        return

    avg_rel = float(np.mean(rel_deltas))
    print(f"\n  Avg relative improvement PCM over GL (reconstructed): "
          f"{avg_rel*100:+.1f}%")

    # Clinical threshold status
    st_ant_pcm = pcm["st_j60_anterior_main_mean"]
    print(f"\n  PCM ST60_anterior = {st_ant_pcm:.4f} mV  "
          f"(STEMI threshold = 0.10 mV → "
          f"{'NOT SAFE' if st_ant_pcm >= 0.05 else 'marginal'}, "
          f"CaLiNet goal: push below 0.02 mV)")

    print()
    if avg_rel > 0.15:
        print("  -> Scenario A: PCM beats GL on reconstructed morphology too.")
        print("     Battle plan: move CaLiNet to OOD / long-gap (E2, E4).")
    elif avg_rel < 0.05:
        print("  -> Scenario B (GOLDEN): PCM matches GL on reconstructed")
        print("     morphology despite winning on PCC. CaLiNet's residual")
        print("     fills this calibration-can't-handle-morphology gap.")
    else:
        print(f"  -> Scenario C: PCM modestly better on reconstructed")
        print(f"     morphology (avg {avg_rel*100:.1f}%). Morphology IS the")
        print(f"     main battleground; execution must be tight.")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    np.random.seed(cfg.seed)

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
    if args.max_records:
        split_meta = split_meta.iloc[: args.max_records]

    in_idx, target_idx = resolve_lead_indices(cfg.input_leads, cfg.target_leads)
    calib_samples = cfg.calib_samples
    target_samples = cfg.target_samples
    gap_samples = int(args.gap * cfg.sampling_rate)

    groups = classify_leads(cfg.target_leads, cfg.input_leads)
    recon_idx_within_target = [cfg.target_leads.index(n) for n in groups["reconstructed"]]

    print("=" * 72)
    print(f"Step 3a - Morphology + PCC on {args.split.upper()} fold, gap={args.gap}s")
    print("=" * 72)
    print(f"  records:     {len(split_meta)}")
    print(f"  input:        {groups['input']}")
    print(f"  derivable:    {groups['derivable']}")
    print(f"  reconstructed:{groups['reconstructed']}")

    filter_kwargs = {
        "low_hz": cfg.bandpass_low_hz, "high_hz": cfg.bandpass_high_hz,
        "fs": cfg.sampling_rate, "order": cfg.bandpass_order,
    } if cfg.use_bandpass else None

    methods = ("GlobalLinear", "PCM", "TemplateReplay")
    reports: dict[str, list[MorphologyReport]] = {m: [] for m in methods}
    pcc_store: dict[str, list[float]] = {m: [] for m in methods}

    t0 = time.time()
    for _ecg_id, sig, _row in tqdm(
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

        mu = normalizer.mu[target_idx]
        sigma = normalizer.sigma[target_idx]
        Yt_mv = ep.Yt * (sigma + 1e-8) + mu

        # GlobalLinear -------------------------------------------------
        Yt_pred = apply_transform(ep.Xt, W_global, b_global)
        Yt_pred_mv = Yt_pred * (sigma + 1e-8) + mu
        r = compute_morphology(Yt_mv, Yt_pred_mv,
                               cfg.target_leads, cfg.input_leads,
                               cfg.sampling_rate)
        if r is not None:
            reports["GlobalLinear"].append(r)
        pcc = per_lead_pcc(ep.Yt, Yt_pred)
        pcc_store["GlobalLinear"].append(float(pcc[recon_idx_within_target].mean()))

        # PCM ---------------------------------------------------------
        W_i, b_i = calibrate_patient(
            ep.Xc, ep.Yc, W_global, b_global,
            lam_W=cfg.ridge_lambda_W, lam_b=cfg.ridge_lambda_b,
        )
        Yt_pred = apply_transform(ep.Xt, W_i, b_i)
        Yt_pred_mv = Yt_pred * (sigma + 1e-8) + mu
        r = compute_morphology(Yt_mv, Yt_pred_mv,
                               cfg.target_leads, cfg.input_leads,
                               cfg.sampling_rate)
        if r is not None:
            reports["PCM"].append(r)
        pcc = per_lead_pcc(ep.Yt, Yt_pred)
        pcc_store["PCM"].append(float(pcc[recon_idx_within_target].mean()))

        # TemplateReplay ----------------------------------------------
        Yt_pred = template_replay(
            Xc_target=ep.Yc, Xt_input=ep.Xt,
            Yt_shape=ep.Yt.shape,
            fs=cfg.sampling_rate,
            input_leads=cfg.input_leads,
            target_leads=cfg.target_leads,
        )
        Yt_pred_mv = Yt_pred * (sigma + 1e-8) + mu
        r = compute_morphology(Yt_mv, Yt_pred_mv,
                               cfg.target_leads, cfg.input_leads,
                               cfg.sampling_rate)
        if r is not None:
            reports["TemplateReplay"].append(r)
        pcc = per_lead_pcc(ep.Yt, Yt_pred)
        pcc_store["TemplateReplay"].append(float(pcc[recon_idx_within_target].mean()))

    print(f"  processed in {(time.time()-t0)/60:.1f} min")

    results = {m: aggregate_morphology(rs) for m, rs in reports.items()}
    pcc_agg = {m: {
        "pcc_recon": float(np.mean(pcc_store[m])),
        "pcc_recon_std": float(np.std(pcc_store[m])),
    } for m in methods}

    # Save CSV
    rows = []
    for m in methods:
        row = {"method": m, **results[m], **pcc_agg[m]}
        rows.append(row)
    df = pd.DataFrame(rows)
    out = results_dir / f"morphology_{args.split}_gap{args.gap:g}.csv"
    df.to_csv(out, index=False)
    print(f"  saved to {out}")

    print_main_table(results, pcc_agg)
    print_sanity_table(results)
    print_supp_table(results)
    scenario_diagnosis(results["GlobalLinear"], results["PCM"])


if __name__ == "__main__":
    main()
