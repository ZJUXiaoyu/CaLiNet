"""Step 4.8: CPSC2018 OOD evaluation for the full method ladder.

Evaluates 6 reconstruction methods on out-of-distribution CPSC2018:

  1. GL                  global linear (W_global, b_global) from PTB-XL train
  2. PCM                 per-patient ridge calibration on the 2s calib block
  3. CNN baseline        plain 1D U-Net (no calibration)
  4. Transformer         (no calibration)
  5. UNetWithAnchor                CNN + W_global anchor (no calibration)
  6. CaLiNet-E           ours: per-patient ridge + FiLM-conditioned U-Net

Two-level aggregation (decision Q in design notes):
  - per-record metric = mean of per-segment metrics within the record
  - final metric      = mean of per-record metrics across records
This makes long records and short records contribute equally.

Per-segment outputs include `gap_samples` so a follow-up "performance vs
gap length" plot can be made from the dumped CSV without re-running.

ρ_i is monitored on CaLiNet-E to flag distribution shift (OOD records
with ρ < 0.85 will trigger soft fallback toward W_global).

Usage:
    python scripts/10_eval_cpsc2018.py --root data/cpsc2018/
    python scripts/10_eval_cpsc2018.py --root data/cpsc2018/ --max_records 100   # mini sanity
    python scripts/10_eval_cpsc2018.py --root data/cpsc2018/ --methods GL PCM    # mini, GL+PCM only
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calinet.config import CaLiNetConfig
from calinet.data.cpsc2018 import iter_cpsc2018_episodes
from calinet.data.episodes import resolve_lead_indices
from calinet.data.normalizer import GlobalNormalizer
from calinet.eval.metrics import per_lead_pcc
from calinet.eval.morphology import classify_leads, compute_morphology
from calinet.models.baselines import CNNBaseline, TransformerBaseline
from calinet.models.calibration import (
    apply_transform, calibrate_patient,
)
from calinet.models.calinet_e import CaLiNetE
from calinet.models.unet_anchor import UNetWithAnchor


ALL_METHODS = ["GL", "PCM", "CNN", "Transformer", "UNetWithAnchor", "CaLiNet-E"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--root", required=True, help="path to CPSC2018 .mat directory")
    p.add_argument("--max_records", type=int, default=None)
    p.add_argument("--methods", nargs="+", default=ALL_METHODS,
                   choices=ALL_METHODS,
                   help="subset of methods to run (default: all)")
    p.add_argument("--out", default="results/cpsc2018_ood.csv")
    p.add_argument("--device", default=None)
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


# ----------------------------------------------------------------------
# Per-segment metric helper
# ----------------------------------------------------------------------
def _segment_metrics(
    Yt_norm: np.ndarray, Yp_norm: np.ndarray,
    sigma: np.ndarray, mu: np.ndarray,
    cfg: CaLiNetConfig, recon_idx: list[int],
) -> dict:
    """Compute per-segment metrics from normalized predictions."""
    pcc_lead = per_lead_pcc(Yt_norm, Yp_norm)
    pcc = float(np.mean(pcc_lead[recon_idx]))

    diff = Yt_norm[:, recon_idx] - Yp_norm[:, recon_idx]
    num = float(np.sqrt((diff ** 2).mean()))
    den = float(np.sqrt((Yt_norm[:, recon_idx] ** 2).mean()))
    nrmse = num / max(den, 1e-8)

    Yt_mv = Yt_norm * sigma + mu
    Yp_mv = Yp_norm * sigma + mu
    rep = compute_morphology(
        Yt_mv, Yp_mv, cfg.target_leads, cfg.input_leads, cfg.sampling_rate,
    )
    if rep is None:
        return dict(pcc=pcc, nrmse=nrmse, morph=np.nan,
                    r_amp=np.nan, st60_ant=np.nan, t_amp=np.nan)
    morph = (rep.r_amp_err_main / 1.0
             + rep.st_j60_anterior_main / 0.1
             + rep.t_amp_err_main / 0.3)
    return dict(
        pcc=pcc, nrmse=nrmse, morph=morph,
        r_amp=rep.r_amp_err_main,
        st60_ant=rep.st_j60_anterior_main,
        t_amp=rep.t_amp_err_main,
    )


# ----------------------------------------------------------------------
# Per-method predict functions — all consume one episode, return Y_pred (Lt, n_out) NORMALIZED
# ----------------------------------------------------------------------
def predict_gl(ep, W_global, b_global) -> np.ndarray:
    return apply_transform(ep.Xt, W_global, b_global)


def predict_pcm(ep, W_global, b_global, lam_W, lam_b) -> tuple[np.ndarray, dict]:
    W_i, b_i = calibrate_patient(
        ep.Xc, ep.Yc, W_global, b_global, lam_W=lam_W, lam_b=lam_b,
    )
    return apply_transform(ep.Xt, W_i, b_i), {"W_i": W_i, "b_i": b_i}


@torch.no_grad()
def predict_dl_simple(model, Xt: np.ndarray, device: torch.device) -> np.ndarray:
    """For CNN / Transformer / UNetWithAnchor — input (Lt, n_in) -> (Lt, n_out) normalized."""
    x = torch.from_numpy(Xt).unsqueeze(0).to(device).float()   # (1, Lt, n_in)
    y = model(x)
    return y.squeeze(0).cpu().numpy()


@torch.no_grad()
def predict_calinet_e(
    model: CaLiNetE, ep, device: torch.device,
) -> tuple[np.ndarray, float]:
    """Run CaLiNet-E on one episode; return (Y_pred normalized, rho)."""
    # Build a 1-sample batch in channel-first convention used by the model.
    batch = {
        "x_calib": torch.from_numpy(ep.Xc.T)[None].to(device).float(),
        "y_calib": torch.from_numpy(ep.Yc.T)[None].to(device).float(),
        "x_test":  torch.from_numpy(ep.Xt.T)[None].to(device).float(),
        "y_test":  torch.from_numpy(ep.Yt.T)[None].to(device).float(),
    }
    Y = model(batch).squeeze(0).cpu().numpy()
    # Recompute rho cheaply (model.forward already used it; CPU-side)
    from calinet.models.rho import calibration_quality
    from calinet.models.calibration import calibrate_patient_batch_torch
    Xc_t = batch["x_calib"].transpose(1, 2).float()
    Yc_t = batch["y_calib"].transpose(1, 2).float()
    W_i, b_i = calibrate_patient_batch_torch(
        Xc_t, Yc_t, model.W_global, model.b_global,
        lam_W=model.lam_W, lam_b=model.lam_b,
    )
    q = calibration_quality(
        Xc_t[0].cpu().numpy(), Yc_t[0].cpu().numpy(),
        W_i[0].cpu().numpy(), b_i[0].cpu().numpy(),
        fs=model.sampling_rate, cfg=model.rho_cfg,
        rpeak_lead_idx_in_Xc=model.rpeak_lead_idx_in_Xc,
    )
    return Y, float(q["rho"])


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------
def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    artifact_dir = Path(cfg.artifact_dir)
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- artifacts ---
    normalizer = GlobalNormalizer.load(artifact_dir / "normalizer.npz")
    gw = np.load(artifact_dir / "global_W.npz")
    W_global = gw["W_global"].astype(np.float32)
    b_global = gw["b_global"].astype(np.float32)

    in_idx, target_idx = resolve_lead_indices(cfg.input_leads, cfg.target_leads)
    groups = classify_leads(cfg.target_leads, cfg.input_leads)
    recon_idx = [cfg.target_leads.index(n) for n in groups["reconstructed"]]
    mu    = normalizer.mu[target_idx]
    sigma = normalizer.sigma[target_idx]

    # --- models ---
    methods = list(args.methods)
    print("=" * 78)
    print(f"CPSC2018 OOD evaluation  methods={methods}  device={device}")
    print("=" * 78)
    print(f"  reconstructed: {groups['reconstructed']} (idx={recon_idx})")
    print(f"  root:          {args.root}")
    print()

    cnn_model = transformer_model = unet_anchor_model = calinet_e_model = None
    if "CNN" in methods:
        cnn_model = CNNBaseline(
            n_in=len(cfg.input_leads), n_out=len(cfg.target_leads),
            channels=cfg.unet_channels, pad_to_multiple=cfg.pad_to_multiple,
        ).to(device)
        ckpt = torch.load(artifact_dir / "cnn_baseline_best.pth", map_location=device)
        cnn_model.load_state_dict(ckpt["model"])
        cnn_model.eval()
    if "Transformer" in methods:
        transformer_model = TransformerBaseline(
            n_in=len(cfg.input_leads), n_out=len(cfg.target_leads),
        ).to(device)
        ckpt = torch.load(artifact_dir / "transformer_baseline_best.pth", map_location=device)
        transformer_model.load_state_dict(ckpt["model"])
        transformer_model.eval()
    if "UNetWithAnchor" in methods:
        unet_anchor_model = UNetWithAnchor.from_artifacts(
            artifact_dir,
            n_in=len(cfg.input_leads), n_out=len(cfg.target_leads),
            channels=cfg.unet_channels, embedding_dim=cfg.embedding_dim,
            pad_to_multiple=cfg.pad_to_multiple,
        ).to(device)
        ckpt = torch.load(artifact_dir / "unet_anchor_full_best.pth", map_location=device)
        unet_anchor_model.load_state_dict(ckpt["model"])
        unet_anchor_model.eval()
    if "CaLiNet-E" in methods:
        calinet_e_model = CaLiNetE.from_artifacts(
            artifact_dir,
            n_in=len(cfg.input_leads), n_out=len(cfg.target_leads),
            channels=cfg.unet_channels, embedding_dim=cfg.embedding_dim,
            pad_to_multiple=cfg.pad_to_multiple,
            sampling_rate=cfg.sampling_rate,
            lam_W=cfg.ridge_lambda_W, lam_b=cfg.ridge_lambda_b,
            force_rho_one=False,
        ).to(device)
        ckpt = torch.load(artifact_dir / "calinet_e_full_best.pth", map_location=device)
        calinet_e_model.load_state_dict(ckpt["model"])
        calinet_e_model.eval()

    # --- iterate episodes ---
    bandpass_low = cfg.bandpass_low_hz if cfg.use_bandpass else None
    bandpass_high = cfg.bandpass_high_hz if cfg.use_bandpass else None
    bandpass_order = cfg.bandpass_order

    rows = []                 # one row per (record, segment, method)
    rho_values = []           # CaLiNet-E rho per segment

    t0 = time.time()
    iterator = iter_cpsc2018_episodes(
        root=args.root,
        in_idx=in_idx, target_idx=target_idx,
        calib_samples=cfg.calib_samples,
        target_samples=cfg.target_samples,
        normalizer=normalizer,
        bandpass_low=bandpass_low,
        bandpass_high=bandpass_high,
        bandpass_order=bandpass_order,
        sampling_rate_target=cfg.sampling_rate,
        max_records=args.max_records,
    )

    for rec_id, episodes in tqdm(iterator, desc="records", leave=False):
        for ep in episodes:
            for method in methods:
                if method == "GL":
                    Yp = predict_gl(ep, W_global, b_global)
                elif method == "PCM":
                    Yp, _ = predict_pcm(
                        ep, W_global, b_global,
                        cfg.ridge_lambda_W, cfg.ridge_lambda_b,
                    )
                elif method == "CNN":
                    Yp = predict_dl_simple(cnn_model, ep.Xt, device)
                elif method == "Transformer":
                    Yp = predict_dl_simple(transformer_model, ep.Xt, device)
                elif method == "UNetWithAnchor":
                    Yp = predict_dl_simple(unet_anchor_model, ep.Xt, device)
                elif method == "CaLiNet-E":
                    Yp, rho = predict_calinet_e(calinet_e_model, ep, device)
                    rho_values.append({
                        "record_id": rec_id, "seg_index": ep.seg_index,
                        "gap_samples": ep.gap_samples, "rho": rho,
                    })
                else:
                    continue

                m = _segment_metrics(ep.Yt, Yp, sigma, mu, cfg, recon_idx)
                rows.append({
                    "record_id": rec_id,
                    "seg_index": ep.seg_index,
                    "gap_samples": ep.gap_samples,
                    "method": method,
                    **m,
                })

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n  saved per-segment metrics to {out_path}")
    if rho_values:
        rho_df = pd.DataFrame(rho_values)
        rho_path = out_path.with_name(out_path.stem + "_rho.csv")
        rho_df.to_csv(rho_path, index=False)
        print(f"  saved CaLiNet-E rho per segment to {rho_path}")

    # --- two-level aggregation ---
    print()
    print("=" * 78)
    print("Two-level aggregation: mean within record, then across records")
    print("=" * 78)
    summary_rows = []
    for method in methods:
        d = df[df["method"] == method]
        # level 1: per-record mean
        per_rec = d.groupby("record_id")[
            ["pcc", "nrmse", "morph", "r_amp", "st60_ant", "t_amp"]
        ].mean()
        # level 2: across-record mean
        agg = per_rec.mean(axis=0)
        summary_rows.append({
            "method":   method,
            "n_records": len(per_rec),
            "n_segments": len(d),
            "PCC":      agg["pcc"],
            "nrmse":    agg["nrmse"],
            "morph":    agg["morph"],
            "R_amp":    agg["r_amp"],
            "ST60_ant": agg["st60_ant"],
            "T_amp":    agg["t_amp"],
        })
    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x),
    ))
    summary_path = out_path.with_name(out_path.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\n  saved summary to {summary_path}")

    if rho_values:
        rhos = np.array([r["rho"] for r in rho_values])
        print()
        print("CaLiNet-E rho on CPSC2018 (OOD distribution shift indicator):")
        print(f"  n      = {len(rhos)}")
        print(f"  median = {np.median(rhos):.4f}  (target: >= 0.85; PTB-XL clean was 0.957)")
        print(f"  p05    = {np.percentile(rhos, 5):.4f}")
        print(f"  p95    = {np.percentile(rhos, 95):.4f}")
        if np.median(rhos) < 0.85:
            print("  [WARN] median rho < 0.85: significant distribution shift; "
                  "CaLiNet-E will fall back toward GL on many records.")

    print(f"\nelapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
