"""Step 4.1: Anchor tau_n and tau_m from PCM on the validation fold.

For every val record, compute on the test segment, reconstructed leads only:
  nrmse_recon  : RMSE of (Yt - Y_hat) / RMS(Yt)   in NORMALIZED space
  morph_err_norm = R_amp/1.0 + ST60_ant/0.1 + T_amp/0.3   in mV

Then take the median across records and set:
  tau_n = median(PCM nrmse_recon)         * 1.5
  tau_m = median(PCM morph_err_norm)      * 1.5

(So that PCM gets exp(-x/tau) ≈ 0.51 — leaves headroom for TCAE / CaLiNet-E
to improve.)

Result: writes checkpoints/val_score_tau.npz with tau_n / tau_m / medians.
These are LOCKED — all subsequent methods (TCAE, CaLiNet-E, CaLiNet-F)
use the same tau values for fair val score comparison.

Usage:
    python scripts/06_anchor_val_tau.py --split val
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calinet.config import CaLiNetConfig
from calinet.data.episodes import extract_episode, resolve_lead_indices
from calinet.data.normalizer import GlobalNormalizer
from calinet.data.ptbxl import get_split_metadata, iter_records, load_metadata
from calinet.eval.metrics import per_lead_pcc
from calinet.eval.morphology import classify_leads, compute_morphology
from calinet.models.calibration import apply_transform, calibrate_patient


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--max_records", type=int, default=None)
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


def nrmse_recon(Yt: np.ndarray, Yt_pred: np.ndarray, recon_idx: list[int]) -> float:
    """||Yt - Y_hat||_F / ||Yt||_F  on reconstructed leads (normalized space)."""
    diff = Yt[:, recon_idx] - Yt_pred[:, recon_idx]
    num = float(np.sqrt((diff ** 2).mean()))
    den = float(np.sqrt((Yt[:, recon_idx] ** 2).mean()))
    return num / max(den, 1e-8)


def morph_components(
    Yt_mv: np.ndarray, Yt_pred_mv: np.ndarray,
    cfg: CaLiNetConfig,
) -> tuple[float, float, float, float]:
    """Return (morph_err_norm, R_amp, ST60_ant, T_amp). NaN on failure."""
    rep = compute_morphology(
        Yt_mv, Yt_pred_mv, cfg.target_leads, cfg.input_leads, cfg.sampling_rate,
    )
    if rep is None:
        return float("nan"), float("nan"), float("nan"), float("nan")
    morph = (rep.r_amp_err_main / 1.0
             + rep.st_j60_anterior_main / 0.1
             + rep.t_amp_err_main / 0.3)
    return morph, rep.r_amp_err_main, rep.st_j60_anterior_main, rep.t_amp_err_main


def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    np.random.seed(cfg.seed)

    artifact_dir = Path(cfg.artifact_dir)
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
    groups = classify_leads(cfg.target_leads, cfg.input_leads)
    recon_idx = [cfg.target_leads.index(n) for n in groups["reconstructed"]]

    mu = normalizer.mu[target_idx]
    sigma = normalizer.sigma[target_idx]

    filter_kwargs = {
        "low_hz": cfg.bandpass_low_hz, "high_hz": cfg.bandpass_high_hz,
        "fs": cfg.sampling_rate, "order": cfg.bandpass_order,
    } if cfg.use_bandpass else None

    print("=" * 72)
    print(f"Step 4.1 - Anchoring tau_n / tau_m from PCM on {args.split} fold")
    print("=" * 72)
    print(f"  records: {len(split_meta)}")

    nrmses_pcm, nrmses_gl = [], []
    morphs_pcm, morphs_gl = [], []
    pccs_pcm,   pccs_gl   = [], []
    rA_pcm, st_pcm, tA_pcm = [], [], []
    rA_gl,  st_gl,  tA_gl  = [], [], []

    t0 = time.time()
    for _id, sig, _row in tqdm(
        iter_records(cfg.ptbxl_path, split_meta, cfg.sampling_rate, filter_kwargs),
        total=len(split_meta), desc="records", leave=False,
    ):
        ep = extract_episode(
            sig, in_idx, target_idx,
            calib_samples=cfg.calib_samples,
            target_samples=cfg.target_samples,
            gap_samples=0, calib_start=0,
            normalizer=normalizer,
        )
        if ep is None:
            continue

        Yt_mv = ep.Yt * (sigma + 1e-8) + mu

        # GL
        Yt_pred = apply_transform(ep.Xt, W_global, b_global)
        nrmses_gl.append(nrmse_recon(ep.Yt, Yt_pred, recon_idx))
        m, rA, st, tA = morph_components(
            Yt_mv, Yt_pred * (sigma + 1e-8) + mu, cfg)
        morphs_gl.append(m)
        rA_gl.append(rA); st_gl.append(st); tA_gl.append(tA)
        pcc = per_lead_pcc(ep.Yt, Yt_pred)
        pccs_gl.append(float(np.mean(pcc[recon_idx])))

        # PCM
        W_i, b_i = calibrate_patient(
            ep.Xc, ep.Yc, W_global, b_global,
            lam_W=cfg.ridge_lambda_W, lam_b=cfg.ridge_lambda_b,
        )
        Yt_pred = apply_transform(ep.Xt, W_i, b_i)
        nrmses_pcm.append(nrmse_recon(ep.Yt, Yt_pred, recon_idx))
        m, rA, st, tA = morph_components(
            Yt_mv, Yt_pred * (sigma + 1e-8) + mu, cfg)
        morphs_pcm.append(m)
        rA_pcm.append(rA); st_pcm.append(st); tA_pcm.append(tA)
        pcc = per_lead_pcc(ep.Yt, Yt_pred)
        pccs_pcm.append(float(np.mean(pcc[recon_idx])))

    print(f"  done in {(time.time()-t0)/60:.1f} min, n={len(nrmses_pcm)}")

    nrmses_pcm = np.array(nrmses_pcm, dtype=np.float64)
    morphs_pcm = np.array(morphs_pcm, dtype=np.float64)
    nrmses_gl  = np.array(nrmses_gl,  dtype=np.float64)
    morphs_gl  = np.array(morphs_gl,  dtype=np.float64)
    morphs_pcm_clean = morphs_pcm[~np.isnan(morphs_pcm)]
    morphs_gl_clean  = morphs_gl[~np.isnan(morphs_gl)]

    p_pcm_n = np.percentile(nrmses_pcm,       [5, 50, 95])
    p_pcm_m = np.percentile(morphs_pcm_clean, [5, 50, 95])
    p_gl_n  = np.percentile(nrmses_gl,        [5, 50, 95])
    p_gl_m  = np.percentile(morphs_gl_clean,  [5, 50, 95])

    print()
    print("=" * 72)
    print("PCM nrmse_recon (test segment, reconstructed leads, normalized)")
    print("-" * 72)
    print(f"  PCM   p05={p_pcm_n[0]:.4f}   p50={p_pcm_n[1]:.4f}   p95={p_pcm_n[2]:.4f}")
    print(f"  GL    p05={p_gl_n[0]:.4f}    p50={p_gl_n[1]:.4f}    p95={p_gl_n[2]:.4f}")
    print()
    print("PCM morph_err_norm (R_amp/1.0 + ST60_ant/0.1 + T_amp/0.3)")
    print("-" * 72)
    print(f"  PCM   p05={p_pcm_m[0]:.3f}   p50={p_pcm_m[1]:.3f}   p95={p_pcm_m[2]:.3f}")
    print(f"  GL    p05={p_gl_m[0]:.3f}    p50={p_gl_m[1]:.3f}    p95={p_gl_m[2]:.3f}")
    print("=" * 72)

    tau_n = float(p_pcm_n[1] * 1.5)
    tau_m = float(p_pcm_m[1] * 1.5)

    print()
    print("Anchored val-score taus (LOCKED — used by TCAE / CaLiNet-E / CaLiNet-F)")
    print("-" * 72)
    print(f"  tau_n = {tau_n:.4f}   (PCM nrmse_recon median * 1.5)")
    print(f"  tau_m = {tau_m:.4f}   (PCM morph_err_norm median * 1.5)")
    print()
    print("  Sanity: PCM at median should give")
    print(f"    exp(-nrmse_p50 / tau_n)        = {np.exp(-p_pcm_n[1] / tau_n):.3f}  (target ~0.51)")
    print(f"    exp(-morph_p50 / tau_m)        = {np.exp(-p_pcm_m[1] / tau_m):.3f}  (target ~0.51)")

    # ------------------------------------------------------------------
    # Measured val_score for GL and PCM (07_train_tcae.validate() uses
    # mean across records, not median; reproduce same aggregation here
    # so epoch-0 sanity check has a strict reference, not an estimate).
    # ------------------------------------------------------------------
    def _aggregate(pccs, nrmses, morphs, rA, st, tA):
        pccs    = np.array(pccs,    dtype=np.float64)
        nrmses  = np.array(nrmses,  dtype=np.float64)
        morphs  = np.array(morphs,  dtype=np.float64)
        rA      = np.array(rA,      dtype=np.float64)
        st      = np.array(st,      dtype=np.float64)
        tA      = np.array(tA,      dtype=np.float64)
        pcc_m   = float(np.nanmean(pccs))
        nrmse_m = float(np.nanmean(nrmses))
        morph_m = float(np.nanmean(morphs))
        rA_m    = float(np.nanmean(rA))
        st_m    = float(np.nanmean(st))
        tA_m    = float(np.nanmean(tA))
        score = (0.3 * pcc_m
                 + 0.4 * float(np.exp(-nrmse_m / tau_n))
                 + 0.3 * float(np.exp(-morph_m / tau_m)))
        return dict(pcc=pcc_m, nrmse=nrmse_m, morph=morph_m,
                    r_amp=rA_m, st60_ant=st_m, t_amp=tA_m, score=score)

    gl  = _aggregate(pccs_gl,  nrmses_gl,  morphs_gl,  rA_gl,  st_gl,  tA_gl)
    pcm = _aggregate(pccs_pcm, nrmses_pcm, morphs_pcm, rA_pcm, st_pcm, tA_pcm)

    print()
    print("Measured val_score (mean aggregation, matching 07_train_tcae.validate)")
    print("-" * 72)
    print(f"  GL : score={gl['score']:.4f}  PCC={gl['pcc']:.4f}  nrmse={gl['nrmse']:.4f}  "
          f"morph={gl['morph']:.4f}  ST60_ant={gl['st60_ant']:.4f} mV")
    print(f"  PCM: score={pcm['score']:.4f}  PCC={pcm['pcc']:.4f}  nrmse={pcm['nrmse']:.4f}  "
          f"morph={pcm['morph']:.4f}  ST60_ant={pcm['st60_ant']:.4f} mV")
    print(f"  PCM - GL: {pcm['score'] - gl['score']:+.4f}")

    out = artifact_dir / "val_score_tau.npz"
    np.savez(out,
             tau_n=tau_n, tau_m=tau_m,
             # legacy median fields (kept for backward compatibility)
             pcm_nrmse_p50=p_pcm_n[1], pcm_nrmse_p95=p_pcm_n[2],
             pcm_morph_p50=p_pcm_m[1], pcm_morph_p95=p_pcm_m[2],
             gl_nrmse_p50=p_gl_n[1],   gl_morph_p50=p_gl_m[1],
             # measured val_score and components (mean-aggregated)
             gl_val_score=gl["score"],
             gl_pcc=gl["pcc"],
             gl_nrmse=gl["nrmse"],
             gl_morph=gl["morph"],
             gl_r_amp=gl["r_amp"],
             gl_st60_ant=gl["st60_ant"],
             gl_t_amp=gl["t_amp"],
             pcm_val_score=pcm["score"],
             pcm_pcc=pcm["pcc"],
             pcm_nrmse=pcm["nrmse"],
             pcm_morph=pcm["morph"],
             pcm_r_amp=pcm["r_amp"],
             pcm_st60_ant=pcm["st60_ant"],
             pcm_t_amp=pcm["t_amp"])
    print(f"\n  saved to {out}")


if __name__ == "__main__":
    main()
