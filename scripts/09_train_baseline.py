"""Step 4.7: Train plain DL baselines (CNN / Transformer) on PTB-XL.

Forward signature: model(Xt) where Xt is (B, T, n_in). No calibration
branch, no W_global anchor, no FiLM. Trains from random init.

Loss / val flow are identical to 07_train_unet_anchor.py so the resulting
val_score is directly comparable to GL / PCM / 1D U-Net w/ anchor / CaLiNet-E.

Usage:
    python scripts/09_train_baseline.py --model cnn         --epochs 50 --tag cnn_baseline
    python scripts/09_train_baseline.py --model transformer --epochs 50 --tag transformer_baseline
    python scripts/09_train_baseline.py --model cnn --epochs 1 --max_train 500 --max_val 500
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calinet.config import CaLiNetConfig
from calinet.data.dataset import EpisodicPTBXL
from calinet.data.episodes import resolve_lead_indices
from calinet.data.normalizer import GlobalNormalizer
from calinet.eval.metrics import per_lead_pcc
from calinet.eval.morphology import classify_leads, compute_morphology
from calinet.models.baselines import CNNBaseline, TransformerBaseline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--model", choices=["cnn", "transformer"], required=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_train", type=int, default=None)
    p.add_argument("--max_val",   type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--tag", default=None)
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


def recon_mse(Y_pred, Y_true, recon_idx):
    return F.mse_loss(Y_pred[:, :, recon_idx], Y_true[:, :, recon_idx])


def _morph_err_norm(report):
    if report is None:
        return float("nan")
    return (report.r_amp_err_main / 1.0
            + report.st_j60_anterior_main / 0.1
            + report.t_amp_err_main / 0.3)


@torch.no_grad()
def validate(
    model, val_loader, normalizer, cfg,
    recon_idx, target_idx, tau_n, tau_m, device, desc="val",
):
    model.eval()
    mu    = normalizer.mu[target_idx]
    sigma = normalizer.sigma[target_idx]
    pccs, nrmses, morphs = [], [], []
    r_amp_vals, st_vals, t_amp_vals = [], [], []

    for batch in tqdm(val_loader, desc=desc, leave=False):
        Xt = batch["x_test"].to(device).transpose(1, 2).float()
        Yt = batch["y_test"].to(device).transpose(1, 2).float()
        Y_pred = model(Xt)

        Yt_np = Yt.cpu().numpy()
        Yp_np = Y_pred.cpu().numpy()
        for b in range(Yt_np.shape[0]):
            pcc = per_lead_pcc(Yt_np[b], Yp_np[b])
            pccs.append(float(np.mean(pcc[recon_idx])))
            diff = Yt_np[b, :, recon_idx] - Yp_np[b, :, recon_idx]
            num = float(np.sqrt((diff ** 2).mean()))
            den = float(np.sqrt((Yt_np[b, :, recon_idx] ** 2).mean()))
            nrmses.append(num / max(den, 1e-8))
            Yt_mv = Yt_np[b] * sigma + mu
            Yp_mv = Yp_np[b] * sigma + mu
            rep = compute_morphology(
                Yt_mv, Yp_mv,
                cfg.target_leads, cfg.input_leads, cfg.sampling_rate,
            )
            m = _morph_err_norm(rep)
            if not np.isnan(m):
                morphs.append(m)
                r_amp_vals.append(rep.r_amp_err_main)
                st_vals.append(rep.st_j60_anterior_main)
                t_amp_vals.append(rep.t_amp_err_main)

    pcc_m   = float(np.nanmean(pccs))
    nrmse_m = float(np.nanmean(nrmses))
    morph_m = float(np.nanmean(morphs)) if morphs else float("nan")
    val_score = (
        0.3 * pcc_m
        + 0.4 * float(np.exp(-nrmse_m / tau_n))
        + 0.3 * float(np.exp(-morph_m / tau_m))
    )
    return {
        "pcc_recon":         pcc_m,
        "nrmse_recon":       nrmse_m,
        "morph_err_norm":    morph_m,
        "r_amp_err_mv":      float(np.nanmean(r_amp_vals)) if r_amp_vals else float("nan"),
        "st60_anterior_mv":  float(np.nanmean(st_vals))    if st_vals    else float("nan"),
        "t_amp_err_mv":      float(np.nanmean(t_amp_vals)) if t_amp_vals else float("nan"),
        "val_score":         val_score,
    }


def fmt_metrics(m):
    return (f"score={m['val_score']:.4f}  "
            f"PCC={m['pcc_recon']:.4f}  nrmse={m['nrmse_recon']:.4f}  "
            f"morph={m['morph_err_norm']:.4f}  "
            f"(R_amp={m['r_amp_err_mv']:.4f}  "
            f"ST60_ant={m['st60_anterior_mv']:.4f}  "
            f"T_amp={m['t_amp_err_mv']:.4f})")


def build_model(name, cfg):
    n_in  = len(cfg.input_leads)
    n_out = len(cfg.target_leads)
    if name == "cnn":
        return CNNBaseline(
            n_in=n_in, n_out=n_out,
            channels=cfg.unet_channels,
            pad_to_multiple=cfg.pad_to_multiple,
        )
    if name == "transformer":
        return TransformerBaseline(
            n_in=n_in, n_out=n_out,
            d_model=192, n_layers=6, n_heads=8, ffn=768, dropout=0.1,
        )
    raise ValueError(name)


def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    tag = args.tag or f"{args.model}_baseline"

    artifact_dir = Path(cfg.artifact_dir)
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    tau_path = artifact_dir / "val_score_tau.npz"
    if not tau_path.exists():
        raise FileNotFoundError(f"{tau_path} not found.")
    tz = np.load(tau_path)
    tau_n = float(tz["tau_n"])
    tau_m = float(tz["tau_m"])
    pcm_val_score = float(tz["pcm_val_score"])
    gl_val_score  = float(tz["gl_val_score"])

    normalizer = GlobalNormalizer.load(artifact_dir / "normalizer.npz")
    model = build_model(args.model, cfg).to(device)
    n_params = model.n_params()

    _, target_idx = resolve_lead_indices(cfg.input_leads, cfg.target_leads)
    groups = classify_leads(cfg.target_leads, cfg.input_leads)
    recon_idx = [cfg.target_leads.index(n) for n in groups["reconstructed"]]

    print("=" * 88)
    print(f"Step 4.7 - Train baseline  model={args.model}  tag={tag}  device={device}")
    print("=" * 88)
    print(f"  params:            {n_params:,}")
    print(f"  reconstructed:     {groups['reconstructed']} (idx={recon_idx})")
    print(f"  anchored tau:      tau_n={tau_n:.4f}  tau_m={tau_m:.4f}")
    print(f"  GL  val_score:     {gl_val_score:.4f}")
    print(f"  PCM val_score:     {pcm_val_score:.4f}")
    print()

    print("loading datasets...")
    train_ds = EpisodicPTBXL(
        cfg, split="train", normalizer=normalizer, mode="train",
        max_records=args.max_train, seed=cfg.seed,
    )
    val_ds = EpisodicPTBXL(
        cfg, split="val", normalizer=normalizer, mode="eval",
        gap_seconds=0.0, max_records=args.max_val, seed=cfg.seed,
    )
    print(f"  train:  {len(train_ds)} records")
    print(f"  val:    {len(val_ds)} records")

    batch_size = args.batch_size or cfg.batch_size
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
    )

    print("\nepoch 0 (untrained, random init -> low score expected) ...")
    m0 = validate(
        model, val_loader, normalizer, cfg,
        recon_idx, target_idx, tau_n, tau_m, device, desc="val@0",
    )
    print(f"  {args.model}@epoch0:  {fmt_metrics(m0)}")

    epochs = args.epochs or cfg.max_epochs
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    curve = []
    best_score = m0["val_score"]
    best_epoch = 0
    no_improve = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"train@{epoch}", leave=False):
            Xt = batch["x_test"].to(device).transpose(1, 2).float()
            Yt = batch["y_test"].to(device).transpose(1, 2).float()
            Y_pred = model(Xt)
            loss = recon_mse(Y_pred, Yt, recon_idx)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            train_losses.append(float(loss.item()))
        scheduler.step()
        train_loss = float(np.mean(train_losses))

        m = validate(
            model, val_loader, normalizer, cfg,
            recon_idx, target_idx, tau_n, tau_m, device,
            desc=f"val@{epoch}",
        )
        curve.append({"epoch": epoch, "train_loss": train_loss, **m,
                      "lr": optimizer.param_groups[0]["lr"]})

        marker = ""
        if m["val_score"] > best_score:
            best_score = m["val_score"]
            best_epoch = epoch
            no_improve = 0
            marker = " *best"
            torch.save({
                "model":   model.state_dict(),
                "epoch":   epoch,
                "metrics": m,
                "cfg":     vars(cfg),
                "arch":    args.model,
            }, artifact_dir / f"{tag}_best.pth")
        else:
            no_improve += 1

        print(f"epoch {epoch:3d}  train_loss={train_loss:.5f}  "
              f"{fmt_metrics(m)}{marker}")
        if no_improve >= cfg.early_stop_patience:
            print(f"  early stop (patience={cfg.early_stop_patience} exceeded)")
            break

    df = pd.DataFrame(curve)
    curve_path = results_dir / f"{tag}_training_curve.csv"
    df.to_csv(curve_path, index=False)
    print(f"\nsaved training curve to {curve_path}")

    print()
    print("=" * 88)
    print(f"Final summary (best epoch = {best_epoch})")
    print("-" * 88)
    if best_epoch >= 1:
        best = df.iloc[best_epoch - 1]
        print(f"  {args.model} (best):  val_score={best['val_score']:.4f}  "
              f"PCC={best['pcc_recon']:.4f}  nrmse={best['nrmse_recon']:.4f}  "
              f"morph={best['morph_err_norm']:.4f}  "
              f"ST60_ant={best['st60_anterior_mv']:.4f} mV")
    print(f"  PCM:           val_score={pcm_val_score:.4f}")
    print(f"  GL:            val_score={gl_val_score:.4f}")
    if best_epoch >= 1:
        print(f"  {args.model} - PCM:   {best['val_score'] - pcm_val_score:+.4f}")
        print(f"  {args.model} - GL:    {best['val_score'] - gl_val_score:+.4f}")
    print(f"\nelapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
