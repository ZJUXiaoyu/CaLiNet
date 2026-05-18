"""Step 4.4-4.5: Train UNetWithAnchor backbone on PTB-XL.

UNetWithAnchor forward:
    Y_pred = X · W_global + b_global + R_θ(X)

Locked training design (v1.2):
  - Loss: MSE on RECONSTRUCTED leads only (V1, V3, V4, V5, V6).
          input + derivable leads are NOT in the loss.
  - Validation: run full val fold each epoch, compute
          PCC_recon, nrmse_recon, morph_err_norm (R_amp + ST60_ant + T_amp)
          val_score = 0.3·PCC + 0.4·exp(-nrmse/tau_n) + 0.3·exp(-morph/tau_m)
          where tau_n, tau_m come from checkpoints/val_score_tau.npz.
  - Sanity at epoch 0 (before training): val_score should ≈ GL baseline
          (because backbone final conv is zero-init → Y_pred = X·W_global).
          Refuse to train if this check fails.
  - Early stopping on val_score with patience 8.
  - AdamW + CosineAnnealingLR, grad_clip=1.0.

Artifacts written:
    checkpoints/unet_anchor_best.pth           best model state_dict + cfg snapshot
    results/unet_anchor_training_curve.csv     per-epoch metrics
    results/unet_anchor_val_final.txt          final val summary vs GL/PCM

Usage:
    python scripts/07_train_unet_anchor.py
    python scripts/07_train_unet_anchor.py --max_train 500 --max_val 200 --epochs 2   # smoke
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
from calinet.data.normalizer import GlobalNormalizer
from calinet.eval.metrics import per_lead_pcc
from calinet.eval.morphology import classify_leads, compute_morphology
from calinet.models.unet_anchor import UNetWithAnchor


# ----------------------------------------------------------------------
# CLI / config
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--epochs", type=int, default=None, help="override cfg.max_epochs")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max_train", type=int, default=None)
    p.add_argument("--max_val",   type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--tag", default="unet_anchor")
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
# Loss / validation
# ----------------------------------------------------------------------
def recon_mse(Y_pred: torch.Tensor, Y_true: torch.Tensor,
              recon_idx: list[int]) -> torch.Tensor:
    """MSE on reconstructed leads only. Tensors shape (B, T, 12)."""
    return F.mse_loss(Y_pred[:, :, recon_idx], Y_true[:, :, recon_idx])


def _morph_err_norm(report) -> float:
    """R_amp/1.0 + ST60_ant/0.1 + T_amp/0.3 — PCM-anchored composite."""
    if report is None:
        return float("nan")
    return (report.r_amp_err_main / 1.0
            + report.st_j60_anterior_main / 0.1
            + report.t_amp_err_main / 0.3)


@torch.no_grad()
def validate(
    model: UNetWithAnchor,
    val_loader: DataLoader,
    normalizer: GlobalNormalizer,
    cfg: CaLiNetConfig,
    recon_idx: list[int],
    target_idx: list[int],
    tau_n: float, tau_m: float,
    device: torch.device,
    desc: str = "val",
) -> dict:
    """Run full val pass; return dict of metrics + val_score."""
    model.eval()
    mu    = normalizer.mu[target_idx]
    sigma = normalizer.sigma[target_idx]

    pccs, nrmses, morphs = [], [], []
    r_amp_vals, st_vals, t_amp_vals = [], [], []

    for batch in tqdm(val_loader, desc=desc, leave=False):
        Xt = batch["x_test"].to(device).transpose(1, 2).float()    # (B, T, n_in)
        Yt = batch["y_test"].to(device).transpose(1, 2).float()    # (B, T, n_out)

        Y_pred = model(Xt)                                         # (B, T, n_out)

        # PCC / nrmse on reconstructed leads (normalized space)
        Yt_np = Yt.cpu().numpy()
        Yp_np = Y_pred.cpu().numpy()
        for b in range(Yt_np.shape[0]):
            pcc = per_lead_pcc(Yt_np[b], Yp_np[b])
            pccs.append(float(np.mean(pcc[recon_idx])))

            diff = Yt_np[b, :, recon_idx] - Yp_np[b, :, recon_idx]
            num = float(np.sqrt((diff ** 2).mean()))
            den = float(np.sqrt((Yt_np[b, :, recon_idx] ** 2).mean()))
            nrmses.append(num / max(den, 1e-8))

            # Invert to mV for morphology
            Yt_mv = Yt_np[b] * sigma + mu
            Yp_mv = Yp_np[b] * sigma + mu
            rep = compute_morphology(
                Yt_mv, Yp_mv,
                cfg.target_leads, cfg.input_leads, cfg.sampling_rate,
            )
            morph = _morph_err_norm(rep)
            if not np.isnan(morph):
                morphs.append(morph)
                r_amp_vals.append(rep.r_amp_err_main)
                st_vals.append(rep.st_j60_anterior_main)
                t_amp_vals.append(rep.t_amp_err_main)

    pcc_m   = float(np.nanmean(pccs))
    nrmse_m = float(np.nanmean(nrmses))
    morph_m = float(np.nanmean(morphs)) if morphs else float("nan")
    r_amp_m = float(np.nanmean(r_amp_vals)) if r_amp_vals else float("nan")
    st_m    = float(np.nanmean(st_vals))    if st_vals    else float("nan")
    t_amp_m = float(np.nanmean(t_amp_vals)) if t_amp_vals else float("nan")

    val_score = (
        0.3 * pcc_m
        + 0.4 * float(np.exp(-nrmse_m / tau_n))
        + 0.3 * float(np.exp(-morph_m / tau_m))
    )

    return {
        "pcc_recon":         pcc_m,
        "nrmse_recon":       nrmse_m,
        "morph_err_norm":    morph_m,
        "r_amp_err_mv":      r_amp_m,
        "st60_anterior_mv":  st_m,
        "t_amp_err_mv":      t_amp_m,
        "val_score":         val_score,
    }


def fmt_metrics(m: dict) -> str:
    return (f"score={m['val_score']:.4f}  "
            f"PCC={m['pcc_recon']:.4f}  nrmse={m['nrmse_recon']:.4f}  "
            f"morph={m['morph_err_norm']:.4f}  "
            f"(R_amp={m['r_amp_err_mv']:.4f}  "
            f"ST60_ant={m['st60_anterior_mv']:.4f}  "
            f"T_amp={m['t_amp_err_mv']:.4f})")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    artifact_dir = Path(cfg.artifact_dir)
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Load anchored tau from val_score_tau.npz ---
    tau_path = artifact_dir / "val_score_tau.npz"
    if not tau_path.exists():
        raise FileNotFoundError(
            f"{tau_path} not found — run scripts/06_anchor_val_tau.py first."
        )
    tz = np.load(tau_path)
    tau_n = float(tz["tau_n"])
    tau_m = float(tz["tau_m"])

    # Measured GL / PCM val_scores (written by 06_anchor_val_tau.py).
    # No estimation fallback — epoch-0 sanity must compare against a
    # value computed with the same aggregation as validate().
    if "gl_val_score" not in tz.files or "pcm_val_score" not in tz.files:
        raise RuntimeError(
            f"{tau_path} is missing measured gl_val_score / pcm_val_score. "
            f"Re-run scripts/06_anchor_val_tau.py to refresh it."
        )
    gl_val_score  = float(tz["gl_val_score"])
    pcm_val_score = float(tz["pcm_val_score"])

    # --- Normalizer ---
    normalizer = GlobalNormalizer.load(artifact_dir / "normalizer.npz")

    # --- Build model ---
    model = UNetWithAnchor.from_artifacts(
        artifact_dir,
        n_in=len(cfg.input_leads),
        n_out=len(cfg.target_leads),
        channels=cfg.unet_channels,
        embedding_dim=cfg.embedding_dim,
        pad_to_multiple=cfg.pad_to_multiple,
    ).to(device)
    n_params = model.n_params()

    # --- Lead indices ---
    from calinet.data.episodes import resolve_lead_indices
    _, target_idx = resolve_lead_indices(cfg.input_leads, cfg.target_leads)
    groups = classify_leads(cfg.target_leads, cfg.input_leads)
    recon_idx = [cfg.target_leads.index(n) for n in groups["reconstructed"]]

    # --- Print header ---
    print("=" * 88)
    print(f"Step 4.5 - Train UNetWithAnchor  (tag={args.tag}, device={device})")
    print("=" * 88)
    print(f"  params:            {n_params:,}")
    print(f"  reconstructed:     {groups['reconstructed']} (idx={recon_idx})")
    print(f"  anchored tau:      tau_n={tau_n:.4f}  tau_m={tau_m:.4f}")
    print(f"  GL  val_score:     {gl_val_score:.4f}")
    print(f"  PCM val_score:     {pcm_val_score:.4f}")
    print()

    # --- Datasets ---
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

    # --- Epoch 0 sanity check ---
    print("\nepoch 0 sanity (untrained, backbone ~ zero-init -> should ≈ GL) ...")
    m0 = validate(
        model, val_loader, normalizer, cfg,
        recon_idx, target_idx, tau_n, tau_m, device, desc="val@0",
    )
    print(f"  UNetWithAnchor@epoch0:  {fmt_metrics(m0)}")
    print(f"  GL baseline:  val_score={gl_val_score:.4f}")
    delta = m0["val_score"] - gl_val_score
    if abs(delta) > 0.01:
        print(f"  [!!] epoch-0 val_score deviates from GL by {delta:+.4f}")
        print(f"      expected: |delta| < 0.01.")
        print(f"      likely cause: backbone final-conv not zero-initialized, "
              f"or linear branch wiring differs from GL.")
        print(f"      refusing to train. fix and rerun.")
        return
    print(f"  [OK] delta = {delta:+.4f} (< 0.01)")

    # --- Optimizer / scheduler ---
    epochs = args.epochs or cfg.max_epochs
    lr = args.lr or cfg.lr
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # --- Training loop ---
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
            }, artifact_dir / f"{args.tag}_best.pth")
        else:
            no_improve += 1

        print(f"epoch {epoch:3d}  train_loss={train_loss:.5f}  {fmt_metrics(m)}{marker}")

        if no_improve >= cfg.early_stop_patience:
            print(f"  early stop (patience={cfg.early_stop_patience} exceeded)")
            break

    # --- Save curve ---
    df = pd.DataFrame(curve)
    curve_path = results_dir / f"{args.tag}_training_curve.csv"
    df.to_csv(curve_path, index=False)
    print(f"\nsaved training curve to {curve_path}")

    # --- Final summary ---
    print()
    print("=" * 88)
    print(f"Final summary (best epoch = {best_epoch})")
    print("-" * 88)
    best = df.iloc[best_epoch - 1] if best_epoch >= 1 else None
    if best is not None:
        print(f"  UNetWithAnchor (best):  val_score={best['val_score']:.4f}  "
              f"PCC={best['pcc_recon']:.4f}  nrmse={best['nrmse_recon']:.4f}  "
              f"morph={best['morph_err_norm']:.4f}  "
              f"ST60_ant={best['st60_anterior_mv']:.4f} mV")
    print(f"  PCM:          val_score={pcm_val_score:.4f}")
    print(f"  GL:           val_score={gl_val_score:.4f}")
    print()
    if best is not None:
        dpcm = best['val_score'] - pcm_val_score
        dgl  = best['val_score'] - gl_val_score
        print(f"  UNetWithAnchor - PCM:   {dpcm:+.4f}  (CaLiNet-E framing signal)")
        print(f"  UNetWithAnchor - GL:    {dgl:+.4f}")
    print(f"\nelapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
