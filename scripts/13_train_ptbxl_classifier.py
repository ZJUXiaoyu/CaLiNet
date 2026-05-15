"""Step 2.A: Train PTB-XL 5-class diagnostic classifier (Ribeiro-style 1D ResNet).

Pre-committed protocol — DO NOT change without explicit approval:
  Input:        12 leads × 5000 samples @ 500 Hz
  Architecture: Ribeiro 2020 NatComm 1D residual CNN
                init conv (kernel=17, 64 ch) → 4 residual blocks
                (channels 64→128→196→256→320, stride-4 downsample each)
                → BN+ReLU+global avg pool → Linear → 5 sigmoid units
  Labels:       only records with at least one likelihood=100 SCP statement
                that maps to a super-class (NORM/MI/STTC/CD/HYP).
  Loss:         BCEWithLogitsLoss
  Optimizer:    AdamW lr=1e-3 cosine→1e-5
  Batch size:   32
  Epochs:       50 (early stop on internal 10% val loss; patience=10)
  Augmentation: NONE
  Preprocessing: bandpass 0.05-50 Hz + GlobalNormalizer (fold 1-8 stats)
  Random seed:  42

Threshold optimization (after training):
  On the same 10% val split, grid-search per-class threshold over
  [0.05, 0.95] step 0.01, maximize per-class F1 independently.

Outputs:
  checkpoints/ptbxl_classifier_best.pth
  checkpoints/ptbxl_thresholds.npy
  results/ptbxl_classifier_training_log.csv
"""
from __future__ import annotations

import ast
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, '.')
from calinet.config import CaLiNetConfig
from calinet.data.normalizer import GlobalNormalizer
from calinet.data.ptbxl import (
    PTBXL_LEAD_ORDER, get_split_metadata, iter_records, load_metadata,
)

SUPER_CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']


# ----------------------------------------------------------------------
# Architecture
# ----------------------------------------------------------------------
class ResidualUnit(nn.Module):
    """Ribeiro-style pre-activation residual block with downsampling."""
    def __init__(self, n_in: int, n_out: int, kernel: int = 17,
                 dropout: float = 0.2, downsample: int = 4):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(n_in)
        self.conv1 = nn.Conv1d(n_in, n_out, kernel, padding=kernel // 2)
        self.bn2 = nn.BatchNorm1d(n_out)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_out, n_out, kernel, stride=downsample,
                               padding=kernel // 2)
        self.skip_pool = nn.MaxPool1d(downsample) if downsample > 1 else nn.Identity()
        self.skip_conv = (
            nn.Conv1d(n_in, n_out, 1) if n_in != n_out else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(x))
        h = self.conv1(h)
        h = F.relu(self.bn2(h))
        h = self.dropout(h)
        h = self.conv2(h)

        skip = self.skip_pool(x)
        skip = self.skip_conv(skip)

        if h.shape[-1] != skip.shape[-1]:
            m = min(h.shape[-1], skip.shape[-1])
            h = h[..., :m]
            skip = skip[..., :m]
        return h + skip


class RibeiroClassifier(nn.Module):
    """1D residual CNN (5-class multi-label) per Ribeiro 2020."""
    def __init__(self, n_classes: int = 5, n_leads: int = 12,
                 kernel_size: int = 17, dropout: float = 0.2):
        super().__init__()
        self.init_conv = nn.Conv1d(n_leads, 64, kernel_size,
                                   padding=kernel_size // 2)
        self.init_bn = nn.BatchNorm1d(64)
        self.block1 = ResidualUnit(64,  128, kernel_size, dropout, 4)
        self.block2 = ResidualUnit(128, 196, kernel_size, dropout, 4)
        self.block3 = ResidualUnit(196, 256, kernel_size, dropout, 4)
        self.block4 = ResidualUnit(256, 320, kernel_size, dropout, 4)
        self.final_bn = nn.BatchNorm1d(320)
        self.head = nn.Linear(320, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_leads, T)
        h = F.relu(self.init_bn(self.init_conv(x)))
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = F.relu(self.final_bn(h))
        h = F.adaptive_avg_pool1d(h, 1).squeeze(-1)
        return self.head(h)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def filter_to_likelihood100_with_super(metadata: pd.DataFrame, scp_to_super: dict) -> pd.DataFrame:
    """Keep only records with >=1 SCP statement at likelihood=100 mapping to a super-class."""
    SC = set(SUPER_CLASSES)

    def super_vec(scp_dict: dict) -> np.ndarray:
        v = np.zeros(5, dtype=np.float32)
        for code, lik in scp_dict.items():
            if float(lik) != 100.0:
                continue
            sc = scp_to_super.get(code)
            if isinstance(sc, str) and sc in SC:
                v[SUPER_CLASSES.index(sc)] = 1.0
        return v

    md = metadata.copy()
    md['super_vec'] = md['scp_codes'].apply(super_vec)
    md['n_super'] = md['super_vec'].apply(lambda v: int(v.sum()))
    md = md[md['n_super'] >= 1].copy()
    return md


def load_signals_into_memory(
    ptbxl_path: Path,
    metadata: pd.DataFrame,
    sampling_rate: int,
    filter_kwargs: dict | None,
    normalizer: GlobalNormalizer,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Load + preprocess + normalize all records into a contiguous tensor.

    Returns (X, Y, ids) where X is (N, 12, n_samples) channel-first float32.
    """
    X_list, Y_list, id_list = [], [], []
    for ecg_id, sig, row in tqdm(
        iter_records(ptbxl_path, metadata, sampling_rate, filter_kwargs),
        total=len(metadata), desc='load+norm', leave=False,
    ):
        if sig.shape[0] < n_samples:
            continue
        sig = sig[:n_samples]
        sig_n = normalizer(sig).astype(np.float32)        # (T, 12)
        X_list.append(sig_n.T.copy())                      # (12, T)
        Y_list.append(metadata.loc[ecg_id, 'super_vec'])
        id_list.append(int(ecg_id))
    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0).astype(np.float32)
    return X, Y, id_list


class TensorMultilabelDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int):
        return self.X[i], self.Y[i]


# ----------------------------------------------------------------------
# Train + threshold opt
# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             criterion: nn.Module) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses, all_logits, all_y = [], [], []
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        logits = model(x)
        losses.append(criterion(logits, y).item())
        all_logits.append(logits.cpu().numpy())
        all_y.append(y.cpu().numpy())
    return (
        float(np.mean(losses)),
        np.concatenate(all_logits, 0),
        np.concatenate(all_y, 0),
    )


def optimize_thresholds(probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    thresholds = np.zeros(probs.shape[1], dtype=np.float32)
    for c in range(probs.shape[1]):
        best_t, best_f1 = 0.5, -1.0
        for t in np.arange(0.05, 0.96, 0.01):
            pred = (probs[:, c] > t).astype(int)
            f1 = f1_score(y_true[:, c], pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        thresholds[c] = float(best_t)
    return thresholds


def main():
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    with open('configs/default.yaml', encoding='utf-8') as f:
        d = yaml.safe_load(f)
    for k in ("input_leads", "target_leads", "train_gap_seconds",
              "unet_channels", "film_layers", "aug_noise_snr_db",
              "aug_drift_hz", "aug_amp_scale", "train_folds",
              "ridge_lambda_W_grid", "ridge_lambda_b_grid"):
        if k in d and isinstance(d[k], list):
            d[k] = tuple(d[k])
    cfg = CaLiNetConfig(**d)
    artifact_dir = Path(cfg.artifact_dir)
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    # --- Metadata ---
    print('loading PTB-XL metadata...')
    md = load_metadata(cfg.ptbxl_path)
    scp_df = pd.read_csv(Path(cfg.ptbxl_path) / 'scp_statements.csv', index_col=0)
    scp_to_super = scp_df[scp_df['diagnostic'] == 1]['diagnostic_class'].to_dict()
    md = filter_to_likelihood100_with_super(md, scp_to_super)
    print(f'  records (likelihood=100, has super-class): {len(md)}')

    train_md = get_split_metadata(md, cfg.train_folds)            # folds 1-8
    val_md   = get_split_metadata(md, cfg.val_fold)               # fold 9
    print(f'  fold 1-8 records: {len(train_md)}')
    print(f'  fold 9   records: {len(val_md)}')

    # Class distribution
    Y_train_dist = np.stack(train_md['super_vec'].values).sum(axis=0)
    print(f'  fold 1-8 per-class positives: {dict(zip(SUPER_CLASSES, Y_train_dist.astype(int).tolist()))}')

    # --- Preprocess (cached) ---
    cache_path = artifact_dir / 'ptbxl_train_cache.npz'
    filter_kwargs = {
        'low_hz':  cfg.bandpass_low_hz,
        'high_hz': cfg.bandpass_high_hz,
        'fs':      cfg.sampling_rate,
        'order':   cfg.bandpass_order,
    } if cfg.use_bandpass else None
    normalizer = GlobalNormalizer.load(artifact_dir / 'normalizer.npz')
    n_samples = 5000   # 10s @ 500Hz

    if cache_path.exists():
        print(f'  loading cache from {cache_path.name} ...')
        z = np.load(cache_path)
        X_all, Y_all, ids_all = z['X'], z['Y'], z['ids'].tolist()
    else:
        print('  loading + preprocessing fold 1-8 signals ...')
        X_all, Y_all, ids_all = load_signals_into_memory(
            Path(cfg.ptbxl_path), train_md, cfg.sampling_rate,
            filter_kwargs, normalizer, n_samples,
        )
        np.savez(cache_path, X=X_all, Y=Y_all, ids=np.array(ids_all))
        print(f'  cached: {cache_path}  shape={X_all.shape}')
    print(f'  fold 1-8 loaded: X={X_all.shape}  Y={Y_all.shape}')

    # --- Internal 90/10 split (seeded) ---
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X_all))
    n_val = int(0.10 * len(X_all))
    val_idx = idx[:n_val]
    tr_idx  = idx[n_val:]
    X_tr, Y_tr = X_all[tr_idx], Y_all[tr_idx]
    X_va, Y_va = X_all[val_idx], Y_all[val_idx]
    print(f'  train: {len(X_tr)}  internal val: {len(X_va)}')

    train_ds = TensorMultilabelDataset(X_tr, Y_tr)
    val_ds   = TensorMultilabelDataset(X_va, Y_va)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False,
                              num_workers=0)

    # --- Model ---
    model = RibeiroClassifier(n_classes=5, n_leads=12, kernel_size=17,
                              dropout=0.2).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  model params: {n_params:,}')

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

    # --- Training loop ---
    best_val = float('inf')
    best_state = None
    no_improve = 0
    log = []
    t0 = time.time()
    for epoch in range(1, 51):
        model.train()
        losses = []
        for x, y in tqdm(train_loader, desc=f'train@{epoch}', leave=False):
            x = x.to(device); y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        train_loss = float(np.mean(losses))
        val_loss, val_logits, val_y = evaluate(model, val_loader, device, criterion)
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))
        # quick macro F1 at threshold 0.5 just for monitoring
        val_f1_05 = f1_score(val_y, (val_probs > 0.5).astype(int),
                             average='macro', zero_division=0)

        marker = ''
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            marker = ' *best'
        else:
            no_improve += 1
        log.append({
            'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
            'val_f1_macro_t05': val_f1_05,
            'lr': optimizer.param_groups[0]['lr'],
        })
        print(f'epoch {epoch:3d}  train_loss={train_loss:.4f}  '
              f'val_loss={val_loss:.4f}  val_F1@0.5={val_f1_05:.4f}{marker}')
        if no_improve >= 10:
            print(f'  early stop @ epoch {epoch} (patience 10)')
            break

    pd.DataFrame(log).to_csv(results_dir / 'ptbxl_classifier_training_log.csv', index=False)

    # --- Restore best + threshold optimization on internal val ---
    print(f'\nrestoring best (val_loss={best_val:.4f} @ epoch {best_epoch})')
    model.load_state_dict(best_state)
    _, val_logits, val_y = evaluate(model, val_loader, device, criterion)
    val_probs = 1.0 / (1.0 + np.exp(-val_logits))

    thresholds = optimize_thresholds(val_probs, val_y)
    binary = (val_probs > thresholds).astype(int)
    val_f1_macro_opt = f1_score(val_y, binary, average='macro', zero_division=0)
    val_f1_per_class = [
        f1_score(val_y[:, c], binary[:, c], zero_division=0)
        for c in range(5)
    ]

    print('\nFitted thresholds (per class, fold 1-8 internal val):')
    for cls, t, f1c in zip(SUPER_CLASSES, thresholds, val_f1_per_class):
        print(f'  {cls:6s} threshold={t:.2f}   F1={f1c:.4f}')
    print(f'\n  internal-val macro F1 (with opt thresholds): {val_f1_macro_opt:.4f}')

    # --- Save ---
    torch.save({
        'model':       best_state,
        'epoch':       best_epoch,
        'val_loss':    best_val,
        'val_f1_macro': val_f1_macro_opt,
        'classes':     SUPER_CLASSES,
        'arch_kwargs': {'n_classes': 5, 'n_leads': 12, 'kernel_size': 17, 'dropout': 0.2},
        'cfg':         vars(cfg),
        'seed':        SEED,
    }, artifact_dir / 'ptbxl_classifier_best.pth')
    np.save(artifact_dir / 'ptbxl_thresholds.npy', thresholds)

    print(f'\nSaved: {artifact_dir / "ptbxl_classifier_best.pth"}')
    print(f'Saved: {artifact_dir / "ptbxl_thresholds.npy"}')
    print(f'\nElapsed: {(time.time()-t0)/60:.1f} min')

    # --- Pre-committed sanity gate ---
    if val_f1_macro_opt < 0.70:
        print(f'\n[STOP] internal-val macro F1 ({val_f1_macro_opt:.4f}) < 0.70')
        print('Per protocol: do NOT proceed to Step 2.B. Debug classifier.')
    else:
        print(f'\n[OK] internal-val macro F1 ({val_f1_macro_opt:.4f}) >= 0.70')
        print('Proceed to Step 2.B (validity check on fold 9 originals).')


if __name__ == '__main__':
    main()
