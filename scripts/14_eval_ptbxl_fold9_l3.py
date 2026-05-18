"""Step 2.B + 2.C + 2.D: PTB-XL fold 9 diagnostic-level evaluation.

Loads the trained PTB-XL classifier (Step 2.A) and runs:
  2.B  Validity gate: classifier on fold 9 ORIGINAL 12-lead signals.
       Macro F1 must fall in [0.70, 0.90]; outside this range = STOP.
  2.C  7-method evaluation: reconstruct fold 9 records with each
       method (Original / GL / Transformer / 1D U-Net / 1D U-Net w/ anchor /
       PCM / CaLiNet-E), run trained classifier, compute F1.
  2.D  Bootstrap 95% CI (1000 iter, paired resampling) + paired
       CaLiNet-E vs PCM delta + p(delta>0).

Pre-committed rules:
  - Architecture: not changed.
  - Thresholds: pre-fit on fold 1-8 internal val (Step 2.A); not refit here.
  - Bootstrap seed: 42.
  - Adverse results are reported, not fixed by changing protocol.

Outputs:
  results/diagnostic_all_methods_ptbxl.csv
  results/diagnostic_bootstrap_ci_ptbxl.csv
  results/l3_two_dataset_comparison.txt
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from tqdm import tqdm

sys.path.insert(0, '.')
from calinet.config import CaLiNetConfig
from calinet.data.normalizer import GlobalNormalizer
from calinet.data.episodes import resolve_lead_indices
from calinet.data.ptbxl import (
    get_split_metadata, iter_records, load_metadata,
)
from calinet.models.calibration import apply_transform, calibrate_patient
from calinet.models.calinet_e import CaLiNetE
from calinet.models.unet_anchor import UNetWithAnchor
from calinet.models.baselines import CNNBaseline, TransformerBaseline

# Re-import classifier from training script (same architecture)
sys.path.insert(0, 'scripts')
import importlib.util
_spec = importlib.util.spec_from_file_location(
    'ptbxl_train_module', 'scripts/13_train_ptbxl_classifier.py'
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
RibeiroClassifier = _mod.RibeiroClassifier
filter_to_likelihood100_with_super = _mod.filter_to_likelihood100_with_super
SUPER_CLASSES = _mod.SUPER_CLASSES


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

    # ------------------------------------------------------------------
    # Load classifier + thresholds
    # ------------------------------------------------------------------
    cls_path = artifact_dir / 'ptbxl_classifier_best.pth'
    th_path  = artifact_dir / 'ptbxl_thresholds.npy'
    if not cls_path.exists() or not th_path.exists():
        raise FileNotFoundError(
            'Classifier or thresholds not found — run Step 2.A first.'
        )
    ckpt = torch.load(cls_path, map_location=device)
    classifier = RibeiroClassifier(**ckpt['arch_kwargs']).to(device)
    classifier.load_state_dict(ckpt['model'])
    classifier.eval()
    thresholds = np.load(th_path).astype(np.float32)
    print(f'Loaded classifier (epoch {ckpt["epoch"]}, val_loss={ckpt["val_loss"]:.4f})')
    print(f'Thresholds: {dict(zip(SUPER_CLASSES, thresholds.round(2).tolist()))}')

    # ------------------------------------------------------------------
    # Fold 9 metadata
    # ------------------------------------------------------------------
    print('\nLoading PTB-XL fold 9 ...')
    md = load_metadata(cfg.ptbxl_path)
    scp_df = pd.read_csv(Path(cfg.ptbxl_path) / 'scp_statements.csv', index_col=0)
    scp_to_super = scp_df[scp_df['diagnostic'] == 1]['diagnostic_class'].to_dict()
    md = filter_to_likelihood100_with_super(md, scp_to_super)
    fold9 = get_split_metadata(md, cfg.val_fold)
    print(f'  fold 9 records: {len(fold9)}')

    # Load + preprocess fold 9 signals
    cache_path = artifact_dir / 'ptbxl_fold9_cache.npz'
    normalizer = GlobalNormalizer.load(artifact_dir / 'normalizer.npz')
    filter_kwargs = {
        'low_hz':  cfg.bandpass_low_hz,
        'high_hz': cfg.bandpass_high_hz,
        'fs':      cfg.sampling_rate,
        'order':   cfg.bandpass_order,
    } if cfg.use_bandpass else None
    n_samples = 5000

    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        X9 = z['X']        # (N, 12, 5000) channel-first, normalized
        Y9 = z['Y']        # (N, 5)
        ids9 = z['ids'].tolist()
        print(f'  loaded cache: X9={X9.shape}')
    else:
        X9_list, Y9_list, ids9 = [], [], []
        for ecg_id, sig, _row in tqdm(
            iter_records(cfg.ptbxl_path, fold9, cfg.sampling_rate, filter_kwargs),
            total=len(fold9), desc='load fold9', leave=False,
        ):
            if sig.shape[0] < n_samples:
                continue
            sig = sig[:n_samples]
            sig_n = normalizer(sig).astype(np.float32)
            X9_list.append(sig_n.T.copy())
            Y9_list.append(fold9.loc[ecg_id, 'super_vec'])
            ids9.append(int(ecg_id))
        X9 = np.stack(X9_list, axis=0)
        Y9 = np.stack(Y9_list, axis=0).astype(np.float32)
        np.savez(cache_path, X=X9, Y=Y9, ids=np.array(ids9))
        print(f'  cached: {cache_path}  shape={X9.shape}')

    # ------------------------------------------------------------------
    # Step 2.B: classifier validity on fold 9 originals
    # ------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('Step 2.B: classifier validity on fold 9 ORIGINAL signals')
    print('=' * 70)

    @torch.no_grad()
    def predict_classifier(X_tensor: np.ndarray, batch=64) -> np.ndarray:
        """Apply classifier on (N, 12, 5000) numpy → (N, 5) sigmoid probs."""
        probs = []
        for i in range(0, len(X_tensor), batch):
            x = torch.from_numpy(X_tensor[i:i + batch]).to(device).float()
            p = torch.sigmoid(classifier(x)).cpu().numpy()
            probs.append(p)
        return np.concatenate(probs, axis=0)

    probs_orig = predict_classifier(X9)
    binary_orig = (probs_orig > thresholds).astype(int)
    f1_orig = f1_score(Y9, binary_orig, average='macro', zero_division=0)
    f1_orig_per_class = [
        f1_score(Y9[:, c], binary_orig[:, c], zero_division=0)
        for c in range(5)
    ]
    print(f'\n  Original macro F1 = {f1_orig:.4f}')
    for cls, f in zip(SUPER_CLASSES, f1_orig_per_class):
        print(f'    {cls:6s} F1 = {f:.4f}')

    if f1_orig < 0.70:
        print(f'\n[STOP] classifier F1 ({f1_orig:.4f}) < 0.70 — too weak. Debug before continuing.')
        return
    if f1_orig > 0.90:
        print(f'\n[STOP] classifier F1 ({f1_orig:.4f}) > 0.90 — possible leakage. Investigate.')
        return
    print(f'\n  [OK] F1 in [0.70, 0.90] — proceed to Step 2.C')

    # ------------------------------------------------------------------
    # Step 2.C: 7-method reconstruction + classification
    # ------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('Step 2.C: reconstruct fold 9 with 6 methods + classify')
    print('=' * 70)

    # --- Load reconstruction models ---
    gw = np.load(artifact_dir / 'global_W.npz')
    W_global = gw['W_global'].astype(np.float32)
    b_global = gw['b_global'].astype(np.float32)

    cnn_model = CNNBaseline(n_in=len(cfg.input_leads), n_out=len(cfg.target_leads),
                             channels=cfg.unet_channels, pad_to_multiple=cfg.pad_to_multiple).to(device)
    cnn_model.load_state_dict(torch.load(artifact_dir / 'cnn_baseline_best.pth', map_location=device)['model'])
    cnn_model.eval()

    tr_model = TransformerBaseline(n_in=len(cfg.input_leads), n_out=len(cfg.target_leads)).to(device)
    tr_model.load_state_dict(torch.load(artifact_dir / 'transformer_baseline_best.pth', map_location=device)['model'])
    tr_model.eval()

    unet_anchor_model = UNetWithAnchor.from_artifacts(artifact_dir, n_in=len(cfg.input_leads), n_out=len(cfg.target_leads),
                                      channels=cfg.unet_channels, embedding_dim=cfg.embedding_dim,
                                      pad_to_multiple=cfg.pad_to_multiple).to(device)
    unet_anchor_model.load_state_dict(torch.load(artifact_dir / 'unet_anchor_full_best.pth', map_location=device)['model'])
    unet_anchor_model.eval()

    calinet_model = CaLiNetE.from_artifacts(artifact_dir, n_in=len(cfg.input_leads), n_out=len(cfg.target_leads),
                                             channels=cfg.unet_channels, embedding_dim=cfg.embedding_dim,
                                             pad_to_multiple=cfg.pad_to_multiple, sampling_rate=cfg.sampling_rate,
                                             lam_W=cfg.ridge_lambda_W, lam_b=cfg.ridge_lambda_b,
                                             force_rho_one=False).to(device)
    calinet_model.load_state_dict(torch.load(artifact_dir / 'calinet_e_full_best.pth', map_location=device)['model'])
    calinet_model.eval()

    in_idx, target_idx = resolve_lead_indices(cfg.input_leads, cfg.target_leads)

    methods = ['GL', 'Transformer', '1D U-Net', '1D U-Net w/ anchor', 'PCM', 'CaLiNet-E']
    # X9 is (N, 12, 5000) normalized channel-first. Build one reconstructed
    # tensor per method by replacing samples [calib_samples:5000] with the
    # method's reconstruction (still in normalized space; classifier ate
    # normalized inputs in training so this is the right space).
    recon_arrays = {m: X9.copy() for m in methods}

    print(f'\nReconstructing fold 9 ({len(X9)} records) ...')
    with torch.no_grad():
        for idx in tqdm(range(len(X9)), desc='records', leave=False):
            sig_norm = X9[idx].T  # (5000, 12) in normalized space
            Xc = sig_norm[:cfg.calib_samples][:, in_idx]  # (Lc, n_in)
            Yc = sig_norm[:cfg.calib_samples][:, target_idx]  # (Lc, n_out)

            W_i, b_i = calibrate_patient(
                Xc, Yc, W_global, b_global,
                lam_W=cfg.ridge_lambda_W, lam_b=cfg.ridge_lambda_b,
            )

            chunks = {m: [] for m in methods}
            pos = cfg.calib_samples
            while pos + cfg.target_samples <= n_samples:
                Xt_np = sig_norm[pos:pos + cfg.target_samples][:, in_idx]
                Xt_t = torch.from_numpy(Xt_np)[None].to(device).float()

                chunks['GL'].append(apply_transform(Xt_np, W_global, b_global))
                chunks['PCM'].append(apply_transform(Xt_np, W_i, b_i))
                chunks['1D U-Net'].append(cnn_model(Xt_t).squeeze(0).cpu().numpy())
                chunks['Transformer'].append(tr_model(Xt_t).squeeze(0).cpu().numpy())
                chunks['1D U-Net w/ anchor'].append(unet_anchor_model(Xt_t).squeeze(0).cpu().numpy())

                batch = {
                    'x_calib': torch.from_numpy(Xc.T)[None].to(device).float(),
                    'y_calib': torch.from_numpy(Yc.T)[None].to(device).float(),
                    'x_test':  Xt_t.transpose(1, 2),
                    'y_test':  torch.zeros(1, len(cfg.target_leads), cfg.target_samples).to(device),
                }
                chunks['CaLiNet-E'].append(calinet_model(batch).squeeze(0).cpu().numpy())
                pos += cfg.target_samples

            # Place reconstructions into recon_arrays (channel-first)
            for m in methods:
                recon_norm = np.concatenate(chunks[m], axis=0)  # (T_recon, 12)
                # Fill from cfg.calib_samples onward
                fill_end = cfg.calib_samples + len(recon_norm)
                fill_end = min(fill_end, n_samples)
                actual_len = fill_end - cfg.calib_samples
                recon_arrays[m][idx, :, cfg.calib_samples:fill_end] = recon_norm[:actual_len].T

    # --- Classify each method ---
    print('\nClassifying each method ...')
    method_binaries = {'Original 12-lead': binary_orig}
    for m in methods:
        probs = predict_classifier(recon_arrays[m])
        method_binaries[m] = (probs > thresholds).astype(int)

    # --- F1 table ---
    rows = []
    ordering = ['Original 12-lead', 'GL', 'Transformer', '1D U-Net',
                '1D U-Net w/ anchor', 'PCM', 'CaLiNet-E']
    for name in ordering:
        b = method_binaries[name]
        f1m = f1_score(Y9, b, average='macro', zero_division=0)
        per_class = [f1_score(Y9[:, c], b[:, c], zero_division=0) for c in range(5)]
        rows.append([name, f1m, *per_class])

    df = pd.DataFrame(rows, columns=['Method', 'F1_macro', *SUPER_CLASSES])
    df.to_csv(results_dir / 'diagnostic_all_methods_ptbxl.csv', index=False)

    print('\n' + '=' * 100)
    print(f'{"Method":<24} {"F1_macro":<10} ' +
          '  '.join(f'{c:>7}' for c in SUPER_CLASSES))
    print('-' * 100)
    for name, f1m, *per in rows:
        print(f'{name:<24} {f1m:<10.4f} ' +
              '  '.join(f'{p:>7.4f}' for p in per))

    # ------------------------------------------------------------------
    # Step 2.D: Bootstrap CI (1000 iter, paired resampling)
    # ------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('Step 2.D: Bootstrap 95% CI (paired, 1000 iter, seed=42)')
    print('=' * 70)
    n_iter = 1000
    rng = np.random.default_rng(SEED)
    n_records = len(Y9)

    ci_rows = []
    for name in ordering:
        b = method_binaries[name]
        boot = np.empty(n_iter)
        for i in range(n_iter):
            idx = rng.integers(0, n_records, size=n_records)
            boot[i] = f1_score(Y9[idx], b[idx], average='macro', zero_division=0)
        pt = f1_score(Y9, b, average='macro', zero_division=0)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        ci_rows.append([name, pt, lo, hi])
    ci_df = pd.DataFrame(ci_rows, columns=['Method', 'F1_macro', 'CI_lo', 'CI_hi'])
    ci_df.to_csv(results_dir / 'diagnostic_bootstrap_ci_ptbxl.csv', index=False)

    print()
    for name, pt, lo, hi in ci_rows:
        print(f'  {name:<24} F1={pt:.4f}  CI=[{lo:.4f}, {hi:.4f}]')

    # Paired CaLiNet-E vs PCM
    rng2 = np.random.default_rng(SEED)
    ce = method_binaries['CaLiNet-E']
    pc = method_binaries['PCM']
    deltas = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng2.integers(0, n_records, size=n_records)
        f1_ce = f1_score(Y9[idx], ce[idx], average='macro', zero_division=0)
        f1_pc = f1_score(Y9[idx], pc[idx], average='macro', zero_division=0)
        deltas[i] = f1_ce - f1_pc
    delta_med = np.median(deltas)
    delta_lo, delta_hi = np.percentile(deltas, [2.5, 97.5])
    p_pos = (deltas > 0).mean()

    print()
    print(f'  Paired CaLiNet-E vs PCM:')
    print(f'    delta median = {delta_med:+.4f}')
    print(f'    delta 95% CI = [{delta_lo:+.4f}, {delta_hi:+.4f}]')
    print(f'    p(delta > 0) = {p_pos:.3f}')
    sig = (delta_lo > 0) or (delta_hi < 0)
    print(f'    statistically significant: {"YES" if sig else "NO"}')

    # CaLiNet-E vs Original CI overlap
    ce_lo = next(r[2] for r in ci_rows if r[0] == 'CaLiNet-E')
    ce_hi = next(r[3] for r in ci_rows if r[0] == 'CaLiNet-E')
    or_lo = next(r[2] for r in ci_rows if r[0] == 'Original 12-lead')
    or_hi = next(r[3] for r in ci_rows if r[0] == 'Original 12-lead')
    overlap_orig = not (ce_hi < or_lo or ce_lo > or_hi)
    print(f'\n  CaLiNet-E vs Original CI overlap: {overlap_orig}')

    # ------------------------------------------------------------------
    # Two-dataset comparison table
    # ------------------------------------------------------------------
    code_test_table = {
        'Original 12-lead':     (0.904, 0.854, 0.940),
        'GL':                   (0.906, 0.863, 0.939),
        'Transformer':          (0.784, 0.701, 0.842),
        '1D U-Net':             (0.688, 0.609, 0.751),
        '1D U-Net w/ anchor':   (0.901, 0.852, 0.938),
        'PCM':                  (0.904, 0.862, 0.939),
        'CaLiNet-E':            (0.909, 0.865, 0.944),
    }
    code_test_paired = (0.004, -0.013, 0.023)  # CaLiNet-E - PCM

    lines = []
    lines.append('=' * 110)
    lines.append('L3 diagnostic-level — two-dataset comparison')
    lines.append('=' * 110)
    lines.append('')
    lines.append(f'{"Method":<25} | {"CODE-test":<28} | {"PTB-XL fold 9":<28}')
    lines.append(f'{" ":<25} | {"F1 [95% CI]":<28} | {"F1 [95% CI]":<28}')
    lines.append('-' * 25 + '-+-' + '-' * 28 + '-+-' + '-' * 28)
    for name in ordering:
        code = code_test_table[name]
        ptb = next(r for r in ci_rows if r[0] == name)
        code_s = f'{code[0]:.4f} [{code[1]:.4f},{code[2]:.4f}]'
        ptb_s  = f'{ptb[1]:.4f} [{ptb[2]:.4f},{ptb[3]:.4f}]'
        lines.append(f'{name:<25} | {code_s:<28} | {ptb_s:<28}')
    lines.append('-' * 25 + '-+-' + '-' * 28 + '-+-' + '-' * 28)
    code_p = f'{code_test_paired[0]:+.4f} [{code_test_paired[1]:+.4f},{code_test_paired[2]:+.4f}]'
    ptb_p  = f'{delta_med:+.4f} [{delta_lo:+.4f},{delta_hi:+.4f}]'
    lines.append(f'{"Paired delta CE-PCM":<25} | {code_p:<28} | {ptb_p:<28}')
    code_sig = 'NO'
    ptb_sig = 'YES' if sig else 'NO'
    lines.append(f'{"Statistically sig?":<25} | {code_sig:<28} | {ptb_sig:<28}')
    lines.append('=' * 110)

    out_txt = '\n'.join(lines)
    print('\n' + out_txt)
    (results_dir / 'l3_two_dataset_comparison.txt').write_text(out_txt, encoding='utf-8')
    print(f'\nSaved: {results_dir / "diagnostic_all_methods_ptbxl.csv"}')
    print(f'Saved: {results_dir / "diagnostic_bootstrap_ci_ptbxl.csv"}')
    print(f'Saved: {results_dir / "l3_two_dataset_comparison.txt"}')


if __name__ == '__main__':
    main()
