"""Extended L3 diagnostic evaluation across all reconstruction methods.

Evaluates 6 reconstruction methods on CODE-test (827 records) via the
Ribeiro et al. (2020) pretrained classifier with optimized per-class
thresholds, producing the diagnostic-level analog of Tables 1 / 2.

Methods evaluated (same checkpoints as PTB-XL/CPSC2018 tables):
    GL                     global linear (W_global, b_global)
    PCM                    per-patient ridge calibration
    Transformer            from-scratch Transformer baseline
    1D U-Net               from-scratch CNN baseline (no anchor)
    1D U-Net w/ anchor     TCAE: CNN backbone + W_global anchor
    CaLiNet-E              ours (calibration + non-linear + soft fallback)

For each method, the same temporal layout is used (2s calibration block +
2s non-overlapping test segments). Methods that do not consume calibration
(GL, deep baselines) still operate on the test-segment portion of the
record; the calibration block is left as the original signal so that the
final fed-into-classifier tensor has identical layout across methods.

Output: results/diagnostic_all_methods.csv + console summary including
F1 retention (= F1_macro_method / F1_macro_original).
"""
import sys; sys.path.insert(0, '.')
import numpy as np
import h5py
import torch
import tensorflow as tf
from pathlib import Path
from scipy.signal import resample_poly
from sklearn.metrics import f1_score
import pandas as pd
import yaml

from calinet.config import CaLiNetConfig
from calinet.data.normalizer import GlobalNormalizer
from calinet.data.preprocessing import apply_filter
from calinet.data.episodes import resolve_lead_indices
from calinet.models.calinet_e import CaLiNetE
from calinet.models.tcae import TCAE
from calinet.models.baselines import CNNBaseline, TransformerBaseline
from calinet.models.calibration import apply_transform, calibrate_patient

# --- Config ---
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# --- Load models ---
print("Loading models...")
gw = np.load(artifact_dir / "global_W.npz")
W_global = gw["W_global"].astype(np.float32)
b_global = gw["b_global"].astype(np.float32)

cnn_model = CNNBaseline(
    n_in=len(cfg.input_leads), n_out=len(cfg.target_leads),
    channels=cfg.unet_channels, pad_to_multiple=cfg.pad_to_multiple,
).to(device)
cnn_model.load_state_dict(torch.load(artifact_dir / "cnn_baseline_best.pth", map_location=device)["model"])
cnn_model.eval()

tr_model = TransformerBaseline(
    n_in=len(cfg.input_leads), n_out=len(cfg.target_leads),
).to(device)
tr_model.load_state_dict(torch.load(artifact_dir / "transformer_baseline_best.pth", map_location=device)["model"])
tr_model.eval()

tcae_model = TCAE.from_artifacts(
    artifact_dir, n_in=len(cfg.input_leads), n_out=len(cfg.target_leads),
    channels=cfg.unet_channels, embedding_dim=cfg.embedding_dim,
    pad_to_multiple=cfg.pad_to_multiple,
).to(device)
tcae_model.load_state_dict(torch.load(artifact_dir / "tcae_full_best.pth", map_location=device)["model"])
tcae_model.eval()

calinet_model = CaLiNetE.from_artifacts(
    artifact_dir, n_in=len(cfg.input_leads), n_out=len(cfg.target_leads),
    channels=cfg.unet_channels, embedding_dim=cfg.embedding_dim,
    pad_to_multiple=cfg.pad_to_multiple, sampling_rate=cfg.sampling_rate,
    lam_W=cfg.ridge_lambda_W, lam_b=cfg.ridge_lambda_b, force_rho_one=False,
).to(device)
calinet_model.load_state_dict(torch.load(artifact_dir / "calinet_e_full_best.pth", map_location=device)["model"])
calinet_model.eval()

normalizer = GlobalNormalizer.load(artifact_dir / "normalizer.npz")
in_idx, target_idx = resolve_lead_indices(cfg.input_leads, cfg.target_leads)
mu_t = normalizer.mu[target_idx]
sigma_t = normalizer.sigma[target_idx]

# --- Load CODE-test ---
print("Loading CODE-test...")
f = h5py.File(r'c:\ZJU\ECG\data\CODE-test\ecg_tracings.hdf5', 'r')
tracings_orig = f['tracings'][:].astype(np.float32)  # (827, 4096, 12) CODE order, 1e-4 V
f.close()

CODE_TO_STD = [0, 1, 2, 5, 3, 4, 6, 7, 8, 9, 10, 11]
# STD_TO_CODE is the INVERSE permutation of CODE_TO_STD.
# (aVR/aVL/aVF form a 3-cycle, NOT self-inverse — earlier version had this bug.)
# CODE pos 3 (AVL) ← STD pos 4 (aVL); CODE pos 4 (AVF) ← STD pos 5 (aVF); CODE pos 5 (AVR) ← STD pos 3 (aVR)
STD_TO_CODE = [0, 1, 2, 4, 5, 3, 6, 7, 8, 9, 10, 11]

tracings_mv_std = tracings_orig[:, :, CODE_TO_STD] * 0.1
tracings_500 = resample_poly(tracings_mv_std, up=5, down=4, axis=1).astype(np.float32)

def find_bounds(sig, threshold=1e-4):
    energy = np.abs(sig).sum(axis=1)
    nz = np.where(energy > threshold)[0]
    return (int(nz[0]), int(nz[-1]) + 1) if len(nz) > 0 else (0, sig.shape[0])

# --- Reconstruction loop ---
# Output containers: one (827, 4096, 12) tensor per method, starting from orig copy
methods_to_run = ['GL', 'Transformer', '1D U-Net', '1D U-Net w/ anchor', 'PCM', 'CaLiNet-E']
recon_arrays = {m: tracings_orig.copy() for m in methods_to_run}
n_processed = 0

print("Running reconstructions...")
with torch.no_grad():
    for idx in range(827):
        sig = tracings_500[idx]
        start, end = find_bounds(sig)
        sig_trimmed = sig[start:end]

        if len(sig_trimmed) < cfg.calib_samples + cfg.target_samples:
            continue

        if cfg.use_bandpass:
            sig_bp = apply_filter(sig_trimmed, low_hz=cfg.bandpass_low_hz,
                                  high_hz=cfg.bandpass_high_hz,
                                  fs=cfg.sampling_rate, order=cfg.bandpass_order)
        else:
            sig_bp = sig_trimmed
        sig_norm = normalizer(sig_bp).astype(np.float32)

        # Calibration block (used by PCM and CaLiNet-E)
        Xc_full = sig_norm[:cfg.calib_samples][:, in_idx]
        Yc_full = sig_norm[:cfg.calib_samples][:, target_idx]
        Xc_full_np = Xc_full
        Yc_full_np = Yc_full

        # PCM: fit ridge once on calibration block
        W_i, b_i = calibrate_patient(
            Xc_full_np, Yc_full_np, W_global, b_global,
            lam_W=cfg.ridge_lambda_W, lam_b=cfg.ridge_lambda_b,
        )

        # Per-method per-chunk reconstruction
        per_method_chunks = {m: [] for m in methods_to_run}
        pos = cfg.calib_samples
        while pos + cfg.target_samples <= len(sig_norm):
            Xt_np = sig_norm[pos:pos + cfg.target_samples][:, in_idx]
            Xt_t = torch.from_numpy(Xt_np)[None].to(device).float()  # (1, T, n_in)

            # GL
            Y_gl = apply_transform(Xt_np, W_global, b_global)
            per_method_chunks['GL'].append(Y_gl)

            # PCM
            Y_pcm = apply_transform(Xt_np, W_i, b_i)
            per_method_chunks['PCM'].append(Y_pcm)

            # 1D U-Net (CNN baseline)
            Y_cnn = cnn_model(Xt_t).squeeze(0).cpu().numpy()
            per_method_chunks['1D U-Net'].append(Y_cnn)

            # Transformer
            Y_tr = tr_model(Xt_t).squeeze(0).cpu().numpy()
            per_method_chunks['Transformer'].append(Y_tr)

            # 1D U-Net w/ anchor (TCAE)
            Y_tcae = tcae_model(Xt_t).squeeze(0).cpu().numpy()
            per_method_chunks['1D U-Net w/ anchor'].append(Y_tcae)

            # CaLiNet-E
            batch = {
                "x_calib": torch.from_numpy(Xc_full.T)[None].to(device).float(),
                "y_calib": torch.from_numpy(Yc_full.T)[None].to(device).float(),
                "x_test":  Xt_t.transpose(1, 2),
                "y_test":  torch.zeros(1, len(cfg.target_leads), cfg.target_samples).to(device),
            }
            Y_cal = calinet_model(batch).squeeze(0).cpu().numpy()
            per_method_chunks['CaLiNet-E'].append(Y_cal)

            pos += cfg.target_samples

        if not per_method_chunks['GL']:
            continue

        # Insert each method's reconstruction back into the (4096, 12) array
        start_400 = int(start * 4 / 5)
        calib_400 = int(cfg.calib_samples * 4 / 5)
        insert_start = start_400 + calib_400

        for m in methods_to_run:
            recon_norm = np.concatenate(per_method_chunks[m], axis=0)
            # Denormalize to mV
            recon_mv = recon_norm * sigma_t + mu_t
            # 500 -> 400 Hz
            recon_400 = resample_poly(recon_mv, up=4, down=5, axis=0).astype(np.float32)
            # Standard -> CODE lead order
            recon_code = recon_400[:, STD_TO_CODE]
            # mV -> 1e-4 V scale (the unit Ribeiro expects)
            recon_scale = recon_code * 10.0

            insert_end = min(insert_start + len(recon_scale), 4096)
            actual_len = insert_end - insert_start
            if actual_len > 0 and actual_len <= len(recon_scale):
                recon_arrays[m][idx, insert_start:insert_end] = recon_scale[:actual_len]

        n_processed += 1
        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/827 records")

print(f"\nProcessed {n_processed}/827 records.\n")

# --- Run Ribeiro classifier on each reconstructed array ---
print("Running Ribeiro classifier on each method...")
ribeiro_model = tf.keras.models.load_model(
    r'c:\ZJU\ECG\ribeiro_model\model\model.hdf5', compile=False
)
thresholds = np.load('checkpoints/ribeiro_thresholds.npy')

# Original (reference)
preds_orig = ribeiro_model.predict(tracings_orig, batch_size=32, verbose=0)
binary_orig = (preds_orig > thresholds).astype(int)

# Each method
method_binaries = {}
for m in methods_to_run:
    preds = ribeiro_model.predict(recon_arrays[m], batch_size=32, verbose=0)
    method_binaries[m] = (preds > thresholds).astype(int)
    print(f"  {m}: done")

# --- F1 computation ---
gold = pd.read_csv('c:/ZJU/ECG/data/CODE-test/annotations/gold_standard.csv').values
classes = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST']

rows = []
def add_row(name, binary):
    f1_mac = f1_score(gold, binary, average='macro')
    per_class = [f1_score(gold[:, i], binary[:, i]) for i in range(6)]
    rows.append([name, f1_mac, *per_class])
    return f1_mac

f1_orig = add_row('Original 12-lead', binary_orig)
# Order rows to match the user's spec
method_order = ['GL', 'Transformer', '1D U-Net', '1D U-Net w/ anchor', 'PCM', 'CaLiNet-E']
method_f1 = {}
for m in method_order:
    f1m = add_row(m, method_binaries[m])
    method_f1[m] = f1m

df = pd.DataFrame(rows, columns=['Method', 'F1_macro', *classes])
df.to_csv('results/diagnostic_all_methods.csv', index=False)

# Console pretty-print
print("\n" + "=" * 100)
print(f"{'Method':<24} {'F1_macro':<10} " + "  ".join(f"{c:>6}" for c in classes))
print("-" * 100)
for r in rows:
    name, f1m, *per = r
    print(f"{name:<24} {f1m:<10.4f} " + "  ".join(f"{p:>6.3f}" for p in per))

print("\n" + "=" * 100)
print(f"F1 retention (= F1_macro / Original F1_macro = {f1_orig:.4f})")
print("-" * 100)
for m in method_order:
    ret = method_f1[m] / f1_orig * 100
    print(f"  {m:<24} F1_macro={method_f1[m]:.4f}  retention={ret:.1f}%")

print(f"\nSaved: results/diagnostic_all_methods.csv")

# Sanity check vs predicted
print("\nSanity check against user's predicted ordering:")
print("  Expected: GL ~80% < deep baselines 88-92% < PCM ~95-96% < CaLiNet-E 97.8%")
print(f"  Got:      GL {method_f1['GL']/f1_orig*100:.1f}%")
print(f"            1D U-Net {method_f1['1D U-Net']/f1_orig*100:.1f}%")
print(f"            Transformer {method_f1['Transformer']/f1_orig*100:.1f}%")
print(f"            U-Net w/ anchor {method_f1['1D U-Net w/ anchor']/f1_orig*100:.1f}%")
print(f"            PCM {method_f1['PCM']/f1_orig*100:.1f}%")
print(f"            CaLiNet-E {method_f1['CaLiNet-E']/f1_orig*100:.1f}%")

if method_f1['CaLiNet-E'] < max(method_f1[m] for m in method_order if m != 'CaLiNet-E'):
    print("\n[WARN] Another method beat CaLiNet-E on F1_macro — investigate before reporting")
else:
    print("\n[OK] CaLiNet-E remains the top method")
