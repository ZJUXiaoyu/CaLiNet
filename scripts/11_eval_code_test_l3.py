"""L3 Diagnostic-Level: CaLiNet-E reconstructed -> Ribeiro classifier -> F1."""
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

# Load CaLiNet-E
calinet_model = CaLiNetE.from_artifacts(
    artifact_dir, n_in=len(cfg.input_leads), n_out=len(cfg.target_leads),
    channels=cfg.unet_channels, embedding_dim=cfg.embedding_dim,
    pad_to_multiple=cfg.pad_to_multiple, sampling_rate=cfg.sampling_rate,
    lam_W=cfg.ridge_lambda_W, lam_b=cfg.ridge_lambda_b, force_rho_one=False,
).to(device)
ckpt = torch.load(artifact_dir / "calinet_e_full_best.pth", map_location=device)
calinet_model.load_state_dict(ckpt["model"])
calinet_model.eval()

normalizer = GlobalNormalizer.load(artifact_dir / "normalizer.npz")
in_idx, target_idx = resolve_lead_indices(cfg.input_leads, cfg.target_leads)
mu_t = normalizer.mu[target_idx]
sigma_t = normalizer.sigma[target_idx]

# Load CODE-test
f = h5py.File(r'c:\ZJU\ECG\data\CODE-test\ecg_tracings.hdf5', 'r')
tracings_orig = f['tracings'][:].astype(np.float32)
f.close()

# Lead reorder: CODE {I,II,III,AVL,AVF,AVR,V1-V6} -> Standard {I,II,III,aVR,aVL,aVF,V1-V6}
CODE_TO_STD = [0, 1, 2, 5, 3, 4, 6, 7, 8, 9, 10, 11]
# STD_TO_CODE is the INVERSE permutation (3-cycle on positions 3/4/5, not self-inverse).
STD_TO_CODE = [0, 1, 2, 4, 5, 3, 6, 7, 8, 9, 10, 11]

tracings_mv_std = tracings_orig[:, :, CODE_TO_STD] * 0.1  # mV, standard order
tracings_500 = resample_poly(tracings_mv_std, up=5, down=4, axis=1).astype(np.float32)

def find_bounds(sig, threshold=1e-4):
    energy = np.abs(sig).sum(axis=1)
    nz = np.where(energy > threshold)[0]
    return (int(nz[0]), int(nz[-1]) + 1) if len(nz) > 0 else (0, sig.shape[0])

# CaLiNet-E inference
print("Running CaLiNet-E on CODE-test...")
reconstructed_ribeiro = tracings_orig.copy()
n_reconstructed = 0

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

        Xc = sig_norm[:cfg.calib_samples][:, in_idx]
        Yc = sig_norm[:cfg.calib_samples][:, target_idx]

        chunks = []
        pos = cfg.calib_samples
        while pos + cfg.target_samples <= len(sig_norm):
            Xt = sig_norm[pos:pos + cfg.target_samples][:, in_idx]
            batch = {
                "x_calib": torch.from_numpy(Xc.T)[None].to(device).float(),
                "y_calib": torch.from_numpy(Yc.T)[None].to(device).float(),
                "x_test":  torch.from_numpy(Xt.T)[None].to(device).float(),
                "y_test":  torch.zeros(1, len(cfg.target_leads), cfg.target_samples).to(device),
            }
            Yp = calinet_model(batch).squeeze(0).cpu().numpy()
            Yp_mv = Yp * sigma_t + mu_t
            chunks.append(Yp_mv)
            pos += cfg.target_samples

        if not chunks:
            continue

        recon_mv = np.concatenate(chunks, axis=0)  # (T, 12) mV std order
        recon_400 = resample_poly(recon_mv, up=4, down=5, axis=0).astype(np.float32)
        recon_code = recon_400[:, STD_TO_CODE]
        recon_scale = recon_code * 10.0  # mV -> 1e-4 V scale

        # Place reconstructed test portion into output
        start_400 = int(start * 4 / 5)
        calib_400 = int(cfg.calib_samples * 4 / 5)
        insert_start = start_400 + calib_400
        insert_end = min(insert_start + len(recon_scale), 4096)
        actual_len = insert_end - insert_start
        if actual_len > 0 and actual_len <= len(recon_scale):
            reconstructed_ribeiro[idx, insert_start:insert_end] = recon_scale[:actual_len]
            n_reconstructed += 1

print(f"  reconstructed: {n_reconstructed} / 827")

# Run Ribeiro on both
print("Running Ribeiro classifier...")
ribeiro_model = tf.keras.models.load_model(
    r'c:\ZJU\ECG\ribeiro_model\model\model.hdf5', compile=False
)
preds_orig = ribeiro_model.predict(tracings_orig, batch_size=32, verbose=0)
preds_recon = ribeiro_model.predict(reconstructed_ribeiro, batch_size=32, verbose=0)

thresholds = np.load('checkpoints/ribeiro_thresholds.npy')
binary_orig = (preds_orig > thresholds).astype(int)
binary_recon = (preds_recon > thresholds).astype(int)

gold = pd.read_csv('c:/ZJU/ECG/data/CODE-test/annotations/gold_standard.csv').values
classes = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST']

print("\n" + "=" * 78)
print("L3 Diagnostic-Level Evaluation (CODE-test, n=827)")
print("=" * 78)
header = f"{'Method':<25} {'F1_macro':<10} " + "  ".join(f"{c:>6}" for c in classes)
print(header)
print("-" * len(header))

for name, binary in [("Original 12-lead", binary_orig),
                     ("CaLiNet-E reconstructed", binary_recon)]:
    f1_mac = f1_score(gold, binary, average='macro')
    per_class = [f1_score(gold[:, i], binary[:, i]) for i in range(6)]
    print(f"{name:<25} {f1_mac:<10.4f} " +
          "  ".join(f"{fc:>6.3f}" for fc in per_class))

f1_orig_mac = f1_score(gold, binary_orig, average='macro')
f1_recon_mac = f1_score(gold, binary_recon, average='macro')
print(f"\nF1 retention: {f1_recon_mac/f1_orig_mac*100:.1f}% (recon/orig)")
print(f"F1 drop: {f1_orig_mac - f1_recon_mac:.4f}")

if f1_recon_mac > f1_orig_mac:
    print("\n[WARN] Reconstructed F1 > Original F1 -- check for data leakage!")
else:
    print("\n[OK] Reconstructed F1 < Original F1 (information-theoretic expectation met)")
