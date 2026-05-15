"""Figure 4: Waveform reconstruction examples stratified by rho."""
import sys; sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from scipy.signal import resample_poly
import scipy.io as sio
import yaml

from calinet.config import CaLiNetConfig
from calinet.data.normalizer import GlobalNormalizer
from calinet.data.preprocessing import apply_filter
from calinet.data.episodes import resolve_lead_indices
from calinet.models.calinet_e import CaLiNetE
from calinet.models.calibration import apply_transform, calibrate_patient

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
gw = np.load(artifact_dir / "global_W.npz")
W_global = gw["W_global"].astype(np.float32)
b_global = gw["b_global"].astype(np.float32)
in_idx, target_idx = resolve_lead_indices(cfg.input_leads, cfg.target_leads)
mu_t = normalizer.mu[target_idx]
sigma_t = normalizer.sigma[target_idx]

records = [('A2252', 0.40, 'Low rho=0.40'), ('A4191', 0.80, 'Mid rho=0.80'), ('A4202', 0.97, 'High rho=0.97')]
leads_to_plot = ['V1', 'V3', 'V5']
lead_indices_in_target = [cfg.target_leads.index(l) for l in leads_to_plot]

fig, axes = plt.subplots(3, 3, figsize=(14, 8), sharex='col')

for col, (rec_id, rho_val, title_str) in enumerate(records):
    mat = sio.loadmat(f'c:\\ZJU\\ECG\\data\\cpsc2018_raw\\{rec_id}.mat')
    sig_raw = mat['val'].astype(np.float64).T / 1000.0
    sig_500 = sig_raw.astype(np.float32)

    if cfg.use_bandpass:
        sig_500 = apply_filter(sig_500, low_hz=cfg.bandpass_low_hz,
                               high_hz=cfg.bandpass_high_hz,
                               fs=cfg.sampling_rate, order=cfg.bandpass_order)
    sig_norm = normalizer(sig_500).astype(np.float32)

    Xc = sig_norm[:cfg.calib_samples][:, in_idx]
    Yc = sig_norm[:cfg.calib_samples][:, target_idx]
    Xt = sig_norm[cfg.calib_samples:cfg.calib_samples + cfg.target_samples][:, in_idx]
    Yt = sig_norm[cfg.calib_samples:cfg.calib_samples + cfg.target_samples][:, target_idx]

    Yp_gl = apply_transform(Xt, W_global, b_global)
    W_i, b_i = calibrate_patient(Xc, Yc, W_global, b_global,
                                  lam_W=cfg.ridge_lambda_W, lam_b=cfg.ridge_lambda_b)
    Yp_pcm = apply_transform(Xt, W_i, b_i)

    with torch.no_grad():
        batch = {
            "x_calib": torch.from_numpy(Xc.T)[None].to(device).float(),
            "y_calib": torch.from_numpy(Yc.T)[None].to(device).float(),
            "x_test":  torch.from_numpy(Xt.T)[None].to(device).float(),
            "y_test":  torch.from_numpy(Yt.T)[None].to(device).float(),
        }
        Yp_calinet = calinet_model(batch).squeeze(0).cpu().numpy()

    Yt_mv = Yt * sigma_t + mu_t
    Yp_gl_mv = Yp_gl * sigma_t + mu_t
    Yp_pcm_mv = Yp_pcm * sigma_t + mu_t
    Yp_calinet_mv = Yp_calinet * sigma_t + mu_t

    t = np.arange(cfg.target_samples) / cfg.sampling_rate

    for row, lead_idx in enumerate(lead_indices_in_target):
        ax = axes[row, col]
        ax.plot(t, Yt_mv[:, lead_idx], 'k-', linewidth=1.2, label='Ground Truth', alpha=0.9)
        ax.plot(t, Yp_gl_mv[:, lead_idx], 'b--', linewidth=0.8, label='GL', alpha=0.6)
        ax.plot(t, Yp_pcm_mv[:, lead_idx], 'g-.', linewidth=0.9, label='PCM', alpha=0.7)
        ax.plot(t, Yp_calinet_mv[:, lead_idx], 'r-', linewidth=1.0, label='CaLiNet-E', alpha=0.85)

        if row == 0:
            ax.set_title(f'{title_str}\n({rec_id})', fontsize=11)
        if col == 0:
            ax.set_ylabel(f'{leads_to_plot[row]} (mV)', fontsize=10)
        if row == 2:
            ax.set_xlabel('Time (s)', fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.2)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))
plt.suptitle('Reconstruction Examples on CPSC2018 (OOD)', fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig('results/paper_figures/Fig4_waveform_examples.png', dpi=300, bbox_inches='tight')
plt.savefig('results/paper_figures/Fig4_waveform_examples.pdf', bbox_inches='tight')
print("Saved: results/paper_figures/Fig4_waveform_examples.png/.pdf")
