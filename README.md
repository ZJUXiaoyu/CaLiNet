# CaLiNet: Calibration-aware Reduced-Lead ECG Reconstruction

**CaLiNet-E** reconstructs 12-lead ECG from 3 leads (I, II, V2) using a brief 12-lead calibration segment, combining per-patient linear calibration with a non-linear residual network.

## Key Results

| Dataset | Method | PCC | ST60 error (mV) |
|---|---|---|---|
| PTB-XL (in-dist) | CaLiNet-E | **0.967** | **0.041** |
| PTB-XL (in-dist) | PCM (linear only) | 0.944 | 0.051 |
| CPSC2018 (OOD) | CaLiNet-E | **0.941** | **0.063** |
| CPSC2018 (OOD) | PCM (linear only) | 0.912 | 0.078 |

Diagnostic-level evaluation (Ribeiro et al. classifier): CaLiNet-E reconstructions retain **99.2%** of original 12-lead diagnostic F1 on PTB-XL (macro F1 0.729 vs 0.735).

## Method

CaLiNet-E forward pass:

```
Y_pred = X_test @ W_eff + b_eff + R_theta(X_test, e_i)
```

where:
- `W_eff = rho_i * W_i + (1 - rho_i) * W_global` (soft fallback by calibration quality)
- `R_theta` is a 1D U-Net with FiLM conditioning (zero-initialized, so epoch 0 = PCM exactly)
- `e_i` encodes the patient-specific calibration deviation

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and PyTorch 2.0+. For diagnostic-level evaluation, also install `tensorflow-cpu`.

## Data

Download the following datasets (not included due to size):

- **PTB-XL**: https://physionet.org/content/ptb-xl/1.0.3/
- **CPSC2018**: `python scripts/fetch_cpsc2018.py` (downloads from PhysioNet)
- **CODE-test**: https://zenodo.org/records/3765780

Place data under a `data/` directory (see `configs/default.yaml` for expected paths).

## Reproducing Results

Run scripts in numerical order:

```bash
# Step 1-5: Baselines + calibration quality
python scripts/01_fit_global.py
python scripts/02_run_baselines.py
python scripts/03_run_morphology.py
python scripts/04_run_E12.py
python scripts/05_diagnose_rho.py

# Step 6: Anchor validation score thresholds
python scripts/06_anchor_val_tau.py

# Step 7-9: Train models
python scripts/07_train_tcae.py --epochs 50 --tag tcae_full
python scripts/08_train_calinet_e.py --epochs 50 --tag calinet_e_full
python scripts/09_train_baseline.py --model cnn --epochs 50 --tag cnn_baseline
python scripts/09_train_baseline.py --model transformer --epochs 50 --tag transformer_baseline

# Step 10: CPSC2018 OOD evaluation
python scripts/fetch_cpsc2018.py
python scripts/10_eval_cpsc2018.py --root data/cpsc2018_raw

# Step 11-14: Diagnostic-level evaluation
python scripts/13_train_ptbxl_classifier.py
python scripts/14_eval_ptbxl_fold9_l3.py
```

## Pretrained Weights

Model checkpoints (~50 MB each) are available upon request or will be uploaded to Zenodo upon publication. Contact the authors for early access.

## Project Structure

```
calinet/
├── config.py              Configuration dataclass
├── data/                  Data loading + preprocessing
│   ├── cpsc2018.py        CPSC2018 OOD loader
│   ├── dataset.py         Episodic PTB-XL dataset
│   ├── episodes.py        Calibration/test episode extraction
│   ├── normalizer.py      Z-score normalization
│   ├── preprocessing.py   Bandpass filtering
│   └── ptbxl.py           PTB-XL metadata + record loading
├── eval/                  Evaluation metrics
│   ├── metrics.py         PCC, RMSE
│   ├── morphology.py      Clinical morphology (R-amp, ST60, T-amp)
│   ├── rpeak.py           R-peak detection
│   └── perturbations.py   E12 robustness perturbations
└── models/
    ├── backbone.py        Shared 1D U-Net with FiLM
    ├── baselines.py       CNN + Transformer baselines
    ├── calibration.py     Ridge regression (numpy + GPU-vectorized torch)
    ├── calinet_e.py       CaLiNet-E model
    ├── rho.py             Calibration quality score
    ├── tcae.py            TCAE (U-Net + linear anchor)
    └── template_replay.py Template replay baseline
```

## Citation

```bibtex
@article{calinet2026,
  title={CaLiNet: Calibration-aware Linear-Nonlinear Network for Reduced-Lead ECG Reconstruction},
  author={[Authors]},
  journal={[Journal]},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE).
