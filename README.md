# CaLiNet: Calibration-aware Reduced-Lead ECG Reconstruction

> ⚠️ **Status: manuscript under peer review.** The associated paper is currently
> under review at a peer-reviewed journal. This repository is shared for
> transparency and reproducibility. Citation details will be updated upon
> acceptance.

Code accompanying the paper *"Reconstructing 12-lead ECG from 3 leads with
brief patient-specific calibration and comprehensive evaluation"*
(Zhang, Cai, Duan, Lu, 2026).

CaLiNet-E reconstructs the full 12-lead ECG from 3 leads (I, II, V2) using
a brief 2-second 12-lead calibration segment at device setup. It combines
per-patient linear calibration with a non-linear residual network, achieving
state-of-the-art reconstruction performance while preserving downstream
diagnostic accuracy.

## Key Results

| Dataset | Method | PCC | nRMSE | ST60 anterior (mV) | Macro F1 |
|---|---|---|---|---|---|
| PTB-XL (in-dist) | **CaLiNet-E** | **0.967** | **0.220** | **0.041** | — |
| PTB-XL (in-dist) | PCM (linear only) | 0.944 | 0.300 | 0.051 | — |
| CPSC2018 (OOD) | **CaLiNet-E** | **0.941** | **0.295** | **0.063** | — |
| CODE-test (n=827) | **CaLiNet-E** | — | — | — | **0.909** vs 0.904 original |
| PTB-XL fold 9 (n=1,709) | **CaLiNet-E** | — | — | — | **0.729** vs 0.735 original (99.2% retention) |

## Method

CaLiNet-E forward pass:
```
Y_pred = X_test @ W_eff + b_eff + R_theta(X_test, e_i)
```

where:
- `W_eff = rho_i * W_i + (1 - rho_i) * W_global` — soft fallback by calibration quality score
- `R_theta` is a 1D U-Net with FiLM conditioning (zero-initialised, so epoch 0 ≡ PCM exactly)
- `e_i` encodes the patient-specific calibration deviation

A central empirical finding: **non-anchored deep methods (1D U-Net,
Transformer) reach F1 retention 62–87% despite signal-level Pearson
correlation 0.92–0.93, while adding a fixed population-level linear anchor
to the same backbone recovers 97–100% F1 retention**, suggesting that
grounding deep ECG reconstruction in a clinically validated linear baseline
is necessary for preserving diagnostic-level fidelity.

## Paper terminology ↔ code modules

| Paper name | Code module |
|---|---|
| CaLiNet-E (our method) | `calinet/models/calinet_e.py` |
| 1D U-Net w/ global-linear anchor | `calinet/models/unet_anchor.py` |
| 1D U-Net (baseline) | `calinet/models/baselines.py` (CNN class) |
| Transformer (baseline) | `calinet/models/baselines.py` (Transformer class) |
| PCM (per-patient ridge) | `calinet/models/calibration.py` |
| GL (population ridge) | `calinet/models/calibration.py` (global mode) |
| Calibration quality score ρ_i | `calinet/models/rho.py` |

## Installation

```bash
git clone https://github.com/ZJUXiaoyu/CaLiNet.git
cd CaLiNet
pip install -r requirements.txt
```

Requires Python 3.10+ and PyTorch 2.0+. For diagnostic-level evaluation,
also install `tensorflow-cpu` (used by the Ribeiro et al. pretrained
classifier).

## Datasets

The following datasets are required but not included in this repository:

| Dataset | Source | Used for |
|---|---|---|
| PTB-XL | https://physionet.org/content/ptb-xl/1.0.3/ | Training, val, fold 9 L3 |
| CPSC2018 | `python scripts/fetch_cpsc2018.py` | OOD evaluation |
| CODE-test | https://doi.org/10.5281/zenodo.3765780 | Ribeiro classifier L3 evaluation |

Place data under a `data/` directory (see `configs/default.yaml` for
expected paths).

## Reproducing Results

Scripts are designed to run in numerical order.

### Step 1: regenerate calibration artifacts (not in git)

```bash
python scripts/01_fit_global.py
python scripts/05_diagnose_rho.py --split val --max_clean 100 --max_per_pert 30
```

### Step 2-5: linear baselines and morphology evaluation

```bash
python scripts/02_run_baselines.py
python scripts/03_run_morphology.py
python scripts/04_run_E12.py
python scripts/06_anchor_val_tau.py
```

### Step 7-9: train deep models (≈ 1.5 h each on RTX 4090)

```bash
python scripts/07_train_unet_anchor.py --epochs 50 --tag unet_anchor_full
python scripts/08_train_calinet_e.py    --epochs 50 --tag calinet_e_full
python scripts/09_train_baseline.py     --model cnn         --epochs 50 --tag cnn_baseline
python scripts/09_train_baseline.py     --model transformer --epochs 50 --tag transformer_baseline
```

### Step 10-14: evaluation (reproduces all paper tables)

```bash
# Table 2 (CPSC2018 OOD)
python scripts/fetch_cpsc2018.py
python scripts/10_eval_cpsc2018.py --root data/cpsc2018_raw

# Table 3 (diagnostic-level, both datasets)
python scripts/13_train_ptbxl_classifier.py
python scripts/14_eval_ptbxl_fold9_l3.py
```

Expected end-to-end training time on a single NVIDIA RTX 4090: ≈ 8 hours.

## Pretrained Weights

Trained model checkpoints (≈ 50 MB each, ≈ 250 MB total for all 5 models)
are **not included in this repository**. They will be released on Zenodo
with a citable DOI **upon paper acceptance**. This release scope is consistent
with standard practice in the ECG deep learning literature (e.g., Ribeiro
et al. 2020 *Nature Communications*, Strodthoff et al. 2021 *IEEE JBHI*).

The repository contains the complete training and evaluation pipeline:
all paper results can be reproduced from publicly available datasets
using the provided scripts.

For peer review access, weights are available on request via the
corresponding author (lvxd@zju.edu.cn).

## Project Structure

```
calinet/
├── config.py                Configuration dataclass
├── data/                    Data loading + preprocessing
│   ├── cpsc2018.py          CPSC2018 OOD loader
│   ├── dataset.py           Episodic PTB-XL dataset
│   ├── episodes.py          Calibration/test episode extraction
│   ├── normalizer.py        Z-score normalisation
│   ├── preprocessing.py     Bandpass filtering
│   └── ptbxl.py             PTB-XL metadata + record loading
├── eval/                    Evaluation metrics
│   ├── metrics.py           PCC, RMSE
│   ├── morphology.py        Clinical morphology (R-amp, ST60, T-amp)
│   ├── rpeak.py             R-peak detection
│   └── perturbations.py     Calibration robustness perturbations
└── models/
    ├── backbone.py          Shared 1D U-Net with FiLM conditioning
    ├── baselines.py         CNN + Transformer non-calibrated baselines
    ├── calibration.py       Ridge regression (numpy + GPU-vectorised torch)
    ├── calinet_e.py         CaLiNet-E (the main method)
    ├── rho.py               Calibration quality score ρ_i
    ├── unet_anchor.py       1D U-Net with fixed global-linear anchor
    └── template_replay.py   Template replay baseline
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{zhang2026calinet,
  title  = {Reconstructing 12-lead ECG from 3 leads with brief patient-specific
            calibration and comprehensive evaluation},
  author = {Zhang, Xinyu and Cai, Hailing and Duan, Huilong and Lu, Xudong},
  journal = {[journal] -- under review},
  year   = {2026},
  note   = {Manuscript under peer review}
}
```

## License

MIT License. See [LICENSE](LICENSE).

## Contact

**Corresponding author:** Prof. Xudong Lu (lvxd@zju.edu.cn)  
**First author:** Xinyu Zhang  
College of Biomedical Engineering and Instrument Science, Zhejiang University
