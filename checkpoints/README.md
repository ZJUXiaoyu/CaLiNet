# Pretrained Checkpoints and Auxiliary Artifacts

This directory contains a mix of files distributed under three different
release scopes.

## Files tracked in git (small, distributed with this repository)

| File | Description | Size |
|---|---|---|
| `normalizer.npz` | Z-score normaliser (mean/std per lead) fit on PTB-XL training folds | < 1 KB |
| `ribeiro_thresholds.npy` | Per-class probability thresholds (6 floats) for the Ribeiro et al. 6-class classifier, optimised to reproduce the released companion binary predictions | < 1 KB |
| `ptbxl_thresholds.npy` | Per-class probability thresholds (5 floats) for the self-trained PTB-XL 5-super-class classifier | < 1 KB |
| `val_score_tau.npz` | Composite validation-score temperature parameters (τ_n, τ_m) anchored at 1.5× the PCM median on PTB-XL validation | < 6 KB |

These small files are configuration outputs of trivial size; they are tracked
in git for reviewer convenience.

## Regenerable artifacts (not in git; reproduced in seconds by existing scripts)

| File | Reproduced by | Runtime |
|---|---|---|
| `global_W.npz` | `python scripts/01_fit_global.py` | ~30 s |
| `rho_config.npz` | `python scripts/05_diagnose_rho.py --split val --max_clean 100 --max_per_pert 30` | ~60 s |

Both artifacts are deterministic functions of public PTB-XL data. Re-running
the scripts above produces artifacts numerically identical (to within
floating-point precision) to those used in all paper experiments.

## Trained model checkpoints (not in git; Zenodo upon acceptance)

Trained model weights for CaLiNet-E and all baselines (~250 MB total)
will be released on Zenodo with a citable DOI upon paper acceptance.
Expected files:

- `calinet_e_best.pt` — CaLiNet-E main model
- `unet_anchor_best.pt` — 1D U-Net with global-linear anchor
- `cnn_baseline_best.pt` — Non-anchored 1D U-Net
- `transformer_baseline_best.pt`
- `ptbxl_classifier_best.pt` — PTB-XL fold 9 diagnostic classifier

For peer review access prior to acceptance, weights are available on
request via the corresponding author (lvxd@zju.edu.cn).
