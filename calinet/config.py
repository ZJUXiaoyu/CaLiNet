"""Central configuration dataclass for CaLiNet v1.0.

All hyperparameters live here. Modify via CLI override or by passing a
CaLiNetConfig instance. Defaults match the v1.0 spec.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class CaLiNetConfig:
    # ---- Paths ----
    ptbxl_path: str = "../data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    cpsc_path:  str = ""  # filled later
    code_path:  str = ""  # filled later
    artifact_dir: str = "checkpoints"
    results_dir:  str = "results"

    # ---- Data ----
    sampling_rate: int = 500
    input_leads:  Tuple[str, ...] = ("I", "II", "V2")
    target_leads: Tuple[str, ...] = (
        "I", "II", "III", "aVR", "aVL", "aVF",
        "V1", "V2", "V3", "V4", "V5", "V6",
    )

    # ---- Episode sampling ----
    calib_seconds:  float = 2.0
    target_seconds: float = 2.0
    train_gap_seconds: Tuple[float, ...] = (0.0, 1.0, 2.0, 5.0)

    # ---- Preprocessing ----
    use_bandpass:        bool  = True
    bandpass_low_hz:     float = 0.5
    bandpass_high_hz:    float = 0.0   # 0 → high-pass only (baseline removal)
    bandpass_order:      int   = 4

    # ---- Calibration (decoupled W and b regularization) ----
    ridge_lambda_W: float = 1.0
    ridge_lambda_b: float = 0.1
    ridge_lambda_W_grid: Tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    ridge_lambda_b_grid: Tuple[float, ...] = (0.0, 0.1, 1.0)

    # ---- Quality score rho_i ----
    rho_w_cond: float = 0.3
    rho_w_fit:  float = 0.5
    rho_w_beat: float = 0.2
    kappa_max:  float = 1e4
    tau_fit:    float = 0.1

    # ---- Model ----
    embedding_dim: int = 128
    unet_channels: Tuple[int, ...] = (32, 64, 128, 256)
    film_layers:   Tuple[str, ...] = ("bottleneck", "enc_3", "enc_4")
    pad_to_multiple: int = 16
    use_film_conditioning: bool = True

    # ---- Training ----
    batch_size:    int   = 32
    num_workers:   int   = 4
    lr:            float = 1e-3
    weight_decay:  float = 1e-5
    max_epochs:    int   = 100
    early_stop_patience: int = 10
    grad_clip:     float = 1.0

    # ---- Validation score weights ----
    val_w_pcc_precordial: float = 0.6
    val_w_pcc_missing:    float = 0.2
    val_w_nrmse:          float = 0.2

    # ---- Augmentation ----
    aug_mode: str = "input_only"   # 'input_only' | 'shared' | 'none'
    aug_noise_snr_db: Tuple[float, float] = (20.0, 40.0)
    aug_drift_hz:     Tuple[float, float] = (0.1, 0.5)
    aug_drift_amp:    float = 0.05
    aug_amp_scale:    Tuple[float, float] = (0.9, 1.1)

    # ---- Ablation flags ----
    use_morphology_loss: bool = False
    morphology_loss_lambda: float = 0.1

    # ---- Reproducibility ----
    seed: int = 42

    # ---- Splits (PTB-XL official strat_fold) ----
    train_folds: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
    val_fold:    int = 9
    test_fold:   int = 10

    # Derived properties --------------------------------------------------
    @property
    def calib_samples(self) -> int:
        return int(self.calib_seconds * self.sampling_rate)

    @property
    def target_samples(self) -> int:
        return int(self.target_seconds * self.sampling_rate)

    @property
    def n_input_leads(self) -> int:
        return len(self.input_leads)

    @property
    def n_target_leads(self) -> int:
        return len(self.target_leads)

    def lead_indices(self, leads: Tuple[str, ...]) -> list[int]:
        return [self.target_leads.index(l) for l in leads]

    @property
    def input_lead_idx(self) -> list[int]:
        return self.lead_indices(self.input_leads)
