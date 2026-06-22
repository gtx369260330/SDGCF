"""Configuration for the SDGCF sleep-staging project."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


COMPARISON_MODEL_TRAINING_PRESETS: Dict[str, Dict[str, Any]] = {
    "sdgcf_fixed_graph": {
        "LR": 5e-4,
        "DROPOUT": 0.25,
        "LABEL_SMOOTHING": 0.03,
        "FOCAL_BLEND": 0.20,
        "AUXILIARY_LOSS_WEIGHT": 0.06,
        "MODALITY_DROPOUT_PROB": 0.02,
        "CHANNEL_NOISE_STD": 0.02,
    },
    "concat_transformer": {
        "LR": 3e-4,
        "DROPOUT": 0.30,
        "LABEL_SMOOTHING": 0.04,
        "FOCAL_BLEND": 0.15,
        "MODALITY_DROPOUT_PROB": 0.03,
        "CHANNEL_NOISE_STD": 0.015,
    },
    "multimodal_concat": {
        "LR": 4e-4,
        "DROPOUT": 0.28,
        "LABEL_SMOOTHING": 0.035,
        "FOCAL_BLEND": 0.18,
        "AUXILIARY_LOSS_WEIGHT": 0.05,
        "MODALITY_DROPOUT_PROB": 0.025,
        "CHANNEL_NOISE_STD": 0.015,
    },
}


def get_comparison_model_training_preset(model_name: str) -> Dict[str, Any]:
    """Return a copy of the reproducible training preset for a comparison model."""
    return dict(COMPARISON_MODEL_TRAINING_PRESETS.get(model_name.lower(), {}))


def apply_comparison_model_training_preset(cfg: "Config", model_name: str) -> Dict[str, Any]:
    """Apply architecture-aware baseline defaults before explicit CLI overrides."""
    preset = get_comparison_model_training_preset(model_name)
    for key, value in preset.items():
        setattr(cfg, key, value)
    return preset


@dataclass
class Config:
    # Data
    DATA_PATH: str = "./data/data_multimodal_eeg_eog_3ch"
    SAVE_DIR: str = "./results_enhanced"
    NUM_CLASSES: int = 5

    CHANNEL_NAMES: List[str] = field(
        default_factory=lambda: ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"]
    )
    CLASS_NAMES: List[str] = field(default_factory=lambda: ["W", "N1", "N2", "N3", "REM"])
    INPUT_CHANNELS: int = 3

    # Split
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.10
    TEST_RATIO: float = 0.20
    RANDOM_SEED: int = 42
    DATA_FRACTION: float = 1.0
    SUBSET_STRATIFIED: bool = True

    # Training
    BATCH_SIZE: int = 128
    NUM_WORKERS: int = 0
    PIN_MEMORY: bool = False
    EPOCHS: int = 60
    LR: float = 5e-4
    WEIGHT_DECAY: float = 1e-4
    EARLY_STOPPING_PATIENCE: int = 5
    USE_AMP: bool = True
    USE_CLASS_WEIGHT: bool = True
    GRAD_CLIP_NORM: float = 3.0
    TORCH_NUM_THREADS: int = 4
    WARMUP_EPOCHS: int = 5
    LABEL_SMOOTHING: float = 0.03
    FOCAL_GAMMA: float = 1.5
    FOCAL_BLEND: float = 0.20
    AUXILIARY_LOSS_WEIGHT: float = 0.06
    ATTENTION_ENTROPY_WEIGHT: float = 0.01
    ATTENTION_ENTROPY_FLOOR: float = 0.30
    NODE_DIVERSITY_WEIGHT: float = 0.005
    NODE_DIVERSITY_MARGIN: float = 0.8
    MODALITY_DROPOUT_PROB: float = 0.02
    CHANNEL_NOISE_STD: float = 0.02
    FUSION_LR_MULTIPLIER: float = 1.25

    # Shared modality encoder and graph attention
    EMBED_DIM: int = 192
    ENCODER_HIDDEN: int = 64
    KERNEL_SIZES: Tuple[int, ...] = (3, 7, 15)
    DROPOUT: float = 0.25
    GRAPH_HEADS: int = 2
    TRANSFORMER_HEADS: int = 2
    TRANSFORMER_LAYERS: int = 1

    # Dynamic graph fusion
    GRAPH_ALPHA_INIT: float = 0.08

    # XGBoost baseline on handcrafted channel features
    XGBOOST_N_ESTIMATORS: int = 300
    XGBOOST_MAX_DEPTH: int = 4
    XGBOOST_LEARNING_RATE: float = 0.05
    XGBOOST_SUBSAMPLE: float = 0.85
    XGBOOST_COLSAMPLE_BYTREE: float = 0.85
    XGBOOST_REG_LAMBDA: float = 2.0
    XGBOOST_MIN_CHILD_WEIGHT: float = 1.0
    XGBOOST_TREE_METHOD: str = "hist"
    XGBOOST_FFT_BINS: int = 8

    # Traditional machine-learning baselines on the same handcrafted
    # channel features used by the XGBoost baseline.
    CLASSICAL_MAX_TRAIN_SAMPLES: Optional[int] = None
    RANDOM_FOREST_N_ESTIMATORS: int = 300
    RANDOM_FOREST_MAX_DEPTH: Optional[int] = 18
    RANDOM_FOREST_MIN_SAMPLES_LEAF: int = 2
    LINEAR_SVM_C: float = 1.0
    LINEAR_SVM_MAX_ITER: int = 5000
    LINEAR_SVM_MAX_TRAIN_SAMPLES: Optional[int] = 100000
    LOGISTIC_REGRESSION_C: float = 1.0
    LOGISTIC_REGRESSION_MAX_ITER: int = 1000
    LOGISTIC_REGRESSION_MAX_TRAIN_SAMPLES: Optional[int] = 100000

    # Missing/noisy modality robustness tests
    NOISE_STD: float = 0.20

    def make_dirs(self) -> None:
        save_dir = Path(self.SAVE_DIR)
        for subdir in ["logs", "checkpoints", "metrics", "figures"]:
            (save_dir / subdir).mkdir(parents=True, exist_ok=True)
