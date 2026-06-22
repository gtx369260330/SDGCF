"""Centralized naming rules for SDGCF experiments and figures."""
from __future__ import annotations

CANONICAL_MODEL_KEY = "sdgcf"
CANONICAL_MODEL_DISPLAY_NAME = "SDGCF"
SDGCF_FIGURE_PREFIX = "sdgcf"

MODEL_DISPLAY_NAMES = {
    "sdgcf": "SDGCF",
    "sdgcf_fixed_graph": "SDGCF w/ Fixed GraphConv",
    "multimodal_concat": "SDGCF w/o Dynamic Graph",
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "svm_linear": "Linear SVM",
    "logistic_regression": "Logistic Regression",
    "simple_concat": "Simple-Concatenation CNN",
    "concat_transformer": "Modality Transformer",
    "single_fpz": "Single Fpz-Cz",
    "single_pz": "Single Pz-Oz",
    "single_eog": "Single EOG",
}


def display_model_name(model_name: str) -> str:
    """Return the paper/figure display name for an internal model key."""
    key = str(model_name)
    return MODEL_DISPLAY_NAMES.get(key.lower(), key)


def sdgcf_figure_name(name: str) -> str:
    """Prefix cross-model figure files with the canonical SDGCF project name."""
    clean_name = str(name)
    prefix = f"{SDGCF_FIGURE_PREFIX}_"
    return clean_name if clean_name.startswith(prefix) else f"{prefix}{clean_name}"
