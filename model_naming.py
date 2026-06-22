"""Centralized naming rules for SDGCF experiments."""
from __future__ import annotations

CANONICAL_MODEL_KEY = "sdgcf"
CANONICAL_MODEL_DISPLAY_NAME = "SDGCF"

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
    """Return the display name for an internal model key."""
    key = str(model_name)
    return MODEL_DISPLAY_NAMES.get(key.lower(), key)
