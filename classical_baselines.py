"""Traditional ML baselines on handcrafted EEG/EOG channel features."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from data_loader import create_dataloaders
from utils import get_checkpoint_path, save_json, set_seed
from xgboost_baseline import (
    _evaluate_predictions,
    _save_test_outputs,
    build_feature_names,
    collect_xgboost_features,
)


CLASSICAL_BASELINE_MODEL_NAMES = ("random_forest", "svm_linear", "logistic_regression")
MODEL_TRAINING_POLICY = {
    "random_forest": "random_forest_handcrafted_channel_features",
    "svm_linear": "linear_svm_handcrafted_channel_features",
    "logistic_regression": "logistic_regression_handcrafted_channel_features",
}


def is_classical_baseline(model_name: str) -> bool:
    """Return whether a model key is handled by this module."""
    return str(model_name).lower() in CLASSICAL_BASELINE_MODEL_NAMES


def get_classical_training_policy(model_name: str) -> str:
    """Return the manifest/logging training policy for a classical baseline."""
    return MODEL_TRAINING_POLICY[str(model_name).lower()]


def _threads(cfg) -> int:
    return max(1, int(getattr(cfg, "TORCH_NUM_THREADS", 1) or 1))


def _model_specific_cap(cfg, model_name: str) -> int | None:
    name = model_name.lower()
    fallback = getattr(cfg, "CLASSICAL_MAX_TRAIN_SAMPLES", None)
    if name == "svm_linear":
        return getattr(cfg, "LINEAR_SVM_MAX_TRAIN_SAMPLES", fallback)
    if name == "logistic_regression":
        return getattr(cfg, "LOGISTIC_REGRESSION_MAX_TRAIN_SAMPLES", fallback)
    return fallback


def _select_train_subset(labels: np.ndarray, max_samples: int | None, seed: int) -> np.ndarray:
    labels = np.asarray(labels)
    indices = np.arange(len(labels))
    if max_samples is None or int(max_samples) <= 0 or int(max_samples) >= len(indices):
        return indices

    max_samples = int(max_samples)
    unique, counts = np.unique(labels, return_counts=True)
    can_stratify = max_samples >= len(unique) and np.all(counts >= 2)
    selected, _ = train_test_split(
        indices,
        train_size=max_samples,
        random_state=seed,
        stratify=labels if can_stratify else None,
    )
    return np.sort(selected)


def _build_estimator(cfg, model_name: str):
    name = model_name.lower()
    if name == "random_forest":
        params = {
            "n_estimators": int(cfg.RANDOM_FOREST_N_ESTIMATORS),
            "max_depth": cfg.RANDOM_FOREST_MAX_DEPTH,
            "min_samples_leaf": int(cfg.RANDOM_FOREST_MIN_SAMPLES_LEAF),
            "class_weight": "balanced_subsample" if cfg.USE_CLASS_WEIGHT else None,
            "random_state": int(cfg.RANDOM_SEED),
            "n_jobs": _threads(cfg),
        }
        return RandomForestClassifier(**params), params

    if name == "svm_linear":
        params = {
            "C": float(cfg.LINEAR_SVM_C),
            "class_weight": "balanced" if cfg.USE_CLASS_WEIGHT else None,
            "max_iter": int(cfg.LINEAR_SVM_MAX_ITER),
            "dual": False,
            "random_state": int(cfg.RANDOM_SEED),
        }
        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LinearSVC(**params)),
            ]
        )
        return estimator, params

    if name == "logistic_regression":
        params = {
            "C": float(cfg.LOGISTIC_REGRESSION_C),
            "class_weight": "balanced" if cfg.USE_CLASS_WEIGHT else None,
            "max_iter": int(cfg.LOGISTIC_REGRESSION_MAX_ITER),
            "random_state": int(cfg.RANDOM_SEED),
            "n_jobs": _threads(cfg),
        }
        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(**params)),
            ]
        )
        return estimator, params

    raise ValueError(f"Unknown classical baseline: {model_name}")


def _estimator_classes(model) -> Iterable[int]:
    classes = getattr(model, "classes_", None)
    if classes is not None:
        return classes
    if isinstance(model, Pipeline):
        return getattr(model.steps[-1][1], "classes_", [])
    return []


def _align_probabilities(probabilities: np.ndarray, classes: Iterable[int], num_classes: int) -> np.ndarray:
    raw = np.asarray(probabilities, dtype=float)
    aligned = np.zeros((raw.shape[0], int(num_classes)), dtype=float)
    classes = list(classes)
    if not classes and raw.shape[1] == num_classes:
        return raw
    for column, cls in enumerate(classes):
        cls = int(cls)
        if 0 <= cls < num_classes and column < raw.shape[1]:
            aligned[:, cls] = raw[:, column]
    row_sum = aligned.sum(axis=1, keepdims=True)
    empty = row_sum.squeeze(axis=1) <= 0
    if np.any(empty):
        aligned[empty] = 1.0 / float(num_classes)
        row_sum = aligned.sum(axis=1, keepdims=True)
    return aligned / np.maximum(row_sum, 1e-12)


def _softmax_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    scores = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / np.maximum(exp_scores.sum(axis=1, keepdims=True), 1e-12)


def _predict_probabilities(model, features: np.ndarray, num_classes: int) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(features)
    elif hasattr(model, "decision_function"):
        raw = _softmax_scores(model.decision_function(features))
    else:
        pred = model.predict(features)
        raw = np.zeros((len(pred), num_classes), dtype=float)
        raw[np.arange(len(pred)), np.asarray(pred, dtype=int)] = 1.0
    return _align_probabilities(raw, _estimator_classes(model), num_classes)


def _slice_list(values: list[Any], indices: np.ndarray) -> list[Any]:
    return [values[int(index)] for index in indices]


def _evaluate_estimator(model, data: Dict[str, Any], cfg, subset_indices: np.ndarray | None = None) -> Dict[str, Any]:
    if subset_indices is None:
        features = data["features"]
        labels = data["labels"]
        subjects = data["subjects"]
        indices = data["indices"]
    else:
        features = data["features"][subset_indices]
        labels = data["labels"][subset_indices]
        subjects = _slice_list(data["subjects"], subset_indices)
        indices = _slice_list(data["indices"], subset_indices)
    probabilities = _predict_probabilities(model, features, cfg.NUM_CLASSES)
    return _evaluate_predictions(labels, probabilities, cfg.CLASS_NAMES, subjects, indices, features=features)


def train_classical_model(cfg, model_name: str, device_str: str | None = None) -> Dict[str, Any]:
    """Train one classical baseline using handcrafted channel statistics."""
    del device_str
    name = model_name.lower()
    if not is_classical_baseline(name):
        raise ValueError(f"Unknown classical baseline: {model_name}")

    cfg.make_dirs()
    set_seed(cfg.RANDOM_SEED)
    train_loader, val_loader, _, _, meta = create_dataloaders(cfg)
    train_data = collect_xgboost_features(train_loader, cfg)
    val_data = collect_xgboost_features(val_loader, cfg)

    train_cap = _model_specific_cap(cfg, name)
    fit_indices = _select_train_subset(train_data["labels"], train_cap, int(cfg.RANDOM_SEED))
    estimator, params = _build_estimator(cfg, name)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        estimator.fit(train_data["features"][fit_indices], train_data["labels"][fit_indices])

    train_metrics = _evaluate_estimator(estimator, train_data, cfg, subset_indices=fit_indices)
    val_metrics = _evaluate_estimator(estimator, val_data, cfg)
    checkpoint_path = get_checkpoint_path(cfg.SAVE_DIR, name)
    feature_names = build_feature_names(cfg.CHANNEL_NAMES, cfg.XGBOOST_FFT_BINS)
    joblib.dump(
        {
            "model_name": name,
            "model": estimator,
            "feature_names": feature_names,
            "params": params,
            "cfg": cfg.__dict__,
            "meta": meta,
            "training_policy": get_classical_training_policy(name),
            "fit_sample_count": int(len(fit_indices)),
            "available_train_sample_count": int(len(train_data["labels"])),
            "best_val_macro_f1": val_metrics["macro_f1"],
        },
        checkpoint_path,
    )

    history = [
        {
            "epoch": 1,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "fit_sample_count": int(len(fit_indices)),
            "available_train_sample_count": int(len(train_data["labels"])),
        }
    ]
    logs_dir = Path(cfg.SAVE_DIR) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(logs_dir / f"training_history_{name}.csv", index=False)
    save_json(
        {
            "model_name": name,
            "best_val_macro_f1": val_metrics["macro_f1"],
            "checkpoint": str(checkpoint_path),
            "training_policy": get_classical_training_policy(name),
            "feature_count": len(feature_names),
            "params": params,
            "fit_sample_count": int(len(fit_indices)),
            "available_train_sample_count": int(len(train_data["labels"])),
            "data_shape": meta.get("X_shape"),
        },
        logs_dir / f"train_summary_{name}.json",
    )
    print(
        f"Model: {name} | handcrafted channel features | Checkpoint: {checkpoint_path}\n"
        f"Train Macro-F1={train_metrics['macro_f1']:.4f} | Val Macro-F1={val_metrics['macro_f1']:.4f}"
    )
    return {"history": history, "checkpoint": str(checkpoint_path), "best_val_macro_f1": val_metrics["macro_f1"]}


def test_classical_model(
    cfg,
    model_name: str,
    missing_condition: str = "none",
    collect_outputs: bool = True,
    save_details: bool = True,
) -> Dict[str, Any]:
    """Evaluate one classical baseline from its saved checkpoint."""
    name = model_name.lower()
    if not is_classical_baseline(name):
        raise ValueError(f"Unknown classical baseline: {model_name}")

    cfg.make_dirs()
    set_seed(cfg.RANDOM_SEED)
    checkpoint_path = get_checkpoint_path(cfg.SAVE_DIR, name)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            f"Train first: python main.py --model {name} --mode train"
        )
    checkpoint = joblib.load(checkpoint_path)
    model = checkpoint["model"]

    _, _, test_loader, _, _ = create_dataloaders(cfg)
    test_data = collect_xgboost_features(
        test_loader,
        cfg,
        missing_condition=missing_condition,
        noise_std=cfg.NOISE_STD,
    )
    metrics = _evaluate_estimator(model, test_data, cfg)
    if not collect_outputs:
        metrics["features"] = None
    _save_test_outputs(cfg, name, missing_condition, metrics, save_details=save_details)
    print(
        f"Test {name} [{missing_condition}] | Acc={metrics['accuracy']:.4f} "
        f"Macro-F1={metrics['macro_f1']:.4f} Weighted-F1={metrics['weighted_f1']:.4f}"
    )
    return metrics
