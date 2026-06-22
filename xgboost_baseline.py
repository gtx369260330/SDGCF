"""XGBoost baseline on handcrafted EEG/EOG channel features."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_sample_weight

from data_loader import create_dataloaders
from utils import calculate_metrics, get_checkpoint_path, save_json, set_seed


MODEL_NAME = "xgboost"
METRIC_EXCLUDE_KEYS = {
    "y_true",
    "y_pred",
    "probabilities",
    "subjects",
    "indices",
    "features",
    "classification_report",
}


def _require_xgboost():
    try:
        import xgboost as xgb  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The XGBoost baseline requires the optional package 'xgboost'. "
            "Install it with: pip install xgboost"
        ) from exc
    return xgb


def _as_numpy_batch(batch: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
    x = batch["x"].detach().cpu().numpy().astype(np.float32, copy=False)
    y = batch["y"].detach().cpu().numpy().astype(int, copy=False)
    subjects = [str(subject) for subject in batch["subject_id"]]
    indices = batch["index"].detach().cpu().numpy().astype(int).tolist()
    return x, y, subjects, indices


def apply_missing_or_noise_numpy(
    x: np.ndarray,
    condition: str,
    noise_std: float = 0.2,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply the same missing/noisy modality protocol used by neural models."""
    x = np.array(x, copy=True)
    if condition in ["none", "clean", None]:
        return x

    rng = rng or np.random.default_rng()
    channels = x.shape[1]
    if condition == "missing_fpz":
        x[:, 0, :] = 0
    elif condition == "missing_pz":
        x[:, 1, :] = 0
    elif condition == "missing_eog":
        x[:, 2, :] = 0
    elif condition == "missing_random_one":
        for index in range(x.shape[0]):
            x[index, rng.integers(0, channels), :] = 0
    elif condition == "missing_two":
        for index in range(x.shape[0]):
            missing = rng.choice(channels, size=min(2, channels), replace=False)
            x[index, missing, :] = 0
    elif condition == "noise_fpz":
        x[:, 0, :] += rng.normal(0.0, noise_std, size=x[:, 0, :].shape).astype(np.float32)
    elif condition == "noise_pz":
        x[:, 1, :] += rng.normal(0.0, noise_std, size=x[:, 1, :].shape).astype(np.float32)
    elif condition == "noise_eog":
        x[:, 2, :] += rng.normal(0.0, noise_std, size=x[:, 2, :].shape).astype(np.float32)
    else:
        raise ValueError(f"Unknown missing/noise condition: {condition}")
    return x


def build_feature_names(channel_names: List[str], fft_bins: int) -> List[str]:
    base_names = [
        "mean",
        "std",
        "min",
        "max",
        "median",
        "q25",
        "q75",
        "iqr",
        "rms",
        "abs_mean",
        "zero_cross_rate",
        "diff_abs_mean",
        "diff_std",
        "skew",
        "kurtosis",
    ]
    names: List[str] = []
    for channel in channel_names:
        clean_channel = channel.replace(" ", "_").replace("/", "_")
        names.extend([f"{clean_channel}_{name}" for name in base_names])
        names.extend([f"{clean_channel}_fft_log_power_bin_{idx + 1}" for idx in range(fft_bins)])
        names.extend([f"{clean_channel}_fft_rel_power_bin_{idx + 1}" for idx in range(fft_bins)])
    return names


def extract_epoch_features(x: np.ndarray, fft_bins: int = 8) -> np.ndarray:
    """Extract compact per-channel statistical and coarse spectral features."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected X [N,C,T], got {x.shape}")

    eps = 1e-8
    features = []
    for channel_index in range(x.shape[1]):
        signal = x[:, channel_index, :]
        mean = signal.mean(axis=1)
        std = signal.std(axis=1) + eps
        q25 = np.percentile(signal, 25, axis=1)
        q75 = np.percentile(signal, 75, axis=1)
        centered = signal - mean[:, None]
        diff = np.diff(signal, axis=1)
        zero_cross_rate = (np.signbit(signal[:, 1:]) != np.signbit(signal[:, :-1])).mean(axis=1)

        channel_features = [
            mean,
            std,
            signal.min(axis=1),
            signal.max(axis=1),
            np.median(signal, axis=1),
            q25,
            q75,
            q75 - q25,
            np.sqrt(np.mean(signal * signal, axis=1) + eps),
            np.mean(np.abs(signal), axis=1),
            zero_cross_rate,
            np.mean(np.abs(diff), axis=1),
            diff.std(axis=1),
            np.mean(centered**3, axis=1) / (std**3),
            np.mean(centered**4, axis=1) / (std**4),
        ]

        power = np.abs(np.fft.rfft(signal, axis=1)) ** 2
        if power.shape[1] > 1:
            spectral_indices = np.arange(1, power.shape[1])
        else:
            spectral_indices = np.arange(power.shape[1])
        bands = np.array_split(spectral_indices, max(1, int(fft_bins)))
        total_power = power[:, spectral_indices].sum(axis=1) + eps if len(spectral_indices) else power.sum(axis=1) + eps
        for band in bands:
            if len(band) == 0:
                band_power = np.zeros(signal.shape[0], dtype=np.float32)
            else:
                band_power = power[:, band].sum(axis=1)
            channel_features.append(np.log1p(band_power / max(1, len(band))))
        for band in bands:
            if len(band) == 0:
                band_power = np.zeros(signal.shape[0], dtype=np.float32)
            else:
                band_power = power[:, band].sum(axis=1)
            channel_features.append(band_power / total_power)
        features.append(np.stack(channel_features, axis=1))

    return np.concatenate(features, axis=1).astype(np.float32, copy=False)


def collect_xgboost_features(
    loader,
    cfg,
    missing_condition: str = "none",
    noise_std: float = 0.2,
) -> Dict[str, Any]:
    rng = np.random.default_rng(int(cfg.RANDOM_SEED))
    features, labels, subjects, indices = [], [], [], []
    for batch in loader:
        x, y, batch_subjects, batch_indices = _as_numpy_batch(batch)
        x = apply_missing_or_noise_numpy(x, missing_condition, noise_std=noise_std, rng=rng)
        features.append(extract_epoch_features(x, fft_bins=cfg.XGBOOST_FFT_BINS))
        labels.append(y)
        subjects.extend(batch_subjects)
        indices.extend(batch_indices)
    return {
        "features": np.concatenate(features, axis=0),
        "labels": np.concatenate(labels, axis=0),
        "subjects": subjects,
        "indices": indices,
    }


def _add_probability_metrics(metrics: Dict[str, Any], y_true: np.ndarray, probabilities: np.ndarray, class_names: List[str]) -> None:
    try:
        labels = np.arange(len(class_names))
        if len(np.unique(y_true)) >= 2 and probabilities.shape[1] == len(class_names):
            y_binary = label_binarize(y_true, classes=labels)
            metrics["macro_auroc_ovr"] = float(
                roc_auc_score(y_true, probabilities, labels=labels, multi_class="ovr", average="macro")
            )
            metrics["weighted_auroc_ovr"] = float(
                roc_auc_score(y_true, probabilities, labels=labels, multi_class="ovr", average="weighted")
            )
            metrics["macro_auprc_ovr"] = float(average_precision_score(y_binary, probabilities, average="macro"))
            metrics["weighted_auprc_ovr"] = float(average_precision_score(y_binary, probabilities, average="weighted"))
    except Exception as exc:
        metrics["probability_metric_warning"] = str(exc)


def _evaluate_predictions(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: List[str],
    subjects: List[str],
    indices: List[int],
    features: np.ndarray | None = None,
) -> Dict[str, Any]:
    y_pred = probabilities.argmax(axis=1)
    metrics = calculate_metrics(y_true, y_pred, class_names=class_names)
    _add_probability_metrics(metrics, y_true, probabilities, class_names)
    metrics.update(
        {
            "y_true": y_true.astype(int).tolist(),
            "y_pred": y_pred.astype(int).tolist(),
            "probabilities": probabilities.astype(float).tolist(),
            "subjects": subjects,
            "indices": indices,
            "features": features,
        }
    )
    return metrics


def _save_predictions(out_dir: Path, model_name: str, condition: str, metrics: Dict[str, Any], class_names: List[str]) -> None:
    y_true = np.asarray(metrics["y_true"], dtype=int)
    y_pred = np.asarray(metrics["y_pred"], dtype=int)
    probabilities = np.asarray(metrics["probabilities"], dtype=float)
    predictions = pd.DataFrame(
        {
            "sample_index": metrics["indices"],
            "subject_id": metrics["subjects"],
            "y_true": y_true,
            "y_true_name": [class_names[index] for index in y_true],
            "y_pred": y_pred,
            "y_pred_name": [class_names[index] for index in y_pred],
            "correct": (y_true == y_pred).astype(int),
        }
    )
    for index, class_name in enumerate(class_names):
        predictions[f"prob_{class_name}"] = probabilities[:, index]
    predictions["pred_confidence"] = probabilities.max(axis=1)
    predictions["true_class_probability"] = probabilities[np.arange(len(y_true)), y_true]
    predictions.to_csv(out_dir / f"predictions_{model_name}_{condition}.csv", index=False)
    predictions[
        ["sample_index", "subject_id", "y_true", "y_pred"] + [f"prob_{name}" for name in class_names]
    ].to_csv(out_dir / f"probabilities_{model_name}_{condition}.csv", index=False)


def _save_test_outputs(cfg, model_name: str, condition: str, metrics: Dict[str, Any], save_details: bool) -> None:
    out_dir = Path(cfg.SAVE_DIR) / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        {key: value for key, value in metrics.items() if key not in METRIC_EXCLUDE_KEYS},
        out_dir / f"test_metrics_{model_name}_{condition}.json",
    )
    if save_details:
        np.save(out_dir / f"confusion_matrix_{model_name}_{condition}.npy", np.asarray(metrics["confusion_matrix"]))
        pd.DataFrame(metrics["classification_report"]).transpose().to_csv(
            out_dir / f"classification_report_{model_name}_{condition}.csv"
        )
        _save_predictions(out_dir, model_name, condition, metrics, cfg.CLASS_NAMES)
    if metrics.get("features") is not None:
        np.savez(
            out_dir / f"features_{model_name}_{condition}.npz",
            features=np.asarray(metrics["features"], dtype=np.float32),
            y_true=np.asarray(metrics["y_true"]),
            y_pred=np.asarray(metrics["y_pred"]),
            probabilities=np.asarray(metrics["probabilities"]),
            subjects=np.asarray(metrics["subjects"]),
        )


def train_xgboost_model(cfg, model_name: str = MODEL_NAME, device_str: str | None = None) -> Dict[str, Any]:
    del device_str
    xgb = _require_xgboost()
    cfg.make_dirs()
    set_seed(cfg.RANDOM_SEED)

    train_loader, val_loader, _, _, meta = create_dataloaders(cfg)
    train_data = collect_xgboost_features(train_loader, cfg)
    val_data = collect_xgboost_features(val_loader, cfg)

    params = {
        "objective": "multi:softprob",
        "num_class": cfg.NUM_CLASSES,
        "eval_metric": "mlogloss",
        "n_estimators": cfg.XGBOOST_N_ESTIMATORS,
        "max_depth": cfg.XGBOOST_MAX_DEPTH,
        "learning_rate": cfg.XGBOOST_LEARNING_RATE,
        "subsample": cfg.XGBOOST_SUBSAMPLE,
        "colsample_bytree": cfg.XGBOOST_COLSAMPLE_BYTREE,
        "reg_lambda": cfg.XGBOOST_REG_LAMBDA,
        "min_child_weight": cfg.XGBOOST_MIN_CHILD_WEIGHT,
        "tree_method": cfg.XGBOOST_TREE_METHOD,
        "random_state": cfg.RANDOM_SEED,
        "n_jobs": max(1, int(cfg.TORCH_NUM_THREADS)),
        "verbosity": 0,
    }
    model = xgb.XGBClassifier(**params)
    sample_weight = compute_sample_weight("balanced", train_data["labels"]) if cfg.USE_CLASS_WEIGHT else None
    model.fit(
        train_data["features"],
        train_data["labels"],
        sample_weight=sample_weight,
        eval_set=[(val_data["features"], val_data["labels"])],
        verbose=False,
    )

    train_metrics = _evaluate_predictions(
        train_data["labels"],
        model.predict_proba(train_data["features"]),
        cfg.CLASS_NAMES,
        train_data["subjects"],
        train_data["indices"],
    )
    val_metrics = _evaluate_predictions(
        val_data["labels"],
        model.predict_proba(val_data["features"]),
        cfg.CLASS_NAMES,
        val_data["subjects"],
        val_data["indices"],
    )

    checkpoint_path = get_checkpoint_path(cfg.SAVE_DIR, model_name)
    feature_names = build_feature_names(cfg.CHANNEL_NAMES, cfg.XGBOOST_FFT_BINS)
    joblib.dump(
        {
            "model_name": model_name,
            "model": model,
            "feature_names": feature_names,
            "params": params,
            "cfg": cfg.__dict__,
            "meta": meta,
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
            "n_estimators": cfg.XGBOOST_N_ESTIMATORS,
            "max_depth": cfg.XGBOOST_MAX_DEPTH,
            "learning_rate": cfg.XGBOOST_LEARNING_RATE,
        }
    ]
    logs_dir = Path(cfg.SAVE_DIR) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(logs_dir / f"training_history_{model_name}.csv", index=False)
    save_json(
        {
            "model_name": model_name,
            "best_val_macro_f1": val_metrics["macro_f1"],
            "checkpoint": str(checkpoint_path),
            "training_policy": "xgboost_handcrafted_channel_features",
            "feature_count": len(feature_names),
            "params": params,
            "data_shape": meta.get("X_shape"),
        },
        logs_dir / f"train_summary_{model_name}.json",
    )
    print(
        f"Model: {model_name} | XGBoost handcrafted features | Checkpoint: {checkpoint_path}\n"
        f"Train Macro-F1={train_metrics['macro_f1']:.4f} | Val Macro-F1={val_metrics['macro_f1']:.4f}"
    )
    return {"history": history, "checkpoint": str(checkpoint_path), "best_val_macro_f1": val_metrics["macro_f1"]}


def test_xgboost_model(
    cfg,
    model_name: str = MODEL_NAME,
    missing_condition: str = "none",
    collect_outputs: bool = True,
    save_details: bool = True,
) -> Dict[str, Any]:
    _require_xgboost()
    cfg.make_dirs()
    set_seed(cfg.RANDOM_SEED)

    checkpoint_path = get_checkpoint_path(cfg.SAVE_DIR, model_name)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            f"Train first: python main.py --model {model_name} --mode train"
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
    probabilities = model.predict_proba(test_data["features"])
    metrics = _evaluate_predictions(
        test_data["labels"],
        probabilities,
        cfg.CLASS_NAMES,
        test_data["subjects"],
        test_data["indices"],
        features=test_data["features"] if collect_outputs else None,
    )
    _save_test_outputs(cfg, model_name, missing_condition, metrics, save_details=save_details)
    print(
        f"Test {model_name} [{missing_condition}] | Acc={metrics['accuracy']:.4f} "
        f"Macro-F1={metrics['macro_f1']:.4f} Weighted-F1={metrics['weighted_f1']:.4f}"
    )
    return metrics
