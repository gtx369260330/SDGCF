"""Evaluation, ablation and robustness testing for SDGCF."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

from config import Config
from data_loader import create_dataloaders
from train import _configure_safe_cuda_attention, _select_device, build_model, train_model
from utils import calculate_metrics, get_checkpoint_path, safe_torch_load, save_json, set_seed


METRIC_EXCLUDE_KEYS = {
    "y_true",
    "y_pred",
    "probabilities",
    "subjects",
    "indices",
    "attention_array",
    "features",
    "classification_report",
}


def apply_missing_or_noise(x: torch.Tensor, condition: str, noise_std: float = 0.2) -> torch.Tensor:
    """Apply one missing-channel or additive-noise condition."""
    x = x.clone()
    channels = x.size(1)
    if condition in ["none", "clean", None]:
        return x
    if condition == "missing_fpz":
        x[:, 0, :] = 0
    elif condition == "missing_pz":
        x[:, 1, :] = 0
    elif condition == "missing_eog":
        x[:, 2, :] = 0
    elif condition == "missing_random_one":
        for index in range(x.size(0)):
            x[index, random.randrange(channels), :] = 0
    elif condition == "missing_two":
        for index in range(x.size(0)):
            x[index, random.sample(range(channels), k=min(2, channels)), :] = 0
    elif condition == "noise_fpz":
        x[:, 0, :] += torch.randn_like(x[:, 0, :]) * noise_std
    elif condition == "noise_pz":
        x[:, 1, :] += torch.randn_like(x[:, 1, :]) * noise_std
    elif condition == "noise_eog":
        x[:, 2, :] += torch.randn_like(x[:, 2, :]) * noise_std
    else:
        raise ValueError(f"Unknown missing/noise condition: {condition}")
    return x


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    device,
    class_names: List[str],
    missing_condition: str = "none",
    noise_std: float = 0.2,
    collect_outputs: bool = True,
) -> Dict[str, Any]:
    """Evaluate a model and optionally collect graph outputs."""
    model.eval()
    y_true, y_pred, probabilities = [], [], []
    subjects, indices = [], []
    attentions, fused_features = [], []
    graph_alphas = []

    for batch in tqdm(loader, desc=f"Eval[{missing_condition}]", leave=False):
        x = apply_missing_or_noise(
            batch["x"].to(device, non_blocking=True),
            missing_condition,
            noise_std=noise_std,
        )
        y = batch["y"].to(device, non_blocking=True)
        out = model(x, return_features=True)
        logits = out["logits"]
        pred = logits.argmax(dim=1)
        probability = torch.softmax(logits, dim=1)

        y_true.extend(y.detach().cpu().numpy().tolist())
        y_pred.extend(pred.detach().cpu().numpy().tolist())
        probabilities.extend(probability.detach().cpu().numpy().tolist())
        subjects.extend(batch["subject_id"])
        indices.extend(batch["index"].numpy().tolist())

        if collect_outputs:
            if out.get("attention") is not None:
                attentions.append(out["attention"].detach().cpu().numpy())
            if out.get("fused") is not None:
                fused_features.append(out["fused"].detach().cpu().numpy())
            if out.get("graph_alpha") is not None:
                graph_alphas.append(float(out["graph_alpha"].detach().cpu().item()))

    metrics = calculate_metrics(y_true, y_pred, class_names=class_names)
    metrics.update(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "probabilities": probabilities,
            "subjects": subjects,
            "indices": indices,
            "attention_array": np.concatenate(attentions, axis=0) if attentions else None,
            "features": np.concatenate(fused_features, axis=0) if fused_features else None,
            "graph_alpha_mean": float(np.mean(graph_alphas)) if graph_alphas else None,
        }
    )

    try:
        y_array = np.asarray(y_true)
        probability_array = np.asarray(probabilities)
        labels = np.arange(len(class_names))
        if len(np.unique(y_array)) >= 2 and probability_array.shape[1] == len(class_names):
            y_binary = label_binarize(y_array, classes=labels)
            metrics["macro_auroc_ovr"] = float(
                roc_auc_score(y_array, probability_array, labels=labels, multi_class="ovr", average="macro")
            )
            metrics["weighted_auroc_ovr"] = float(
                roc_auc_score(y_array, probability_array, labels=labels, multi_class="ovr", average="weighted")
            )
            metrics["macro_auprc_ovr"] = float(average_precision_score(y_binary, probability_array, average="macro"))
            metrics["weighted_auprc_ovr"] = float(
                average_precision_score(y_binary, probability_array, average="weighted")
            )
    except Exception as exc:
        metrics["probability_metric_warning"] = str(exc)
    return metrics


def load_trained_model(cfg: Config, model_name: str, device):
    model = build_model(model_name, cfg).to(device)
    checkpoint_path = get_checkpoint_path(cfg.SAVE_DIR, model_name)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            f"Train first: python main.py --model {model_name} --mode train"
        )
    checkpoint = safe_torch_load(checkpoint_path, map_location=device)
    try:
        model.load_state_dict(checkpoint["model_state"])
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint is incompatible with the current '{model_name}' architecture: {checkpoint_path}. "
            "Retrain the model before evaluation."
        ) from exc
    return model


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


def test_model(
    cfg: Config,
    model_name: str = "sdgcf",
    device_str: Optional[str] = None,
    missing_condition: str = "none",
    collect_outputs: bool = True,
    save_details: bool = True,
) -> Dict[str, Any]:
    if model_name.lower() == "xgboost":
        from xgboost_baseline import test_xgboost_model

        return test_xgboost_model(
            cfg,
            model_name=model_name,
            missing_condition=missing_condition,
            collect_outputs=collect_outputs,
            save_details=save_details,
        )
    if model_name.lower() in {"random_forest", "svm_linear", "logistic_regression"}:
        from classical_baselines import test_classical_model

        return test_classical_model(
            cfg,
            model_name=model_name,
            missing_condition=missing_condition,
            collect_outputs=collect_outputs,
            save_details=save_details,
        )

    cfg.make_dirs()
    device = _select_device(device_str)
    set_seed(cfg.RANDOM_SEED)
    if device.type == "cuda":
        _configure_safe_cuda_attention()
    if cfg.TORCH_NUM_THREADS:
        torch.set_num_threads(int(cfg.TORCH_NUM_THREADS))
    _, _, test_loader, _, _ = create_dataloaders(cfg)
    model = load_trained_model(cfg, model_name, device)
    metrics = evaluate_model(
        model,
        test_loader,
        device,
        cfg.CLASS_NAMES,
        missing_condition,
        cfg.NOISE_STD,
        collect_outputs=collect_outputs,
    )

    out_dir = Path(cfg.SAVE_DIR) / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        {key: value for key, value in metrics.items() if key not in METRIC_EXCLUDE_KEYS},
        out_dir / f"test_metrics_{model_name}_{missing_condition}.json",
    )
    if save_details:
        np.save(out_dir / f"confusion_matrix_{model_name}_{missing_condition}.npy", np.asarray(metrics["confusion_matrix"]))
        pd.DataFrame(metrics["classification_report"]).transpose().to_csv(
            out_dir / f"classification_report_{model_name}_{missing_condition}.csv"
        )
        _save_predictions(out_dir, model_name, missing_condition, metrics, cfg.CLASS_NAMES)

    if metrics["attention_array"] is not None:
        np.save(out_dir / f"graph_attention_{model_name}_{missing_condition}.npy", metrics["attention_array"])
    if metrics["features"] is not None:
        np.savez(
            out_dir / f"features_{model_name}_{missing_condition}.npz",
            features=metrics["features"],
            y_true=np.asarray(metrics["y_true"]),
            y_pred=np.asarray(metrics["y_pred"]),
            probabilities=np.asarray(metrics["probabilities"]),
            subjects=np.asarray(metrics["subjects"]),
        )

    print(
        f"Test {model_name} [{missing_condition}] | Acc={metrics['accuracy']:.4f} "
        f"Macro-F1={metrics['macro_f1']:.4f} Weighted-F1={metrics['weighted_f1']:.4f}"
    )
    return metrics


def run_ablation(cfg: Config, device_str: Optional[str] = None, train_if_missing: bool = True) -> pd.DataFrame:
    models = [
        "single_fpz",
        "single_pz",
        "single_eog",
        "xgboost",
        "random_forest",
        "svm_linear",
        "logistic_regression",
        "simple_concat",
        "concat_transformer",
        "multimodal_concat",
        "sdgcf_fixed_graph",
        "sdgcf",
    ]
    rows = []
    for model_name in models:
        checkpoint = get_checkpoint_path(cfg.SAVE_DIR, model_name)
        if train_if_missing and not checkpoint.exists():
            train_model(cfg, model_name, device_str=device_str)
        metrics = test_model(cfg, model_name, device_str=device_str)
        rows.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_auroc_ovr": metrics.get("macro_auroc_ovr", np.nan),
                "macro_auprc_ovr": metrics.get("macro_auprc_ovr", np.nan),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(Path(cfg.SAVE_DIR) / "metrics" / "ablation_results.csv", index=False)
    print(result)
    return result


def run_missing_modality(cfg: Config, device_str: Optional[str] = None) -> pd.DataFrame:
    conditions = [
        "none",
        "missing_fpz",
        "missing_pz",
        "missing_eog",
        "missing_random_one",
        "missing_two",
        "noise_fpz",
        "noise_pz",
        "noise_eog",
    ]
    models = ["simple_concat", "concat_transformer", "multimodal_concat", "sdgcf"]
    rows = []
    for model_name in models:
        checkpoint = get_checkpoint_path(cfg.SAVE_DIR, model_name)
        if not checkpoint.exists():
            train_model(cfg, model_name, device_str=device_str)
        clean_macro_f1 = None
        for condition in conditions:
            metrics = test_model(
                cfg,
                model_name,
                device_str=device_str,
                missing_condition=condition,
                collect_outputs=False,
                save_details=False,
            )
            if condition == "none":
                clean_macro_f1 = metrics["macro_f1"]
            drop = 0.0 if condition == "none" else (clean_macro_f1 - metrics["macro_f1"]) / max(1e-8, clean_macro_f1) * 100
            rows.append(
                {
                    "model": model_name,
                    "condition": condition,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "weighted_f1": metrics["weighted_f1"],
                    "macro_f1_drop_percent": drop,
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(Path(cfg.SAVE_DIR) / "metrics" / "missing_modality_results.csv", index=False)
    print(result)
    return result
