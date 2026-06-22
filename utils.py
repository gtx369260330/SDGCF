"""Common utilities."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, Any, Optional, Iterable

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


def set_seed(seed: int = 42) -> None:
    """Set random seeds without crashing on CUDA-visible but unsupported GPUs.

    Some newer NVIDIA GPUs can be visible to PyTorch while the installed PyTorch
    build does not contain kernels for that compute capability. In that state,
    torch.manual_seed() may internally touch CUDA and raise a CUDA error even
    when the experiment is intended to run on CPU. This function therefore falls
    back to the CPU default generator when CUDA seeding fails.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        torch.manual_seed(seed)
    except Exception as exc:
        print(f"[Seed warning] torch.manual_seed failed because CUDA appears unusable: {exc}")
        try:
            torch.random.default_generator.manual_seed(seed)
        except Exception:
            pass

    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception as exc:
        print(f"[Seed warning] torch.cuda.manual_seed_all skipped: {exc}")

    try:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


class EarlyStopping:
    """Early stopping on a maximized metric, e.g. val_macro_f1."""
    def __init__(self, patience: int = 8, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.best: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        if self.best is None:
            self.best = metric
            self.counter = 0
            return True
        improved = metric > self.best if self.mode == "max" else metric < self.best
        if improved:
            self.best = metric
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


def calculate_metrics(y_true: Iterable[int], y_pred: Iterable[int], class_names=None) -> Dict[str, Any]:
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    labels = list(range(len(class_names))) if class_names is not None else sorted(np.unique(y_true).tolist())

    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred, labels=labels)
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except Exception:
        mcc = 0.0
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    per_p, per_r, per_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names if class_names is not None else None,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(balanced_acc),
        "cohen_kappa": float(kappa),
        "mcc": float(mcc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "per_class_precision": per_p.tolist(),
        "per_class_recall": per_r.tolist(),
        "per_class_f1": per_f1.tolist(),
        "support": support.tolist(),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def get_checkpoint_path(save_dir: str | Path, model_name: str) -> Path:
    return Path(save_dir) / "checkpoints" / f"best_{model_name}.pt"


def safe_torch_load(path: str | Path, map_location="cpu"):
    """Load checkpoint with compatibility across PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
