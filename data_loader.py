"""Data loading utilities for Sleep-EDF EEG-EOG sleep staging."""
from __future__ import annotations

from pathlib import Path
import os
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.utils.class_weight import compute_class_weight


class SleepEpochDataset(Dataset):
    """Sleep epoch dataset.

    X shape: [N, C, T], y shape: [N].
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subjects: Optional[np.ndarray] = None,
        indices: Optional[np.ndarray] = None,
        modality_indices: Optional[List[int]] = None,
    ) -> None:
        if indices is not None:
            self.X = X[indices]
            self.y = y[indices]
            self.subjects = subjects[indices] if subjects is not None else np.arange(len(self.y))
        else:
            self.X = X
            self.y = y
            self.subjects = subjects if subjects is not None else np.arange(len(y))
        self.modality_indices = modality_indices

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.X[idx]
        if self.modality_indices is not None:
            x = x[self.modality_indices, :]
        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "y": torch.tensor(int(self.y[idx]), dtype=torch.long),
            "subject_id": str(self.subjects[idx]),
            "index": int(idx),
        }


def _ensure_nct(X: np.ndarray) -> np.ndarray:
    """Try to ensure X has shape [N, C, T]."""
    if X.ndim != 3:
        raise ValueError(f"X must be 3D [N, C, T], got shape {X.shape}")
    # If shape is [N, T, C] and C is small, transpose.
    if X.shape[1] > 16 and X.shape[2] <= 16:
        X = np.transpose(X, (0, 2, 1))
    return X.astype(np.float32)


def load_arrays(data_path: str | Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load data from .npz or directory with .npy files.

    Supported .npz keys:
    - X_all, y_all, subject_id_all
    - X, y, subjects

    Supported directory files:
    - X_all.npy, y_all.npy, subject_id_all.npy
    - or X_train/y_train/X_test/y_test for fallback random-like use.
    """
    data_path = Path(data_path)
    if data_path.is_file() and data_path.suffix == ".npz":
        data = np.load(data_path, allow_pickle=True)
        X = data["X_all"] if "X_all" in data else data["X"]
        y = data["y_all"] if "y_all" in data else data["y"]
        subjects = None
        for key in ["subject_id_all", "subjects", "subject_ids"]:
            if key in data:
                subjects = data[key]
                break
        return _ensure_nct(X), y.astype(np.int64), subjects

    if data_path.is_dir():
        x_all = data_path / "X_all.npy"
        y_all = data_path / "y_all.npy"
        sid_all = data_path / "subject_id_all.npy"
        if x_all.exists() and y_all.exists():
            X = np.load(x_all, allow_pickle=True)
            y = np.load(y_all, allow_pickle=True)
            subjects = np.load(sid_all, allow_pickle=True) if sid_all.exists() else None
            return _ensure_nct(X), y.astype(np.int64), subjects

        # Fallback for already split arrays: concatenate and create pseudo subjects.
        x_train, y_train = data_path / "X_train.npy", data_path / "y_train.npy"
        x_test, y_test = data_path / "X_test.npy", data_path / "y_test.npy"
        if x_train.exists() and y_train.exists() and x_test.exists() and y_test.exists():
            Xtr, ytr = np.load(x_train), np.load(y_train)
            Xte, yte = np.load(x_test), np.load(y_test)
            X = np.concatenate([Xtr, Xte], axis=0)
            y = np.concatenate([ytr, yte], axis=0)
            subjects = np.arange(len(y))
            return _ensure_nct(X), y.astype(np.int64), subjects

    raise FileNotFoundError(
        f"Cannot find supported data files in {data_path}. Expected .npz or X_all.npy/y_all.npy."
    )


def split_by_subject(
    y: np.ndarray,
    subjects: Optional[np.ndarray],
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    n = len(y)
    all_idx = np.arange(n)
    if subjects is None:
        train_idx, tmp_idx = train_test_split(
            all_idx, train_size=train_ratio, random_state=seed, stratify=y if len(np.unique(y)) > 1 else None
        )
        rel_val = val_ratio / max(1e-8, (val_ratio + test_ratio))
        val_idx, test_idx = train_test_split(
            tmp_idx, train_size=rel_val, random_state=seed, stratify=y[tmp_idx] if len(np.unique(y[tmp_idx])) > 1 else None
        )
        return {"train": train_idx, "val": val_idx, "test": test_idx}

    subjects = np.asarray(subjects)
    # First split train vs temp by groups.
    gss = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
    train_idx, tmp_idx = next(gss.split(all_idx, y, groups=subjects))
    # Split temp into val/test by groups.
    tmp_subjects = subjects[tmp_idx]
    rel_val = val_ratio / max(1e-8, (val_ratio + test_ratio))
    gss2 = GroupShuffleSplit(n_splits=1, train_size=rel_val, random_state=seed + 1)
    val_rel, test_rel = next(gss2.split(tmp_idx, y[tmp_idx], groups=tmp_subjects))
    val_idx, test_idx = tmp_idx[val_rel], tmp_idx[test_rel]
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def _subset_indices(
    indices: np.ndarray,
    y: np.ndarray,
    fraction: float = 1.0,
    seed: int = 42,
    stratified: bool = True,
) -> np.ndarray:
    """Return a deterministic subset of a split for quick experiments.

    The subset is sampled *after* subject/group splitting, so train/val/test leakage
    control is unchanged. With stratified=True, at least one sample from each present
    class is retained whenever possible.
    """
    indices = np.asarray(indices)
    fraction = float(fraction)
    if fraction <= 0 or fraction > 1:
        raise ValueError(f"DATA_FRACTION must be in (0, 1], got {fraction}")
    if fraction >= 0.999999 or len(indices) == 0:
        return indices
    rng = np.random.default_rng(seed)
    target_n = max(1, int(round(len(indices) * fraction)))
    if not stratified:
        chosen = rng.choice(indices, size=target_n, replace=False)
        return np.sort(chosen)

    chosen_parts = []
    labels = y[indices]
    for cls in np.unique(labels):
        cls_idx = indices[labels == cls]
        cls_n = max(1, int(round(len(cls_idx) * fraction)))
        cls_n = min(cls_n, len(cls_idx))
        chosen_parts.append(rng.choice(cls_idx, size=cls_n, replace=False))
    chosen = np.concatenate(chosen_parts) if chosen_parts else np.array([], dtype=indices.dtype)

    # Rounding class-by-class can make the subset slightly larger/smaller than target_n.
    if len(chosen) > target_n:
        chosen = rng.choice(chosen, size=target_n, replace=False)
    elif len(chosen) < target_n:
        remaining = np.setdiff1d(indices, chosen, assume_unique=False)
        if len(remaining) > 0:
            extra_n = min(target_n - len(chosen), len(remaining))
            extra = rng.choice(remaining, size=extra_n, replace=False)
            chosen = np.concatenate([chosen, extra])
    return np.sort(chosen)


def compute_weights(y_train: np.ndarray, num_classes: int = 5) -> torch.Tensor:
    classes = np.arange(num_classes)
    present = np.unique(y_train)
    weights = np.ones(num_classes, dtype=np.float32)
    computed = compute_class_weight(class_weight="balanced", classes=present, y=y_train)
    for cls, w in zip(present, computed):
        weights[int(cls)] = float(w)
    return torch.tensor(weights, dtype=torch.float32)


def create_dataloaders(cfg, modality_indices: Optional[List[int]] = None):
    X, y, subjects = load_arrays(cfg.DATA_PATH)
    if X.shape[1] < 3 and modality_indices is None:
        print(f"Warning: X has {X.shape[1]} channels. SDGCF expects 3 channels.")
    splits = split_by_subject(
        y,
        subjects,
        train_ratio=cfg.TRAIN_RATIO,
        val_ratio=cfg.VAL_RATIO,
        test_ratio=cfg.TEST_RATIO,
        seed=cfg.RANDOM_SEED,
    )

    data_fraction = float(getattr(cfg, "DATA_FRACTION", 1.0))
    if data_fraction < 1.0:
        stratified = bool(getattr(cfg, "SUBSET_STRATIFIED", True))
        original_sizes = {k: int(len(v)) for k, v in splits.items()}
        splits = {
            k: _subset_indices(v, y, data_fraction, seed=int(cfg.RANDOM_SEED) + i * 997, stratified=stratified)
            for i, (k, v) in enumerate(splits.items())
        }
        subset_sizes = {k: int(len(v)) for k, v in splits.items()}
        print(f"[Data fraction] DATA_FRACTION={data_fraction:.3f} | original={original_sizes} | subset={subset_sizes}")

    train_set = SleepEpochDataset(X, y, subjects, splits["train"], modality_indices)
    val_set = SleepEpochDataset(X, y, subjects, splits["val"], modality_indices)
    test_set = SleepEpochDataset(X, y, subjects, splits["test"], modality_indices)

    requested_workers = int(getattr(cfg, "NUM_WORKERS", 0))
    # Windows spawn-based multiprocessing can crash with very large in-memory numpy
    # arrays, producing OSError: [Errno 22] Invalid argument or "pickle data was
    # truncated". For this project the safest default is single-process loading.
    if os.name == "nt" and requested_workers > 0:
        print(f"[DataLoader warning] Windows detected; overriding NUM_WORKERS={requested_workers} to 0 to avoid multiprocessing pickle errors.")
        requested_workers = 0
    pin_memory = bool(getattr(cfg, "PIN_MEMORY", False)) and requested_workers >= 0
    common_kwargs = dict(num_workers=requested_workers, pin_memory=pin_memory)
    if requested_workers > 0:
        common_kwargs.update(persistent_workers=True, prefetch_factor=2)

    train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE, shuffle=True, **common_kwargs)
    val_loader = DataLoader(val_set, batch_size=cfg.BATCH_SIZE, shuffle=False, **common_kwargs)
    test_loader = DataLoader(test_set, batch_size=cfg.BATCH_SIZE, shuffle=False, **common_kwargs)
    class_weights = compute_weights(y[splits["train"]], cfg.NUM_CLASSES)
    meta = {
        "X_shape": X.shape,
        "splits": {k: v.tolist() for k, v in splits.items()},
        "subjects": subjects,
        "data_fraction": float(getattr(cfg, "DATA_FRACTION", 1.0)),
    }
    return train_loader, val_loader, test_loader, class_weights, meta
