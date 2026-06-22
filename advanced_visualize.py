"""Paper-ready visualization utilities for SDGCF experiment batches.

All figures are saved as PNG, PDF and SVG. The module is intentionally robust:
missing files create a small placeholder figure rather than crashing the whole batch.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.decomposition import PCA

from model_naming import (
    CANONICAL_MODEL_DISPLAY_NAME,
    CANONICAL_MODEL_KEY,
    display_model_name,
    sdgcf_figure_name,
)


CLASS_NAMES_DEFAULT = ["W", "N1", "N2", "N3", "REM"]
CHANNEL_SHORT = ["Fpz-Cz", "Pz-Oz", "EOG"]
METRIC_COLUMNS = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]


# A clean paper style. No seaborn dependency is used.
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 120,
    "savefig.dpi": 600,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _figure_dir(save_dir: str | Path, subdir: str = "paper_figures") -> Path:
    fig_dir = Path(save_dir) / "figures" / subdir
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def _save(fig: plt.Figure, out_dir: Path, name: str, tight: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    # Export high-resolution PNG. Vector export can hang on some Windows/Python/Matplotlib
    # combinations, so the default batch pipeline prioritizes robust completion.
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def _placeholder(out_dir: Path, name: str, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=10, wrap=True)
    _save(fig, out_dir, name)


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_metrics_file(save_dir: str | Path, model_name: str, condition: str = "none") -> Path:
    return Path(save_dir) / "metrics" / f"test_metrics_{model_name}_{condition}.json"


def _find_predictions_file(save_dir: str | Path, model_name: str, condition: str = "none") -> Path:
    return Path(save_dir) / "metrics" / f"predictions_{model_name}_{condition}.csv"


def _find_report_file(save_dir: str | Path, model_name: str, condition: str = "none") -> Path:
    return Path(save_dir) / "metrics" / f"classification_report_{model_name}_{condition}.csv"


def _find_cm_file(save_dir: str | Path, model_name: str, condition: str = "none") -> Path:
    return Path(save_dir) / "metrics" / f"confusion_matrix_{model_name}_{condition}.npy"


def _ece(y_true: np.ndarray, y_pred: np.ndarray, confidence: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    accs = np.full(n_bins, np.nan)
    confs = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins)
    ece_val = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidence >= lo) & (confidence < hi if i < n_bins - 1 else confidence <= hi)
        if not np.any(mask):
            continue
        counts[i] = mask.sum()
        accs[i] = (y_true[mask] == y_pred[mask]).mean()
        confs[i] = confidence[mask].mean()
        ece_val += (mask.mean()) * abs(accs[i] - confs[i])
    centers = (bins[:-1] + bins[1:]) / 2
    return centers, accs, counts, float(ece_val)


def plot_training_curve(save_dir: str | Path, model_name: str, out_dir: Path) -> None:
    model_display = display_model_name(model_name)
    path = Path(save_dir) / "logs" / f"training_history_{model_name}.csv"
    if not path.exists():
        _placeholder(out_dir, "training_curve", "Training curve", f"Missing {path.name}")
        return
    df = pd.read_csv(path)
    fig, ax1 = plt.subplots(figsize=(7.5, 4.8))
    if {"train_loss", "val_loss"}.issubset(df.columns):
        ax1.plot(df["epoch"], df["train_loss"], marker="o", linewidth=1.8, label="Training loss")
        ax1.plot(df["epoch"], df["val_loss"], marker="s", linewidth=1.8, label="Validation loss")
        ax1.set_ylabel("Loss")
        ax2 = ax1.twinx()
        ax2.plot(df["epoch"], df["val_accuracy"], linestyle="--", linewidth=1.8, label="Validation accuracy")
        ax2.plot(df["epoch"], df["val_macro_f1"], linestyle="--", linewidth=1.8, label="Validation macro-F1")
        ax2.set_ylabel("Score")
        ax2.set_ylim(0, 1.05)
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="best", frameon=False)
    else:
        metric_columns = [column for column in ["train_accuracy", "val_accuracy", "train_macro_f1", "val_macro_f1"] if column in df.columns]
        if not metric_columns:
            _placeholder(out_dir, "training_curve", "Training curve", f"No plottable metric columns in {path.name}")
            plt.close(fig)
            return
        for column in metric_columns:
            ax1.plot(df["epoch"], df[column], marker="o", linewidth=1.8, label=column.replace("_", " "))
        ax1.set_ylabel("Score")
        ax1.set_ylim(0, 1.05)
        ax1.legend(loc="best", frameon=False)
    ax1.set_xlabel("Epoch")
    ax1.grid(alpha=0.25, axis="y")
    ax1.set_title(f"Training dynamics: {model_display}")
    _save(fig, out_dir, "training_curve")


def plot_confusion_matrix(save_dir: str | Path, model_name: str, class_names: Sequence[str], out_dir: Path) -> None:
    model_display = display_model_name(model_name)
    path = _find_cm_file(save_dir, model_name)
    if not path.exists():
        _placeholder(out_dir, "confusion_matrix", "Confusion matrix", f"Missing {path.name}")
        return
    cm = np.load(path)
    for normalized in [False, True]:
        data = cm.astype(float)
        if normalized:
            data = data / np.maximum(data.sum(axis=1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(5.7, 4.8))
        im = ax.imshow(data, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(class_names)), class_names)
        ax.set_yticks(range(len(class_names)), class_names)
        ax.set_xlabel("Predicted stage")
        ax.set_ylabel("True stage")
        ax.set_title(f"{'Normalized ' if normalized else ''}confusion matrix: {model_display}")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                txt = f"{data[i, j]:.2f}" if normalized else f"{int(data[i, j])}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        _save(fig, out_dir, "confusion_matrix_normalized" if normalized else "confusion_matrix")


def plot_per_class_metrics(save_dir: str | Path, model_name: str, class_names: Sequence[str], out_dir: Path) -> None:
    model_display = display_model_name(model_name)
    path = _find_report_file(save_dir, model_name)
    if not path.exists():
        _placeholder(out_dir, "per_class_metrics", "Per-class metrics", f"Missing {path.name}")
        return
    df = pd.read_csv(path, index_col=0)
    if not all(c in df.index for c in class_names):
        _placeholder(out_dir, "per_class_metrics", "Per-class metrics", "Classification report does not contain all class names.")
        return
    vals = df.loc[list(class_names), ["precision", "recall", "f1-score"]]
    x = np.arange(len(class_names))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(x - width, vals["precision"], width, label="Precision")
    ax.bar(x, vals["recall"], width, label="Recall")
    ax.bar(x + width, vals["f1-score"], width, label="F1-score")
    ax.set_xticks(x, class_names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    ax.set_title(f"Stage-wise classification performance: {model_display}")
    _save(fig, out_dir, "per_class_metrics")


def plot_confidence_and_calibration(save_dir: str | Path, model_name: str, class_names: Sequence[str], out_dir: Path) -> None:
    model_display = display_model_name(model_name)
    pred_path = _find_predictions_file(save_dir, model_name)
    if not pred_path.exists():
        _placeholder(out_dir, "confidence_distribution", "Confidence distribution", f"Missing {pred_path.name}")
        return
    df = pd.read_csv(pred_path)
    if "pred_confidence" not in df.columns:
        _placeholder(out_dir, "confidence_distribution", "Confidence distribution", "Prediction probabilities were not saved.")
        return
    correct = df[df["correct"] == 1]["pred_confidence"].values
    incorrect = df[df["correct"] == 0]["pred_confidence"].values
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    bins = np.linspace(0, 1, 21)
    ax.hist(correct, bins=bins, alpha=0.65, label="Correct")
    ax.hist(incorrect, bins=bins, alpha=0.65, label="Incorrect")
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Epoch count")
    ax.set_title(f"Confidence distribution: {model_display}")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, out_dir, "confidence_distribution")

    centers, accs, counts, ece_val = _ece(
        df["y_true"].values.astype(int),
        df["y_pred"].values.astype(int),
        df["pred_confidence"].values.astype(float),
        n_bins=10,
    )
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, label="Perfect calibration")
    mask = ~np.isnan(accs)
    ax.plot(centers[mask], accs[mask], marker="o", linewidth=1.8, label=f"Model (ECE={ece_val:.3f})")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="best")
    ax.set_title(f"Reliability diagram: {model_display}")
    _save(fig, out_dir, "calibration_reliability_diagram")


def plot_roc_pr_curves(save_dir: str | Path, model_name: str, class_names: Sequence[str], out_dir: Path) -> None:
    model_display = display_model_name(model_name)
    pred_path = _find_predictions_file(save_dir, model_name)
    if not pred_path.exists():
        _placeholder(out_dir, "roc_curves", "ROC curves", f"Missing {pred_path.name}")
        return
    df = pd.read_csv(pred_path)
    prob_cols = [f"prob_{c}" for c in class_names]
    if not all(c in df.columns for c in prob_cols):
        _placeholder(out_dir, "roc_curves", "ROC curves", "Probability columns were not saved.")
        return
    y = df["y_true"].values.astype(int)
    probs = df[prob_cols].values.astype(float)
    y_bin = np.zeros_like(probs)
    y_bin[np.arange(len(y)), y] = 1

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for i, name in enumerate(class_names):
        if y_bin[:, i].sum() == 0 or y_bin[:, i].sum() == len(y_bin):
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        ax.plot(fpr, tpr, linewidth=1.6, label=f"{name} AUC={auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"One-vs-rest ROC curves: {model_display}")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    _save(fig, out_dir, "roc_curves")

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for i, name in enumerate(class_names):
        if y_bin[:, i].sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(y_bin[:, i], probs[:, i])
        ap = average_precision_score(y_bin[:, i], probs[:, i])
        ax.plot(recall, precision, linewidth=1.6, label=f"{name} AP={ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"One-vs-rest precision-recall curves: {model_display}")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    _save(fig, out_dir, "precision_recall_curves")


def plot_graph_attention(save_dir: str | Path, model_name: str, class_names: Sequence[str], out_dir: Path) -> None:
    model_display = display_model_name(model_name)
    att_path = Path(save_dir) / "metrics" / f"graph_attention_{model_name}_none.npy"
    pred_path = _find_predictions_file(save_dir, model_name)
    if not att_path.exists() or not pred_path.exists():
        _placeholder(out_dir, "graph_attention", "Dynamic graph attention", f"Missing {att_path.name} or {pred_path.name}")
        return
    att = np.load(att_path)
    df = pd.read_csv(pred_path)
    mean_att = att.mean(axis=0)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(mean_att, cmap="Purples", vmin=0, vmax=1)
    ax.set_xticks(range(3), CHANNEL_SHORT)
    ax.set_yticks(range(3), CHANNEL_SHORT)
    ax.set_xlabel("Attended modality")
    ax.set_ylabel("Query modality")
    ax.set_title(f"Mean dynamic graph attention: {model_display}")
    for i in range(mean_att.shape[0]):
        for j in range(mean_att.shape[1]):
            ax.text(j, i, f"{mean_att[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, out_dir, "graph_attention")

    n_classes = len(class_names)
    fig = plt.figure(figsize=(max(12.5, 2.55 * n_classes + 1.8), 3.9))
    grid = fig.add_gridspec(
        1,
        n_classes + 1,
        width_ratios=[1.0] * n_classes + [0.055],
        wspace=0.34,
    )
    axes = []
    for cls in range(n_classes):
        if cls == 0:
            ax = fig.add_subplot(grid[0, cls])
        else:
            ax = fig.add_subplot(grid[0, cls], sharex=axes[0], sharey=axes[0])
        axes.append(ax)

    im = None
    for cls, ax in enumerate(axes):
        mask = df["y_true"].values == cls
        mat = att[mask].mean(axis=0) if np.any(mask) else np.zeros((3, 3))
        im = ax.imshow(mat, cmap="Purples", vmin=0, vmax=1)
        ax.set_title(class_names[cls], pad=8)
        ax.set_aspect("equal")
        ax.set_xticks(range(3), CHANNEL_SHORT, rotation=45, ha="right")
        if cls == 0:
            ax.set_yticks(range(3), CHANNEL_SHORT)
            ax.set_ylabel("Query modality")
        else:
            ax.tick_params(axis="y", labelleft=False)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    cax = fig.add_subplot(grid[0, -1])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Attention weight", rotation=270, labelpad=14)
    fig.suptitle(f"Stage-wise graph attention: {model_display}", y=0.98)
    fig.supxlabel("Attended modality", y=0.03)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.83, bottom=0.24)
    _save(fig, out_dir, "graph_attention_stagewise", tight=False)


def plot_feature_embedding(save_dir: str | Path, model_name: str, class_names: Sequence[str], out_dir: Path, max_points: int = 3000) -> None:
    """Project fused features into 2-D with PCA for a fast, deterministic paper figure."""
    model_display = display_model_name(model_name)
    path = Path(save_dir) / "metrics" / f"features_{model_name}_none.npz"
    if not path.exists():
        _placeholder(out_dir, "pca_fused_features", "Feature visualization", f"Missing {path.name}")
        return
    data = np.load(path, allow_pickle=True)
    X = data["features"]
    y = data["y_true"].astype(int)
    if len(X) < 3:
        _placeholder(out_dir, "pca_fused_features", "Feature visualization", "Too few samples for feature projection.")
        return
    if len(X) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=max_points, replace=False)
        X, y = X[idx], y[idx]
    X = X.reshape(X.shape[0], -1)
    emb = PCA(n_components=2, random_state=42).fit_transform(X)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    for cls, name in enumerate(class_names):
        mask = y == cls
        if np.any(mask):
            ax.scatter(emb[mask, 0], emb[mask, 1], s=10, alpha=0.65, label=name, linewidths=0, rasterized=True)
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.set_title(f"Fused feature embedding: {model_display}")
    ax.legend(frameon=False, markerscale=2, ncol=1)
    ax.grid(alpha=0.2)
    _save(fig, out_dir, "pca_fused_features")


def plot_hypnogram(save_dir: str | Path, model_name: str, class_names: Sequence[str], out_dir: Path) -> None:
    model_display = display_model_name(model_name)
    path = _find_predictions_file(save_dir, model_name)
    if not path.exists():
        _placeholder(out_dir, "hypnogram_prediction", "Hypnogram", f"Missing {path.name}")
        return
    df = pd.read_csv(path)
    if df.empty:
        _placeholder(out_dir, "hypnogram_prediction", "Hypnogram", "Prediction file is empty.")
        return
    # Pick the subject with the most epochs to make the figure informative.
    subject = df["subject_id"].value_counts().index[0]
    d = df[df["subject_id"] == subject].sort_values("sample_index").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    ax.step(np.arange(len(d)), d["y_true"], where="mid", linewidth=1.8, label="Ground truth")
    ax.step(np.arange(len(d)), d["y_pred"], where="mid", linewidth=1.3, alpha=0.8, label="Prediction")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.invert_yaxis()
    ax.set_xlabel("Epoch index")
    ax.set_ylabel("Sleep stage")
    ax.set_title(f"Hypnogram comparison for subject {subject}: {model_display}")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.grid(alpha=0.18)
    _save(fig, out_dir, "hypnogram_prediction")


def plot_error_analysis(save_dir: str | Path, model_name: str, class_names: Sequence[str], out_dir: Path) -> None:
    model_display = display_model_name(model_name)
    path = _find_predictions_file(save_dir, model_name)
    if not path.exists():
        _placeholder(out_dir, "error_pairs", "Error analysis", f"Missing {path.name}")
        return
    df = pd.read_csv(path)
    err = df[df["y_true"] != df["y_pred"]].copy()
    if err.empty:
        _placeholder(out_dir, "error_pairs", "Error analysis", "No misclassified samples were found.")
        return
    err["pair"] = err.apply(lambda r: f"{class_names[int(r.y_true)]}->{class_names[int(r.y_pred)]}", axis=1)
    counts = err["pair"].value_counts().head(12)
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.bar(counts.index, counts.values)
    ax.set_xticks(np.arange(len(counts)), counts.index, rotation=35, ha="right")
    ax.set_ylabel("Number of errors")
    ax.set_title(f"Most frequent confusion pairs: {model_display}")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, out_dir, "error_pairs")


def generate_model_figures(save_dir: str | Path, model_name: str, class_names: Sequence[str] = CLASS_NAMES_DEFAULT) -> None:
    """Generate all paper figures for one model result folder."""
    out_dir = _figure_dir(save_dir)
    tasks = [
        ("training_curve", lambda: plot_training_curve(save_dir, model_name, out_dir)),
        ("confusion_matrix", lambda: plot_confusion_matrix(save_dir, model_name, class_names, out_dir)),
        ("per_class_metrics", lambda: plot_per_class_metrics(save_dir, model_name, class_names, out_dir)),
        ("confidence_calibration", lambda: plot_confidence_and_calibration(save_dir, model_name, class_names, out_dir)),
        ("roc_pr_curves", lambda: plot_roc_pr_curves(save_dir, model_name, class_names, out_dir)),
        ("graph_attention", lambda: plot_graph_attention(save_dir, model_name, class_names, out_dir)),
        ("feature_embedding", lambda: plot_feature_embedding(save_dir, model_name, class_names, out_dir)),
        ("hypnogram", lambda: plot_hypnogram(save_dir, model_name, class_names, out_dir)),
        ("error_analysis", lambda: plot_error_analysis(save_dir, model_name, class_names, out_dir)),
    ]
    for name, fn in tasks:
        print(f"[Figure] {model_name}: {name} ...", flush=True)
        fn()
        print(f"[Figure] {model_name}: {name} done", flush=True)


def _model_type(model: str) -> str:
    if model.startswith("single_") or model in [
        "simple_concat",
        "concat_transformer",
        "xgboost",
        "random_forest",
        "svm_linear",
        "logistic_regression",
    ]:
        return "baseline"
    if model in ["multimodal_concat", "sdgcf_fixed_graph"]:
        return "ablation"
    return "proposed"


def generate_summary_figures(summary_dir: str | Path, class_names: Sequence[str] = CLASS_NAMES_DEFAULT) -> None:
    """Generate cross-model comparison figures from summary CSV files."""
    summary_dir = Path(summary_dir)
    fig_dir = summary_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = summary_dir / "all_model_metrics.csv"
    if not metrics_path.exists():
        _placeholder(fig_dir, "model_comparison", "Model comparison", f"Missing {metrics_path.name}")
        return
    df = pd.read_csv(metrics_path)
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "success"].copy()
    if "macro_f1" in df.columns:
        df = df.dropna(subset=["macro_f1"]).copy()
    if df.empty:
        _placeholder(fig_dir, "model_comparison", "Model comparison", "No successful experiment was recorded.")
        return
    df = df.sort_values("macro_f1", ascending=True)
    df["model_display"] = df["model"].map(display_model_name)

    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.45 * len(df))))
    ax.barh(df["model_display"], df["macro_f1"])
    ax.set_xlabel("Macro-F1")
    ax.set_xlim(0, 1.05)
    ax.set_title(f"{CANONICAL_MODEL_DISPLAY_NAME} model comparison")
    ax.grid(axis="x", alpha=0.25)
    for i, v in enumerate(df["macro_f1"]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    _save(fig, fig_dir, sdgcf_figure_name("model_comparison_macro_f1"))

    metrics = [m for m in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "macro_auroc_ovr", "macro_auprc_ovr"] if m in df.columns]
    heat = df.set_index("model")[metrics].sort_values("macro_f1", ascending=False)
    heat.index = [display_model_name(model) for model in heat.index]
    fig, ax = plt.subplots(figsize=(1.25 * len(metrics) + 3.0, 0.42 * len(heat) + 2.5))
    im = ax.imshow(heat.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(metrics)), metrics, rotation=35, ha="right")
    ax.set_yticks(range(len(heat)), heat.index)
    ax.set_title(f"{CANONICAL_MODEL_DISPLAY_NAME} metric heatmap across models")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat.values[i, j]
            ax.text(j, i, "NA" if np.isnan(val) else f"{val:.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    _save(fig, fig_dir, sdgcf_figure_name("metric_heatmap"))

    # Ablation delta relative to proposed model.
    if CANONICAL_MODEL_KEY in df["model"].values:
        proposed_f1 = float(df.loc[df["model"] == CANONICAL_MODEL_KEY, "macro_f1"].iloc[0])
        ab = df[df["model"].map(_model_type) == "ablation"].copy()
        if not ab.empty:
            ab["delta_macro_f1_vs_sdgcf"] = ab["macro_f1"] - proposed_f1
            ab.to_csv(summary_dir / "ablation_delta_vs_sdgcf.csv", index=False)
            ab = ab.sort_values("delta_macro_f1_vs_sdgcf")
            fig, ax = plt.subplots(figsize=(8.0, 4.3))
            ax.barh(ab["model"].map(display_model_name), ab["delta_macro_f1_vs_sdgcf"])
            ax.axvline(0, linestyle="--", linewidth=1.0)
            ax.set_xlabel(f"Macro-F1 difference relative to {CANONICAL_MODEL_DISPLAY_NAME}")
            ax.set_title(f"{CANONICAL_MODEL_DISPLAY_NAME} ablation impact analysis")
            ax.grid(axis="x", alpha=0.25)
            _save(fig, fig_dir, sdgcf_figure_name("ablation_delta_vs_sdgcf"))

    # Radar chart for top models.
    top = df.sort_values("macro_f1", ascending=False).head(min(5, len(df)))
    radar_metrics = [m for m in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"] if m in top.columns]
    if len(top) > 0 and len(radar_metrics) >= 3:
        angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
        angles += angles[:1]
        fig = plt.figure(figsize=(6.4, 6.4))
        ax = fig.add_subplot(111, polar=True)
        for _, row in top.iterrows():
            values = [float(row[m]) for m in radar_metrics]
            values += values[:1]
            ax.plot(angles, values, linewidth=1.6, label=display_model_name(row["model"]))
            ax.fill(angles, values, alpha=0.08)
        ax.set_thetagrids(np.degrees(angles[:-1]), radar_metrics)
        ax.set_ylim(0.5, 0.8)
        ax.set_yticks(np.arange(0.5, 0.81, 0.05))
        ax.set_title(f"{CANONICAL_MODEL_DISPLAY_NAME} multi-metric radar comparison")
        ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.13), frameon=False)
        _save(fig, fig_dir, sdgcf_figure_name("top_model_radar"))

