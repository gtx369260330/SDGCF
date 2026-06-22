"""Standalone visualizations for missing/noisy modality robustness tests."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_naming import CANONICAL_MODEL_DISPLAY_NAME, display_model_name, sdgcf_figure_name


CONDITION_ORDER = [
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

CONDITION_LABELS = {
    "none": "Clean",
    "missing_fpz": "Missing Fpz-Cz",
    "missing_pz": "Missing Pz-Oz",
    "missing_eog": "Missing EOG",
    "missing_random_one": "Missing random one",
    "missing_two": "Missing two",
    "noise_fpz": "Noise Fpz-Cz",
    "noise_pz": "Noise Pz-Oz",
    "noise_eog": "Noise EOG",
}


plt.rcParams.update(
    {
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
    }
)


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def _ordered_conditions(values: Sequence[str]) -> list[str]:
    present = set(values)
    return [condition for condition in CONDITION_ORDER if condition in present]


def _prepare_results(results: pd.DataFrame) -> pd.DataFrame:
    required = {"model", "condition", "macro_f1", "macro_f1_drop_percent"}
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Robustness CSV is missing columns: {sorted(missing)}")
    results = results.copy()
    results["condition"] = results["condition"].astype(str)
    if results.empty:
        raise ValueError("Robustness CSV is empty.")
    return results


def plot_macro_f1_lines(results: pd.DataFrame, out_dir: Path) -> None:
    """Compare absolute Macro-F1 under clean, missing and noisy conditions."""
    conditions = _ordered_conditions(results["condition"].tolist())
    x = np.arange(len(conditions))
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    for model, group in results.groupby("model", sort=False):
        values = group.set_index("condition").reindex(conditions)["macro_f1"].to_numpy(dtype=float)
        ax.plot(x, values, marker="o", linewidth=1.8, markersize=5, label=display_model_name(model))
    ax.set_xticks(x, [CONDITION_LABELS.get(condition, condition) for condition in conditions], rotation=30, ha="right")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{CANONICAL_MODEL_DISPLAY_NAME} robustness under missing and noisy modalities")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    _save(fig, out_dir, sdgcf_figure_name("robustness_macro_f1_comparison"))


def plot_drop_lines(results: pd.DataFrame, out_dir: Path) -> None:
    """Compare relative Macro-F1 degradation against each model's clean score."""
    corrupted = results[results["condition"] != "none"].copy()
    conditions = _ordered_conditions(corrupted["condition"].tolist())
    x = np.arange(len(conditions))
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    for model, group in corrupted.groupby("model", sort=False):
        values = group.set_index("condition").reindex(conditions)["macro_f1_drop_percent"].to_numpy(dtype=float)
        ax.plot(x, values, marker="o", linewidth=1.8, markersize=5, label=display_model_name(model))
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(x, [CONDITION_LABELS.get(condition, condition) for condition in conditions], rotation=30, ha="right")
    ax.set_ylabel("Macro-F1 drop (%)")
    ax.set_title(f"{CANONICAL_MODEL_DISPLAY_NAME} relative performance drop under modality corruption")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    _save(fig, out_dir, sdgcf_figure_name("robustness_macro_f1_drop_comparison"))


def _plot_heatmap(
    matrix: pd.DataFrame,
    out_dir: Path,
    name: str,
    title: str,
    *,
    cmap: str,
    value_format: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(1.55 * len(matrix.columns) + 3.5, 0.52 * len(matrix.index) + 2.7))
    im = ax.imshow(matrix.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(matrix.index)), [CONDITION_LABELS.get(value, value) for value in matrix.index])
    ax.set_title(title)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iat[row, column]
            text = "NA" if np.isnan(value) else value_format.format(value)
            ax.text(column, row, text, ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    _save(fig, out_dir, name)


def plot_heatmaps(results: pd.DataFrame, out_dir: Path) -> None:
    """Render absolute and relative robustness heatmaps."""
    conditions = _ordered_conditions(results["condition"].tolist())
    macro_f1 = results.pivot_table(index="condition", columns="model", values="macro_f1", aggfunc="mean").reindex(conditions)
    macro_f1.columns = [display_model_name(model) for model in macro_f1.columns]
    _plot_heatmap(
        macro_f1,
        out_dir,
        sdgcf_figure_name("robustness_macro_f1_heatmap"),
        f"{CANONICAL_MODEL_DISPLAY_NAME} Macro-F1 under missing and noisy modalities",
        cmap="magma",
        value_format="{:.3f}",
        vmin=0,
        vmax=1,
    )

    corrupted = results[results["condition"] != "none"]
    corrupted_conditions = _ordered_conditions(corrupted["condition"].tolist())
    drop = corrupted.pivot_table(
        index="condition",
        columns="model",
        values="macro_f1_drop_percent",
        aggfunc="mean",
    ).reindex(corrupted_conditions)
    drop.columns = [display_model_name(model) for model in drop.columns]
    _plot_heatmap(
        drop,
        out_dir,
        sdgcf_figure_name("robustness_macro_f1_drop_heatmap"),
        f"{CANONICAL_MODEL_DISPLAY_NAME} relative Macro-F1 drop under modality corruption",
        cmap="Reds",
        value_format="{:.1f}%",
    )


def generate_robustness_figures(csv_path: str | Path, out_dir: str | Path) -> Path:
    """Generate standalone robustness-comparison figures from one CSV file."""
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    results = _prepare_results(pd.read_csv(csv_path))
    plot_macro_f1_lines(results, out_dir)
    plot_drop_lines(results, out_dir)
    plot_heatmaps(results, out_dir)
    return out_dir
