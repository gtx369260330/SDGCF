"""CLI wrapper for figure generation.

Use this after experiments if you want to regenerate all paper figures:
    python generate_figures_cli.py --root_dir all_experiment_results
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from advanced_visualize import generate_model_figures, generate_summary_figures
from model_naming import display_model_name

CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]
SUMMARY_METRIC_KEYS = [
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "cohen_kappa",
    "mcc",
    "macro_auroc_ovr",
    "macro_auprc_ovr",
]
SUMMARY_ABLATION_MODELS = {"multimodal_concat", "sdgcf_fixed_graph", "sdgcf"}


def _metrics_to_summary_row(model: str, result_dir: Path, metrics: dict) -> dict:
    row = {
        "model": model,
        "result_dir": str(result_dir),
        "status": "success",
        "error": "",
    }
    for key in SUMMARY_METRIC_KEYS:
        row[key] = metrics.get(key)
    return row


def _write_summary_tables(summary_dir: Path, rows: list[dict]) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(summary_dir / "all_model_metrics.csv", index=False)
    if df.empty:
        return

    successful = df[df["status"].astype(str).str.lower() == "success"].copy()
    successful = successful.dropna(subset=["macro_f1"])
    if successful.empty:
        return

    successful.sort_values("macro_f1", ascending=False).to_csv(
        summary_dir / "model_ranking_by_macro_f1.csv",
        index=False,
    )
    ablation_mask = successful["model"].isin(SUMMARY_ABLATION_MODELS)
    successful[ablation_mask].to_csv(summary_dir / "ablation_results.csv", index=False)
    successful[~ablation_mask].to_csv(summary_dir / "baseline_comparison_results.csv", index=False)


def refresh_summary_from_existing_metrics(root: Path) -> int:
    """Rebuild summary CSV files from all existing *_result metric JSON files."""
    rows = []
    for folder in sorted(root.glob("*_result")):
        if not folder.is_dir():
            continue
        model = folder.name[:-len("_result")]
        metrics_path = folder / "metrics" / f"test_metrics_{model}_none.json"
        if not metrics_path.exists():
            candidates = sorted((folder / "metrics").glob("test_metrics_*_none.json"))
            if not candidates:
                continue
            metrics_path = candidates[0]
        with open(metrics_path, "r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        rows.append(_metrics_to_summary_row(model, folder, metrics))

    if rows:
        _write_summary_tables(root / "summary_results", rows)
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="./all_experiment_results")
    parser.add_argument("--model", type=str, default=None, help="Generate figures for one model result folder only.")
    parser.add_argument("--summary_only", action="store_true")
    parser.add_argument("--model_figures_only", action="store_true", help="Generate only per-model figures and skip summary figures.")
    parser.add_argument("--chain_summary", action="store_true", help="After per-model figures, replace this process with a fresh summary-only figure process.")
    args = parser.parse_args()
    root = Path(args.root_dir)
    if not args.summary_only:
        folders = []
        if args.model:
            folders = [root / f"{args.model}_result"]
        else:
            folders = sorted(root.glob("*_result"))
        for folder in folders:
            model = folder.name[:-len("_result")]
            metrics = folder / "metrics" / f"test_metrics_{model}_none.json"
            if metrics.exists():
                print(f"[Figures] {model} ({display_model_name(model)})", flush=True)
                generate_model_figures(folder, model, CLASS_NAMES)
            else:
                print(f"[Figures] skip {folder.name}; missing {metrics.name}", flush=True)
    summary = root / "summary_results"
    if args.chain_summary and args.model_figures_only and not args.summary_only:
        print("[Figures] restarting fresh process for summary figures", flush=True)
        os.execv(os.sys.executable, [os.sys.executable, str(Path(__file__).resolve()), "--root_dir", str(root), "--summary_only"])

    if (not args.model_figures_only) and (args.summary_only or args.model is None) and summary.exists():
        count = refresh_summary_from_existing_metrics(root)
        if count:
            print(f"[Figures] refreshed summary metrics from {count} model result folders", flush=True)
        print(f"[Figures] summary", flush=True)
        generate_summary_figures(summary, CLASS_NAMES)
    print("Figure generation finished.", flush=True)


if __name__ == "__main__":
    main()
