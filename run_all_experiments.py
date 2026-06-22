"""Sequential experiment runner for SDGCF and its comparison models."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import torch

from config import Config, apply_comparison_model_training_preset, get_comparison_model_training_preset
from evaluate import test_model
from model_naming import CANONICAL_MODEL_DISPLAY_NAME, CANONICAL_MODEL_KEY, display_model_name
from robustness_visualize import generate_robustness_figures
from train import _select_device, train_model
from utils import get_checkpoint_path, save_json
from classical_baselines import get_classical_training_policy, is_classical_baseline


ALL_MODELS = [
    "single_fpz",
    "single_pz",
    "single_eog",
    "sdgcf",
    "sdgcf_fixed_graph",
    "simple_concat",
    "xgboost",
    "random_forest",
    "svm_linear",
    "logistic_regression",
    "concat_transformer",
    "multimodal_concat",

]
DEFAULT_MISSING_MODELS = ["simple_concat", "concat_transformer", "multimodal_concat", "sdgcf"]
# DEFAULT_MISSING_MODELS = ["sdgcf"]
MISSING_CONDITIONS = [
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


class Tee:
    """Write stdout and stderr to both console and a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextmanager
def tee_stdout(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    original_out, original_err = sys.stdout, sys.stderr
    with open(log_path, "a", encoding="utf-8") as log_file:
        sys.stdout = Tee(original_out, log_file)
        sys.stderr = Tee(original_err, log_file)
        try:
            yield
        finally:
            sys.stdout = original_out
            sys.stderr = original_err


def parse_model_list(text: str | None) -> List[str]:
    if not text or text.strip().lower() in ["all", "default"]:
        return ALL_MODELS.copy()
    models = [model.strip() for model in text.split(",") if model.strip()]
    unknown = [model for model in models if model not in ALL_MODELS]
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}. Available: {ALL_MODELS}")
    return models


def make_cfg(args, save_dir: Path, model_name: str | None = None) -> Config:
    cfg = Config()
    if model_name is not None:
        apply_comparison_model_training_preset(cfg, model_name)
    cfg.SAVE_DIR = str(save_dir)
    if args.data_path is not None:
        cfg.DATA_PATH = args.data_path
    if args.epochs is not None:
        cfg.EPOCHS = args.epochs
    if args.batch_size is not None:
        cfg.BATCH_SIZE = args.batch_size
    if args.lr is not None:
        cfg.LR = args.lr
    if args.data_fraction is not None:
        cfg.DATA_FRACTION = args.data_fraction
    if args.embed_dim is not None:
        cfg.EMBED_DIM = int(args.embed_dim)
    if args.encoder_hidden is not None:
        cfg.ENCODER_HIDDEN = int(args.encoder_hidden)
    if args.dropout is not None:
        cfg.DROPOUT = float(args.dropout)
    if args.graph_alpha_init is not None:
        cfg.GRAPH_ALPHA_INIT = float(args.graph_alpha_init)
    if args.label_smoothing is not None:
        cfg.LABEL_SMOOTHING = float(args.label_smoothing)
    if args.focal_blend is not None:
        cfg.FOCAL_BLEND = float(args.focal_blend)
    if args.auxiliary_loss_weight is not None:
        cfg.AUXILIARY_LOSS_WEIGHT = float(args.auxiliary_loss_weight)
    if args.modality_dropout is not None:
        cfg.MODALITY_DROPOUT_PROB = float(args.modality_dropout)
    if args.channel_noise is not None:
        cfg.CHANNEL_NOISE_STD = float(args.channel_noise)
    if args.num_workers is not None:
        cfg.NUM_WORKERS = args.num_workers
    if args.seed is not None:
        cfg.RANDOM_SEED = args.seed
    if args.no_amp:
        cfg.USE_AMP = False
    if args.no_class_weight:
        cfg.USE_CLASS_WEIGHT = False
    cfg.make_dirs()
    return cfg


def metrics_to_row(model_name: str, result_dir: Path, metrics: Dict[str, Any], status: str = "success", error: str = ""):
    return {
        "model": model_name,
        "result_dir": str(result_dir),
        "status": status,
        "error": error,
        "accuracy": metrics.get("accuracy", np.nan),
        "balanced_accuracy": metrics.get("balanced_accuracy", np.nan),
        "macro_precision": metrics.get("macro_precision", np.nan),
        "macro_recall": metrics.get("macro_recall", np.nan),
        "macro_f1": metrics.get("macro_f1", np.nan),
        "weighted_f1": metrics.get("weighted_f1", np.nan),
        "cohen_kappa": metrics.get("cohen_kappa", np.nan),
        "mcc": metrics.get("mcc", np.nan),
        "macro_auroc_ovr": metrics.get("macro_auroc_ovr", np.nan),
        "macro_auprc_ovr": metrics.get("macro_auprc_ovr", np.nan),
    }


def training_policy_for_model(model_name: str) -> str:
    if model_name == "xgboost":
        return "xgboost_handcrafted_channel_features"
    if is_classical_baseline(model_name):
        return get_classical_training_policy(model_name)
    return "hybrid_focal_ce_with_graph_regularization"


def read_saved_metrics(model_name: str, result_dir: Path) -> Dict[str, Any]:
    path = result_dir / "metrics" / f"test_metrics_{model_name}_none.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        import json

        return json.load(f)


def write_summary_tables(rows: list[dict[str, Any]], summary_dir: Path) -> pd.DataFrame:
    metrics = pd.DataFrame(rows)
    metrics.to_csv(summary_dir / "all_model_metrics.csv", index=False)
    successful = metrics[metrics["status"] == "success"].copy()
    if not successful.empty:
        successful.sort_values("macro_f1", ascending=False).to_csv(
            summary_dir / "model_ranking_by_macro_f1.csv",
            index=False,
        )
        ablation_mask = successful["model"].isin(["multimodal_concat", "sdgcf_fixed_graph", "sdgcf"])
        successful[ablation_mask].to_csv(
            summary_dir / "ablation_results.csv",
            index=False,
        )
        successful[~ablation_mask].to_csv(
            summary_dir / "baseline_comparison_results.csv",
            index=False,
        )
    return metrics


def copy_compact_outputs(result_dir: Path, summary_dir: Path, model_name: str) -> None:
    compact_dir = summary_dir / "compact_outputs" / model_name
    compact_dir.mkdir(parents=True, exist_ok=True)
    for relative_path in [
        f"metrics/test_metrics_{model_name}_none.json",
        f"metrics/classification_report_{model_name}_none.csv",
        f"metrics/predictions_{model_name}_none.csv",
        f"logs/training_history_{model_name}.csv",
    ]:
        source = result_dir / relative_path
        if source.exists():
            shutil.copy2(source, compact_dir / source.name)


def run_figure_subprocess(root_dir: Path, model_name: str | None = None, summary_only: bool = False) -> None:
    command = [sys.executable, str(Path(__file__).resolve().parent / "generate_figures_cli.py"), "--root_dir", str(root_dir)]
    if model_name is not None:
        command.extend(["--model", model_name])
    if summary_only:
        command.append("--summary_only")
    subprocess.run(command, check=True)


def run_one_model(model_name: str, args, root_dir: Path):
    result_dir = root_dir / f"{model_name}_result"
    cfg = make_cfg(args, result_dir, model_name)
    save_json(
        {
            "model_name": model_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "data_path": cfg.DATA_PATH,
            "save_dir": cfg.SAVE_DIR,
            "epochs": cfg.EPOCHS,
            "batch_size": cfg.BATCH_SIZE,
            "lr": cfg.LR,
            "data_fraction": cfg.DATA_FRACTION,
            "training_policy": training_policy_for_model(model_name),
            "embed_dim": cfg.EMBED_DIM,
            "encoder_hidden": cfg.ENCODER_HIDDEN,
            "dropout": cfg.DROPOUT,
            "graph_alpha_init": cfg.GRAPH_ALPHA_INIT,
            "label_smoothing": cfg.LABEL_SMOOTHING,
            "focal_blend": cfg.FOCAL_BLEND,
            "auxiliary_loss_weight": cfg.AUXILIARY_LOSS_WEIGHT,
            "modality_dropout_prob": cfg.MODALITY_DROPOUT_PROB,
            "channel_noise_std": cfg.CHANNEL_NOISE_STD,
            "seed": cfg.RANDOM_SEED,
            "device_arg": args.device,
            "model_specific_training_preset": get_comparison_model_training_preset(model_name),
        },
        result_dir / "experiment_config.json",
    )
    checkpoint = get_checkpoint_path(cfg.SAVE_DIR, model_name)
    if args.skip_existing and checkpoint.exists():
        print(f"[Skip train] Existing checkpoint: {checkpoint}")
    else:
        train_model(cfg, model_name=model_name, device_str=args.device)
    metrics = test_model(cfg, model_name=model_name, device_str=args.device)
    if args.with_figures:
        run_figure_subprocess(root_dir, model_name=model_name)
    return metrics_to_row(model_name, result_dir, metrics)


def run_missing_for_selected_models(
    models: Iterable[str],
    args,
    root_dir: Path,
    summary_dir: Path,
    *,
    output_csv: Path | None = None,
    noise_std: float | None = None,
) -> pd.DataFrame:
    rows = []
    for model_name in models:
        result_dir = root_dir / f"{model_name}_result"
        cfg = make_cfg(args, result_dir, model_name)
        if noise_std is not None:
            cfg.NOISE_STD = float(noise_std)
        if not get_checkpoint_path(cfg.SAVE_DIR, model_name).exists():
            print(f"[Missing modality] Skip {model_name}; checkpoint not found.")
            continue
        clean_macro_f1 = None
        for condition in MISSING_CONDITIONS:
            metrics = test_model(
                cfg,
                model_name=model_name,
                device_str=args.device,
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
                    "result_dir": str(result_dir),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result.to_csv(summary_dir / "missing_modality_results.csv", index=False)
        if output_csv is not None:
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(output_csv, index=False)
    return result


def run_all(args) -> Path:
    root_dir = Path(args.root_dir).resolve()
    summary_dir = root_dir / "summary_results"
    summary_dir.mkdir(parents=True, exist_ok=True)
    models = parse_model_list(args.models)
    requested_device = args.device or "auto"
    args.device = str(_select_device(args.device))

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root_dir": str(root_dir),
        "data_path": args.data_path,
        "models": models,
        "data_fraction": args.data_fraction,
        "proposed_model": CANONICAL_MODEL_KEY,
        "proposed_model_display": CANONICAL_MODEL_DISPLAY_NAME,
        "device_arg": requested_device,
        "selected_device": args.device,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "status": "running",
        "experiments": [],
    }
    save_json(manifest, summary_dir / "experiment_manifest.json")

    rows = []
    with tee_stdout(summary_dir / "run_all_console.log"):
        print(f"========== {CANONICAL_MODEL_DISPLAY_NAME} automatic experiment runner ==========")
        print(f"Models: {models}")
        print(f"Device argument: {requested_device} | selected: {args.device}")
        for index, model_name in enumerate(models, start=1):
            result_dir = root_dir / f"{model_name}_result"
            try:
                print(f"\n[{index}/{len(models)}] Start model: {model_name} ({display_model_name(model_name)})")
                row = run_one_model(model_name, args, root_dir)
                manifest["experiments"].append({"model": model_name, "status": "success", "result_dir": str(result_dir)})
                copy_compact_outputs(result_dir, summary_dir, model_name)
            except Exception as exc:
                print(f"[ERROR] {model_name} failed: {exc}\n{traceback.format_exc()}")
                row = metrics_to_row(model_name, result_dir, {}, status="failed", error=str(exc))
                manifest["experiments"].append({"model": model_name, "status": "failed", "error": str(exc)})
                if args.stop_on_error:
                    rows.append(row)
                    break
            rows.append(row)
            pd.DataFrame(rows).to_csv(summary_dir / "all_model_metrics.csv", index=False)
            save_json(manifest, summary_dir / "experiment_manifest.json")

        metrics = pd.DataFrame(rows)
        write_summary_tables(rows, summary_dir)
        if args.run_missing:
            missing_models = parse_model_list(args.missing_models) if args.missing_models else DEFAULT_MISSING_MODELS
            robustness_dir = summary_dir / "robustness_comparison"
            robustness_csv = robustness_dir / "robustness_results.csv"
            results = run_missing_for_selected_models(
                missing_models,
                args,
                root_dir,
                summary_dir,
                output_csv=robustness_csv,
            )
            if not results.empty:
                generate_robustness_figures(robustness_csv, robustness_dir)
        if args.with_figures:
            run_figure_subprocess(root_dir, summary_only=True)

    manifest["status"] = "finished"
    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    save_json(manifest, summary_dir / "experiment_manifest.json")
    return summary_dir


def refresh_summary_from_existing(args) -> Path:
    root_dir = Path(args.root_dir).resolve()
    summary_dir = root_dir / "summary_results"
    summary_dir.mkdir(parents=True, exist_ok=True)
    models = parse_model_list(args.models)
    rows = []
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root_dir": str(root_dir),
        "models": models,
        "proposed_model": CANONICAL_MODEL_KEY,
        "proposed_model_display": CANONICAL_MODEL_DISPLAY_NAME,
        "status": "refreshed_from_existing_metrics",
        "experiments": [],
    }
    for model_name in models:
        result_dir = root_dir / f"{model_name}_result"
        try:
            metrics = read_saved_metrics(model_name, result_dir)
            rows.append(metrics_to_row(model_name, result_dir, metrics))
            copy_compact_outputs(result_dir, summary_dir, model_name)
            manifest["experiments"].append({"model": model_name, "status": "success", "result_dir": str(result_dir)})
        except Exception as exc:
            rows.append(metrics_to_row(model_name, result_dir, {}, status="missing", error=str(exc)))
            manifest["experiments"].append({"model": model_name, "status": "missing", "error": str(exc)})
    write_summary_tables(rows, summary_dir)
    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    save_json(manifest, summary_dir / "experiment_manifest.json")
    if args.with_figures:
        run_figure_subprocess(root_dir, summary_only=True)
    return summary_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SDGCF comparison experiments sequentially.")
    parser.add_argument("--data_path", type=str, default="./data/data_multimodal_eeg_eog_3ch")
    parser.add_argument("--root_dir", type=str, default="./all_experiment_results")
    parser.add_argument("--models", type=str, default="all")
    parser.add_argument("--missing_models", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--data_fraction", type=float, default=None)
    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--encoder_hidden", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--graph_alpha_init", type=float, default=None)
    parser.add_argument("--label_smoothing", type=float, default=None)
    parser.add_argument("--focal_blend", type=float, default=None)
    parser.add_argument("--auxiliary_loss_weight", type=float, default=None)
    parser.add_argument("--modality_dropout", type=float, default=None)
    parser.add_argument("--channel_noise", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="auto, cuda, cuda:0 or cpu")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--stop_on_error", action="store_true")
    parser.add_argument("--run_missing", action="store_true")
    parser.add_argument("--with_figures", action="store_true")
    parser.add_argument("--refresh_summary_only", action="store_true", help="Scan existing test metrics and regenerate summary CSV/figures without training or testing.")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_class_weight", action="store_true")
    return parser


def main():
    args = build_argparser().parse_args()
    if args.refresh_summary_only:
        summary_dir = refresh_summary_from_existing(args)
    else:
        summary_dir = run_all(args)
    print(f"\nAll done. Summary saved to: {summary_dir}")


if __name__ == "__main__":
    main()
