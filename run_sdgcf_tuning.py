"""Run a compact SDGCF hyperparameter search without overwriting prior results."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import Config
from data_loader import create_dataloaders
from evaluate import evaluate_model, load_trained_model
from train import _select_device, train_model
from utils import get_checkpoint_path, save_json, set_seed


PRESETS = {
    "compact": {
        "EMBED_DIM": 128,
        "ENCODER_HIDDEN": 48,
        "DROPOUT": 0.20,
        "LR": 5e-4,
        "GRAPH_ALPHA_INIT": 0.06,
        "FOCAL_BLEND": 0.15,
        "AUXILIARY_LOSS_WEIGHT": 0.05,
    },
    "balanced": {
        "EMBED_DIM": 192,
        "ENCODER_HIDDEN": 64,
        "DROPOUT": 0.20,
        "LR": 5e-4,
        "GRAPH_ALPHA_INIT": 0.08,
        "FOCAL_BLEND": 0.20,
        "AUXILIARY_LOSS_WEIGHT": 0.08,
    },
    "expressive": {
        "EMBED_DIM": 256,
        "ENCODER_HIDDEN": 96,
        "DROPOUT": 0.25,
        "LR": 3e-4,
        "GRAPH_ALPHA_INIT": 0.10,
        "FOCAL_BLEND": 0.25,
        "AUXILIARY_LOSS_WEIGHT": 0.08,
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run compact SDGCF tuning presets.")
    parser.add_argument("--data_path", default="./data/data_multimodal_eeg_eog_3ch")
    parser.add_argument("--root_dir", default="./sdgcf_tuning_results")
    parser.add_argument("--presets", default="compact,balanced,expressive")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data_fraction", type=float, default=0.30)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip_existing", action="store_true")
    return parser


def make_config(args, name: str) -> Config:
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {sorted(PRESETS)}")
    cfg = Config()
    cfg.DATA_PATH = args.data_path
    cfg.SAVE_DIR = str(Path(args.root_dir).resolve() / name)
    cfg.EPOCHS = int(args.epochs)
    cfg.BATCH_SIZE = int(args.batch_size)
    cfg.DATA_FRACTION = float(args.data_fraction)
    for key, value in PRESETS[name].items():
        setattr(cfg, key, value)
    cfg.make_dirs()
    return cfg


def validate_preset(cfg: Config, device_str: str):
    """Evaluate one tuning preset on validation data only."""
    device = _select_device(device_str)
    set_seed(cfg.RANDOM_SEED)
    _, val_loader, _, _, _ = create_dataloaders(cfg)
    model = load_trained_model(cfg, "sdgcf", device)
    return evaluate_model(model, val_loader, device, cfg.CLASS_NAMES, collect_outputs=False)


def main() -> None:
    args = build_parser().parse_args()
    preset_names = [name.strip() for name in args.presets.split(",") if name.strip()]
    rows = []
    for preset_name in preset_names:
        cfg = make_config(args, preset_name)
        save_json(
            {
                "preset": preset_name,
                "model_name": "sdgcf",
                "data_fraction": cfg.DATA_FRACTION,
                "epochs": cfg.EPOCHS,
                "batch_size": cfg.BATCH_SIZE,
                "parameters": PRESETS[preset_name],
            },
            Path(cfg.SAVE_DIR) / "preset_config.json",
        )
        checkpoint = get_checkpoint_path(cfg.SAVE_DIR, "sdgcf")
        if not (args.skip_existing and checkpoint.exists()):
            train_model(cfg, "sdgcf", device_str=args.device)
        metrics = validate_preset(cfg, device_str=args.device)
        rows.append(
            {
                "preset": preset_name,
                "val_accuracy": metrics["accuracy"],
                "val_balanced_accuracy": metrics["balanced_accuracy"],
                "val_macro_f1": metrics["macro_f1"],
                "val_weighted_f1": metrics["weighted_f1"],
                "val_macro_auroc_ovr": metrics.get("macro_auroc_ovr"),
                "val_macro_auprc_ovr": metrics.get("macro_auprc_ovr"),
                "result_dir": cfg.SAVE_DIR,
            }
        )
    result = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
    output = Path(args.root_dir).resolve() / "tuning_results.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result)
    print(f"Tuning summary saved to: {output}")


if __name__ == "__main__":
    main()
