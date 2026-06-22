"""Run or visualize standalone missing/noisy modality robustness comparisons."""
from __future__ import annotations

import argparse
from pathlib import Path

from model_naming import CANONICAL_MODEL_DISPLAY_NAME
from robustness_visualize import generate_robustness_figures
from run_all_experiments import (
    DEFAULT_MISSING_MODELS,
    make_cfg,
    parse_model_list,
    run_missing_for_selected_models,
)
from train import _select_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate standalone robustness-comparison results and figures.")
    parser.add_argument("--root_dir", default="./all_experiment_results")
    parser.add_argument("--models", default=",".join(DEFAULT_MISSING_MODELS))
    parser.add_argument("--data_path", default="./data/data_multimodal_eeg_eog_3ch")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0 or cpu")
    parser.add_argument("--data_fraction", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--noise_std", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--visualize_only", action="store_true", help="Reuse the existing robustness CSV without evaluation.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root_dir = Path(args.root_dir).resolve()
    summary_dir = root_dir / "summary_results"
    out_dir = Path(args.output_dir).resolve() if args.output_dir else summary_dir / "robustness_comparison"
    csv_path = out_dir / "robustness_results.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.visualize_only:
        # Reuse the shared experiment configuration builder without training.
        args.lr = None
        args.epochs = None
        args.embed_dim = None
        args.encoder_hidden = None
        args.dropout = None
        args.graph_alpha_init = None
        args.label_smoothing = None
        args.focal_blend = None
        args.auxiliary_loss_weight = None
        args.modality_dropout = None
        args.channel_noise = None
        args.no_amp = False
        args.no_class_weight = False
        args.device = str(_select_device(args.device))

        models = parse_model_list(args.models)
        results = run_missing_for_selected_models(
            models,
            args,
            root_dir,
            summary_dir,
            output_csv=csv_path,
            noise_std=args.noise_std,
        )
        if results.empty:
            raise RuntimeError("No robustness result was generated. Check whether the requested model checkpoints exist.")
    elif not csv_path.exists():
        raise FileNotFoundError(f"Robustness CSV not found: {csv_path}")

    generate_robustness_figures(csv_path, out_dir)
    print(f"{CANONICAL_MODEL_DISPLAY_NAME} robustness CSV: {csv_path}")
    print(f"{CANONICAL_MODEL_DISPLAY_NAME} robustness figures: {out_dir}")


if __name__ == "__main__":
    main()
