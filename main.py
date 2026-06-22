"""Command-line entry for SDGCF EEG/EOG sleep staging."""
from __future__ import annotations

import argparse

from config import Config, apply_comparison_model_training_preset
from evaluate import run_ablation, run_missing_modality, test_model
from train import train_model


MODEL_CHOICES = [
    "sdgcf",
    "sdgcf_fixed_graph",
    "multimodal_concat",
    "xgboost",
    "random_forest",
    "svm_linear",
    "logistic_regression",
    "simple_concat",
    "concat_transformer",
    "single_fpz",
    "single_pz",
    "single_eog",
]


def parse_args():
    parser = argparse.ArgumentParser(description="SDGCF for three-channel EEG/EOG sleep staging")
    parser.add_argument("--model", type=str, default="sdgcf", choices=MODEL_CHOICES)
    parser.add_argument("--mode", type=str, default="ablation", choices=["train", "test", "ablation", "missing_test", "all"])
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None, help="auto, cuda, cuda:0 or cpu")
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
    return parser.parse_args()


def apply_overrides(cfg: Config, args):
    if args.data_path is not None:
        cfg.DATA_PATH = args.data_path
    if args.save_dir is not None:
        cfg.SAVE_DIR = args.save_dir
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
    cfg.make_dirs()
    return cfg


def main():
    args = parse_args()
    cfg = Config()
    apply_comparison_model_training_preset(cfg, args.model)
    cfg = apply_overrides(cfg, args)
    if args.mode == "train":
        train_model(cfg, args.model, device_str=args.device)
    elif args.mode == "test":
        test_model(cfg, args.model, device_str=args.device)
    elif args.mode == "ablation":
        run_ablation(cfg, device_str=args.device, train_if_missing=True)
    elif args.mode == "missing_test":
        run_missing_modality(cfg, device_str=args.device)
    elif args.mode == "all":
        train_model(cfg, args.model, device_str=args.device)
        test_model(cfg, args.model, device_str=args.device)
        run_ablation(cfg, device_str=args.device, train_if_missing=True)
        run_missing_modality(cfg, device_str=args.device)


if __name__ == "__main__":
    main()
