#!/usr/bin/env bash
set -e
# One-click full experiment pipeline for Linux/WSL.
DATA_PATH="./data/data_multimodal_eeg_eog_3ch"
ROOT_DIR="./all_experiment_results"
# DEVICE can be auto / cpu / cuda / cuda:0.
DEVICE="auto"

python run_all_experiments.py --data_path "$DATA_PATH" --root_dir "$ROOT_DIR" --epochs 30 --batch_size 128 --device "$DEVICE" --run_missing
python generate_figures_cli.py --root_dir "$ROOT_DIR" --model_figures_only
python generate_figures_cli.py --root_dir "$ROOT_DIR" --summary_only
