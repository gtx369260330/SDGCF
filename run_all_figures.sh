#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
ROOT_DIR="./all_experiment_results"

echo "[1/3] Checking experiment result folder: ${ROOT_DIR}"
test -d "$ROOT_DIR"

echo "[2/3] Generating per-model figures..."
python generate_figures_cli.py --root_dir "$ROOT_DIR" --model_figures_only

echo "[3/3] Generating summary figures..."
python generate_figures_cli.py --root_dir "$ROOT_DIR" --summary_only

echo "[DONE] Per-model figures: ${ROOT_DIR}/model_name_result/figures/paper_figures"
echo "[DONE] Summary figures:   ${ROOT_DIR}/summary_results/figures"
