@echo off
chcp 65001 >nul
REM One-click full experiment pipeline for Windows.
REM Modify DATA_PATH if your data folder is different.
set DATA_PATH=.\data\data_multimodal_eeg_eog_3ch
set ROOT_DIR=.\all_experiment_results
REM DEVICE can be auto / cpu / cuda / cuda:0.
REM auto will use GPU only when PyTorch CUDA kernels are compatible with your graphics card; otherwise it falls back to CPU.
set DEVICE=auto

REM Stage 1: train and evaluate all models. Each model is saved as model_name_result.
python run_all_experiments.py --data_path "%DATA_PATH%" --root_dir "%ROOT_DIR%" --epochs 30 --batch_size 128 --device %DEVICE% --run_missing

REM Stage 2: generate paper-ready figures from saved metrics and predictions.
python generate_figures_cli.py --root_dir "%ROOT_DIR%" --model_figures_only
python generate_figures_cli.py --root_dir "%ROOT_DIR%" --summary_only

pause
