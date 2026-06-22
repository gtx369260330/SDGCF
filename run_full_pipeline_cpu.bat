@echo off
chcp 65001 >nul
REM CPU-safe version. Use this when CUDA reports: no kernel image is available for execution on the device.
set DATA_PATH=.\data\data_multimodal_eeg_eog_3ch
set ROOT_DIR=.\all_experiment_results_cpu

python run_all_experiments.py --data_path "%DATA_PATH%" --root_dir "%ROOT_DIR%" --epochs 30 --batch_size 128 --device cpu --run_missing --no_amp
python generate_figures_cli.py --root_dir "%ROOT_DIR%" --model_figures_only
python generate_figures_cli.py --root_dir "%ROOT_DIR%" --summary_only

pause
