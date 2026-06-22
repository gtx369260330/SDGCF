@echo off
chcp 65001 >nul
REM Quick smoke test on CPU with 1 epoch and the proposed model.
set DATA_PATH=.\data\data_multimodal_eeg_eog_3ch
python run_all_experiments.py --data_path "%DATA_PATH%" --root_dir ".\debug_results" --models sdgcf --epochs 1 --batch_size 8 --data_fraction 0.01 --device cpu --no_amp
pause
