@echo off
chcp 65001 >nul
REM SDGCF experiment on 30%% of each split using CPU.
python run_all_experiments.py --data_path ./data/data_multimodal_eeg_eog_3ch --root_dir ./sdgcf_30pct_results --models simple_concat,concat_transformer,multimodal_concat,sdgcf --epochs 30 --batch_size 128 --data_fraction 0.3 --device cpu --num_workers 0 --no_amp
pause
