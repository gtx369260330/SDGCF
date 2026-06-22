@echo off
chcp 65001 >nul
REM Full SDGCF experiment. If your PyTorch CUDA build is incompatible, use --device cpu.
python run_all_experiments.py --data_path ./data/data_multimodal_eeg_eog_3ch --root_dir ./sdgcf_full_results --models simple_concat,concat_transformer,multimodal_concat,sdgcf --epochs 80 --batch_size 128 --data_fraction 1.0 --device auto --num_workers 0
pause
