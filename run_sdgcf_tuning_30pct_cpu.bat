@echo off
chcp 65001 >nul
REM Compare three SDGCF presets on 30%% of each split.
python run_sdgcf_tuning.py ^
  --data_path ./data/data_multimodal_eeg_eog_3ch ^
  --root_dir ./sdgcf_tuning_30pct_results ^
  --epochs 30 ^
  --batch_size 128 ^
  --data_fraction 0.3 ^
  --device cpu
pause
