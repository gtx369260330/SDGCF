@echo off
python generate_robustness_cli.py --root_dir ./all_experiment_results --models simple_concat,concat_transformer,multimodal_concat,sdgcf --device auto
