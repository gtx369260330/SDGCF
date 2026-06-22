# SDGCF EEG/EOG Sleep Staging

SDGCF means **Simple Dynamic Graph Concatenation Fusion**. It is a compact
three-channel EEG/EOG sleep-staging model designed around explicit and
interpretable modality interaction.

## Method

The input contains three synchronized channels:

```text
x = [EEG Fpz-Cz, EEG Pz-Oz, EOG horizontal]  # [B, 3, T]
```

Each channel is encoded independently as one modality node:

```text
u_i = MultiScaleTemporalEncoder_i(x_i)       # [B, D]
H = stack(u_1, u_2, u_3)                     # [B, 3, D]
```

Multi-head graph attention dynamically derives modality relations from node
content similarity:

```text
score_ij = Q_i K_j^T / sqrt(d)
A_ij = softmax_j(score_ij)
H_graph = GraphAttention(H, A)
```

The three graph-updated nodes are concatenated directly. There is no
reliability gate and no TCN branch:

```text
z = [h_graph_1 || h_graph_2 || h_graph_3]     # [B, 3D]
y_hat = Classifier(z)
```

## Models

```text
sdgcf              SDGCF, proposed simple dynamic graph fusion model
multimodal_concat  SDGCF w/o Dynamic Graph, ablation without graph interaction
simple_concat      Simple-Concatenation CNN baseline
concat_transformer Modality Transformer baseline
xgboost            XGBoost on handcrafted channel statistics/spectral features
random_forest      Random Forest on the same handcrafted channel features
svm_linear         Linear SVM on the same handcrafted channel features
logistic_regression Logistic Regression on the same handcrafted channel features
single_fpz         Single Fpz-Cz baseline
single_pz          Single Pz-Oz baseline
single_eog         Single EOG baseline
```

Internal CLI/checkpoint names stay lowercase, for example `sdgcf`. Figure
titles, legends and report display names use the canonical paper name `SDGCF`.
Cross-model summary and robustness figures are saved with the `sdgcf_` prefix.

## Quick Check

```bash
python main.py --model sdgcf --mode train --save_dir ./results_enhanced --device cpu --epochs 1 --batch_size 8 --data_fraction 0.01
python main.py --model sdgcf --mode test --save_dir ./results_enhanced --device cpu --batch_size 8 --data_fraction 0.01
```

## Comparison Experiments

```bash
python run_all_experiments.py --models all --epochs 30 --batch_size 128 --device auto
```

To add or refresh comparison figures without retraining already completed
models, first run only the missing comparison models, then rebuild the summary
from saved metrics:

```bash
python run_all_experiments.py --models random_forest,svm_linear,logistic_regression --root_dir ./all_experiment_results --skip_existing --with_figures --device cpu
python run_all_experiments.py --root_dir ./all_experiment_results --refresh_summary_only --with_figures
```

Windows launchers:

```text
run_quick_debug_cpu.bat
run_sdgcf_30pct_cpu.bat
run_sdgcf_full_auto.bat
run_sdgcf_tuning_30pct_cpu.bat
run_full_pipeline.bat
```

## Tuning

Run three SDGCF presets on 30% of each split:

```bash
python run_sdgcf_tuning.py --epochs 30 --batch_size 128 --data_fraction 0.3 --device auto
```

The presets write to separate folders under `sdgcf_tuning_results/`. Select
the best preset by validation Macro-F1, then train the selected configuration
on the full dataset and evaluate the test set once.

## Outputs

Each result folder contains:

```text
checkpoints/   best model checkpoint
logs/          training history and summary
metrics/       test metrics, predictions and graph attention matrices
figures/       generated visualizations
```

The simplified SDGCF architecture requires fresh checkpoints. Legacy
pre-SDGCF result folders remain useful for traceability but are not compatible
with the current model definition.

## Robustness Comparison

Run missing-modality and noisy-modality tests for selected trained models:

```bash
python generate_robustness_cli.py --root_dir ./all_experiment_results --models simple_concat,concat_transformer,multimodal_concat,sdgcf --device auto
```

The standalone comparison CSV and figures are saved under:

```text
all_experiment_results/summary_results/robustness_comparison/
```

Use `--visualize_only` to regenerate figures from the existing CSV without
rerunning model evaluation.

On Windows, the same comparison can be started with:

```bat
run_robustness_comparison.bat
```

## Comparison-model Presets

`concat_transformer` and `multimodal_concat` use architecture-aware training
presets from `config.py`. The presets only adjust optimization and
regularization parameters; they do not reduce the model dimensions. Explicit
CLI arguments such as `--lr` and `--dropout` still override the presets.
