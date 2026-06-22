@echo off
chcp 65001 >nul
cd /d "%~dp0"

set ROOT_DIR=.\all_experiment_results

echo [1/3] Checking experiment result folder: %ROOT_DIR%
if not exist "%ROOT_DIR%" (
    echo [ERROR] Cannot find %ROOT_DIR%.
    exit /b 1
)

echo [2/3] Generating per-model figures...
python generate_figures_cli.py --root_dir "%ROOT_DIR%" --model_figures_only
if errorlevel 1 exit /b 1

echo [3/3] Generating summary figures...
python generate_figures_cli.py --root_dir "%ROOT_DIR%" --summary_only
if errorlevel 1 exit /b 1

echo [DONE] Per-model figures: %ROOT_DIR%\model_name_result\figures\paper_figures
echo [DONE] Summary figures:   %ROOT_DIR%\summary_results\figures
pause
