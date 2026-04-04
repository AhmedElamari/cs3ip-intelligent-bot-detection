# CS3IP: Intelligent Bot Detection - Interpretable ML

## Overview
This project implements an interpretable bot detection pipeline for social media data with an emphasis on explainability, reproducibility, and clean experimentation. It supports multiple supervised models, feature engineering grounded in domain knowledge, and benchmarking with XAI tooling (SHAP, LIME, feature importance).

## Key Capabilities
- Structured data loading for TwiBot-20 JSON
- Feature engineering with leakage-aware account age computation
- Train/validation/test splits with reproducible random state
- Multiple supervised models with a common interface (LR, RF, XGBoost, SVM, DT, TabNet)
- **TabNet deep learning model** — sequential attention with intrinsic interpretability (feature masks)
- Balanced hyperparameter optimisation for TabNet via Optuna (40-60 trial balanced search)
- Benchmarking with comparison tables, plots, and statistically grounded evaluation
- Bootstrap 95% confidence intervals per model metric
- Pairwise model significance: paired bootstrap delta test + McNemar exact test (Holm-Bonferroni corrected)
- Explainability using SHAP (TreeExplainer for XGBoost), LIME, and feature importance analysis
- Optional cost-aware adversarial robustness audit with flip-rate, confidence-drop, SHAP stability, pivot tracking, and feature resilience scoring

## Project Structure
```
cs3ip-intelligent-bot-detection/
|-- DataLoader.py                 # TwiBot-20 JSON loader and flattening
|-- FeatureEngineering.py         # Feature extraction
|-- Preprocessing.py              # Cleaning and split helpers
|-- main.py                       # Single-model pipeline
|-- run_benchmark.py              # Multi-model benchmark CLI + XAI
|-- config/
|   |-- config.py                 # Config management
|-- models/                       # Model implementations
|-- benchmarking/                 # Benchmark runner and metrics
|-- explainability/               # SHAP/LIME/feature importance tools
|-- tests/                        # Smoke tests
|-- results/                      # Generated outputs (gitignored)
```

## Installation
Recommended:
```bash
pip install -r requirements.txt
```

Minimum:
```bash
pip install pandas numpy>=1.23.5,<2.0 scikit-learn>=1.5.0 xgboost>=2.0.0
```

Optional XAI tooling:
```bash
pip install shap lime matplotlib seaborn
```

Optional deep-learning profile (TabNet + HPO):
```bash
# CPU
pip install -r requirements-dl.txt
# GPU (adjust CUDA version to match your driver)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-dl.txt
```

> **Migration note** — the `gradient_boosting` model key was replaced by `xgboost` in the
> config, CLI, and model registry.  Any existing config files or `--models gradient_boosting`
> invocations must be updated to use `xgboost`.

## Usage

## Quickstart
Example files assumed:
- `data/train.json`, `data/dev.json`, `data/test.json` (embedded labels)
- Download the TwiBot-20 dataset from the [official repository](https://github.com/BunsenFeng/TwiBot-20).

Run a single model:
```bash
python main.py --model random_forest
# With TabNet (requires: pip install -r requirements-dl.txt)
python main.py --model tabnet
```
Expected output (console):
- Training/validation/test sizes
- Validation + test metrics
- Confusion matrix and classification report

Run a benchmark with explainability:
```bash
python run_benchmark.py --explain --save-plots
```
Run the optional robustness audit:
```bash
python run_benchmark.py --models random_forest xgboost --explain --robustness-analysis --save-plots
```
Expected output (filesystem):
- `results/benchmark_YYYYMMDD_HHMMSS/model_comparison.csv`
- `results/benchmark_YYYYMMDD_HHMMSS/benchmark_report.md`
- `results/benchmark_YYYYMMDD_HHMMSS/benchmark_report.txt` (compatibility mirror)
- `results/benchmark_YYYYMMDD_HHMMSS/performance_comparison.png`
- `results/benchmark_YYYYMMDD_HHMMSS/feature_importance.csv` — raw per-model feature importances
- `results/benchmark_YYYYMMDD_HHMMSS/feature_importance_comparison.csv`
- `results/benchmark_YYYYMMDD_HHMMSS/run_metadata.json` â€” runtime, dataset, package, git, and artifact provenance
- `results/benchmark_YYYYMMDD_HHMMSS/results.json` — structured benchmark export contract with ranking metadata
- `results/benchmark_YYYYMMDD_HHMMSS/metric_confidence_intervals.csv` — 95% bootstrap CIs per model/metric
- `results/benchmark_YYYYMMDD_HHMMSS/pairwise_significance.csv` — delta, CI, and p-values for every model pair

- `results/benchmark_YYYYMMDD_HHMMSS/robustness_summary.csv` — profile-level attack metrics
- `results/benchmark_YYYYMMDD_HHMMSS/feature_attack_results.csv` — per-feature attack metrics
- `results/benchmark_YYYYMMDD_HHMMSS/shap_rank_stability.csv` — SHAP stability rows when available
- `results/benchmark_YYYYMMDD_HHMMSS/feature_resilience_scores.csv` — FRS values when SHAP succeeds
- `results/benchmark_YYYYMMDD_HHMMSS/shap_pivot_features.csv` — features that gain or lose explanatory prominence
- `results/benchmark_YYYYMMDD_HHMMSS/robustness_report.json` — machine-readable robustness summary

### Single Model Pipeline
```bash
python main.py --model random_forest
```

Options:
- `--model`: `random_forest`, `logistic_regression`, `svm`, `tabnet` (tabnet requires the DL profile)
- `--smote`: enable SMOTE
- `--scale`: enable feature scaling
- `--features`: select top-k features

### Benchmarking and Explainability
```bash
python run_benchmark.py --explain --save-plots
```

Explainability runs when either `--explain` is passed or `explainability.enabled` is `true` in the loaded config.
`--save-plots` forces plot saving on from the CLI, but plots can also be saved through `output.save_plots` in config.

Options:
- `--config`: load YAML or JSON config
- `--models`: specify models to run (e.g. `logistic_regression random_forest xgboost tabnet`)
- `--smote` / `--scale`: override preprocessing settings
- `--robustness-analysis`: enable the optional adversarial robustness audit
- `--robustness-profiles`: override the default profiles (`cheap_only realistic_mixed`)
- `--robustness-max-shap-samples`: cap SHAP rows used during robustness analysis

Outputs are saved under `results/benchmark_YYYYMMDD_HHMMSS/`.

### TabNet Hyperparameter Optimisation
Run balanced Optuna HPO (40-60 trials) for TabNet before or after a benchmark:
```python
from benchmarking.tabnet_optuna import optimize_tabnet
result = optimize_tabnet(X_train, y_train, X_val, y_val, n_trials=50, output_path="results/hpo.json")
```
The best params are saved as a validated `HPOResultV1` JSON artifact and can be loaded back via `load_hpo_result("results/hpo.json")`.

## Configuration
Configuration is centralized in `config/config.py` and supports YAML/JSON. Use `create_default_config()` to generate a template file and adjust model parameters, preprocessing options, explainability settings, and the optional `robustness.*` controls.

## Data Notes
- The pipeline expects TwiBot-20 JSON with labels embedded in the data.
- Split files under `data/` (train/dev/test) are required for runs.
- Large datasets are intentionally not tracked in git; keep them local under `data/`.
- Account age uses a reference date derived from the training split to avoid leakage into validation/test distributions.
- Numeric features are aligned to the actual training data columns (including tweet counts and related activity features).

## Data Download
1) Download the TwiBot-20 dataset from the
   [TwiBot-20 repository](https://github.com/BunsenFeng/TwiBot-20).
2) Extract the archive.
3) Copy `train.json`, `dev.json`, and `test.json` into the local `data/` folder.

The `data/` folder is gitignored, so these files stay local and will not be committed.

## Testing
```bash
python -m unittest discover -s tests -v
```
Smoke tests are dependency-aware and will skip when optional libraries are missing.

## License
Academic research project (CS3IP, University of Reading).
