# CS3IP: Intelligent Bot Detection - Interpretable ML

## Overview
This project implements an interpretable bot detection pipeline for social media data with an emphasis on explainability, reproducibility, and clean experimentation. It supports multiple supervised models, feature engineering grounded in domain knowledge, and benchmarking with XAI tooling (SHAP, LIME, feature importance).

## Key Capabilities
- Structured data loading for TwiBot-20 JSON
- Feature engineering with leakage-aware account age computation
- Train/validation/test splits with reproducible random state
- Multiple supervised models with a common interface (LR, RF, XGBoost, SVM, DT, TabNet)
- **TabNet deep learning model** — sequential attention with intrinsic interpretability (feature masks)
- **Generalised Optuna HPO** for all registered models (`logistic_regression`, `svm`, `decision_tree`, `random_forest`, `xgboost`, `tabnet`) with `val_f1` on the train/val split, TPE sampler (seed 2112), cached `HPOResultV1` artifacts under `results/hpo_cache/<model>/<sha256>.json`
- Benchmarking with comparison tables, plots, and statistically grounded evaluation
- Bootstrap 95% confidence intervals per model metric
- Pairwise model significance: paired bootstrap delta test + McNemar exact test (Holm-Bonferroni corrected)
- Explainability using SHAP (TreeExplainer for XGBoost), LIME, and feature importance analysis
- Optional cost-aware adversarial robustness audit with flip-rate, confidence-drop on true bots, full-test-set Macro-F1 / PR-AUC degradation per profile, and dissertation figures (`robustness_profile_degradation.png`, feature vulnerability table/chart)

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
|-- benchmarking/                 # Benchmark runner, metrics, `hpo/` shared Optuna service
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

**Optuna** is included in `requirements.txt` for model HPO. Optional deep-learning profile (TabNet training):
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
python run_benchmark.py --explain
```
Run the optional robustness audit:
```bash
python run_benchmark.py --models random_forest xgboost --explain --robustness-analysis
```
Recommended **dissertation core** run (full HPO, **every** model from config when you omit `--models`, bootstrap CIs + pairwise significance, full CSV/JSON/report bundle and dissertation scoreboard; **skips** XAI/SHAP and robustness to save time; still writes PR-curve and confusion-matrix figures below):
```bash
python run_benchmark.py --dissertation-core
```
Expected output (filesystem):
- `results/benchmark_YYYYMMDD_HHMMSS/model_comparison.csv`
- `results/benchmark_YYYYMMDD_HHMMSS/dissertation_scoreboard.csv`, `dissertation_scoreboard.md`, and `dissertation_scoreboard.tex` — dissertation Table 8.2 style baseline scoreboard (F1-macro / F1-weighted, PR-AUC, MCC, balanced accuracy; Markdown/LaTeX bold best column values)
- `results/benchmark_YYYYMMDD_HHMMSS/benchmark_report.md`
- `results/benchmark_YYYYMMDD_HHMMSS/pr_curves_comparison.png` — test-set Precision-Recall curves for the top 3 models by scoreboard order (F1-Macro, ROC-AUC); legend includes PR-AUC per model; dashed line = positive-class prevalence on the test set
- `results/benchmark_YYYYMMDD_HHMMSS/confusion_matrix_best_model_normalized.png` — normalized by true label (rows sum to 1); best model by scoreboard; rows = True label, columns = Predicted label; class order Human, Bot
- `results/benchmark_YYYYMMDD_HHMMSS/confusion_matrix_best_model_raw.png` — raw-count companion (same axes/order)
- `results/benchmark_YYYYMMDD_HHMMSS/feature_importance.csv` — raw per-model feature importances
- `results/benchmark_YYYYMMDD_HHMMSS/run_metadata.json` - runtime, dataset, package, git, and artifact provenance
- `results/benchmark_YYYYMMDD_HHMMSS/results.json` — structured benchmark export contract with ranking metadata (includes per-model `hpo` audit when tuning ran)
- `results/benchmark_YYYYMMDD_HHMMSS/hpo_summary.json` — cache hit vs fresh study, trial counts, best `val_f1`, artifact paths
- `results/hpo_cache/<model>/<signature>.json` — persisted `HPOResultV1` for reuse when the run signature matches
- `results/benchmark_YYYYMMDD_HHMMSS/metric_confidence_intervals.csv` — 95% bootstrap CIs per model/metric
- `results/benchmark_YYYYMMDD_HHMMSS/pairwise_significance.csv` — delta, CI, and p-values for every model pair

- `results/benchmark_YYYYMMDD_HHMMSS/robustness_summary.csv` — profile-level flip / confidence metrics on true bots
- `results/benchmark_YYYYMMDD_HHMMSS/feature_attack_results.csv` — per-feature single-attack metrics
- `results/benchmark_YYYYMMDD_HHMMSS/robustness_degradation.csv` — Macro-F1 and PR-AUC on the full test set (baseline vs each adversarial profile; bots perturbed only)
- `results/benchmark_YYYYMMDD_HHMMSS/robustness_profile_degradation.png` — grouped Macro-F1 bars for top-3 scoreboard models (baseline, `cheap_only`, `realistic_mixed`)
- `results/benchmark_YYYYMMDD_HHMMSS/top_feature_vulnerabilities.csv` — top flip-rate features for the best model (single-feature attacks)
- `results/benchmark_YYYYMMDD_HHMMSS/feature_attack_flip_rates_best_model.png` — horizontal bar chart companion to the CSV above
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
- `--no-tune`: skip HPO; use hyperparameters from config only
- `--retune`: ignore the HPO cache and run a fresh Optuna study
- `--hpo-trials N`: override trial count for this run (otherwise `config.hpo.trials_per_model.<model>`)

HPO defaults are in `config/config.py` under `hpo.*` (enabled by default). A run summary is written to `results/hpo_summary.json`.

### Benchmarking and Explainability
```bash
python run_benchmark.py --explain
```

Explainability runs when either `--explain` is passed or `explainability.enabled` is `true` in the loaded config.
Passing `--explain` also sets `output.save_plots` so XAI PNG exports (e.g. `feature_importance_comparison.png`) are written; you can toggle `output.save_plots` in config alone if needed.

Options:
- `--config`: load YAML or JSON config
- `--models`: specify models to run (e.g. `logistic_regression random_forest xgboost tabnet`)
- `--smote` / `--scale`: override preprocessing settings
- `--no-tune` / `--retune` / `--hpo-trials`: same semantics as `main.py` (per-model HPO before training)
- `--robustness-analysis`: enable the optional adversarial robustness audit
- `--robustness-profiles`: override the default profiles (`cheap_only realistic_mixed`)

Outputs are saved under `results/benchmark_YYYYMMDD_HHMMSS/`.

**Metric note:** HPO maximises **validation F1** (`val_f1`). The printed `[BEST]` benchmark line still ranks models by **test F1** (existing `get_best_model('f1')` default).

### Hyperparameter search spaces (v1)

| Model | Tuned hyperparameters |
|-------|------------------------|
| `logistic_regression` | `C`, `solver` |
| `svm` | `C`, `kernel`, `gamma` (mode-dependent) |
| `decision_tree` | `max_depth`, `min_samples_split`, `min_samples_leaf`, `criterion` |
| `random_forest` | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features` |
| `xgboost` | `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda` |
| `tabnet` | Same ranges as legacy TabNet HPO (`n_d`/`n_a`, `n_steps`, `gamma`, `lambda_sparse`, `learning_rate`, batch sizes, `momentum`, `mask_type`; MedianPruner) |

Programmatic TabNet-only entry point (delegates to the shared service):

```python
from benchmarking.tabnet_optuna import optimize_tabnet
result = optimize_tabnet(X_train, y_train, X_val, y_val, n_trials=50, output_path="results/hpo.json")
```

Artifacts are `HPOResultV1` JSON; load with `load_hpo_result(path)`.

## Configuration
Configuration is centralized in `config/config.py` and supports YAML/JSON. Use `create_default_config()` to generate a template file and adjust model parameters, preprocessing options, explainability settings, and the optional `robustness.*` controls.

## Data Notes
- The pipeline expects TwiBot-20 JSON with labels embedded in the data.
- Split files under `data/` (train/dev/test) are required for runs.
- Large datasets are intentionally not tracked in git; keeping them local under `data/`.
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
