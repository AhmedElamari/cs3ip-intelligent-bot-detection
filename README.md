# CS3IP: Intelligent Bot Detection - Interpretable ML

## Overview
This project implements an interpretable bot detection pipeline for social media data with an emphasis on explainability, reproducibility, and clean experimentation. It supports multiple supervised models, feature engineering grounded in domain knowledge, and benchmarking with XAI tooling (SHAP, LIME, feature importance).

## Key Capabilities
- Structured data loading for TwiBot-20 JSON
- Feature engineering with leakage-aware account age computation
- Train/validation/test splits with reproducible random state
- Multiple supervised models with a common interface
- Benchmarking with comparison tables and plots
- Explainability using SHAP, LIME, and feature importance analysis

## Project Structure
```
cs3ip-intelligent-bot-detection/
|-- DataLoader.py                 # TwiBot-20 JSON loader and flattening
|-- FeatureEngineering.py         # Feature extraction
|-- Preprocessing.py              # Cleaning and split helpers
|-- main.py                       # Single-model pipeline
|-- benchmark.py                  # Multi-model benchmark + XAI
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
pip install pandas numpy scikit-learn
```

Optional XAI tooling:
```bash
pip install shap lime matplotlib seaborn
```

## Usage

## Quickstart
Example files assumed:
- `data/train.json`, `data/dev.json`, `data/test.json` (embedded labels)
- Download the TwiBot-20 dataset from the [official repository](https://github.com/BunsenFeng/TwiBot-20)

Run a single model:
```bash
python main.py --model random_forest
```
Expected output (console):
- Training/validation/test sizes
- Validation + test metrics
- Confusion matrix and classification report

Run a benchmark with explainability:
```bash
python benchmark.py --explain --save-plots
```
Expected output (filesystem):
- `results/benchmark_YYYYMMDD_HHMMSS/model_comparison.csv`
- `results/benchmark_YYYYMMDD_HHMMSS/benchmark_report.txt`
- `results/benchmark_YYYYMMDD_HHMMSS/performance_comparison.png`
- `results/benchmark_YYYYMMDD_HHMMSS/feature_importance_comparison.csv`

### Single Model Pipeline
```bash
python main.py --model random_forest
```

Options:
- `--model`: `random_forest`, `logistic_regression`, `svm`
- `--smote`: enable SMOTE
- `--scale`: enable feature scaling
- `--features`: select top-k features

### Benchmarking and Explainability
```bash
python benchmark.py --explain --save-plots
```

Options:
- `--config`: load YAML or JSON config
- `--models`: specify models to run
- `--smote` / `--scale`: override preprocessing settings

Outputs are saved under `results/benchmark_YYYYMMDD_HHMMSS/`.

## Configuration
Configuration is centralized in `config/config.py` and supports YAML/JSON. Use `create_default_config()` to generate a template file and adjust model parameters, preprocessing options, and explainability settings.

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
