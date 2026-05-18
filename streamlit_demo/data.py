"""Approved VIVA demo snapshot values tied to on-disk benchmark artifacts."""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_ASSET_DIR = _REPO_ROOT / "demo_assets"
SHAP_SUMMARY_RF_PATH = DEMO_ASSET_DIR / "shap_summary_random_forest.png"
LIVE_PREDICTOR_JOBLIB_PATH = DEMO_ASSET_DIR / "live_predictor.joblib"

# --- Provenance (paths relative to repo root) ---
# Tab 1: six-model dissertation-style run (XGBoost #1 on test F1; RF tuned, stronger PR-AUC/ROC-AUC).
DEMO_ARENA_BENCHMARK_REL = "results/benchmark_20260517_101200"
# Tab 2 SHAP figure + Tab 3 live RF: Optuna-tuned RF on official TwiBot-20 train split.
DEMO_RF_TUNED_XAI_REL = "results/benchmark_20260517_091938"
DEMO_RF_HPO_CACHE_REL = (
    "results/hpo_cache/random_forest/"
    "e076c10b28698ab7ae24b52bbf79b89fbc22d4a1286ea8db8d5bdcf91529162f.json"
)

# Engineered numeric feature count for official split + default preprocessing (matches XAI run).
DEMO_ENGINEERED_FEATURE_COUNT = 24
# Tab 3 sliders/toggles the user can edit; remainder filled with training medians.
LIVE_UI_EDITABLE_FEATURE_COUNT = 8
DEMO_LIVE_MEDIAN_FILLED_COUNT = DEMO_ENGINEERED_FEATURE_COUNT - LIVE_UI_EDITABLE_FEATURE_COUNT

# From results/benchmark_20260517_101200/model_comparison.csv + results.json test_metrics.pr_auc
# Rows ordered for the VIVA narrative: peak F1 (XGBoost), then Random Forest as the dissertation
# reliability / calibration anchor, then remaining models by benchmark F1 rank.
MODEL_ARENA_ROWS: list[dict[str, float | str | bool]] = [
    {
        "name": "XGBoost",
        "f1": 0.8541,
        "pr_auc": 0.8384,
        "roc_auc": 0.8604,
        "mcc": 0.6638,
        "train_seconds": 1.15,
        "champion": True,
    },
    {
        "name": "Random Forest",
        "f1": 0.8463,
        "pr_auc": 0.8607,
        "roc_auc": 0.8704,
        "mcc": 0.6403,
        "train_seconds": 2.91,
        "champion": False,
    },
    {
        "name": "Decision Tree",
        "f1": 0.8537,
        "pr_auc": 0.8109,
        "roc_auc": 0.8492,
        "mcc": 0.6644,
        "train_seconds": 0.13,
        "champion": False,
    },
    {
        "name": "SVM",
        "f1": 0.8519,
        "pr_auc": 0.795,
        "roc_auc": 0.8273,
        "mcc": 0.6578,
        "train_seconds": 41.34,
        "champion": False,
    },
    {
        "name": "Logistic Regression",
        "f1": 0.8475,
        "pr_auc": 0.8137,
        "roc_auc": 0.8382,
        "mcc": 0.645,
        "train_seconds": 0.06,
        "champion": False,
    },
    {
        "name": "TabNet",
        "f1": 0.8395,
        "pr_auc": 0.8118,
        "roc_auc": 0.8454,
        "mcc": 0.6227,
        "train_seconds": 276.67,
        "champion": False,
    },
]

# From tuned RF benchmark feature_importance.csv (random_forest column), top 10 by value
FEATURE_IMPORTANCE_ROWS: list[tuple[str, float]] = [
    ("is_verified", 0.3171),
    ("followers_to_friends_ratio", 0.1407),
    ("followers_count", 0.0862),
    ("listed_count", 0.0858),
    ("followers_per_day", 0.0802),
    ("friends_count", 0.0387),
    ("account_age_days", 0.0373),
    ("statuses_count", 0.0306),
    ("description_length", 0.0296),
    ("tweets_per_day", 0.0277),
]

RESILIENCE_ROWS: list[tuple[str, str, float, str]] = [
    (
        "is_verified",
        "High",
        0.08,
        "Platform-controlled — adversary cannot fake without buying access.",
    ),
    (
        "followers_to_friends_ratio",
        "Medium",
        0.23,
        "Requires a coordinated follower farm.",
    ),
    (
        "tweets_per_day",
        "Low",
        0.41,
        "Trivially evaded by throttling posting frequency.",
    ),
    (
        "default_profile_image",
        "Low",
        0.19,
        "A single image upload defeats this signal.",
    ),
    (
        "screen_name_has_digits",
        "Low",
        0.31,
        "One rename and the cue disappears.",
    ),
]

DEMO_DATASET_META: dict[str, str] = {
    "account_count_label": "n = 11,826 accounts",
    "class_ratio_label": "Class ratio ≈ 1.8 : 1",
}
