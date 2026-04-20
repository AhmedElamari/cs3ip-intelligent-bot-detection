"""
Bot Detection Pipeline
======================
Main script to run the complete bot detection pipeline:
JSON loading -> Feature engineering -> Preprocessing -> Model training -> Evaluation
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from config import Config
from benchmarking.hpo.service import (
    HPOCliOverrides,
    merge_hpo_into_config_params,
    resolve_hpo,
)
from DataLoader import load_twibot_splits_as_dict
from FeatureEngineering import BotFeatureExtractor, derive_reference_date
from Preprocessing import BotDetector
from training import train_and_evaluate
from pipeline_utils import apply_time_split_if_enabled

REPO_ROOT = Path(__file__).resolve().parent
TWIBOT20_DATA_DIR = REPO_ROOT / "data"
DEFAULTS = Config.DEFAULTS
DEFAULT_RANDOM_STATE = DEFAULTS['random_state']
DEFAULT_TEST_SIZE = DEFAULTS['test_size']
DEFAULT_VAL_SIZE = DEFAULTS['val_size']


def load_and_prepare_data() -> dict:
    """Load TwiBot-20 JSON split data from the local data/ directory."""
    print(f"Detected pre-split dataset under {TWIBOT20_DATA_DIR} (train/dev/test).")
    splits = load_twibot_splits_as_dict(TWIBOT20_DATA_DIR)
    for name, df in splits.items():
        print(f"{name} split: {len(df)} samples")
    return splits


def engineer_features(df: pd.DataFrame, reference_date: pd.Timestamp = None) -> pd.DataFrame:
    """Apply feature engineering to the dataset."""
    print("\nExtracting features...")
    extractor = BotFeatureExtractor(reference_date=reference_date)
    df = extractor.extract_all_features(df)
    feature_names = extractor.get_feature_names()
    print(f"Extracted {len(feature_names)} features: {feature_names}")
    return df


def preprocess_split(detector: BotDetector, df: pd.DataFrame) -> pd.DataFrame:
    """Apply preprocessing to a single data split."""
    detector.data = df
    return detector.preprocess()


def run_pipeline(
    model_type: str = 'random_forest',
    use_smote: bool = False,
    use_scaling: bool = False,
    num_features: Optional[int] = None,
    use_time_split: bool = False,
    val_size: float = DEFAULT_VAL_SIZE,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    no_tune: bool = False,
    retune: bool = False,
    hpo_trials: Optional[int] = None,
    config: Optional[Config] = None,
):
    """Run the complete bot detection pipeline on TwiBot-20.
    
    Args:
        model_type: Model to use ('random_forest', 'logistic_regression', 'svm')
        use_smote: Apply SMOTE for class balancing
        use_scaling: Apply feature scaling
        num_features: Number of features to select (None = all)
        use_time_split: Use chronological splitting to combat data drift
        val_size: Fraction of data for validation split (time split only)
        test_size: Fraction of data for test split (time split only)
        random_state: Random seed for time-split shuffling
    """
    
    cfg = config or Config()
    if no_tune:
        cfg.set("hpo.enabled", False)

    print("=" * 60)
    print("BOT DETECTION PIPELINE")
    print("=" * 60)
    
    # Step 1: Load data
    splits = load_and_prepare_data()

    split_frames = {name: splits[name].copy() for name in ('train', 'val', 'test')}

    # Check for labels
    for name, df in split_frames.items():
        if 'label' not in df.columns:
            raise ValueError(f"No 'label' column found in {name} split")
        df = df.dropna(subset=['label'])
        if df.empty:
            raise ValueError(
                f"{name} split has no labeled rows after dropping missing labels."
            )
        split_frames[name] = df

    train_df = split_frames['train']
    val_df = split_frames['val']
    test_df = split_frames['test']

    # Apply time-stratified split if enabled (combats data drift)
    if use_time_split:
        print("\nApplying time-stratified split (chronological ordering)...")
        # Derive reference date from ALL data BEFORE splitting:
        # - Avoids negative "age" features for newer accounts.
        # - With chronological splitting, all accounts already exist at or before
        #   this max-date reference, so this does not leak future labels.
        train_df, val_df, test_df, reference_date = apply_time_split_if_enabled(
            train_df,
            val_df,
            test_df,
            use_time_split=True,
            val_size=val_size,
            test_size=test_size,
            time_col='account_creation_date',
            random_state=random_state
        )
    else:
        reference_date = derive_reference_date(train_df)
    if use_time_split:
        print(f"  Train (oldest):   {len(train_df)} samples")
        print(f"  Val (middle):     {len(val_df)} samples")
        print(f"  Test (newest):    {len(test_df)} samples")

    # Log reference date for transparency
    if reference_date is not None:
        print(f"\nReference date for age features: {reference_date.date()}")

    # Feature engineering on each split
    train_df = engineer_features(train_df, reference_date=reference_date)
    val_df = engineer_features(val_df, reference_date=reference_date)
    test_df = engineer_features(test_df, reference_date=reference_date)

    # Preprocessing
    print("\nPreprocessing data...")
    detector = BotDetector()

    train_df = preprocess_split(detector, train_df)
    val_df = preprocess_split(detector, val_df)
    test_df = preprocess_split(detector, test_df)

    # Extract features
    feature_names = (
        train_df.drop(columns=['label'])
        .select_dtypes(include=[np.number])
        .columns
        .tolist()
    )
    if not feature_names:
        raise ValueError("No numeric features found after preprocessing.")
    missing_val = sorted(set(feature_names) - set(val_df.columns))
    missing_test = sorted(set(feature_names) - set(test_df.columns))
    if missing_val or missing_test:
        raise ValueError(
            "Feature mismatch detected between train and eval splits. "
            f"Missing in val: {missing_val}; missing in test: {missing_test}"
        )

    X_train = train_df[feature_names]
    y_train = train_df['label']
    X_val = val_df[feature_names]
    y_val = val_df['label']
    X_test = test_df[feature_names]
    y_test = test_df['label']

    print("\nUsing original splits:")
    print(f"Training set:   {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set:       {len(X_test)} samples")
    
    # Post-processing steps
    
    # Step 5: Handle class imbalance (optional)
    if use_smote:
        print("\nApplying SMOTE for class balancing...")
        X_train, y_train = detector.handle_imbalance(X_train, y_train, method='smote')
        print(f"After SMOTE: {len(X_train)} training samples")
    
    # Step 6: scaling (LR/SVM) inside train_and_evaluate when should_scale
    should_scale = (use_scaling or model_type in ('logistic_regression', 'svm')) and model_type != 'tabnet'
    if should_scale:
        print("\nApplying feature scaling in training (LR/SVM).")
    
    # Step 7: Feature selection (optional)
    if num_features:
        print(f"\nSelecting top {num_features} features...")
        X_train = detector.select_features(X_train, y_train, k=num_features)
        X_val = detector.apply_feature_selection(X_val)
        X_test = detector.apply_feature_selection(X_test)
        feature_names = [feature_names[i] for i in detector.selected_features]
    
    # Step 8: Calculate class weights
    class_weights = detector.get_class_weights(y_train)
    print(f"\nClass weights: {class_weights}")

    live_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else feature_names

    hpo_result, hpo_audit = resolve_hpo(
        model_type,
        cfg,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_names_ordered=list(live_names),
        data_dir=TWIBOT20_DATA_DIR,
        enable_scaling=should_scale,
        class_weights=class_weights,
        cli=HPOCliOverrides(
            no_tune=no_tune,
            retune=retune,
            hpo_trials=hpo_trials,
        ),
    )
    if hpo_result.get("status") != "skipped" and hpo_result.get("best_params"):
        merge_hpo_into_config_params(cfg, model_type, hpo_result["best_params"])

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "hpo_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "HPOSummaryV1",
                "models": {
                    model_type: {
                        **hpo_audit,
                        "best_score": hpo_result.get("best_score"),
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Step 9: Train and evaluate
    results = train_and_evaluate(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        model_type=model_type,
        class_weights=class_weights,
        feature_names=live_names,
        enable_scaling=should_scale,
        config=cfg,
    )
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Bot Detection Pipeline (TwiBot-20 only)'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='random_forest',
        choices=['random_forest', 'logistic_regression', 'svm', 'tabnet'],
        help='Model type to use (tabnet requires: pip install -r requirements-dl.txt)'
    )
    parser.add_argument(
        '--smote',
        action='store_true',
        help='Use SMOTE for class balancing'
    )
    parser.add_argument(
        '--scale',
        action='store_true',
        help='Apply feature scaling'
    )
    parser.add_argument(
        '--features', '-f',
        type=int,
        default=None,
        help='Number of features to select (uses all if not specified)'
    )
    parser.add_argument(
        '--time-split',
        action='store_true',
        help='Use chronological splitting to combat data drift (train oldest, test newest)'
    )
    parser.add_argument(
        '--val-size',
        type=float,
        default=DEFAULT_VAL_SIZE,
        help='Validation split fraction for time split'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=DEFAULT_TEST_SIZE,
        help='Test split fraction for time split'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help='Random seed for time-split shuffling'
    )
    parser.add_argument(
        '--no-tune',
        action='store_true',
        help='Skip hyperparameter optimisation; use config params only',
    )
    parser.add_argument(
        '--retune',
        action='store_true',
        help='Ignore HPO cache and run a fresh Optuna study',
    )
    parser.add_argument(
        '--hpo-trials',
        type=int,
        default=None,
        help='Override number of Optuna trials for this run',
    )

    args = parser.parse_args()

    if args.no_tune and args.retune:
        parser.error("Cannot use --no-tune together with --retune")

    run_pipeline(
        model_type=args.model,
        use_smote=args.smote,
        use_scaling=args.scale,
        num_features=args.features,
        use_time_split=args.time_split,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.random_state,
        no_tune=args.no_tune,
        retune=args.retune,
        hpo_trials=args.hpo_trials,
    )


if __name__ == '__main__':
    main()
