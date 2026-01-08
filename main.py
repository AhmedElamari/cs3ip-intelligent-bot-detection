"""
Bot Detection Pipeline
======================
Main script to run the complete bot detection pipeline:
JSON loading -> Feature engineering -> Preprocessing -> Model training -> Evaluation
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from DataLoader import load_twibot_splits_as_dict
from FeatureEngineering import BotFeatureExtractor, derive_reference_date
from Preprocessing import BotDetector
from training import train_and_evaluate

REPO_ROOT = Path(__file__).resolve().parent
TWIBOT20_DATA_DIR = REPO_ROOT / "data"


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


def run_pipeline(
    model_type: str = 'random_forest',
    use_smote: bool = False,
    use_scaling: bool = False,
    num_features: Optional[int] = None
):
    """Run the complete bot detection pipeline on TwiBot-20.
    
    Args:
        model_type: Model to use ('random_forest', 'logistic_regression', 'svm')
        use_smote: Apply SMOTE for class balancing
        use_scaling: Apply feature scaling
        num_features: Number of features to select (None = all)
    """
    
    print("=" * 60)
    print("BOT DETECTION PIPELINE")
    print("=" * 60)
    
    # Step 1: Load data
    splits = load_and_prepare_data()

    # Validate and clean each split
    split_names = ['train', 'val', 'test']
    split_frames = {}
    for name in split_names:
        df = splits[name].copy()
        if 'label' not in df.columns:
            raise ValueError(f"No 'label' column found in {name} split")
        df = df.dropna(subset=['label'])
        if df.empty:
            raise ValueError(
                f"{name} split has no labeled rows after dropping missing labels."
            )
        split_frames[name] = df

    train_df, val_df, test_df = (split_frames[name] for name in split_names)

    # Derive reference date from training data only
    reference_date = derive_reference_date(train_df)

    # Feature engineering on each split
    print("\nExtracting features...")
    train_df = engineer_features(train_df, reference_date=reference_date)
    val_df = engineer_features(val_df, reference_date=reference_date)
    test_df = engineer_features(test_df, reference_date=reference_date)

    # Preprocessing each split
    print("\nPreprocessing data...")
    detector = BotDetector()
    processed = {}
    for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        detector.data = df
        processed[name] = detector.preprocess()
    
    train_df, val_df, test_df = processed['train'], processed['val'], processed['test']

    # Extract features
    feature_names = (
        train_df.drop(columns=['label'])
        .select_dtypes(include=[np.number])
        .columns
        .tolist()
    )

    # Ensure all splits have the same features
    for df in (val_df, test_df):
        missing = [col for col in feature_names if col not in df.columns]
        if missing:
            df[missing] = 0

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
    
    # Step 6: Feature scaling (recommended for logistic regression/SVM)
    should_scale = use_scaling or model_type in ('logistic_regression', 'svm')
    if should_scale:
        if use_scaling:
            print("\nApplying feature scaling...")
        else:
            print(f"\nApplying feature scaling for {model_type}...")
        X_train, X_val, X_test = detector.scale_features(X_train, X_val, X_test)
    
    # Step 7: Feature selection (optional)
    if num_features:
        print(f"\nSelecting top {num_features} features...")
        X_train = detector.select_features(X_train, y_train, k=num_features)
        X_val = detector.apply_feature_selection(X_val)
        X_test = detector.apply_feature_selection(X_test)
    
    # Step 8: Calculate class weights
    class_weights = detector.get_class_weights(y_train)
    print(f"\nClass weights: {class_weights}")
    
    # Step 9: Train and evaluate
    results = train_and_evaluate(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        model_type=model_type,
        class_weights=class_weights
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
        choices=['random_forest', 'logistic_regression', 'svm'],
        help='Model type to use'
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
    
    args = parser.parse_args()
    
    run_pipeline(
        model_type=args.model,
        use_smote=args.smote,
        use_scaling=args.scale,
        num_features=args.features
    )


if __name__ == '__main__':
    main()
