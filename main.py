"""
Bot Detection Pipeline
======================
Main script to run the complete bot detection pipeline:
JSON loading -> Feature engineering -> Preprocessing -> Model training -> Evaluation
"""

import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

from DataLoader import load_twibot_splits_as_dict, check_twibot_data_available
from FeatureEngineering import BotFeatureExtractor
from Preprocessing import BotDetector

REPO_ROOT = Path(__file__).resolve().parent
TWIBOT20_DATA_DIR = REPO_ROOT / "data"
LOGGER = logging.getLogger(__name__)


def resolve_data_source() -> dict:
    """
    Determine best available data source.
    
    Returns:
        dict with 'type' ('splits'), 'path', and 'count'
    """
    availability = check_twibot_data_available()
    
    # Require split files with labels
    if availability['total_split_samples'] > 0:
        splits_info = availability['splits_available']
        has_labels = all(s['has_labels'] for s in splits_info.values() if s['exists'])
        if has_labels:
            return {
                'type': 'splits',
                'path': TWIBOT20_DATA_DIR,
                'count': availability['total_split_samples']
            }
    
    raise FileNotFoundError(
        "No TwiBot-20 dataset found. Expected split files in:\n"
        f"  {TWIBOT20_DATA_DIR} (train.json, dev.json, test.json)"
    )


def load_and_prepare_data() -> dict:
    """Load TwiBot-20 JSON split data from the local data/ directory."""
    source = resolve_data_source()
    print(f"Detected pre-split dataset under {source['path']} (train/dev/test).")
    splits = load_twibot_splits_as_dict(source['path'])
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


def train_and_evaluate(
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    model_type: str = 'random_forest',
    class_weights: dict = None
) -> dict:
    """Train model and evaluate on validation and test sets."""
    
    # Select model
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight=class_weights,
            random_state=2112,
            n_jobs=-1
        )
    elif model_type == 'logistic_regression':
        model = LogisticRegression(
            class_weight=class_weights,
            random_state=2112,
            max_iter=1000
        )
    elif model_type == 'svm':
        model = SVC(
            class_weight=class_weights,
            random_state=2112,
            kernel='rbf'
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"\nTraining {model_type}...")
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_val_pred = model.predict(X_val)
    val_metrics = {
        'accuracy': accuracy_score(y_val, y_val_pred),
        'precision': precision_score(y_val, y_val_pred, average='binary', zero_division=0),
        'recall': recall_score(y_val, y_val_pred, average='binary', zero_division=0),
        'f1': f1_score(y_val, y_val_pred, average='binary', zero_division=0)
    }
    
    print(f"\nValidation Results:")
    print(f"  Accuracy:  {val_metrics['accuracy']:.4f}")
    print(f"  Precision: {val_metrics['precision']:.4f}")
    print(f"  Recall:    {val_metrics['recall']:.4f}")
    print(f"  F1 Score:  {val_metrics['f1']:.4f}")
    
    # Evaluate on test set
    y_test_pred = model.predict(X_test)
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred, average='binary', zero_division=0),
        'recall': recall_score(y_test, y_test_pred, average='binary', zero_division=0),
        'f1': f1_score(y_test, y_test_pred, average='binary', zero_division=0)
    }
    
    print(f"\nTest Results:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    
    print(f"\nClassification Report (Test):")
    test_labels = np.unique(np.concatenate([y_test, y_test_pred]))
    target_names = [
        "Human" if lbl == 0 else "Bot" if lbl == 1 else str(lbl)
        for lbl in test_labels
    ]
    print(
        classification_report(
            y_test,
            y_test_pred,
            labels=test_labels,
            target_names=target_names,
            zero_division=0
        )
    )
    
    print(f"\nConfusion Matrix (Test):")
    print(confusion_matrix(y_test, y_test_pred))
    
    return {
        'model': model,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics
    }


def _derive_reference_date(train_df: pd.DataFrame) -> pd.Timestamp:
    """Derive a leakage-safe reference date from the training split."""
    if 'account_creation_date' not in train_df.columns:
        return None
    account_creation = pd.to_datetime(
        train_df['account_creation_date'],
        errors='coerce'
    )
    if account_creation.notna().any():
        ref_date = account_creation.max()
        if ref_date is not None and getattr(ref_date, "tz", None) is None:
            ref_date = ref_date.tz_localize("UTC")
        return ref_date
    return None


def _safe_stratified_split(
    indices: np.ndarray,
    labels: np.ndarray,
    test_size: float,
    random_state: int,
    split_name: str
):
    try:
        return train_test_split(
            indices,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )
    except ValueError as exc:
        LOGGER.warning(
            "Stratified %s split failed (%s). Falling back to unstratified split.",
            split_name,
            exc
        )
        return train_test_split(
            indices,
            labels,
            test_size=test_size,
            random_state=random_state
        )


def run_pipeline(
    model_type: str = 'random_forest',
    use_smote: bool = False,
    use_scaling: bool = False,
    num_features: int = None
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

    train_df = splits['train'].copy()
    val_df = splits['val'].copy()
    test_df = splits['test'].copy()

    # Check for labels
    for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        if 'label' not in df.columns:
            raise ValueError(f"No 'label' column found in {name} split")
        df.dropna(subset=['label'], inplace=True)

    # Derive reference date from training data only
    reference_date = _derive_reference_date(train_df)

    # Feature engineering on each split
    print("\nExtracting features...")
    train_df = engineer_features(train_df, reference_date=reference_date)
    val_df = engineer_features(val_df, reference_date=reference_date)
    test_df = engineer_features(test_df, reference_date=reference_date)

    # Preprocessing
    print("\nPreprocessing data...")
    detector = BotDetector()

    detector.data = train_df
    train_df = detector.preprocess()

    detector.data = val_df
    val_df = detector.preprocess()

    detector.data = test_df
    test_df = detector.preprocess()

    # Extract features
    feature_names = (
        train_df.drop(columns=['label'])
        .select_dtypes(include=[np.number])
        .columns
        .tolist()
    )

    X_train = train_df[feature_names]
    y_train = train_df['label']
    X_val = val_df[feature_names]
    y_val = val_df['label']
    X_test = test_df[feature_names]
    y_test = test_df['label']

    print(f"\nUsing original splits:")
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
