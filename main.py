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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

from DataLoader import (
    TwiBotDataLoader, load_twibot_json,
    load_twibot_splits_as_dict, check_twibot_data_available
)
from FeatureEngineering import BotFeatureExtractor
from Preprocessing import BotDetector

REPO_ROOT = Path(__file__).resolve().parent
TWIBOT20_DATA_PATH = REPO_ROOT / "TwiBot-20_sample.json"
TWIBOT20_DATA_DIR = REPO_ROOT / "data"


def resolve_data_source() -> dict:
    """
    Determine best available data source.
    
    Returns:
        dict with 'type' ('splits' or 'sample'), 'path', and 'count'
    """
    availability = check_twibot_data_available()
    
    # Prefer split files if they have labels and reasonable count
    if availability['total_split_samples'] > 0:
        splits_info = availability['splits_available']
        has_labels = all(s['has_labels'] for s in splits_info.values() if s['exists'])
        if has_labels:
            return {
                'type': 'splits',
                'path': TWIBOT20_DATA_DIR,
                'count': availability['total_split_samples']
            }
    
    # Fall back to sample file
    if availability['sample_available']:
        return {
            'type': 'sample',
            'path': TWIBOT20_DATA_PATH,
            'count': 100  # Known sample size
        }
    
    raise FileNotFoundError(
        "No TwiBot-20 dataset found. Expected either:\n"
        f"  - Split files in {TWIBOT20_DATA_DIR} (train.json, dev.json, test.json)\n"
        f"  - Sample file at {TWIBOT20_DATA_PATH}"
    )


def load_and_prepare_data(
    data_path: str = None,
    use_splits: bool = True
):
    """Load JSON data and prepare it for processing.
    
    Prefers loading as separate splits (dict) to preserve original experimental design.
    
    Args:
        data_path: Explicit path to JSON file (overrides auto-detection)
        use_splits: If True and split files available, use them
        
    Returns:
        Either dict of DataFrames {'train', 'val', 'test'} or single DataFrame
    """
    if data_path:
        # Use explicit path
        print(f"Loading data from: {data_path}")
        loader = TwiBotDataLoader(json_path=data_path)
        df = loader.load()
        print(f"Loaded {len(df)} records")
        if 'label' in df.columns:
            print(f"Label distribution: {df['label'].value_counts().to_dict()}")
        return df
    
    # Auto-detect best data source
    source = resolve_data_source()
    
    # Use original splits (dict) when available - preserves experimental design
    if source['type'] == 'splits' and use_splits:
        print(f"Detected pre-split dataset under {source['path']} (train/dev/test).")
        splits = load_twibot_splits_as_dict(source['path'])
        for name, df in splits.items():
            print(f"{name} split: {len(df)} samples")
        return splits
    
    # Fall back to single file
    print(f"Loading TwiBot-20 sample from: {source['path']}")
    df = load_twibot_json(str(source['path']))
    print(f"Loaded {len(df)} records")
    if 'label' in df.columns:
        print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    return df


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
        print(
            f"\n[WARNING] Stratified {split_name} split failed ({exc}). "
            "Falling back to unstratified split."
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
    num_features: int = None,
    data_path: str = None,
    use_sample: bool = False
):
    """Run the complete bot detection pipeline on TwiBot-20.
    
    Args:
        model_type: Model to use ('random_forest', 'logistic_regression', 'svm')
        use_smote: Apply SMOTE for class balancing
        use_scaling: Apply feature scaling
        num_features: Number of features to select (None = all)
        data_path: Explicit path to JSON data file
        use_sample: Force use of sample file instead of split files
    """
    
    print("=" * 60)
    print("BOT DETECTION PIPELINE")
    print("=" * 60)
    
    # Step 1: Load data
    data = load_and_prepare_data(
        data_path=data_path,
        use_splits=not use_sample
    )
    
    # Handle dict-based splits (preferred - preserves original experimental design)
    if isinstance(data, dict):
        splits = data
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
        
    else:
        # Single DataFrame - split it ourselves
        df = data
        
        # Check if labels exist
        if 'label' not in df.columns:
            print("\nWARNING: No 'label' column found in data.")
            print("For demo purposes, creating synthetic labels...")
            np.random.seed(2112)
            df['label'] = np.random.randint(0, 2, size=len(df))
            print(f"Created synthetic labels: {df['label'].value_counts().to_dict()}")
        
        df = df.dropna(subset=['label'])
        if df.empty:
            raise ValueError(
                "No labeled records available after loading labels. "
                "Check that label IDs match the TwiBot IDs."
            )
        
        indices = df.index.to_numpy()
        labels = df['label'].to_numpy()
        idx_temp, idx_test, labels_temp, _ = _safe_stratified_split(
            indices, labels, test_size=0.1, random_state=2112, split_name="test"
        )
        val_ratio = 0.2 / (1 - 0.1)
        idx_train, idx_val, _, _ = _safe_stratified_split(
            idx_temp, labels_temp, test_size=val_ratio, random_state=2112, split_name="validation"
        )
        
        # Feature engineering
        reference_date = None
        if 'account_creation_date' in df.columns:
            account_creation = pd.to_datetime(
                df.loc[idx_train, 'account_creation_date'], errors='coerce'
            )
            if account_creation.notna().any():
                reference_date = account_creation.max()
            else:
                full_account_creation = pd.to_datetime(
                    df['account_creation_date'], errors='coerce'
                )
                reference_date = pd.Timestamp.utcnow()
                if full_account_creation.dt.tz is not None:
                    reference_date = reference_date.tz_localize(full_account_creation.dt.tz)
        df = engineer_features(df, reference_date=reference_date)
        
        # Preprocessing
        print("\nPreprocessing data...")
        detector = BotDetector()
        detector.data = df
        df = detector.preprocess()
        
        # Split data
        print("\nSplitting data (70% train, 20% validation, 10% test)...")
        feature_names = (
            df.drop(columns=['label'])
            .select_dtypes(include=[np.number])
            .columns
            .tolist()
        )
        X = df[feature_names]
        y = df['label']
        X_train, X_val, X_test = X.loc[idx_train], X.loc[idx_val], X.loc[idx_test]
        y_train, y_val, y_test = y.loc[idx_train], y.loc[idx_val], y.loc[idx_test]
        
        print(f"Training set:   {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set:       {len(X_test)} samples")
    
    # Common processing for both paths
    detector = BotDetector()
    
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
        '--data', '-d',
        type=str,
        default=None,
        help='Explicit path to JSON data file (overrides auto-detection)'
    )
    parser.add_argument(
        '--use-sample',
        action='store_true',
        help='Force use of sample file instead of full split data'
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
        num_features=args.features,
        data_path=args.data,
        use_sample=args.use_sample
    )


if __name__ == '__main__':
    main()
