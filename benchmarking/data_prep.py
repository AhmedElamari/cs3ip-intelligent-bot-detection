"""
Data preparation helpers for the benchmark pipeline.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from DataLoader import load_twibot_splits_as_dict
from FeatureEngineering import BotFeatureExtractor, derive_reference_date
from Preprocessing import BotDetector
from config import Config
from pipeline_utils import safe_stratified_split


def load_data(data_dir: Path) -> dict:
    """Load TwiBot-20 JSON split data from the local data/ directory."""
    print(f"Detected pre-split dataset under {data_dir} (train/dev/test).")
    splits = load_twibot_splits_as_dict(data_dir)
    for name, df in splits.items():
        print(f"{name} split: {len(df)} samples")
    return splits


def prepare_data(data, config: Config) -> tuple:
    """Prepare data: feature engineering, preprocessing, splitting."""
    print("\n" + "=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)

    random_state = config.get('random_state')
    test_size = config.get('test_size', 0.1)
    val_size = config.get('val_size', 0.2)

    # Handle dict-based splits (preferred - preserves original experimental design)
    if isinstance(data, dict):
        splits = data
        train_df = splits['train'].copy()
        val_df = splits['val'].copy()
        test_df = splits['test'].copy()

        # Check for labels in each split
        cleaned = {}
        for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            if 'label' not in df.columns:
                raise ValueError(f"No 'label' column found in {name} split")
            df = df.dropna(subset=['label'])
            if df.empty:
                raise ValueError(
                    f"{name} split has no labeled rows after dropping missing labels."
                )
            cleaned[name] = df

        train_df = cleaned['train']
        val_df = cleaned['val']
        test_df = cleaned['test']

        # Derive reference date from training data only (avoid leakage)
        reference_date = derive_reference_date(train_df)

        # Feature engineering on each split
        print("\nExtracting features...")
        extractor = BotFeatureExtractor(reference_date=reference_date)
        train_df = extractor.extract_all_features(train_df)
        print(f"Extracted {len(extractor.get_feature_names())} features")
        val_df = extractor.extract_all_features(val_df)
        test_df = extractor.extract_all_features(test_df)

        # Preprocessing on each split
        print("\nPreprocessing...")
        detector = BotDetector()

        detector.data = train_df
        train_df = detector.preprocess()

        detector.data = val_df
        val_df = detector.preprocess()

        detector.data = test_df
        test_df = detector.preprocess()

        # Extract features and labels
        feature_names = (
            train_df.drop(columns=['label'])
            .select_dtypes(include=[np.number])
            .columns
            .tolist()
        )

        for df_name, df in (('val', val_df), ('test', test_df)):
            missing = [col for col in feature_names if col not in df.columns]
            if missing:
                df[missing] = 0
            if df_name == 'val':
                val_df = df
            else:
                test_df = df

        X_train = train_df[feature_names]
        y_train = train_df['label']
        X_val = val_df[feature_names]
        y_val = val_df['label']
        X_test = test_df[feature_names]
        y_test = test_df['label']

        print("\nFinal data shapes (using provided splits):")

    else:
        # Single DataFrame - split it ourselves
        df = data

        # Check for labels
        if 'label' not in df.columns:
            print("\n[WARNING] No 'label' column found.")
            print("Creating synthetic labels for demonstration...")
            np.random.seed(random_state)
            df['label'] = np.random.randint(0, 2, size=len(df))

        df = df.dropna(subset=['label'])
        if df.empty:
            raise ValueError(
                "No labeled records available after loading labels. "
                "Check that label IDs match the TwiBot IDs."
            )

        # Split indices first to avoid leakage when deriving reference dates
        indices = df.index.to_numpy()
        labels = df['label'].to_numpy()
        idx_temp, idx_test, labels_temp, _ = safe_stratified_split(
            indices,
            labels,
            test_size=test_size,
            random_state=random_state,
            split_name="test"
        )
        val_ratio = val_size / (1 - test_size)
        idx_train, idx_val, _, _ = safe_stratified_split(
            idx_temp,
            labels_temp,
            test_size=val_ratio,
            random_state=random_state,
            split_name="validation"
        )

        # Feature engineering
        print("\nExtracting features...")
        reference_date = derive_reference_date(df.loc[idx_train])
        extractor = BotFeatureExtractor(reference_date=reference_date)
        df = extractor.extract_all_features(df)
        print(f"Extracted {len(extractor.get_feature_names())} features")

        # Preprocessing
        print("\nPreprocessing...")
        detector = BotDetector()
        detector.data = df
        df = detector.preprocess()

        # Split data
        print(
            f"\nSplitting data (train/val/test: {(1-test_size-val_size)*100:.0f}%/"
            f"{val_size*100:.0f}%/{test_size*100:.0f}%)..."
        )
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

        print("\nFinal data shapes:")

    # Common post-processing for both paths

    # Reset detector for potential reuse
    detector = BotDetector()

    # Handle imbalance if configured
    if config.get('preprocessing.handle_imbalance'):
        method = config.get('preprocessing.imbalance_method', 'smote')
        print(f"\nApplying {method.upper()} for class balancing...")
        X_train, y_train = detector.handle_imbalance(X_train, y_train, method=method)

    # Scale features if configured
    if config.get('preprocessing.scale_features'):
        print("\nScaling features...")
        X_train, X_val, X_test = detector.scale_features(X_train, X_val, X_test)

    # Feature selection if configured
    if config.get('preprocessing.feature_selection'):
        n_features = config.get('preprocessing.n_features', 20)
        print(f"\nSelecting top {n_features} features...")
        X_train = detector.select_features(X_train, y_train, k=n_features)
        X_val = detector.apply_feature_selection(X_val)
        X_test = detector.apply_feature_selection(X_test)
        # Update feature names
        selected_indices = detector.selected_features
        feature_names = [feature_names[i] for i in selected_indices]

    print(f"  Training:   {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    print(f"  Test:       {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names
