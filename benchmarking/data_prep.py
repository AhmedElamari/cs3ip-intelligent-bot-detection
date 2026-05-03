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
from pipeline_utils import apply_time_split_if_enabled


def preprocess_split(detector: BotDetector, df: pd.DataFrame) -> pd.DataFrame:
    """Apply preprocessing to a single data split."""
    detector.data = df
    return detector.preprocess()


def load_data(data_dir: Path) -> dict:
    """Load TwiBot-20 JSON split data from the local data/ directory.

    Args:
        data_dir: Path to the directory containing the TwiBot-20 JSON splits 
        (e.g. data/train.json, data/dev.json, data/test.json)

    Returns:
        Dictionary of data splits keyed by split name to corresponding DataFrame.
        For example, {
            'train': train_df,
            'val': val_df,
            'test': test_df,
        }.
    """
    print(f"Detected pre-split dataset under {data_dir} (train/dev/test).")
    splits = load_twibot_splits_as_dict(data_dir)
    for name, df in splits.items():
        print(f"{name} split: {len(df)} samples")
    return splits


def prepare_data(
    data,
    config: Config,
    return_metadata: bool = False,
    *,
    temporal_protocol: bool = False,
) -> tuple:
    """Prepare data for model training with feature engineering and preprocessing.

    Args:
        data: Dictionary of data splits keyed by split name to corresponding DataFrame
            (e.g. {'train': train_df, 'val': val_df, 'test': test_df}).
        config: Configuration object controlling preprocessing options
            such as imbalance handling, feature scaling policy, and feature selection.

    Returns:
        Tuple containing the training, validation, and test splits plus feature names.
    """
    print("\n" + "=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)

    if not isinstance(data, dict):
        raise TypeError(
            "prepare_data expects a dict with 'train', 'val', and 'test' DataFrames."
        )

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

    if temporal_protocol:
        reference_date = derive_reference_date(
            pd.concat([train_df, val_df, test_df], ignore_index=True)
        )
        use_time_split = False
    elif config.get('time_split'):
        use_time_split = True
    else:
        use_time_split = False

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
            val_size=config.get('val_size', 0.2),
            test_size=config.get('test_size', 0.1),
            time_col='account_creation_date',
            random_state=config.get('random_state', 2112)
        )
    elif not temporal_protocol:
        reference_date = derive_reference_date(train_df)
    if use_time_split:
        print(f"  Train (oldest):   {len(train_df)} samples")
        print(f"  Val (middle):     {len(val_df)} samples")
        print(f"  Test (newest):    {len(test_df)} samples")

    # Log reference date for transparency
    if reference_date is not None:
        print(f"\nReference date for age features: {reference_date.date()}")

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

    train_df = preprocess_split(detector, train_df)
    val_df = preprocess_split(detector, val_df)
    test_df = preprocess_split(detector, test_df)

    # Extract features and labels
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

    print("\nFinal data shapes (using provided splits):")

    # Handle imbalance if configured
    test_metadata = pd.DataFrame({
        'user_id': test_df.get(
            'user_id',
            pd.Series(['n/a'] * len(test_df), index=test_df.index),
        ).astype(str).values,
        'row_index': np.arange(len(test_df)),
        'label': test_df['label'].astype(int).values,
    })

    if config.get('preprocessing.handle_imbalance'):
        method = config.get('preprocessing.imbalance_method', 'smote')
        print(f"\nApplying {method.upper()} for class balancing...")
        X_train, y_train = detector.handle_imbalance(X_train, y_train, method=method)

    # Scaling is per-model in run_benchmark; config key preprocessing.scale_features toggles it.

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

    prepared = (X_train, X_val, X_test, y_train, y_val, y_test, feature_names)
    if return_metadata:
        return (*prepared, test_metadata)
    return prepared
