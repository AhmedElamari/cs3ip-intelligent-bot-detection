"""
Bot Detection Pipeline
======================
Main script to run the complete bot detection pipeline:
JSON loading → Feature engineering → Preprocessing → Model training → Evaluation
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

from DataLoader import TwiBotDataLoader, load_twibot_json
from FeatureEngineering import BotFeatureExtractor
from Preprocessing import BotDetector


def load_and_prepare_data(json_path: str, label_path: str = None) -> pd.DataFrame:
    """Load JSON data and prepare it for processing."""
    print(f"Loading data from: {json_path}")
    
    loader = TwiBotDataLoader(json_path, label_path)
    df = loader.load()
    
    print(f"Loaded {len(df)} records")
    print(f"Columns: {df.columns.tolist()}")
    
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
    print(classification_report(y_test, y_test_pred, target_names=['Human', 'Bot']))
    
    print(f"\nConfusion Matrix (Test):")
    print(confusion_matrix(y_test, y_test_pred))
    
    return {
        'model': model,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics
    }


def run_pipeline(
    json_path: str,
    label_path: str = None,
    model_type: str = 'random_forest',
    use_smote: bool = False,
    use_scaling: bool = False,
    num_features: int = None
):
    """Run the complete bot detection pipeline."""
    
    print("=" * 60)
    print("BOT DETECTION PIPELINE")
    print("=" * 60)
    
    # Step 1: Load data
    df = load_and_prepare_data(json_path, label_path)
    
    # Check if labels exist
    if 'label' not in df.columns:
        print("\n⚠️  WARNING: No 'label' column found in data.")
        print("Please provide a labels file with --labels argument.")
        print("For demo purposes, creating synthetic labels...")
        # Create synthetic labels for demo (random)
        np.random.seed(2112)
        df['label'] = np.random.randint(0, 2, size=len(df))
        print(f"Created synthetic labels: {df['label'].value_counts().to_dict()}")
    
    # Step 2: Feature engineering
    df = engineer_features(df)
    
    # Step 3: Preprocessing
    print("\nPreprocessing data...")
    detector = BotDetector()
    detector.data = df
    detector.preprocess()
    
    # Step 4: Split data
    print("\nSplitting data (70% train, 20% validation, 10% test)...")
    X_train, X_val, X_test, y_train, y_val, y_test = detector.split_data()
    
    print(f"Training set:   {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set:       {len(X_test)} samples")
    
    # Step 5: Handle class imbalance (optional)
    if use_smote:
        print("\nApplying SMOTE for class balancing...")
        X_train, y_train = detector.handle_imbalance(X_train, y_train, method='smote')
        print(f"After SMOTE: {len(X_train)} training samples")
    
    # Step 6: Feature scaling (optional, for logistic regression/SVM)
    if use_scaling:
        print("\nApplying feature scaling...")
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
    parser = argparse.ArgumentParser(description='Bot Detection Pipeline')
    parser.add_argument(
        '--data', '-d',
        type=str,
        default='TwiBot-20_sample.json',
        help='Path to TwiBot-20 JSON file'
    )
    parser.add_argument(
        '--labels', '-l',
        type=str,
        default=None,
        help='Path to labels CSV file (with ID and label columns)'
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
        json_path=args.data,
        label_path=args.labels,
        model_type=args.model,
        use_smote=args.smote,
        use_scaling=args.scale,
        num_features=args.features
    )


if __name__ == '__main__':
    main()
