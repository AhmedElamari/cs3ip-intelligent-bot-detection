import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif


class BotDetector:
    def __init__(self, data_source: Union[str, pd.DataFrame, None] = None):
        """
        Initialize BotDetector.
        
        Args:
            data_source: Path to CSV/JSON file, or a pandas DataFrame directly
        """
        self.data_source = data_source
        self.data = None
        self.model = None
        self.scaler = StandardScaler()
        self.selected_features = None
        self.feature_selector = None

    def load_data(self, data_source: Union[str, pd.DataFrame, None] = None) -> pd.DataFrame:
        """
        Load data from CSV, JSON, or use provided DataFrame.
        
        Args:
            data_source: Path to CSV/JSON file, or pandas DataFrame.
                         If None, uses self.data_source from init.
        
        Returns:
            Loaded DataFrame
        """
        source = data_source or self.data_source
        
        if source is None:
            raise ValueError("No data source provided. Pass a file path or DataFrame.")
        
        if isinstance(source, pd.DataFrame):
            self.data = source.copy()
        elif isinstance(source, (str, Path)):
            path = Path(source)
            if path.suffix.lower() == '.json':
                # Use TwiBotDataLoader for JSON files
                from DataLoader import load_twibot_json
                self.data = load_twibot_json(str(path))
            elif path.suffix.lower() == '.csv':
                self.data = pd.read_csv(path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        else:
            raise TypeError(f"Unsupported data source type: {type(source)}")
        
        return self.data

    def preprocess(self) -> pd.DataFrame:
        """Clean and preprocess data from TwiBot-20"""
        self.data = self.data.dropna(subset=['label'])
        # Fill NaN values - use infer_objects() to avoid FutureWarning about downcasting
        self.data = self.data.fillna(0).infer_objects(copy=False)
        
        #normalise the text fields
        if 'description' in self.data.columns:
            self.data['description'] = self.data['description'].str.lower()
        return self.data

    def split_data(self, test_size: float = 0.1, val_size: float = 0.2):
        """Will split into train/val/test sets (70:20:10 ratio)"""
        X = self.data.drop('label', axis=1)
        # Ensure that only numeric features are passed to the model
        X = X.select_dtypes(include=[np.number])
        y = self.data['label']
        # First split: to separate test set (10%)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=2112
        )
        # Second split: will separate validation set from training set (20% of original = 2/9 of remaining)
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=2112
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def scale_features(self, X_train, X_val, X_test):
        """Apply StandardScaler for Logistic Regression and SVM models"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_val_scaled, X_test_scaled

    def handle_imbalance(self, X_train, y_train, method: str = 'smote'):
        """Handle class imbalance in bot detection datasets"""
        if method == 'smote':
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=2112)
                X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
                return X_resampled, y_resampled
            except ImportError:
                print("imblearn not installed.  Install with: pip install imbalanced-learn")
                return X_train, y_train
        elif method == 'undersample':
            try:
                from imblearn.under_sampling import RandomUnderSampler
                rus = RandomUnderSampler(random_state=2112)
                X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
                return X_resampled, y_resampled
            except ImportError:
                print("imblearn not installed. Install with: pip install imbalanced-learn")
                return X_train, y_train
        return X_train, y_train

    def select_features(self, X_train, y_train, k: int = 20):
        """Select top k features using mutual information"""
        # Ensure k does not exceed number of features
        k = min(k, X_train.shape[1])
        self.feature_selector = SelectKBest(mutual_info_classif, k=k)
        X_selected = self.feature_selector.fit_transform(X_train, y_train)
        self.selected_features = self.feature_selector.get_support(indices=True)
        return X_selected

    def apply_feature_selection(self, X):
        """Apply previously fitted feature selection to new data"""
        if self.feature_selector is None:
            raise ValueError("Feature selector not fitted. Call select_features first.")
        return self.feature_selector.transform(X)

    def get_class_weights(self, y_train) -> dict:
        """Calculate class weights for handling imbalance in model training"""
        y_int = y_train.astype(int)
        class_counts = np.bincount(y_int)
        total = len(y_int)
        if total == 0:
            raise ValueError("Cannot compute class weights: training labels are empty.")

        unique_classes = np.unique(y_int)

        # For binary classification problems with labels 0 and 1, explicitly ensure both are present.
        if set(unique_classes).issubset({0, 1}):
            expected = {0, 1}
            missing_binary = sorted(expected - set(unique_classes))
            if missing_binary:
                raise ValueError(
                    f"Cannot compute class weights: no samples found for classes {missing_binary}. "
                    "Adjust the train/validation split or resampling strategy to include all classes."
                )
        else:
            # For non-binary or non-0/1 label schemes, fall back to checking counts from np.bincount.
            missing_classes = [i for i, count in enumerate(class_counts) if count == 0]
            if missing_classes:
                raise ValueError(
                    f"Cannot compute class weights: no samples found for classes {missing_classes}. "
                    "Adjust the train/validation split or resampling strategy to include all classes."
                )
        weights = {i: total / (len(class_counts) * count) for i, count in enumerate(class_counts)}
        return weights
