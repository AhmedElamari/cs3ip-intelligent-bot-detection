import functools

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif


class BotDetector:
    def __init__(self, random_state: int = 2112):
        """Stateful scaler/selector — callers must fit on train, transform eval splits."""
        self.random_state = int(random_state)
        self.data = None
        self.scaler = StandardScaler()
        self.selected_features = None
        self.feature_selector = None

    def preprocess(self) -> pd.DataFrame:
        """Drop unlabeled rows; impute sparse TwiBot fields (no label invention)."""
        self.data = self.data.dropna(subset=['label'])
        num_cols = self.data.select_dtypes(include='number').columns
        self.data[num_cols] = self.data[num_cols].fillna(0)
        self.data = self.data.fillna('')
        if 'description' in self.data.columns:
            self.data['description'] = self.data['description'].str.lower()
        return self.data

    def scale_features(self, X_train, X_val, X_test):
        """Fit scaler on train only — val/test must not influence mean/variance."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_val_scaled, X_test_scaled

    def handle_imbalance(self, X_train, y_train, method: str = 'smote'):
        """Synthetic/undersampled rows for training only (never val/test)."""
        if method == 'smote':
            try:
                from imblearn.over_sampling import SMOTE
                return self._fit_resample(SMOTE, X_train, y_train)
            except ImportError:
                print("imblearn not installed.  Install with: pip install imbalanced-learn")
                return X_train, y_train
        if method == 'undersample':
            try:
                from imblearn.under_sampling import RandomUnderSampler
                return self._fit_resample(RandomUnderSampler, X_train, y_train)
            except ImportError:
                print("imblearn not installed. Install with: pip install imbalanced-learn")
                return X_train, y_train
        return X_train, y_train

    def _fit_resample(self, resampler_cls, X_train, y_train):
        """Build a resampler and apply it to training data."""
        resampler = resampler_cls(random_state=self.random_state)
        return resampler.fit_resample(X_train, y_train)

    def select_features(self, X_train, y_train, k: int = 20):
        """Mutual information — captures nonlinear relevance vs univariate F-test."""
        # Ensure k does not exceed number of features
        k = min(k, X_train.shape[1])
        score_fn = functools.partial(
            mutual_info_classif, random_state=self.random_state
        )
        self.feature_selector = SelectKBest(score_fn, k=k)
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

        unique_classes = set(np.unique(y_int))
        if unique_classes.issubset({0, 1}):
            missing_classes = sorted({0, 1} - unique_classes)
        else:
            missing_classes = [i for i, count in enumerate(class_counts) if count == 0]

        if missing_classes:
            raise ValueError(
                f"Cannot compute class weights: no samples found for classes {missing_classes}. "
                "Adjust the train/validation split or resampling strategy to include all classes."
            )

        return {
            i: total / (len(class_counts) * count)
            for i, count in enumerate(class_counts)
        }
