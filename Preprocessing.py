import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif


class BotDetector:
    def __init__(self):
        """Initialize preprocessing helpers."""
        self.data = None
        self.scaler = StandardScaler()
        self.selected_features = None
        self.feature_selector = None

    def preprocess(self) -> pd.DataFrame:
        """Clean and preprocess data from TwiBot-20"""
        self.data = self.data.dropna(subset=['label'])
        # Fill NaN values - use infer_objects() to avoid FutureWarning about downcasting
        self.data = self.data.fillna(0).infer_objects(copy=False)
        
        #normalise the text fields
        if 'description' in self.data.columns:
            self.data['description'] = self.data['description'].str.lower()
        return self.data

    def scale_features(self, X_train, X_val, X_test):
        """Apply StandardScaler for Logistic Regression and SVM models"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_val_scaled, X_test_scaled

    def handle_imbalance(self, X_train, y_train, method: str = 'smote'):
        """Handle class imbalance in bot detection datasets.
        
        Args:
            X_train: Training features
            y_train: Training labels
            method: Resampling method ('smote' or 'undersample')
            
        Returns:
            Resampled X_train and y_train, or original data if imblearn unavailable
        """
        try:
            if method == 'smote':
                from imblearn.over_sampling import SMOTE
                sampler = SMOTE(random_state=2112)
            elif method == 'undersample':
                from imblearn.under_sampling import RandomUnderSampler
                sampler = RandomUnderSampler(random_state=2112)
            else:
                return X_train, y_train
            return sampler.fit_resample(X_train, y_train)
        except ImportError:
            print("imblearn not installed. Install with: pip install imbalanced-learn")
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
