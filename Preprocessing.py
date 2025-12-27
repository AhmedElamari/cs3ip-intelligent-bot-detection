import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV

class BotDetector:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self.scaler = StandardScaler()

    def load_data(self) -> pd.DataFrame:
        self.data = pd.read_csv(self.data_path)
        return self.data

    def preprocess(self) -> pd.DataFrame:
        """Clean and preprocess data from TwiBot-20"""
        self.data = self.data.dropna(subset=['label'])
        self.data = self.data.fillna(0)
        
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
   