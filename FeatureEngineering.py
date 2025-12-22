import pandas as pd
import numpy as np  
#from datetime import datetime

class BotFeatureExtractor:
    """Extracting features required"""

    def __init__(self, reference_date: pd.Timestamp | None = None):
        self.feature_names = []
        # Optional fixed reference date for reproducible, leakage-free age features
        self.reference_date = reference_date

    def extract_account_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract account related features"""
        account_creation = pd.to_datetime(df['account_creation_date'])
        # Use a fixed or data-derived reference date to avoid non-reproducibility and data leakage
        if self.reference_date is not None:
            reference_date = self.reference_date
        else:
            reference_date = account_creation.max()
        df['account_age_days'] = (reference_date - account_creation).dt.days
        df['is_verified'] = df['is_verified'].astype(int)
        self.feature_names.extend(['account_age_days', 'is_verified'])
        return df