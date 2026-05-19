import pandas as pd
import numpy as np
from typing import Optional, List


def derive_reference_date(train_df: pd.DataFrame) -> Optional[pd.Timestamp]:
    """Train-max creation date — age features must not use val/test future dates."""
    if 'account_creation_date' not in train_df.columns:
        return None
    account_creation = pd.to_datetime(
        train_df['account_creation_date'],
        errors='coerce'
    )
    ref_date = account_creation.max()
    return None if pd.isna(ref_date) else ref_date


class BotFeatureExtractor:
    """Interpretable behavioural/metadata features (no embeddings) for bot detection."""

    # Caps stop a few extreme accounts dominating linear models and rate features.
    FOLLOWERS_FRIENDS_RATIO_CAP = 1000
    TWEETS_PER_DAY_CAP = 1000
    FAVOURITES_PER_DAY_CAP = 1000
    FOLLOWERS_PER_DAY_CAP = 10000

    def __init__(self, reference_date: Optional[pd.Timestamp] = None):
        self.feature_names: List[str] = []
        # Same anchor for all splits — set by caller from train (or combined under time-split).
        self.reference_date = reference_date

    def extract_account_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract account-related features."""
        df = df.copy()

        # Account age
        if 'account_creation_date' in df.columns:
            account_creation = pd.to_datetime(df['account_creation_date'], errors='coerce')
            # Per-row max() would leak val/test timing into train features.
            reference_date = (
                self.reference_date
                if self.reference_date is not None
                else account_creation.max()
            )
            df['account_age_days'] = (reference_date - account_creation).dt.days
            self.feature_names.append('account_age_days')

        # Verification status
        if 'is_verified' in df.columns:
            df['is_verified'] = df['is_verified'].fillna(0).astype(int)
            self.feature_names.append('is_verified')

        return df

    def _safe_to_numeric(self, series: pd.Series, default: float = 0) -> pd.Series:
        """Safely convert series to numeric, handling strings with spaces."""
        # Strip whitespace from string values
        if series.dtype == object:
            series = series.astype(str).str.strip()
        result = pd.to_numeric(series, errors='coerce').fillna(default)
        # Replace infinity with a large but finite value
        return result.replace([np.inf, -np.inf], default)

    def extract_activity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract activity-related features from user statistics."""
        df = df.copy()

        numeric_columns = [
            'followers_count',
            'friends_count',
            'listed_count',
            'statuses_count',
            'favourites_count',
        ]
        for column in numeric_columns:
            if column in df.columns:
                df[column] = self._safe_to_numeric(df[column])
                self.feature_names.append(column)

        # Classic bot heuristic: high followers/friends ratio (literature-backed signal).
        if {'followers_count', 'friends_count'}.issubset(df.columns):
            # Avoid division by zero and cap extreme values
            ratio = df['followers_count'] / (df['friends_count'] + 1)
            # Cap at a reasonable maximum to limit outliers
            df['followers_to_friends_ratio'] = ratio.clip(
                upper=self.FOLLOWERS_FRIENDS_RATIO_CAP
            )
            self.feature_names.append('followers_to_friends_ratio')

        return df

    def extract_profile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract profile-related features (bot indicators)."""
        df = df.copy()

        boolean_columns = [
            'default_profile',
            'default_profile_image',
            'has_extended_profile',
            'geo_enabled',
            'protected',
        ]
        for column in boolean_columns:
            if column in df.columns:
                df[column] = df[column].fillna(0).astype(int)
                self.feature_names.append(column)

        return df

    def extract_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract text-based features from description."""
        df = df.copy()

        if 'description' in df.columns:
            # Description length
            description = df['description'].fillna('')
            df['description_length'] = description.str.len()
            # Has description
            df['has_description'] = description.ne('').astype(int)
            self.feature_names.extend(['description_length', 'has_description'])

        # Has URL in profile
        if 'url' in df.columns:
            df['has_url'] = df['url'].notna().astype(int)
            self.feature_names.append('has_url')

        # Screen name length (bots often have random long names)
        if 'screen_name' in df.columns:
            screen_name = df['screen_name'].fillna('')
            df['screen_name_length'] = screen_name.str.len()
            # Has digits in screen name (common for bots)
            df['screen_name_has_digits'] = screen_name.str.contains(
                r'\d',
                regex=True
            ).astype(int)
            self.feature_names.extend(
                ['screen_name_length', 'screen_name_has_digits']
            )

        return df

    def extract_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract derived/computed features."""
        df = df.copy()

        # Ensure account_age_days is numeric and handle negative/zero values
        if 'account_age_days' in df.columns:
            df['account_age_days'] = self._safe_to_numeric(df['account_age_days'], default=1)
            # Ensure minimum of 1 day to avoid division issues
            df['account_age_days'] = df['account_age_days'].clip(lower=1)

        if 'account_age_days' in df.columns:
            rate_specs = [
                ('statuses_count', 'tweets_per_day', self.TWEETS_PER_DAY_CAP),
                ('favourites_count', 'favourites_per_day', self.FAVOURITES_PER_DAY_CAP),
                ('followers_count', 'followers_per_day', self.FOLLOWERS_PER_DAY_CAP),
            ]
            for count_col, feature_name, cap in rate_specs:
                if count_col in df.columns:
                    rate = (df[count_col] / df['account_age_days']).clip(upper=cap)
                    df[feature_name] = rate.replace([np.inf, -np.inf], 0)
                    self.feature_names.append(feature_name)

        return df

    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all features in one call."""
        self.feature_names = []  # Reset feature names

        # Order matters: derived rates need account_age_days from earlier steps.
        extractors = (
            self.extract_account_features,
            self.extract_activity_features,
            self.extract_profile_features,
            self.extract_text_features,
            self.extract_derived_features,
        )
        for extractor in extractors:
            df = extractor(df)

        # Remove duplicates from feature names
        self.feature_names = list(dict.fromkeys(self.feature_names))

        return df

    def get_feature_names(self) -> List[str]:
        """Return list of extracted feature names."""
        return self.feature_names.copy()
