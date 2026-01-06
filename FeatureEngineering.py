import pandas as pd
import numpy as np
from typing import Optional, List


def derive_reference_date(train_df: pd.DataFrame) -> Optional[pd.Timestamp]:
    """Derive a leakage-safe reference date from the training split."""
    if 'account_creation_date' not in train_df.columns:
        return None
    account_creation = pd.to_datetime(
        train_df['account_creation_date'],
        errors='coerce'
    )
    ref_date = account_creation.max()
    if not pd.isna(ref_date):
        return ref_date
    tzinfo = account_creation.dt.tz
    fallback = pd.Timestamp("1970-01-01")
    if tzinfo is not None:
        return fallback.tz_localize(tzinfo)
    return fallback


class BotFeatureExtractor:
    """Extract features for bot detection from TwiBot-20 data."""

    # Heuristic caps to limit extreme outliers consistently across splits.
    FOLLOWERS_FRIENDS_RATIO_CAP = 1000
    TWEETS_PER_DAY_CAP = 1000
    FAVOURITES_PER_DAY_CAP = 1000
    FOLLOWERS_PER_DAY_CAP = 10000

    def __init__(self, reference_date: Optional[pd.Timestamp] = None):
        self.feature_names: List[str] = []
        # Optional fixed reference date for reproducible, leakage-free age features
        self.reference_date = reference_date

    def extract_account_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract account-related features."""
        df = df.copy()
        
        # Account age
        if 'account_creation_date' in df.columns:
            account_creation = pd.to_datetime(df['account_creation_date'], errors='coerce')
            # Use a fixed or data-derived reference date to avoid data leakage
            if self.reference_date is not None:
                reference_date = self.reference_date
            else:
                reference_date = account_creation.max()
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
        result = result.replace([np.inf, -np.inf], default)
        return result

    def extract_activity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract activity-related features from user statistics."""
        df = df.copy()
        
        # Follower/following counts
        if 'followers_count' in df.columns:
            df['followers_count'] = self._safe_to_numeric(df['followers_count'])
            self.feature_names.append('followers_count')
        
        if 'friends_count' in df.columns:
            df['friends_count'] = self._safe_to_numeric(df['friends_count'])
            self.feature_names.append('friends_count')
        
        # Followers to friends ratio (important bot indicator)
        if 'followers_count' in df.columns and 'friends_count' in df.columns:
            # Avoid division by zero and cap extreme values
            ratio = df['followers_count'] / (df['friends_count'] + 1)
            # Cap at a reasonable maximum to limit outliers
            df['followers_to_friends_ratio'] = ratio.clip(
                upper=self.FOLLOWERS_FRIENDS_RATIO_CAP
            )
            self.feature_names.append('followers_to_friends_ratio')
        
        # Listed count (how many lists user is on - popularity indicator)
        if 'listed_count' in df.columns:
            df['listed_count'] = self._safe_to_numeric(df['listed_count'])
            self.feature_names.append('listed_count')
        
        # Statuses count (total tweets)
        if 'statuses_count' in df.columns:
            df['statuses_count'] = self._safe_to_numeric(df['statuses_count'])
            self.feature_names.append('statuses_count')
        
        # Favourites count
        if 'favourites_count' in df.columns:
            df['favourites_count'] = self._safe_to_numeric(df['favourites_count'])
            self.feature_names.append('favourites_count')
        
        return df

    def extract_profile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract profile-related features (bot indicators)."""
        df = df.copy()
        
        # Default profile (hasn't customized - common for bots)
        if 'default_profile' in df.columns:
            df['default_profile'] = df['default_profile'].fillna(0).astype(int)
            self.feature_names.append('default_profile')
        
        # Default profile image (hasn't uploaded photo - common for bots)
        if 'default_profile_image' in df.columns:
            df['default_profile_image'] = df['default_profile_image'].fillna(0).astype(int)
            self.feature_names.append('default_profile_image')
        
        # Has extended profile
        if 'has_extended_profile' in df.columns:
            df['has_extended_profile'] = df['has_extended_profile'].fillna(0).astype(int)
            self.feature_names.append('has_extended_profile')
        
        # Geo enabled
        if 'geo_enabled' in df.columns:
            df['geo_enabled'] = df['geo_enabled'].fillna(0).astype(int)
            self.feature_names.append('geo_enabled')
        
        # Protected account
        if 'protected' in df.columns:
            df['protected'] = df['protected'].fillna(0).astype(int)
            self.feature_names.append('protected')
        
        return df

    def extract_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract text-based features from description."""
        df = df.copy()
        
        if 'description' in df.columns:
            # Description length
            df['description_length'] = df['description'].fillna('').str.len()
            self.feature_names.append('description_length')
            
            # Has description
            df['has_description'] = (df['description'].fillna('').str.len() > 0).astype(int)
            self.feature_names.append('has_description')
        
        # Has URL in profile
        if 'url' in df.columns:
            df['has_url'] = df['url'].notna().astype(int)
            self.feature_names.append('has_url')
        
        # Screen name length (bots often have random long names)
        if 'screen_name' in df.columns:
            df['screen_name_length'] = df['screen_name'].fillna('').str.len()
            self.feature_names.append('screen_name_length')
            
            # Has digits in screen name (common for bots)
            df['screen_name_has_digits'] = df['screen_name'].fillna('').str.contains(r'\d', regex=True).astype(int)
            self.feature_names.append('screen_name_has_digits')
        
        return df

    def extract_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract derived/computed features."""
        df = df.copy()
        
        # Ensure account_age_days is numeric and handle negative/zero values
        if 'account_age_days' in df.columns:
            df['account_age_days'] = self._safe_to_numeric(df['account_age_days'], default=1)
            # Ensure minimum of 1 day to avoid division issues
            df['account_age_days'] = df['account_age_days'].clip(lower=1)
        
        # Tweets per day (activity rate)
        if 'statuses_count' in df.columns and 'account_age_days' in df.columns:
            tweets_per_day = df['statuses_count'] / df['account_age_days']
            df['tweets_per_day'] = tweets_per_day.clip(
                upper=self.TWEETS_PER_DAY_CAP
            ).replace([np.inf, -np.inf], 0)
            self.feature_names.append('tweets_per_day')
        
        # Favourites per day
        if 'favourites_count' in df.columns and 'account_age_days' in df.columns:
            fav_per_day = df['favourites_count'] / df['account_age_days']
            df['favourites_per_day'] = fav_per_day.clip(
                upper=self.FAVOURITES_PER_DAY_CAP
            ).replace([np.inf, -np.inf], 0)
            self.feature_names.append('favourites_per_day')
        
        # Followers per day (growth rate)
        if 'followers_count' in df.columns and 'account_age_days' in df.columns:
            followers_per_day = df['followers_count'] / df['account_age_days']
            df['followers_per_day'] = followers_per_day.clip(
                upper=self.FOLLOWERS_PER_DAY_CAP
            ).replace([np.inf, -np.inf], 0)
            self.feature_names.append('followers_per_day')
        
        return df

    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all features in one call."""
        self.feature_names = []  # Reset feature names
        
        df = self.extract_account_features(df)
        df = self.extract_activity_features(df)
        df = self.extract_profile_features(df)
        df = self.extract_text_features(df)
        df = self.extract_derived_features(df)
        
        # Remove duplicates from feature names
        self.feature_names = list(dict.fromkeys(self.feature_names))
        
        return df

    def get_feature_names(self) -> List[str]:
        """Return list of extracted feature names."""
        return self.feature_names.copy()
