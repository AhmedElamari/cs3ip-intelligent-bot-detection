import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


class TwiBotDataLoader:
    """Load and flatten TwiBot-20 JSON dataset into a pandas DataFrame."""

    # Mapping from TwiBot-20 JSON fields to expected column names
    FIELD_MAPPING = {
        'created_at': 'account_creation_date',
        'verified': 'is_verified',
    }

    # Profile fields to extract (flat numeric/boolean features)
    PROFILE_FIELDS = [
        'id', 'id_str', 'name', 'screen_name', 'location', 'description',
        'url', 'protected', 'followers_count', 'friends_count', 'listed_count',
        'created_at', 'favourites_count', 'geo_enabled', 'verified',
        'statuses_count', 'default_profile', 'default_profile_image',
        'has_extended_profile'
    ]

    def __init__(self, json_path: str, label_path: Optional[str] = None):
        """
        Initialize the data loader.

        Args:
            json_path: Path to the TwiBot-20 JSON file
            label_path: Optional path to a separate labels file (CSV with ID, label columns)
        """
        self.json_path = Path(json_path)
        self.label_path = Path(label_path) if label_path else None
        self.raw_data = None
        self.labels = None

    def load_json(self) -> list:
        """Load raw JSON data from file."""
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        return self.raw_data

    def load_labels(self) -> Optional[pd.DataFrame]:
        """Load labels from a separate file if provided."""
        if self.label_path and self.label_path.exists():
            self.labels = pd.read_csv(self.label_path)
            return self.labels
        return None

    def _clean_string(self, value) -> str:
        """Clean string values by stripping whitespace."""
        if isinstance(value, str):
            return value.strip()
        return value

    def _parse_twitter_date(self, date_str: str) -> Optional[pd.Timestamp]:
        """Parse Twitter's date format to pandas Timestamp."""
        if not date_str or pd.isna(date_str):
            return None
        try:
            # Twitter format: "Wed Oct 10 20:19:24 +0000 2018"
            return pd.to_datetime(date_str, format='%a %b %d %H:%M:%S %z %Y')
        except (ValueError, TypeError):
            try:
                # Fallback to flexible parsing
                return pd.to_datetime(date_str)
            except:
                return None

    def _flatten_user(self, user: dict) -> dict:
        """Flatten a single user record from nested JSON to flat dict."""
        flat = {}

        # Extract top-level ID
        flat['user_id'] = self._clean_string(user.get('ID', ''))

        # Extract profile fields
        profile = user.get('profile', {}) or {}
        for field in self.PROFILE_FIELDS:
            value = profile.get(field)
            # Apply field name mapping
            target_field = self.FIELD_MAPPING.get(field, field)
            flat[target_field] = self._clean_string(value) if isinstance(value, str) else value

        # Extract domain (first domain if list)
        domain = user.get('domain', [])
        flat['domain'] = domain[0] if domain else None

        # Extract tweet count from tweet list
        tweets = user.get('tweet', [])
        flat['tweet_count'] = len(tweets) if tweets else 0

        # Extract neighbor counts
        neighbor = user.get('neighbor', {}) or {}
        flat['following_sample_count'] = len(neighbor.get('following', []))
        flat['follower_sample_count'] = len(neighbor.get('follower', []))

        return flat

    def flatten_to_dataframe(self) -> pd.DataFrame:
        """Convert nested JSON data to a flat pandas DataFrame."""
        if self.raw_data is None:
            self.load_json()

        flattened = [self._flatten_user(user) for user in self.raw_data]
        df = pd.DataFrame(flattened)

        # Parse date column
        if 'account_creation_date' in df.columns:
            df['account_creation_date'] = df['account_creation_date'].apply(self._parse_twitter_date)

        # Convert boolean fields to int (handle string 'True'/'False' and actual booleans)
        bool_columns = ['is_verified', 'protected', 'geo_enabled', 
                        'default_profile', 'default_profile_image', 'has_extended_profile']
        for col in bool_columns:
            if col in df.columns:
                # Handle string booleans like 'True', 'False' and actual booleans
                df[col] = df[col].apply(lambda x: 
                    1 if x is True or x == 'True' or x == 'true' or x == 1 
                    else 0 if x is False or x == 'False' or x == 'false' or x == 0 or pd.isna(x) 
                    else int(bool(x))
                )

        # Merge labels if available
        if self.labels is not None:
            # Try to match on ID column
            id_col = 'ID' if 'ID' in self.labels.columns else 'id'
            if id_col in self.labels.columns:
                df = df.merge(
                    self.labels[[id_col, 'label']].rename(columns={id_col: 'user_id'}),
                    on='user_id',
                    how='left'
                )
        elif self.label_path:
            self.load_labels()
            if self.labels is not None:
                id_col = 'ID' if 'ID' in self.labels.columns else 'id'
                if id_col in self.labels.columns:
                    df = df.merge(
                        self.labels[[id_col, 'label']].rename(columns={id_col: 'user_id'}),
                        on='user_id',
                        how='left'
                    )

        return df

    def load(self) -> pd.DataFrame:
        """Main entry point: load and return flattened DataFrame."""
        self.load_json()
        if self.label_path:
            self.load_labels()
        return self.flatten_to_dataframe()


def load_twibot_json(json_path: str, label_path: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to load TwiBot-20 JSON data.

    Args:
        json_path: Path to TwiBot-20 JSON file
        label_path: Optional path to labels CSV file

    Returns:
        Flattened pandas DataFrame ready for preprocessing
    """
    loader = TwiBotDataLoader(json_path, label_path)
    return loader.load()


if __name__ == '__main__':
    # Example usage
    import sys
    
    json_file = sys.argv[1] if len(sys.argv) > 1 else 'TwiBot-20_sample.json'
    label_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    df = load_twibot_json(json_file, label_file)
    print(f"Loaded {len(df)} records with columns:")
    print(df.columns.tolist())
    print(f"\nSample data:")
    print(df.head())
    print(f"\nData types:")
    print(df.dtypes)
