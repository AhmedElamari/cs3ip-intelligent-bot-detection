import json
import numbers
import pandas as pd
from pathlib import Path
from typing import Optional, List, Union


__all__ = [
    "TwiBotDataLoader",
    "load_twibot_splits_as_dict",
]


class TwiBotDataLoader:
    """Load and flatten TwiBot-20 JSON into tabular rows for sklearn pipelines.

    Nested profile/neighbor/tweet JSON is flattened so downstream stages see
    interpretable behavioural/metadata columns (not graph embeddings).
    """

    TWITTER_DATE_FORMAT = "%a %b %d %H:%M:%S %z %Y"

    # Align TwiBot field names with pipeline time-split / age features.
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
    BOOL_COLUMNS = (
        "is_verified",
        "protected",
        "geo_enabled",
        "default_profile",
        "default_profile_image",
        "has_extended_profile",
    )

    def __init__(
        self,
        json_path: Optional[Union[str, Path]] = None,
        json_paths: Optional[List[Union[str, Path]]] = None
    ):
        """
        Initialize the data loader.

        Args:
            json_path: Path to a single TwiBot-20 JSON file
            json_paths: Optional list of JSON file paths to load and combine (for train/dev/test splits)
        """
        if json_paths and json_path:
            raise ValueError("Provide either json_path or json_paths, not both.")
        if json_paths:
            self.json_paths = [Path(p) for p in json_paths]
            self.json_path = None
        elif json_path:
            self.json_path = Path(json_path)
            self.json_paths = None
        else:
            raise ValueError("Either json_path or json_paths must be provided")
        
        self.raw_data = None

    def load_json(self) -> list:
        """Load raw JSON data from file(s)."""
        if self.json_paths:
            # Load and combine multiple files
            self.raw_data = []
            for path in self.json_paths:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.raw_data.extend(data)
        else:
            # Single file
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.raw_data = json.load(f)
        return self.raw_data

    def _clean_string(self, value) -> str:
        """Clean string values by stripping whitespace."""
        if isinstance(value, str):
            return value.strip()
        return value

    def _normalize_id_series(self, series: pd.Series) -> pd.Series:
        """Normalize ID values for consistent comparisons."""
        return series.astype(str).str.strip()

    # Normalises all different types of labels to be consistently 0 or 1.
    def _normalize_embedded_label(self, value) -> Optional[int]:
        """Normalize a single embedded label value (string/number) to 0/1."""
        if value is None or pd.isna(value):
            return None
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in ("0", "1"):
                return int(normalized)
            if normalized in ("bot", "fake"):
                return 1
            if normalized in ("human", "real"):
                return 0
            numeric = pd.to_numeric(normalized, errors="coerce")
            if numeric in (0, 1):
                return int(numeric)
            return None

        if isinstance(value, numbers.Integral):
            return int(value) if value in (0, 1) else None
        if isinstance(value, numbers.Real):
            if value in (0, 1) and float(value).is_integer():
                return int(value)
            return None
        return None

    def _parse_twitter_date(self, date_str: str) -> pd.Timestamp:
        """Parse Twitter's date format to pandas Timestamp."""
        if not date_str or pd.isna(date_str):
            return pd.NaT

        # Fast path: strict Twitter format
        parsed = pd.to_datetime(
            date_str,
            format=self.TWITTER_DATE_FORMAT,
            errors="coerce",
        )
        if not pd.isna(parsed):
            return parsed

        # Fallback: flexible parsing
        return pd.to_datetime(date_str, errors="coerce")

    @staticmethod
    def _to_int_bool(value) -> int:
        """Convert a bool-ish value to 0/1, matching TwiBot-20's mixed typing."""
        if value is True or value == 1:
            return 1
        if value is False or value == 0 or pd.isna(value):
            return 0

        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized == "true":
                return 1
            if normalized == "false":
                return 0
            # TwiBot JSON can encode booleans as numeric strings ("0"/"1").
            numeric = pd.to_numeric(normalized, errors="coerce")
            if numeric in (0, 1):
                return int(numeric)
            # Preserve Python truthiness for arbitrary strings ("0" handled above)
            return 1 if normalized else 0

        return int(bool(value))

    def _flatten_user(self, user: dict) -> dict:
        """Flatten a single user record from nested JSON to flat dict."""
        flat = {}

        # Extract top-level ID
        flat['user_id'] = self._clean_string(user.get('ID', ''))

        # Embedded label (some TwiBot-20 variants include it directly)
        label = self._normalize_embedded_label(user.get("label"))
        if label is not None:
            flat["label"] = label

        # Extract profile fields
        profile = user.get('profile', {}) or {}
        for field in self.PROFILE_FIELDS:
            value = profile.get(field)
            # Apply field name mapping
            target_field = self.FIELD_MAPPING.get(field, field)
            flat[target_field] = self._clean_string(value) if isinstance(value, str) else value

        # Extract domain (first domain if list)
        domain = user.get('domain', [])
        if isinstance(domain, list):
            flat['domain'] = domain[0] if domain else None
        else:
            flat['domain'] = domain

        # Compress graph structure to counts — interpretable, no GNN in this pipeline.
        tweets = user.get('tweet') or []
        flat['tweet_count'] = len(tweets) if tweets else 0

        neighbor = user.get('neighbor') or {}
        flat['following_sample_count'] = len(neighbor.get('following', []) or [])
        flat['follower_sample_count'] = len(neighbor.get('follower', []) or [])

        return flat

    def flatten_to_dataframe(self) -> pd.DataFrame:
        """Convert nested JSON data to a flat pandas DataFrame."""
        if self.raw_data is None:
            self.load_json()

        flattened = [self._flatten_user(user) for user in self.raw_data]
        df = pd.DataFrame(flattened)
        if 'user_id' in df.columns:
            df['user_id'] = self._normalize_id_series(df['user_id'])

        # Parse date column
        if 'account_creation_date' in df.columns:
            df['account_creation_date'] = df['account_creation_date'].apply(self._parse_twitter_date)

        # Convert boolean-ish fields to int (handle strings + mixed typing)
        for col in self.BOOL_COLUMNS:
            if col in df.columns:
                df[col] = df[col].apply(self._to_int_bool)

        return df

    def load(self) -> pd.DataFrame:
        """Main entry point: load and return flattened DataFrame."""
        self.load_json()
        return self.flatten_to_dataframe()


def load_twibot_splits_as_dict(
    data_dir: Union[str, Path] = 'data'
) -> dict:
    """
    Load TwiBot-20 data splits as separate DataFrames.
    
    This preserves the original train/dev/test split design, which typically
    results in better model performance than re-splitting combined data.

    Args:
        data_dir: Path to directory containing train.json, dev.json, test.json

    Returns:
        Dictionary with keys 'train', 'val', 'test' mapping to DataFrames
    """
    data_dir = Path(data_dir)
    
    split_files = {
        'train': data_dir / 'train.json',
        'val': data_dir / 'dev.json',  # TwiBot-20 uses dev.json; we standardize to 'val'
        'test': data_dir / 'test.json'
    }
    
    # Verify all files exist
    missing = [name for name, path in split_files.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing split files: {missing}. "
            f"Expected train.json, dev.json, test.json in {data_dir}"
        )
    
    splits = {}
    for split_name, split_path in split_files.items():
        loader = TwiBotDataLoader(json_path=split_path)
        splits[split_name] = loader.load()
    
    return splits


