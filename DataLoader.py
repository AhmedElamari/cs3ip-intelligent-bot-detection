import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Union


class TwiBotDataLoader:
    """Load and flatten TwiBot-20 JSON dataset into a pandas DataFrame.
    
    Supports both:
    - Single JSON file with optional separate labels CSV
    - Multiple JSON files (train/dev/test splits) with embedded labels
    """

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
    LABEL_ID_CANDIDATES = ('ID', 'id', 'user_id')

    def __init__(
        self,
        json_path: Optional[Union[str, Path]] = None,
        label_path: Optional[str] = None,
        json_paths: Optional[List[Union[str, Path]]] = None
    ):
        """
        Initialize the data loader.

        Args:
            json_path: Path to a single TwiBot-20 JSON file
            label_path: Optional path to a separate labels file (CSV with ID/id/user_id and label columns)
            json_paths: Optional list of JSON file paths to load and combine (for train/dev/test splits)
        """
        if json_paths:
            self.json_paths = [Path(p) for p in json_paths]
            self.json_path = None
        elif json_path:
            self.json_path = Path(json_path)
            self.json_paths = None
        else:
            raise ValueError("Either json_path or json_paths must be provided")
        
        self.label_path = Path(label_path) if label_path else None
        self.raw_data = None
        self.labels = None

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

    def load_labels(self) -> Optional[pd.DataFrame]:
        """Load labels from a separate file if provided."""
        if not self.label_path:
            return None
        if not self.label_path.exists():
            raise FileNotFoundError(
                f"Labels file not found: {self.label_path}. "
                "Provide a valid path with --labels."
            )
        self.labels = pd.read_csv(self.label_path)
        if self.labels.empty:
            raise ValueError(f"Labels file is empty: {self.label_path}")
        if 'label' not in self.labels.columns:
            raise ValueError(
                "Labels file must include a 'label' column with 0/1 values."
            )
        id_col = self._get_label_id_column(self.labels)
        self.labels = self.labels[[id_col, 'label']].copy()
        self.labels[id_col] = self._normalize_id_series(self.labels[id_col])
        self.labels['label'] = self._normalize_label_values(self.labels['label'])
        self._validate_label_values(self.labels['label'])
        return self.labels

    def _clean_string(self, value) -> str:
        """Clean string values by stripping whitespace."""
        if isinstance(value, str):
            return value.strip()
        return value

    def _get_label_id_column(self, labels: pd.DataFrame) -> str:
        """Find the ID column in a labels DataFrame."""
        for candidate in self.LABEL_ID_CANDIDATES:
            if candidate in labels.columns:
                return candidate
        raise ValueError(
            "Labels file must include one of these ID columns: "
            f"{', '.join(self.LABEL_ID_CANDIDATES)}"
        )

    def _normalize_id_series(self, series: pd.Series) -> pd.Series:
        """Normalize ID values for reliable merging."""
        return series.astype(str).str.strip()

    def _normalize_label_values(self, series: pd.Series) -> pd.Series:
        """Normalize label values to numeric 0/1 where possible."""
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(series, errors='coerce')
        normalized = series.astype(str).str.strip().str.lower()
        label_map = {
            'bot': 1,
            'human': 0,
            'fake': 1,
            'real': 0,
        }
        mapped = normalized.map(label_map)
        numeric = pd.to_numeric(normalized, errors='coerce')
        return mapped.fillna(numeric)

    def _validate_label_values(self, series: pd.Series) -> None:
        """Ensure labels are binary after normalization."""
        valid_values = set(series.dropna().unique())
        if not valid_values:
            raise ValueError(
                "Labels file has no valid label values after normalization."
            )
        # Accept both integer and float representations of 0 and 1 (e.g., 0, 1, 0.0, 1.0)
        if not all((v == 0) or (v == 1) for v in valid_values):
            raise ValueError(
                "Labels must be binary (0/1) after normalization."
            )

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

        # Extract embedded label if present (TwiBot-20 format stores as string '0' or '1')
        if 'label' in user:
            label_val = user.get('label')
            if label_val is not None:
                # Convert string labels to int
                if isinstance(label_val, str):
                    label_val = label_val.strip()
                    if label_val in ('0', '1'):
                        flat['label'] = int(label_val)
                    elif label_val.lower() in ('bot', 'human'):
                        flat['label'] = 1 if label_val.lower() == 'bot' else 0
                else:
                    flat['label'] = int(label_val) if label_val in (0, 1) else None

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

        # Extract tweet count from tweet list
        tweets = user.get('tweet', [])
        flat['tweet_count'] = len(tweets) if tweets else 0

        # Extract neighbor counts
        neighbor = user.get('neighbor', {}) or {}
        if neighbor is None:
            neighbor = {}
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

        # Check if labels were embedded and extracted during flattening
        has_embedded_labels = 'label' in df.columns and df['label'].notna().any()

        # Merge external labels if available and embedded labels not present/complete
        if not has_embedded_labels:
            if self.labels is not None:
                df = self._merge_labels(df)
            elif self.label_path:
                self.load_labels()
                if self.labels is not None:
                    df = self._merge_labels(df)

        return df

    def _merge_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge labels into the flattened DataFrame."""
        id_col = self._get_label_id_column(self.labels)
        labels = self.labels[[id_col, 'label']].copy()
        df = df.merge(
            labels.rename(columns={id_col: 'user_id'}),
            on='user_id',
            how='left'
        )
        if df['label'].notna().sum() == 0:
            raise ValueError(
                "Label merge produced zero matches. Check the ID column and format."
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
        label_path: Optional path to labels CSV file (ID/id/user_id and label)

    Returns:
        Flattened pandas DataFrame ready for preprocessing
    """
    loader = TwiBotDataLoader(json_path, label_path)
    return loader.load()


def load_twibot_splits_as_dict(
    data_dir: Union[str, Path] = 'data',
    label_path: Optional[str] = None
) -> dict:
    """
    Load TwiBot-20 data splits as separate DataFrames (pyi-style).
    
    This preserves the original train/dev/test split design, which typically
    results in better model performance than re-splitting combined data.

    Args:
        data_dir: Path to directory containing train.json, dev.json, test.json
        label_path: Optional path to external labels CSV

    Returns:
        Dictionary with keys 'train', 'val', 'test' mapping to DataFrames
    """
    data_dir = Path(data_dir)
    
    split_files = {
        'train': data_dir / 'train.json',
        'val': data_dir / 'dev.json',  # Note: dev.json maps to 'val' key
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
        loader = TwiBotDataLoader(json_path=split_path, label_path=label_path)
        splits[split_name] = loader.load()
    
    return splits


def check_twibot_data_available() -> dict:
    """
    Check what TwiBot-20 data files are available.
    
    Returns:
        Dictionary with file availability and sample counts
    """
    repo_root = Path(__file__).resolve().parent
    data_dir = repo_root / 'data'
    sample_path = repo_root / 'TwiBot-20_sample.json'
    
    result = {
        'sample_available': sample_path.exists(),
        'splits_available': {},
        'total_split_samples': 0
    }
    
    for split in ['train', 'dev', 'test']:
        split_path = data_dir / f'{split}.json'
        if split_path.exists():
            try:
                with open(split_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                count = len(data)
                has_labels = any('label' in user for user in data[:10])
            except Exception:
                count = 0
                has_labels = False
            result['splits_available'][split] = {
                'exists': True,
                'count': count,
                'has_labels': has_labels
            }
            result['total_split_samples'] += count
        else:
            result['splits_available'][split] = {
                'exists': False,
                'count': 0,
                'has_labels': False
            }
    
    return result


if __name__ == '__main__':
    import sys
    
    # Quick demo: load data from CLI arg or sample file
    json_file = sys.argv[1] if len(sys.argv) > 1 else 'TwiBot-20_sample.json'
    label_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    df = load_twibot_json(json_file, label_file)
    print(f"Loaded {len(df)} records, {df.shape[1]} columns")
    if 'label' in df.columns:
        print(f"Labels: {df['label'].value_counts().to_dict()}")
