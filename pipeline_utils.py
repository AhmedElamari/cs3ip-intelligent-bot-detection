import logging
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger(__name__)


def safe_stratified_split(
    indices,
    labels,
    test_size: float,
    random_state: int,
    split_name: str
):
    """Split data with stratification fallback when labels are too sparse."""
    try:
        return train_test_split(
            indices,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )
    except ValueError as exc:
        LOGGER.warning(
            "Stratified %s split failed (%s). Falling back to unstratified split.",
            split_name,
            exc
        )
        return train_test_split(
            indices,
            labels,
            test_size=test_size,
            random_state=random_state
        )


def time_stratified_split(
    df: pd.DataFrame,
    time_column: str = 'account_creation_date',
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically to simulate data drift scenarios.
    
    This function splits data based on temporal ordering, ensuring that:
    - Training data contains the earliest samples
    - Validation data contains middle-period samples  
    - Test data contains the most recent samples
    
    This simulates real-world ML deployment where models are trained on
    historical data and must generalize to future (unseen) data distributions.
    
    Args:
        df: DataFrame containing the data to split
        time_column: Name of the datetime column to use for ordering
        train_size: Proportion of data for training (default: 0.7)
        val_size: Proportion of data for validation (default: 0.15)
        test_size: Proportion of data for testing (default: 0.15)
        
    Returns:
        Tuple of (train_df, val_df, test_df) DataFrames
        
    Raises:
        ValueError: If time_column is not present or split sizes don't sum to 1.0
        
    Example:
        >>> train_df, val_df, test_df = time_stratified_split(
        ...     df, 
        ...     time_column='account_creation_date',
        ...     train_size=0.7,
        ...     val_size=0.15,
        ...     test_size=0.15
        ... )
    """
    # Validate split proportions
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0, rtol=1e-5):
        raise ValueError(
            f"Split sizes must sum to 1.0, got {total:.4f} "
            f"(train={train_size}, val={val_size}, test={test_size})"
        )
    
    if time_column not in df.columns:
        raise ValueError(
            f"Time column '{time_column}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Convert time column to datetime if needed
    df = df.copy()
    df[time_column] = pd.to_datetime(df[time_column], errors='coerce')
    
    # Handle missing timestamps - place them at the beginning (oldest)
    missing_time_mask = df[time_column].isna()
    if missing_time_mask.any():
        n_missing = missing_time_mask.sum()
        LOGGER.warning(
            "Found %d samples with missing timestamps in '%s'. "
            "These will be treated as the oldest samples and placed in training data.",
            n_missing,
            time_column
        )
    
    # Sort by time (NaT values will be at the beginning)
    df_sorted = df.sort_values(by=time_column, na_position='first').reset_index(drop=True)
    
    n_samples = len(df_sorted)
    train_end = int(n_samples * train_size)
    val_end = train_end + int(n_samples * val_size)
    
    train_df = df_sorted.iloc[:train_end].copy()
    val_df = df_sorted.iloc[train_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()
    
    LOGGER.info(
        "Time-stratified split complete: train=%d (%.1f%%), val=%d (%.1f%%), test=%d (%.1f%%)",
        len(train_df), len(train_df) / n_samples * 100,
        len(val_df), len(val_df) / n_samples * 100,
        len(test_df), len(test_df) / n_samples * 100
    )
    
    # Log time ranges for each split
    for name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        valid_times = split_df[time_column].dropna()
        if len(valid_times) > 0:
            LOGGER.info(
                "%s split time range: %s to %s",
                name,
                valid_times.min().strftime('%Y-%m-%d') if not pd.isna(valid_times.min()) else 'N/A',
                valid_times.max().strftime('%Y-%m-%d') if not pd.isna(valid_times.max()) else 'N/A'
            )
    
    return train_df, val_df, test_df


def load_combined_data_with_time_split(
    data_dir,
    time_column: str = 'account_creation_date',
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> dict:
    """
    Load all TwiBot-20 split files, combine them, and re-split chronologically.
    
    This function is useful for simulating data drift scenarios where you want
    to test how well a model trained on older data generalizes to newer data.
    
    Args:
        data_dir: Path to directory containing train.json, dev.json, test.json
        time_column: Name of the datetime column to use for ordering
        train_size: Proportion of data for training (default: 0.7)
        val_size: Proportion of data for validation (default: 0.15)
        test_size: Proportion of data for testing (default: 0.15)
        
    Returns:
        Dictionary with keys 'train', 'val', 'test' mapping to DataFrames
    """
    from DataLoader import TwiBotDataLoader
    from pathlib import Path
    
    data_dir = Path(data_dir)
    
    # Load and combine all splits
    split_files = [
        data_dir / 'train.json',
        data_dir / 'dev.json',
        data_dir / 'test.json'
    ]
    
    existing_files = [f for f in split_files if f.exists()]
    if not existing_files:
        raise FileNotFoundError(
            f"No split files found in {data_dir}. "
            f"Expected train.json, dev.json, or test.json"
        )
    
    loader = TwiBotDataLoader(json_paths=existing_files)
    combined_df = loader.load()
    
    LOGGER.info(
        "Loaded %d total samples from %d files for time-stratified splitting",
        len(combined_df),
        len(existing_files)
    )
    
    # Apply time-stratified split
    train_df, val_df, test_df = time_stratified_split(
        combined_df,
        time_column=time_column,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size
    )
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }
