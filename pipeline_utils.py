import logging
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
    val_size: float = 0.2,
    test_size: float = 0.1,
    time_col: str = 'account_creation_date',
    random_state: int = 2112
) -> tuple:
    """Split data chronologically, then shuffle within each split.
    
    This approach helps combat data drift by training on older samples and
    testing on newer ones, simulating real-world deployment where models
    must generalize to future data.
    
    Args:
        df: DataFrame containing the data to split
        val_size: Fraction of data for validation (default 0.2)
        test_size: Fraction of data for test (default 0.1)
        time_col: Column name containing temporal information
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df, test_df) DataFrames
    """
    if df.empty:
        LOGGER.warning(
            "Received empty DataFrame for time-stratified split; returning empty splits."
        )
        empty = df.copy()
        return empty, empty.copy(), empty.copy()
    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1 (exclusive).")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1 (exclusive).")
    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size must be less than 1.0.")
    if time_col not in df.columns:
        LOGGER.warning(
            "Time column '%s' not found. Falling back to row order; "
            "this may not reflect chronology.",
            time_col
        )
        # Fall back to using row order as proxy for time
        df_sorted = df.reset_index(drop=True)
    else:
        time_values = pd.to_datetime(df[time_col], errors='coerce')
        missing_mask = time_values.isna()
        if missing_mask.any():
            LOGGER.warning(
                "Column '%s' contains %d missing values; treating them as oldest.",
                time_col,
                missing_mask.sum()
            )
        df_sorted = (
            df.assign(
                _time_sort=time_values,
                _original_order=range(len(df))
            )
            .sort_values(
                by=['_time_sort', '_original_order'],
                kind='mergesort',
                na_position='first'
            )
            .drop(columns=['_time_sort', '_original_order'])
            .reset_index(drop=True)
        )
    
    n = len(df_sorted)
    train_end = int(n * (1 - val_size - test_size))
    val_end = int(n * (1 - test_size))
    train_count = train_end
    val_count = val_end - train_end
    test_count = n - val_end
    if min(train_count, val_count, test_count) < 1:
        raise ValueError(
            "Time-stratified split requires at least 1 sample per split; "
            f"got train={train_count}, val={val_count}, test={test_count} from n={n}. "
            "Adjust val_size/test_size or provide more data."
        )
    
    # Split chronologically: train oldest, val middle, test newest
    train_df = (
        df_sorted.iloc[:train_end]
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )
    val_df = (
        df_sorted.iloc[train_end:val_end]
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )
    test_df = (
        df_sorted.iloc[val_end:]
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )
    
    LOGGER.info(
        "Time-stratified split: train=%d (oldest), val=%d (middle), test=%d (newest)",
        len(train_df), len(val_df), len(test_df)
    )
    
    return train_df, val_df, test_df


def apply_time_split_if_enabled(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    use_time_split: bool,
    val_size: float = 0.2,
    test_size: float = 0.1,
    time_col: str = 'account_creation_date',
    random_state: int = 2112
) -> tuple:
    """Apply time-stratified split and derive reference date when enabled."""
    from FeatureEngineering import derive_reference_date

    if use_time_split:
        combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
        reference_date = derive_reference_date(combined)
        train_df, val_df, test_df = time_stratified_split(
            combined,
            val_size=val_size,
            test_size=test_size,
            time_col=time_col,
            random_state=random_state
        )
    else:
        reference_date = derive_reference_date(train_df)

    return train_df, val_df, test_df, reference_date
