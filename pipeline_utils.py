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
    if time_col not in df.columns:
        LOGGER.warning(
            "Time column '%s' not found. Falling back to index-based split.",
            time_col
        )
        # Fall back to using row order as proxy for time
        df_sorted = df.reset_index(drop=True)
    else:
        df_sorted = df.sort_values(time_col).reset_index(drop=True)
    
    n = len(df_sorted)
    train_end = int(n * (1 - val_size - test_size))
    val_end = int(n * (1 - test_size))
    
    # Split chronologically: train oldest, val middle, test newest
    train_df = df_sorted.iloc[:train_end].sample(frac=1, random_state=random_state)
    val_df = df_sorted.iloc[train_end:val_end].sample(frac=1, random_state=random_state)
    test_df = df_sorted.iloc[val_end:].sample(frac=1, random_state=random_state)
    
    LOGGER.info(
        "Time-stratified split: train=%d (oldest), val=%d (middle), test=%d (newest)",
        len(train_df), len(val_df), len(test_df)
    )
    
    return train_df, val_df, test_df
