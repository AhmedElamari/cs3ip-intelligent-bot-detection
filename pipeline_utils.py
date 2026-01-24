import logging
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
