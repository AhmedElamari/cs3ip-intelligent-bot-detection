"""
Models Module
=============
Contains all ML model implementations for bot detection.
Each model follows a common interface defined in base.py.
"""

from .base import BaseModel
from .logistic_regression import LogisticRegressionModel
from .svm import SVMModel
from .decision_tree import DecisionTreeModel
from .random_forest import RandomForestModel
from .gradient_boosting import GradientBoostingModel

__all__ = [
    'BaseModel',
    'LogisticRegressionModel',
    'SVMModel',
    'DecisionTreeModel',
    'RandomForestModel',
    'GradientBoostingModel',
]

# Model registry for easy access
MODEL_REGISTRY = {
    'logistic_regression': LogisticRegressionModel,
    'svm': SVMModel,
    'decision_tree': DecisionTreeModel,
    'random_forest': RandomForestModel,
    'gradient_boosting': GradientBoostingModel,
}


def get_model(name: str, **kwargs):
    """Factory function to get model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)


def get_all_models(**kwargs):
    """Get instances of all available models."""
    return {name: cls(**kwargs) for name, cls in MODEL_REGISTRY.items()}
