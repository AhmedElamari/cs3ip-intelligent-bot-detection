"""
Model factory helpers for the benchmark pipeline.
"""
from typing import Any
from config import Config
from models import (
    LogisticRegressionModel,
    SVMModel,
    DecisionTreeModel,
    RandomForestModel,
    XGBoostModel,
)


def create_models(config: Config) -> dict[str, Any]:
    """Create model instances based on configuration."""
    models = {}

    enabled_models = config.get_enabled_models()

    model_classes = {
        'logistic_regression': LogisticRegressionModel,
        'svm': SVMModel,
        'decision_tree': DecisionTreeModel,
        'random_forest': RandomForestModel,
        'xgboost': XGBoostModel,
    }

    for model_name in enabled_models:
        if model_name in model_classes:
            params = config.get_model_params(model_name)
            models[model_name] = model_classes[model_name](**params)
        else:
            raise ValueError(f"Model {model_name} not found in model classes.")

    return models
