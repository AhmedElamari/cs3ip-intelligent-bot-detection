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
    TabNetModel,
)


def create_models(config: Config) -> dict[str, Any]:
    """Create model instances based on configuration."""
    models = {}

    model_classes = {
        'logistic_regression': LogisticRegressionModel,
        'svm': SVMModel,
        'decision_tree': DecisionTreeModel,
        'random_forest': RandomForestModel,
        'xgboost': XGBoostModel,
        'tabnet': TabNetModel,
    }

    for model_name in config.get_enabled_models():
        if model_name not in model_classes:
            raise ValueError(f"Model {model_name} not found in model classes.")
        params = config.get_model_params(model_name)
        models[model_name] = model_classes[model_name](**params)

    return models
