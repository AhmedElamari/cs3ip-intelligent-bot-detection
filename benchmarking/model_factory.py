"""
Model factory helpers for the benchmark pipeline.
"""

from config import Config
from models import (
    LogisticRegressionModel,
    SVMModel,
    DecisionTreeModel,
    RandomForestModel,
    GradientBoostingModel,
)


def create_models(config: Config) -> dict:
    """
    This function creates model instances based of the configurations provided, 
    in the config object.

    Args:
        config: (Config) configuration object used to determine which models are enabled
        and their parameters.


    Returns:
        Dictionary of model instances keyed by model name to corresponding model class.
        For example, {
            'logistic_regression': LogisticRegressionModel,
            'svm': SVMModel,
            'decision_tree': DecisionTreeModel,
            'random_forest': RandomForestModel,
            'gradient_boosting': GradientBoostingModel,
        }
    """
    models = {}

    enabled_models = config.get_enabled_models()

    model_classes = {
        'logistic_regression': LogisticRegressionModel,
        'svm': SVMModel,
        'decision_tree': DecisionTreeModel,
        'random_forest': RandomForestModel,
        'gradient_boosting': GradientBoostingModel,
    }

    for model_name in enabled_models:
        if model_name in model_classes:
            params = config.get_model_params(model_name)
            models[model_name] = model_classes[model_name](**params)

    return models
