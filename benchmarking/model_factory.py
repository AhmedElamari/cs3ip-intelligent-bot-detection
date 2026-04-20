"""
Model factory helpers for the benchmark pipeline.
"""
from typing import Any

from config import Config
from benchmarking.hpo.factory import build_model


def create_models(config: Config) -> dict[str, Any]:
    """Create model instances based on configuration."""
    models = {}

    for model_name in config.get_enabled_models():
        params = config.get_model_params(model_name)
        models[model_name] = build_model(model_name, params)

    return models
