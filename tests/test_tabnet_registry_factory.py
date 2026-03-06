"""
Registry and factory guard tests for TabNet.

Covers:
  - 'tabnet' present in MODEL_REGISTRY
  - 'tabnet' present in Config.DEFAULTS['models']
  - 'tabnet' disabled by default (safe-by-default)
  - create_models instantiates TabNetModel when tabnet is enabled
  - TabNetModel class is importable (dependency-guarded, not instantiated here)
"""

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None
NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None


class TabNetRegistryTest(unittest.TestCase):
    """Verify tabnet is wired into MODEL_REGISTRY and Config defaults."""

    def test_tabnet_in_model_registry(self):
        from models import MODEL_REGISTRY
        self.assertIn("tabnet", MODEL_REGISTRY)

    def test_tabnet_class_is_tabnetmodel(self):
        from models import MODEL_REGISTRY
        from models.tabnet import TabNetModel
        self.assertIs(MODEL_REGISTRY["tabnet"], TabNetModel)

    def test_tabnet_in_config_defaults(self):
        from config import Config
        self.assertIn("tabnet", Config.DEFAULTS["models"])

    def test_tabnet_disabled_by_default(self):
        """TabNet must be off by default to avoid breaking installs without DL deps."""
        from config import Config
        self.assertFalse(
            Config.DEFAULTS["models"]["tabnet"]["enabled"],
            "tabnet must be disabled by default until pytorch-tabnet is installed."
        )

    def test_tabnet_config_params_present(self):
        from config import Config
        params = Config.DEFAULTS["models"]["tabnet"]["params"]
        for key in ("n_d", "n_a", "n_steps", "gamma", "lambda_sparse",
                    "batch_size", "virtual_batch_size", "momentum",
                    "mask_type", "max_epochs", "patience"):
            self.assertIn(key, params, f"Expected param '{key}' missing from tabnet defaults.")


@unittest.skipUnless(SKLEARN_AVAILABLE and NUMPY_AVAILABLE,
                     "sklearn/numpy not installed")
class TabNetFactoryTest(unittest.TestCase):
    """Verify create_models instantiates TabNet when enabled."""

    def test_factory_creates_tabnet_when_enabled(self):
        from config import Config
        from benchmarking.model_factory import create_models
        from models.tabnet import TabNetModel

        config = Config()
        # Disable all models except tabnet
        for name in config.get("models", {}).keys():
            config.set(f"models.{name}.enabled", False)
        config.set("models.tabnet.enabled", True)

        models = create_models(config)
        self.assertIn("tabnet", models)
        self.assertIsInstance(models["tabnet"], TabNetModel)

    def test_factory_excludes_tabnet_when_disabled(self):
        from config import Config
        from benchmarking.model_factory import create_models

        config = Config()
        config.set("models.tabnet.enabled", False)
        models = create_models(config)
        self.assertNotIn("tabnet", models)

    def test_factory_raises_on_unknown_model(self):
        from config import Config
        from benchmarking.model_factory import create_models

        config = Config()
        for name in config.get("models", {}).keys():
            config.set(f"models.{name}.enabled", False)
        config.set("models.nonexistent_model.enabled", True)

        with self.assertRaises(ValueError):
            create_models(config)


if __name__ == "__main__":
    unittest.main()
