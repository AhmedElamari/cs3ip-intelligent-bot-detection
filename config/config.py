"""
Configuration Management
========================
Centralized configuration for the bot detection pipeline.
Supports YAML configuration files.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import copy


class Config:
    """
    Configuration container for bot detection pipeline.
    
    Provides centralized management of:
        - Model hyperparameters
        - Training settings
        - Feature engineering options
        - Experiment tracking
    """
    
    # Default configuration
    DEFAULTS = {
        'random_state': 2112,
        'test_size': 0.1,
        'val_size': 0.2,
        'time_split': False,  # Use chronological splitting to combat data drift
        
        'preprocessing': {
            'handle_imbalance': False,
            'imbalance_method': 'smote',
            'scale_features': False,
            'feature_selection': False,
            'n_features': 20,
        },
        
        'models': {
            'logistic_regression': {
                'enabled': True,
                'params': {
                    'max_iter': 1000,
                    'C': 1.0,
                    'class_weight': 'balanced',
                    'solver': 'lbfgs',
                }
            },
            'svm': {
                'enabled': True,
                'params': {
                    'kernel': 'rbf',
                    'C': 1.0,
                    'gamma': 'scale',
                    'class_weight': 'balanced',
                    'probability': True,
                }
            },
            'decision_tree': {
                'enabled': True,
                'params': {
                    'max_depth': 10,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'class_weight': 'balanced',
                    'criterion': 'gini',
                }
            },
            'random_forest': {
                'enabled': True,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'class_weight': 'balanced',
                    'n_jobs': -1,
                    'oob_score': True,
                }
            },
            'xgboost': {
                'enabled': True,
                'params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 5,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.0,
                    'reg_lambda': 1.0,
                    'class_weight': 'balanced',
                }
            },
            'tabnet': {
                # Disabled by default: requires `pip install -r requirements-dl.txt`
                'enabled': False,
                'params': {
                    'n_d': 32,
                    'n_a': 32,
                    'n_steps': 3,
                    'gamma': 1.3,
                    'lambda_sparse': 1e-3,
                    'batch_size': 1024,
                    'virtual_batch_size': 128,
                    'momentum': 0.02,
                    'mask_type': 'sparsemax',
                    'max_epochs': 200,
                    'patience': 20,
                    'class_weight': 'balanced',
                }
            },
        },
        
        'explainability': {
            'enabled': True,
            'shap': {
                'enabled': True,
                'max_samples': 100,
            },
            'lime': {
                'enabled': True,
                'num_samples': 5000,
                'num_features': 10,
            },
            'feature_importance': {
                'enabled': True,
                'methods': ['builtin', 'permutation'],
            },
            'poster': {
                'enabled': False,
                'model': 'xgboost',
                'top_n': 10,
            },
        },

        'robustness': {
            'enabled': False,
            'attack_population': 'true_bots',
            'profiles': ['cheap_only', 'realistic_mixed'],
            'evaluate_single_feature_attacks': True,
            'evaluate_bundle_attacks': True,
            'expensive_nudge_fraction': 0.05,
        },
        
        'output': {
            'save_models': True,
            'save_plots': True,
            'output_dir': 'results',
            'experiment_name': None,
        },

        'hpo': {
            'enabled': True,
            'reuse_cache': True,
            'metric': 'val_f1',
            'fail_fast': True,
            'cache_dir': 'results/hpo_cache',
            'sampler_seed': 2112,
            'trials_per_model': {
                'logistic_regression': 15,
                'svm': 20,
                'decision_tree': 15,
                'random_forest': 20,
                'xgboost': 25,
                'tabnet': 30,
            },
            'pruning': {
                'tabnet': {
                    'type': 'median',
                    'n_startup_trials': 5,
                    'n_warmup_steps': 0,
                },
            },
        },
    }
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        """
        Initialize configuration.
        
        Args:
            config_dict: Configuration dictionary to merge with defaults
        """
        self._config = copy.deepcopy(self.DEFAULTS)
        
        if config_dict:
            self._deep_update(self._config, config_dict)
    
    def _deep_update(self, base: dict, update: dict) -> dict:
        """Deep update a nested dictionary."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
        return base
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'models.random_forest.params.n_estimators')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.
        
        Args:
            key: Configuration key
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_model_params(self, model_name: str) -> Dict[str, Any]:
        """
        Get parameters for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model parameters dictionary
        """
        model_config = self.get(f'models.{model_name}', {})
        params = model_config.get('params', {}).copy()
        params['random_state'] = self.get('random_state')
        return params
    
    def get_enabled_models(self) -> list:
        """Get list of enabled model names."""
        models_config = self.get('models', {})
        return [
            name for name, config in models_config.items()
            if config.get('enabled', True)
        ]

    def get_hpo(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Return a copy of the ``hpo`` section; optional per-model trial budget."""
        hpo = copy.deepcopy(self.get('hpo', {}))
        if model_name:
            tpm = hpo.get('trials_per_model') or {}
            if model_name in tpm:
                hpo['trials_for_model'] = tpm[model_name]
        return hpo
    
    def to_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary."""
        return copy.deepcopy(self._config)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """
        Load configuration from YAML file.
        
        Args:
            path: Path to YAML file
            
        Returns:
            Config instance
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML not installed. Install with: pip install pyyaml")
        
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(config_dict)
    
    def to_yaml(self, path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            path: Path to save YAML file
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML not installed. Install with: pip install pyyaml")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def from_json(cls, path: str) -> 'Config':
        """Load configuration from JSON file."""
        import json
        
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(config_dict)
    
    def to_json(self, path: str) -> None:
        """Save configuration to JSON file."""
        import json
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self._config, f, indent=2)
    
    def __repr__(self) -> str:
        return f"Config(models={self.get_enabled_models()})"


def load_config(path: str = None) -> Config:
    """
    Load configuration from file or return defaults.
    
    Args:
        path: Optional path to config file (YAML or JSON)
        
    Returns:
        Config instance
    """
    if path is None:
        return Config()
    
    path = Path(path)
    
    if path.suffix in ['.yml', '.yaml']:
        return Config.from_yaml(path)
    elif path.suffix == '.json':
        return Config.from_json(path)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")


# Example configuration YAML template
CONFIG_TEMPLATE = """
# Bot Detection Pipeline Configuration
# =====================================

random_state: 2112
test_size: 0.1
val_size: 0.2
time_split: false  # Use chronological splitting to combat data drift

preprocessing:
  handle_imbalance: false
  imbalance_method: smote  # smote or undersample
  scale_features: false
  feature_selection: false
  n_features: 20

models:
  logistic_regression:
    enabled: true
    params:
      max_iter: 1000
      C: 1.0
      class_weight: balanced
      solver: lbfgs

  svm:
    enabled: true
    params:
      kernel: rbf
      C: 1.0
      gamma: scale
      class_weight: balanced
      probability: true

  decision_tree:
    enabled: true
    params:
      max_depth: 10
      min_samples_split: 2
      min_samples_leaf: 1
      class_weight: balanced
      criterion: gini

  random_forest:
    enabled: true
    params:
      n_estimators: 100
      max_depth: 10
      class_weight: balanced
      n_jobs: -1
      oob_score: true

  xgboost:
    enabled: true
    params:
      n_estimators: 100
      learning_rate: 0.1
      max_depth: 5
      subsample: 0.8
      colsample_bytree: 0.8
      reg_alpha: 0.0
      reg_lambda: 1.0
      class_weight: balanced

  tabnet:
    # Requires: pip install -r requirements-dl.txt
    enabled: false
    params:
      n_d: 32
      n_a: 32
      n_steps: 3
      gamma: 1.3
      lambda_sparse: 0.001
      batch_size: 1024
      virtual_batch_size: 128
      momentum: 0.02
      mask_type: sparsemax
      max_epochs: 200
      patience: 20
      class_weight: balanced

explainability:
  enabled: true
  shap:
    enabled: true
    max_samples: 100
  lime:
    enabled: true
    num_samples: 5000
    num_features: 10
  feature_importance:
    enabled: true
    methods:
      - builtin
      - permutation
  poster:
    enabled: false
    model: xgboost
    top_n: 10

robustness:
  enabled: false
  attack_population: true_bots
  profiles:
    - cheap_only
    - realistic_mixed
  evaluate_single_feature_attacks: true
  evaluate_bundle_attacks: true
  expensive_nudge_fraction: 0.05

output:
  save_models: true
  save_plots: true
  output_dir: results
  experiment_name: null

hpo:
  enabled: true
  reuse_cache: true
  metric: val_f1
  fail_fast: true
  cache_dir: results/hpo_cache
  sampler_seed: 2112
  trials_per_model:
    logistic_regression: 15
    svm: 20
    decision_tree: 15
    random_forest: 20
    xgboost: 25
    tabnet: 30
  pruning:
    tabnet:
      type: median
      n_startup_trials: 5
      n_warmup_steps: 0
"""


def create_default_config(path: str = 'config/config.yaml') -> None:
    """Create default configuration file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        f.write(CONFIG_TEMPLATE)
    
    print(f"Default configuration created at: {path}")
