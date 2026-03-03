"""
Base Model Class
================
Abstract base class defining the interface for all ML models.
Follows the Strategy pattern for interchangeable model implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import time
import pickle
import warnings
from pathlib import Path


class BaseModel(ABC):
    """
    Abstract base class for all bot detection models.
    
    This class defines a common interface that all model implementations
    must follow, ensuring consistency across different algorithms.
    
    Attributes:
        name (str): Human-readable name of the model
        model: The underlying sklearn model instance
        is_fitted (bool): Whether the model has been trained
        training_time (float): Time taken to train the model
        feature_names (List[str]): Names of features used in training
    """
    
    def __init__(self, name: str, random_state: int = 2112):
        self.name = name
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        self.training_time = 0.0
        self.feature_names: List[str] = []
        self._params: Dict[str, Any] = {}
    
    @abstractmethod
    def _create_model(self, **kwargs) -> Any:
        """Create and return the underlying sklearn model instance."""
        pass
    
    @property
    @abstractmethod
    def is_interpretable(self) -> bool:
        """Return True if the model is inherently interpretable."""
        pass
    
    @property
    @abstractmethod
    def supports_feature_importance(self) -> bool:
        """Return True if the model provides feature importance scores."""
        pass
    
    def set_params(self, **params) -> 'BaseModel':
        """Set model hyperparameters."""
        self._params.update(params)
        self.model = self._create_model(**self._params)
        return self
    
    def get_params(self) -> Dict[str, Any]:
        """Get current model parameters."""
        if self.model is not None:
            return self.model.get_params()
        return self._params
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[List[str]] = None,
        **kwargs
    ) -> 'BaseModel':
        """
        Train the model on the provided data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            feature_names: Optional list of feature names
            **kwargs: Additional arguments passed to the underlying fit method
            
        Returns:
            self: The fitted model instance
        """
        if self.model is None:
            self.model = self._create_model(**self._params)
        
        if feature_names is not None:
            self.feature_names = feature_names
        elif isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
        else:
            self.feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        
        start_time = time.time()
        self.model.fit(X_train, y_train, **kwargs)
        self.training_time = time.time() - start_time
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predicted labels
        """
        self._check_fitted()
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability predictions if supported.
        
        Args:
            X: Features to predict on
            
        Returns:
            Probability predictions for each class
        """
        self._check_fitted()
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # Fall back to decision function if available
            if hasattr(self.model, 'decision_function'):
                decision = self.model.decision_function(X)
                # Convert to probability-like values using sigmoid
                proba = 1 / (1 + np.exp(-decision))
                return np.column_stack([1 - proba, proba])
            raise NotImplementedError(f"{self.name} does not support probability predictions")
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        prefix: str = ''
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Features to evaluate on
            y: True labels
            prefix: Optional prefix for metric names
            
        Returns:
            Dictionary of evaluation metrics
        """
        self._check_fitted()
        
        y_pred = self.predict(X)
        
        metrics = {
            f'{prefix}accuracy': accuracy_score(y, y_pred),
            f'{prefix}precision': precision_score(y, y_pred, average='binary', zero_division=0),
            f'{prefix}recall': recall_score(y, y_pred, average='binary', zero_division=0),
            f'{prefix}f1': f1_score(y, y_pred, average='binary', zero_division=0),
        }
        
        # Add ROC-AUC if model supports probability predictions
        try:
            y_proba = self.predict_proba(X)[:, 1]
            metrics[f'{prefix}roc_auc'] = roc_auc_score(y, y_proba)
        except (NotImplementedError, AttributeError):
            pass
        
        return metrics
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores if supported.
        
        Returns:
            Dictionary mapping feature names to importance scores,
            or None if not supported
        """
        if not self.supports_feature_importance:
            return None
        
        self._check_fitted()
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_).flatten()
        else:
            return None
        
        return dict(zip(self.feature_names, importances))
    
    def get_confusion_matrix(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Get confusion matrix for predictions."""
        self._check_fitted()
        y_pred = self.predict(X)
        return confusion_matrix(y, y_pred)
    
    def get_classification_report(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_names: List[str] = None
    ) -> str:
        """Get detailed classification report."""
        self._check_fitted()
        y_pred = self.predict(X)
        if target_names is None:
            target_names = ['Human', 'Bot']
        return classification_report(y, y_pred, target_names=target_names)
    
    def save(self, path: str) -> None:
        """Save the model to disk."""
        self._check_fitted()
        path = self._validate_output_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'name': self.name,
                'feature_names': self.feature_names,
                'params': self._params,
                'training_time': self.training_time,
            }, f)
    
    @classmethod
    def load(cls, path: str, trusted_source: bool = False) -> 'BaseModel':
        """Load a model from disk.

        Args:
            path: Pickle file path created by ``save``.
            trusted_source: Must be True to load. Pickle can execute arbitrary code.
        """
        path = cls._validate_output_path(path)
        if not trusted_source:
            raise ValueError(
                "Refusing to load pickle from an untrusted source. "
                "Pass trusted_source=True only for files you fully trust."
            )
        warnings.warn(
            "Loading a pickled model. Only load artifacts from trusted sources.",
            UserWarning,
            stacklevel=2,
        )
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls.__new__(cls)
        instance.model = data['model']
        instance.name = data['name']
        instance.feature_names = data['feature_names']
        instance._params = data['params']
        instance.training_time = data['training_time']
        instance.is_fitted = True
        instance.random_state = instance._params.get('random_state', 2112)
        
        return instance
    
    def _check_fitted(self) -> None:
        """Check if the model has been fitted."""
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} has not been fitted. Call fit() first.")

    @staticmethod
    def _validate_output_path(path: str) -> Path:
        """Restrict model artifacts to the current workspace."""
        resolved = Path(path).expanduser().resolve()
        workspace = Path(__file__).resolve().parent.parent
        if resolved != workspace and workspace not in resolved.parents:
            raise ValueError(
                f"Path must stay within workspace: {workspace}"
            )
        return resolved
    
    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.__class__.__name__}(name='{self.name}', status={status})"
