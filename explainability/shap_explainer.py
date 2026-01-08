"""
SHAP Explainer
==============
SHapley Additive exPlanations for model interpretability.

SHAP values provide consistent and locally accurate attribution
of predictions to features, based on game theory.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from pathlib import Path


class SHAPExplainer:
    """
    SHAP-based model explainer for bot detection.
    
    SHAP (SHapley Additive exPlanations) uses game theory to
    explain individual predictions by computing the contribution
    of each feature.
    
    Key concepts:
        - SHAP value: The contribution of a feature to the prediction
        - Base value: The average model output
        - Local explanation: Why a specific sample was classified as it was
        - Global explanation: Overall feature importance across all samples
    """
    
    def __init__(self, model, feature_names: List[str] = None):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model (can be sklearn model or BaseModel instance)
            feature_names: List of feature names
        """
        self.model = self._get_underlying_model(model)
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        self._is_fitted = False
    
    def _get_underlying_model(self, model):
        """Extract underlying sklearn model if wrapped."""
        if hasattr(model, 'model'):  # BaseModel wrapper
            return model.model
        return model
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], max_samples: int = 100) -> 'SHAPExplainer':
        """
        Fit the SHAP explainer on background data.
        
        Args:
            X: Background data for SHAP explainer
            max_samples: Maximum samples to use for background
            
        Returns:
            self
        """
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP not installed. Install with: pip install shap")
        
        # Sample background data if too large
        if len(X) > max_samples:
            if isinstance(X, pd.DataFrame):
                X_background = X.sample(n=max_samples, random_state=2112)
            else:
                rng = np.random.RandomState(2112)
                indices = rng.choice(len(X), max_samples, replace=False)
                X_background = X[indices]
        else:
            X_background = X
        
        # Choose appropriate explainer based on model type
        model_name = type(self.model).__name__.lower()
        
        if 'tree' in model_name or 'forest' in model_name or 'gradient' in model_name:
            # Tree-based models
            self.explainer = shap.TreeExplainer(self.model)
        else:
            # Linear and other models
            self.explainer = shap.Explainer(self.model, X_background)
        
        if self.feature_names is None:
            if isinstance(X, pd.DataFrame):
                self.feature_names = X.columns.tolist()
            else:
                self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        self._is_fitted = True
        return self
    
    def explain(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Compute SHAP values for samples.
        
        Args:
            X: Samples to explain
            
        Returns:
            SHAP values array
        """
        if not self._is_fitted:
            raise RuntimeError("Explainer not fitted. Call fit() first.")
        
        import shap
        
        self.shap_values = self.explainer(X)
        
        # Handle different SHAP value formats
        if hasattr(self.shap_values, 'values'):
            values = self.shap_values.values
            # For binary classification, take positive class
            if len(values.shape) == 3:
                values = values[:, :, 1]
            return values
        return self.shap_values
    
    def get_global_importance(self) -> Dict[str, float]:
        """
        Get global feature importance from SHAP values.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.shap_values is None:
            raise RuntimeError("No SHAP values computed. Call explain() first.")
        
        import shap
        
        if hasattr(self.shap_values, 'values'):
            values = self.shap_values.values
            if len(values.shape) == 3:
                values = values[:, :, 1]
        else:
            values = self.shap_values
        
        # Mean absolute SHAP value per feature
        importance = np.abs(values).mean(axis=0)
        return dict(zip(self.feature_names, importance))
    
    def explain_instance(self, instance: np.ndarray, instance_idx: int = 0) -> Dict[str, float]:
        """
        Explain a single instance prediction.
        
        Args:
            instance: Single sample or array of samples
            instance_idx: Index of instance to explain
            
        Returns:
            Dictionary of feature contributions
        """
        shap_values = self.explain(instance)
        
        if len(shap_values.shape) > 1:
            instance_shap = shap_values[instance_idx]
        else:
            instance_shap = shap_values
        
        return dict(zip(self.feature_names, instance_shap))
    
    def plot_summary(self, X: Union[np.ndarray, pd.DataFrame], max_display: int = 15):
        """
        Create SHAP summary plot showing feature importance and effects.
        
        Args:
            X: Data to explain
            max_display: Maximum features to display
            
        Returns:
            matplotlib figure
        """
        import shap
        import matplotlib.pyplot as plt
        
        if self.shap_values is None:
            self.explain(X)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values,
            X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        return plt.gcf()
    
    def plot_waterfall(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        instance_idx: int = 0,
        max_display: int = 15
    ):
        """
        Create waterfall plot for a single instance.
        
        Shows how each feature contributes to push the prediction
        from the base value.
        
        Args:
            X: Data containing instance
            instance_idx: Index of instance to plot
            max_display: Maximum features to display
            
        Returns:
            matplotlib figure
        """
        import shap
        import matplotlib.pyplot as plt
        
        if self.shap_values is None:
            self.explain(X)
        
        shap.waterfall_plot(
            self.shap_values[instance_idx],
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        return plt.gcf()
    
    def plot_force(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        instance_idx: int = 0
    ):
        """
        Create force plot for a single instance.
        
        Args:
            X: Data containing instance
            instance_idx: Index of instance to plot
            
        Returns:
            SHAP force plot visualization
        """
        import shap
        
        if self.shap_values is None:
            self.explain(X)
        
        return shap.force_plot(
            self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
            self.shap_values[instance_idx].values if hasattr(self.shap_values[instance_idx], 'values') else self.shap_values[instance_idx],
            X[instance_idx] if hasattr(X, '__getitem__') else X,
            feature_names=self.feature_names
        )
    
    def plot_dependence(
        self,
        feature: str,
        X: Union[np.ndarray, pd.DataFrame],
        interaction_feature: str = None
    ):
        """
        Create dependence plot showing how a feature affects predictions.
        
        Args:
            feature: Feature to analyze
            X: Data to explain
            interaction_feature: Optional second feature for interaction
            
        Returns:
            matplotlib figure
        """
        import shap
        import matplotlib.pyplot as plt
        
        if self.shap_values is None:
            self.explain(X)
        
        shap.dependence_plot(
            feature,
            self.shap_values.values if hasattr(self.shap_values, 'values') else self.shap_values,
            X,
            feature_names=self.feature_names,
            interaction_index=interaction_feature,
            show=False
        )
        plt.tight_layout()
        return plt.gcf()
    
    def save_explanations(self, path: str) -> None:
        """Save SHAP values to disk."""
        if self.shap_values is None:
            raise RuntimeError("No SHAP values to save. Call explain() first.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'shap_values': self.shap_values,
                'feature_names': self.feature_names,
            }, f)
