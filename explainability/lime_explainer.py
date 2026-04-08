"""LIME (Local Interpretable Model-agnostic Explanations) for individual predictions."""

from typing import Any, Dict, List, Optional, Union, Callable
import numpy as np
import pandas as pd


class LIMEExplainer:
    """LIME-based explainer for bot detection; model-agnostic local explanations."""
    
    def __init__(
        self,
        model,
        feature_names: List[str] = None,
        class_names: List[str] = None
    ):
        """
        Initialize LIME explainer.
        
        Args:
            model: Trained model with predict_proba method
            feature_names: List of feature names
            class_names: Names for each class
        """
        self.model = self._get_underlying_model(model)
        self.feature_names = feature_names
        self.class_names = class_names or ['Human', 'Bot']
        self.explainer = None
        self._is_fitted = False
    
    def _get_underlying_model(self, model):
        """Extract underlying sklearn model if wrapped."""
        if hasattr(model, 'model'):
            return model.model
        return model
    
    def _get_predict_fn(self) -> Callable:
        """Get prediction function for LIME."""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba
        else:
            # Fallback for models without predict_proba
            def predict_fn(X):
                pred = self.model.predict(X)
                return np.column_stack([1 - pred, pred])
            return predict_fn
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], mode: str = 'tabular') -> 'LIMEExplainer':
        """Fit LIME explainer on training data."""
        try:
            from lime import lime_tabular
        except ImportError:
            raise ImportError("LIME not installed. Install with: pip install lime")
        
        if self.feature_names is None:
            if isinstance(X, pd.DataFrame):
                self.feature_names = X.columns.tolist()
            else:
                self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Convert to numpy if needed
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        self.explainer = lime_tabular.LimeTabularExplainer(
            training_data=X_array,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode='classification',
            random_state=2112
        )
        
        self._is_fitted = True
        return self
    
    def explain_instance(
        self,
        instance: np.ndarray,
        num_features: int = 10,
        num_samples: int = 5000
    ) -> Dict[str, Any]:
        """
        Explain a single instance prediction.
        
        Args:
            instance: Single sample to explain (1D array)
            num_features: Number of features in explanation
            num_samples: Number of samples for LIME perturbation
            
        Returns:
            Dictionary with explanation details
        """
        if not self._is_fitted:
            raise RuntimeError("Explainer not fitted. Call fit() first.")
        
        # Ensure instance is 1D
        if len(instance.shape) > 1:
            instance = instance.flatten()
        
        explanation = self.explainer.explain_instance(
            instance,
            self._get_predict_fn(),
            num_features=num_features,
            num_samples=num_samples
        )
        
        # Extract feature contributions
        feature_contributions = dict(explanation.as_list())
        
        # Get prediction probabilities
        predict_proba = self._get_predict_fn()(instance.reshape(1, -1))[0]
        
        return {
            'feature_contributions': feature_contributions,
            'prediction_proba': dict(zip(self.class_names, predict_proba)),
            'predicted_class': self.class_names[np.argmax(predict_proba)],
            'local_prediction': explanation.local_pred[0] if hasattr(explanation, 'local_pred') else None,
            'intercept': explanation.intercept[1] if hasattr(explanation, 'intercept') else None,
            'score': explanation.score if hasattr(explanation, 'score') else None,
            '_raw_explanation': explanation
        }
    
    def explain_batch(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        num_features: int = 10,
        num_samples: int = 5000
    ) -> List[Dict[str, Any]]:
        """
        Explain multiple instances.
        
        Args:
            X: Samples to explain
            num_features: Number of features in each explanation
            num_samples: Number of LIME samples
            
        Returns:
            List of explanation dictionaries
        """
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        explanations = []
        
        for i in range(len(X_array)):
            exp = self.explain_instance(X_array[i], num_features, num_samples)
            explanations.append(exp)
        
        return explanations
    
    def get_important_features(
        self,
        instance: np.ndarray,
        num_features: int = 10
    ) -> List[tuple]:
        """
        Get most important features for an instance.
        
        Args:
            instance: Sample to explain
            num_features: Number of features to return
            
        Returns:
            List of (feature_name, contribution) tuples sorted by importance
        """
        explanation = self.explain_instance(instance, num_features)
        contributions = explanation['feature_contributions']
        
        # Sort by absolute contribution
        sorted_features = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        return sorted_features
    
    def plot_explanation(
        self,
        instance: np.ndarray,
        num_features: int = 10,
        figsize: tuple = (10, 6)
    ):
        """
        Plot LIME explanation for an instance.
        
        Args:
            instance: Sample to explain
            num_features: Number of features to show
            figsize: Figure size
            
        Returns:
            matplotlib figure
        """
        import matplotlib.pyplot as plt
        
        explanation = self.explain_instance(instance, num_features)
        contributions = explanation['feature_contributions']
        
        # Prepare data for plotting
        features = list(contributions.keys())
        values = list(contributions.values())
        colors = ['green' if v > 0 else 'red' for v in values]
        
        fig, ax = plt.subplots(figsize=figsize)
        y_pos = range(len(features))
        ax.barh(y_pos, values, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Feature Contribution')
        ax.set_title(f"LIME Explanation (Predicted: {explanation['predicted_class']})")
        
        plt.tight_layout()
        return fig
    
    def plot_as_notebook(
        self,
        instance: np.ndarray,
        num_features: int = 10
    ):
        """
        Display LIME's built-in notebook visualization.
        
        Args:
            instance: Sample to explain
            num_features: Number of features to show
        """
        explanation = self.explain_instance(instance, num_features)
        raw_exp = explanation['_raw_explanation']
        return raw_exp.show_in_notebook(show_table=True)
    
    def compare_instances(
        self,
        instance1: np.ndarray,
        instance2: np.ndarray,
        num_features: int = 10
    ) -> Dict[str, Any]:
        """
        Compare explanations for two instances.
        
        Useful for understanding why the model classifies
        similar instances differently.
        
        Args:
            instance1: First sample
            instance2: Second sample
            num_features: Number of features in explanations
            
        Returns:
            Comparison dictionary
        """
        exp1 = self.explain_instance(instance1, num_features)
        exp2 = self.explain_instance(instance2, num_features)
        
        features1 = set(exp1['feature_contributions'].keys())
        features2 = set(exp2['feature_contributions'].keys())
        
        common_features = features1 & features2
        
        comparison = {
            'instance1_prediction': exp1['predicted_class'],
            'instance2_prediction': exp2['predicted_class'],
            'instance1_proba': exp1['prediction_proba'],
            'instance2_proba': exp2['prediction_proba'],
            'common_important_features': list(common_features),
            'feature_differences': {}
        }
        
        for feature in common_features:
            comparison['feature_differences'][feature] = {
                'instance1': exp1['feature_contributions'].get(feature, 0),
                'instance2': exp2['feature_contributions'].get(feature, 0),
                'difference': exp1['feature_contributions'].get(feature, 0) - 
                             exp2['feature_contributions'].get(feature, 0)
            }
        
        return comparison
