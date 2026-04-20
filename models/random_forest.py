"""
Random Forest Model
===================
Ensemble of decision trees for robust classification.
Provides aggregate feature importance.
"""

from typing import Any, Optional, Dict, List
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from .base import BaseModel


class RandomForestModel(BaseModel):
    """
    Random Forest classifier for bot detection.
    
    Random Forest is an ensemble of decision trees, where each
    tree is trained on a random subset of data and features.
    The final prediction is made by majority voting.
    
    Advantages:
        - Robust to overfitting (compared to single trees)
        - Handles high-dimensional data well
        - Provides feature importance scores
        - Works well out-of-the-box
        
    Disadvantages:
        - Less interpretable than single trees
        - Can be slow for real-time predictions
        - May struggle with very imbalanced data
    """
    
    def __init__(self, random_state: int = 2112, **kwargs):
        super().__init__(name='Random Forest', random_state=random_state)
        self._params = {
            'random_state': random_state,
            'n_estimators': kwargs.get('n_estimators', 100),
            'max_depth': kwargs.get('max_depth', 10),
            'min_samples_split': kwargs.get('min_samples_split', 2),
            'min_samples_leaf': kwargs.get('min_samples_leaf', 1),
            'max_features': kwargs.get('max_features', 'sqrt'),
            'class_weight': kwargs.get('class_weight', 'balanced'),
            'n_jobs': kwargs.get('n_jobs', -1),
            'oob_score': kwargs.get('oob_score', True),
        }
        self.model = self._create_model(**self._params)
    
    def _create_model(self, **kwargs) -> RandomForestClassifier:
        return RandomForestClassifier(**kwargs)
    
    @property
    def is_interpretable(self) -> bool:
        return False  # Ensemble is not directly interpretable
    
    @property
    def supports_feature_importance(self) -> bool:
        return True
    
    def get_oob_score(self) -> Optional[float]:
        """
        Get out-of-bag score (estimate of generalization error).
        
        Only available if oob_score=True during training.
        """
        self._check_fitted()
        if hasattr(self.model, 'oob_score_'):
            return self.model.oob_score_
        return None
    
    def get_feature_importance_std(self) -> Dict[str, float]:
        """
        Get standard deviation of feature importances across trees.
        
        High std indicates the feature importance varies significantly
        across different trees, suggesting instability.
        """
        self._check_fitted()
        importances = np.array([tree.feature_importances_ for tree in self.model.estimators_])
        std = np.std(importances, axis=0)
        return dict(zip(self.feature_names, std))
    
    def get_top_features(self, n: int = 10) -> List[tuple]:
        """
        Get top N most important features.
        
        Args:
            n: Number of top features to return
            
        Returns:
            List of (feature_name, importance) tuples
        """
        self._check_fitted()
        importance_dict = self.get_feature_importance()
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:n]
    
    def plot_feature_importance(self, n_features: int = 15, figsize: tuple = (10, 8)):
        """
        Plot feature importances with error bars.
        
        Args:
            n_features: Number of top features to show
            figsize: Figure size
            
        Returns:
            matplotlib figure
        """
        import matplotlib.pyplot as plt
        
        self._check_fitted()
        
        importances = self.model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in self.model.estimators_], axis=0)
        
        # Sort by importance
        indices = np.argsort(importances)[::-1][:n_features]
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(
            range(len(indices)),
            importances[indices],
            xerr=std[indices],
            align='center',
            color='steelblue',
            ecolor='gray'
        )
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([self.feature_names[i] for i in indices])
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Random Forest Feature Importances (Top {n_features})')
        plt.tight_layout()
        
        return fig
