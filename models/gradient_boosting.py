"""
Gradient Boosting Model
=======================
Sequential ensemble that learns from previous errors.
State-of-the-art performance but less interpretable.
"""

from typing import Any, Optional, Dict, List
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from .base import BaseModel


class GradientBoostingModel(BaseModel):
    """
    Gradient Boosting classifier for bot detection.
    
    Gradient Boosting builds trees sequentially, where each tree
    tries to correct the errors of the previous ones. This often
    leads to excellent predictive performance.
    
    Advantages:
        - Often achieves best-in-class performance
        - Handles mixed feature types
        - Built-in feature importance
        - Less prone to overfitting with proper tuning
        
    Disadvantages:
        - Less interpretable than single models
        - Slower training due to sequential nature
        - Sensitive to hyperparameters
        - Can overfit with too many trees

    Notes:
        GradientBoostingClassifier does not support class_weight directly.
        When class_weight is provided, the model converts it to sample weights
        during fit to address class imbalance.
    """
    
    def __init__(self, random_state: int = 2112, **kwargs):
        super().__init__(name='Gradient Boosting', random_state=random_state)
        self._params = {
            'random_state': random_state,
            'n_estimators': kwargs.get('n_estimators', 100),
            'learning_rate': kwargs.get('learning_rate', 0.1),
            'max_depth': kwargs.get('max_depth', 5),
            'min_samples_split': kwargs.get('min_samples_split', 2),
            'min_samples_leaf': kwargs.get('min_samples_leaf', 1),
            'subsample': kwargs.get('subsample', 0.8),
            'max_features': kwargs.get('max_features', 'sqrt'),
            'class_weight': kwargs.get('class_weight', 'balanced'),
        }
        self.model = self._create_model(**self._params)
    
    def _create_model(self, **kwargs) -> GradientBoostingClassifier:
        kwargs.pop('class_weight', None)
        return GradientBoostingClassifier(**kwargs)

    @staticmethod
    def _compute_sample_weight(y: np.ndarray, class_weight) -> Optional[np.ndarray]:
        if class_weight is None:
            return None
        if class_weight == 'balanced':
            y_int = np.asarray(y).astype(int)
            classes, counts = np.unique(y_int, return_counts=True)
            if len(classes) == 0:
                return None
            total = len(y_int)
            weight_map = {
                cls: total / (len(classes) * count) for cls, count in zip(classes, counts, strict=True)
            }
        elif isinstance(class_weight, dict):
            weight_map = {int(k): v for k, v in class_weight.items()}
        else:
            raise ValueError(f"Unsupported class_weight value: {class_weight}")
        y_int = np.asarray(y).astype(int)
        unknown = sorted(set(np.unique(y_int)) - set(weight_map.keys()))
        if unknown:
            raise ValueError(f"Unexpected class labels: {unknown}")
        return np.array([weight_map[int(label)] for label in y_int], dtype=float)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[List[str]] = None,
        **kwargs
    ) -> 'GradientBoostingModel':
        sample_weight = self._compute_sample_weight(y_train, self._params.get('class_weight'))
        if sample_weight is not None:
            kwargs = dict(kwargs)
            kwargs['sample_weight'] = sample_weight
        super().fit(X_train, y_train, feature_names=feature_names, **kwargs)
        return self
    
    @property
    def is_interpretable(self) -> bool:
        return False  # Boosting ensemble is complex
    
    @property
    def supports_feature_importance(self) -> bool:
        return True
    
    def get_staged_predictions(self, X) -> np.ndarray:
        """
        Get predictions at each boosting stage.
        
        Useful for understanding how predictions evolve
        as more trees are added.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of predictions at each stage
        """
        self._check_fitted()
        return np.array(list(self.model.staged_predict(X)))
    
    def get_staged_scores(self, X, y) -> List[float]:
        """
        Get accuracy scores at each boosting stage.
        
        Args:
            X: Features
            y: True labels
            
        Returns:
            List of accuracy scores at each stage
        """
        from sklearn.metrics import accuracy_score
        
        self._check_fitted()
        scores = []
        for pred in self.model.staged_predict(X):
            scores.append(accuracy_score(y, pred))
        return scores
    
    def get_training_deviance(self) -> np.ndarray:
        """
        Get training deviance (loss) at each stage.
        
        Useful for detecting overfitting.
        """
        self._check_fitted()
        return self.model.train_score_
    
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
    
    def plot_learning_curve(self, X_test, y_test, figsize: tuple = (10, 6)):
        """
        Plot learning curve showing train/test deviance vs iterations.
        
        Args:
            X_test: Test features
            y_test: Test labels
            figsize: Figure size
            
        Returns:
            matplotlib figure
        """
        import matplotlib.pyplot as plt
        
        self._check_fitted()
        
        # Get test deviance at each stage
        test_scores = []
        for pred in self.model.staged_decision_function(X_test):
            # Compute deviance (log loss)
            from sklearn.metrics import log_loss
            pred_proba = 1 / (1 + np.exp(-pred))  # sigmoid
            pred_proba = np.clip(pred_proba, 1e-15, 1 - 1e-15)
            test_scores.append(log_loss(y_test, pred_proba))
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(range(1, len(self.model.train_score_) + 1), 
                self.model.train_score_, 'b-', label='Training Deviance')
        ax.plot(range(1, len(test_scores) + 1), 
                test_scores, 'r-', label='Test Deviance')
        ax.set_xlabel('Boosting Iterations')
        ax.set_ylabel('Deviance')
        ax.set_title('Gradient Boosting Learning Curve')
        ax.legend()
        plt.tight_layout()
        
        return fig
    
    def plot_feature_importance(self, n_features: int = 15, figsize: tuple = (10, 8)):
        """
        Plot feature importances.
        
        Args:
            n_features: Number of top features to show
            figsize: Figure size
            
        Returns:
            matplotlib figure
        """
        import matplotlib.pyplot as plt
        
        self._check_fitted()
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:n_features]
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(
            range(len(indices)),
            importances[indices],
            align='center',
            color='forestgreen'
        )
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([self.feature_names[i] for i in indices])
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Gradient Boosting Feature Importances (Top {n_features})')
        plt.tight_layout()
        
        return fig
