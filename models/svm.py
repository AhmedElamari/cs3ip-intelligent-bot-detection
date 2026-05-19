"""
Support Vector Machine Model
============================
SVM classifier with various kernel options.
Less interpretable but powerful for complex decision boundaries.
"""

from typing import Any
from sklearn.svm import SVC
from .base import BaseModel


class SVMModel(BaseModel):
    """
    Support Vector Machine classifier for bot detection.
    
    SVMs find the optimal hyperplane that separates classes
    with maximum margin. Different kernels allow capturing
    various patterns in the data.
    
    Advantages:
        - Effective in high-dimensional spaces
        - Memory efficient (uses support vectors)
        - Versatile through different kernel functions
        
    Disadvantages:
        - Less interpretable, especially with non-linear kernels
        - Computationally expensive for large datasets
        - Sensitive to feature scaling
    """
    
    def __init__(self, random_state: int = 2112, **kwargs):
        super().__init__(name='Support Vector Machine', random_state=random_state)
        self._params = {
            'random_state': random_state,
            'kernel': kwargs.get('kernel', 'rbf'),
            'C': kwargs.get('C', 1.0),
            'gamma': kwargs.get('gamma', 'scale'),
            'class_weight': kwargs.get('class_weight', 'balanced'),
            # Platt scaling: needed for ROC/PR/threshold analysis (costly but standard).
            'probability': kwargs.get('probability', True),
        }
        self.model = self._create_model(**self._params)
    
    def _create_model(self, **kwargs) -> SVC:
        return SVC(**kwargs)
    
    @property
    def is_interpretable(self) -> bool:
        # RBF/poly are black-box; linear kernel ≈ weighted feature audit.
        return self._params.get('kernel') == 'linear'
    
    @property
    def supports_feature_importance(self) -> bool:
        return self._params.get('kernel') == 'linear'
    
    def get_support_vector_count(self) -> int:
        """Get the number of support vectors."""
        self._check_fitted()
        return len(self.model.support_)
    
    def get_support_vector_ratio(self) -> float:
        """
        Get the ratio of support vectors to training samples.
        
        A high ratio might indicate overfitting or that the
        decision boundary is complex.
        """
        self._check_fitted()
        if not hasattr(self.model, 'shape_fit_'):
            raise RuntimeError("Training sample count not available on the fitted model.")
        n_samples = self.model.shape_fit_[0]
        return len(self.model.support_) / n_samples
