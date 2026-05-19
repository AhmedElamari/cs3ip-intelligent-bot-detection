"""
Logistic Regression Model
=========================
Linear model for binary classification.
Highly interpretable through coefficient analysis.
"""

from typing import Any, Dict, Optional
from sklearn.linear_model import LogisticRegression
from .base import BaseModel


class LogisticRegressionModel(BaseModel):
    """
    Logistic Regression classifier for bot detection.
    
    This is a highly interpretable model where coefficients
    directly indicate feature importance and direction of effect.
    
    Advantages:
        - Highly interpretable
        - Fast training and prediction
        - Provides probability estimates
        - Works well with linearly separable data
        
    Disadvantages:
        - Assumes linear relationship between features and log-odds
        - May underperform on complex, non-linear patterns
    """
    
    def __init__(self, random_state: int = 2112, **kwargs):
        super().__init__(name='Logistic Regression', random_state=random_state)
        self._params = {
            'random_state': random_state,
            'max_iter': kwargs.get('max_iter', 1000),
            'C': kwargs.get('C', 1.0),
            'class_weight': kwargs.get('class_weight', 'balanced'),
            'solver': kwargs.get('solver', 'lbfgs'),
        }
        self.model = self._create_model(**self._params)
    
    def _create_model(self, **kwargs) -> LogisticRegression:
        return LogisticRegression(**kwargs)
    
    @property
    def is_interpretable(self) -> bool:
        return True  # Coefficients = direct bot/human odds direction per feature.
    
    @property
    def supports_feature_importance(self) -> bool:
        return True
    
    def get_coefficients(self) -> Optional[Dict[str, float]]:
        """
        Get model coefficients mapped to feature names.
        
        Positive coefficients indicate features that increase
        the probability of being classified as a bot.
        
        Returns:
            Dictionary of feature names to coefficients
        """
        self._check_fitted()
        coefficients = self.model.coef_.flatten()
        return dict(zip(self.feature_names, coefficients))
    
    def get_odds_ratios(self) -> Optional[Dict[str, float]]:
        """
        Get odds ratios for each feature.
        
        Odds ratio > 1 means the feature increases bot probability.
        Odds ratio < 1 means the feature decreases bot probability.
        
        Returns:
            Dictionary of feature names to odds ratios
        """
        import numpy as np
        self._check_fitted()
        coefficients = self.model.coef_.flatten()
        odds_ratios = np.exp(coefficients)
        return dict(zip(self.feature_names, odds_ratios))
