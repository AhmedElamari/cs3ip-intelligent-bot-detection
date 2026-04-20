"""
Explainability Module (XAI)
===========================
Provides interpretability tools for ML models including:
- SHAP (SHapley Additive exPlanations)
- LIME (Local Interpretable Model-agnostic Explanations)
- Feature importance analysis
"""

from .shap_explainer import SHAPExplainer
from .lime_explainer import LIMEExplainer
from .feature_importance import FeatureImportanceAnalyzer

__all__ = [
    'SHAPExplainer',
    'LIMEExplainer',
    'FeatureImportanceAnalyzer',
]
