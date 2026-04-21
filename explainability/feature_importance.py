"""
Feature Importance Analyzer
===========================
Unified feature importance analysis across different models and methods.
Helps track how bot detection features evolve over time.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from pathlib import Path


def _importance_value(v: Any) -> float:
    if isinstance(v, dict):
        return float(v.get("mean", 0))
    return float(v)


class FeatureImportanceAnalyzer:
    """
    Unified feature importance analyzer for bot detection.
    
    Provides multiple methods to analyze feature importance:
        - Model-specific importance (built-in feature_importances_)
        - Permutation importance
        - SHAP-based importance
        - Statistical importance (correlation, mutual information)
    
    Also supports tracking feature importance over time to
    understand how bot behavior evolves.
    """
    
    def __init__(self, feature_names: List[str] = None):
        """
        Initialize the analyzer.
        
        Args:
            feature_names: List of feature names
        """
        self.feature_names = feature_names
        self.importance_history: List[Dict[str, Any]] = []
    
    def analyze_model_importance(
        self,
        model,
        method: str = 'builtin'
    ) -> Dict[str, float]:
        """
        Get feature importance from a trained model.
        
        Args:
            model: Trained model (sklearn or BaseModel wrapper)
            method: Importance method ('builtin', 'coefficient')
            
        Returns:
            Dictionary of feature importances
        """
        # Extract underlying model if wrapped
        sklearn_model = model.model if hasattr(model, 'model') else model
        
        if hasattr(model, 'feature_names') and model.feature_names:
            self.feature_names = model.feature_names
        
        if method == 'builtin':
            if hasattr(sklearn_model, 'feature_importances_'):
                importances = sklearn_model.feature_importances_
            elif hasattr(sklearn_model, 'coef_'):
                importances = np.abs(sklearn_model.coef_).flatten()
            else:
                raise ValueError(f"Model {type(sklearn_model)} does not support built-in importance")
        elif method == 'coefficient':
            if not hasattr(sklearn_model, 'coef_'):
                raise ValueError("Model does not have coefficients")
            importances = sklearn_model.coef_.flatten()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if self.feature_names is None:
            self.feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        return dict(zip(self.feature_names, importances))
    
    def compute_permutation_importance(
        self,
        model,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        n_repeats: int = 10,
        random_state: int = 2112
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute permutation importance.
        
        Measures the decrease in model performance when a feature
        is randomly shuffled, breaking its relationship with the target.
        
        Args:
            model: Trained model
            X: Features
            y: True labels
            n_repeats: Number of times to permute each feature
            random_state: Random seed
            
        Returns:
            Dictionary with mean and std importance for each feature
        """
        from sklearn.inspection import permutation_importance
        
        sklearn_model = model.model if hasattr(model, 'model') else model
        
        result = permutation_importance(
            sklearn_model, X, y,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1
        )
        
        if self.feature_names is None:
            if isinstance(X, pd.DataFrame):
                self.feature_names = X.columns.tolist()
            else:
                self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        importance_dict = {}
        for i, name in enumerate(self.feature_names):
            importance_dict[name] = {
                'mean': result.importances_mean[i],
                'std': result.importances_std[i],
            }
        
        return importance_dict
    
    def compute_correlation_importance(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        method: str = 'pearson'
    ) -> Dict[str, float]:
        """
        Compute correlation-based feature importance.
        
        Args:
            X: Features
            y: Labels
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Dictionary of absolute correlations with target
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        
        if self.feature_names is None:
            self.feature_names = X.columns.tolist()
        
        correlations = {}
        for col in X.columns:
            if method == 'pearson':
                corr = np.corrcoef(X[col], y)[0, 1]
            elif method == 'spearman':
                from scipy.stats import spearmanr
                corr, _ = spearmanr(X[col], y)
            elif method == 'kendall':
                from scipy.stats import kendalltau
                corr, _ = kendalltau(X[col], y)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            correlations[col] = abs(corr) if not np.isnan(corr) else 0.0
        
        return correlations
    
    def compute_mutual_information(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute mutual information between features and target.
        
        Mutual information measures the amount of information
        obtained about the target by observing the feature.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Dictionary of mutual information scores
        """
        from sklearn.feature_selection import mutual_info_classif
        
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        if self.feature_names is None:
            if isinstance(X, pd.DataFrame):
                self.feature_names = X.columns.tolist()
            else:
                self.feature_names = [f'feature_{i}' for i in range(X_array.shape[1])]
        
        mi_scores = mutual_info_classif(X_array, y, random_state=2112)
        
        return dict(zip(self.feature_names, mi_scores))
    
    def get_top_features(
        self,
        importance_dict: Dict[str, float],
        n: int = 10,
        ascending: bool = False
    ) -> List[tuple]:
        """
        Get top N features by importance.
        
        Args:
            importance_dict: Feature importance dictionary
            n: Number of features to return
            ascending: If True, return least important features
            
        Returns:
            List of (feature_name, importance) tuples
        """
        sorted_features = sorted(
            importance_dict.items(),
            key=lambda x: _importance_value(x[1]),
            reverse=not ascending
        )
        return sorted_features[:n]
    
    def compare_importances(
        self,
        importance_dicts: Dict[str, Dict[str, float]],
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        Compare feature importances across different methods/models.
        
        Args:
            importance_dicts: Dictionary mapping method names to importance dicts
            normalize: Whether to normalize importances to [0, 1]
            
        Returns:
            DataFrame with features as rows and methods as columns
        """
        all_features = set()
        for imp_dict in importance_dicts.values():
            all_features.update(imp_dict.keys())
        
        data = {}
        for method, imp_dict in importance_dicts.items():
            values = []
            for feature in sorted(all_features):
                val = imp_dict.get(feature, 0)
                if isinstance(val, dict):
                    val = val.get('mean', 0)
                values.append(val)
            
            if normalize and max(values) > 0:
                max_val = max(values)
                values = [v / max_val for v in values]
            
            data[method] = values
        
        df = pd.DataFrame(data, index=sorted(all_features))
        df.index.name = 'feature'
        
        return df
    
    def track_importance(
        self,
        importance_dict: Dict[str, float],
        timestamp: str = None,
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        Track feature importance over time.
        
        Useful for monitoring how bot detection features
        change in importance as bots evolve.
        
        Args:
            importance_dict: Current feature importances
            timestamp: Optional timestamp label
            metadata: Optional additional metadata
        """
        from datetime import datetime
        
        entry = {
            'timestamp': timestamp or datetime.now().isoformat(),
            'importances': importance_dict.copy(),
            'metadata': metadata or {}
        }
        
        self.importance_history.append(entry)
    
    def get_importance_trends(self) -> pd.DataFrame:
        """
        Get feature importance trends over time.
        
        Returns:
            DataFrame with timestamps as index and features as columns
        """
        if not self.importance_history:
            raise ValueError("No importance history. Call track_importance() first.")
        
        data = {}
        for entry in self.importance_history:
            data[entry['timestamp']] = entry['importances']
        
        df = pd.DataFrame(data).T
        df.index.name = 'timestamp'
        
        return df
    
    def identify_evolving_features(
        self,
        threshold: float = 0.1
    ) -> Dict[str, Dict[str, float]]:
        """
        Identify features whose importance has changed significantly.
        
        Useful for detecting when bots adapt their behavior.
        
        Args:
            threshold: Minimum change to consider significant
            
        Returns:
            Dictionary of features with significant changes
        """
        if len(self.importance_history) < 2:
            raise ValueError("Need at least 2 history entries to detect evolution")
        
        first = self.importance_history[0]['importances']
        last = self.importance_history[-1]['importances']
        
        evolving = {}
        all_features = set(first.keys()) | set(last.keys())
        
        for feature in all_features:
            first_val = first.get(feature, 0)
            last_val = last.get(feature, 0)
            change = last_val - first_val
            
            if abs(change) >= threshold:
                evolving[feature] = {
                    'initial': first_val,
                    'current': last_val,
                    'change': change,
                    'percent_change': (change / first_val * 100) if first_val != 0 else float('inf')
                }
        
        return evolving
    
    def plot_importance(
        self,
        importance_dict: Dict[str, float],
        n_features: int = 15,
        figsize: tuple = (10, 8),
        title: str = 'Feature Importance'
    ):
        """
        Plot feature importance as horizontal bar chart.
        
        Args:
            importance_dict: Feature importances
            n_features: Number of features to display
            figsize: Figure size
            title: Plot title
            
        Returns:
            matplotlib figure
        """
        import matplotlib.pyplot as plt
        
        # Get top features
        top_features = self.get_top_features(importance_dict, n_features)
        features, values = zip(*top_features)
        
        # Handle dict values (e.g., permutation importance)
        values = [v['mean'] if isinstance(v, dict) else v for v in values]
        
        fig, ax = plt.subplots(figsize=figsize)
        y_pos = range(len(features))
        ax.barh(y_pos, values, color='steelblue', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(title)
        
        plt.tight_layout()
        return fig
    
    def plot_importance_comparison(
        self,
        comparison_df: pd.DataFrame,
        n_features: int = 15,
        figsize: tuple = (12, 8)
    ):
        """
        Plot comparison of feature importances across methods.
        
        Args:
            comparison_df: DataFrame from compare_importances()
            n_features: Number of features to display
            figsize: Figure size
            
        Returns:
            matplotlib figure
        """
        import matplotlib.pyplot as plt
        
        # Get top features by average importance
        avg_importance = comparison_df.mean(axis=1)
        top_features = avg_importance.nlargest(n_features).index
        
        plot_df = comparison_df.loc[top_features]
        
        fig, ax = plt.subplots(figsize=figsize)
        plot_df.plot(kind='barh', ax=ax, alpha=0.7)
        ax.invert_yaxis()
        ax.set_xlabel('Normalized Importance')
        ax.set_title('Feature Importance Comparison Across Methods')
        ax.legend(title='Method', loc='lower right')
        
        plt.tight_layout()
        return fig
    
    def save_history(self, path: str) -> None:
        """Save importance history to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(path, 'w') as f:
            json.dump(self.importance_history, f, indent=2)
    
    def load_history(self, path: str) -> None:
        """Load importance history from disk."""
        import json
        with open(path, 'r') as f:
            self.importance_history = json.load(f)
