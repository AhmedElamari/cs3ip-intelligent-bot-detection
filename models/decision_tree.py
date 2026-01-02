"""
Decision Tree Model
===================
Tree-based classifier with high interpretability.
Can be visualized as a flowchart of decisions.
"""

from typing import Any, Optional, Dict
from sklearn.tree import DecisionTreeClassifier
from .base import BaseModel


class DecisionTreeModel(BaseModel):
    """
    Decision Tree classifier for bot detection.
    
    Decision trees make predictions by learning simple
    decision rules from the features, creating an easily
    interpretable tree structure.
    
    Advantages:
        - Highly interpretable (can be visualized)
        - No feature scaling required
        - Handles non-linear relationships
        - Can capture feature interactions
        
    Disadvantages:
        - Prone to overfitting
        - Sensitive to small changes in data
        - Can create biased trees if classes are imbalanced
    """
    
    def __init__(self, random_state: int = 2112, **kwargs):
        super().__init__(name='Decision Tree', random_state=random_state)
        self._params = {
            'random_state': random_state,
            'max_depth': kwargs.get('max_depth', 10),
            'min_samples_split': kwargs.get('min_samples_split', 2),
            'min_samples_leaf': kwargs.get('min_samples_leaf', 1),
            'class_weight': kwargs.get('class_weight', 'balanced'),
            'criterion': kwargs.get('criterion', 'gini'),
        }
        self.model = self._create_model(**self._params)
    
    def _create_model(self, **kwargs) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(**kwargs)
    
    @property
    def is_interpretable(self) -> bool:
        return True
    
    @property
    def supports_feature_importance(self) -> bool:
        return True
    
    def get_tree_depth(self) -> int:
        """Get the actual depth of the trained tree."""
        self._check_fitted()
        return self.model.get_depth()
    
    def get_n_leaves(self) -> int:
        """Get the number of leaves in the tree."""
        self._check_fitted()
        return self.model.get_n_leaves()
    
    def get_decision_path(self, X) -> Dict[str, Any]:
        """
        Get the decision path for samples.
        
        Returns information about which nodes each sample
        traversed to reach its prediction.
        """
        self._check_fitted()
        indicator, n_nodes_ptr = self.model.decision_path(X)
        return {
            'indicator': indicator,
            'n_nodes_ptr': n_nodes_ptr,
        }
    
    def export_tree_rules(self, max_depth: int = 3) -> str:
        """
        Export tree as human-readable rules.
        
        Args:
            max_depth: Maximum depth to show
            
        Returns:
            String representation of tree rules
        """
        from sklearn.tree import export_text
        self._check_fitted()
        return export_text(
            self.model,
            feature_names=self.feature_names,
            max_depth=max_depth
        )
    
    def plot_tree(self, figsize: tuple = (20, 10), max_depth: int = 3):
        """
        Plot the decision tree.
        
        Args:
            figsize: Figure size
            max_depth: Maximum depth to display
            
        Returns:
            matplotlib figure
        """
        import matplotlib.pyplot as plt
        from sklearn.tree import plot_tree
        
        self._check_fitted()
        fig, ax = plt.subplots(figsize=figsize)
        plot_tree(
            self.model,
            feature_names=self.feature_names,
            class_names=['Human', 'Bot'],
            filled=True,
            rounded=True,
            max_depth=max_depth,
            ax=ax
        )
        plt.title(f'Decision Tree (depth={self.get_tree_depth()}, leaves={self.get_n_leaves()})')
        return fig
