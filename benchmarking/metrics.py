"""
Metrics Calculator
==================
Comprehensive metrics calculation for model evaluation.

Includes:
  - Point-estimate metrics (accuracy, F1, ROC-AUC, PR-AUC, MCC, kappa …)
  - Bootstrap confidence intervals (percentile method, deterministic RNG)
  - Paired bootstrap delta test for cross-model comparisons
  - McNemar exact test for paired classification outcomes
  - Holm-Bonferroni multiple-comparison correction
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    cohen_kappa_score, balanced_accuracy_score, log_loss,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)


class MetricsCalculator:
    """
    Comprehensive metrics calculator for bot detection evaluation.
    
    Computes a wide range of classification metrics including:
        - Basic: Accuracy, Precision, Recall, F1
        - Advanced: ROC-AUC, PR-AUC, MCC, Cohen's Kappa
        - Threshold-independent: ROC curve, PR curve
    """
    
    METRIC_DESCRIPTIONS = {
        'accuracy': 'Overall correctness of predictions',
        'precision': 'Proportion of predicted bots that are actual bots',
        'recall': 'Proportion of actual bots correctly identified (sensitivity)',
        'f1': 'Harmonic mean of precision and recall',
        'specificity': 'Proportion of actual humans correctly identified',
        'balanced_accuracy': 'Average of recall for each class',
        'roc_auc': 'Area under ROC curve (ranking quality)',
        'pr_auc': 'Area under Precision-Recall curve',
        'mcc': 'Matthews Correlation Coefficient (-1 to 1)',
        'cohen_kappa': 'Agreement beyond chance',
        'log_loss': 'Cross-entropy loss (lower is better)',
    }
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialize metrics calculator.
        
        Args:
            class_names: Names for classes ['Human', 'Bot']
        """
        self.class_names = class_names or ['Human', 'Bot']
    
    def compute_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute all available metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Probability predictions (optional)
            
        Returns:
            Dictionary of metric names to values
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
        
        # Confusion matrix derived
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Advanced metrics
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        
        # Probability-based metrics
        if y_proba is not None:
            proba = y_proba[:, 1] if len(y_proba.shape) > 1 else y_proba
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, proba)
            except ValueError:
                metrics['roc_auc'] = 0.0
            try:
                metrics['pr_auc'] = average_precision_score(y_true, proba)
            except ValueError:
                metrics['pr_auc'] = 0.0
            try:
                metrics['log_loss'] = log_loss(y_true, y_proba)
            except ValueError:
                metrics['log_loss'] = float('inf')
        
        # Count metrics
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        
        return metrics
    
    def compute_basic_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute basic classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with accuracy, precision, recall, f1
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
        }
    
    def get_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: str = None
    ) -> np.ndarray:
        """
        Get confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: 'true', 'pred', 'all', or None
            
        Returns:
            Confusion matrix array
        """
        return confusion_matrix(y_true, y_pred, normalize=normalize)
    
    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_dict: bool = False
    ) -> Union[str, Dict]:
        """
        Get detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            output_dict: If True, return as dictionary
            
        Returns:
            Classification report string or dictionary
        """
        return classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=output_dict,
            zero_division=0
        )
    
    def get_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Get ROC curve data.
        
        Args:
            y_true: True labels
            y_proba: Probability predictions
            
        Returns:
            Dictionary with fpr, tpr, thresholds
        """
        proba = y_proba[:, 1] if len(y_proba.shape) > 1 else y_proba
        fpr, tpr, thresholds = roc_curve(y_true, proba)
        return {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
        }
    
    def get_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Get Precision-Recall curve data.
        
        Args:
            y_true: True labels
            y_proba: Probability predictions
            
        Returns:
            Dictionary with precision, recall, thresholds
        """
        proba = y_proba[:, 1] if len(y_proba.shape) > 1 else y_proba
        precision, recall, thresholds = precision_recall_curve(y_true, proba)
        return {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
        }
    
    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        metric: str = 'f1'
    ) -> Dict[str, float]:
        """
        Find optimal classification threshold.
        
        Args:
            y_true: True labels
            y_proba: Probability predictions
            metric: Metric to optimize ('f1', 'balanced_accuracy', 'youden')
            
        Returns:
            Dictionary with optimal threshold and metric value
        """
        proba = y_proba[:, 1] if len(y_proba.shape) > 1 else y_proba
        
        best_threshold = 0.5
        best_score = 0
        
        for threshold in np.arange(0.1, 0.9, 0.01):
            y_pred = (proba >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'balanced_accuracy':
                score = balanced_accuracy_score(y_true, y_pred)
            elif metric == 'youden':
                # Youden's J statistic = sensitivity + specificity - 1
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                score = sensitivity + specificity - 1
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return {
            'optimal_threshold': best_threshold,
            'best_score': best_score,
            'metric': metric,
        }
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: bool = False,
        figsize: tuple = (8, 6)
    ):
        """
        Plot confusion matrix heatmap.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Whether to normalize
            figsize: Figure size
            
        Returns:
            matplotlib figure
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        cm = self.get_confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            cm, annot=True, fmt='.2f' if normalize else 'd',
            cmap='Blues', ax=ax,
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        
        plt.tight_layout()
        return fig
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        figsize: tuple = (8, 6)
    ):
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_proba: Probability predictions
            figsize: Figure size
            
        Returns:
            matplotlib figure
        """
        import matplotlib.pyplot as plt
        
        roc_data = self.get_roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba[:, 1] if len(y_proba.shape) > 1 else y_proba)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(roc_data['fpr'], roc_data['tpr'], 'b-', label=f'ROC (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        figsize: tuple = (8, 6)
    ):
        """
        Plot Precision-Recall curve.
        
        Args:
            y_true: True labels
            y_proba: Probability predictions
            figsize: Figure size
            
        Returns:
            matplotlib figure
        """
        import matplotlib.pyplot as plt
        
        pr_data = self.get_precision_recall_curve(y_true, y_proba)
        auc = average_precision_score(y_true, y_proba[:, 1] if len(y_proba.shape) > 1 else y_proba)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(pr_data['recall'], pr_data['precision'], 'b-', label=f'PR (AUC = {auc:.3f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
        """
        Format metrics as a readable string.
        
        Args:
            metrics: Dictionary of metrics
            precision: Decimal places
            
        Returns:
            Formatted string
        """
        lines = []
        for name, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"  {name}: {value:.{precision}f}")
            else:
                lines.append(f"  {name}: {value}")
        return '\n'.join(lines)

    # ------------------------------------------------------------------
    # Statistical inference utilities
    # ------------------------------------------------------------------

    def _compute_metric(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray],
        metric: str,
    ) -> float:
        """Compute a single named metric on provided arrays."""
        if metric == 'accuracy':
            return accuracy_score(y_true, y_pred)
        if metric == 'precision':
            return precision_score(y_true, y_pred, average='binary', zero_division=0)
        if metric == 'recall':
            return recall_score(y_true, y_pred, average='binary', zero_division=0)
        if metric == 'f1':
            return f1_score(y_true, y_pred, average='binary', zero_division=0)
        if metric == 'balanced_accuracy':
            return balanced_accuracy_score(y_true, y_pred)
        if metric == 'mcc':
            return matthews_corrcoef(y_true, y_pred)
        if metric == 'roc_auc':
            if y_proba is None:
                return float('nan')
            proba = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
            try:
                return roc_auc_score(y_true, proba)
            except ValueError:
                return float('nan')
        if metric == 'pr_auc':
            if y_proba is None:
                return float('nan')
            proba = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
            try:
                return average_precision_score(y_true, proba)
            except ValueError:
                return float('nan')
        raise ValueError(f"Unsupported metric for bootstrapping: {metric}")

    def bootstrap_metric_ci(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        metric: str = 'f1',
        n_bootstrap: int = 1000,
        alpha: float = 0.05,
        random_state: int = 2112,
    ) -> Tuple[float, float, float]:
        """
        Compute a bootstrap percentile confidence interval for a single metric.

        Args:
            y_true: Ground-truth labels.
            y_pred: Predicted labels.
            y_proba: Predicted probabilities (optional; required for roc_auc/pr_auc).
            metric: Metric name (f1, roc_auc, pr_auc, mcc, accuracy, …).
            n_bootstrap: Number of bootstrap resamples.
            alpha: Significance level; produces (alpha/2, 1-alpha/2) interval.
            random_state: Seed for reproducibility.

        Returns:
            Tuple of (lower_bound, point_estimate, upper_bound).
        """
        rng = np.random.RandomState(random_state)
        n = len(y_true)
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_proba is not None:
            y_proba = np.asarray(y_proba)

        point = self._compute_metric(y_true, y_pred, y_proba, metric)

        boot_scores = []
        for _ in range(n_bootstrap):
            idx = rng.randint(0, n, n)
            bt = y_true[idx]
            bp = y_pred[idx]
            bpr = y_proba[idx] if y_proba is not None else None
            if len(np.unique(bt)) < 2:
                continue
            score = self._compute_metric(bt, bp, bpr, metric)
            if not np.isnan(score):
                boot_scores.append(score)

        if not boot_scores:
            return (float('nan'), point, float('nan'))

        lower = float(np.percentile(boot_scores, 100 * alpha / 2))
        upper = float(np.percentile(boot_scores, 100 * (1 - alpha / 2)))
        return (lower, point, upper)

    def bootstrap_delta_ci(
        self,
        y_true: np.ndarray,
        preds_a: np.ndarray,
        preds_b: np.ndarray,
        probas_a: Optional[np.ndarray] = None,
        probas_b: Optional[np.ndarray] = None,
        metric: str = 'f1',
        n_bootstrap: int = 1000,
        alpha: float = 0.05,
        random_state: int = 2112,
    ) -> Dict[str, float]:
        """
        Bootstrap-based paired comparison: CI for (metric_B - metric_A) and
        a two-sided p-value estimate via the permutation principle.

        Args:
            y_true: Shared ground-truth labels.
            preds_a: Predictions from model A.
            preds_b: Predictions from model B.
            probas_a: Probabilities from model A (optional).
            probas_b: Probabilities from model B (optional).
            metric: Metric name to compare.
            n_bootstrap: Bootstrap resamples.
            alpha: CI level.
            random_state: Seed for reproducibility.

        Returns:
            Dict with keys: delta, ci_lower, ci_upper, p_value.
        """
        rng = np.random.RandomState(random_state)
        n = len(y_true)
        y_true = np.asarray(y_true)
        preds_a = np.asarray(preds_a)
        preds_b = np.asarray(preds_b)
        if probas_a is not None:
            probas_a = np.asarray(probas_a)
        if probas_b is not None:
            probas_b = np.asarray(probas_b)

        score_a = self._compute_metric(y_true, preds_a, probas_a, metric)
        score_b = self._compute_metric(y_true, preds_b, probas_b, metric)
        observed_delta = score_b - score_a

        boot_deltas = []
        for _ in range(n_bootstrap):
            idx = rng.randint(0, n, n)
            bt = y_true[idx]
            if len(np.unique(bt)) < 2:
                continue
            bpa = preds_a[idx]
            bpb = preds_b[idx]
            bpra = probas_a[idx] if probas_a is not None else None
            bprb = probas_b[idx] if probas_b is not None else None
            sa = self._compute_metric(bt, bpa, bpra, metric)
            sb = self._compute_metric(bt, bpb, bprb, metric)
            if not (np.isnan(sa) or np.isnan(sb)):
                boot_deltas.append(sb - sa)

        if not boot_deltas:
            return {'delta': observed_delta, 'ci_lower': float('nan'),
                    'ci_upper': float('nan'), 'p_value': float('nan')}

        boot_deltas = np.array(boot_deltas)
        ci_lower = float(np.percentile(boot_deltas, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_deltas, 100 * (1 - alpha / 2)))
        # Two-sided p-value: proportion of bootstrap deltas on the opposite side
        p_value = float(np.mean(np.abs(boot_deltas - boot_deltas.mean()) >=
                                abs(observed_delta - boot_deltas.mean())))
        return {
            'delta': observed_delta,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value,
        }

    @staticmethod
    def mcnemar_test(
        y_true: np.ndarray,
        preds_a: np.ndarray,
        preds_b: np.ndarray,
    ) -> Dict[str, float]:
        """
        McNemar's exact test for paired binary classifiers.

        Compares the classification outcomes of model A vs model B on the
        same test set.  The test statistic uses the exact binomial p-value
        (no continuity correction) when b+c < 25, otherwise the asymptotic
        chi-squared approximation.

        Args:
            y_true: Ground-truth binary labels (0/1).
            preds_a: Binary predictions from model A.
            preds_b: Binary predictions from model B.

        Returns:
            Dict with keys: b (A-right B-wrong), c (A-wrong B-right),
            statistic, p_value, test_type.
        """
        from scipy.stats import binom, chi2

        y_true = np.asarray(y_true)
        preds_a = np.asarray(preds_a)
        preds_b = np.asarray(preds_b)

        correct_a = (preds_a == y_true)
        correct_b = (preds_b == y_true)

        b = int(np.sum(correct_a & ~correct_b))   # A right, B wrong
        c = int(np.sum(~correct_a & correct_b))    # A wrong, B right

        if b + c == 0:
            return {'b': b, 'c': c, 'statistic': 0.0, 'p_value': 1.0,
                    'test_type': 'exact'}

        if b + c < 25:
            # Exact binomial two-sided
            p_value = float(2 * min(
                binom.cdf(min(b, c), b + c, 0.5),
                1 - binom.cdf(min(b, c) - 1, b + c, 0.5),
            ))
            statistic = float(min(b, c))
            test_type = 'exact'
        else:
            statistic = float((abs(b - c) - 1) ** 2 / (b + c))
            p_value = float(1 - chi2.cdf(statistic, df=1))
            test_type = 'chi2'

        return {'b': b, 'c': c, 'statistic': statistic, 'p_value': p_value,
                'test_type': test_type}

    @staticmethod
    def holm_bonferroni(p_values: List[float]) -> List[float]:
        """
        Apply Holm-Bonferroni step-down correction to a list of p-values.

        Args:
            p_values: Uncorrected p-values.

        Returns:
            Corrected p-values (same order as input).
        """
        n = len(p_values)
        if n == 0:
            return []
        order = np.argsort(p_values)
        sorted_p = np.array(p_values)[order]
        # Multiply by (n, n-1, …, 1) then take cumulative max to enforce
        # monotonicity of adjusted p-values (standard Holm step-down rule).
        corrected = np.maximum.accumulate(
            sorted_p * np.arange(n, 0, -1)
        ).clip(max=1.0)
        result = np.empty(n)
        result[order] = corrected
        return result.tolist()
