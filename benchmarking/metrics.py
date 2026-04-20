"""Metrics calculator: point estimates, bootstrap CIs, paired delta test, McNemar, Holm-Bonferroni."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    cohen_kappa_score, balanced_accuracy_score, log_loss,
    confusion_matrix, classification_report, precision_recall_curve
)


class MetricsCalculator:
    """Metrics for bot detection: accuracy, F1, ROC-AUC, PR-AUC, MCC, bootstrap CIs, McNemar, Holm-Bonferroni."""
    
    METRIC_DESCRIPTIONS = {
        'accuracy': 'Overall correctness of predictions',
        'precision': 'Proportion of predicted bots that are actual bots',
        'recall': 'Proportion of actual bots correctly identified (sensitivity)',
        'f1': 'Harmonic mean of precision and recall (binary positive class)',
        'f1_macro': 'Unweighted mean of per-class F1 scores',
        'f1_weighted': 'Support-weighted mean of per-class F1 scores',
        'specificity': 'Proportion of actual humans correctly identified',
        'balanced_accuracy': 'Average of recall for each class',
        'roc_auc': 'Area under ROC curve (ranking quality)',
        'pr_auc': 'Area under Precision-Recall curve',
        'mcc': 'Matthews Correlation Coefficient (-1 to 1)',
        'cohen_kappa': 'Agreement beyond chance',
        'log_loss': 'Cross-entropy loss (lower is better)',
    }
    
    def __init__(self, class_names: List[str] = None):
        """Initialize with optional class names (default: Human, Bot)."""
        self.class_names = class_names or ['Human', 'Bot']

    @staticmethod
    def _positive_class_proba(y_proba: np.ndarray) -> np.ndarray:
        """Return 1-D probabilities for the positive class."""
        proba = np.asarray(y_proba)
        return proba[:, 1] if proba.ndim > 1 else proba

    @staticmethod
    def _valid_bootstrap_samples(
        rng: np.random.RandomState,
        n: int,
        n_bootstrap: int,
        y_true: np.ndarray,
    ):
        """Yield bootstrap indices where both classes appear in the resample."""
        for _ in range(n_bootstrap):
            idx = rng.randint(0, n, n)
            if len(np.unique(y_true[idx])) >= 2:
                yield idx
    
    def compute_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Compute all available metrics for y_true, y_pred, and optional y_proba."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
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
            proba = self._positive_class_proba(y_proba)
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
        return confusion_matrix(y_true, y_pred, labels=[0, 1], normalize=normalize)
    
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
        proba = self._positive_class_proba(y_proba)
        precision, recall, thresholds = precision_recall_curve(y_true, proba)
        return {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
        }
    
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
        basic_metric_fns = {
            'accuracy': accuracy_score,
            'precision': lambda yt, yp: precision_score(yt, yp, average='binary', zero_division=0),
            'recall': lambda yt, yp: recall_score(yt, yp, average='binary', zero_division=0),
            'f1': lambda yt, yp: f1_score(yt, yp, average='binary', zero_division=0),
            'f1_macro': lambda yt, yp: f1_score(yt, yp, average='macro', zero_division=0),
            'f1_weighted': lambda yt, yp: f1_score(yt, yp, average='weighted', zero_division=0),
            'balanced_accuracy': balanced_accuracy_score,
            'mcc': matthews_corrcoef,
        }
        metric_fn = basic_metric_fns.get(metric)
        if metric_fn is not None:
            return metric_fn(y_true, y_pred)

        if metric in ('roc_auc', 'pr_auc'):
            if y_proba is None:
                return float('nan')
            proba = self._positive_class_proba(y_proba)
            scorer = roc_auc_score if metric == 'roc_auc' else average_precision_score
            try:
                return scorer(y_true, proba)
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
        for idx in self._valid_bootstrap_samples(rng, n, n_bootstrap, y_true):
            bt, bp = y_true[idx], y_pred[idx]
            bpr = y_proba[idx] if y_proba is not None else None
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
        for idx in self._valid_bootstrap_samples(rng, n, n_bootstrap, y_true):
            bt = y_true[idx]
            bpra = probas_a[idx] if probas_a is not None else None
            bprb = probas_b[idx] if probas_b is not None else None
            sa = self._compute_metric(bt, preds_a[idx], bpra, metric)
            sb = self._compute_metric(bt, preds_b[idx], bprb, metric)
            if not (np.isnan(sa) or np.isnan(sb)):
                boot_deltas.append(sb - sa)

        if not boot_deltas:
            return {'delta': observed_delta, 'ci_lower': float('nan'),
                    'ci_upper': float('nan'), 'p_value': float('nan')}

        boot_deltas = np.array(boot_deltas)
        ci_lower = float(np.percentile(boot_deltas, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_deltas, 100 * (1 - alpha / 2)))
        # Two-sided p-value under null H0: delta == 0 (null-centered, not mean-centered).
        # This avoids non-informative p-values when observed_delta is close to boot mean.
        p_left = float(np.mean(boot_deltas <= 0.0))
        p_right = float(np.mean(boot_deltas >= 0.0))
        p_value = float(min(1.0, 2.0 * min(p_left, p_right)))
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

        try:
            from scipy.stats import binom, chi2
        except ImportError:
            return {
                'b': b,
                'c': c,
                'statistic': float('nan'),
                'p_value': float('nan'),
                'test_type': 'unavailable',
            }

        if b + c < 25:
            # Exact binomial two-sided
            p_value = float(min(1.0, 2 * min(
                binom.cdf(min(b, c), b + c, 0.5),
                1 - binom.cdf(min(b, c) - 1, b + c, 0.5),
            )))
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

        Non-finite inputs (NaN, +/-inf) are excluded from the multiplicity
        count and returned as NaN so they do not inflate corrections on
        valid hypotheses.

        Args:
            p_values: Uncorrected p-values (may contain NaN or inf).

        Returns:
            Corrected p-values in the same order as input; non-finite
            positions are set to NaN.
        """
        arr = np.asarray(p_values, dtype=float)
        if arr.size == 0:
            return []

        out = np.full(arr.size, np.nan)
        finite_mask = np.isfinite(arr)
        if not np.any(finite_mask):
            return out.tolist()

        finite_idx = np.where(finite_mask)[0]
        p = np.clip(arr[finite_mask], 0.0, 1.0)
        m = p.size

        order = np.argsort(p, kind='mergesort')
        # Multiply by (m, m-1, …, 1) then take cumulative max to enforce
        # monotonicity, then clip to [0, 1] (standard Holm step-down rule).
        adjusted = np.minimum(1.0, np.maximum.accumulate(p[order] * np.arange(m, 0, -1)))
        out[finite_idx[order]] = adjusted
        return out.tolist()
