"""
TabNet Model
============
PyTorch TabNet wrapper implementing the project BaseModel interface.

TabNet (Arik & Pfister, 2021) uses sequential attention to perform sparse,
instance-wise feature selection at each decision step.  This gives it
intrinsic interpretability (feature-mask outputs) while achieving
performance competitive with tree ensembles on tabular data.

Key features of this wrapper:
- Ghost Batch Normalisation: virtual_batch_size must be <= batch_size and
  is auto-capped when training set is smaller than the configured batch.
- Hardware safety: automatic CPU fallback; bounded CUDA OOM retry.
- Determinism: NumPy + PyTorch seeds fixed to random_state.
- Early stopping: pass X_val/y_val via prepare_eval_set() before fit().
- Imbalance: class_weight='balanced' or dict -> per-sample weights.

Install:  pip install -r requirements-dl.txt
"""

import time
import warnings
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseModel


def _require_tabnet():
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "pytorch-tabnet is not installed.  Install the deep-learning "
            "profile:\n  pip install -r requirements-dl.txt"
        ) from exc


def _sample_weights(y: np.ndarray, class_weight) -> Optional[np.ndarray]:
    """Convert class_weight specification to a per-sample weight array."""
    if class_weight is None:
        return None
    y_int = np.asarray(y, dtype=int)
    if class_weight == "balanced":
        classes, counts = np.unique(y_int, return_counts=True)
        total = len(y_int)
        wmap = {c: total / (len(classes) * n) for c, n in zip(classes, counts)}
    elif isinstance(class_weight, dict):
        wmap = {int(k): v for k, v in class_weight.items()}
        missing = sorted(int(x) for x in (set(y_int) - set(wmap)))
        if missing:
            raise ValueError(f"class_weight dict missing label(s): {missing}")
    else:
        raise ValueError(f"Unsupported class_weight: {class_weight!r}")
    return np.array([wmap[int(lbl)] for lbl in y_int], dtype=np.float64)


class TabNetModel(BaseModel):
    """
    TabNet classifier for bot detection.

    Wraps ``pytorch_tabnet.tab_model.TabNetClassifier`` and satisfies the
    ``BaseModel`` contract so it integrates into both the single-model CLI
    (``main.py``) and the benchmark pipeline (``run_benchmark.py``).

    Key parameters (all override-able via ``**kwargs``):
        n_d / n_a   : Width of decision and attention embedding layers.
        n_steps     : Number of sequential attention steps.
        gamma       : Relaxation factor for feature re-use (1 = no re-use).
        lambda_sparse : Sparsity regularisation coefficient.
        batch_size  : Mini-batch size for Ghost Batch Normalisation.
        virtual_batch_size : Ghost batch size (must be <= batch_size).
        momentum    : Ghost BN momentum.
        mask_type   : 'sparsemax' (default) or 'entmax'.
        max_epochs  : Training epoch cap.
        patience    : Early-stopping patience (epochs without improvement).
        class_weight: 'balanced', None, or class->weight dict.
    """

    def __init__(self, random_state: int = 2112, **kwargs):
        super().__init__(name="TabNet", random_state=random_state)
        self._params = {
            "random_state": random_state,
            "n_d": kwargs.get("n_d", 32),
            "n_a": kwargs.get("n_a", 32),
            "n_steps": kwargs.get("n_steps", 3),
            "gamma": kwargs.get("gamma", 1.3),
            "lambda_sparse": kwargs.get("lambda_sparse", 1e-3),
            "learning_rate": kwargs.get("learning_rate", 2e-2),
            "device_name": kwargs.get("device_name", "auto"),
            "batch_size": kwargs.get("batch_size", 1024),
            "virtual_batch_size": kwargs.get("virtual_batch_size", 128),
            "momentum": kwargs.get("momentum", 0.02),
            "mask_type": kwargs.get("mask_type", "sparsemax"),
            "max_epochs": kwargs.get("max_epochs", 200),
            "patience": kwargs.get("patience", 20),
            "class_weight": kwargs.get("class_weight", "balanced"),
            # Categorical embedding indices/dims (empty = all numeric)
            "cat_idxs": kwargs.get("cat_idxs", []),
            "cat_dims": kwargs.get("cat_dims", []),
        }
        # Eval set for early stopping - set via prepare_eval_set()
        self._eval_set: Optional[list] = None
        self.model = None  # Created lazily in fit()

    # ------------------------------------------------------------------
    # BaseModel abstract interface
    # ------------------------------------------------------------------

    def _create_model(self, **kwargs) -> Any:
        _require_tabnet()
        from pytorch_tabnet.tab_model import TabNetClassifier
        seed = kwargs.get("random_state", 2112)
        self._seed_everything(seed)
        return TabNetClassifier(
            n_d=kwargs["n_d"],
            n_a=kwargs["n_a"],
            n_steps=kwargs["n_steps"],
            gamma=kwargs["gamma"],
            lambda_sparse=kwargs["lambda_sparse"],
            optimizer_params={"lr": kwargs.get("learning_rate", 2e-2)},
            cat_idxs=kwargs.get("cat_idxs", []),
            cat_dims=kwargs.get("cat_dims", []),
            momentum=kwargs["momentum"],
            mask_type=kwargs["mask_type"],
            device_name=kwargs.get("device_name", "auto"),
            seed=seed,
            verbose=0,
        )

    @property
    def is_interpretable(self) -> bool:
        return True

    @property
    def supports_feature_importance(self) -> bool:
        return True

    def get_runtime_metadata(self) -> Dict[str, Any]:
        """Return requested vs actual compute device after fit (TabNet / PyTorch)."""
        requested = str(self._params.get("device_name", "auto"))
        out: Dict[str, Any] = {
            "requested_device": requested,
            "actual_device": None,
            "cuda_available": None,
        }
        try:
            import torch

            out["cuda_available"] = bool(torch.cuda.is_available())
        except ImportError:
            out["cuda_available"] = None

        if self.model is None:
            return out

        try:
            network = getattr(self.model, "network", None)
            if network is not None:
                param = next(network.parameters(), None)
                if param is not None:
                    out["actual_device"] = str(param.device)
                    return out
        except (StopIteration, RuntimeError, AttributeError):
            pass

        if out["actual_device"] is None:
            if requested == "auto":
                out["actual_device"] = "cuda" if out["cuda_available"] else "cpu"
            else:
                out["actual_device"] = requested
        return out

    # ------------------------------------------------------------------
    # Eval-set hook for early stopping
    # ------------------------------------------------------------------

    def prepare_eval_set(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "TabNetModel":
        """Register validation data for early stopping inside fit()."""
        self._eval_set = [(np.asarray(X_val, dtype=np.float32),
                           np.asarray(y_val, dtype=int))]
        return self

    # ------------------------------------------------------------------
    # fit / predict / predict_proba
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[List[str]] = None,
        **kwargs,
    ) -> "TabNetModel":
        _require_tabnet()

        X_np = np.asarray(X_train, dtype=np.float32)
        # Defense-in-depth: guarantee no NaN/Inf reaches TabNet (benchmark path
        # may not run TabNetPrep; training path already does explicit imputation).
        X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)
        y_np = np.asarray(y_train, dtype=int)

        # Feature names
        if feature_names is not None:
            self.feature_names = feature_names
        elif hasattr(X_train, "columns"):
            self.feature_names = list(X_train.columns)
        else:
            self.feature_names = [f"feature_{i}" for i in range(X_np.shape[1])]

        # Hardware-safe batch sizes
        batch_size, virtual_batch_size = self._safe_batch_sizes(
            len(X_np),
            self._params["batch_size"],
            self._params["virtual_batch_size"],
        )

        # Build model
        params = dict(self._params)
        params.update({"batch_size": batch_size, "virtual_batch_size": virtual_batch_size})
        self.model = self._create_model(**params)

        # Class weights -> per-sample weights (TabNet: 0 = none, array = custom)
        weights = _sample_weights(y_np, self._params.get("class_weight"))
        weights_arg = weights if weights is not None else 0

        eval_set = self._eval_set or []
        eval_name = ["val"] if eval_set else []

        start = time.time()
        self._fit_with_oom_retry(
            X_np, y_np, weights_arg, eval_set, eval_name,
            batch_size, virtual_batch_size, params,
        )
        self.training_time = time.time() - start
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X_np = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        return self.model.predict(X_np)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        # TabNetClassifier.predict_proba returns Nx2 for binary classification
        X_np = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        return self.model.predict_proba(X_np)

    # ------------------------------------------------------------------
    # Feature importance (uses TabNet's built-in attention masks)
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Return mean attention-mask importance from pytorch-tabnet."""
        self._check_fitted()
        if not hasattr(self.model, "feature_importances_"):
            return None
        return dict(zip(self.feature_names, self.model.feature_importances_))

    # ------------------------------------------------------------------
    # Serialisation override (torch-native + pickle fallback)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        import pickle
        from pathlib import Path
        self._check_fitted()
        p = self._validate_output_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Save TabNet model via its own save mechanism
        tabnet_path = str(p.with_suffix(""))
        self.model.save_model(tabnet_path)
        meta = {
            "name": self.name,
            "feature_names": self.feature_names,
            "params": {k: v for k, v in self._params.items()
                       if not isinstance(v, np.ndarray)},
            "training_time": self.training_time,
            "tabnet_path": tabnet_path,
        }
        with open(p, "wb") as f:
            pickle.dump(meta, f)

    @classmethod
    def load(cls, path: str, trusted_source: bool = False) -> "TabNetModel":
        """Load a TabNetModel from disk.

        Restores both the metadata from pickle and the underlying
        TabNetClassifier model from its native save format.

        Args:
            path: Pickle file path created by ``save``.
            trusted_source: Must be True to load. Pickle can execute arbitrary code.

        Returns:
            TabNetModel instance with restored state.
        """
        import pickle
        import warnings

        p = cls._validate_output_path(path)
        if not trusted_source:
            raise ValueError(
                "Refusing to load pickle from an untrusted source. "
                "Pass trusted_source=True only for files you fully trust."
            )
        warnings.warn(
            "Loading a pickled model. Only load artifacts from trusted sources.",
            UserWarning,
            stacklevel=2,
        )
        with open(p, "rb") as f:
            data = pickle.load(f)

        # Reconstruct TabNetModel instance
        instance = cls.__new__(cls)
        instance.name = data.get("name", "TabNet")
        instance.feature_names = data.get("feature_names", [])
        instance._params = data.get("params", {})
        instance.training_time = data.get("training_time", 0.0)
        instance.is_fitted = True
        instance.random_state = instance._params.get("random_state", 2112)
        instance._eval_set = None

        # Restore TabNet model from native checkpoint
        tabnet_path = data.get("tabnet_path")
        if tabnet_path:
            _require_tabnet()
            from pathlib import Path
            from pytorch_tabnet.tab_model import TabNetClassifier
            instance.model = TabNetClassifier()
            zip_path = tabnet_path if str(tabnet_path).endswith(".zip") else f"{tabnet_path}.zip"
            if not Path(zip_path).exists():
                raise FileNotFoundError(f"TabNet checkpoint not found: {zip_path}")
            instance.model.load_model(zip_path)
        else:
            # Fallback: use pickled model if available
            instance.model = data.get("model")

        return instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _seed_everything(seed: int) -> None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            pass

    @staticmethod
    def _safe_batch_sizes(n_samples: int, batch_size: int, virtual_batch_size: int):
        """Ensure batch/virtual_batch sizes are consistent with dataset size."""
        batch_size = min(batch_size, n_samples)
        virtual_batch_size = min(virtual_batch_size, batch_size)
        # virtual_batch_size must divide batch_size cleanly
        while batch_size % virtual_batch_size != 0 and virtual_batch_size > 1:
            virtual_batch_size -= 1
        return batch_size, max(1, virtual_batch_size)

    def _fit_with_oom_retry(
        self, X_np, y_np, weights_arg, eval_set, eval_name,
        batch_size, virtual_batch_size, params, max_retries=3,
    ) -> None:
        """Attempt fit; on CUDA OOM halve batch sizes and retry."""
        for attempt in range(max_retries):
            try:
                self.model.fit(
                    X_np, y_np,
                    eval_set=eval_set,
                    eval_name=eval_name,
                    weights=weights_arg,
                    max_epochs=params["max_epochs"],
                    patience=params["patience"],
                    batch_size=batch_size,
                    virtual_batch_size=virtual_batch_size,
                )
                return
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or attempt == max_retries - 1:
                    raise
                warnings.warn(
                    f"CUDA OOM on attempt {attempt + 1}. "
                    f"Halving batch sizes and retrying.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                batch_size = max(16, batch_size // 2)
                virtual_batch_size = max(8, virtual_batch_size // 2)
                # Re-validate divisibility constraint: virtual_batch_size must divide batch_size
                batch_size, virtual_batch_size = self._safe_batch_sizes(
                    len(X_np), batch_size, virtual_batch_size
                )
                # Rebuild model with adjusted sizes
                params = dict(params)
                params["batch_size"] = batch_size
                params["virtual_batch_size"] = virtual_batch_size
                self.model = self._create_model(**params)

    def get_params(self) -> Dict[str, Any]:
        return dict(self._params)
