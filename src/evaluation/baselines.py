"""Classical (non-deep-learning) anomaly detection baselines and supervised oracle.

All detectors adhere to a common ``BaseFeatureDetector`` interface operating on
standardized Bode tensors ``(N, seq_len, n_features)`` and supporting both:

1. ``fit_val_f1`` calibration (threshold chosen from healthy validation score
   percentiles to maximize F1 on a labeled validation partition), and
2. ``unsupervised_percentile`` calibration (threshold chosen exclusively from
   healthy validation scores with zero anomaly labels).
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence, Union

import numpy as np
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM

from .metrics import compute_classification_metrics

DEFAULT_PERCENTILES: Sequence[float] = (70.0, 80.0, 90.0, 95.0, 98.0, 99.0, 99.5, 99.9)


def _f1_compat(labels: np.ndarray, predictions: np.ndarray) -> float:
    try:
        return float(f1_score(labels, predictions, zero_division=0))
    except TypeError:
        return float(f1_score(labels, predictions))


class BaseFeatureDetector(ABC):
    """Abstract base class for feature-space anomaly detectors."""

    def __init__(self) -> None:
        self.threshold_: Optional[float] = None
        self.calibration_info_: Optional[Dict[str, Union[str, float]]] = None
        self._is_fitted: bool = False

    @staticmethod
    def _flatten(data: np.ndarray) -> np.ndarray:
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim == 1:
            return arr[:, None]
        return arr.reshape(arr.shape[0], -1)

    def fit(self, train_data: np.ndarray) -> "BaseFeatureDetector":
        """Fit detector exclusively on healthy training samples."""
        X = self._flatten(train_data)
        if X.shape[0] == 0:
            raise ValueError("Cannot fit detector on empty training array")
        self._fit_flat(X)
        self._is_fitted = True
        return self

    @abstractmethod
    def _fit_flat(self, X: np.ndarray) -> None:
        """Fit underlying statistical model on 2-D array ``(N, D)``."""

    @abstractmethod
    def _score_flat(self, X: np.ndarray) -> np.ndarray:
        """Return continuous anomaly scores (higher = more anomalous)."""

    def score(self, data: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Detector must be fitted before scoring")
        return np.asarray(self._score_flat(self._flatten(data)), dtype=np.float64)

    def calibrate_threshold(
        self,
        val_data: np.ndarray,
        val_labels: Optional[np.ndarray] = None,
        percentiles: Sequence[float] = DEFAULT_PERCENTILES,
        unsupervised_percentile: Optional[float] = None,
    ) -> Dict[str, Union[str, float]]:
        """Calibrate decision threshold on validation scores."""
        scores = self.score(val_data)

        if unsupervised_percentile is not None or val_labels is None:
            pct = 95.0 if unsupervised_percentile is None else float(unsupervised_percentile)
            if val_labels is not None:
                normal_scores = scores[np.asarray(val_labels) == 0]
            else:
                normal_scores = scores
            if normal_scores.size == 0:
                raise ValueError("Unsupervised calibration requires healthy validation samples")
            threshold = float(np.percentile(normal_scores, pct))
            info: Dict[str, Union[str, float]] = {
                "mode": "unsupervised_percentile",
                "normal_percentile": pct,
                "threshold": threshold,
            }
            self.threshold_ = threshold
            self.calibration_info_ = info
            return info

        labels = np.asarray(val_labels, dtype=int)
        normal_mask = labels == 0
        if not np.any(normal_mask) or np.all(normal_mask):
            raise ValueError("fit_val_f1 calibration requires both healthy and anomalous labels")

        best_info: Optional[Dict[str, Union[str, float]]] = None
        for pct in percentiles:
            th = float(np.percentile(scores[normal_mask], float(pct)))
            f1 = _f1_compat(labels, (scores > th).astype(int))
            cand: Dict[str, Union[str, float]] = {
                "mode": "fit_val_f1",
                "normal_percentile": float(pct),
                "threshold": th,
                "fit_validation_f1": f1,
            }
            if best_info is None or f1 > float(best_info["fit_validation_f1"]):
                best_info = cand

        assert best_info is not None
        self.threshold_ = float(best_info["threshold"])
        self.calibration_info_ = best_info
        return best_info

    def predict(self, data: np.ndarray) -> np.ndarray:
        if self.threshold_ is None:
            raise RuntimeError("Threshold not calibrated. Call calibrate_threshold() first.")
        return (self.score(data) > self.threshold_).astype(int)

    def evaluate(self, test_data: np.ndarray, test_labels: np.ndarray) -> Dict[str, object]:
        scores = self.score(test_data)
        if self.threshold_ is None:
            raise RuntimeError("Threshold not calibrated. Call calibrate_threshold() first.")
        predictions = (scores > self.threshold_).astype(int)
        labels = np.asarray(test_labels, dtype=int)
        cm = compute_classification_metrics(predictions=predictions, labels=labels, scores=scores)
        return {
            "accuracy": float(cm.accuracy),
            "precision": float(cm.precision),
            "recall": float(cm.recall),
            "f1": float(cm.f1),
            "auc_roc": float(cm.auc_roc),
            "auc_pr": float(cm.auc_pr),
            "threshold": float(self.threshold_),
            "scores": scores,
            "predictions": predictions,
        }


class RawKNNDetector(BaseFeatureDetector):
    """Euclidean k-NN distance detector directly on flattened input space."""

    def __init__(self, n_neighbors: int = 1) -> None:
        super().__init__()
        self.n_neighbors = int(n_neighbors)
        self._nn: Optional[NearestNeighbors] = None

    def _fit_flat(self, X: np.ndarray) -> None:
        k = min(self.n_neighbors, X.shape[0])
        self._nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
        self._nn.fit(X)

    def _score_flat(self, X: np.ndarray) -> np.ndarray:
        assert self._nn is not None
        dists, _ = self._nn.kneighbors(X)
        return dists.mean(axis=1)


class MahalanobisDetector(BaseFeatureDetector):
    """Regularized Mahalanobis distance detector on flattened input space."""

    def __init__(self, ridge: Optional[float] = None) -> None:
        super().__init__()
        self.ridge = ridge
        self._cov = None

    def _fit_flat(self, X: np.ndarray) -> None:
        if self.ridge is None:
            self._cov = LedoitWolf().fit(X)
        else:
            rng = np.random.default_rng(42)
            jitter = float(self.ridge) * rng.standard_normal(X.shape)
            self._cov = EmpiricalCovariance().fit(X + jitter)

    def _score_flat(self, X: np.ndarray) -> np.ndarray:
        assert self._cov is not None
        return np.sqrt(np.clip(self._cov.mahalanobis(X), 0.0, None))


class PCAReconstructionDetector(BaseFeatureDetector):
    """Linear PCA subspace reconstruction-error detector."""

    def __init__(self, n_components: int = 10, seed: int = 42) -> None:
        super().__init__()
        self.n_components = int(n_components)
        self.seed = int(seed)
        self._pca: Optional[PCA] = None

    def _fit_flat(self, X: np.ndarray) -> None:
        k = min(self.n_components, X.shape[0], X.shape[1])
        self._pca = PCA(n_components=k, random_state=self.seed)
        self._pca.fit(X)

    def _score_flat(self, X: np.ndarray) -> np.ndarray:
        assert self._pca is not None
        recon = self._pca.inverse_transform(self._pca.transform(X))
        return np.mean((X - recon) ** 2, axis=1)


class IsolationForestDetector(BaseFeatureDetector):
    """Tree-ensemble Isolation Forest detector."""

    def __init__(self, n_estimators: int = 200, seed: int = 42) -> None:
        super().__init__()
        self.n_estimators = int(n_estimators)
        self.seed = int(seed)
        self._iforest: Optional[IsolationForest] = None

    def _fit_flat(self, X: np.ndarray) -> None:
        self._iforest = IsolationForest(
            n_estimators=self.n_estimators, random_state=self.seed, n_jobs=-1
        )
        self._iforest.fit(X)

    def _score_flat(self, X: np.ndarray) -> np.ndarray:
        assert self._iforest is not None
        return -self._iforest.score_samples(X)


class OneClassSVMDetector(BaseFeatureDetector):
    """Kernel One-Class SVM boundary detector."""

    def __init__(self, nu: float = 0.05, kernel: str = "rbf") -> None:
        super().__init__()
        self.nu = float(nu)
        self.kernel = kernel
        self._ocsvm: Optional[OneClassSVM] = None

    def _fit_flat(self, X: np.ndarray) -> None:
        self._ocsvm = OneClassSVM(nu=self.nu, kernel=self.kernel, gamma="scale")
        self._ocsvm.fit(X)

    def _score_flat(self, X: np.ndarray) -> np.ndarray:
        assert self._ocsvm is not None
        return -self._ocsvm.score_samples(X)


class SupervisedOracleDetector(BaseFeatureDetector):
    """Supervised gradient-boosted oracle quantifying empirical Bayes ceiling."""

    def __init__(self, seed: int = 42) -> None:
        super().__init__()
        self.seed = int(seed)
        self._clf: Optional[HistGradientBoostingClassifier] = None

    def _fit_flat(self, X: np.ndarray) -> None:
        raise NotImplementedError("Use fit_oracle(train_normals, fit_val_data, fit_val_labels)")

    def fit_oracle(
        self,
        train_normals: np.ndarray,
        fit_val_data: np.ndarray,
        fit_val_labels: np.ndarray,
    ) -> "SupervisedOracleDetector":
        X_tr = self._flatten(train_normals)
        X_fv = self._flatten(fit_val_data)
        y_fv = np.asarray(fit_val_labels, dtype=int)
        X_all = np.vstack([X_tr, X_fv])
        y_all = np.concatenate([np.zeros(len(X_tr), dtype=int), y_fv])
        min_leaf = min(20, max(2, int(y_all.sum()) // 4))
        self._clf = HistGradientBoostingClassifier(
            random_state=self.seed, min_samples_leaf=min_leaf
        )
        self._clf.fit(X_all, y_all)
        self._is_fitted = True
        self.threshold_ = 0.5
        self.calibration_info_ = {"mode": "supervised_posterior", "threshold": 0.5}
        return self

    def _score_flat(self, X: np.ndarray) -> np.ndarray:
        assert self._clf is not None
        return self._clf.predict_proba(X)[:, 1]
