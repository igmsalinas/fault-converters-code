"""Framework-independent anomaly scoring for exported CARLA embeddings."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def compress_reference_bank(
    reference_embeddings: np.ndarray,
    n_prototypes: int,
    normalize: bool = True,
    random_state: int = 42,
) -> np.ndarray:
    """Compress a reference embedding bank into spherical/Euclidean K-Means prototypes."""
    from sklearn.cluster import KMeans

    ref = np.asarray(reference_embeddings, dtype=np.float32)
    if ref.ndim != 2 or len(ref) == 0:
        raise ValueError("Reference embeddings must be a non-empty 2-D array.")
    if n_prototypes < 1:
        raise ValueError("n_prototypes must be >= 1.")
    if normalize:
        ref = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-8)
    if n_prototypes >= len(ref):
        return ref.copy()

    km = KMeans(n_clusters=int(n_prototypes), random_state=random_state, n_init=10)
    km.fit(ref)
    centers = km.cluster_centers_.astype(np.float32)
    if normalize:
        centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)
    return centers


class EmbeddingAnomalyScorer:
    """Reusable CARLA scorer with reference-bank state fitted exactly once."""

    def __init__(
        self,
        reference_embeddings: np.ndarray,
        method: str = "knn",
        k: int = 5,
        normalize: bool = True,
        n_prototypes: Optional[int] = None,
        random_state: int = 42,
    ):
        reference = np.asarray(reference_embeddings, dtype=np.float32)
        if reference.ndim != 2 or len(reference) == 0:
            raise ValueError("Reference embeddings must be a non-empty 2-D array.")
        self.method = method
        self.normalize = normalize
        if normalize:
            reference = reference / (
                np.linalg.norm(reference, axis=1, keepdims=True) + 1e-8
            )
        if n_prototypes is not None and int(n_prototypes) < len(reference):
            reference = compress_reference_bank(
                reference,
                n_prototypes=int(n_prototypes),
                normalize=normalize,
                random_state=random_state,
            )
        self.reference_embeddings = reference
        self.k = min(int(k), len(reference)) if method == "knn" and int(k) >= 1 else int(k)
        self.shrinkage_: Optional[float] = None

        if method == "knn":
            if not 1 <= self.k <= len(reference):
                raise ValueError(f"k must be between 1 and {len(reference)}")
            self._ref_f64 = reference.astype(np.float64)
            self._ref_sq_f64 = np.sum(self._ref_f64 * self._ref_f64, axis=1)[None, :]
        elif method == "centroid":
            self._centroid = np.mean(reference, axis=0, keepdims=True)
        elif method == "mahalanobis":
            self._mean = np.mean(reference, axis=0)
            covariance = np.cov(reference, rowvar=False)
            covariance += np.eye(covariance.shape[0]) * 1e-6
            self._covariance_inverse = np.linalg.pinv(covariance).astype(np.float32)
        elif method == "ledoit_wolf":
            from sklearn.covariance import LedoitWolf

            ref_f64 = reference.astype(np.float64)
            lw = LedoitWolf(store_precision=False).fit(ref_f64)
            cov_lw = lw.covariance_
            tr_cov = float(np.trace(cov_lw))
            tr_cov_sq = float(np.sum(cov_lw * cov_lw))
            # Spectral floor at 0.5 * participation-ratio manifold variance:
            # preserves exact Ledoit-Wolf whitening on active manifold axes while
            # preventing collapsed ambient dimensions (d >> rank_eff) from blowing up.
            manifold_var = (tr_cov_sq / tr_cov) if tr_cov > 1e-12 else 1.0
            eig_floor = max(0.4 * manifold_var, 1e-6)
            eigvals, eigvecs = np.linalg.eigh(cov_lw)
            clamped_inv = 1.0 / np.maximum(eigvals, eig_floor)
            precision = (eigvecs * clamped_inv[None, :]) @ eigvecs.T
            self._mean = lw.location_.astype(np.float32)
            self._covariance_inverse = precision.astype(np.float32)
            self.shrinkage_ = float(lw.shrinkage_)
        elif method != "cosine":
            raise ValueError(f"Unknown anomaly scoring method: {method}")

    def score(
        self, embeddings: np.ndarray, exclude_self: bool = False
    ) -> np.ndarray:
        """Score one or more projection embeddings."""
        values = np.asarray(embeddings, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("Embeddings must be a 2-D array.")
        if values.shape[1] != self.reference_embeddings.shape[1]:
            raise ValueError("Embedding and reference dimensions do not match.")
        if self.normalize:
            values = values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-8)

        if self.method == "knn":
            val_f64 = values.astype(np.float64)
            val_sq = np.sum(val_f64 * val_f64, axis=1, keepdims=True)
            sq_dists = np.maximum(
                0.0, val_sq + self._ref_sq_f64 - 2.0 * (val_f64 @ self._ref_f64.T)
            )
            k_eff = self.k + 1 if exclude_self else self.k
            k_eff = min(k_eff, sq_dists.shape[1])
            if k_eff < sq_dists.shape[1]:
                part = np.partition(sq_dists, kth=k_eff - 1, axis=1)[:, :k_eff]
            else:
                part = sq_dists
            if exclude_self and k_eff > 1:
                part = np.sort(part, axis=1)[:, 1:]
            return np.mean(np.sqrt(part), axis=1).astype(np.float32)

        if self.method == "cosine":
            similarities = values @ self.reference_embeddings.T
            if self.k > 0:
                k_eff = min(self.k, similarities.shape[1])
                nearest = np.partition(similarities, kth=similarities.shape[1] - k_eff, axis=1)[
                    :, -k_eff:
                ]
                similarities = np.mean(nearest, axis=1)
            else:
                similarities = np.mean(similarities, axis=1)
            return (1.0 - similarities).astype(np.float32)

        if self.method == "centroid":
            return np.linalg.norm(values - self._centroid, axis=1).astype(np.float32)

        difference = values - self._mean
        quad = np.maximum(
            0.0, np.sum((difference @ self._covariance_inverse) * difference, axis=1)
        )
        return np.sqrt(quad).astype(np.float32)

    def with_compressed_bank(
        self, n_prototypes: int, random_state: int = 42
    ) -> "EmbeddingAnomalyScorer":
        """Return a new scorer whose reference bank is compressed to M prototypes."""
        compressed = compress_reference_bank(
            self.reference_embeddings,
            n_prototypes=n_prototypes,
            normalize=self.normalize,
            random_state=random_state,
        )
        return EmbeddingAnomalyScorer(
            compressed,
            method=self.method,
            k=min(self.k, len(compressed)) if self.method == "knn" else self.k,
            normalize=self.normalize,
        )

    def recalibrate(
        self,
        quantized_reference_embeddings: np.ndarray,
        quantized_val_normal_embeddings: np.ndarray,
        percentile: float = 97.5,
        method: Optional[str] = None,
        k: Optional[int] = None,
        n_prototypes: Optional[int] = None,
    ) -> Tuple["EmbeddingAnomalyScorer", float]:
        """Rebuild reference bank and healthy quantile threshold in a target (e.g. INT8) runtime domain."""
        scorer = EmbeddingAnomalyScorer(
            quantized_reference_embeddings,
            method=method or self.method,
            k=int(k if k is not None else self.k),
            normalize=self.normalize,
            n_prototypes=n_prototypes,
        )
        val_scores = scorer.score(quantized_val_normal_embeddings)
        threshold = float(np.percentile(val_scores, percentile))
        return scorer, threshold

    __call__ = score
