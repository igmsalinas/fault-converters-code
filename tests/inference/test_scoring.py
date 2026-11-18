"""Unit tests for EmbeddingAnomalyScorer, Ledoit-Wolf shrinkage, bank compression, and INT8 recalibration."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from src.inference.scoring import EmbeddingAnomalyScorer
from src.losses.contrastive import anomaly_score_from_embeddings


def _make_low_rank_manifold_problem(seed: int = 123):
    """Simulate unit-sphere CARLA embeddings with low intrinsic rank (rank=4 in R^64)."""
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.normal(size=(64, 4)))
    basis = basis.astype(np.float32)

    # Healthy reference and healthy test samples lie strictly in span(basis)
    z_ref_lat = rng.normal(loc=0.0, scale=0.3, size=(200, 4)).astype(np.float32)
    z_ref_lat[:, 0] += 1.0
    ref = z_ref_lat @ basis.T

    z_norm_lat = rng.normal(loc=0.0, scale=0.3, size=(120, 4)).astype(np.float32)
    z_norm_lat[:, 0] += 1.0
    test_norm = z_norm_lat @ basis.T

    # True anomalies shift within span(basis)
    z_anom_lat = rng.normal(loc=0.0, scale=0.3, size=(120, 4)).astype(np.float32)
    z_anom_lat[:, 0] += 1.0
    z_anom_lat[:, 1] += 1.25
    test_anom = z_anom_lat @ basis.T

    # Out-of-subspace measurement/quantization jitter at identical instrument floor
    jitter_norm = rng.normal(scale=0.012, size=test_norm.shape).astype(np.float32)
    jitter_anom = rng.normal(scale=0.012, size=test_anom.shape).astype(np.float32)
    test_x = np.vstack([test_norm + jitter_norm, test_anom + jitter_anom])
    test_y = np.concatenate([np.zeros(120, dtype=int), np.ones(120, dtype=int)])
    return ref, test_x, test_y


def test_ledoit_wolf_prevents_nullspace_explosion_vs_ridge_mahalanobis():
    ref, test_x, test_y = _make_low_rank_manifold_problem()

    scorer_ridge = EmbeddingAnomalyScorer(ref, method="mahalanobis", normalize=True)
    scorer_lw = EmbeddingAnomalyScorer(ref, method="ledoit_wolf", normalize=True)

    auc_ridge = roc_auc_score(test_y, scorer_ridge.score(test_x))
    auc_lw = roc_auc_score(test_y, scorer_lw.score(test_x))

    # Unshrunk 1e-6 ridge blows up null-space jitter by 10^6; Ledoit-Wolf stays near-ideal
    assert auc_lw > 0.94
    assert auc_lw > auc_ridge + 0.25


def test_knn_exact_distance_matches_sklearn_nearest_neighbors():
    rng = np.random.default_rng(7)
    ref = rng.normal(size=(150, 32)).astype(np.float32)
    query = rng.normal(size=(17, 32)).astype(np.float32)

    for normalize in (True, False):
        scorer = EmbeddingAnomalyScorer(ref, method="knn", k=5, normalize=normalize)
        actual = scorer.score(query)

        ref_check = (
            ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-8)
            if normalize
            else ref
        )
        q_check = (
            query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)
            if normalize
            else query
        )
        nn = NearestNeighbors(n_neighbors=5).fit(ref_check)
        dists, _ = nn.kneighbors(q_check)
        expected = np.mean(dists, axis=1).astype(np.float32)

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_prototype_bank_compression_reduces_memory_and_preserves_auc():
    ref, test_x, test_y = _make_low_rank_manifold_problem(seed=99)
    full_scorer = EmbeddingAnomalyScorer(ref, method="knn", k=5, normalize=True)
    compressed_scorer = full_scorer.with_compressed_bank(
        n_prototypes=24, random_state=42
    )

    assert compressed_scorer.reference_embeddings.shape == (24, ref.shape[1])
    norms = np.linalg.norm(compressed_scorer.reference_embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    auc_full = roc_auc_score(test_y, full_scorer.score(test_x))
    auc_comp = roc_auc_score(test_y, compressed_scorer.score(test_x))
    assert auc_comp > 0.96
    assert abs(auc_full - auc_comp) < 0.03


def test_quantized_backend_recalibration_rescues_decision_flip_rate():
    rng = np.random.default_rng(2026)
    ref_fp32 = rng.normal(loc=0.0, scale=0.05, size=(250, 32)).astype(np.float32)
    ref_fp32[:, 0] += 1.0
    val_fp32 = rng.normal(loc=0.0, scale=0.05, size=(120, 32)).astype(np.float32)
    val_fp32[:, 0] += 1.0

    test_norm_fp32 = rng.normal(loc=0.0, scale=0.05, size=(100, 32)).astype(np.float32)
    test_norm_fp32[:, 0] += 1.0
    test_anom_fp32 = rng.normal(loc=0.0, scale=0.05, size=(100, 32)).astype(np.float32)
    test_anom_fp32[:, 0] += 1.0
    test_anom_fp32[:, 1] += 0.55
    test_fp32 = np.vstack([test_norm_fp32, test_anom_fp32])

    scorer_fp32 = EmbeddingAnomalyScorer(ref_fp32, method="knn", k=5, normalize=True)
    th_fp32 = float(np.percentile(scorer_fp32.score(val_fp32), 95.0))
    decisions_fp32 = scorer_fp32.score(test_fp32) > th_fp32

    # Simulate systematic INT8 angular zero-point drift + 8-bit grid rounding
    # on a tight CARLA cluster (reproduces the ~45%-50% healthy->FP flip rate
    # seen in JSC when scoring INT8 embeddings against an FP32 bank/threshold)
    bias_shift = np.zeros((1, 32), dtype=np.float32)
    bias_shift[0, 2:6] = 0.16

    def quantize_sim(arr: np.ndarray) -> np.ndarray:
        shifted = arr + bias_shift
        scale = 0.008
        return (np.round(shifted / scale) * scale).astype(np.float32)

    ref_int8 = quantize_sim(ref_fp32)
    val_int8 = quantize_sim(val_fp32)
    test_int8 = quantize_sim(test_fp32)

    # Naive scoring against FP32 bank + FP32 threshold causes massive decision flips
    naive_decisions = scorer_fp32.score(test_int8) > th_fp32
    naive_flip_rate = float(np.mean(naive_decisions != decisions_fp32))
    assert naive_flip_rate > 0.25

    # Backend-calibrated scorer + healthy-val threshold restores decision concordance
    cal_scorer, cal_th = scorer_fp32.recalibrate(
        quantized_reference_embeddings=ref_int8,
        quantized_val_normal_embeddings=val_int8,
        percentile=95.0,
    )
    cal_decisions = cal_scorer.score(test_int8) > cal_th
    cal_flip_rate = float(np.mean(cal_decisions != decisions_fp32))
    assert cal_flip_rate < 0.06


def test_anomaly_score_from_embeddings_supports_ledoit_wolf_and_self_exclusion():
    rng = np.random.default_rng(11)
    ref = rng.normal(size=(40, 12)).astype(np.float32)
    query = rng.normal(size=(9, 12)).astype(np.float32)

    scores_fn = anomaly_score_from_embeddings(query, ref, method="ledoit_wolf")
    scores_cls = EmbeddingAnomalyScorer(ref, method="ledoit_wolf").score(query)
    np.testing.assert_allclose(scores_fn, scores_cls, rtol=1e-5, atol=1e-5)

    # Self-exclusion check on knn
    self_scores = anomaly_score_from_embeddings(ref, ref, method="knn", k=3)
    assert np.all(self_scores > 0.0)
