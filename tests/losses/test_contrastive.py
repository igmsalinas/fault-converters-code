import numpy as np
from src.losses.contrastive import (
    anomaly_score_from_embeddings,
    CARLALoss,
    EmbeddingAnomalyScorer,
    NTXentLoss,
    reconstruction_loss,
)


def test_reconstruction_loss():
    y_true = np.random.normal(0, 1, size=(4, 24, 5)).astype(np.float32)
    y_pred = np.copy(y_true)
    y_pred += 0.1  # slight error

    # MSE should be positive
    loss_mse = float(reconstruction_loss(y_true, y_pred, "mse"))
    assert loss_mse > 0
    assert np.isclose(loss_mse, 0.01, atol=1e-3)

    # MAE should be positive
    loss_mae = float(reconstruction_loss(y_true, y_pred, "mae"))
    assert loss_mae > 0
    assert np.isclose(loss_mae, 0.1, atol=1e-3)


def test_nt_xent_loss():
    batch_size = 4
    embed_dim = 16
    n_neg = 2

    anchor = np.random.normal(0, 1, size=(batch_size, embed_dim)).astype(np.float32)
    positive = anchor + np.random.normal(0, 0.1, size=(batch_size, embed_dim)).astype(
        np.float32
    )
    negatives = np.random.normal(5, 1, size=(batch_size, n_neg, embed_dim)).astype(
        np.float32
    )

    loss_fn = NTXentLoss(temperature=0.1)

    # In-batch negatives
    loss_in_batch = float(loss_fn(anchor, positive))
    assert loss_in_batch > 0

    # Explicit negatives
    loss_explicit = float(loss_fn(anchor, positive, negatives))
    assert loss_explicit > 0


def test_carla_loss():
    batch_size = 4
    seq_len = 24
    features = 5
    embed_dim = 16
    n_neg = 2

    x_orig = np.random.normal(0, 1, size=(batch_size, seq_len, features)).astype(
        np.float32
    )
    x_recon = x_orig + 0.1

    z_anchor = np.random.normal(0, 1, size=(batch_size, embed_dim)).astype(np.float32)
    z_pos = z_anchor + 0.1
    z_neg = np.random.normal(5, 1, size=(batch_size, n_neg, embed_dim)).astype(
        np.float32
    )

    loss_fn = CARLALoss(reconstruction_weight=1.0, contrastive_weight=1.0)

    total, recon, contrast = loss_fn(x_orig, x_recon, z_anchor, z_pos, z_neg)

    assert float(total) > 0
    assert float(recon) > 0
    assert float(contrast) > 0
    assert np.isclose(float(total), float(recon) + float(contrast))


def test_reusable_embedding_scorer_matches_reference_function():
    rng = np.random.default_rng(42)
    reference = rng.normal(size=(20, 8)).astype(np.float32)
    query = rng.normal(size=(5, 8)).astype(np.float32)

    for method, k in [
        ("knn", 3),
        ("cosine", 3),
        ("centroid", 3),
        ("mahalanobis", 3),
    ]:
        expected = anomaly_score_from_embeddings(
            query, reference, method=method, k=k
        )
        actual = EmbeddingAnomalyScorer(
            reference, method=method, k=k
        ).score(query)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
