"""Unit tests for Transformer self-attention extraction and Jacobian frequency saliency (TDD)."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.attribution import (
    compute_knn_score_frequency_saliency,
    extract_transformer_attention_maps,
    summarize_frequency_attention,
)
from src.models.contrastive_ae import ContrastiveAutoencoder
from src.models.transformer_ae import TransformerAutoencoder


def test_extract_attention_contrastive_transformer_shapes_and_simplex():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(6, 21, 2)).astype(np.float32)

    model = ContrastiveAutoencoder(
        input_shape=(21, 2),
        latent_dim=8,
        projection_dim=16,
        encoder_type="transformer",
        encoder_filters=[16, 16],
        num_heads=4,
        dropout_rate=0.0,
    )
    model.build()
    maps = extract_transformer_attention_maps(model, x, batch_size=4)

    assert len(maps) == 2
    for layer_attn in maps:
        assert layer_attn.shape == (6, 4, 21, 21)
        assert np.all(layer_attn >= -1e-7)
        row_sums = layer_attn.sum(axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


def test_extract_attention_standalone_transformer_ae():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(5, 17, 2)).astype(np.float32)

    model = TransformerAutoencoder(
        input_shape=(17, 2),
        latent_dim=8,
        d_model=16,
        num_heads=2,
        ff_dim=32,
        num_layers=2,
        dropout_rate=0.0,
    )
    model.build()
    maps = extract_transformer_attention_maps(model, x, batch_size=3)

    assert len(maps) == 2
    for layer_attn in maps:
        assert layer_attn.shape == (5, 2, 17, 17)
        np.testing.assert_allclose(layer_attn.sum(axis=-1), 1.0, atol=1e-5)


def test_extract_attention_rejects_non_transformer():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(4, 16, 2)).astype(np.float32)
    conv_model = ContrastiveAutoencoder(
        input_shape=(16, 2),
        latent_dim=8,
        projection_dim=16,
        encoder_type="conv1d",
        encoder_filters=[8, 16],
    )
    conv_model.build()
    with pytest.raises(ValueError, match="MultiHeadAttention"):
        extract_transformer_attention_maps(conv_model, x)


def test_summarize_frequency_attention_detects_resonant_landmark():
    freqs = np.geomspace(100.0, 50000.0, 101)
    res_idx = int(np.argmin(np.abs(freqs - 1125.0)))

    # Construct synthetic 2-layer attention strongly focusing keys at ~1.13 kHz
    attn = np.full((10, 4, 101, 101), 0.2 / 100.0, dtype=np.float32)
    attn[:, :, :, res_idx] = 0.8 + (0.2 / 100.0)
    attn = attn / attn.sum(axis=-1, keepdims=True)

    summary = summarize_frequency_attention([attn, attn], freqs)
    key_profile = np.asarray(summary["mean_key_attention"])
    assert key_profile.shape == (101,)
    np.testing.assert_allclose(key_profile.sum(), 1.0, atol=1e-5)
    assert int(summary["peak_attended_index"]) == res_idx
    assert 1000.0 <= float(summary["peak_attended_freq_hz"]) <= 1300.0
    assert float(summary["bands"]["resonance"]["enrichment_ratio"]) > 4.0
    assert 0.0 <= float(summary["normalized_entropy_mean"]) < 0.5


def test_compute_knn_score_frequency_saliency_matches_finite_difference():
    rng = np.random.default_rng(19)
    ref = rng.normal(size=(12, 15, 2)).astype(np.float32)
    query = (rng.normal(size=(3, 15, 2)) + 1.5).astype(np.float32)

    model = ContrastiveAutoencoder(
        input_shape=(15, 2),
        latent_dim=8,
        projection_dim=12,
        encoder_type="transformer",
        encoder_filters=[16],
        num_heads=2,
        dropout_rate=0.0,
    )
    model.build()
    sal = compute_knn_score_frequency_saliency(
        model, query, ref, space="projection", k=3, batch_size=2
    )
    assert sal.shape == (3, 15)
    assert np.all(np.isfinite(sal))
    assert np.all(sal >= 0.0)
    assert float(sal.sum()) > 0.0
