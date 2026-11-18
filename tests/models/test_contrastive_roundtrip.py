"""Checkpoint round-trip tests for every deployable CARLA encoder."""

import numpy as np
import pytest

from src.models.contrastive_ae import ContrastiveAutoencoder


@pytest.mark.parametrize("encoder_type", ["conv1d", "gru", "lstm", "mlp", "transformer"])
def test_carla_projection_roundtrip(tmp_path, encoder_type):
    model = ContrastiveAutoencoder(
        input_shape=(16, 2),
        latent_dim=4,
        projection_dim=4,
        encoder_type=encoder_type,
        encoder_filters=[4, 8],
        encoder_units=[8, 4],
        num_heads=2,
        projection_hidden_dim=8,
        dropout_rate=0.0,
    )
    model.build()
    inputs = np.random.default_rng(42).normal(size=(3, 16, 2)).astype(np.float32)
    expected = model.project(inputs)

    checkpoint = tmp_path / encoder_type
    model.save(str(checkpoint))
    restored = ContrastiveAutoencoder.load(str(checkpoint))
    actual = restored.project(inputs)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_carla_load_rejects_missing_projection_checkpoint(tmp_path):
    model = ContrastiveAutoencoder(
        input_shape=(16, 2),
        latent_dim=4,
        projection_dim=4,
        encoder_type="mlp",
        encoder_units=[8, 4],
        projection_hidden_dim=8,
        dropout_rate=0.0,
    )
    model.build()
    checkpoint = tmp_path / "missing_projection"
    model.save(str(checkpoint))
    (checkpoint / "projection_head.weights.npz").unlink()
    (checkpoint / "projection_head.weights.h5").unlink()

    with pytest.raises(FileNotFoundError, match="projection_head"):
        ContrastiveAutoencoder.load(str(checkpoint))
