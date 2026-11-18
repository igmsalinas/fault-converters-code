import pytest
import numpy as np

from unittest.mock import patch, MagicMock
from src.training.trainer import Trainer, TrainingConfig
from src.training.carla_trainer import CARLATrainer, CARLAConfig


@pytest.fixture
def mock_trainer_data():
    seq_len = 24
    features = 5
    # Use 16 samples minimum for validation split testing
    n_samples = 32

    data = np.random.normal(0, 1, size=(n_samples, seq_len, features)).astype(
        np.float32
    )
    return data, data


def test_trainer_initialization_and_fit(mock_trainer_data, tmp_path):
    train_data, val_data = mock_trainer_data
    seq_len, features = train_data.shape[1:]

    config = TrainingConfig(
        epochs=1,
        batch_size=8,
        use_gpu=False,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_dir=str(tmp_path / "logs"),
    )

    trainer = Trainer(config, experiment_name="test_exp")

    # Test model creation
    model = trainer.create_model(
        model_type="conv1d_ae",
        input_shape=(seq_len, features),
        filters=[16],
        latent_dim=8,
    )
    assert model is not None
    trainer.compile_model()

    # Test minimal training execution doesn't crash
    history = trainer.train(train_data, val_data, verbose=0)
    assert history is not None
    assert "loss" in history.history


def test_trainer_gpu_initialization(mock_trainer_data, tmp_path):
    train_data, val_data = mock_trainer_data
    seq_len, features = train_data.shape[1:]

    config = TrainingConfig(
        epochs=1,
        batch_size=8,
        use_gpu=True,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_dir=str(tmp_path / "logs"),
    )

    # Mock tensorflow to simulate finding a GPU
    with patch("tensorflow.config.list_physical_devices") as mock_list_devices:
        # Create a mock device
        mock_device = MagicMock()
        mock_list_devices.return_value = [mock_device]

        with patch("src.training.base_trainer.logger.info") as mock_logger_info:
            with patch(
                "tensorflow.config.experimental.set_memory_growth"
            ) as mock_set_memory:
                trainer = Trainer(config, experiment_name="test_gpu_exp")

                # Check that list_physical_devices was called for GPU
                mock_list_devices.assert_called_with("GPU")

                # Check that set_memory_growth was called
                mock_set_memory.assert_called_with(mock_device, True)

                # Check logger
                mock_logger_info.assert_any_call(
                    "Found 1 GPU(s): [<MagicMock id='" + str(id(mock_device)) + "'>]"
                    if not mock_logger_info.call_args_list
                    else mock_logger_info.call_args_list[0][0][0]
                )

                # Quick check that it can still create a model
                model = trainer.create_model(
                    model_type="conv1d_ae",
                    input_shape=(seq_len, features),
                    filters=[16],
                    latent_dim=8,
                )
                assert model is not None


def test_carla_trainer_synthetic_injection_and_fit(mock_trainer_data, tmp_path):
    train_data, val_data = mock_trainer_data
    seq_len, features = train_data.shape[1:]

    config = CARLAConfig(
        epochs=1,
        batch_size=8,
        use_gpu=False,
        n_negative_per_sample=1,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_dir=str(tmp_path / "logs"),
    )

    trainer = CARLATrainer(config, experiment_name="test_carla_exp")

    trainer.create_model(
        input_shape=(seq_len, features),
        encoder_type="conv1d",
        encoder_filters=[16],
        latent_dim=8,
        projection_dim=16,
    )
    trainer.setup_training()

    # Test batch preparation logic with neg samples
    anchors, positives, negatives, is_normal = trainer._prepare_batch(train_data[:4])
    assert anchors.shape == (4, seq_len, features)
    assert positives.shape == (4, seq_len, features)
    # negatives should be (batch_size, n_neg, seq, feature)
    assert negatives.shape == (4, 1, seq_len, features)

    # Test minimal training loop
    history = trainer.train(train_data, val_data, verbose=0)
    assert history is not None
    assert len(trainer.history["loss"]) == 1
    assert trainer.history["loss"][0] > 0

    # Test anomaly scoring function (kNN)
    scores = trainer.compute_anomaly_scores(val_data[:4], method="knn", k=2)
    assert scores.shape == (4,)


def test_carla_validation_uses_partial_batch_and_sample_weighting(tmp_path):
    config = CARLAConfig(
        epochs=1,
        batch_size=8,
        use_gpu=False,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_dir=str(tmp_path / "logs"),
    )
    trainer = CARLATrainer(config, experiment_name="partial_batch")
    values = np.arange(10, dtype=np.float32).reshape(10, 1, 1)

    trainer._prepare_batch = lambda batch: (batch, batch, batch, None)
    trainer._validate_step = lambda anchors, _positives, _negatives: (
        float(np.mean(anchors)),
        float(np.mean(anchors)),
        float(np.mean(anchors)),
    )

    loss, reconstruction, contrastive = trainer._validate_epoch(values)

    assert loss == pytest.approx(4.5)
    assert reconstruction == pytest.approx(4.5)
    assert contrastive == pytest.approx(4.5)
    assert len(trainer._validation_batches) == 2

    trainer._prepare_batch = lambda _batch: (_ for _ in ()).throw(
        AssertionError("validation perturbations were regenerated")
    )
    repeated = trainer._validate_epoch(values)
    assert repeated == pytest.approx((4.5, 4.5, 4.5))


def test_carla_trainer_physics_subset_lofo_and_recon_only(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.normal(0.0, 1.0, size=(24, 16, 2)).astype(np.float32)
    freqs = np.logspace(2, 4, 16)

    # 1. Leave-one-fault-out subset of physics fault names must not fail AnomalyType parsing
    lofo_modes = ["capacitance_drop", "inductor_saturation", "inductor_esr", "switch_degradation"]
    config = CARLAConfig(
        epochs=1,
        batch_size=8,
        use_gpu=False,
        use_physics_anomalies=True,
        anomaly_types=lofo_modes,
        contrastive_weight=0.0,
        reconstruction_weight=10.0,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_dir=str(tmp_path / "logs"),
    )
    trainer = CARLATrainer(config, experiment_name="lofo_recon_test")
    trainer.create_model(
        input_shape=(16, 2),
        encoder_type="conv1d",
        encoder_filters=[16],
        latent_dim=8,
        projection_dim=16,
    )
    trainer.set_physics_context(frequencies=freqs, scaler=None, use_real_imag=True)
    trainer.setup_training()
    assert trainer.anomaly_injector.fault_mode_names == lofo_modes

    # 2. Training with contrastive_weight=0.0 (pure AE) must succeed and support reconstruction & latent_knn scoring
    history = trainer.train(data, data[:8], verbose=0)
    assert len(history["loss"]) == 1
    recon_scores = trainer.compute_anomaly_scores(data[:6], method="reconstruction")
    latent_scores = trainer.compute_anomaly_scores(data[:6], method="latent_knn", k=3)
    assert recon_scores.shape == (6,)
    assert latent_scores.shape == (6,)
    assert np.all(np.isfinite(recon_scores))
    assert np.all(np.isfinite(latent_scores))


def test_carla_trainer_fra_denoising_mode(tmp_path):
    from src.data.preprocessor import StandardScaler

    rng = np.random.default_rng(11)
    raw_ri = rng.normal(5.0, 2.0, size=(16, 16, 2)).astype(np.float32)
    scaler = StandardScaler(per_feature=True).fit(raw_ri)
    norm_data = scaler.transform(raw_ri).astype(np.float32)
    freqs = np.logspace(2, 4, 16)

    config = CARLAConfig(
        epochs=1,
        batch_size=8,
        use_gpu=False,
        use_physics_anomalies=True,
        fra_jitter_db=0.30,
        fra_jitter_deg=1.50,
        fra_clean_prob=0.0,
        denoising_reconstruction=True,
        checkpoint_dir=str(tmp_path / "checkpoints_fra"),
        log_dir=str(tmp_path / "logs_fra"),
    )
    trainer = CARLATrainer(config, experiment_name="fra_denoising_test")
    trainer.create_model(
        input_shape=(16, 2),
        encoder_type="conv1d",
        encoder_filters=[16],
        latent_dim=8,
        projection_dim=16,
    )
    trainer.set_physics_context(frequencies=freqs, scaler=scaler, use_real_imag=True)
    trainer.setup_training()

    anchors, positives, negatives, _ = trainer._prepare_batch(norm_data[:8])
    assert anchors.shape == (8, 16, 2)
    assert positives.shape == (8, 16, 2)
    assert negatives.shape == (8, 1, 16, 2)
    # With fra_clean_prob=0.0, noisy anchors must differ from clean norm_data
    assert not np.allclose(anchors, norm_data[:8])
    assert not np.allclose(positives, anchors)

    history = trainer.train(norm_data, norm_data[:8], verbose=0)
    assert len(history["loss"]) == 1
    scores = trainer.compute_anomaly_scores(norm_data[:6])
    assert scores.shape == (6,)
    assert np.all(np.isfinite(scores))


def test_trainer_gpu_init_when_context_already_initialized(tmp_path):
    """Verify BaseTrainer._setup_gpu tolerates RuntimeError after TF GPU context init."""
    import tensorflow as tf

    _ = tf.constant([1.0])  # Force eager context initialization
    config = CARLAConfig(
        epochs=1,
        batch_size=8,
        use_gpu=True,
        checkpoint_dir=str(tmp_path / "checkpoints_gpu_locked"),
        log_dir=str(tmp_path / "logs_gpu_locked"),
    )
    mock_device = MagicMock()
    with patch("tensorflow.config.list_physical_devices", return_value=[mock_device]):
        with patch(
            "tensorflow.config.experimental.set_memory_growth",
            side_effect=RuntimeError(
                "Physical devices cannot be modified after being initialized"
            ),
        ):
            trainer = CARLATrainer(config, experiment_name="gpu_locked_test")
            assert trainer is not None

