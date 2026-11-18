import json

from src.training.carla_hyperparameter_search import (
    CARLAHyperparameterSearch,
    TrialResult,
)
from src.training.carla_trainer import CARLAConfig


def _config(encoder_type="conv1d", **overrides):
    config = {
        "encoder_type": encoder_type,
        "latent_dim": 32,
        "projection_dim": 64,
        "encoder_filters": [64, 128],
        "encoder_units": [128, 64],
        "kernel_size": 3,
        "num_heads": 2,
        "learning_rate": 0.0003,
        "reconstruction_weight": 10.0,
        "contrastive_weight": 1.0,
        "temperature": 0.1,
        "anomaly_ratio": 0.5,
        "scoring_method": "knn",
        "k_neighbors": 5,
        "dropout_rate": 0.0,
    }
    config.update(overrides)
    return config


def _trial(config):
    return TrialResult(
        trial_id=0,
        config=config,
        val_loss=0.1,
        val_recon_loss=0.1,
        val_contrast_loss=0.1,
        roc_auc=0.9,
        pr_auc=0.9,
        f1_score=0.9,
        training_time=1.0,
        epochs_trained=1,
    )


def test_effective_config_key_ignores_only_inactive_encoder_fields():
    baseline = _config()
    nuisance_change = _config(encoder_units=[512, 256], num_heads=8)
    active_change = _config(encoder_filters=[128, 256])

    key = CARLAHyperparameterSearch._effective_config_key
    assert key(baseline) == key(nuisance_change)
    assert key(baseline) != key(active_change)

    recurrent = _config("gru")
    recurrent_nuisance = _config(
        "gru", encoder_filters=[128, 256, 512], kernel_size=5, num_heads=8
    )
    recurrent_active = _config("gru", encoder_units=[256, 128])
    assert key(recurrent) == key(recurrent_nuisance)
    assert key(recurrent) != key(recurrent_active)


def test_candidate_generation_rejects_effective_duplicate(tmp_path):
    search = CARLAHyperparameterSearch(
        input_shape=(101, 2), project_name="test", directory=str(tmp_path)
    )
    baseline = _config()
    search.trials = [_trial(baseline)]
    candidates = iter([
        _config(encoder_units=[512, 256], num_heads=8),
        _config(encoder_filters=[128, 256]),
    ])

    selected = search._get_or_create_config(1, generate_fn=lambda: next(candidates))

    assert selected["encoder_filters"] == [128, 256]
    saved = json.loads(
        (tmp_path / "test" / "trial_1" / "trial_config.json").read_text()
    )
    assert saved == selected


def test_corrected_carla_protocol_has_explicit_seeds_and_one_negative_ratio(tmp_path):
    config = CARLAConfig(seed=17, resume_from_latest=False)
    search = CARLAHyperparameterSearch(
        input_shape=(101, 2), project_name="seeded", directory=str(tmp_path), seed=23
    )

    assert config.seed == 17
    assert config.anomaly_ratio == 1.0
    assert config.n_negative_per_sample == 1
    assert config.resume_from_latest is False
    assert search.seed == 23
