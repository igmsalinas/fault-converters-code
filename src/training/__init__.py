"""Training pipeline and utilities."""

import keras

from .trainer import Trainer, TrainingConfig
from .callbacks import get_callbacks, EarlyStoppingWithRestore
# Keras Tuner is a workstation training dependency and is intentionally absent
# from the JetPack 4.6 inference environment. Keep the package importable there
# so CARLA checkpoint loading/evaluation does not require an unused tuner.
try:
    from .hyperparameter_search import HyperparameterSearch
except ModuleNotFoundError as error:
    if error.name != "keras_tuner":
        raise
    HyperparameterSearch = None
from .carla_trainer import CARLATrainer, CARLAConfig
from .carla_hyperparameter_search import CARLAHyperparameterSearch, CARLASearchSpace

# These convenience exports use Keras 3 serialization decorators. They are
# workstation training utilities, not dependencies of JetPack checkpoint
# evaluation, so avoid importing them under the Nano's Keras 2.7 stack.
_optimizer_exports = []
if hasattr(getattr(keras, "saving", None), "register_keras_serializable"):
    from .optimizers import (
        BaseLRSchedule,
        BaseOptimizer,
        ConstantLR,
        CosineLR,
        WarmupCosineLR,
        ExponentialLR,
        StepLR,
        PolynomialLR,
        InverseTimeLR,
        OneCycleLR,
        CyclicLR,
        AdamOptimizer,
        AdamWOptimizer,
        NadamOptimizer,
        SGDOptimizer,
        RMSpropOptimizer,
        LionOptimizer,
        AdagradOptimizer,
        AdadeltaOptimizer,
        LR_SCHEDULE_REGISTRY,
        OPTIMIZER_REGISTRY,
        get_lr_schedule,
        get_optimizer,
        OptimizerConfig,
        create_lr_schedule,
        create_optimizer,
    )
    _optimizer_exports = [
        "BaseLRSchedule", "ConstantLR", "CosineLR", "WarmupCosineLR",
        "ExponentialLR", "StepLR", "PolynomialLR", "InverseTimeLR",
        "OneCycleLR", "CyclicLR", "LR_SCHEDULE_REGISTRY", "get_lr_schedule",
        "BaseOptimizer", "AdamOptimizer", "AdamWOptimizer", "NadamOptimizer",
        "SGDOptimizer", "RMSpropOptimizer", "LionOptimizer", "AdagradOptimizer",
        "AdadeltaOptimizer", "OPTIMIZER_REGISTRY", "get_optimizer",
        "OptimizerConfig", "create_lr_schedule", "create_optimizer",
    ]

__all__ = [
    # Trainer
    "Trainer",
    "TrainingConfig",
    "get_callbacks",
    "EarlyStoppingWithRestore",
    "HyperparameterSearch",
    "CARLATrainer",
    "CARLAConfig",
    "CARLAHyperparameterSearch",
    "CARLASearchSpace",
] + _optimizer_exports
