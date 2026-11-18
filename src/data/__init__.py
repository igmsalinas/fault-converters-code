"""Data loading, preprocessing, and dataset management."""

from .component_ranges import (
    ComponentRange,
    classify_component,
    classify_variations,
    find_ranges_file,
    load_ranges,
    load_ranges_for,
)
from .dataset import (
    PowerConverterDataset,
    BuckConverterDataset,
    TransferFunctionData,
)
from .loader import (
    DataLoader,
    load_simulation_file,
    load_all_simulations,
)
from .preprocessor import (
    DataPreprocessor,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)
from .generator import (
    SequenceGenerator,
    AugmentedGenerator,
)
from .anomaly_injection import (
    AnomalyInjector,
    AnomalyConfig,
    AnomalyType,
    create_anomaly_injector,
)
try:
    from .physics_anomaly import (
        PhysicsAnomalyInjector,
        PhysicsAnomalyType,
        TransferFunctionModel,
        PowerConverter,
        BuckConverter,
        BoostConverter,
        BuckBoostConverter,
        FaultMode,
        DEFAULT_FAULT_MODES,
        available_fault_modes,
        make_converter,
        available_converters,
    )
    _PHYSICS_EXPORTS = [
        "PhysicsAnomalyInjector", "PhysicsAnomalyType", "TransferFunctionModel",
        "PowerConverter", "BuckConverter", "BoostConverter", "BuckBoostConverter",
        "FaultMode", "DEFAULT_FAULT_MODES", "available_fault_modes",
        "make_converter", "available_converters",
    ]
except ImportError:
    # Physics-based training is optional for inference-only JetPack installs.
    # Importing src.data must not pull in an incompatible system SciPy when the
    # deployment pipeline only needs loaders, datasets, and preprocessing.
    _PHYSICS_EXPORTS = []

__all__ = [
    # Dataset
    "PowerConverterDataset",
    "BuckConverterDataset",
    "TransferFunctionData",
    # Component ranges (shared normal/anomalous spec)
    "ComponentRange",
    "classify_component",
    "classify_variations",
    "find_ranges_file",
    "load_ranges",
    "load_ranges_for",
    # Loader
    "DataLoader",
    "load_simulation_file",
    "load_all_simulations",
    # Preprocessor
    "DataPreprocessor",
    "StandardScaler",
    "MinMaxScaler",
    "RobustScaler",
    # Generator
    "SequenceGenerator",
    "AugmentedGenerator",
    # Anomaly Injection
    "AnomalyInjector",
    "AnomalyConfig",
    "AnomalyType",
    "create_anomaly_injector",
] + _PHYSICS_EXPORTS
