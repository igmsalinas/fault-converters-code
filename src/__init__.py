"""
Power Converter Anomaly Detection
================================

A research ML project for detecting anomalies in simulated transfer functions
of Power Converters using autoencoder-based approaches.

Author: UC3M Research
Date: January 2026

Key Components:
- data: Data loading, preprocessing, and generation
- models: Autoencoder architectures (Conv1D, LSTM, VAE, Transformer)
- training: Training pipelines with hyperparameter search
- evaluation: Metrics and visualization tools
- visualization: Transfer function plotting and comparison utilities
- utils: Configuration and helper utilities
"""

# Ubuntu 18.04's SciPy 0.19 imports NumPy's historical public nosetester path.
# NumPy 1.19 (required by NVIDIA's JetPack 4.6 TensorFlow wheel) moved that
# module under ``testing._private``. Alias it before any SciPy/sklearn import so
# the inference environment can use JetPack's system scientific packages.
try:
    import sys
    from numpy.testing._private import decorators as _numpy_decorators
    from numpy.testing._private import nosetester as _numpy_nosetester
    sys.modules.setdefault("numpy.testing.decorators", _numpy_decorators)
    sys.modules.setdefault("numpy.testing.nosetester", _numpy_nosetester)
except ImportError:
    pass

__version__ = "0.1.0"
__author__ = "UC3M Research"
