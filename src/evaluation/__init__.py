"""Evaluation and metrics for anomaly detection."""

from .metrics import (
    AnomalyMetrics,
    compute_reconstruction_metrics,
    compute_classification_metrics,
)
from .threshold import (
    ThresholdSelector,
    compute_optimal_threshold,
)
from .baselines import (
    BaseFeatureDetector,
    IsolationForestDetector,
    MahalanobisDetector,
    OneClassSVMDetector,
    PCAReconstructionDetector,
    RawKNNDetector,
    SupervisedOracleDetector,
)
from .fault_analysis import (
    DEFAULT_ANOMALY_PCT_BANDS,
    compute_per_fault_recall,
    compute_prevalence_adjusted_prauc,
    inject_fra_measurement_noise,
    paired_sign_flip_test,
)
try:
    from .attribution import (
        DEFAULT_FREQUENCY_BANDS_HZ,
        compute_knn_score_frequency_saliency,
        extract_transformer_attention_maps,
        summarize_frequency_attention,
    )
    _ATTRIBUTION_EXPORTS = [
        "DEFAULT_FREQUENCY_BANDS_HZ",
        "extract_transformer_attention_maps",
        "summarize_frequency_attention",
        "compute_knn_score_frequency_saliency",
    ]
except ImportError:
    _ATTRIBUTION_EXPORTS = []
try:
    from .visualization import (
        plot_reconstruction_error_distribution,
        plot_roc_curve,
        plot_precision_recall_curve,
        plot_confusion_matrix,
        plot_reconstructions,
        plot_latent_space,
        plot_training_history,
    )
    _VISUALIZATION_EXPORTS = [
        "plot_reconstruction_error_distribution", "plot_roc_curve",
        "plot_precision_recall_curve", "plot_confusion_matrix",
        "plot_reconstructions", "plot_latent_space", "plot_training_history",
    ]
except ImportError:
    # Plotting dependencies are deliberately optional on headless edge targets.
    _VISUALIZATION_EXPORTS = []

__all__ = [
    # Metrics
    "AnomalyMetrics",
    "compute_reconstruction_metrics",
    "compute_classification_metrics",
    # Threshold
    "ThresholdSelector",
    "compute_optimal_threshold",
    # Baselines
    "BaseFeatureDetector",
    "RawKNNDetector",
    "MahalanobisDetector",
    "PCAReconstructionDetector",
    "IsolationForestDetector",
    "OneClassSVMDetector",
    "SupervisedOracleDetector",
    # Fault & robustness diagnostics
    "DEFAULT_ANOMALY_PCT_BANDS",
    "compute_per_fault_recall",
    "compute_prevalence_adjusted_prauc",
    "inject_fra_measurement_noise",
    "paired_sign_flip_test",
] + _ATTRIBUTION_EXPORTS + _VISUALIZATION_EXPORTS
