import numpy as np
import src.evaluate as evaluate_module

from src.evaluation.metrics import (
    bootstrap_classification_intervals,
    compute_classification_metrics,
)
from src.evaluation.threshold import compute_optimal_threshold


def test_compute_classification_metrics():
    # 8 normal (0), 2 anomalous (1)
    labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])

    # Perfect scores (high scores for anomalies)
    scores = np.array([0.1, 0.2, 0.1, 0.3, 0.15, 0.25, 0.1, 0.2, 0.9, 0.95])
    predictions = (scores > 0.5).astype(int)

    metrics = compute_classification_metrics(predictions, labels, scores)

    # Verify perfect separation
    assert np.isclose(metrics.auc_roc, 1.0)
    assert np.isclose(metrics.auc_pr, 1.0)


def test_compute_optimal_threshold():
    normal_errors = np.array([0.1, 0.2, 0.3, 0.4])

    threshold = compute_optimal_threshold(
        normal_errors, method="percentile", percentile=95
    )
    assert threshold > 0.35

    threshold_std = compute_optimal_threshold(normal_errors, method="std", n_std=3)
    assert threshold_std > 0.4


def test_evaluate_metric_helpers_support_legacy_sklearn(monkeypatch):
    """JetPack 4.6 sklearn predates the zero_division keyword."""
    calls = {"f1": 0, "fbeta": 0}

    def legacy_f1(labels, predictions, **kwargs):
        calls["f1"] += 1
        if "zero_division" in kwargs:
            raise TypeError("unexpected keyword argument 'zero_division'")
        return 0.75

    def legacy_fbeta(labels, predictions, beta, **kwargs):
        calls["fbeta"] += 1
        if "zero_division" in kwargs:
            raise TypeError("unexpected keyword argument 'zero_division'")
        assert beta == 2.0
        return 0.625

    monkeypatch.setattr(evaluate_module, "f1_score", legacy_f1)
    monkeypatch.setattr(evaluate_module, "fbeta_score", legacy_fbeta)
    labels = np.array([0, 1])
    predictions = np.array([0, 1])

    assert evaluate_module._f1_score_compat(labels, predictions) == 0.75
    assert evaluate_module._fbeta_score_compat(
        labels, predictions, beta=2.0
    ) == 0.625
    assert calls == {"f1": 2, "fbeta": 2}


def test_threshold_selector_fbeta():
    from src.evaluation.threshold import ThresholdSelector
    # Normal errors cluster lower, anomalies cluster higher
    normal_errors = np.array([0.1, 0.2, 0.15, 0.25, 0.05])
    anomaly_errors = np.array([0.4, 0.5, 0.6, 0.8, 0.35])

    # Fit with beta = 1 (F1-score)
    selector_f1 = ThresholdSelector(method="fbeta", beta=1.0)
    selector_f1.fit(normal_errors, anomaly_errors)

    # Fit with beta = 2 (F2-score prioritizing recall)
    selector_f2 = ThresholdSelector(method="fbeta", beta=2.0)
    selector_f2.fit(normal_errors, anomaly_errors)

    assert selector_f1.threshold_ > 0.0
    assert selector_f2.threshold_ > 0.0
    # Higher beta weights recall more heavily, which tends to lower the optimal threshold to catch more true positives
    assert selector_f2.threshold_ <= selector_f1.threshold_


def test_stratified_bootstrap_is_deterministic_and_bounded():
    labels = np.array([0] * 20 + [1] * 20)
    scores = np.linspace(0.0, 1.0, len(labels))
    predictions = (scores > 0.45).astype(int)

    first = bootstrap_classification_intervals(
        predictions, labels, scores, n_resamples=50, seed=7
    )
    second = bootstrap_classification_intervals(
        predictions, labels, scores, n_resamples=50, seed=7
    )

    assert first == second
    for interval in first.values():
        assert 0.0 <= interval["lower"] <= interval["upper"] <= 1.0
