import numpy as np
import pytest

from src.evaluation.baselines import (
    BaseFeatureDetector,
    IsolationForestDetector,
    MahalanobisDetector,
    OneClassSVMDetector,
    PCAReconstructionDetector,
    RawKNNDetector,
    SupervisedOracleDetector,
)
from src.evaluation.fault_analysis import (
    compute_per_fault_recall,
    compute_prevalence_adjusted_prauc,
    inject_fra_measurement_noise,
    paired_sign_flip_test,
)


@pytest.fixture
def synthetic_bode_splits():
    rng = np.random.default_rng(123)
    train_norm = rng.normal(0.0, 0.2, size=(120, 16, 2)).astype(np.float32)
    val_norm = rng.normal(0.0, 0.2, size=(30, 16, 2)).astype(np.float32)
    val_anom = rng.normal(1.2, 0.3, size=(30, 16, 2)).astype(np.float32)
    fit_val_x = np.concatenate([val_norm, val_anom], axis=0)
    fit_val_y = np.array([0] * 30 + [1] * 30)

    test_norm = rng.normal(0.0, 0.2, size=(40, 16, 2)).astype(np.float32)
    test_anom = rng.normal(1.2, 0.3, size=(40, 16, 2)).astype(np.float32)
    test_x = np.concatenate([test_norm, test_anom], axis=0)
    test_y = np.array([0] * 40 + [1] * 40)
    return train_norm, val_norm, fit_val_x, fit_val_y, test_x, test_y


def test_base_detector_is_abstract():
    with pytest.raises(TypeError):
        BaseFeatureDetector()


@pytest.mark.parametrize(
    "detector_cls,kwargs",
    [
        (RawKNNDetector, {"n_neighbors": 3}),
        (MahalanobisDetector, {}),
        (PCAReconstructionDetector, {"n_components": 5}),
        (IsolationForestDetector, {"n_estimators": 50, "seed": 7}),
        (OneClassSVMDetector, {"nu": 0.05}),
    ],
)
def test_classical_detectors_separate_clean_anomalies(
    synthetic_bode_splits, detector_cls, kwargs
):
    train_norm, val_norm, fit_val_x, fit_val_y, test_x, test_y = synthetic_bode_splits
    det = detector_cls(**kwargs).fit(train_norm)

    # Semi-supervised threshold calibration on fit_val
    cal = det.calibrate_threshold(fit_val_x, fit_val_y)
    assert "threshold" in cal
    metrics = det.evaluate(test_x, test_y)
    assert metrics["f1"] > 0.90
    assert metrics["auc_roc"] > 0.95

    # Strictly unsupervised calibration (healthy validation only, zero anomaly labels)
    cal_unsup = det.calibrate_threshold(val_norm, unsupervised_percentile=95.0)
    assert cal_unsup["mode"] == "unsupervised_percentile"
    metrics_unsup = det.evaluate(test_x, test_y)
    assert metrics_unsup["f1"] > 0.85


def test_supervised_oracle_detector(synthetic_bode_splits):
    train_norm, _, fit_val_x, fit_val_y, test_x, test_y = synthetic_bode_splits
    oracle = SupervisedOracleDetector(seed=11).fit_oracle(train_norm, fit_val_x, fit_val_y)
    metrics = oracle.evaluate(test_x, test_y)
    assert metrics["f1"] > 0.95
    assert metrics["auc_roc"] >= 0.97


def test_per_fault_recall_separates_single_and_multi_faults():
    variations = [
        {"Cout": 0.0, "Rds_2": 0.0},  # healthy
        {"Cout": -50.0, "Rds_2": 0.0},  # single Cout (detected)
        {"Cout": -40.0, "Esr_C": 400.0},  # multi Cout + Esr_C (detected)
        {"Rds_2": 800.0},  # single Rds_2 (missed)
        {"Rds_1": 500.0, "Rds_2": 600.0},  # multi Rds_1 + Rds_2 (detected)
    ]
    preds = np.array([0, 1, 1, 0, 1])
    breakdown = compute_per_fault_recall(preds, variations)

    assert breakdown["Cout"]["n_any"] == 2
    assert breakdown["Cout"]["recall_any"] == pytest.approx(1.0)
    assert breakdown["Cout"]["n_single"] == 1
    assert breakdown["Cout"]["recall_single"] == pytest.approx(1.0)

    assert breakdown["Rds_2"]["n_any"] == 2
    assert breakdown["Rds_2"]["recall_any"] == pytest.approx(0.5)
    assert breakdown["Rds_2"]["n_single"] == 1
    assert breakdown["Rds_2"]["recall_single"] == pytest.approx(0.0)


def test_inject_fra_measurement_noise():
    raw = np.zeros((10, 101, 2), dtype=np.float64)
    raw[..., 0] = 34.0
    raw[..., 1] = -45.0

    # Zero noise is exact identity
    clean = inject_fra_measurement_noise(raw, 0.0, 0.0, seed=0)
    assert np.array_equal(clean, raw)

    noisy = inject_fra_measurement_noise(raw, sigma_mag_db=0.5, sigma_phase_deg=2.0, seed=0)
    assert noisy.shape == raw.shape
    assert 0.35 < np.std(noisy[..., 0] - raw[..., 0]) < 0.65
    assert 1.5 < np.std(noisy[..., 1] - raw[..., 1]) < 2.5


def test_prevalence_adjusted_prauc_monotonic_in_prevalence():
    rng = np.random.default_rng(0)
    labels = np.array([0] * 100 + [1] * 100)
    scores = np.concatenate([rng.normal(0.0, 1.0, 100), rng.normal(1.5, 1.0, 100)])

    pr_low = compute_prevalence_adjusted_prauc(labels, scores, target_prevalence=0.05)
    pr_mid = compute_prevalence_adjusted_prauc(labels, scores, target_prevalence=0.20)
    pr_high = compute_prevalence_adjusted_prauc(labels, scores, target_prevalence=0.60)
    assert 0.0 < pr_low < pr_mid < pr_high <= 1.0


def test_paired_sign_flip_test_exact():
    a = [0.89, 0.88, 0.90, 0.87, 0.89, 0.88, 0.91, 0.90]
    b = [0.85, 0.84, 0.86, 0.83, 0.85, 0.84, 0.87, 0.86]
    res = paired_sign_flip_test(a, b)
    assert res["mean_diff"] == pytest.approx(0.04)
    # 8/8 strictly positive -> exact two-sided sign-flip p = 2 / 2^8 = 1/128 = 0.0078125
    assert res["p_value_two_sided"] == pytest.approx(2.0 / 256.0)
