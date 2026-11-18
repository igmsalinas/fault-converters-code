#!/usr/bin/env python3
"""Comprehensive technical verification and ablation study for manuscript resubmission.

Produces reproducible artifacts under ``papers/results/mlwa_study/`` and stores
newly trained checkpoints under ``experiments/dataset_01_mlwa_study/`` without
modifying the frozen companion-paper directory ``experiments/dataset_01_run_06_corrected/``.

Blocks:
1. PSIM data deduplication & exact State-Space Averaged (SSA) physics audit.
2. Classical (non-DL) baselines, frozen CARLA backbones, per-fault recall,
   prevalence-adjusted PR-AUC, and strictly unsupervised threshold calibration.
3. Swept-sine FRA measurement-noise robustness sweep.
4. 5-seed paired ablation (recon_only_mse, recon_only_latent, heuristic,
   physics_legacy, physics_exact_ssa) + strict Leave-One-Fault-Out (LOFO).
"""

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import keras
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src  # noqa: F401
from sklearn.metrics import f1_score, roc_auc_score

from src.data.dataset import PowerConverterDataset
from src.data.physics_anomaly import (
    DEFAULT_FAULT_MODES,
    BuckConverter,
    PowerConverter,
    available_fault_modes,
    evaluate_transfer_ratio_error,
)
from src.data.preprocessor import DataPreprocessor
from src.evaluation.baselines import (
    IsolationForestDetector,
    MahalanobisDetector,
    OneClassSVMDetector,
    PCAReconstructionDetector,
    RawKNNDetector,
    SupervisedOracleDetector,
)
from src.evaluation.fault_analysis import (
    DEFAULT_ANOMALY_PCT_BANDS,
    compute_per_fault_recall,
    compute_prevalence_adjusted_prauc,
    inject_fra_measurement_noise,
    paired_sign_flip_test,
)
from src.evaluation.metrics import compute_classification_metrics
from src.inference.predictor import AnomalyPredictor
from src.losses.contrastive import anomaly_score_from_embeddings
from src.models.contrastive_ae import ContrastiveAutoencoder
from src.training.carla_trainer import CARLAConfig, CARLATrainer

METHODS = ("knn", "centroid", "cosine", "mahalanobis")
K_VALUES = (1, 3, 5, 10, 20, 50)
PERCENTILES = (70.0, 80.0, 90.0, 95.0, 98.0, 99.0, 99.5, 99.9)

FAULT_MODE_TO_DATASET_COMP = {
    "capacitor_esr": "Esr_C",
    "capacitance_drop": "Cout",
    "inductor_saturation": "Lout",
    "inductor_esr": "Esr_L",
    "switch_degradation": "Rds_1",
}


def _f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    try:
        return float(f1_score(labels, predictions, zero_division=0))
    except TypeError:
        return float(f1_score(labels, predictions))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _model_weights_sha256(model) -> str:
    digest = hashlib.sha256()
    arrays = list(model.autoencoder.get_weights()) + list(
        model.projection_head.get_weights()
    )
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _active_fault_components(
    variations: Dict[str, float],
    tol: float = 1e-3,
) -> List[str]:
    return [
        comp
        for comp, (lo, hi) in DEFAULT_ANOMALY_PCT_BANDS.items()
        if (lo - tol) <= float(variations.get(comp, 0.0)) <= (hi + tol)
    ]


def _load_frozen_dataset(
    data_dir: Path,
    main_model_dir: Path,
    cache_dir: Path,
) -> PowerConverterDataset:
    preprocessor = DataPreprocessor.load(str(main_model_dir / "preprocessor"))
    dataset = PowerConverterDataset(
        data_dir=str(data_dir),
        cache_dir=str(cache_dir),
        preprocessor=preprocessor,
    )
    dataset.load()
    dataset.load_split_indices(str(main_model_dir / "split_indices.npz"))
    dataset.preprocess(fit=False)
    return dataset


# ---------------------------------------------------------------------------
# Block 1: PSIM Data & Physics Audit
# ---------------------------------------------------------------------------
def run_data_and_physics_audit(
    dataset: PowerConverterDataset,
    data_dir: Path,
    out_dir: Path,
) -> Dict[str, Any]:
    print("\n=== [Block 1] Running PSIM Data & Exact SSA Physics Audit ===")
    manifest_path = data_dir / "manifest.csv"
    rout_vals = set()
    n_manifest_rows = 0
    n_healthy_rows = 0
    n_fault_rows = 0
    unique_healthy_vecs = set()
    unique_fault_vecs = set()

    with manifest_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_manifest_rows += 1
            if "Rout" in row:
                rout_vals.add(round(float(row["Rout"]), 8))
            vec = tuple(
                round(float(row[c]), 10)
                for c in ("Cout", "Esr_C", "Lout", "Esr_L", "Rds_1", "Rds_2")
            )
            label = str(row.get("label", "")).strip().lower()
            if label in ("0", "normal", "healthy", "false"):
                n_healthy_rows += 1
                unique_healthy_vecs.add(vec)
            else:
                n_fault_rows += 1
                unique_fault_vecs.add(vec)

    splits = dataset._splits
    all_unique_idx = np.concatenate(
        [splits["train_idx"], splits["fit_val_idx"], splits["test_idx"]]
    )
    freqs = np.asarray(dataset.frequencies, dtype=float)
    omega = 2.0 * np.pi * freqs
    nom = BuckConverter()

    errs_legacy = []
    errs_ssa_bidir = []
    errs_ssa_psim_diode = []
    errs_rds2_single_bidir = []
    errs_rds2_single_diode = []

    for idx in all_unique_idx:
        raw = dataset._raw_data[int(idx)]  # (101, 2): [mag_dB, phase_deg]
        mag_lin = 10.0 ** (raw[:, 0] / 20.0)
        ph_rad = np.deg2rad(raw[:, 1])
        h_psim = mag_lin * np.exp(1j * ph_rad)
        norm_psim = max(float(np.linalg.norm(h_psim)), 1e-30)

        v = dataset._metadata[int(idx)].variations
        m_c = 1.0 + float(v.get("Cout", 0.0)) / 100.0
        m_esrc = 1.0 + float(v.get("Esr_C", 0.0)) / 100.0
        m_l = 1.0 + float(v.get("Lout", 0.0)) / 100.0
        m_esrl = 1.0 + float(v.get("Esr_L", 0.0)) / 100.0
        m_rds1 = 1.0 + float(v.get("Rds_1", 0.0)) / 100.0
        m_rds2 = 1.0 + float(v.get("Rds_2", 0.0)) / 100.0

        common = dict(
            capacitance=nom.capacitance * m_c,
            esr_cap=nom.esr_cap * m_esrc,
            inductance=nom.inductance * m_l,
            esr_ind=nom.esr_ind * m_esrl,
        )
        c_leg = BuckConverter(
            **common,
            rds=nom.rds * m_rds1,
            rds_low=nom.rds * m_rds2,
            include_parasitic_drop=False,
        )
        c_bidir = BuckConverter(
            **common,
            rds=nom.rds * m_rds1,
            rds_low=nom.rds * m_rds2,
            include_parasitic_drop=True,
        )
        c_diode = BuckConverter(
            **common,
            rds=nom.rds * m_rds1,
            rds_low=0.0,
            include_parasitic_drop=True,
        )

        h_leg = c_leg.transfer_function().frequency_response(omega)
        h_bidir = c_bidir.transfer_function().frequency_response(omega)
        h_diode = c_diode.transfer_function().frequency_response(omega)

        e_leg = float(np.linalg.norm(h_leg - h_psim)) / norm_psim
        e_bidir = float(np.linalg.norm(h_bidir - h_psim)) / norm_psim
        e_diode = float(np.linalg.norm(h_diode - h_psim)) / norm_psim

        errs_legacy.append(e_leg)
        errs_ssa_bidir.append(e_bidir)
        errs_ssa_psim_diode.append(e_diode)

        act = _active_fault_components(v)
        if act == ["Rds_2"]:
            errs_rds2_single_bidir.append(e_bidir)
            errs_rds2_single_diode.append(e_diode)

    def _summary(arr: Sequence[float]) -> Dict[str, float]:
        a = np.asarray(arr, dtype=float)
        return {
            "count": int(a.size),
            "median_pct": float(np.median(a) * 100.0),
            "mean_pct": float(np.mean(a) * 100.0),
            "p95_pct": float(np.percentile(a, 95) * 100.0),
            "max_pct": float(np.max(a) * 100.0),
        }

    ratio_err_legacy = evaluate_transfer_ratio_error(
        freqs,
        converter=BuckConverter(include_parasitic_drop=False),
        n_samples=1000,
        seed=42,
    )
    ratio_err_ssa_psim = evaluate_transfer_ratio_error(
        freqs,
        converter=BuckConverter(include_parasitic_drop=True, rds_low=0.0),
        n_samples=1000,
        seed=42,
    )
    ratio_err_ssa_bidir = evaluate_transfer_ratio_error(
        freqs,
        converter=BuckConverter(include_parasitic_drop=True, rds_low=10e-3),
        n_samples=1000,
        seed=42,
    )

    audit = {
        "manifest_summary": {
            "total_rows": n_manifest_rows,
            "healthy_rows": n_healthy_rows,
            "fault_rows": n_fault_rows,
            "unique_healthy_component_vectors": len(unique_healthy_vecs),
            "unique_fault_component_vectors": len(unique_fault_vecs),
            "total_unique_states": len(unique_healthy_vecs) + len(unique_fault_vecs),
            "rout_unique_values_ohm": sorted(rout_vals),
            "split_counts": {
                "train_healthy": int(len(splits["train_idx"])),
                "fit_val_total": int(len(splits["fit_val_idx"])),
                "fit_val_healthy": int(np.sum(splits["fit_val_labels"] == 0)),
                "fit_val_anomalous": int(np.sum(splits["fit_val_labels"] == 1)),
                "test_total": int(len(splits["test_idx"])),
                "test_healthy": int(np.sum(splits["test_labels"] == 0)),
                "test_anomalous": int(np.sum(splits["test_labels"] == 1)),
            },
        },
        "psim_vs_analytic_rel_l2_error": {
            "legacy_eq2_lossless_dc": _summary(errs_legacy),
            "exact_ssa_bidirectional_rds2": _summary(errs_ssa_bidir),
            "exact_ssa_psim_body_diode_rds2_zero": _summary(errs_ssa_psim_diode),
            "single_fault_rds2_under_bidirectional_ssa": _summary(errs_rds2_single_bidir),
            "single_fault_rds2_under_psim_body_diode_ssa": _summary(errs_rds2_single_diode),
        },
        "multiplicative_transfer_ratio_approximation": {
            "legacy_eq2": ratio_err_legacy,
            "exact_ssa_psim_rds2_zero": ratio_err_ssa_psim,
            "exact_ssa_hardware_rds2_10mohm": ratio_err_ssa_bidir,
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data_physics_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(
        f"  PSIM L2 median error: legacy={audit['psim_vs_analytic_rel_l2_error']['legacy_eq2_lossless_dc']['median_pct']:.3f}% "
        f"-> exact_ssa_psim={audit['psim_vs_analytic_rel_l2_error']['exact_ssa_psim_body_diode_rds2_zero']['median_pct']:.4f}%"
    )
    return audit


# ---------------------------------------------------------------------------
# Block 2: Classical Baselines, Frozen Backbones & Unsupervised Calibration
# ---------------------------------------------------------------------------
def _summarize_detector_predictions(
    name: str,
    family: str,
    scores: np.ndarray,
    threshold: float,
    test_labels: np.ndarray,
    active5_mask: np.ndarray,
    test_variations: List[Dict[str, float]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    preds = (scores > threshold).astype(int)
    cm = compute_classification_metrics(predictions=preds, labels=test_labels, scores=scores)

    labels_a5 = test_labels[active5_mask]
    scores_a5 = scores[active5_mask]
    preds_a5 = preds[active5_mask]
    cm_a5 = compute_classification_metrics(predictions=preds_a5, labels=labels_a5, scores=scores_a5)

    pfr = compute_per_fault_recall(preds, test_variations)
    passive_single_mask = np.array(
        [
            _active_fault_components(v) in (["Cout"], ["Esr_C"], ["Lout"], ["Esr_L"])
            for v in test_variations
        ],
        dtype=bool,
    )

    row = {
        "detector": name,
        "family": family,
        "threshold": float(threshold),
        "accuracy": float(cm.accuracy),
        "precision": float(cm.precision),
        "recall": float(cm.recall),
        "f1": float(cm.f1),
        "auc_roc": float(cm.auc_roc),
        "auc_pr_pi59": float(cm.auc_pr),
        "auc_pr_pi10": compute_prevalence_adjusted_prauc(test_labels, scores, 0.10),
        "auc_pr_pi05": compute_prevalence_adjusted_prauc(test_labels, scores, 0.05),
        "active5_f1": float(cm_a5.f1),
        "active5_recall": float(cm_a5.recall),
        "active5_precision": float(cm_a5.precision),
        "active5_auc_roc": float(cm_a5.auc_roc),
        "active5_auc_pr_pi10": compute_prevalence_adjusted_prauc(labels_a5, scores_a5, 0.10),
        "recall_passive_single": float(preds[passive_single_mask].mean()),
        "recall_rds1_single": float(pfr["Rds_1"]["recall_single"]),
        "recall_rds2_single": float(pfr["Rds_2"]["recall_single"]),
    }

    fault_row: Dict[str, Any] = {"detector": name, "family": family}
    for comp, stats in pfr.items():
        fault_row[f"{comp}_n_any"] = stats["n_any"]
        fault_row[f"{comp}_recall_any"] = stats["recall_any"]
        if comp != "multi_fault":
            fault_row[f"{comp}_n_single"] = stats["n_single"]
            fault_row[f"{comp}_recall_single"] = stats["recall_single"]

    return row, fault_row


def run_baselines_and_calibration(
    dataset: PowerConverterDataset,
    frozen_exp_dir: Path,
    out_dir: Path,
) -> Dict[str, Any]:
    print("\n=== [Block 2] Evaluating Classical Baselines, Frozen CARLA & Calibration ===")
    splits = dataset._splits
    test_variations = [dataset._metadata[int(i)].variations for i in splits["test_idx"]]
    active5_mask = np.array(
        [_active_fault_components(v) != ["Rds_2"] for v in test_variations],
        dtype=bool,
    )
    test_labels = np.asarray(dataset.test_labels, dtype=int)
    fit_val_labels = np.asarray(dataset.fit_val_labels, dtype=int)
    fit_val_normal_mask = fit_val_labels == 0

    classical_specs = [
        ("Raw k-NN (k=1)", RawKNNDetector(n_neighbors=1)),
        ("Raw k-NN (k=5)", RawKNNDetector(n_neighbors=5)),
        ("Mahalanobis (Ledoit-Wolf)", MahalanobisDetector(ridge=None)),
        ("Mahalanobis (Ridge 1e-6)", MahalanobisDetector(ridge=1e-6)),
        ("PCA Reconstruction (d=10)", PCAReconstructionDetector(n_components=10, seed=42)),
        ("Isolation Forest (200)", IsolationForestDetector(n_estimators=200, seed=42)),
        ("One-Class SVM (RBF)", OneClassSVMDetector(nu=0.05, kernel="rbf")),
    ]

    comparison_rows: List[Dict[str, Any]] = []
    fault_rows: List[Dict[str, Any]] = []
    calibration_rows: List[Dict[str, Any]] = []
    fitted_classical: Dict[str, Any] = {}

    for name, det in classical_specs:
        det.fit(dataset.train_data)
        cal_fv = det.calibrate_threshold(dataset.fit_val_data, fit_val_labels)
        scores_test = det.score(dataset.test_data)
        row, frow = _summarize_detector_predictions(
            name, "classical_unsupervised", scores_test, float(cal_fv["threshold"]),
            test_labels, active5_mask, test_variations
        )
        row["selected_normal_percentile"] = float(cal_fv["normal_percentile"])
        comparison_rows.append(row)
        fault_rows.append(frow)
        fitted_classical[name] = det

        if name in ("Raw k-NN (k=5)", "Mahalanobis (Ledoit-Wolf)", "PCA Reconstruction (d=10)"):
            val_normal_scores = det.score(dataset.fit_val_data[fit_val_normal_mask])
            for mode_label, th in [
                ("fit_val_f1", float(cal_fv["threshold"])),
                ("unsupervised_p95", float(np.percentile(val_normal_scores, 95.0))),
                ("unsupervised_p97.5", float(np.percentile(val_normal_scores, 97.5))),
            ]:
                r_cal, _ = _summarize_detector_predictions(
                    name, mode_label, scores_test, th,
                    test_labels, active5_mask, test_variations
                )
                calibration_rows.append(
                    {
                        "detector": name,
                        "calibration_mode": mode_label,
                        "threshold": th,
                        "precision": r_cal["precision"],
                        "recall": r_cal["recall"],
                        "f1": r_cal["f1"],
                        "active5_f1": r_cal["active5_f1"],
                        "active5_recall": r_cal["active5_recall"],
                    }
                )

    # Supervised GBM oracle (empirical upper bound)
    oracle = SupervisedOracleDetector(seed=42)
    oracle.fit_oracle(dataset.train_data, dataset.fit_val_data, fit_val_labels)
    oracle_scores = oracle.score(dataset.test_data)
    row_or, frow_or = _summarize_detector_predictions(
        "Supervised GBM Oracle", "supervised_ceiling", oracle_scores, 0.5,
        test_labels, active5_mask, test_variations
    )
    row_or["selected_normal_percentile"] = float("nan")
    comparison_rows.append(row_or)
    fault_rows.append(frow_or)

    # Frozen CARLA models from dataset_01_run_06_corrected
    frozen_predictors: Dict[str, AnomalyPredictor] = {}
    for subdir, label in [
        ("carla_conv1d", "CARLA-Conv1D (frozen)"),
        ("carla_mlp", "CARLA-MLP (frozen)"),
        ("carla_transformer", "CARLA-Transformer (frozen)"),
        ("carla_lstm", "CARLA-LSTM (frozen)"),
        ("carla_gru", "CARLA-GRU (frozen)"),
    ]:
        mdir = frozen_exp_dir / subdir
        if not mdir.exists():
            continue
        pred = AnomalyPredictor(str(mdir))
        res_test = pred.predict(dataset.test_data, preprocess=False)
        scores_test = np.asarray(res_test["anomaly_score"], dtype=float)
        th_data = json.loads((mdir / "threshold.json").read_text())
        row_c, frow_c = _summarize_detector_predictions(
            label, "carla_frozen", scores_test, float(pred.threshold),
            test_labels, active5_mask, test_variations
        )
        row_c["selected_normal_percentile"] = float(
            th_data.get("selection", {}).get("normal_score_percentile", float("nan"))
        )
        comparison_rows.append(row_c)
        fault_rows.append(frow_c)
        frozen_predictors[label] = pred

        if subdir in ("carla_conv1d", "carla_mlp"):
            res_fv = pred.predict(dataset.fit_val_data, preprocess=False)
            fv_scores = np.asarray(res_fv["anomaly_score"], dtype=float)
            val_normal_scores = fv_scores[fit_val_normal_mask]
            for mode_label, th in [
                ("fit_val_f1", float(pred.threshold)),
                ("unsupervised_p95", float(np.percentile(val_normal_scores, 95.0))),
                ("unsupervised_p97.5", float(np.percentile(val_normal_scores, 97.5))),
            ]:
                r_cal, _ = _summarize_detector_predictions(
                    label, mode_label, scores_test, th,
                    test_labels, active5_mask, test_variations
                )
                calibration_rows.append(
                    {
                        "detector": label,
                        "calibration_mode": mode_label,
                        "threshold": th,
                        "precision": r_cal["precision"],
                        "recall": r_cal["recall"],
                        "f1": r_cal["f1"],
                        "active5_f1": r_cal["active5_f1"],
                        "active5_recall": r_cal["active5_recall"],
                    }
                )

    _write_csv(out_dir / "baselines_comparison.csv", comparison_rows)
    _write_csv(out_dir / "per_fault_recall.csv", fault_rows)
    _write_csv(out_dir / "calibration_comparison.csv", calibration_rows)
    print(f"  Wrote {len(comparison_rows)} detectors to baselines_comparison.csv")
    return {
        "fitted_classical": fitted_classical,
        "frozen_predictors": frozen_predictors,
        "active5_mask": active5_mask,
        "test_variations": test_variations,
    }


# ---------------------------------------------------------------------------
# Block 3: FRA Swept-Sine Measurement Noise Robustness Sweep
# ---------------------------------------------------------------------------
def run_fra_noise_sweep(
    dataset: PowerConverterDataset,
    block2_ctx: Dict[str, Any],
    out_dir: Path,
) -> List[Dict[str, Any]]:
    print("\n=== [Block 3] Running FRA Measurement-Noise Robustness Sweep ===")
    splits = dataset._splits
    raw_test = dataset._raw_data[splits["test_idx"]]
    raw_fit_val = dataset._raw_data[splits["fit_val_idx"]]
    test_labels = np.asarray(dataset.test_labels, dtype=int)
    fit_val_labels = np.asarray(dataset.fit_val_labels, dtype=int)
    active5_mask = block2_ctx["active5_mask"]

    noise_levels = [
        (0.00, 0.00),
        (0.05, 0.25),
        (0.15, 0.75),
        (0.30, 1.50),
        (0.60, 3.00),
    ]
    mc_seeds = (100, 101, 102, 103, 104)

    detectors_to_test: List[Tuple[str, Any, str]] = []
    for cname in (
        "Raw k-NN (k=1)",
        "Raw k-NN (k=5)",
        "Mahalanobis (Ledoit-Wolf)",
        "Mahalanobis (Ridge 1e-6)",
        "PCA Reconstruction (d=10)",
    ):
        detectors_to_test.append((cname, block2_ctx["fitted_classical"][cname], "classical"))
    for pname in ("CARLA-Conv1D (frozen)", "CARLA-MLP (frozen)"):
        if pname in block2_ctx["frozen_predictors"]:
            detectors_to_test.append((pname, block2_ctx["frozen_predictors"][pname], "carla"))

    rows: List[Dict[str, Any]] = []
    for sigma_db, sigma_deg in noise_levels:
        draws = [0] if (sigma_db == 0.0 and sigma_deg == 0.0) else list(mc_seeds)
        per_det_metrics: Dict[str, List[Dict[str, float]]] = {
            name: [] for name, _, _ in detectors_to_test
        }
        for s in draws:
            noisy_raw_test = inject_fra_measurement_noise(
                raw_test, sigma_db, sigma_deg, seed=s
            )
            noisy_proc_test = dataset.preprocessor.transform(noisy_raw_test)

            noisy_raw_fv = inject_fra_measurement_noise(
                raw_fit_val, sigma_db, sigma_deg, seed=s + 1000
            )
            noisy_proc_fv = dataset.preprocessor.transform(noisy_raw_fv)

            for name, obj, kind in detectors_to_test:
                if kind == "classical":
                    clean_th = float(obj.threshold_)
                    scores_test = obj.score(noisy_proc_test)
                    scores_fv = obj.score(noisy_proc_fv)
                else:
                    clean_th = float(obj.threshold)
                    scores_test = np.asarray(
                        obj.predict(noisy_proc_test, preprocess=False)["anomaly_score"],
                        dtype=float,
                    )
                    scores_fv = np.asarray(
                        obj.predict(noisy_proc_fv, preprocess=False)["anomaly_score"],
                        dtype=float,
                    )

                preds_fixed = (scores_test > clean_th).astype(int)
                f1_fixed = _f1(test_labels, preds_fixed)
                f1_fixed_a5 = _f1(test_labels[active5_mask], preds_fixed[active5_mask])
                auc_full = float(roc_auc_score(test_labels, scores_test))
                auc_a5 = float(roc_auc_score(test_labels[active5_mask], scores_test[active5_mask]))

                # Noise-matched recalibrated threshold on noisy validation
                best_recal_th = clean_th
                best_fv_f1 = -1.0
                fv_norm = fit_val_labels == 0
                for pct in PERCENTILES:
                    th_c = float(np.percentile(scores_fv[fv_norm], pct))
                    f1_c = _f1(fit_val_labels, (scores_fv > th_c).astype(int))
                    if f1_c > best_fv_f1:
                        best_fv_f1 = f1_c
                        best_recal_th = th_c
                preds_recal = (scores_test > best_recal_th).astype(int)
                f1_recal = _f1(test_labels, preds_recal)
                f1_recal_a5 = _f1(test_labels[active5_mask], preds_recal[active5_mask])

                per_det_metrics[name].append(
                    {
                        "f1_fixed_th": f1_fixed,
                        "active5_f1_fixed_th": f1_fixed_a5,
                        "f1_recal_th": f1_recal,
                        "active5_f1_recal_th": f1_recal_a5,
                        "auc_roc": auc_full,
                        "active5_auc_roc": auc_a5,
                    }
                )

        for name, _, kind in detectors_to_test:
            m_list = per_det_metrics[name]
            rows.append(
                {
                    "detector": name,
                    "kind": kind,
                    "sigma_mag_db": sigma_db,
                    "sigma_phase_deg": sigma_deg,
                    "mc_runs": len(m_list),
                    "f1_fixed_th_mean": float(np.mean([m["f1_fixed_th"] for m in m_list])),
                    "f1_fixed_th_std": float(np.std([m["f1_fixed_th"] for m in m_list])),
                    "active5_f1_fixed_th_mean": float(
                        np.mean([m["active5_f1_fixed_th"] for m in m_list])
                    ),
                    "f1_recal_th_mean": float(np.mean([m["f1_recal_th"] for m in m_list])),
                    "active5_f1_recal_th_mean": float(
                        np.mean([m["active5_f1_recal_th"] for m in m_list])
                    ),
                    "auc_roc_mean": float(np.mean([m["auc_roc"] for m in m_list])),
                    "active5_auc_roc_mean": float(
                        np.mean([m["active5_auc_roc"] for m in m_list])
                    ),
                }
            )

    _write_csv(out_dir / "fra_noise_robustness.csv", rows)
    print(f"  Wrote {len(rows)} FRA noise sweep rows to fra_noise_robustness.csv")
    return rows


# ---------------------------------------------------------------------------
# Block 4: 5-Seed Paired Ablation + Leave-One-Fault-Out (LOFO)
# ---------------------------------------------------------------------------
def _select_embedding_detector(
    eval_embeddings_fn,
    ref_embeddings: np.ndarray,
    fit_val_data: np.ndarray,
    fit_val_labels: np.ndarray,
    normalize: bool = True,
    batch_size: int = 128,
) -> Dict[str, Any]:
    val_emb = eval_embeddings_fn(fit_val_data)
    normal = fit_val_labels == 0
    best: Optional[Dict[str, Any]] = None
    for method in METHODS:
        k_values = K_VALUES if method == "knn" else (5,)
        for k in k_values:
            if k > len(ref_embeddings):
                continue
            scores = anomaly_score_from_embeddings(
                val_emb,
                ref_embeddings,
                method=method,
                k=k,
                normalize=normalize,
                batch_size=batch_size,
            )
            for pct in PERCENTILES:
                th = float(np.percentile(scores[normal], pct))
                score = _f1(fit_val_labels, (scores > th).astype(int))
                cand = {
                    "scoring_method": method,
                    "k_neighbors": int(k),
                    "normal_percentile": float(pct),
                    "threshold": th,
                    "fit_validation_f1": score,
                    "normalize": normalize,
                }
                if best is None or score > best["fit_validation_f1"]:
                    best = cand
    assert best is not None
    return best


def _select_reconstruction_detector(
    model: ContrastiveAutoencoder,
    fit_val_data: np.ndarray,
    fit_val_labels: np.ndarray,
) -> Dict[str, Any]:
    scores = model.compute_reconstruction_error(fit_val_data)
    normal = fit_val_labels == 0
    best: Optional[Dict[str, Any]] = None
    for pct in PERCENTILES:
        th = float(np.percentile(scores[normal], pct))
        score = _f1(fit_val_labels, (scores > th).astype(int))
        cand = {
            "scoring_method": "reconstruction_mse",
            "k_neighbors": 0,
            "normal_percentile": float(pct),
            "threshold": th,
            "fit_validation_f1": score,
        }
        if best is None or score > best["fit_validation_f1"]:
            best = cand
    assert best is not None
    return best


def _get_embeddings_batched(
    model: ContrastiveAutoencoder,
    data: np.ndarray,
    key: str = "projection",
    batch_size: int = 128,
) -> np.ndarray:
    chunks = []
    for i in range(0, len(data), batch_size):
        emb = model.get_embeddings(data[i : i + batch_size])
        chunks.append(np.asarray(emb[key]))
    out = np.concatenate(chunks, axis=0)
    if key == "projection":
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        out = out / (norms + 1e-8)
    return out


def _train_or_load_carla_checkpoint(
    selected: Dict[str, Any],
    regime: str,
    seed: int,
    run_dir: Path,
    dataset: PowerConverterDataset,
    epochs: int,
    batch_size: int,
    converter_spec: Optional[PowerConverter] = None,
    fault_modes: Optional[List[str]] = None,
    contrastive_weight_override: Optional[float] = None,
    use_physics: bool = True,
    existing_readonly_dir: Optional[Path] = None,
) -> Tuple[ContrastiveAutoencoder, Dict[str, Any]]:
    # 1. Check read-only frozen checkpoint first (e.g. seeds 200042..200044)
    if existing_readonly_dir is not None:
        ro_best = existing_readonly_dir / "checkpoints" / "best"
        if (ro_best / "autoencoder.weights.h5").exists():
            prov_path = existing_readonly_dir / "ablation_training_provenance.json"
            prov = json.loads(prov_path.read_text()) if prov_path.exists() else {}
            model = ContrastiveAutoencoder.load(str(ro_best), config=selected)
            return model, {
                "regime": regime,
                "seed": seed,
                "reused_readonly_checkpoint": str(ro_best),
                "initial_model_weights_sha256": prov.get("initial_model_weights_sha256", ""),
                "training_seconds": float(prov.get("training_seconds", 0.0)),
                "epochs_trained": int(prov.get("epochs_trained", 0)),
            }

    # 2. Check our own study checkpoint directory for resume support
    best_ckpt = run_dir / "checkpoints" / "best"
    prov_file = run_dir / "training_provenance.json"
    if (best_ckpt / "autoencoder.weights.h5").exists() and prov_file.exists():
        prov = json.loads(prov_file.read_text())
        model = ContrastiveAutoencoder.load(str(best_ckpt), config=selected)
        return model, prov

    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    cw = (
        float(selected["contrastive_weight"])
        if contrastive_weight_override is None
        else float(contrastive_weight_override)
    )
    config = CARLAConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=float(selected["learning_rate"]),
        reconstruction_weight=float(selected["reconstruction_weight"]),
        contrastive_weight=cw,
        temperature=float(selected["temperature"]),
        anomaly_ratio=1.0,
        seed=seed,
        use_physics_anomalies=use_physics,
        anomaly_types=fault_modes,
        resume_from_latest=False,
        scoring_method=str(selected["scoring_method"]),
        k_neighbors=int(selected["k_neighbors"]),
        checkpoint_dir=str(run_dir / "checkpoints"),
        log_dir=str(run_dir / "logs"),
        early_stopping=True,
        patience=20,
        save_best_only=True,
    )
    trainer = CARLATrainer(config, experiment_name=f"{regime}_{seed}")
    trainer.create_model(
        input_shape=dataset.input_shape,
        latent_dim=int(selected["latent_dim"]),
        projection_dim=int(selected["projection_dim"]),
        encoder_type=str(selected["encoder_type"]),
        encoder_filters=selected["encoder_filters"],
        encoder_units=selected.get("encoder_units", [256, 128, 64]),
        kernel_size=int(selected["kernel_size"]),
        num_heads=int(selected["num_heads"]),
        dropout_rate=float(selected["dropout_rate"]),
    )
    init_sha = _model_weights_sha256(trainer.model)
    trainer.set_physics_context(
        frequencies=dataset.frequencies,
        scaler=dataset.preprocessor.scaler,
        use_real_imag=dataset.preprocessor.use_real_imag,
        converter_spec=converter_spec,
    )
    trainer.setup_training()

    t0 = time.perf_counter()
    history = trainer.train(dataset.train_data, dataset.val_data, verbose=0)
    elapsed = time.perf_counter() - t0

    prov = {
        "regime": regime,
        "seed": seed,
        "reused_readonly_checkpoint": None,
        "initial_model_weights_sha256": init_sha,
        "final_best_weights_sha256": _model_weights_sha256(trainer.model),
        "training_seconds": elapsed,
        "epochs_trained": len(history["loss"]),
    }
    prov_file.write_text(json.dumps(prov, indent=2) + "\n")
    print(
        f"  Trained {regime} (seed={seed}) in {elapsed:.1f}s ({len(history['loss'])} epochs, init={init_sha[:10]})"
    )
    return trainer.model, prov


def _evaluate_trained_model(
    model: ContrastiveAutoencoder,
    dataset: PowerConverterDataset,
    scoring_space: str,
    active5_mask: np.ndarray,
    test_variations: List[Dict[str, float]],
    noisy_test_proc_015: np.ndarray,
    noisy_test_proc_030: np.ndarray,
    fit_val_mask: Optional[np.ndarray] = None,
    batch_size: int = 128,
) -> Dict[str, Any]:
    fv_data = dataset.fit_val_data if fit_val_mask is None else dataset.fit_val_data[fit_val_mask]
    fv_labels = (
        np.asarray(dataset.fit_val_labels, dtype=int)
        if fit_val_mask is None
        else np.asarray(dataset.fit_val_labels, dtype=int)[fit_val_mask]
    )
    test_labels = np.asarray(dataset.test_labels, dtype=int)

    if scoring_space == "reconstruction_mse":
        det = _select_reconstruction_detector(model, fv_data, fv_labels)
        scores_test = model.compute_reconstruction_error(dataset.test_data)
        scores_n015 = model.compute_reconstruction_error(noisy_test_proc_015)
        scores_n030 = model.compute_reconstruction_error(noisy_test_proc_030)
    else:
        key = "latent" if scoring_space == "latent" else "projection"
        normalize = key == "projection"
        ref_emb = _get_embeddings_batched(model, dataset.train_data, key=key, batch_size=batch_size)
        det = _select_embedding_detector(
            lambda x: _get_embeddings_batched(model, x, key=key, batch_size=batch_size),
            ref_emb,
            fv_data,
            fv_labels,
            normalize=normalize,
            batch_size=batch_size,
        )
        test_emb = _get_embeddings_batched(model, dataset.test_data, key=key, batch_size=batch_size)
        scores_test = anomaly_score_from_embeddings(
            test_emb,
            ref_emb,
            method=det["scoring_method"],
            k=det["k_neighbors"],
            normalize=normalize,
            batch_size=batch_size,
        )
        n015_emb = _get_embeddings_batched(model, noisy_test_proc_015, key=key, batch_size=batch_size)
        scores_n015 = anomaly_score_from_embeddings(
            n015_emb,
            ref_emb,
            method=det["scoring_method"],
            k=det["k_neighbors"],
            normalize=normalize,
            batch_size=batch_size,
        )
        n030_emb = _get_embeddings_batched(model, noisy_test_proc_030, key=key, batch_size=batch_size)
        scores_n030 = anomaly_score_from_embeddings(
            n030_emb,
            ref_emb,
            method=det["scoring_method"],
            k=det["k_neighbors"],
            normalize=normalize,
            batch_size=batch_size,
        )

    th = float(det["threshold"])
    fv_normal_mask = fv_labels == 0
    if scoring_space == "reconstruction_mse":
        fv_norm_scores = model.compute_reconstruction_error(fv_data[fv_normal_mask])
    else:
        key = "latent" if scoring_space == "latent" else "projection"
        normalize = key == "projection"
        fv_norm_emb = _get_embeddings_batched(
            model, fv_data[fv_normal_mask], key=key, batch_size=batch_size
        )
        fv_norm_scores = anomaly_score_from_embeddings(
            fv_norm_emb,
            ref_emb,
            method=det["scoring_method"],
            k=det["k_neighbors"],
            normalize=normalize,
            batch_size=batch_size,
        )
    th_unsup_975 = float(np.percentile(fv_norm_scores, 97.5))

    row, _ = _summarize_detector_predictions(
        scoring_space, scoring_space, scores_test, th,
        test_labels, active5_mask, test_variations
    )
    row_u975, _ = _summarize_detector_predictions(
        scoring_space, "unsupervised_p97.5", scores_test, th_unsup_975,
        test_labels, active5_mask, test_variations
    )
    pfr = compute_per_fault_recall((scores_test > th).astype(int), test_variations)
    row.update(
        {
            "scoring_method": det["scoring_method"],
            "k_neighbors": det["k_neighbors"],
            "normal_percentile": det["normal_percentile"],
            "fit_validation_f1": det["fit_validation_f1"],
            "unsup_p975_f1": float(row_u975["f1"]),
            "unsup_p975_active5_f1": float(row_u975["active5_f1"]),
            "unsup_p975_precision": float(row_u975["precision"]),
            "recall_esrl_any": float(pfr["Esr_L"]["recall_any"]),
            "recall_esrc_any": float(pfr["Esr_C"]["recall_any"]),
            "recall_cout_single": float(pfr["Cout"]["recall_single"]),
            "recall_lout_single": float(pfr["Lout"]["recall_single"]),
            "fra_015db_f1": _f1(test_labels, (scores_n015 > th).astype(int)),
            "fra_015db_active5_f1": _f1(
                test_labels[active5_mask], (scores_n015[active5_mask] > th).astype(int)
            ),
            "fra_030db_f1": _f1(test_labels, (scores_n030 > th).astype(int)),
            "fra_030db_active5_f1": _f1(
                test_labels[active5_mask], (scores_n030[active5_mask] > th).astype(int)
            ),
            "scores": scores_test,
            "predictions": (scores_test > th).astype(int),
        }
    )
    return row


def run_multiseed_ablation_and_lofo(
    dataset: PowerConverterDataset,
    frozen_exp_dir: Path,
    study_exp_dir: Path,
    out_dir: Path,
    seeds: List[int],
    epochs: int,
    batch_size: int,
) -> Dict[str, Any]:
    print(f"\n=== [Block 4] Running {len(seeds)}-Seed Paired Ablation & Strict LOFO ===")
    selected = json.loads((frozen_exp_dir / "carla_conv1d" / "model_config.json").read_text())
    splits = dataset._splits
    test_variations = [dataset._metadata[int(i)].variations for i in splits["test_idx"]]
    active5_mask = np.array(
        [_active_fault_components(v) != ["Rds_2"] for v in test_variations],
        dtype=bool,
    )

    raw_test = dataset._raw_data[splits["test_idx"]]
    noisy_test_proc_015 = dataset.preprocessor.transform(
        inject_fra_measurement_noise(raw_test, 0.15, 0.75, seed=100)
    )
    noisy_test_proc_030 = dataset.preprocessor.transform(
        inject_fra_measurement_noise(raw_test, 0.30, 1.50, seed=100)
    )

    ro_ablation_root = frozen_exp_dir / "ablations" / "negative_sampler"
    ablation_rows: List[Dict[str, Any]] = []
    cached_in_dist_ssa_model_200042: Optional[ContrastiveAutoencoder] = None

    for seed in seeds:
        # 1. Pure AE (contrastive_weight = 0.0) -> evaluate BOTH MSE and latent z scoring
        recon_model, recon_prov = _train_or_load_carla_checkpoint(
            selected=selected,
            regime="recon_only",
            seed=seed,
            run_dir=study_exp_dir / "ablation" / "recon_only" / f"seed_{seed}",
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            contrastive_weight_override=0.0,
            use_physics=False,
        )
        for sub_name, space in [
            ("recon_only_mse", "reconstruction_mse"),
            ("recon_only_latent", "latent"),
        ]:
            ev = _evaluate_trained_model(
                recon_model, dataset, space, active5_mask, test_variations,
                noisy_test_proc_015, noisy_test_proc_030, batch_size=batch_size
            )
            ev.pop("scores")
            ev.pop("predictions")
            ablation_rows.append(
                {
                    "regime": sub_name,
                    "seed": seed,
                    "initial_weights_sha256": recon_prov["initial_model_weights_sha256"],
                    "epochs_trained": recon_prov["epochs_trained"],
                    **ev,
                }
            )

        # 2. Heuristic CARLA
        ro_heur = ro_ablation_root / "heuristic" / f"seed_{seed}"
        heur_model, heur_prov = _train_or_load_carla_checkpoint(
            selected=selected,
            regime="heuristic",
            seed=seed,
            run_dir=study_exp_dir / "ablation" / "heuristic" / f"seed_{seed}",
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            use_physics=False,
            existing_readonly_dir=ro_heur if ro_heur.exists() else None,
        )
        ev_h = _evaluate_trained_model(
            heur_model, dataset, "projection", active5_mask, test_variations,
            noisy_test_proc_015, noisy_test_proc_030, batch_size=batch_size
        )
        ev_h.pop("scores")
        ev_h.pop("predictions")
        ablation_rows.append(
            {
                "regime": "heuristic",
                "seed": seed,
                "initial_weights_sha256": heur_prov["initial_model_weights_sha256"],
                "epochs_trained": heur_prov["epochs_trained"],
                **ev_h,
            }
        )

        # 3. Physics Legacy (Eq. 2 lossless DC: Gd0 = Vin)
        ro_phys = ro_ablation_root / "physics" / f"seed_{seed}"
        leg_model, leg_prov = _train_or_load_carla_checkpoint(
            selected=selected,
            regime="physics_legacy",
            seed=seed,
            run_dir=study_exp_dir / "ablation" / "physics_legacy" / f"seed_{seed}",
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            converter_spec=BuckConverter(include_parasitic_drop=False),
            use_physics=True,
            existing_readonly_dir=ro_phys if ro_phys.exists() else None,
        )
        ev_l = _evaluate_trained_model(
            leg_model, dataset, "projection", active5_mask, test_variations,
            noisy_test_proc_015, noisy_test_proc_030, batch_size=batch_size
        )
        ev_l.pop("scores")
        ev_l.pop("predictions")
        ablation_rows.append(
            {
                "regime": "physics_legacy",
                "seed": seed,
                "initial_weights_sha256": leg_prov["initial_model_weights_sha256"],
                "epochs_trained": leg_prov["epochs_trained"],
                **ev_l,
            }
        )

        # 4. Physics Exact SSA (Parasitic series drop + single-switch modulation)
        ssa_model, ssa_prov = _train_or_load_carla_checkpoint(
            selected=selected,
            regime="physics_exact_ssa",
            seed=seed,
            run_dir=study_exp_dir / "ablation" / "physics_exact_ssa" / f"seed_{seed}",
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            converter_spec=BuckConverter(include_parasitic_drop=True, rds_low=0.0),
            use_physics=True,
        )
        if seed == seeds[0]:
            cached_in_dist_ssa_model_200042 = ssa_model
        ev_s = _evaluate_trained_model(
            ssa_model, dataset, "projection", active5_mask, test_variations,
            noisy_test_proc_015, noisy_test_proc_030, batch_size=batch_size
        )
        ev_s.pop("scores")
        ev_s.pop("predictions")
        ablation_rows.append(
            {
                "regime": "physics_exact_ssa",
                "seed": seed,
                "initial_weights_sha256": ssa_prov["initial_model_weights_sha256"],
                "epochs_trained": ssa_prov["epochs_trained"],
                **ev_s,
            }
        )

        # Persist incrementally after every seed
        _write_csv(out_dir / "ablation_multiseed.csv", ablation_rows)

    # Summarize paired statistics across seeds
    regimes = [
        "recon_only_mse",
        "recon_only_latent",
        "heuristic",
        "physics_legacy",
        "physics_exact_ssa",
    ]
    key_metrics = [
        "f1",
        "recall",
        "precision",
        "auc_roc",
        "auc_pr_pi10",
        "auc_pr_pi05",
        "active5_f1",
        "active5_recall",
        "active5_auc_roc",
        "unsup_p975_f1",
        "unsup_p975_active5_f1",
        "recall_rds1_single",
        "recall_esrl_any",
        "recall_passive_single",
        "fra_015db_f1",
        "fra_030db_f1",
    ]
    by_rs = {(r["regime"], r["seed"]): r for r in ablation_rows}
    summary: Dict[str, Any] = {
        "seeds": seeds,
        "regimes": {},
        "paired_comparisons": {},
    }
    for reg in regimes:
        rows_r = [by_rs[(reg, s)] for s in seeds]
        summary["regimes"][reg] = {
            m: {
                "mean": float(np.mean([x[m] for x in rows_r])),
                "sample_std": float(np.std([x[m] for x in rows_r], ddof=1))
                if len(rows_r) > 1
                else 0.0,
                "values": [float(x[m]) for x in rows_r],
            }
            for m in key_metrics
        }

    pairs = [
        ("physics_exact_ssa", "physics_legacy"),
        ("physics_exact_ssa", "heuristic"),
        ("physics_exact_ssa", "recon_only_mse"),
        ("physics_exact_ssa", "recon_only_latent"),
        ("physics_legacy", "heuristic"),
    ]
    for reg_a, reg_b in pairs:
        pair_key = f"{reg_a}_vs_{reg_b}"
        summary["paired_comparisons"][pair_key] = {
            m: paired_sign_flip_test(
                [by_rs[(reg_a, s)][m] for s in seeds],
                [by_rs[(reg_b, s)][m] for s in seeds],
            )
            for m in (
                "f1",
                "active5_f1",
                "unsup_p975_f1",
                "unsup_p975_active5_f1",
                "auc_roc",
                "active5_auc_roc",
                "auc_pr_pi05",
                "recall_rds1_single",
                "fra_015db_f1",
            )
        }

    (out_dir / "ablation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Strict Leave-One-Fault-Out (LOFO) on seed[0]
    print(f"\n=== [Block 4b] Running Strict Leave-One-Fault-Out (seed={seeds[0]}) ===")
    lofo_seed = seeds[0]
    all_modes = available_fault_modes()
    fit_val_variations = [dataset._metadata[int(i)].variations for i in splits["fit_val_idx"]]

    assert cached_in_dist_ssa_model_200042 is not None
    indist_eval = _evaluate_trained_model(
        cached_in_dist_ssa_model_200042, dataset, "projection", active5_mask,
        test_variations, noisy_test_proc_015, noisy_test_proc_030, batch_size=batch_size
    )
    indist_pfr = compute_per_fault_recall(indist_eval["predictions"], test_variations)

    lofo_rows: List[Dict[str, Any]] = []
    for held_out_mode in all_modes:
        held_comp = FAULT_MODE_TO_DATASET_COMP[held_out_mode]
        kept_modes = [m for m in all_modes if m != held_out_mode]

        # Exclude ANY fit_val anomaly involving held_comp so threshold selection
        # also has zero exposure to the held-out physical fault mode.
        fv_keep_mask = np.array(
            [
                (int(lbl) == 0) or (held_comp not in _active_fault_components(v))
                for lbl, v in zip(dataset.fit_val_labels, fit_val_variations)
            ],
            dtype=bool,
        )

        lofo_model, lofo_prov = _train_or_load_carla_checkpoint(
            selected=selected,
            regime=f"lofo_no_{held_out_mode}",
            seed=lofo_seed,
            run_dir=study_exp_dir / "lofo" / f"no_{held_out_mode}_seed_{lofo_seed}",
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            converter_spec=BuckConverter(include_parasitic_drop=True, rds_low=0.0),
            fault_modes=kept_modes,
            use_physics=True,
        )
        ev_lofo = _evaluate_trained_model(
            lofo_model,
            dataset,
            "projection",
            active5_mask,
            test_variations,
            noisy_test_proc_015,
            noisy_test_proc_030,
            fit_val_mask=fv_keep_mask,
            batch_size=batch_size,
        )
        pfr_lofo = compute_per_fault_recall(ev_lofo["predictions"], test_variations)

        # Single-fault and any-fault ROC-AUC on healthy test vs held-out fault test
        single_held_mask = np.array(
            [_active_fault_components(v) == [held_comp] for v in test_variations],
            dtype=bool,
        )
        any_held_mask = np.array(
            [held_comp in _active_fault_components(v) for v in test_variations],
            dtype=bool,
        )
        healthy_test_mask = np.asarray(dataset.test_labels, dtype=int) == 0
        y_int = np.asarray(dataset.test_labels, dtype=int)

        if single_held_mask.any():
            sub_single = healthy_test_mask | single_held_mask
            held_single_auc_lofo = float(
                roc_auc_score(y_int[sub_single], ev_lofo["scores"][sub_single])
            )
            held_single_auc_indist = float(
                roc_auc_score(y_int[sub_single], indist_eval["scores"][sub_single])
            )
        else:
            held_single_auc_lofo = float("nan")
            held_single_auc_indist = float("nan")

        sub_any = healthy_test_mask | any_held_mask
        held_any_auc_lofo = float(
            roc_auc_score(y_int[sub_any], ev_lofo["scores"][sub_any])
        )
        held_any_auc_indist = float(
            roc_auc_score(y_int[sub_any], indist_eval["scores"][sub_any])
        )

        lofo_rows.append(
            {
                "held_out_fault_mode": held_out_mode,
                "dataset_component": held_comp,
                "n_test_single_fault": int(pfr_lofo[held_comp]["n_single"]),
                "n_test_any_fault": int(pfr_lofo[held_comp]["n_any"]),
                "lofo_recall_single": float(pfr_lofo[held_comp]["recall_single"]),
                "indist_recall_single": float(indist_pfr[held_comp]["recall_single"]),
                "lofo_recall_any": float(pfr_lofo[held_comp]["recall_any"]),
                "indist_recall_any": float(indist_pfr[held_comp]["recall_any"]),
                "lofo_single_auc_roc": held_single_auc_lofo,
                "indist_single_auc_roc": held_single_auc_indist,
                "lofo_any_auc_roc": held_any_auc_lofo,
                "indist_any_auc_roc": held_any_auc_indist,
                "lofo_overall_f1": float(ev_lofo["f1"]),
                "lofo_active5_f1": float(ev_lofo["active5_f1"]),
                "epochs_trained": int(lofo_prov["epochs_trained"]),
            }
        )
        _write_csv(out_dir / "lofo_results.csv", lofo_rows)

    print("  Completed strict LOFO across all 5 physical fault modes.")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frozen-experiment",
        default="experiments/dataset_01_run_06_corrected",
    )
    parser.add_argument(
        "--study-experiment",
        default="experiments/dataset_01_mlwa_study",
    )
    parser.add_argument(
        "--data-dir",
        default="data/buck/buck_data/dataset_01",
    )
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument(
        "--out-dir",
        default="papers/results/mlwa_study",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[200042, 200043, 200044, 200045, 200046],
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--skip-block4",
        action="store_true",
        help="Run only Blocks 1-3 (fast audit, baselines, FRA noise sweep).",
    )
    parser.add_argument(
        "--only-block4",
        action="store_true",
        help="Run only Block 4 (5-seed paired ablation + LOFO).",
    )
    args = parser.parse_args()

    frozen_exp = ROOT / args.frozen_experiment
    study_exp = ROOT / args.study_experiment
    data_dir = ROOT / args.data_dir
    cache_dir = ROOT / args.cache_dir
    out_dir = ROOT / args.out_dir

    dataset = _load_frozen_dataset(data_dir, frozen_exp / "carla_conv1d", cache_dir)

    if not args.only_block4:
        run_data_and_physics_audit(dataset, data_dir, out_dir)
        b2_ctx = run_baselines_and_calibration(dataset, frozen_exp, out_dir)
        run_fra_noise_sweep(dataset, b2_ctx, out_dir)

    if not args.skip_block4:
        run_multiseed_ablation_and_lofo(
            dataset=dataset,
            frozen_exp_dir=frozen_exp,
            study_exp_dir=study_exp,
            out_dir=out_dir,
            seeds=list(args.seeds),
            epochs=args.epochs,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
