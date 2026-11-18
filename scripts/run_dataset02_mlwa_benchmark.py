#!/usr/bin/env python3
"""Complete MLWA Benchmark on Balanced Variable-Load PSIM Dataset (dataset_02).

Executes on `data/buck/buck_data_v2/dataset_02` (800 real Altair PSIM 2026 AC sweeps:
500 healthy + 300 faulty with variable operating load `Rout in [0.5, 2.0]x` and
round-robin LHS fault balance across Cout, Esr_C, Lout, Esr_L, Rds_1):

1. Data & Exact SSA Variable-Load Physics Audit (`data_physics_audit.json`).
2. Stratified Split (`split_indices.npz` + `preprocessor/`):
   300 healthy train, 100 healthy + 90 stratified fault fit_val,
   100 healthy + 210 stratified fault test (~35 pure single-faults/component).
3. Classical Baselines + Supervised Oracle (`baselines_comparison.csv`,
   `per_fault_recall.csv`, `zero_label_baselines.csv`).
4. 5-Seed Paired Deep Ablation (`ablation_multiseed.csv`, `ablation_summary.json`):
   - `recon_only_mse`
   - `recon_only_latent`
   - `heuristic`
   - `physics_ssa_fixed_load` (`adapt_load=False, strict_margin=False`)
   - `physics_ssa_load_adapt` (`adapt_load=True, strict_margin=True`)
   - `physics_ssa_load_adapt_fra` (`adapt_load=True, strict_margin=True, fra_jitter_db=0.25`)
5. 5-Architecture Backbone Study across 5 seeds (`encoder_comparison.csv`):
   `conv1d`, `mlp`, `transformer`, `lstm`, `gru`.
6. 5-Mode Strict Leave-One-Fault-Out (`lofo_results.csv`) with ~35 pure
   single-fault test curves for EVERY component (including Esr_L and Esr_C).
"""

from __future__ import annotations

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
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src  # noqa: F401
from src.data.dataset import PowerConverterDataset
from src.data.physics_anomaly import (
    BuckConverter,
    PhysicsAnomalyInjector,
    PowerConverter,
    available_fault_modes,
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
from src.inference.scoring import EmbeddingAnomalyScorer
from src.models.contrastive_ae import ContrastiveAutoencoder
from src.training.carla_trainer import CARLAConfig, CARLATrainer

METHODS = ("knn", "centroid", "cosine", "ledoit_wolf")
K_VALUES = (1, 3, 5, 10, 20)
PERCENTILES = (70.0, 80.0, 90.0, 95.0, 97.5, 98.0, 99.0, 99.5)

FAULT_MODE_TO_DATASET_COMP = {
    "capacitor_esr": "Esr_C",
    "capacitance_drop": "Cout",
    "inductor_saturation": "Lout",
    "inductor_esr": "Esr_L",
    "switch_degradation": "Rds_1",
}


def _f1(labels: np.ndarray, preds: np.ndarray) -> float:
    return float(f1_score(labels, preds, zero_division=0))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _model_weights_sha256(model: ContrastiveAutoencoder) -> str:
    digest = hashlib.sha256()
    arrays = list(model.autoencoder.get_weights()) + list(model.projection_head.get_weights())
    for val in arrays:
        arr = np.ascontiguousarray(np.asarray(val))
        digest.update(str(tuple(arr.shape)).encode("ascii"))
        digest.update(str(arr.dtype).encode("ascii"))
        digest.update(arr.tobytes(order="C"))
    return digest.hexdigest()


def _active_fault_components(variations: Dict[str, float], tol: float = 1e-3) -> List[str]:
    return [
        comp
        for comp, (lo, hi) in DEFAULT_ANOMALY_PCT_BANDS.items()
        if (lo - tol) <= float(variations.get(comp, 0.0)) <= (hi + tol)
    ]


def _fault_stratum(variations: Dict[str, float]) -> str:
    act = _active_fault_components(variations)
    if len(act) == 1:
        return f"single_{act[0]}"
    if len(act) > 1:
        return "multi_fault"
    return "unknown_fault"


def prepare_or_load_stratified_dataset02(
    data_dir: Path,
    study_exp_dir: Path,
    cache_dir: Path,
    split_seed: int = 102,
) -> PowerConverterDataset:
    split_path = study_exp_dir / "split_indices.npz"
    preproc_dir = study_exp_dir / "preprocessor"
    study_exp_dir.mkdir(parents=True, exist_ok=True)

    if split_path.exists() and (preproc_dir / "config.json").exists():
        preprocessor = DataPreprocessor.load(str(preproc_dir))
        dataset = PowerConverterDataset(
            data_dir=str(data_dir),
            cache_dir=str(cache_dir),
            preprocessor=preprocessor,
        )
        dataset.load()
        dataset.load_split_indices(str(split_path))
        dataset.preprocess(fit=False)
        return dataset

    dataset = PowerConverterDataset(
        data_dir=str(data_dir),
        cache_dir=str(cache_dir),
        preprocessor=DataPreprocessor(normalization_method="standard", use_real_imag=True),
    )
    dataset.load()

    rng = np.random.default_rng(split_seed)
    normal_idx = np.asarray(dataset.loader.get_normal_indices(), dtype=np.int64)
    anomaly_idx = np.asarray(dataset.loader.get_anomaly_indices(), dtype=np.int64)

    perm_norm = rng.permutation(len(normal_idx))
    normal_shuf = normal_idx[perm_norm]
    n_train = int(round(0.60 * len(normal_shuf)))
    n_val = int(round(0.20 * len(normal_shuf)))
    train_idx = normal_shuf[:n_train]
    val_idx = normal_shuf[n_train : n_train + n_val]
    test_norm_idx = normal_shuf[n_train + n_val :]

    # Stratify anomalies across single_Cout, single_Esr_C, single_Lout, single_Esr_L, single_Rds_1, multi_fault
    strata: Dict[str, List[int]] = {}
    for idx in anomaly_idx:
        st = _fault_stratum(dataset._metadata[int(idx)].variations)
        strata.setdefault(st, []).append(int(idx))

    fv_anom_list: List[int] = []
    te_anom_list: List[int] = []
    for st in sorted(strata.keys()):
        arr = np.asarray(strata[st], dtype=np.int64)
        arr = arr[rng.permutation(len(arr))]
        n_fv_st = max(1, int(round(0.30 * len(arr))))
        fv_anom_list.extend(arr[:n_fv_st].tolist())
        te_anom_list.extend(arr[n_fv_st:].tolist())

    fit_val_anom_idx = np.asarray(fv_anom_list, dtype=np.int64)
    test_anom_idx = np.asarray(te_anom_list, dtype=np.int64)

    fit_val_idx = np.concatenate([val_idx, fit_val_anom_idx])
    fit_val_labels = np.concatenate(
        [np.zeros(len(val_idx), dtype=np.int64), np.ones(len(fit_val_anom_idx), dtype=np.int64)]
    )
    fv_perm = rng.permutation(len(fit_val_idx))
    fit_val_idx = fit_val_idx[fv_perm]
    fit_val_labels = fit_val_labels[fv_perm]

    test_idx = np.concatenate([test_norm_idx, test_anom_idx])
    test_labels = np.concatenate(
        [np.zeros(len(test_norm_idx), dtype=np.int64), np.ones(len(test_anom_idx), dtype=np.int64)]
    )
    te_perm = rng.permutation(len(test_idx))
    test_idx = test_idx[te_perm]
    test_labels = test_labels[te_perm]

    np.savez(
        split_path,
        train_idx=train_idx,
        val_idx=val_idx,
        fit_val_idx=fit_val_idx,
        fit_val_labels=fit_val_labels,
        test_idx=test_idx,
        test_labels=test_labels,
    )
    dataset.load_split_indices(str(split_path))
    dataset.preprocess(fit=True)
    dataset.preprocessor.save(str(preproc_dir))
    return dataset


def run_dataset02_physics_audit(
    dataset: PowerConverterDataset,
    data_dir: Path,
    out_dir: Path,
) -> Dict[str, Any]:
    print("\n=== [Block 1] Running dataset_02 Variable-Load & Exact SSA Audit ===")
    manifest_path = data_dir / "manifest.csv"
    n_rows = 0
    n_healthy = 0
    n_fault = 0
    rout_healthy: List[float] = []
    rout_fault: List[float] = []
    unique_vecs = set()

    with manifest_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        cols = [
            c
            for c in ("Cout", "Esr_C", "Lout", "Esr_L", "Rds_1", "Rout")
            if c in (reader.fieldnames or [])
        ]
        for row in reader:
            n_rows += 1
            vec = tuple(round(float(row[c]), 10) for c in cols)
            unique_vecs.add(vec)
            r_val = float(row.get("Rout", 1.0))
            lbl = str(row.get("label", "")).strip().lower()
            if lbl in ("0", "normal", "healthy", "false"):
                n_healthy += 1
                rout_healthy.append(r_val)
            else:
                n_fault += 1
                rout_fault.append(r_val)

    freqs = np.asarray(dataset.frequencies, dtype=float)
    omega = 2.0 * np.pi * freqs
    nom = BuckConverter()

    errs_fixed_load_ssa: List[float] = []
    errs_true_load_ssa: List[float] = []
    errs_true_load_legacy: List[float] = []

    stratum_counts_all: Dict[str, int] = {}
    stratum_counts_test: Dict[str, int] = {}

    for idx in range(dataset.num_samples):
        raw = dataset._raw_data[idx]
        h_psim = (10.0 ** (raw[:, 0] / 20.0)) * np.exp(1j * np.deg2rad(raw[:, 1]))
        norm_psim = max(float(np.linalg.norm(h_psim)), 1e-30)

        v = dataset._metadata[idx].variations
        if dataset._metadata[idx].label == "anomalous":
            st = _fault_stratum(v)
            stratum_counts_all[st] = stratum_counts_all.get(st, 0) + 1

        m_c = 1.0 + float(v.get("Cout", 0.0)) / 100.0
        m_esrc = 1.0 + float(v.get("Esr_C", 0.0)) / 100.0
        m_l = 1.0 + float(v.get("Lout", 0.0)) / 100.0
        m_esrl = 1.0 + float(v.get("Esr_L", 0.0)) / 100.0
        m_rds1 = 1.0 + float(v.get("Rds_1", 0.0)) / 100.0
        m_rout = 1.0 + float(v.get("Rout", 0.0)) / 100.0

        common = dict(
            capacitance=nom.capacitance * m_c,
            esr_cap=nom.esr_cap * m_esrc,
            inductance=nom.inductance * m_l,
            esr_ind=nom.esr_ind * m_esrl,
            rds=nom.rds * m_rds1,
            rds_low=0.0,
        )
        c_fixed = BuckConverter(**common, load=nom.load, include_parasitic_drop=True)
        c_true = BuckConverter(
            **common, load=nom.load * m_rout, include_parasitic_drop=True
        )
        c_leg = BuckConverter(
            **common, load=nom.load * m_rout, include_parasitic_drop=False
        )

        errs_fixed_load_ssa.append(
            float(np.linalg.norm(c_fixed.transfer_function().frequency_response(omega) - h_psim))
            / norm_psim
        )
        errs_true_load_ssa.append(
            float(np.linalg.norm(c_true.transfer_function().frequency_response(omega) - h_psim))
            / norm_psim
        )
        errs_true_load_legacy.append(
            float(np.linalg.norm(c_leg.transfer_function().frequency_response(omega) - h_psim))
            / norm_psim
        )

    splits = dataset._splits
    for idx, lbl in zip(splits["test_idx"], splits["test_labels"]):
        if int(lbl) == 1:
            st = _fault_stratum(dataset._metadata[int(idx)].variations)
            stratum_counts_test[st] = stratum_counts_test.get(st, 0) + 1

    # Also evaluate load estimation accuracy of PhysicsAnomalyInjector(adapt_load=True) on healthy curves
    inj_adapt = PhysicsAnomalyInjector(
        frequencies=freqs,
        converter=BuckConverter(include_parasitic_drop=True, rds_low=0.0),
        adapt_load=True,
        strict_margin=True,
    )
    healthy_indices = splits["train_idx"]
    est_routs = []
    true_routs = []
    for idx in healthy_indices:
        raw = dataset._raw_data[int(idx)]
        h_s = (10.0 ** (raw[:, 0] / 20.0)) * np.exp(1j * np.deg2rad(raw[:, 1]))
        est_routs.append(nom.load * inj_adapt._estimate_load_multiplier(h_s))
        v = dataset._metadata[int(idx)].variations
        true_routs.append(nom.load * (1.0 + float(v.get("Rout", 0.0)) / 100.0))
    rel_load_est_err = np.abs(np.asarray(est_routs) - np.asarray(true_routs)) / np.asarray(true_routs)

    def _s(arr: Sequence[float]) -> Dict[str, float]:
        a = np.asarray(arr, dtype=float)
        return {
            "count": int(a.size),
            "median_pct": float(np.median(a) * 100.0),
            "mean_pct": float(np.mean(a) * 100.0),
            "p95_pct": float(np.percentile(a, 95) * 100.0),
            "max_pct": float(np.max(a) * 100.0),
        }

    audit = {
        "manifest_summary": {
            "total_rows": n_rows,
            "healthy_rows": n_healthy,
            "fault_rows": n_fault,
            "unique_component_vectors": len(unique_vecs),
            "duplicate_rows": n_rows - len(unique_vecs),
            "rout_healthy_min_max_multiplier": [float(min(rout_healthy)), float(max(rout_healthy))],
            "rout_fault_min_max_multiplier": [float(min(rout_fault)), float(max(rout_fault))],
            "fault_strata_total": stratum_counts_all,
            "fault_strata_test": stratum_counts_test,
            "split_counts": {
                "train_healthy": int(len(splits["train_idx"])),
                "fit_val_healthy": int(np.sum(splits["fit_val_labels"] == 0)),
                "fit_val_anomalous": int(np.sum(splits["fit_val_labels"] == 1)),
                "test_healthy": int(np.sum(splits["test_labels"] == 0)),
                "test_anomalous": int(np.sum(splits["test_labels"] == 1)),
            },
        },
        "psim_vs_analytic_rel_l2_error": {
            "ssa_ignoring_variable_rout_nominal_4ohm": _s(errs_fixed_load_ssa),
            "legacy_lossless_with_true_rout": _s(errs_true_load_legacy),
            "exact_ssa_with_true_rout": _s(errs_true_load_ssa),
        },
        "anchor_load_estimation_rel_error": _s(rel_load_est_err),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data_physics_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(
        f"  dataset_02 unique={len(unique_vecs)}/{n_rows} | "
        f"SSA L2 median: fixed_load={audit['psim_vs_analytic_rel_l2_error']['ssa_ignoring_variable_rout_nominal_4ohm']['median_pct']:.3f}% "
        f"-> true_load={audit['psim_vs_analytic_rel_l2_error']['exact_ssa_with_true_rout']['median_pct']:.4f}%"
    )
    return audit


def _summarize_predictions(
    name: str,
    family: str,
    scores: np.ndarray,
    threshold: float,
    test_labels: np.ndarray,
    test_variations: List[Dict[str, float]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    preds = (scores > threshold).astype(int)
    cm = compute_classification_metrics(predictions=preds, labels=test_labels, scores=scores)
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
        "auc_pr_test": float(cm.auc_pr),
        "auc_pr_pi10": float(compute_prevalence_adjusted_prauc(test_labels, scores, 0.10)),
        "auc_pr_pi05": float(compute_prevalence_adjusted_prauc(test_labels, scores, 0.05)),
        "recall_passive_single": float(preds[passive_single_mask].mean()),
        "recall_cout_single": float(pfr["Cout"]["recall_single"]),
        "recall_esrc_single": float(pfr["Esr_C"]["recall_single"]),
        "recall_lout_single": float(pfr["Lout"]["recall_single"]),
        "recall_esrl_single": float(pfr["Esr_L"]["recall_single"]),
        "recall_rds1_single": float(pfr["Rds_1"]["recall_single"]),
        "recall_multi_fault": float(pfr["multi_fault"]["recall_any"]),
    }
    frow: Dict[str, Any] = {"detector": name, "family": family}
    for comp in ("Cout", "Esr_C", "Lout", "Esr_L", "Rds_1", "multi_fault"):
        stats = pfr[comp]
        frow[f"{comp}_n_any"] = stats["n_any"]
        frow[f"{comp}_recall_any"] = stats["recall_any"]
        if comp != "multi_fault":
            frow[f"{comp}_n_single"] = stats["n_single"]
            frow[f"{comp}_recall_single"] = stats["recall_single"]
    return row, frow


def run_classical_baselines_dataset02(
    dataset: PowerConverterDataset,
    out_dir: Path,
) -> Dict[str, Any]:
    print("\n=== [Block 2] Classical Baselines & Zero-Label Protocol on dataset_02 ===")
    splits = dataset._splits
    test_variations = [dataset._metadata[int(i)].variations for i in splits["test_idx"]]
    test_labels = np.asarray(dataset.test_labels, dtype=int)
    fv_labels = np.asarray(dataset.fit_val_labels, dtype=int)
    fv_norm_mask = fv_labels == 0

    specs = [
        ("Raw k-NN (k=1)", RawKNNDetector(n_neighbors=1)),
        ("Raw k-NN (k=5)", RawKNNDetector(n_neighbors=5)),
        ("Mahalanobis (Ledoit-Wolf)", MahalanobisDetector(ridge=None)),
        ("Mahalanobis (Ridge 1e-6)", MahalanobisDetector(ridge=1e-6)),
        ("PCA Reconstruction (d=10)", PCAReconstructionDetector(n_components=10, seed=42)),
        ("Isolation Forest (200)", IsolationForestDetector(n_estimators=200, seed=42)),
        ("One-Class SVM (RBF)", OneClassSVMDetector(nu=0.05, kernel="rbf")),
    ]

    comp_rows: List[Dict[str, Any]] = []
    fault_rows: List[Dict[str, Any]] = []
    zero_rows: List[Dict[str, Any]] = []

    for name, det in specs:
        det.fit(dataset.train_data)
        cal = det.calibrate_threshold(dataset.fit_val_data, fv_labels)
        s_test = det.score(dataset.test_data)
        row, frow = _summarize_predictions(
            name, "classical_val_f1", s_test, float(cal["threshold"]), test_labels, test_variations
        )
        row["selected_normal_percentile"] = float(cal["normal_percentile"])
        comp_rows.append(row)
        fault_rows.append(frow)

        s_fv_norm = det.score(dataset.fit_val_data[fv_norm_mask])
        for q_label, q in [("p95", 95.0), ("p97.5", 97.5)]:
            th_z = float(np.percentile(s_fv_norm, q))
            rz, _ = _summarize_predictions(
                name, f"zero_label_{q_label}", s_test, th_z, test_labels, test_variations
            )
            rz["healthy_quantile"] = q_label
            zero_rows.append(rz)

    oracle = SupervisedOracleDetector(seed=42)
    oracle.fit_oracle(dataset.train_data, dataset.fit_val_data, fv_labels)
    s_or = oracle.score(dataset.test_data)
    row_or, frow_or = _summarize_predictions(
        "Supervised GBM Oracle", "supervised_ceiling", s_or, 0.5, test_labels, test_variations
    )
    row_or["selected_normal_percentile"] = float("nan")
    comp_rows.append(row_or)
    fault_rows.append(frow_or)

    _write_csv(out_dir / "baselines_comparison.csv", comp_rows)
    _write_csv(out_dir / "per_fault_recall.csv", fault_rows)
    _write_csv(out_dir / "zero_label_baselines.csv", zero_rows)
    print(f"  Saved baselines_comparison.csv ({len(comp_rows)} detectors)")
    return {"test_variations": test_variations}


def _get_embeddings_batched(
    model: ContrastiveAutoencoder,
    data: np.ndarray,
    key: str = "projection",
    batch_size: int = 128,
) -> np.ndarray:
    chunks = []
    for i in range(0, len(data), batch_size):
        emb = model.get_embeddings(data[i : i + batch_size])
        chunks.append(np.asarray(emb[key], dtype=np.float32))
    out = np.concatenate(chunks, axis=0)
    if key == "projection":
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        out = out / np.maximum(norms, 1e-8)
    return out


def _select_embedding_detector(
    eval_emb_fn,
    ref_emb: np.ndarray,
    fv_data: np.ndarray,
    fv_labels: np.ndarray,
    normalize: bool = True,
) -> Tuple[EmbeddingAnomalyScorer, Dict[str, Any]]:
    val_emb = eval_emb_fn(fv_data)
    normal = fv_labels == 0
    best_scorer: Optional[EmbeddingAnomalyScorer] = None
    best_meta: Optional[Dict[str, Any]] = None
    for method in METHODS:
        ks = K_VALUES if method == "knn" else (5,)
        for k in ks:
            if k > len(ref_emb):
                continue
            sc = EmbeddingAnomalyScorer(ref_emb, method=method, k=k, normalize=normalize)
            scores = sc.score(val_emb)
            for pct in PERCENTILES:
                th = float(np.percentile(scores[normal], pct))
                f1_val = _f1(fv_labels, (scores > th).astype(int))
                cand = {
                    "scoring_method": method,
                    "k_neighbors": int(k),
                    "normal_percentile": float(pct),
                    "threshold": th,
                    "fit_validation_f1": f1_val,
                }
                if best_meta is None or f1_val > best_meta["fit_validation_f1"]:
                    best_meta = cand
                    best_scorer = sc
    assert best_scorer is not None and best_meta is not None
    return best_scorer, best_meta


def _select_reconstruction_detector(
    model: ContrastiveAutoencoder,
    fv_data: np.ndarray,
    fv_labels: np.ndarray,
) -> Dict[str, Any]:
    scores = model.compute_reconstruction_error(fv_data)
    normal = fv_labels == 0
    best: Optional[Dict[str, Any]] = None
    for pct in PERCENTILES:
        th = float(np.percentile(scores[normal], pct))
        f1_val = _f1(fv_labels, (scores > th).astype(int))
        cand = {
            "scoring_method": "reconstruction_mse",
            "k_neighbors": 0,
            "normal_percentile": float(pct),
            "threshold": th,
            "fit_validation_f1": f1_val,
        }
        if best is None or f1_val > best["fit_validation_f1"]:
            best = cand
    assert best is not None
    return best


def _train_or_load_model(
    arch_cfg: Dict[str, Any],
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
    adapt_load: bool = False,
    strict_margin: bool = False,
    fra_jitter_db: float = 0.0,
    fra_jitter_deg: float = 0.0,
) -> Tuple[ContrastiveAutoencoder, Dict[str, Any]]:
    best_ckpt = run_dir / "checkpoints" / "best"
    prov_file = run_dir / "training_provenance.json"
    if (best_ckpt / "autoencoder.weights.h5").exists() and prov_file.exists():
        prov = json.loads(prov_file.read_text())
        model = ContrastiveAutoencoder.load(str(best_ckpt), config=arch_cfg)
        return model, prov

    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    cw = (
        float(arch_cfg["contrastive_weight"])
        if contrastive_weight_override is None
        else float(contrastive_weight_override)
    )
    config = CARLAConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=float(arch_cfg["learning_rate"]),
        reconstruction_weight=float(arch_cfg["reconstruction_weight"]),
        contrastive_weight=cw,
        temperature=float(arch_cfg["temperature"]),
        anomaly_ratio=1.0,
        seed=seed,
        use_physics_anomalies=use_physics,
        anomaly_types=fault_modes,
        resume_from_latest=False,
        scoring_method="knn",
        k_neighbors=5,
        fra_jitter_db=fra_jitter_db,
        fra_jitter_deg=fra_jitter_deg,
        fra_clean_prob=0.25,
        denoising_reconstruction=True,
        adapt_load=adapt_load,
        strict_margin=strict_margin,
        checkpoint_dir=str(run_dir / "checkpoints"),
        log_dir=str(run_dir / "logs"),
        early_stopping=True,
        patience=25,
        save_best_only=True,
    )
    trainer = CARLATrainer(config, experiment_name=f"ds02_{regime}_{seed}")
    trainer.create_model(
        input_shape=dataset.input_shape,
        latent_dim=int(arch_cfg["latent_dim"]),
        projection_dim=int(arch_cfg["projection_dim"]),
        encoder_type=str(arch_cfg["encoder_type"]),
        encoder_filters=arch_cfg["encoder_filters"],
        encoder_units=arch_cfg.get("encoder_units", [256, 128, 64]),
        kernel_size=int(arch_cfg["kernel_size"]),
        num_heads=int(arch_cfg["num_heads"]),
        dropout_rate=float(arch_cfg["dropout_rate"]),
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
        "encoder_type": str(arch_cfg["encoder_type"]),
        "initial_model_weights_sha256": init_sha,
        "final_best_weights_sha256": _model_weights_sha256(trainer.model),
        "training_seconds": elapsed,
        "epochs_trained": len(history["loss"]),
    }
    prov_file.write_text(json.dumps(prov, indent=2) + "\n")
    print(f"  Trained {regime} (seed={seed}) in {elapsed:.1f}s ({len(history['loss'])} epochs)")
    return trainer.model, prov


def _evaluate_deep_model(
    model: ContrastiveAutoencoder,
    dataset: PowerConverterDataset,
    scoring_space: str,
    test_variations: List[Dict[str, float]],
    noisy_test_015: np.ndarray,
    noisy_fv_norm_015: np.ndarray,
    noisy_tr_015: np.ndarray,
    fit_val_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    fv_data = dataset.fit_val_data if fit_val_mask is None else dataset.fit_val_data[fit_val_mask]
    fv_labels = (
        np.asarray(dataset.fit_val_labels, dtype=int)
        if fit_val_mask is None
        else np.asarray(dataset.fit_val_labels, dtype=int)[fit_val_mask]
    )
    fv_norm_mask = fv_labels == 0
    test_labels = np.asarray(dataset.test_labels, dtype=int)

    if scoring_space == "reconstruction_mse":
        det_meta = _select_reconstruction_detector(model, fv_data, fv_labels)
        scores_test = model.compute_reconstruction_error(dataset.test_data)
        fv_norm_scores = model.compute_reconstruction_error(fv_data[fv_norm_mask])
        th_zero_975 = float(np.percentile(fv_norm_scores, 97.5))
        th_zero_95 = float(np.percentile(fv_norm_scores, 95.0))
        scores_zero = scores_test

        s_n015 = model.compute_reconstruction_error(noisy_test_015)
        th_n015_m = float(np.percentile(model.compute_reconstruction_error(noisy_fv_norm_015), 97.5))
        fra015_matched_f1 = _f1(test_labels, (s_n015 > th_n015_m).astype(int))
    else:
        key = "latent" if scoring_space == "latent" else "projection"
        norm = key == "projection"
        ref_emb = _get_embeddings_batched(model, dataset.train_data, key=key)
        sc_sel, det_meta = _select_embedding_detector(
            lambda x: _get_embeddings_batched(model, x, key=key),
            ref_emb,
            fv_data,
            fv_labels,
            normalize=norm,
        )
        te_emb = _get_embeddings_batched(model, dataset.test_data, key=key)
        scores_test = sc_sel.score(te_emb)

        # Strict Zero-Anomaly-Label protocol (fixed knn k=5 + p97.5 / p95 on healthy validation)
        sc_knn5 = EmbeddingAnomalyScorer(ref_emb, method="knn", k=5, normalize=norm)
        fv_norm_emb = _get_embeddings_batched(model, fv_data[fv_norm_mask], key=key)
        s_fv_knn5 = sc_knn5.score(fv_norm_emb)
        th_zero_975 = float(np.percentile(s_fv_knn5, 97.5))
        th_zero_95 = float(np.percentile(s_fv_knn5, 95.0))
        scores_zero = sc_knn5.score(te_emb)

        # Instrument-matched FRA 0.15 dB zero-label F1
        tr_n015_emb = _get_embeddings_batched(model, noisy_tr_015, key=key)
        fvn_n015_emb = _get_embeddings_batched(model, noisy_fv_norm_015, key=key)
        te_n015_emb = _get_embeddings_batched(model, noisy_test_015, key=key)
        sc_n015 = EmbeddingAnomalyScorer(tr_n015_emb, method="knn", k=5, normalize=norm)
        th_n015_m = float(np.percentile(sc_n015.score(fvn_n015_emb), 97.5))
        fra015_matched_f1 = _f1(test_labels, (sc_n015.score(te_n015_emb) > th_n015_m).astype(int))

    th_val = float(det_meta["threshold"])
    row, _ = _summarize_predictions(
        scoring_space, scoring_space, scores_test, th_val, test_labels, test_variations
    )
    row_z975, _ = _summarize_predictions(
        scoring_space, "zero_label_knn5_p97.5", scores_zero, th_zero_975, test_labels, test_variations
    )
    row_z95, _ = _summarize_predictions(
        scoring_space, "zero_label_knn5_p95", scores_zero, th_zero_95, test_labels, test_variations
    )
    row.update(
        {
            "scoring_method": det_meta["scoring_method"],
            "k_neighbors": det_meta["k_neighbors"],
            "normal_percentile": det_meta["normal_percentile"],
            "fit_validation_f1": det_meta["fit_validation_f1"],
            "zero_knn5_p975_f1": float(row_z975["f1"]),
            "zero_knn5_p975_precision": float(row_z975["precision"]),
            "zero_knn5_p975_recall": float(row_z975["recall"]),
            "zero_knn5_p95_f1": float(row_z95["f1"]),
            "zero_knn5_auc_roc": float(row_z975["auc_roc"]),
            "zero_knn5_auc_pr_pi05": float(row_z975["auc_pr_pi05"]),
            "fra015_matched_p975_f1": float(fra015_matched_f1),
            "scores": scores_test,
            "predictions": (scores_test > th_val).astype(int),
        }
    )
    return row


def run_ablation_encoders_and_lofo_dataset02(
    dataset: PowerConverterDataset,
    frozen_exp_dir: Path,
    study_exp_dir: Path,
    out_dir: Path,
    seeds: List[int],
    epochs: int,
    batch_size: int,
) -> None:
    print(f"\n=== [Block 3] {len(seeds)}-Seed Paired Ablation on dataset_02 ===")
    conv_cfg = json.loads((frozen_exp_dir / "carla_conv1d" / "model_config.json").read_text())
    splits = dataset._splits
    test_variations = [dataset._metadata[int(i)].variations for i in splits["test_idx"]]
    fv_norm_mask = np.asarray(dataset.fit_val_labels, dtype=int) == 0

    raw_tr = dataset._raw_data[splits["train_idx"]]
    raw_fvn = dataset._raw_data[splits["fit_val_idx"]][fv_norm_mask]
    raw_te = dataset._raw_data[splits["test_idx"]]
    noisy_tr_015 = dataset.preprocessor.transform(
        inject_fra_measurement_noise(raw_tr, 0.15, 0.75, seed=301)
    )
    noisy_fv_norm_015 = dataset.preprocessor.transform(
        inject_fra_measurement_noise(raw_fvn, 0.15, 0.75, seed=302)
    )
    noisy_test_015 = dataset.preprocessor.transform(
        inject_fra_measurement_noise(raw_te, 0.15, 0.75, seed=303)
    )

    ssa_spec = BuckConverter(include_parasitic_drop=True, rds_low=0.0)
    ablation_rows: List[Dict[str, Any]] = []
    cached_load_adapt_seed0: Optional[ContrastiveAutoencoder] = None

    for seed in seeds:
        # 1. Pure AE (MSE & Latent)
        rec_model, rec_prov = _train_or_load_model(
            arch_cfg=conv_cfg,
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
            ev = _evaluate_deep_model(
                rec_model, dataset, space, test_variations,
                noisy_test_015, noisy_fv_norm_015, noisy_tr_015,
            )
            ev.pop("scores")
            ev.pop("predictions")
            ablation_rows.append(
                {
                    "regime": sub_name,
                    "seed": seed,
                    "epochs_trained": rec_prov["epochs_trained"],
                    **ev,
                }
            )

        # 2. Heuristic CARLA
        heur_model, heur_prov = _train_or_load_model(
            arch_cfg=conv_cfg,
            regime="heuristic",
            seed=seed,
            run_dir=study_exp_dir / "ablation" / "heuristic" / f"seed_{seed}",
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            use_physics=False,
        )
        ev_h = _evaluate_deep_model(
            heur_model, dataset, "projection", test_variations,
            noisy_test_015, noisy_fv_norm_015, noisy_tr_015,
        )
        ev_h.pop("scores")
        ev_h.pop("predictions")
        ablation_rows.append(
            {"regime": "heuristic", "seed": seed, "epochs_trained": heur_prov["epochs_trained"], **ev_h}
        )

        # 3. Physics SSA (Fixed Nominal Load Ratio: adapt_load=False, strict_margin=False)
        fix_model, fix_prov = _train_or_load_model(
            arch_cfg=conv_cfg,
            regime="physics_ssa_fixed_load",
            seed=seed,
            run_dir=study_exp_dir / "ablation" / "physics_ssa_fixed_load" / f"seed_{seed}",
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            converter_spec=ssa_spec,
            use_physics=True,
            adapt_load=False,
            strict_margin=False,
        )
        ev_f = _evaluate_deep_model(
            fix_model, dataset, "projection", test_variations,
            noisy_test_015, noisy_fv_norm_015, noisy_tr_015,
        )
        ev_f.pop("scores")
        ev_f.pop("predictions")
        ablation_rows.append(
            {
                "regime": "physics_ssa_fixed_load",
                "seed": seed,
                "epochs_trained": fix_prov["epochs_trained"],
                **ev_f,
            }
        )

        # 4. Physics SSA Load-Adaptive + Strict Margin (adapt_load=True, strict_margin=True)
        la_model, la_prov = _train_or_load_model(
            arch_cfg=conv_cfg,
            regime="physics_ssa_load_adapt",
            seed=seed,
            run_dir=study_exp_dir / "ablation" / "physics_ssa_load_adapt" / f"seed_{seed}",
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            converter_spec=ssa_spec,
            use_physics=True,
            adapt_load=True,
            strict_margin=True,
        )
        if seed == seeds[0]:
            cached_load_adapt_seed0 = la_model
        ev_la = _evaluate_deep_model(
            la_model, dataset, "projection", test_variations,
            noisy_test_015, noisy_fv_norm_015, noisy_tr_015,
        )
        ev_la.pop("scores")
        ev_la.pop("predictions")
        ablation_rows.append(
            {
                "regime": "physics_ssa_load_adapt",
                "seed": seed,
                "epochs_trained": la_prov["epochs_trained"],
                **ev_la,
            }
        )

        # 5. Physics SSA Load-Adaptive + FRA Denoising
        fra_model, fra_prov = _train_or_load_model(
            arch_cfg=conv_cfg,
            regime="physics_ssa_load_adapt_fra",
            seed=seed,
            run_dir=study_exp_dir / "ablation" / "physics_ssa_load_adapt_fra" / f"seed_{seed}",
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            converter_spec=ssa_spec,
            use_physics=True,
            adapt_load=True,
            strict_margin=True,
            fra_jitter_db=0.25,
            fra_jitter_deg=1.25,
        )
        ev_fra = _evaluate_deep_model(
            fra_model, dataset, "projection", test_variations,
            noisy_test_015, noisy_fv_norm_015, noisy_tr_015,
        )
        ev_fra.pop("scores")
        ev_fra.pop("predictions")
        ablation_rows.append(
            {
                "regime": "physics_ssa_load_adapt_fra",
                "seed": seed,
                "epochs_trained": fra_prov["epochs_trained"],
                **ev_fra,
            }
        )

        _write_csv(out_dir / "ablation_multiseed.csv", ablation_rows)

    regimes = [
        "recon_only_mse",
        "recon_only_latent",
        "heuristic",
        "physics_ssa_fixed_load",
        "physics_ssa_load_adapt",
        "physics_ssa_load_adapt_fra",
    ]
    key_metrics = [
        "f1",
        "precision",
        "recall",
        "auc_roc",
        "auc_pr_pi10",
        "auc_pr_pi05",
        "zero_knn5_p975_f1",
        "zero_knn5_p95_f1",
        "zero_knn5_auc_roc",
        "zero_knn5_auc_pr_pi05",
        "fra015_matched_p975_f1",
        "recall_passive_single",
        "recall_cout_single",
        "recall_esrc_single",
        "recall_lout_single",
        "recall_esrl_single",
        "recall_rds1_single",
    ]
    by_rs = {(r["regime"], r["seed"]): r for r in ablation_rows}
    summary: Dict[str, Any] = {"seeds": seeds, "regimes": {}, "paired_comparisons": {}}
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
        ("physics_ssa_load_adapt", "physics_ssa_fixed_load"),
        ("physics_ssa_load_adapt", "heuristic"),
        ("physics_ssa_load_adapt", "recon_only_mse"),
        ("physics_ssa_load_adapt", "recon_only_latent"),
        ("physics_ssa_load_adapt_fra", "physics_ssa_load_adapt"),
    ]
    for reg_a, reg_b in pairs:
        summary["paired_comparisons"][f"{reg_a}_vs_{reg_b}"] = {
            m: paired_sign_flip_test(
                [by_rs[(reg_a, s)][m] for s in seeds],
                [by_rs[(reg_b, s)][m] for s in seeds],
            )
            for m in ("f1", "zero_knn5_p975_f1", "auc_roc", "auc_pr_pi05", "fra015_matched_p975_f1")
        }
    (out_dir / "ablation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # -----------------------------------------------------------------------
    # Block 4: 5-Architecture Backbone Study across seeds
    # -----------------------------------------------------------------------
    print(f"\n=== [Block 4] 5-Architecture Backbone Comparison on dataset_02 ===")
    enc_rows: List[Dict[str, Any]] = []
    for subdir, arch_label in [
        ("carla_conv1d", "Conv1D"),
        ("carla_mlp", "MLP"),
        ("carla_transformer", "Transformer"),
        ("carla_lstm", "LSTM"),
        ("carla_gru", "GRU"),
    ]:
        acfg = json.loads((frozen_exp_dir / subdir / "model_config.json").read_text())
        for seed in seeds:
            if subdir == "carla_conv1d":
                rdir = study_exp_dir / "ablation" / "physics_ssa_load_adapt" / f"seed_{seed}"
            else:
                rdir = study_exp_dir / "encoders" / subdir / f"seed_{seed}"
            mdl, prov = _train_or_load_model(
                arch_cfg=acfg,
                regime=f"enc_{subdir}",
                seed=seed,
                run_dir=rdir,
                dataset=dataset,
                epochs=epochs,
                batch_size=batch_size,
                converter_spec=ssa_spec,
                use_physics=True,
                adapt_load=True,
                strict_margin=True,
            )
            ev_e = _evaluate_deep_model(
                mdl, dataset, "projection", test_variations,
                noisy_test_015, noisy_fv_norm_015, noisy_tr_015,
            )
            ev_e.pop("scores")
            ev_e.pop("predictions")
            enc_rows.append(
                {
                    "architecture": arch_label,
                    "seed": seed,
                    "epochs_trained": prov["epochs_trained"],
                    "training_seconds": prov["training_seconds"],
                    **ev_e,
                }
            )
            _write_csv(out_dir / "encoder_comparison.csv", enc_rows)

    # -----------------------------------------------------------------------
    # Block 5: 5-Mode Strict Leave-One-Fault-Out (LOFO)
    # -----------------------------------------------------------------------
    print(f"\n=== [Block 5] 5-Mode Strict Leave-One-Fault-Out on dataset_02 (seed={seeds[0]}) ===")
    lofo_seed = seeds[0]
    all_modes = available_fault_modes()
    fv_variations = [dataset._metadata[int(i)].variations for i in splits["fit_val_idx"]]

    assert cached_load_adapt_seed0 is not None
    indist_ev = _evaluate_deep_model(
        cached_load_adapt_seed0, dataset, "projection", test_variations,
        noisy_test_015, noisy_fv_norm_015, noisy_tr_015,
    )
    indist_pfr = compute_per_fault_recall(indist_ev["predictions"], test_variations)

    lofo_rows: List[Dict[str, Any]] = []
    for held_mode in all_modes:
        held_comp = FAULT_MODE_TO_DATASET_COMP[held_mode]
        kept_modes = [m for m in all_modes if m != held_mode]
        fv_keep_mask = np.array(
            [
                (int(lbl) == 0) or (held_comp not in _active_fault_components(v))
                for lbl, v in zip(dataset.fit_val_labels, fv_variations)
            ],
            dtype=bool,
        )

        lofo_mdl, lofo_prov = _train_or_load_model(
            arch_cfg=conv_cfg,
            regime=f"lofo_no_{held_mode}",
            seed=lofo_seed,
            run_dir=study_exp_dir / "lofo" / f"no_{held_mode}_seed_{lofo_seed}",
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            converter_spec=ssa_spec,
            fault_modes=kept_modes,
            use_physics=True,
            adapt_load=True,
            strict_margin=True,
        )
        ev_lofo = _evaluate_deep_model(
            lofo_mdl, dataset, "projection", test_variations,
            noisy_test_015, noisy_fv_norm_015, noisy_tr_015,
            fit_val_mask=fv_keep_mask,
        )
        pfr_lofo = compute_per_fault_recall(ev_lofo["predictions"], test_variations)

        single_held_mask = np.array(
            [_active_fault_components(v) == [held_comp] for v in test_variations], dtype=bool
        )
        any_held_mask = np.array(
            [held_comp in _active_fault_components(v) for v in test_variations], dtype=bool
        )
        healthy_test_mask = np.asarray(dataset.test_labels, dtype=int) == 0
        y_int = np.asarray(dataset.test_labels, dtype=int)

        sub_single = healthy_test_mask | single_held_mask
        held_single_auc_lofo = (
            float(roc_auc_score(y_int[sub_single], ev_lofo["scores"][sub_single]))
            if single_held_mask.any()
            else float("nan")
        )
        held_single_auc_indist = (
            float(roc_auc_score(y_int[sub_single], indist_ev["scores"][sub_single]))
            if single_held_mask.any()
            else float("nan")
        )

        sub_any = healthy_test_mask | any_held_mask
        held_any_auc_lofo = float(roc_auc_score(y_int[sub_any], ev_lofo["scores"][sub_any]))
        held_any_auc_indist = float(roc_auc_score(y_int[sub_any], indist_ev["scores"][sub_any]))

        lofo_rows.append(
            {
                "held_out_fault_mode": held_mode,
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
                "lofo_zero_knn5_p975_f1": float(ev_lofo["zero_knn5_p975_f1"]),
                "epochs_trained": int(lofo_prov["epochs_trained"]),
            }
        )
        _write_csv(out_dir / "lofo_results.csv", lofo_rows)

    print("  Completed 5-mode strict LOFO on dataset_02.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/buck/buck_data_v2/dataset_02")
    parser.add_argument("--frozen-experiment", default="experiments/dataset_01_run_06_corrected")
    parser.add_argument("--study-experiment", default="experiments/dataset_02_mlwa_study")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--out-dir", default="papers/results/mlwa_study_dataset02")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[200042, 200043, 200044, 200045, 200046]
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    data_dir = ROOT / args.data_dir
    frozen_exp = ROOT / args.frozen_experiment
    study_exp = ROOT / args.study_experiment
    cache_dir = ROOT / args.cache_dir
    out_dir = ROOT / args.out_dir

    dataset = prepare_or_load_stratified_dataset02(data_dir, study_exp, cache_dir)
    run_dataset02_physics_audit(dataset, data_dir, out_dir)
    run_classical_baselines_dataset02(dataset, out_dir)
    run_ablation_encoders_and_lofo_dataset02(
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
