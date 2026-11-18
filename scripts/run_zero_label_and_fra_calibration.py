#!/usr/bin/env python3
"""Zero-Anomaly-Label Calibration & Instrument-Matched FRA Sweep on dataset_01.

Evaluates all 30 pre-trained 5-seed checkpoints (recon_only, heuristic,
physics_legacy, physics_exact_ssa, physics_exact_ssa_fra_denoising) and
classical baselines under:
1. Strict Zero-Anomaly-Label Protocol (fixed scorer `knn k=5` or `ledoit_wolf`,
   threshold = healthy validation `p95` or `p97.5`; zero anomaly labels used
   in training, scorer selection, or thresholding).
2. FRA Instrument-Matched vs Clean-Commissioned Sweep across 0.00..0.60 dB
   removing the `1e-6` ridge null-space artifact via `knn (k=5)` and
   spectral-floored `ledoit_wolf`.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src  # noqa: F401
from src.data.dataset import PowerConverterDataset
from src.data.preprocessor import DataPreprocessor
from src.evaluation.baselines import (
    IsolationForestDetector,
    MahalanobisDetector,
    OneClassSVMDetector,
    PCAReconstructionDetector,
    RawKNNDetector,
)
from src.evaluation.fault_analysis import (
    DEFAULT_ANOMALY_PCT_BANDS,
    compute_per_fault_recall,
    compute_prevalence_adjusted_prauc,
    inject_fra_measurement_noise,
)
from src.evaluation.metrics import compute_classification_metrics
from src.inference.scoring import EmbeddingAnomalyScorer
from src.models.contrastive_ae import ContrastiveAutoencoder

SEEDS = (200042, 200043, 200044, 200045, 200046)


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


def _active_fault_components(variations: Dict[str, float], tol: float = 1e-3) -> List[str]:
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


def _get_embeddings_batched(
    model: ContrastiveAutoencoder,
    data: np.ndarray,
    key: str = "projection",
    batch_size: int = 256,
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


def _resolve_checkpoint_dir(
    regime_dir_name: str,
    seed: int,
    frozen_exp: Path,
    study_exp: Path,
) -> Path:
    if regime_dir_name in ("heuristic", "physics_legacy"):
        ro_sub = "heuristic" if regime_dir_name == "heuristic" else "physics"
        ro_cand = frozen_exp / "ablations" / "negative_sampler" / ro_sub / f"seed_{seed}" / "checkpoints" / "best"
        if (ro_cand / "autoencoder.weights.h5").exists():
            return ro_cand
    cand = study_exp / "ablation" / regime_dir_name / f"seed_{seed}" / "checkpoints" / "best"
    if not (cand / "autoencoder.weights.h5").exists():
        raise FileNotFoundError(f"Missing checkpoint for {regime_dir_name} seed={seed}: {cand}")
    return cand


def _eval_metrics(
    scores_test: np.ndarray,
    th: float,
    test_labels: np.ndarray,
    active5_mask: np.ndarray,
    passive_single_mask: np.ndarray,
    test_variations: List[Dict[str, float]],
) -> Dict[str, float]:
    preds = (scores_test > th).astype(int)
    cm = compute_classification_metrics(predictions=preds, labels=test_labels, scores=scores_test)
    cm_a5 = compute_classification_metrics(
        predictions=preds[active5_mask],
        labels=test_labels[active5_mask],
        scores=scores_test[active5_mask],
    )
    pfr = compute_per_fault_recall(preds, test_variations)
    return {
        "threshold": float(th),
        "precision": float(cm.precision),
        "recall": float(cm.recall),
        "f1": float(cm.f1),
        "auc_roc": float(cm.auc_roc),
        "auc_pr_pi10": float(compute_prevalence_adjusted_prauc(test_labels, scores_test, 0.10)),
        "auc_pr_pi05": float(compute_prevalence_adjusted_prauc(test_labels, scores_test, 0.05)),
        "active5_precision": float(cm_a5.precision),
        "active5_recall": float(cm_a5.recall),
        "active5_f1": float(cm_a5.f1),
        "active5_auc_roc": float(cm_a5.auc_roc),
        "recall_passive_single": float(preds[passive_single_mask].mean()),
        "recall_rds1_single": float(pfr["Rds_1"]["recall_single"]),
    }


def main() -> None:
    frozen_exp = ROOT / "experiments" / "dataset_01_run_06_corrected"
    study_exp = ROOT / "experiments" / "dataset_01_mlwa_study"
    data_dir = ROOT / "data" / "buck" / "buck_data" / "dataset_01"
    cache_dir = ROOT / "cache"
    out_dir = ROOT / "papers" / "results" / "mlwa_study"
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = json.loads((frozen_exp / "carla_conv1d" / "model_config.json").read_text())
    dataset = _load_frozen_dataset(data_dir, frozen_exp / "carla_conv1d", cache_dir)
    splits = dataset._splits

    test_variations = [dataset._metadata[int(i)].variations for i in splits["test_idx"]]
    active5_mask = np.array(
        [_active_fault_components(v) != ["Rds_2"] for v in test_variations],
        dtype=bool,
    )
    passive_single_mask = np.array(
        [
            _active_fault_components(v) in (["Cout"], ["Esr_C"], ["Lout"], ["Esr_L"])
            for v in test_variations
        ],
        dtype=bool,
    )
    test_labels = np.asarray(dataset.test_labels, dtype=int)
    fv_labels = np.asarray(dataset.fit_val_labels, dtype=int)
    fv_norm_mask = fv_labels == 0
    fv_norm_data = dataset.fit_val_data[fv_norm_mask]

    print("=== [Block A] Strict Zero-Anomaly-Label Protocol ===")
    zero_rows: List[Dict[str, Any]] = []

    classical_specs = [
        ("Raw k-NN (k=1)", RawKNNDetector(n_neighbors=1)),
        ("Raw k-NN (k=5)", RawKNNDetector(n_neighbors=5)),
        ("Mahalanobis (Ledoit-Wolf)", MahalanobisDetector(ridge=None)),
        ("PCA Reconstruction (d=10)", PCAReconstructionDetector(n_components=10, seed=42)),
        ("Isolation Forest (200)", IsolationForestDetector(n_estimators=200, seed=42)),
        ("One-Class SVM (RBF)", OneClassSVMDetector(nu=0.05, kernel="rbf")),
    ]
    fitted_classical: Dict[str, Any] = {}
    for name, det in classical_specs:
        det.fit(dataset.train_data)
        fitted_classical[name] = det
        scores_fv_norm = det.score(fv_norm_data)
        scores_test = det.score(dataset.test_data)
        for q_label, q in [("p95", 95.0), ("p97.5", 97.5)]:
            th = float(np.percentile(scores_fv_norm, q))
            m = _eval_metrics(
                scores_test, th, test_labels, active5_mask, passive_single_mask, test_variations
            )
            zero_rows.append(
                {
                    "detector": name,
                    "family": "classical",
                    "seed": 0,
                    "fixed_scorer": "native",
                    "healthy_quantile": q_label,
                    **m,
                }
            )

    # Load all deep checkpoints once so we can reuse SSA & SSA-FRA in Block B
    deep_models: Dict[Tuple[str, int], ContrastiveAutoencoder] = {}
    regime_dirs = [
        "recon_only",
        "heuristic",
        "physics_legacy",
        "physics_exact_ssa",
        "physics_exact_ssa_fra_denoising",
    ]
    for rdir in regime_dirs:
        for seed in SEEDS:
            ckpt = _resolve_checkpoint_dir(rdir, seed, frozen_exp, study_exp)
            deep_models[(rdir, seed)] = ContrastiveAutoencoder.load(str(ckpt), config=selected)
        print(f"  Loaded 5 seeds for {rdir}")

    for seed in SEEDS:
        # 1. recon_only_mse
        m_rec = deep_models[("recon_only", seed)]
        s_fv_mse = m_rec.compute_reconstruction_error(fv_norm_data)
        s_te_mse = m_rec.compute_reconstruction_error(dataset.test_data)
        for q_label, q in [("p95", 95.0), ("p97.5", 97.5)]:
            th = float(np.percentile(s_fv_mse, q))
            m = _eval_metrics(
                s_te_mse, th, test_labels, active5_mask, passive_single_mask, test_variations
            )
            zero_rows.append(
                {
                    "detector": "recon_only_mse",
                    "family": "deep_ae",
                    "seed": seed,
                    "fixed_scorer": "reconstruction_mse",
                    "healthy_quantile": q_label,
                    **m,
                }
            )

        # 2. Embedding-based deep regimes under fixed knn(k=5) and ledoit_wolf
        emb_specs = [
            ("recon_only_latent", "recon_only", "latent", False),
            ("heuristic", "heuristic", "projection", True),
            ("physics_legacy", "physics_legacy", "projection", True),
            ("physics_exact_ssa", "physics_exact_ssa", "projection", True),
            ("physics_exact_ssa_fra_denoising", "physics_exact_ssa_fra_denoising", "projection", True),
        ]
        for det_name, rdir, key, norm in emb_specs:
            mdl = deep_models[(rdir, seed)]
            ref_emb = _get_embeddings_batched(mdl, dataset.train_data, key=key)
            fv_emb = _get_embeddings_batched(mdl, fv_norm_data, key=key)
            te_emb = _get_embeddings_batched(mdl, dataset.test_data, key=key)
            for scorer_name in ("knn", "ledoit_wolf"):
                scorer = EmbeddingAnomalyScorer(
                    ref_emb, method=scorer_name, k=5, normalize=norm
                )
                s_fv = scorer.score(fv_emb)
                s_te = scorer.score(te_emb)
                for q_label, q in [("p95", 95.0), ("p97.5", 97.5)]:
                    th = float(np.percentile(s_fv, q))
                    m = _eval_metrics(
                        s_te, th, test_labels, active5_mask, passive_single_mask, test_variations
                    )
                    zero_rows.append(
                        {
                            "detector": det_name,
                            "family": "deep_contrastive" if key == "projection" else "deep_ae",
                            "seed": seed,
                            "fixed_scorer": f"{scorer_name}_k5" if scorer_name == "knn" else scorer_name,
                            "healthy_quantile": q_label,
                            **m,
                        }
                    )

    _write_csv(out_dir / "zero_label_protocol.csv", zero_rows)

    # Summarize zero-label protocol (mean +/- std across seeds)
    zero_summary: Dict[str, Any] = {}
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for r in zero_rows:
        gkey = (r["detector"], r["fixed_scorer"], r["healthy_quantile"])
        grouped.setdefault(gkey, []).append(r)

    for (det_name, sc_name, q_label), rlist in grouped.items():
        skey = f"{det_name}__{sc_name}__{q_label}"
        zero_summary[skey] = {
            "n_runs": len(rlist),
            **{
                metric: {
                    "mean": float(np.mean([x[metric] for x in rlist])),
                    "std": float(np.std([x[metric] for x in rlist], ddof=1))
                    if len(rlist) > 1
                    else 0.0,
                }
                for metric in (
                    "f1",
                    "active5_f1",
                    "precision",
                    "recall",
                    "auc_roc",
                    "active5_auc_roc",
                    "auc_pr_pi05",
                    "recall_rds1_single",
                    "recall_passive_single",
                )
            },
        }
    (out_dir / "zero_label_summary.json").write_text(json.dumps(zero_summary, indent=2) + "\n")
    print(f"  Saved zero_label_protocol.csv ({len(zero_rows)} rows) & zero_label_summary.json")

    # -----------------------------------------------------------------------
    # Block B: FRA Instrument-Matched vs Clean-Commissioned Sweep
    # -----------------------------------------------------------------------
    print("\n=== [Block B] FRA Instrument-Matched vs Clean-Commissioned Sweep ===")
    raw_train = dataset._raw_data[splits["train_idx"]]
    raw_fv_norm = dataset._raw_data[splits["fit_val_idx"]][fv_norm_mask]
    raw_test = dataset._raw_data[splits["test_idx"]]

    noise_levels = [
        (0.00, 0.00),
        (0.05, 0.25),
        (0.10, 0.50),
        (0.15, 0.75),
        (0.30, 1.50),
        (0.60, 3.00),
    ]
    mc_draws = (200, 201, 202)

    # Precompute clean reference embeddings + clean p97.5 thresholds
    clean_knn5 = RawKNNDetector(n_neighbors=5).fit(dataset.train_data)
    clean_knn5_th975 = float(np.percentile(clean_knn5.score(fv_norm_data), 97.5))

    clean_maha = MahalanobisDetector(ridge=None).fit(dataset.train_data)
    clean_maha_th975 = float(np.percentile(clean_maha.score(fv_norm_data), 97.5))

    clean_deep_scorers: Dict[Tuple[str, int, str], Tuple[EmbeddingAnomalyScorer, float]] = {}
    for rdir in ("physics_exact_ssa", "physics_exact_ssa_fra_denoising"):
        for seed in SEEDS:
            mdl = deep_models[(rdir, seed)]
            ref_c = _get_embeddings_batched(mdl, dataset.train_data, key="projection")
            fv_c = _get_embeddings_batched(mdl, fv_norm_data, key="projection")
            for sc_name in ("knn", "ledoit_wolf"):
                sc = EmbeddingAnomalyScorer(ref_c, method=sc_name, k=5, normalize=True)
                th975 = float(np.percentile(sc.score(fv_c), 97.5))
                clean_deep_scorers[(rdir, seed, sc_name)] = (sc, th975)

    fra_rows: List[Dict[str, Any]] = []
    for sigma_db, sigma_deg in noise_levels:
        draws = [0] if sigma_db == 0.0 else list(mc_draws)
        accum: Dict[Tuple[str, str, str], List[Dict[str, float]]] = {}

        for dseed in draws:
            if sigma_db == 0.0:
                tr_proc = dataset.train_data
                fvn_proc = fv_norm_data
                te_proc = dataset.test_data
            else:
                tr_proc = dataset.preprocessor.transform(
                    inject_fra_measurement_noise(raw_train, sigma_db, sigma_deg, seed=dseed + 10)
                )
                fvn_proc = dataset.preprocessor.transform(
                    inject_fra_measurement_noise(raw_fv_norm, sigma_db, sigma_deg, seed=dseed + 20)
                )
                te_proc = dataset.preprocessor.transform(
                    inject_fra_measurement_noise(raw_test, sigma_db, sigma_deg, seed=dseed + 30)
                )

            # 1. Classical Raw k-NN (k=5)
            s_knn_clean = clean_knn5.score(te_proc)
            accum.setdefault(("Raw k-NN (k=5)", "native", "clean_commissioned"), []).append(
                {
                    "f1": _f1(test_labels, (s_knn_clean > clean_knn5_th975).astype(int)),
                    "active5_f1": _f1(
                        test_labels[active5_mask],
                        (s_knn_clean[active5_mask] > clean_knn5_th975).astype(int),
                    ),
                    "auc_roc": float(roc_auc_score(test_labels, s_knn_clean)),
                    "active5_auc_roc": float(
                        roc_auc_score(test_labels[active5_mask], s_knn_clean[active5_mask])
                    ),
                    "auc_pr_pi05": float(
                        compute_prevalence_adjusted_prauc(test_labels, s_knn_clean, 0.05)
                    ),
                }
            )

            matched_knn5 = RawKNNDetector(n_neighbors=5).fit(tr_proc)
            th_knn_m = float(np.percentile(matched_knn5.score(fvn_proc), 97.5))
            s_knn_m = matched_knn5.score(te_proc)
            accum.setdefault(("Raw k-NN (k=5)", "native", "instrument_matched"), []).append(
                {
                    "f1": _f1(test_labels, (s_knn_m > th_knn_m).astype(int)),
                    "active5_f1": _f1(
                        test_labels[active5_mask],
                        (s_knn_m[active5_mask] > th_knn_m).astype(int),
                    ),
                    "auc_roc": float(roc_auc_score(test_labels, s_knn_m)),
                    "active5_auc_roc": float(
                        roc_auc_score(test_labels[active5_mask], s_knn_m[active5_mask])
                    ),
                    "auc_pr_pi05": float(
                        compute_prevalence_adjusted_prauc(test_labels, s_knn_m, 0.05)
                    ),
                }
            )

            # 2. Classical Mahalanobis (Ledoit-Wolf)
            s_maha_c = clean_maha.score(te_proc)
            accum.setdefault(("Mahalanobis (Ledoit-Wolf)", "native", "clean_commissioned"), []).append(
                {
                    "f1": _f1(test_labels, (s_maha_c > clean_maha_th975).astype(int)),
                    "active5_f1": _f1(
                        test_labels[active5_mask],
                        (s_maha_c[active5_mask] > clean_maha_th975).astype(int),
                    ),
                    "auc_roc": float(roc_auc_score(test_labels, s_maha_c)),
                    "active5_auc_roc": float(
                        roc_auc_score(test_labels[active5_mask], s_maha_c[active5_mask])
                    ),
                    "auc_pr_pi05": float(
                        compute_prevalence_adjusted_prauc(test_labels, s_maha_c, 0.05)
                    ),
                }
            )
            matched_maha = MahalanobisDetector(ridge=None).fit(tr_proc)
            th_maha_m = float(np.percentile(matched_maha.score(fvn_proc), 97.5))
            s_maha_m = matched_maha.score(te_proc)
            accum.setdefault(("Mahalanobis (Ledoit-Wolf)", "native", "instrument_matched"), []).append(
                {
                    "f1": _f1(test_labels, (s_maha_m > th_maha_m).astype(int)),
                    "active5_f1": _f1(
                        test_labels[active5_mask],
                        (s_maha_m[active5_mask] > th_maha_m).astype(int),
                    ),
                    "auc_roc": float(roc_auc_score(test_labels, s_maha_m)),
                    "active5_auc_roc": float(
                        roc_auc_score(test_labels[active5_mask], s_maha_m[active5_mask])
                    ),
                    "auc_pr_pi05": float(
                        compute_prevalence_adjusted_prauc(test_labels, s_maha_m, 0.05)
                    ),
                }
            )

            # 3. Deep 5-seed CARLA-SSA and CARLA-SSA-FRA
            for rdir, label in [
                ("physics_exact_ssa", "CARLA-SSA"),
                ("physics_exact_ssa_fra_denoising", "CARLA-SSA-FRA"),
            ]:
                for seed in SEEDS:
                    mdl = deep_models[(rdir, seed)]
                    te_emb = _get_embeddings_batched(mdl, te_proc, key="projection")
                    tr_emb = (
                        clean_deep_scorers[(rdir, seed, "knn")][0].reference_embeddings
                        if sigma_db == 0.0
                        else _get_embeddings_batched(mdl, tr_proc, key="projection")
                    )
                    fv_emb = _get_embeddings_batched(mdl, fvn_proc, key="projection")

                    for sc_name in ("knn", "ledoit_wolf"):
                        sc_clean, th_clean = clean_deep_scorers[(rdir, seed, sc_name)]
                        s_c = sc_clean.score(te_emb)
                        accum.setdefault((label, sc_name, "clean_commissioned"), []).append(
                            {
                                "f1": _f1(test_labels, (s_c > th_clean).astype(int)),
                                "active5_f1": _f1(
                                    test_labels[active5_mask],
                                    (s_c[active5_mask] > th_clean).astype(int),
                                ),
                                "auc_roc": float(roc_auc_score(test_labels, s_c)),
                                "active5_auc_roc": float(
                                    roc_auc_score(test_labels[active5_mask], s_c[active5_mask])
                                ),
                                "auc_pr_pi05": float(
                                    compute_prevalence_adjusted_prauc(test_labels, s_c, 0.05)
                                ),
                            }
                        )

                        sc_m = (
                            sc_clean
                            if sigma_db == 0.0
                            else EmbeddingAnomalyScorer(tr_emb, method=sc_name, k=5, normalize=True)
                        )
                        th_m = (
                            th_clean
                            if sigma_db == 0.0
                            else float(np.percentile(sc_m.score(fv_emb), 97.5))
                        )
                        s_m = s_c if sigma_db == 0.0 else sc_m.score(te_emb)
                        accum.setdefault((label, sc_name, "instrument_matched"), []).append(
                            {
                                "f1": _f1(test_labels, (s_m > th_m).astype(int)),
                                "active5_f1": _f1(
                                    test_labels[active5_mask],
                                    (s_m[active5_mask] > th_m).astype(int),
                                ),
                                "auc_roc": float(roc_auc_score(test_labels, s_m)),
                                "active5_auc_roc": float(
                                    roc_auc_score(test_labels[active5_mask], s_m[active5_mask])
                                ),
                                "auc_pr_pi05": float(
                                    compute_prevalence_adjusted_prauc(test_labels, s_m, 0.05)
                                ),
                            }
                        )

        for (det_label, sc_name, comm_mode), mlist in accum.items():
            fra_rows.append(
                {
                    "detector": det_label,
                    "scorer": sc_name,
                    "commissioning_mode": comm_mode,
                    "sigma_mag_db": sigma_db,
                    "sigma_phase_deg": sigma_deg,
                    "n_evals": len(mlist),
                    "f1_mean": float(np.mean([x["f1"] for x in mlist])),
                    "f1_std": float(np.std([x["f1"] for x in mlist])),
                    "active5_f1_mean": float(np.mean([x["active5_f1"] for x in mlist])),
                    "active5_f1_std": float(np.std([x["active5_f1"] for x in mlist])),
                    "auc_roc_mean": float(np.mean([x["auc_roc"] for x in mlist])),
                    "active5_auc_roc_mean": float(np.mean([x["active5_auc_roc"] for x in mlist])),
                    "auc_pr_pi05_mean": float(np.mean([x["auc_pr_pi05"] for x in mlist])),
                }
            )
        print(f"  Completed FRA level {sigma_db:.2f} dB / {sigma_deg:.2f} deg")

    _write_csv(out_dir / "fra_instrument_matched_sweep.csv", fra_rows)
    print(f"  Saved fra_instrument_matched_sweep.csv ({len(fra_rows)} rows)")


if __name__ == "__main__":
    main()
