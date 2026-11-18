"""JSC Companion Inference & INT8 Quantization Study (Read-Only on dataset_01_run_06_corrected).

Diagnoses and resolves two key edge-deployment bottlenecks identified in the JSC study:
1. Reference-bank scoring bottleneck: compares legacy sklearn NearestNeighbors (N=2419, 4.8 MB)
   against Fast BLAS k-NN, Spherical K-Means Prototype Compression (M=128, 64, 32, 16),
   Centroid (M=1), and Ledoit-Wolf shrinkage precision scoring.
2. CPU INT8 decision-flip collapse (49.8%--54.6% discordance in TFLite INT8 / ONNX INT8):
   demonstrates root causes (1e-6 Mahalanobis null-space quantization explosion + FP32-vs-INT8
   bank/threshold mismatch + int8 wraparound) and quantifies recovery under zero-anomaly-label
   INT8-domain bank and threshold recalibration.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocessor import DataPreprocessor
from src.deployment.runners import create_onnx_runner, create_tflite_runner
from src.deployment.utils import load_deployment_datasets
from src.evaluation.threshold import ThresholdSelector
from src.inference.scoring import EmbeddingAnomalyScorer


def _opt_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(
        ThresholdSelector(method="f1")
        .fit(scores[labels == 0], scores[labels == 1])
        .threshold_
    )

EXP_ROOT = ROOT / "experiments" / "dataset_01_run_06_corrected"
DATA_DIR = ROOT / "data" / "buck" / "buck_data"
OUT_DIR = ROOT / "papers" / "results" / "jsc_inference_study"
MODELS = ("conv1d", "mlp", "transformer", "lstm", "gru")


def _batched_infer(runner: Callable[[np.ndarray], np.ndarray], x: np.ndarray, batch_size: int = 1) -> np.ndarray:
    """Run sample-by-sample (batch=1) to support fixed-batch TFLite/ONNX edge graphs."""
    outs = [runner(x[i : i + batch_size].astype(np.float32)) for i in range(0, len(x), batch_size)]
    return np.concatenate(outs, axis=0).astype(np.float32)


def _create_clipped_tflite_runner(tflite_path: str) -> Callable[[np.ndarray], np.ndarray]:
    """TFLite runner with proper [-128, 127] saturation before int8 cast."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    in_idx, out_idx = in_det["index"], out_det["index"]
    is_int8_in = in_det["dtype"] == np.int8
    in_scale, in_zp = in_det.get("quantization", (1.0, 0))
    is_int8_out = out_det["dtype"] == np.int8
    out_scale, out_zp = out_det.get("quantization", (1.0, 0))

    def runner(x: np.ndarray) -> np.ndarray:
        if is_int8_in:
            q = np.round(x.astype(np.float32) / in_scale + in_zp)
            x_prep = np.clip(q, -128, 127).astype(np.int8)
        else:
            x_prep = np.asarray(x, dtype=np.float32)
        interpreter.set_tensor(in_idx, x_prep)
        interpreter.invoke()
        raw = interpreter.get_tensor(out_idx)
        if is_int8_out:
            return (raw.astype(np.float32) - out_zp) * out_scale
        return raw.astype(np.float32)

    return runner


def _time_single_sample(fn: Callable[[np.ndarray], np.ndarray], sample: np.ndarray, warmup: int = 30, runs: int = 500) -> Dict[str, float]:
    for _ in range(warmup):
        fn(sample)
    times = np.empty(runs, dtype=np.float64)
    for i in range(runs):
        t0 = time.perf_counter()
        fn(sample)
        times[i] = (time.perf_counter() - t0) * 1000.0
    return {
        "latency_mean_ms": float(np.mean(times)),
        "latency_std_ms": float(np.std(times, ddof=1)),
        "latency_p50_ms": float(np.median(times)),
    }


def run_scorer_pareto_benchmark(
    ref_fp32: np.ndarray,
    val_fp32: np.ndarray,
    val_labels: np.ndarray,
    test_fp32: np.ndarray,
    test_labels: np.ndarray,
    graph_size_mb: float,
) -> List[Dict[str, object]]:
    """Benchmark accuracy, memory, and single-sample latency across reference-bank scorers."""
    rows: List[Dict[str, object]] = []
    sample = test_fp32[:1].astype(np.float32)
    val_norm_fp32 = val_fp32[val_labels == 0]

    # 0. Legacy sklearn NearestNeighbors (what JSC used prior to BLAS optimization)
    ref_norm = ref_fp32 / (np.linalg.norm(ref_fp32, axis=1, keepdims=True) + 1e-8)
    nn_legacy = NearestNeighbors(n_neighbors=5).fit(ref_norm)

    def legacy_knn_call(z: np.ndarray) -> np.ndarray:
        zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
        d, _ = nn_legacy.kneighbors(zn)
        return np.mean(d, axis=1)

    t_leg = _time_single_sample(legacy_knn_call, sample)
    val_s = legacy_knn_call(val_fp32)
    test_s = legacy_knn_call(test_fp32)
    th_sup = _opt_f1_threshold(val_s, val_labels)
    th_unsup = float(np.percentile(legacy_knn_call(val_norm_fp32), 95.0))
    bank_mb = (ref_fp32.shape[0] * ref_fp32.shape[1] * 4) / (1024 * 1024)
    rows.append(
        {
            "scorer_variant": "legacy_sklearn_knn_N2419",
            "n_bank_vectors": int(ref_fp32.shape[0]),
            "bank_memory_mb": round(bank_mb, 4),
            "total_detector_mb": round(graph_size_mb + bank_mb, 4),
            "scorer_latency_mean_ms": round(t_leg["latency_mean_ms"], 5),
            "scorer_latency_p50_ms": round(t_leg["latency_p50_ms"], 5),
            "auc_roc": round(float(roc_auc_score(test_labels, test_s)), 4),
            "f1_val_opt": round(float(f1_score(test_labels, (test_s > th_sup).astype(int))), 4),
            "f1_unsup_p95": round(float(f1_score(test_labels, (test_s > th_unsup).astype(int))), 4),
        }
    )

    configs = [
        ("fast_blas_knn_N2419", "knn", 5, None),
        ("proto_knn_M128", "knn", 5, 128),
        ("proto_knn_M64", "knn", 5, 64),
        ("proto_knn_M32", "knn", 5, 32),
        ("proto_knn_M16", "knn", 5, 16),
        ("centroid_M1", "centroid", 1, None),
        ("ledoit_wolf_precision", "ledoit_wolf", 5, None),
        ("ridge_mahalanobis_1e6", "mahalanobis", 5, None),
    ]

    for name, method, k, n_proto in configs:
        scorer = EmbeddingAnomalyScorer(
            ref_fp32, method=method, k=k, normalize=True, n_prototypes=n_proto, random_state=42
        )
        timing = _time_single_sample(scorer.score, sample)
        val_s = scorer.score(val_fp32)
        test_s = scorer.score(test_fp32)
        th_sup = _opt_f1_threshold(val_s, val_labels)
        th_unsup = float(np.percentile(scorer.score(val_norm_fp32), 95.0))

        if method == "centroid":
            n_vecs = 1
            mem_bytes = ref_fp32.shape[1] * 4
        elif method in ("ledoit_wolf", "mahalanobis"):
            n_vecs = ref_fp32.shape[1] + 1  # mean + d x d precision matrix
            mem_bytes = (ref_fp32.shape[1] + ref_fp32.shape[1] ** 2) * 4
        else:
            n_vecs = scorer.reference_embeddings.shape[0]
            mem_bytes = scorer.reference_embeddings.nbytes

        b_mb = mem_bytes / (1024 * 1024)
        rows.append(
            {
                "scorer_variant": name,
                "n_bank_vectors": int(n_vecs),
                "bank_memory_mb": round(b_mb, 4),
                "total_detector_mb": round(graph_size_mb + b_mb, 4),
                "scorer_latency_mean_ms": round(timing["latency_mean_ms"], 5),
                "scorer_latency_p50_ms": round(timing["latency_p50_ms"], 5),
                "auc_roc": round(float(roc_auc_score(test_labels, test_s)), 4),
                "f1_val_opt": round(float(f1_score(test_labels, (test_s > th_sup).astype(int))), 4),
                "f1_unsup_p95": round(float(f1_score(test_labels, (test_s > th_unsup).astype(int))), 4),
            }
        )
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    int8_rows: List[Dict[str, object]] = []
    pareto_all: Dict[str, List[Dict[str, object]]] = {}

    for model_key in MODELS:
        model_dir = EXP_ROOT / f"carla_{model_key}"
        deploy_dir = model_dir / "deployment"
        det_cfg = json.loads((deploy_dir / "detector_config.json").read_text())
        orig_method = det_cfg["scoring_method"]
        orig_k = int(det_cfg.get("k_neighbors", 5))
        orig_th = float(det_cfg["threshold"])

        preprocessor = DataPreprocessor.load(str(model_dir / "preprocessor"))
        dataset, _ = load_deployment_datasets(
            data_dir=str(DATA_DIR),
            cache_dir=str(ROOT / "cache"),
            preprocessor=preprocessor,
            split_indices_path=str(model_dir / "split_indices.npz"),
        )
        x_train = dataset.train_data
        x_val = dataset.fit_val_data
        y_val = dataset.fit_val_labels
        x_val_norm = x_val[y_val == 0]
        x_test = dataset.test_data
        y_test = dataset.test_labels

        ref_fp32 = np.load(deploy_dir / "reference_embeddings.npy")

        # Baseline FP32 embeddings via TFLite FP32 (0.00% discordance with Keras FP32)
        tflite_fp32_runner = create_tflite_runner(str(deploy_dir / "model_fp32.tflite"))
        val_fp32 = _batched_infer(tflite_fp32_runner, x_val, batch_size=1)
        test_fp32 = _batched_infer(tflite_fp32_runner, x_test, batch_size=1)

        orig_fp32_scorer = EmbeddingAnomalyScorer(ref_fp32, method=orig_method, k=orig_k, normalize=True)
        fp32_orig_preds = orig_fp32_scorer.score(test_fp32) > orig_th

        # Reference FP32 kNN + unsupervised p95 baseline decisions (fair one-class target)
        knn_fp32_scorer = EmbeddingAnomalyScorer(ref_fp32, method="knn", k=5, normalize=True)
        th_fp32_knn_p95 = float(np.percentile(knn_fp32_scorer.score(val_fp32[y_val == 0]), 95.0))
        fp32_knn_p95_preds = knn_fp32_scorer.score(test_fp32) > th_fp32_knn_p95
        th_fp32_knn_opt = _opt_f1_threshold(knn_fp32_scorer.score(val_fp32), y_val)
        fp32_knn_opt_preds = knn_fp32_scorer.score(test_fp32) > th_fp32_knn_opt

        lw_fp32_scorer = EmbeddingAnomalyScorer(ref_fp32, method="ledoit_wolf", k=5, normalize=True)
        th_fp32_lw_p95 = float(np.percentile(lw_fp32_scorer.score(val_fp32[y_val == 0]), 95.0))
        fp32_lw_p95_preds = lw_fp32_scorer.score(test_fp32) > th_fp32_lw_p95

        if model_key in ("conv1d", "mlp"):
            g_mb = (deploy_dir / "model_fp32.tflite").stat().st_size / (1024 * 1024)
            pareto_all[model_key] = run_scorer_pareto_benchmark(
                ref_fp32, val_fp32, y_val, test_fp32, y_test, g_mb
            )

        # Evaluate TFLite INT8 (clipped) and ONNX INT8
        backends = [
            ("tflite_int8", _create_clipped_tflite_runner(str(deploy_dir / "model_int8.tflite")), deploy_dir / "model_int8.tflite"),
            ("onnx_int8", create_onnx_runner(str(deploy_dir / "model_int8.onnx")), deploy_dir / "model_int8.onnx"),
        ]

        for b_name, runner, model_file in backends:
            if runner is None or not model_file.exists():
                continue
            graph_mb = model_file.stat().st_size / (1024 * 1024)
            ref_q = _batched_infer(runner, x_train, batch_size=1)
            val_q = _batched_infer(runner, x_val, batch_size=1)
            val_norm_q = val_q[y_val == 0]
            test_q = _batched_infer(runner, x_test, batch_size=1)

            # Strategy 1: Naive JSC (FP32 bank + original method + FP32 threshold)
            s_naive = orig_fp32_scorer.score(test_q)
            p_naive = s_naive > orig_th

            # Strategy 2: Zero-Label INT8 Recalibration with k-NN (k=5, full bank)
            knn_q_scorer, th_knn_q_p95 = knn_fp32_scorer.recalibrate(
                ref_q, val_norm_q, percentile=95.0, method="knn", k=5
            )
            s_knn_q = knn_q_scorer.score(test_q)
            p_knn_q_p95 = s_knn_q > th_knn_q_p95
            th_knn_q_opt = _opt_f1_threshold(knn_q_scorer.score(val_q), y_val)
            p_knn_q_opt = s_knn_q > th_knn_q_opt

            # Strategy 3: Zero-Label INT8 Recalibration with k-NN + 32 Spherical Prototypes
            proto_q_scorer, th_proto_q_p95 = knn_fp32_scorer.recalibrate(
                ref_q, val_norm_q, percentile=95.0, method="knn", k=5, n_prototypes=32
            )
            s_proto_q = proto_q_scorer.score(test_q)
            p_proto_q_p95 = s_proto_q > th_proto_q_p95
            th_proto_q_opt = _opt_f1_threshold(proto_q_scorer.score(val_q), y_val)
            p_proto_q_opt = s_proto_q > th_proto_q_opt

            # Strategy 4: Zero-Label INT8 Recalibration with Spectral-Floored Ledoit-Wolf
            lw_q_scorer, th_lw_q_p95 = lw_fp32_scorer.recalibrate(
                ref_q, val_norm_q, percentile=95.0, method="ledoit_wolf"
            )
            s_lw_q = lw_q_scorer.score(test_q)
            p_lw_q_p95 = s_lw_q > th_lw_q_p95

            strat_specs = [
                (
                    "1_naive_fp32_bank_orig_scorer",
                    orig_method,
                    ref_fp32.nbytes / (1024 * 1024),
                    s_naive,
                    p_naive,
                    fp32_orig_preds,
                ),
                (
                    "2_int8_recal_knn5_val_opt",
                    "knn_k5",
                    ref_q.nbytes / (1024 * 1024),
                    s_knn_q,
                    p_knn_q_opt,
                    fp32_knn_opt_preds,
                ),
                (
                    "3_int8_recal_knn5_zero_label_p95",
                    "knn_k5",
                    ref_q.nbytes / (1024 * 1024),
                    s_knn_q,
                    p_knn_q_p95,
                    fp32_knn_p95_preds,
                ),
                (
                    "4_int8_recal_proto32_val_opt",
                    "knn_proto32",
                    proto_q_scorer.reference_embeddings.nbytes / (1024 * 1024),
                    s_proto_q,
                    p_proto_q_opt,
                    fp32_knn_opt_preds,
                ),
                (
                    "5_int8_recal_proto32_zero_label_p95",
                    "knn_proto32",
                    proto_q_scorer.reference_embeddings.nbytes / (1024 * 1024),
                    s_proto_q,
                    p_proto_q_p95,
                    fp32_knn_p95_preds,
                ),
                (
                    "6_int8_recal_ledoit_wolf_zero_label_p95",
                    "ledoit_wolf",
                    (512 + 512 * 512) * 4 / (1024 * 1024),
                    s_lw_q,
                    p_lw_q_p95,
                    fp32_lw_p95_preds,
                ),
            ]

            for s_label, s_meth, b_mb, scores_vec, preds_vec, ref_preds_vec in strat_specs:
                int8_rows.append(
                    {
                        "architecture": model_key,
                        "orig_jsc_scorer": orig_method,
                        "backend": b_name,
                        "strategy": s_label,
                        "scorer": s_meth,
                        "graph_mb": round(graph_mb, 4),
                        "bank_mb": round(b_mb, 4),
                        "total_detector_mb": round(graph_mb + b_mb, 4),
                        "auc_roc": round(float(roc_auc_score(y_test, scores_vec)), 4),
                        "precision": round(float(precision_score(y_test, preds_vec.astype(int), zero_division=0)), 4),
                        "recall": round(float(recall_score(y_test, preds_vec.astype(int), zero_division=0)), 4),
                        "f1": round(float(f1_score(y_test, preds_vec.astype(int), zero_division=0)), 4),
                        "discordance_vs_fp32_pct": round(float(np.mean(preds_vec != ref_preds_vec) * 100.0), 2),
                    }
                )
            print(f"[DONE] {model_key} | {b_name}", flush=True)

    int8_csv = OUT_DIR / "int8_recalibration_rescue.csv"
    with open(int8_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(int8_rows[0].keys()))
        writer.writeheader()
        writer.writerows(int8_rows)

    for mk, p_rows in pareto_all.items():
        p_csv = OUT_DIR / f"scorer_pareto_{mk}.csv"
        with open(p_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(p_rows[0].keys()))
            writer.writeheader()
            writer.writerows(p_rows)

    summary = {"int8_rows": len(int8_rows), "pareto_models": list(pareto_all.keys())}
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved JSC inference & INT8 study to {OUT_DIR}")


if __name__ == "__main__":
    main()
