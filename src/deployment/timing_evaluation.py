"""
Timing and Classification Evaluation Pipeline
==============================================

Evaluates optimized models (TFLite, ONNX, TensorRT) on the test dataset,
computes precision, recall, F1, AUC-ROC, and measures latency.
Saves comparison reports in the deployment output directory.
"""

from pathlib import Path
import gc
import json
import numpy as np
import keras
from typing import Dict, Any, List, Optional, Callable

from .utils import (
    get_file_size_mb,
    TIMING_BATCH_SIZE,
    TIMING_WARMUP,
    TIMING_RUNS,
    CONSISTENCY_SAMPLES,
    CONSISTENCY_ATOL,
    CONSISTENCY_RTOL,
    CONSISTENCY_DISAGREEMENT_WARN,
)
from .runners import (
    create_keras_runner,
    create_tflite_runner,
    create_onnx_runner,
    create_tensorrt_runner,
    run_inference_loop,
    measure_inference_latency,
    compare_runner_outputs,
)
from ..utils.logger import get_logger
from ..evaluation.metrics import compute_classification_metrics
from ..inference.predictor import AnomalyPredictor
from ..inference.scoring import EmbeddingAnomalyScorer
from .utils import build_carla_embedding_model

logger = get_logger(__name__)


def _evaluate_with_runner(
    name: str,
    runner,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    threshold: float,
    scorer: Optional[EmbeddingAnomalyScorer] = None,
    evaluation_batch_size: int = TIMING_BATCH_SIZE,
    timing_warmup: int = TIMING_WARMUP,
    timing_runs: int = TIMING_RUNS,
    reference_scores: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Run full classification evaluation using a generic runner callable.

    Latency is measured under the uniform protocol (single-sample batch n=1,
    15 warm-up, 150 timed runs) so every backend is directly comparable.
    Classification metrics are batch-invariant, so backend outputs are gathered
    over the full test set for correctness.

    Args:
        name: Display name for logging.
        runner: Inference callable (from runner factories).
        test_data: Test dataset array.
        test_labels: Ground truth labels (0 = normal, 1 = anomaly).
        threshold: Anomaly decision threshold.
        scorer: Reusable embedding scorer for CARLA. When omitted, outputs are
            interpreted as autoencoder reconstructions.
        evaluation_batch_size: Batch size for the full detector-output pass. It is
            fixed to the same value for every backend by the public evaluators.

    Returns:
        Dictionary with latency and classification metrics.
    """
    logger.info(f"Evaluating {name}...")

    # 1. Use the same shared loop as the timing-and-memory benchmark.
    x_single = test_data[:TIMING_BATCH_SIZE].astype(np.float32)
    timing = measure_inference_latency(
        runner, x_single, num_warmup=timing_warmup, num_runs=timing_runs
    )

    # 2. Measure the complete detector separately with the host-side reference
    #    search included. GPU model latency above remains CUDA-event timed.
    def detector_runner(x):
        outputs = runner(x)
        scores = (
            scorer.score(outputs)
            if scorer is not None
            else np.mean(np.square(x - outputs), axis=(1, 2))
        )
        return scores[:, np.newaxis]

    detector_runner.output_shape = (1,)
    detector_timing = measure_inference_latency(
        detector_runner,
        x_single,
        num_warmup=timing_warmup,
        num_runs=timing_runs,
    )

    # 3. Compute detector scores over the full held-out set.
    outputs = run_inference_loop(
        runner, test_data, batch_size=evaluation_batch_size
    )
    scores = (
        scorer.score(outputs)
        if scorer is not None
        else np.mean(np.square(test_data - outputs), axis=(1, 2))
    )
    predictions = (scores > threshold).astype(int)

    full_test_disagreement = None
    if reference_scores is not None:
        reference_scores = np.asarray(reference_scores).reshape(-1)
        if reference_scores.shape != scores.shape:
            raise ValueError(
                "Reference-score shape %s does not match candidate-score shape %s"
                % (reference_scores.shape, scores.shape)
            )
        reference_predictions = (reference_scores > threshold).astype(int)
        full_test_disagreement = float(
            np.mean(reference_predictions != predictions)
        )

    # 3. Compute classification metrics
    metrics = compute_classification_metrics(predictions, test_labels, scores=scores)

    result = {
        "latency_per_sample_mean_ms": timing["latency_mean_ms"],
        "latency_per_sample_std_ms": timing["latency_std_ms"],
        "latency_per_sample_min_ms": timing["latency_min_ms"],
        "latency_per_sample_max_ms": timing["latency_max_ms"],
        "timing_source": timing["timing_source"],
        "timing_scope": timing["timing_scope"],
        "end_to_end_latency_mean_ms": timing["end_to_end_latency_mean_ms"],
        "end_to_end_latency_std_ms": timing["end_to_end_latency_std_ms"],
        "end_to_end_latency_min_ms": timing["end_to_end_latency_min_ms"],
        "end_to_end_latency_max_ms": timing["end_to_end_latency_max_ms"],
        "detector_end_to_end_latency_mean_ms": detector_timing["latency_mean_ms"],
        "detector_end_to_end_latency_std_ms": detector_timing["latency_std_ms"],
        "detector_timing_source": detector_timing["timing_source"],
        "detector_timing_scope": detector_timing["timing_scope"],
        "timing_batch_size": timing["timing_batch_size"],
        "timing_warmup_runs": timing["timing_warmup_runs"],
        "timing_runs": timing["timing_runs"],
        "evaluation_batch_size": int(evaluation_batch_size),
        "accuracy": float(metrics.accuracy),
        "precision": float(metrics.precision),
        "recall": float(metrics.recall),
        "f1": float(metrics.f1),
        "auc_roc": float(metrics.auc_roc),
    }
    if full_test_disagreement is not None:
        result["full_test_consistency_samples"] = int(len(scores))
        result["full_test_anomaly_disagreement_rate_vs_baseline"] = (
            full_test_disagreement
        )
    return result


def evaluate_keras_model(
    model: keras.Model,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    threshold: float,
    scorer: Optional[EmbeddingAnomalyScorer] = None,
    timing_warmup: int = TIMING_WARMUP,
    timing_runs: int = TIMING_RUNS,
    reference_scores: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Run full test evaluation on baseline Keras model."""
    runner = create_keras_runner(model)
    return _evaluate_with_runner(
        "Keras FP32 Model", runner, test_data, test_labels, threshold,
        scorer=scorer, evaluation_batch_size=TIMING_BATCH_SIZE,
        timing_warmup=timing_warmup, timing_runs=timing_runs,
        reference_scores=reference_scores,
    )


def evaluate_tflite_model(
    tflite_path: str,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    threshold: float,
    runner: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    scorer: Optional[EmbeddingAnomalyScorer] = None,
    reference_scores: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Run full test evaluation on TFLite quantized models."""
    runner = runner or create_tflite_runner(tflite_path)
    return _evaluate_with_runner(
        f"TFLite ({tflite_path})", runner, test_data, test_labels, threshold, scorer,
        reference_scores=reference_scores,
    )


def evaluate_onnx_model(
    onnx_path: str,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    threshold: float,
    runner: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    scorer: Optional[EmbeddingAnomalyScorer] = None,
    reference_scores: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Run full test evaluation on ONNX model."""
    runner = runner or create_onnx_runner(onnx_path)
    if runner is None:
        return {}
    return _evaluate_with_runner(
        f"ONNX ({onnx_path})", runner, test_data, test_labels, threshold, scorer,
        reference_scores=reference_scores,
    )


def evaluate_tensorrt_model(
    engine_path: str,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    threshold: float,
    runner: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    scorer: Optional[EmbeddingAnomalyScorer] = None,
    reference_scores: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Run full test evaluation on TensorRT engine.

    Args:
        engine_path: Path to a serialized .engine file.
        test_data: Numpy array of test data.
        test_labels: Ground truth labels.
        threshold: Anomaly decision threshold.

    Returns:
        Dictionary with latency and classification metrics,
        or empty dict if TRT is unavailable.
    """
    runner = runner or create_tensorrt_runner(engine_path)
    if runner is None:
        return {}
    return _evaluate_with_runner(
        f"TensorRT ({engine_path})", runner, test_data, test_labels, threshold, scorer,
        reference_scores=reference_scores,
    )


def run_deployment_evaluations(
    model_dir: str,
    deployment_dir: str,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    include_keras: bool = True,
    predictor: Optional[AnomalyPredictor] = None,
) -> List[Dict[str, Any]]:
    """Loads and runs evaluation metrics for all optimized models in the deployment directory."""
    logger.info("Starting unified deployment evaluation benchmark...")
    dep_path = Path(deployment_dir)
    results = []

    # Load threshold
    predictor = predictor or AnomalyPredictor(model_dir=model_dir)
    threshold = predictor.threshold
    if threshold is None:
        raise RuntimeError(
            "Deployment evaluation requires a fit-validation threshold; "
            "run src.evaluate before benchmarking converted backends."
        )

    scorer = None
    if predictor.reference_embeddings is not None:
        scorer = EmbeddingAnomalyScorer(
            predictor.reference_embeddings,
            method=predictor.scoring_method,
            k=predictor.k_neighbors,
        )
        reference_model = build_carla_embedding_model(predictor.model)
    else:
        reference_model = predictor.model.autoencoder
    reference_runner = create_keras_runner(reference_model)
    reference_outputs = run_inference_loop(
        reference_runner, test_data, batch_size=TIMING_BATCH_SIZE
    )
    reference_scores = (
        scorer.score(reference_outputs)
        if scorer is not None
        else np.mean(np.square(test_data - reference_outputs), axis=(1, 2))
    )
    if not np.all(np.isfinite(reference_scores)):
        raise RuntimeError("Keras reference detector produced non-finite test scores")

    def _check_consistency(candidate_runner, label):
        consistency = compare_runner_outputs(
            reference_runner,
            candidate_runner,
            test_data,
            threshold=threshold,
            batch_size=TIMING_BATCH_SIZE,
            max_samples=CONSISTENCY_SAMPLES,
            atol=CONSISTENCY_ATOL,
            rtol=CONSISTENCY_RTOL,
            score_function=scorer.score if scorer is not None else None,
        )
        disagreement = consistency.get("anomaly_disagreement_rate_vs_baseline", 0.0)
        if disagreement > CONSISTENCY_DISAGREEMENT_WARN:
            logger.warning(
                f"{label} differs from the Keras baseline on "
                f"{disagreement:.2%} of preflight anomaly decisions."
            )
        if not consistency["output_within_tolerance"]:
            logger.warning(
                f"{label} output is outside the standard numerical "
                f"tolerance (atol={CONSISTENCY_ATOL}, rtol={CONSISTENCY_RTOL})."
            )
        # Numerical/decision drift is a scientific result, not a reason to
        # suppress the requested evaluation. Shape and finite-value failures
        # still raise above; viable but discordant paths run to completion and
        # are explicitly flagged for fidelity.
        consistency["preflight_passed"] = True
        consistency["fidelity_passed"] = bool(
            disagreement <= CONSISTENCY_DISAGREEMENT_WARN
        )
        return consistency

    # 1. The production pipeline gets the CPU baseline from benchmark_cpu.py,
    # which starts with CUDA disabled before TensorFlow imports. Standalone
    # callers/tests may opt into the in-process baseline.
    if include_keras:
        try:
            keras_res = _evaluate_with_runner(
                "Keras FP32 Model",
                reference_runner,
                test_data,
                test_labels,
                threshold,
                scorer,
                evaluation_batch_size=TIMING_BATCH_SIZE,
                reference_scores=reference_scores,
            )
            keras_res["model_name"] = "Keras FP32 (Baseline)"
            keras_res["size_mb"] = get_file_size_mb(str(Path(model_dir) / "best_model.weights.h5"))
            keras_res.update({
                "consistency_samples": min(len(test_data), CONSISTENCY_SAMPLES),
                "output_mse_vs_baseline": 0.0,
                "output_mean_abs_error_vs_baseline": 0.0,
                "output_max_abs_error_vs_baseline": 0.0,
                "output_relative_l2_vs_baseline": 0.0,
                "output_within_tolerance": True,
                "anomaly_disagreement_rate_vs_baseline": 0.0,
            })
            results.append(keras_res)
        except Exception as e:
            logger.error(f"Keras baseline evaluation failed: {e}")

    # 2. Discover every converted backend. Runners are instantiated one at a
    # time below so the Nano never retains several runtime sessions and GPU
    # contexts in its 4-GB unified memory.
    candidates = []
    tflite_files = {
        "TFLite FP32": "model_fp32.tflite",
        "TFLite FP16": "model_float16.tflite",
        "TFLite INT8": "model_int8.tflite",
    }
    for label, filename in tflite_files.items():
        tflite_path = dep_path / filename
        if tflite_path.exists():
            candidates.append((
                "tflite", label, tflite_path,
                lambda path=str(tflite_path): create_tflite_runner(path),
            ))

    onnx_files = {
        "ONNX FP32 (CPU)": "model.onnx",
        "ONNX FP16 (CPU)": "model_fp16.onnx",
        "ONNX INT8 (CPU)": "model_int8.onnx",
    }
    for label, filename in onnx_files.items():
        onnx_path = dep_path / filename
        if not onnx_path.exists():
            continue
        candidates.append((
            "onnx", label, onnx_path,
            lambda path=str(onnx_path): create_onnx_runner(path),
        ))

    for engine_path in sorted(dep_path.glob("*.engine")):
        name_suffix = engine_path.stem.split("_")[-1] if "_" in engine_path.stem else "fp16"
        label = f"TensorRT {name_suffix.upper()} (GPU)"
        candidates.append((
            "tensorrt", label, engine_path,
            lambda path=str(engine_path): create_tensorrt_runner(path),
        ))

    # 3. Validate every backend against the same Keras detector outputs before
    #    any converted backend enters warm-up, timing, or full evaluation.
    logger.info("Running cross-backend consistency preflight...")
    validated_candidates = []
    preflight_records = []
    for backend, label, path, runner_factory in candidates:
        runner = None
        try:
            runner = runner_factory()
            if runner is None:
                raise RuntimeError("runtime is unavailable")
            consistency = _check_consistency(runner, label)
            preflight_records.append({
                "model_name": label,
                "backend": backend,
                "path": str(path),
                **consistency,
            })
            validated_candidates.append((backend, label, path, runner_factory, consistency))
        except Exception as e:
            logger.error(f"Consistency preflight failed for {label}; skipping: {e}")
            preflight_records.append({
                "model_name": label,
                "backend": backend,
                "path": str(path),
                "preflight_passed": False,
                "runtime_initialization_failed": True,
                "reason": str(e),
                "anomaly_disagreement_rate_vs_baseline": None,
            })
        finally:
            del runner
            gc.collect()

    preflight_path = dep_path / "cross_backend_consistency.json"
    with open(preflight_path, "w") as handle:
        json.dump(preflight_records, handle, indent=2)
    logger.info(f"Saved cross-backend preflight report to {preflight_path}")

    # 4. Every executable runner proceeds to the standardized evaluation loop.
    # Fidelity-rejected paths are deliberately retained as scientific results.
    for backend, label, path, runner_factory, consistency in validated_candidates:
        runner = None
        try:
            runner = runner_factory()
            if runner is None:
                raise RuntimeError("runtime is unavailable")
            if backend == "tflite":
                result = evaluate_tflite_model(
                    str(path), test_data, test_labels, threshold, runner=runner,
                    scorer=scorer,
                    reference_scores=reference_scores,
                )
            elif backend == "onnx":
                result = evaluate_onnx_model(
                    str(path), test_data, test_labels, threshold, runner=runner,
                    scorer=scorer,
                    reference_scores=reference_scores,
                )
            else:
                result = evaluate_tensorrt_model(
                    str(path), test_data, test_labels, threshold, runner=runner,
                    scorer=scorer,
                    reference_scores=reference_scores,
                )
            if result:
                result["model_name"] = label
                result["size_mb"] = get_file_size_mb(str(path))
                result["detector_size_mb"] = result["size_mb"] + get_file_size_mb(
                    str(dep_path / "reference_embeddings.npy")
                )
                result.update(consistency)
                full_disagreement = result.get(
                    "full_test_anomaly_disagreement_rate_vs_baseline"
                )
                if full_disagreement is not None:
                    result["preflight_anomaly_disagreement_rate_vs_baseline"] = (
                        consistency.get("anomaly_disagreement_rate_vs_baseline")
                    )
                    result["anomaly_disagreement_rate_vs_baseline"] = (
                        full_disagreement
                    )
                    result["fidelity_passed"] = bool(
                        full_disagreement <= CONSISTENCY_DISAGREEMENT_WARN
                    )
                    for record in preflight_records:
                        if record.get("model_name") == label:
                            record.update({
                                "evaluation_completed": True,
                                "preflight_anomaly_disagreement_rate_vs_baseline": (
                                    consistency.get(
                                        "anomaly_disagreement_rate_vs_baseline"
                                    )
                                ),
                                "full_test_consistency_samples": int(len(test_data)),
                                "full_test_anomaly_disagreement_rate_vs_baseline": (
                                    full_disagreement
                                ),
                                "anomaly_disagreement_rate_vs_baseline": (
                                    full_disagreement
                                ),
                                "fidelity_passed": result["fidelity_passed"],
                            })
                            break
                results.append(result)
        except Exception as e:
            logger.error(f"Failed to evaluate {label}: {e}")
            for record in preflight_records:
                if record.get("model_name") == label:
                    record.update({
                        "evaluation_completed": False,
                        "runtime_evaluation_failed": True,
                        "fidelity_passed": False,
                        "evaluation_failure_reason": str(e),
                    })
                    break
        finally:
            del runner
            gc.collect()

    # Replace the preflight-only version with the full-test fidelity records.
    with open(preflight_path, "w") as handle:
        json.dump(preflight_records, handle, indent=2)

    # Print comparison table
    logger.info("================================== CLASSIFICATION & TIMING EVALUATION SUMMARY ==================================")
    logger.info(f"{'Model Format':<22} | {'Size (MB)':<10} | {'Latency/S Mean ± Std (ms)':<28} | {'F1-Score':<10} | {'ROC-AUC':<10} | {'Recall':<10}")
    logger.info("-" * 110)
    for r in results:
        logger.info(
            f"{r['model_name']:<22} | {r['size_mb']:<10.3f} | {r['latency_per_sample_mean_ms']:<7.4f} ± {r['latency_per_sample_std_ms']:<7.4f}      | "
            f"{r['f1']:<10.4f} | {r['auc_roc']:<10.4f} | {r['recall']:<10.4f}"
        )
    logger.info("================================================================================================================")

    return results


def save_evaluation_report(results: List[Dict[str, Any]], output_path: str) -> None:
    """Save evaluation metrics to JSON and Markdown reports."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # 1. Save JSON
    import json
    with open(out_p.with_suffix(".json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved evaluation benchmark JSON data to {out_p.with_suffix('.json')}")
    logger.info(f"Saved evaluation benchmark Markdown report to {out_p.with_suffix('.md')}")
