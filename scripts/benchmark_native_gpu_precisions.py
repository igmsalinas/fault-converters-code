#!/usr/bin/env python3
"""Benchmark latest CARLA checkpoints with native TensorFlow GPU precision."""

import argparse
import gc
import json
import platform
import sys
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run(args):
    physical_gpus = tf.config.list_physical_devices("GPU")
    if not physical_gpus:
        raise RuntimeError("A TensorFlow-visible CUDA GPU is required")
    for gpu in physical_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    from src.inference.predictor import AnomalyPredictor
    from src.inference.scoring import EmbeddingAnomalyScorer
    from src.deployment.utils import build_carla_embedding_model
    from src.utils.keras_ops_compat import set_mixed_precision_policy

    # Load and execute the FP32 reference first. For mixed FP16, release this
    # model before constructing the candidate so reported GPU memory is not the
    # sum of two resident checkpoints.
    set_mixed_precision_policy("float32")
    reference = AnomalyPredictor(model_dir=args.model_dir)
    if reference.reference_embeddings is None:
        raise RuntimeError(
            "Native GPU precision study requires a CARLA reference embedding bank"
        )
    if reference.threshold is None:
        raise RuntimeError("A fit-validation threshold is required for evaluation")
    threshold = float(reference.threshold)

    from src.deployment.runners import (
        compare_output_arrays,
        create_keras_runner,
        run_inference_loop,
    )
    from src.deployment.timing_evaluation import evaluate_keras_model
    from src.deployment.utils import load_deployment_datasets

    _, splits = load_deployment_datasets(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        max_files=args.max_files,
        preprocessor=reference.preprocessor,
        split_indices_path=str(Path(args.model_dir) / "split_indices.npz"),
    )
    test_data = splits["test"]
    test_labels = splits["test_labels"]
    reference_model = build_carla_embedding_model(reference.model)
    reference_runner = create_keras_runner(reference_model)
    scorer = EmbeddingAnomalyScorer(
        reference.reference_embeddings,
        method=reference.scoring_method,
        k=reference.k_neighbors,
    )
    reference_outputs = run_inference_loop(reference_runner, test_data, batch_size=1)
    reference_scores = scorer.score(reference_outputs)

    policy = "mixed_float16" if args.precision == "fp16" else "float32"
    if args.precision == "fp16":
        del reference_runner, reference_model, reference
        keras.backend.clear_session()
        gc.collect()
        set_mixed_precision_policy(policy)
        candidate = AnomalyPredictor(model_dir=args.model_dir)
    else:
        candidate = reference
    if candidate.threshold is None or float(candidate.threshold) != threshold:
        raise RuntimeError("Candidate and FP32 reference thresholds do not match")
    candidate_model = build_carla_embedding_model(candidate.model)
    candidate_runner = create_keras_runner(candidate_model)
    sample_count = min(len(test_data), args.consistency_samples)
    candidate_outputs = run_inference_loop(
        candidate_runner, test_data[:sample_count], batch_size=1
    )
    consistency = compare_output_arrays(
        reference_outputs[:sample_count],
        candidate_outputs,
        test_data[:sample_count],
        threshold=threshold,
        score_function=scorer.score,
    )

    tf.config.experimental.reset_memory_stats("GPU:0")
    evaluation = evaluate_keras_model(
        candidate_model,
        test_data,
        test_labels,
        threshold,
        scorer=scorer,
        timing_warmup=args.warmup,
        timing_runs=args.runs,
        reference_scores=reference_scores,
    )
    full_disagreement = evaluation[
        "full_test_anomaly_disagreement_rate_vs_baseline"
    ]
    consistency["preflight_anomaly_disagreement_rate_vs_baseline"] = (
        consistency.get("anomaly_disagreement_rate_vs_baseline")
    )
    consistency["full_test_consistency_samples"] = int(len(test_data))
    consistency["full_test_anomaly_disagreement_rate_vs_baseline"] = (
        full_disagreement
    )
    consistency["anomaly_disagreement_rate_vs_baseline"] = full_disagreement
    memory = tf.config.experimental.get_memory_info("GPU:0")
    result = {
        "device": {
            "hostname": platform.node(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "tensorflow_version": tf.__version__,
            "tensorflow_gpu": tf.config.experimental.get_device_details(
                physical_gpus[0]
            ),
            "l4t_release": (
                Path("/etc/nv_tegra_release").read_text().splitlines()[0]
                if Path("/etc/nv_tegra_release").exists() else None
            ),
        },
        "model_dir": args.model_dir,
        "backend": "keras-gpu",
        "precision": args.precision,
        "policy": policy,
        "protocol": {
            "batch_size": evaluation["timing_batch_size"],
            "warmup_runs": evaluation["timing_warmup_runs"],
            "timed_runs": evaluation["timing_runs"],
            "consistency_samples": args.consistency_samples,
        },
        "evaluation": evaluation,
        "consistency": consistency,
        "tensorflow_gpu_memory": {
            "measurement_scope": "candidate_model_after_reference_release",
            "current_mb": float(memory["current"]) / (1024.0 * 1024.0),
            "peak_mb": float(memory["peak"]) / (1024.0 * 1024.0),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16"), required=True)
    parser.add_argument("--data-dir", default="data/buck/buck_data/dataset_01")
    parser.add_argument("--cache-dir", default="/tmp/native-gpu-benchmark")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--consistency-samples", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--runs", type=int, default=150)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
