"""
Standalone script to run Keras benchmarking on CPU only.
It enforces CUDA_VISIBLE_DEVICES="-1" before loading any TensorFlow components.
"""
import os
import sys
import json
import argparse
from pathlib import Path

# Enforce CPU execution
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Now we can import the rest safely
import numpy as np
from src.inference.predictor import AnomalyPredictor
from src.deployment.timing_memory_benchmark import benchmark_timing_memory
from src.deployment.runners import create_keras_runner
from src.deployment.timing_evaluation import evaluate_keras_model
from src.deployment.utils import get_file_size_mb
from src.deployment.utils import build_carla_embedding_model
from src.inference.scoring import EmbeddingAnomalyScorer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--labels-file", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    # Load Model
    predictor = AnomalyPredictor(model_dir=args.model_dir)
    if predictor.reference_embeddings is None:
        raise RuntimeError("CPU deployment benchmark requires a CARLA embedding bank")
    keras_model = build_carla_embedding_model(predictor.model)
    scorer = EmbeddingAnomalyScorer(
        predictor.reference_embeddings,
        method=predictor.scoring_method,
        k=predictor.k_neighbors,
    )
    if predictor.threshold is None:
        raise RuntimeError(
            "CPU deployment benchmark requires a fit-validation threshold"
        )
    threshold = predictor.threshold

    # Load Data
    test_data = np.load(args.data_file)
    test_labels = np.load(args.labels_file)

    # 1. Timing Benchmark
    print("Running Timing Benchmark on CPU...")
    # Single-sample (n=1) latency under the uniform protocol so CPU and GPU
    # numbers stay directly comparable with run_full_performance_suite.
    x_perf = test_data[:1]
    bench_res = benchmark_timing_memory(
        "Keras FP32 (CPU)", lambda: create_keras_runner(keras_model), x_perf
    )
    checkpoint_dir = Path(args.model_dir) / "checkpoints" / "best"
    bench_res["size_mb"] = sum(
        get_file_size_mb(str(checkpoint_dir / filename))
        for filename in ("encoder.weights.npz", "projection_head.weights.npz")
    )
    bank_size_mb = get_file_size_mb(
        str(checkpoint_dir / "reference_embeddings.npy")
    )
    bench_res["detector_size_mb"] = bench_res["size_mb"] + bank_size_mb

    # 2. Evaluation Benchmark
    print("Running Evaluation Benchmark on CPU...")
    eval_res = evaluate_keras_model(
        keras_model, test_data, test_labels, threshold, scorer=scorer
    )
    eval_res["model_name"] = "Keras FP32 (CPU)"
    eval_res["size_mb"] = bench_res["size_mb"]
    eval_res["detector_size_mb"] = bench_res["detector_size_mb"]
    eval_res["full_test_consistency_samples"] = int(len(test_data))
    eval_res["full_test_anomaly_disagreement_rate_vs_baseline"] = 0.0
    eval_res["anomaly_disagreement_rate_vs_baseline"] = 0.0

    # Save outputs
    out_data = {
        "benchmark": bench_res,
        "evaluation": eval_res
    }
    with open(args.output_file, "w") as f:
        json.dump(out_data, f, indent=2)

if __name__ == "__main__":
    main()
