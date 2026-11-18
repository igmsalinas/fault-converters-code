"""
Deployment Evaluation Script
=============================

Runs benchmarking on converted models.
"""
import argparse
import sys
import json
import subprocess
from pathlib import Path
import numpy as np
from typing import Dict, Any, List

from .deployment import (
    load_deployment_datasets,
    run_full_performance_suite,
    run_deployment_evaluations,
)
from .inference.predictor import AnomalyPredictor
from .utils.logger import get_logger, setup_logger

setup_logger()
logger = get_logger(__name__)

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate deployed models.")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to the trained model directory.")
    parser.add_argument("--data-dir", type=str, default="data/buck/buck_data", help="Directory containing simulation files.")
    parser.add_argument("--cache-dir", type=str, default="cache", help="Directory for caching preprocessed data.")
    parser.add_argument("--normal-threshold", type=float, default=5.0, help="Tolerance threshold.")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum files to load.")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory for deployment files.")
    parser.add_argument(
        "--skip-cpu", action="store_true",
        help="Reuse the existing CPU-baseline report while refreshing converted backends.",
    )
    return parser.parse_args(argv)

def get_converted_models(output_dir: Path) -> Dict[str, str]:
    """Scan output directory for converted models."""
    models = {}
    
    # Check for TFLite
    if (output_dir / "model_fp32.tflite").exists():
        models["tflite_fp32"] = str(output_dir / "model_fp32.tflite")
    if (output_dir / "model_float16.tflite").exists():
        models["tflite_fp16"] = str(output_dir / "model_float16.tflite")
    if (output_dir / "model_int8.tflite").exists():
        models["tflite_int8"] = str(output_dir / "model_int8.tflite")
        
    # Check for ONNX
    if (output_dir / "model.onnx").exists():
        models["onnx_fp32"] = str(output_dir / "model.onnx")
    if (output_dir / "model_fp16.onnx").exists():
        models["onnx_fp16"] = str(output_dir / "model_fp16.onnx")
    if (output_dir / "model_int8.onnx").exists():
        models["onnx_int8"] = str(output_dir / "model_int8.onnx")
        
    # Check for TensorRT
    if (output_dir / "model_fp32.engine").exists():
        models["tensorrt_fp32"] = str(output_dir / "model_fp32.engine")
    if (output_dir / "model_fp16.engine").exists():
        models["tensorrt_fp16"] = str(output_dir / "model_fp16.engine")
    if (output_dir / "model_int8.engine").exists():
        models["tensorrt_int8"] = str(output_dir / "model_int8.engine")
        
    return models

def generate_unified_markdown_report(
    perf_results: List[Dict[str, Any]],
    eval_results: List[Dict[str, Any]],
    output_path: str
):
    md = ["# Edge Compute Unified Deployment Report", "", "This report aggregates two deployment studies: timing & memory under single-sample inference, and classification metric fidelity.", "", "All latencies are measured under a uniform protocol: single-sample batch (n=1), 15 warm-up iterations, 150 timed runs.", ""]

    # Section 1: Timing & Memory (single-sample, n=1)
    md.extend(["## 1. Timing & Memory Profiling (Single-Sample, n=1)", "", "CPU primary latency uses `perf_counter`; GPU primary latency uses CUDA events. End-to-end latency always uses `perf_counter` and includes host overhead/transfers.", "", "| Model Format | Primary Latency | Timing Source | End-to-End Latency | Net RAM Used | Net VRAM Used | Peak RAM | Peak VRAM |", "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"])
    for r in perf_results:
        latency_str = f"{r['latency_mean_ms']:.3f} ± {r['latency_std_ms']:.3f} ms"
        end_to_end_str = f"{r.get('end_to_end_latency_mean_ms', r['latency_mean_ms']):.3f} ± {r.get('end_to_end_latency_std_ms', r['latency_std_ms']):.3f} ms"
        md.append(f"| **{r['model_name']}** | {latency_str} | {r.get('timing_source', 'perf_counter')} | {end_to_end_str} | {r['net_ram_mb']:.3f} MB | {r['net_vram_mb']:.3f} MB | {r['peak_ram_mb']:.3f} MB | {r['peak_vram_mb']:.3f} MB |")
    md.append("")

    # Section 2: Classification Metrics
    md.extend(["## 2. Classification Metrics Evaluation", "", "| Model Format | Graph Size (MB) | Detector Size (MB) | Primary Latency | Timing Source | End-to-End Latency | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Decision Discordance |", "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"])
    for r in eval_results:
        # Support fallback to old scalar values for backward compatibility if old JSONs exist
        mean = r.get('latency_per_sample_mean_ms', r.get('latency_per_sample_ms', 0.0))
        std = r.get('latency_per_sample_std_ms', 0.0)
        latency_str = f"{mean:.4f} ± {std:.4f}"
        e2e_str = f"{r.get('end_to_end_latency_mean_ms', mean):.4f} ± {r.get('end_to_end_latency_std_ms', std):.4f}"
        disagreement = r.get('anomaly_disagreement_rate_vs_baseline', 0.0)
        detector_size = r.get("detector_size_mb", r["size_mb"])
        md.append(f"| **{r['model_name']}** | {r['size_mb']:.3f} MB | {detector_size:.3f} MB | {latency_str} ms | {r.get('timing_source', 'perf_counter')} | {e2e_str} ms | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['auc_roc']:.4f} | {disagreement:.2%} |")
    md.append("")

    out_p = Path(output_path)
    with open(out_p, "w") as f:
        f.write("\n".join(md))
    logger.info(f"Saved unified benchmark report to {out_p}")


def build_unified_results(
    perf_results: List[Dict[str, Any]],
    eval_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge the two deployment studies into a single dict keyed by strategy.

    Each top-level key is a deployment strategy display name (e.g.
    ``"TensorRT FP16 (GPU)"``) mapping to a unified record with ``size_mb`` plus
    ``performance`` and ``evaluation`` sections.
    Any section is ``None`` if that strategy was not covered by the corresponding suite.
    """
    unified: Dict[str, Any] = {}

    # Canonicalize display-name variants so the same strategy merges into one key.
    name_aliases = {"TFLite DYNAMIC": "TFLite Dynamic"}

    def _canon(name: str) -> str:
        return name_aliases.get(name, name)

    def _entry(name: str) -> Dict[str, Any]:
        return unified.setdefault(name, {
            "size_mb": None,
            "detector_size_mb": None,
            "performance": None,
            "evaluation": None,
        })

    # 1. Hardware resource profiling (timing & memory)
    for r in perf_results:
        name = _canon(r.get("model_name"))
        if not name:
            continue
        e = _entry(name)
        if r.get("size_mb") is not None:
            e["size_mb"] = r["size_mb"]
        if r.get("detector_size_mb") is not None:
            e["detector_size_mb"] = r["detector_size_mb"]
        e["performance"] = {
            "latency_mean_ms": r.get("latency_mean_ms"),
            "latency_std_ms": r.get("latency_std_ms"),
            "latency_min_ms": r.get("latency_min_ms"),
            "latency_max_ms": r.get("latency_max_ms"),
            "timing_source": r.get("timing_source"),
            "timing_scope": r.get("timing_scope"),
            "end_to_end_latency_mean_ms": r.get("end_to_end_latency_mean_ms"),
            "end_to_end_latency_std_ms": r.get("end_to_end_latency_std_ms"),
            "detector_end_to_end_latency_mean_ms": r.get(
                "detector_end_to_end_latency_mean_ms"
            ),
            "detector_end_to_end_latency_std_ms": r.get(
                "detector_end_to_end_latency_std_ms"
            ),
            "detector_timing_source": r.get("detector_timing_source"),
            "net_ram_mb": r.get("net_ram_mb"),
            "net_vram_mb": r.get("net_vram_mb"),
            "peak_ram_mb": r.get("peak_ram_mb"),
            "peak_vram_mb": r.get("peak_vram_mb"),
        }

    # 2. Classification metrics
    for r in eval_results:
        name = _canon(r.get("model_name"))
        if not name:
            continue
        e = _entry(name)
        if r.get("size_mb") is not None:
            e["size_mb"] = r["size_mb"]
        if r.get("detector_size_mb") is not None:
            e["detector_size_mb"] = r["detector_size_mb"]
        e["evaluation"] = {
            "latency_per_sample_mean_ms": r.get(
                "latency_per_sample_mean_ms", r.get("latency_per_sample_ms")
            ),
            "latency_per_sample_std_ms": r.get("latency_per_sample_std_ms"),
            "timing_source": r.get("timing_source"),
            "timing_scope": r.get("timing_scope"),
            "end_to_end_latency_mean_ms": r.get("end_to_end_latency_mean_ms"),
            "end_to_end_latency_std_ms": r.get("end_to_end_latency_std_ms"),
            "detector_end_to_end_latency_mean_ms": r.get(
                "detector_end_to_end_latency_mean_ms"
            ),
            "detector_end_to_end_latency_std_ms": r.get(
                "detector_end_to_end_latency_std_ms"
            ),
            "detector_timing_source": r.get("detector_timing_source"),
            "detector_timing_scope": r.get("detector_timing_scope"),
            "accuracy": r.get("accuracy"),
            "precision": r.get("precision"),
            "recall": r.get("recall"),
            "f1": r.get("f1"),
            "auc_roc": r.get("auc_roc"),
            "output_mse_vs_baseline": r.get("output_mse_vs_baseline"),
            "output_max_abs_error_vs_baseline": r.get("output_max_abs_error_vs_baseline"),
            "output_relative_l2_vs_baseline": r.get("output_relative_l2_vs_baseline"),
            "output_within_tolerance": r.get("output_within_tolerance"),
            "fidelity_passed": r.get("fidelity_passed"),
            "score_mean_abs_error_vs_baseline": r.get(
                "score_mean_abs_error_vs_baseline"
            ),
            "score_max_abs_error_vs_baseline": r.get(
                "score_max_abs_error_vs_baseline"
            ),
            "anomaly_disagreement_rate_vs_baseline": r.get("anomaly_disagreement_rate_vs_baseline"),
        }

    return unified


def save_unified_json_report(unified: Dict[str, Any], output_path: str) -> None:
    """Write the unified deployment results dict to a single JSON file."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(unified, f, indent=2)
    logger.info(f"Saved unified deployment JSON report to {out_p}")


def main(argv=None):
    args = parse_args(argv)

    # The consistency reference and TensorRT runners coexist in this process.
    # Prevent TensorFlow from reserving the Nano's entire unified GPU memory so
    # TensorRT/PyCUDA can create their contexts and buffers afterward.
    try:
        import tensorflow as tf
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except (RuntimeError, ValueError) as e:
        logger.warning(f"Could not enable TensorFlow GPU memory growth: {e}")

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        logger.error(f"Model directory does not exist: {model_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else model_dir / "deployment"
    if not output_dir.exists():
        logger.error(f"Deployment output directory does not exist. Run optimization first: {output_dir}")
        sys.exit(1)

    logger.info(f"Deployment evaluation for: {output_dir}")
    existing_unified = {}
    existing_report = output_dir / "unified_deployment_report.json"
    if args.skip_cpu and existing_report.exists():
        with existing_report.open() as handle:
            existing_unified = json.load(handle)

    try:
        predictor = AnomalyPredictor(model_dir=str(model_dir))
        logger.info(f"Loaded baseline predictor successfully.")
    except Exception as e:
        logger.error(f"Failed to load baseline predictor: {e}")
        sys.exit(1)

    try:
        _, splits = load_deployment_datasets(
            data_dir=args.data_dir,
            cache_dir=args.cache_dir,
            normal_threshold=args.normal_threshold,
            max_files=args.max_files,
            preprocessor=predictor.preprocessor,
            split_indices_path=str(model_dir / "split_indices.npz"),
        )
        test_data = splits["test"]
    except Exception as e:
        logger.error(f"Failed to prepare datasets: {e}")
        sys.exit(1)

    converted_models = get_converted_models(output_dir)
    if not converted_models:
        logger.warning("No converted models found in output directory.")

    # ---- Deployment studies ----
    try:
        # ---------------- CPU BENCHMARK RUNNER ----------------
        logger.info("--- Running CPU-only Benchmark via Subprocess ---")
        temp_data_file = str(output_dir / "temp_test_data.npy")
        temp_labels_file = str(output_dir / "temp_test_labels.npy")
        cpu_out_file = str(output_dir / "cpu_res.json")
        np.save(temp_data_file, test_data)
        np.save(temp_labels_file, splits["test_labels"])
        
        cpu_res = None
        if args.skip_cpu:
            logger.info("Reusing the existing isolated CPU baseline.")
        else:
            try:
                import sys
                subprocess.run([
                    sys.executable, "-m", "src.deployment.benchmark_cpu",
                    "--model-dir", str(model_dir),
                    "--data-file", temp_data_file,
                    "--labels-file", temp_labels_file,
                    "--output-file", cpu_out_file
                ], check=True)
                if Path(cpu_out_file).exists():
                    with open(cpu_out_file, "r") as f:
                        cpu_res = json.load(f)
            except Exception as e:
                logger.error(f"Failed to run CPU benchmark: {e}")
        # ------------------------------------------------------

        # Validate all converted backends before the separate performance suite.
        # run_deployment_evaluations performs its all-backend consistency
        # preflight before any converted runner enters a timed/full-data loop.
        logger.info("--- Study 1: Consistency, Classification, and Timing Evaluation ---")
        eval_results = run_deployment_evaluations(
            model_dir=str(model_dir),
            deployment_dir=str(output_dir),
            test_data=test_data,
            test_labels=splits["test_labels"],
            include_keras=False,
            predictor=predictor,
        )

        if cpu_res and "evaluation" in cpu_res:
            eval_results.insert(1, cpu_res["evaluation"])

        validated_converted_models = {
            result["model_name"] for result in eval_results
            if result.get("model_name") != "Keras FP32 (CPU)"
        }

        # Study 2: single-sample timing and memory performance
        logger.info("--- Study 2: Timing and Memory Performance (single input, n=1) ---")
        perf_results = run_full_performance_suite(
            model_dir=str(model_dir),
            deployment_dir=str(output_dir),
            test_data=test_data,
            include_keras=False,
            allowed_converted_models=validated_converted_models,
        )

        if cpu_res and "benchmark" in cpu_res:
            perf_results.insert(1, cpu_res["benchmark"])

        # Merge both studies into a single unified report keyed by deployment strategy
        logger.info("--- Generating Unified Deployment Report ---")
        unified = build_unified_results(perf_results, eval_results)
        if existing_unified:
            existing_unified.update(unified)
            unified = existing_unified
        save_unified_json_report(unified, str(output_dir / "unified_deployment_report.json"))
        generate_unified_markdown_report(
            perf_results, eval_results,
            str(output_dir / "unified_deployment_report.md")
        )

        # Clean up intermediate artifacts
        for tmp in (temp_data_file, temp_labels_file, cpu_out_file):
            try:
                tmp_path = Path(tmp)
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

    except Exception as e:
        logger.error(f"Deployment study run failed: {e}")
        raise

    logger.info(f"Deployment evaluation complete! All reports in: {output_dir}")

if __name__ == "__main__":
    main()
