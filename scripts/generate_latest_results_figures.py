#!/usr/bin/env python3
"""Generate publication figures from the latest CARLA and Jetson reports."""

from __future__ import annotations

import csv
import json
import re
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXPERIMENT = ROOT / "experiments" / "dataset_01_run_06_corrected"
PLOT_DIR = ROOT / "docs" / "plots"
RESULT_DIR = ROOT / "papers" / "results"
MODELS = ["conv1d", "gru", "lstm", "mlp", "transformer"]
DISPLAY = {
    "conv1d": "Conv1D",
    "gru": "GRU",
    "lstm": "LSTM",
    "mlp": "MLP",
    "transformer": "Transformer",
}
COLORS = {"CPU": "#577590", "FP32": "#43aa8b", "FP16": "#f8961e", "INT8": "#e76f51"}

# Figure matrices exclude INT8: the deployment results present only obtainable
# precisions (FP32/FP16). INT8 raw measurements remain in the CSV records.
RESOURCE_BACKENDS = [
    "Keras CPU", "Keras GPU FP32", "Keras GPU FP16",
    "TFLite FP32", "TFLite FP16",
    "ONNX FP32", "ONNX FP16",
    "TRT FP32", "TRT FP16",
]
RESOURCE_NUMERIC_FIELDS = [
    "latency_ms", "detector_end_to_end_latency_ms",
    "throughput_inferences_per_second", "shared_ram_active_mean_delta_mb",
    "shared_ram_active_peak_delta_mb", "process_ram_peak_delta_mb",
    "cuda_memory_peak_delta_mb", "cpu_active_mean_percent",
    "gpu_active_mean_percent", "gpu_max_temperature_c", "idle_input_power_w",
    "active_input_power_w", "dynamic_input_power_w",
    "total_energy_per_inference_mj", "dynamic_energy_per_inference_mj",
    "gpu_rail_active_power_w", "cpu_rail_active_power_w",
    "throughput_per_input_watt",
]


def _resource_backend(path, record):
    runtime = record["backend"]
    precision = (record.get("precision") or "fp32").upper()
    if runtime == "keras-cpu":
        return "Keras CPU", "Keras CPU"
    if runtime.startswith("keras-gpu-") or runtime == "keras-gpu":
        return f"Keras GPU {precision}", f"Keras GPU {precision}"
    if runtime == "tensorrt":
        return f"TensorRT {precision}", f"TensorRT {precision}"
    if runtime == "onnx":
        return f"ONNX {precision}", f"ONNX {precision}"
    return f"TFLite {precision}", f"TFLite {precision}"


def _load(path: Path):
    with path.open() as handle:
        return json.load(handle)


def _resource_repetition(path: Path) -> int:
    match = re.search(r"_repeat(\d+)$", path.stem)
    if match is None:
        raise ValueError(
            f"Resource record is not an explicit repetition: {path.name}"
        )
    return int(match.group(1))


def _resource_records(model: str):
    """Load every isolated resource repetition without best-run selection."""
    dep_dir = EXPERIMENT / f"carla_{model}" / "deployment"
    records = []
    for path in sorted(dep_dir.glob("jetson_resources_*_repeat*.json")):
        record = _load(path)
        report_name, backend = _resource_backend(path, record)
        records.append((backend, report_name, _resource_repetition(path), record))
    return records


def _nested_number(record, *keys):
    value = record
    for key in keys:
        if not isinstance(value, dict):
            return np.nan
        value = value.get(key)
    return float(value) if value is not None else np.nan


def _save(fig, stem: str) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png", "svg"):
        kwargs = {"dpi": 300} if suffix == "png" else {}
        fig.savefig(PLOT_DIR / f"{stem}.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def collect_results():
    classification = {}
    deployment = {}
    rejected = {}
    for model in MODELS:
        model_dir = EXPERIMENT / f"carla_{model}"
        classification[model] = _load(model_dir / "metrics.json")
        report_path = model_dir / "deployment" / "unified_deployment_report.json"
        deployment[model] = _load(report_path) if report_path.exists() else {}
        consistency_path = model_dir / "deployment" / "cross_backend_consistency.json"
        if consistency_path.exists():
            rejected[model] = {
                row["model_name"]: row
                for row in _load(consistency_path)
                if (
                    not row.get("preflight_passed", True)
                    or not row.get("fidelity_passed", True)
                )
            }
        else:
            rejected[model] = {}
        for precision in ("fp32", "fp16"):
            native_path = model_dir / "deployment" / f"native_gpu_{precision}.json"
            if not native_path.exists():
                continue
            native = _load(native_path)
            lat = (native.get("evaluation") or {}).get("latency_per_sample_mean_ms")
            if lat is not None:
                deployment[model][f"Keras GPU {precision.upper()}"] = {
                    "performance": {"latency_mean_ms": lat},
                    "evaluation": native.get("evaluation", {}),
                }
            consistency = native.get("consistency") or {}
            disagreement = (native.get("evaluation") or {}).get(
                "full_test_anomaly_disagreement_rate_vs_baseline",
                consistency.get("anomaly_disagreement_rate_vs_baseline", 0.0)
            )
            if disagreement is not None and disagreement > 0.01:
                rejected[model][f"Keras GPU {precision.upper()}"] = {
                    "anomaly_disagreement_rate_vs_baseline": disagreement,
                    "latency_mean_ms": lat,
                }
    return classification, deployment, rejected


def write_tables(classification, deployment, rejected):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in MODELS:
        metric = classification[model]
        intervals = metric.get("bootstrap_95_ci") or {}
        rows.append({
            "model": DISPLAY[model],
            "accuracy": metric["accuracy"],
            "precision": metric["precision"],
            "recall": metric["recall"],
            "f1": metric["f1"],
            "f1_ci_lower": (intervals.get("f1") or {}).get("lower"),
            "f1_ci_upper": (intervals.get("f1") or {}).get("upper"),
            "roc_auc": metric["auc_roc"],
            "roc_auc_ci_lower": (intervals.get("auc_roc") or {}).get("lower"),
            "roc_auc_ci_upper": (intervals.get("auc_roc") or {}).get("upper"),
            "pr_auc": metric["auc_pr"],
            "pr_auc_ci_lower": (intervals.get("auc_pr") or {}).get("lower"),
            "pr_auc_ci_upper": (intervals.get("auc_pr") or {}).get("upper"),
        })
    with (RESULT_DIR / "latest_carla_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    ablation_summary_path = (
        EXPERIMENT / "ablations" / "negative_sampler" / "summary.json"
    )
    if ablation_summary_path.exists():
        ablation_root = ablation_summary_path.parent
        ablation_summary = _load(ablation_summary_path)
        ablation_rows = []
        for sampler in ("physics", "heuristic"):
            for seed in ablation_summary["seeds"]:
                metrics = _load(
                    ablation_root / sampler / f"seed_{seed}" / "metrics.json"
                )
                ablation_rows.append({
                    "sampler": sampler,
                    "seed": seed,
                    **{
                        metric: metrics[metric]
                        for metric in (
                            "accuracy", "precision", "recall", "f1",
                            "auc_roc", "auc_pr",
                        )
                    },
                })
        with (RESULT_DIR / "negative_sampler_ablation.csv").open(
            "w", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(ablation_rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(ablation_rows)

    deploy_rows = []
    for model in MODELS:
        present_backends = set()
        for backend, values in deployment[model].items():
            present_backends.add(backend)
            evaluation = values.get("evaluation") or {}
            performance = values.get("performance") or {}
            rejected_record = rejected[model].get(backend)
            deploy_rows.append({
                "model": DISPLAY[model],
                "backend": backend,
                "status": "rejected" if rejected_record else "validated",
                "latency_ms": performance.get("latency_mean_ms"),
                "end_to_end_latency_ms": performance.get("end_to_end_latency_mean_ms"),
                "detector_end_to_end_latency_ms": evaluation.get(
                    "detector_end_to_end_latency_mean_ms"
                ),
                "timing_source": performance.get("timing_source"),
                "f1": evaluation.get("f1"),
                "roc_auc": evaluation.get("auc_roc"),
                "decision_discordance": (
                    rejected_record or evaluation
                ).get("anomaly_disagreement_rate_vs_baseline"),
                "peak_process_ram_mb": performance.get("peak_ram_mb"),
                "engine_size_mb": values.get("size_mb"),
                "detector_size_mb": values.get("detector_size_mb"),
            })
        for backend, rejected_record in rejected[model].items():
            if backend in present_backends:
                continue
            deploy_rows.append({
                "model": DISPLAY[model],
                "backend": backend,
                "status": "runtime_failed" if (
                    rejected_record.get("runtime_initialization_failed")
                    or rejected_record.get("runtime_evaluation_failed")
                ) else "rejected",
                "latency_ms": None,
                "end_to_end_latency_ms": None,
                "timing_source": None,
                "f1": None,
                "roc_auc": None,
                "decision_discordance": rejected_record.get(
                    "anomaly_disagreement_rate_vs_baseline"
                ),
                "peak_process_ram_mb": None,
                "engine_size_mb": None,
                "detector_size_mb": None,
            })
    fields = list(deploy_rows[0])
    with (RESULT_DIR / "jetson_deployment_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(deploy_rows)

    resource_rows = []
    for model in MODELS:
        for backend, report_name, repetition, record in _resource_records(model):
            power = record["power"]
            cuda_memory = record.get("cuda_memory", {})
            rejected_record = rejected[model].get(report_name)
            resource_rows.append({
                "model": DISPLAY[model],
                "backend": backend,
                "repetition": repetition,
                "fidelity_status": "rejected" if rejected_record else "validated",
                "decision_discordance": (
                    rejected_record or {}
                ).get("anomaly_disagreement_rate_vs_baseline", 0.0),
                "latency_ms": record["timing"]["latency_mean_ms"],
                "detector_end_to_end_latency_ms": record["detector_timing"][
                    "latency_mean_ms"
                ],
                "timing_source": record["timing"]["timing_source"],
                "throughput_inferences_per_second": record["sustained_throughput_inferences_per_second"],
                "shared_ram_active_mean_delta_mb": record["system_shared_memory"]["active_mean_delta_mb"],
                "shared_ram_active_peak_delta_mb": record["system_shared_memory"]["active_peak_delta_mb"],
                "process_ram_peak_delta_mb": record["process_ram"]["peak_delta_mb"],
                "cuda_memory_peak_delta_mb": cuda_memory.get(
                    "peak_delta_from_context_baseline_mb"
                ),
                "cpu_active_mean_percent": record["utilization"]["cpu_active_mean_percent"],
                "gpu_active_mean_percent": record["utilization"]["gpu_active_mean_percent"],
                "gpu_max_temperature_c": record["temperatures"].get("gpu", {}).get("max_c"),
                "power_sensor_available": power["sensor_available"],
                "idle_input_power_w": power.get("idle_input_mean_w"),
                "active_input_power_w": power.get("active_input_mean_w"),
                "dynamic_input_power_w": power.get("dynamic_input_mean_w"),
                "total_energy_per_inference_mj": power.get(
                    "total_energy_per_inference_mj"
                ),
                "dynamic_energy_per_inference_mj": power.get(
                    "dynamic_energy_per_inference_mj"
                ),
                "gpu_rail_active_power_w": power.get("gpu_rail_active_mean_w"),
                "cpu_rail_active_power_w": power.get("cpu_rail_active_mean_w"),
                "throughput_per_input_watt": power.get("throughput_per_input_watt"),
                "idle_power_samples": (power.get("idle") or {}).get("sample_count"),
                "active_power_samples": (power.get("active") or {}).get("sample_count"),
            })
    if resource_rows:
        with (RESULT_DIR / "jetson_resource_measurements.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(resource_rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(resource_rows)

        grouped_rows = {}
        for row in resource_rows:
            grouped_rows.setdefault(
                (row["model"], row["backend"], row["fidelity_status"]), []
            ).append(row)
        summary_rows = []
        for (model, backend, status), repetitions in sorted(grouped_rows.items()):
            summary = {
                "model": model,
                "backend": backend,
                "fidelity_status": status,
                "repetitions": len(repetitions),
            }
            for field in RESOURCE_NUMERIC_FIELDS:
                values = [
                    float(row[field]) for row in repetitions
                    if row.get(field) is not None
                    and np.isfinite(float(row[field]))
                ]
                summary[field + "_mean"] = float(np.mean(values)) if values else None
                if len(values) > 1:
                    summary[field + "_std"] = float(np.std(values, ddof=1))
                elif values:
                    summary[field + "_std"] = 0.0
                else:
                    summary[field + "_std"] = None
            summary_rows.append(summary)
        with (RESULT_DIR / "jetson_resource_summary.csv").open(
            "w", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(summary_rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(summary_rows)

    native_rows = []
    for model in MODELS:
        dep_dir = EXPERIMENT / f"carla_{model}" / "deployment"
        for precision in ("fp32", "fp16"):
            path = dep_dir / f"native_gpu_{precision}.json"
            if not path.exists():
                continue
            record = _load(path)
            native_rows.append({
                "model": DISPLAY[model],
                "precision": precision.upper(),
                "fidelity_status": (
                    "rejected"
                    if f"Keras GPU {precision.upper()}" in rejected[model]
                    else "validated"
                ),
                "device_hostname": record.get("device", {}).get("hostname"),
                "latency_ms": record["evaluation"]["latency_per_sample_mean_ms"],
                "end_to_end_latency_ms": record["evaluation"]["end_to_end_latency_mean_ms"],
                "detector_end_to_end_latency_ms": record["evaluation"][
                    "detector_end_to_end_latency_mean_ms"
                ],
                "timing_source": record["evaluation"]["timing_source"],
                "output_mse_vs_fp32": record["consistency"]["output_mse_vs_baseline"],
                "decision_discordance": record["consistency"].get("anomaly_disagreement_rate_vs_baseline"),
                "f1": record["evaluation"]["f1"],
                "roc_auc": record["evaluation"]["auc_roc"],
                "tensorflow_gpu_peak_mb": record["tensorflow_gpu_memory"]["peak_mb"],
            })
    if native_rows:
        with (RESULT_DIR / "native_gpu_precision_results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(native_rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(native_rows)

    trt_layer_rows = []
    for model in MODELS:
        dep_dir = EXPERIMENT / f"carla_{model}" / "deployment"
        for path in sorted(dep_dir.glob("model_*.engine.layers.json")):
            payload = _load(path)
            layers = payload.get("Layers") or []
            counts = {"FP32": 0, "FP16": 0, "INT8": 0, "UNKNOWN": 0}
            for layer in layers:
                formats = {
                    str(tensor.get("Format/Datatype", ""))
                    for tensor in (layer.get("Outputs") or [])
                }
                matched = {
                    precision
                    for precision in ("FP32", "FP16", "INT8")
                    if any(precision.lower() in value.lower() for value in formats)
                }
                if matched:
                    for precision in matched:
                        counts[precision] += 1
                else:
                    counts["UNKNOWN"] += 1
            precision_match = re.match(
                r"model_(fp32|fp16|int8)\.engine\.layers\.json$", path.name
            )
            if precision_match is None:
                continue
            trt_layer_rows.append({
                "model": DISPLAY[model],
                "engine_precision_flag": precision_match.group(1).upper(),
                "layer_count": len(layers),
                "layers_with_fp32_outputs": counts["FP32"],
                "layers_with_fp16_outputs": counts["FP16"],
                "layers_with_int8_outputs": counts["INT8"],
                "layers_with_unclassified_outputs": counts["UNKNOWN"],
                "binding_formats_json": json.dumps(
                    payload.get("Bindings") or [], sort_keys=True
                ),
                "interpretation": (
                    "Counts observed TensorRT layer output tensor formats; does not "
                    "claim undocumented internal tactic arithmetic precision."
                ),
                "source_file": str(path.relative_to(ROOT)),
            })
    if trt_layer_rows:
        with (RESULT_DIR / "tensorrt_layer_precision_summary.csv").open(
            "w", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(trt_layer_rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(trt_layer_rows)


from src.visualization import (
    plot_carla_metrics_comparison,
    plot_negative_sampler_ablation as plot_sampler_ablation,
    plot_precision_matrix,
    plot_latency_comparison,
    plot_optimization_fidelity,
    plot_resource_heatmaps,
    plot_power_energy,
    plot_memory_footprint,
    plot_efficiency_throughput,
    plot_thermal_utilization,
    plot_native_gpu_precision,
    plot_speedup_fidelity,
    plot_backend_compatibility,
    draw_buck_converter_schematic,
    plot_bode_deviation,
    plot_paper_bode_degradation,
    plot_carla_architecture,
    apply_publication_style,
)
from src.visualization.analysis import extract_single_component_anomalies


def main():
    apply_publication_style()
    classification, deployment, rejected = collect_results()
    write_tables(classification, deployment, rejected)

    # 1. CARLA performance metrics
    plot_carla_metrics_comparison(
        classification,
        save_paths=[
            PLOT_DIR / "latest_carla_metrics.png",
            PLOT_DIR / "latest_carla_metrics.pdf",
            ROOT / "papers" / "mmas_anomaly_detection" / "figures" / "latest_carla_metrics.pdf",
        ],
    )

    # 2. Negative sampler ablation
    ablation_path = EXPERIMENT / "ablations" / "negative_sampler" / "summary.json"
    if ablation_path.exists():
        ablation_meta = _load(ablation_path)
        summary = ablation_meta["samplers"]
        seeds = ablation_meta["seeds"]
        ablation_root = EXPERIMENT / "ablations" / "negative_sampler"
        individual = {
            sampler: {
                metric: [
                    _load(ablation_root / sampler / f"seed_{seed}" / "metrics.json").get(metric, np.nan)
                    for seed in seeds
                ]
                for metric in ("f1", "recall", "auc_roc", "auc_pr")
            }
            for sampler in ("physics", "heuristic")
        }
        plot_sampler_ablation(
            summary,
            individual_runs=individual,
            seeds=seeds,
            save_paths=[
                PLOT_DIR / "negative_sampler_ablation.png",
                PLOT_DIR / "negative_sampler_ablation.pdf",
                ROOT / "papers" / "mmas_anomaly_detection" / "figures" / "negative_sampler_ablation.pdf",
            ],
        )

    # 3. Precision Matrix
    plot_precision_matrix(
        deployment, rejected,
        save_paths=[
            PLOT_DIR / "jetson_precision_matrix.png",
            PLOT_DIR / "jetson_precision_matrix.pdf",
            ROOT / "papers" / "jsc_edge_deployment" / "figures" / "jetson_precision_matrix.pdf",
        ],
    )

    # 4. Latency comparison
    plot_latency_comparison(
        deployment, rejected,
        save_paths=[
            PLOT_DIR / "jetson_latency_comparison.png",
            PLOT_DIR / "jetson_latency_comparison.pdf",
            ROOT / "papers" / "jsc_edge_deployment" / "figures" / "jetson_latency_comparison.pdf",
        ],
    )

    # 4b. Optimization Fidelity & Accuracy Matrix
    plot_optimization_fidelity(
        deployment, rejected,
        save_paths=[
            PLOT_DIR / "jetson_optimization_fidelity.png",
            PLOT_DIR / "jetson_optimization_fidelity.pdf",
            ROOT / "papers" / "jsc_edge_deployment" / "figures" / "jetson_optimization_fidelity.pdf",
        ],
    )

    # 5. Resource Measurements (Divided into 3 focused 2-panel figures)
    grouped_resources = {}
    for model in MODELS:
        for backend, report_name, _, record in _resource_records(model):
            grouped_resources.setdefault((model, backend), []).append(record)

    if grouped_resources:
        # Combined 6-panel summary
        plot_resource_heatmaps(
            grouped_resources,
            save_paths=[
                PLOT_DIR / "jetson_resource_measurements.png",
                PLOT_DIR / "jetson_resource_measurements.pdf",
            ],
        )
        # Split Figure 1: Power & Energy (2 panels)
        plot_power_energy(
            grouped_resources,
            save_paths=[
                PLOT_DIR / "jetson_power_energy.png",
                PLOT_DIR / "jetson_power_energy.pdf",
                ROOT / "papers" / "jsc_edge_deployment" / "figures" / "jetson_power_energy.pdf",
            ],
        )
        # Split Figure 2: Memory Footprint (2 panels)
        plot_memory_footprint(
            grouped_resources,
            save_paths=[
                PLOT_DIR / "jetson_memory_footprint.png",
                PLOT_DIR / "jetson_memory_footprint.pdf",
                ROOT / "papers" / "jsc_edge_deployment" / "figures" / "jetson_memory_footprint.pdf",
            ],
        )
        # Split Figure 3: Efficiency & Throughput (2 panels)
        plot_efficiency_throughput(
            grouped_resources,
            save_paths=[
                PLOT_DIR / "jetson_efficiency_throughput.png",
                PLOT_DIR / "jetson_efficiency_throughput.pdf",
                ROOT / "papers" / "jsc_edge_deployment" / "figures" / "jetson_efficiency_throughput.pdf",
            ],
        )
        # Thermal & Utilization (3 panels)
        plot_thermal_utilization(
            grouped_resources,
            save_paths=[
                PLOT_DIR / "jetson_utilization_thermal.png",
                PLOT_DIR / "jetson_utilization_thermal.pdf",
                ROOT / "papers" / "jsc_edge_deployment" / "figures" / "jetson_utilization_thermal.pdf",
            ],
        )

    # 6. Native GPU Precisions
    native_records = {}
    for model in MODELS:
        dep_dir = EXPERIMENT / f"carla_{model}" / "deployment"
        fp32_p = dep_dir / "native_gpu_fp32.json"
        fp16_p = dep_dir / "native_gpu_fp16.json"
        native_records[model] = {
            "fp32": {
                "latency_mean_ms": _load(fp32_p)["evaluation"]["latency_per_sample_mean_ms"] if fp32_p.exists() else np.nan,
            },
            "fp16": {
                "latency_mean_ms": _load(fp16_p)["evaluation"]["latency_per_sample_mean_ms"] if fp16_p.exists() else np.nan,
                "disagreement_rate": _load(fp16_p).get("evaluation", {}).get("full_test_anomaly_disagreement_rate_vs_baseline", 0.0) if fp16_p.exists() else 0.0,
            },
        }
    plot_native_gpu_precision(
        native_records,
        save_paths=[
            PLOT_DIR / "native_gpu_precision_comparison.png",
            PLOT_DIR / "native_gpu_precision_comparison.pdf",
            ROOT / "papers" / "jsc_edge_deployment" / "figures" / "native_gpu_precision_comparison.pdf",
        ],
    )

    # 7. Buck Converter Schematic
    draw_buck_converter_schematic(
        save_paths=[
            PLOT_DIR / "buck_converter_schematic.png",
            PLOT_DIR / "buck_converter_schematic.pdf",
            ROOT / "papers" / "mmas_anomaly_detection" / "figures" / "buck_converter_schematic.png",
            ROOT / "papers" / "mmas_anomaly_detection" / "figures" / "buck_converter_schematic.pdf",
        ]
    )

    # 8. Single- and Multi-Component Bode Degradation Plots
    plot_paper_bode_degradation(
        save_paths=[
            PLOT_DIR / "component_degradation_bode.png",
            PLOT_DIR / "component_degradation_bode.pdf",
            ROOT / "papers" / "mmas_anomaly_detection" / "figures" / "component_degradation_bode.png",
            ROOT / "papers" / "mmas_anomaly_detection" / "figures" / "component_degradation_bode.pdf",
        ]
    )

    plot_bode_deviation(
        "Cout",
        save_paths=[
            PLOT_DIR / "anomaly_deviation_Cout.png",
            PLOT_DIR / "anomaly_deviation_Cout.pdf",
            ROOT / "papers" / "mmas_anomaly_detection" / "figures" / "anomaly_deviation_Cout.png",
            ROOT / "papers" / "mmas_anomaly_detection" / "figures" / "anomaly_deviation_Cout.pdf",
        ],
    )
    plot_bode_deviation(
        "Esr_C",
        save_paths=[
            PLOT_DIR / "anomaly_deviation_Esr_C.png",
            PLOT_DIR / "anomaly_deviation_Esr_C.pdf",
            ROOT / "papers" / "mmas_anomaly_detection" / "figures" / "anomaly_deviation_Esr_C.png",
            ROOT / "papers" / "mmas_anomaly_detection" / "figures" / "anomaly_deviation_Esr_C.pdf",
        ],
    )

    # 9. CARLA Modular Architecture Diagram
    plot_carla_architecture(
        save_paths=[
            PLOT_DIR / "arch_diagram_carla.png",
            PLOT_DIR / "arch_diagram_carla.pdf",
            ROOT / "papers" / "mmas_anomaly_detection" / "figures" / "arch_diagram_carla.png",
            ROOT / "papers" / "mmas_anomaly_detection" / "figures" / "arch_diagram_carla.pdf",
        ]
    )


if __name__ == "__main__":
    main()
