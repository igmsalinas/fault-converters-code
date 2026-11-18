"""
Edge Deployment and Hardware Benchmark Visualization Module
============================================================

Provides publication-grade visualizations for edge accelerator benchmarks (Jetson Nano,
TensorRT, TFLite, ONNX, Keras GPU), including runtime-precision matrices with fidelity
rejection badges, log-scale latency comparisons, and multi-panel resource/thermal heatmaps.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .style import apply_publication_style, adaptive_contrast, save_figure, RUNTIME_COLORS

MODELS = ["conv1d", "gru", "lstm", "mlp", "transformer"]
MODEL_DISPLAY = {
    "conv1d": "Conv1D",
    "gru": "GRU",
    "lstm": "LSTM",
    "mlp": "MLP",
    "transformer": "Transformer",
}

RESOURCE_BACKENDS = [
    "Keras CPU (FP32)",
    "ONNX CPU (FP32)",
    "ONNX CPU (FP16)",
    "TFLite (FP32)",
    "TFLite (FP16)",
    "Keras GPU (FP32)",
    "Keras GPU (FP16)",
    "TensorRT (FP32)",
    "TensorRT (FP16)",
]


def _canonical_key(name: str) -> str:
    """Normalize runtime/backend strings to a canonical format."""
    s = name.lower().replace("_", " ").replace("-", " ").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace(" (cpu)", "").replace(" (gpu)", "").replace("cpu", "").replace("gpu", "").strip()
    # Normalize keras, onnx, tflite, tensorrt/trt + fp32/fp16/int8
    is_gpu = "gpu" in name.lower()
    is_cpu = "cpu" in name.lower()
    for prec in ["fp32", "fp16", "int8"]:
        if prec in s:
            if "keras" in s:
                return f"keras_gpu_{prec}" if is_gpu else f"keras_cpu_{prec}"
            if "tensorrt" in s or "trt" in s:
                return f"tensorrt_{prec}"
            if "onnx" in s:
                return f"onnx_{prec}"
            if "tflite" in s:
                return f"tflite_{prec}"
    if "keras" in s:
        return "keras_gpu_fp32" if is_gpu else "keras_cpu_fp32"
    return s


def _lookup_backend_record(data_map: Dict[str, Any], backend_name: str) -> Optional[Dict[str, Any]]:
    """Lookup a backend record with robust fuzzy/canonical normalization."""
    if not isinstance(data_map, dict):
        return None
    if backend_name in data_map:
        return data_map[backend_name]
    
    target_c = _canonical_key(backend_name)
    for k, v in data_map.items():
        if _canonical_key(k) == target_c:
            return v
    return None


def plot_precision_matrix(
    deployment: Dict[str, Any],
    rejected: Dict[str, Any],
    save_paths: Optional[Sequence[Union[str, Path]]] = None,
) -> plt.Figure:
    """Plot comprehensive 45-cell Jetson Nano latency matrix across all models and runtimes."""
    apply_publication_style()

    backends = [
        "Keras FP32 (CPU)",
        "Keras GPU FP32", "Keras GPU FP16",
        "TFLite FP32", "TFLite FP16",
        "ONNX FP32 (CPU)", "ONNX FP16 (CPU)",
        "TensorRT FP32 (GPU)", "TensorRT FP16 (GPU)",
    ]
    labels = [
        "Keras CPU", "Keras GPU FP32", "Keras GPU FP16",
        "TFLite FP32", "TFLite FP16",
        "ONNX FP32", "ONNX FP16",
        "TRT FP32", "TRT FP16",
    ]

    values = np.full((len(backends), len(MODELS)), np.nan)
    rejected_cells = set()
    for row, backend in enumerate(backends):
        for col, model in enumerate(MODELS):
            dep_record = _lookup_backend_record(deployment.get(model, {}), backend)
            rej_record = _lookup_backend_record(rejected.get(model, {}), backend)

            lat = (
                (dep_record.get("performance", {}) if dep_record else {}).get("latency_mean_ms")
                or (dep_record.get("evaluation", {}) if dep_record else {}).get("latency_per_sample_mean_ms")
                or (rej_record.get("latency_mean_ms") if rej_record else None)
                or (rej_record.get("latency_per_sample_mean_ms") if rej_record else None)
            )
            if lat is not None:
                values[row, col] = lat

            # Check if this configuration is fidelity-rejected
            dis = (
                (rej_record.get("anomaly_disagreement_rate_vs_baseline") if rej_record else None)
                or (dep_record.get("evaluation", {}).get("anomaly_disagreement_rate_vs_baseline") if dep_record else 0.0)
            )
            if rej_record is not None or (dis is not None and dis > 0.01):
                rejected_cells.add((row, col))

    masked = np.ma.masked_invalid(np.log10(values))
    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    image = ax.imshow(masked, aspect="auto", cmap="Blues_r")

    v_min = np.nanmin(masked) if np.any(np.isfinite(masked)) else 0
    v_max = np.nanmax(masked) if np.any(np.isfinite(masked)) else 1

    for row in range(len(backends)):
        for col in range(len(MODELS)):
            is_rej = (row, col) in rejected_cells
            if np.isfinite(values[row, col]):
                val = values[row, col]
                txt_color = adaptive_contrast(np.log10(val), v_min, v_max, cmap="Blues_r")
                suffix = "†" if is_rej else ""
                txt = f"{val:.1f}{suffix} ms" if val >= 10 else f"{val:.2f}{suffix} ms"
                if is_rej:
                    # Overlay subtle red hatch patch to indicate fidelity rejection
                    hatch_patch = Rectangle(
                        (col - 0.5, row - 0.5), 1.0, 1.0, fill=False,
                        hatch="//", edgecolor="#b33939", linewidth=1.2, alpha=0.6, zorder=5
                    )
                    ax.add_patch(hatch_patch)
                ax.text(col, row, txt, ha="center", va="center",
                        fontsize=9.5, fontweight="bold", color=txt_color, zorder=6)
            else:
                ax.text(col, row, "—", ha="center", va="center",
                        fontsize=10.0, color="#888888")

    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([MODEL_DISPLAY[m] for m in MODELS], fontsize=11.5, fontweight="bold")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11.0, fontweight="bold")

    ax.grid(False)
    ax.set_xticks(np.arange(-0.5, len(MODELS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="#ffffff", linestyle="-", linewidth=2.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(image, ax=ax, pad=0.025)
    cbar.set_label("Batch-1 Latency (ms, $\\log_{10}$ scale)", fontsize=11.0, fontweight="bold")
    cbar.ax.tick_params(labelsize=9.5)
    ax.set_title("Jetson Nano Batch-1 Inference Latency by Runtime & Precision\n(† Red Hatching Denotes Fidelity-Rejected Paths)",
                 fontsize=12.0, fontweight="bold", pad=14)
    fig.tight_layout()

    save_figure(fig, save_paths)
    return fig


def plot_latency_comparison(
    deployment: Dict[str, Any],
    rejected: Dict[str, Any],
    save_paths: Optional[Sequence[Union[str, Path]]] = None,
) -> plt.Figure:
    """Plot grouped batch-1 latency comparisons across CPU and TensorRT precisions."""
    apply_publication_style()

    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    x = np.arange(len(MODELS))
    width = 0.26
    keys = ["Keras FP32 (CPU)", "TensorRT FP32 (GPU)", "TensorRT FP16 (GPU)"]
    labels = ["Keras CPU (FP32)", "TensorRT FP32 (GPU)", "TensorRT FP16 (GPU)"]
    colors = ["#2b4c6f", "#1b7a6e", "#b33939"]

    for offset, (key, label, color) in enumerate(zip(keys, labels, colors)):
        values = []
        for model in MODELS:
            item = deployment.get(model, {}).get(key, {})
            lat = item.get("performance", {}).get("latency_mean_ms")
            if lat is None:
                rej = rejected.get(model, {}).get(key, {})
                lat = rej.get("latency_mean_ms") or rej.get("latency_per_sample_mean_ms")
            values.append(lat if lat is not None else np.nan)

        bars = ax.bar(
            x + (offset - 1) * width, values, width, label=label,
            color=color, edgecolor="#222222", linewidth=0.8,
        )
        for bar, val in zip(bars, values):
            if np.isfinite(val):
                txt = f"{val:.1f} ms" if val >= 10 else f"{val:.2f} ms"
                ax.text(
                    bar.get_x() + bar.get_width() / 2, val * 1.12, txt,
                    ha="left", va="bottom", rotation=28, fontsize=8.0, fontweight="bold",
                    color="#111111",
                )

    # Speedup callouts for recurrent models
    ax.annotate(
        r"$\mathbf{43\times\ Speedup}$",
        xy=(1.0, 32.0), xytext=(1.0, 160.0),
        arrowprops=dict(facecolor="#1b7a6e", edgecolor="#1b7a6e", arrowstyle="->", lw=1.4),
        fontsize=9.0, fontweight="bold", color="#1b7a6e", ha="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#edf7f5", edgecolor="#1b7a6e", alpha=0.95)
    )
    ax.annotate(
        r"$\mathbf{52\times\ Speedup}$",
        xy=(2.0, 22.0), xytext=(2.0, 120.0),
        arrowprops=dict(facecolor="#1b7a6e", edgecolor="#1b7a6e", arrowstyle="->", lw=1.4),
        fontsize=9.0, fontweight="bold", color="#1b7a6e", ha="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#edf7f5", edgecolor="#1b7a6e", alpha=0.95)
    )

    ax.set_yscale("log")
    ax.set_ylabel("Batch-1 Latency (ms, log scale)", fontsize=11.5, fontweight="bold")
    ax.set_ylim(0.2, 7000)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY[m] for m in MODELS], fontsize=11.5, fontweight="bold")
    ax.tick_params(axis="y", labelsize=10.5)
    ax.grid(axis="y", which="both", linestyle="--", alpha=0.35)
    ax.legend(ncol=3, fontsize=10.0, loc="upper right", framealpha=0.95, edgecolor="#cccccc")
    ax.set_title("Jetson Nano Batch-1 Latency by Runtime & Precision",
                 fontsize=12.5, fontweight="bold", pad=14)
    fig.tight_layout()

    save_figure(fig, save_paths)
    return fig


def plot_optimization_fidelity(
    deployment: Dict[str, Any],
    rejected: Optional[Dict[str, Any]] = None,
    save_paths: Optional[Sequence[Union[str, Path]]] = None,
) -> plt.Figure:
    """
    Plot dual heatmaps comparing (a) Test-Set F1-Score and (b) Anomaly Decision Concordance (%)
    across all optimization runtimes and numerical precisions on the Jetson Nano.
    """
    apply_publication_style()
    rejected = rejected or {}

    backends = [
        "Keras FP32 (CPU)",
        "Keras GPU FP32", "Keras GPU FP16",
        "TFLite FP32", "TFLite FP16",
        "ONNX FP32 (CPU)", "ONNX FP16 (CPU)",
        "TensorRT FP32 (GPU)", "TensorRT FP16 (GPU)",
    ]
    labels = [
        "Keras CPU", "Keras GPU FP32", "Keras GPU FP16",
        "TFLite FP32", "TFLite FP16",
        "ONNX FP32", "ONNX FP16",
        "TRT FP32", "TRT FP16",
    ]

    f1_matrix = np.full((len(backends), len(MODELS)), np.nan)
    concordance_matrix = np.full((len(backends), len(MODELS)), np.nan)

    for row, backend in enumerate(backends):
        for col, model in enumerate(MODELS):
            dep_rec = _lookup_backend_record(deployment.get(model, {}), backend)
            rej_rec = _lookup_backend_record(rejected.get(model, {}), backend)

            ev = (dep_rec.get("evaluation") if dep_rec else None) or {}
            f1 = ev.get("f1")
            dis = (rej_rec.get("anomaly_disagreement_rate_vs_baseline") if rej_rec else None)
            if dis is None:
                dis = ev.get("anomaly_disagreement_rate_vs_baseline", 0.0)

            if f1 is not None:
                f1_matrix[row, col] = f1
            if dis is not None and np.isfinite(dis):
                concordance_matrix[row, col] = (1.0 - dis) * 100.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 6.8))

    # --- Panel A: F1 Score Heatmap ---
    im1 = ax1.imshow(f1_matrix, aspect="auto", cmap="Blues", vmin=0.75, vmax=0.92)
    for row in range(len(backends)):
        for col in range(len(MODELS)):
            if np.isfinite(f1_matrix[row, col]):
                val = f1_matrix[row, col]
                txt_color = adaptive_contrast(val, 0.75, 0.92, cmap="Blues")
                ax1.text(col, row, f"{val:.3f}", ha="center", va="center",
                         fontsize=9.5, fontweight="bold", color=txt_color)
            else:
                ax1.text(col, row, "—", ha="center", va="center", fontsize=10.0, color="#888888")

    ax1.set_xticks(range(len(MODELS)))
    ax1.set_xticklabels([MODEL_DISPLAY[m] for m in MODELS], fontsize=11.0, fontweight="bold")
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels, fontsize=10.5, fontweight="bold")
    ax1.grid(False)
    ax1.set_xticks(np.arange(-0.5, len(MODELS), 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax1.grid(which="minor", color="#ffffff", linestyle="-", linewidth=2.0)
    ax1.tick_params(which="minor", bottom=False, left=False)
    ax1.set_title("(a) Test-Set Anomaly F1-Score", fontsize=12.0, fontweight="bold", pad=12)

    cb1 = fig.colorbar(im1, ax=ax1, pad=0.03)
    cb1.set_label("F1-Score", fontsize=10.5, fontweight="bold")
    cb1.ax.tick_params(labelsize=9.0)

    # --- Panel B: Decision Concordance Heatmap ---
    im2 = ax2.imshow(concordance_matrix, aspect="auto", cmap="YlGn", vmin=70.0, vmax=100.0)
    for row in range(len(backends)):
        for col in range(len(MODELS)):
            if np.isfinite(concordance_matrix[row, col]):
                val = concordance_matrix[row, col]
                txt_color = adaptive_contrast(val, 70.0, 100.0, cmap="YlGn")
                txt = f"{val:.1f}%" if val < 99.95 else "100.0%"
                if val < 99.0:
                    # Highlight decision-discordant cells with bold red border
                    ax2.add_patch(
                        Rectangle((col - 0.48, row - 0.48), 0.96, 0.96,
                                  fill=False, edgecolor="#b33939", linewidth=2.4, zorder=5)
                    )
                ax2.text(col, row, txt, ha="center", va="center",
                         fontsize=9.5, fontweight="bold", color=txt_color, zorder=6)
            else:
                ax2.text(col, row, "—", ha="center", va="center", fontsize=10.0, color="#888888")

    ax2.set_xticks(range(len(MODELS)))
    ax2.set_xticklabels([MODEL_DISPLAY[m] for m in MODELS], fontsize=11.0, fontweight="bold")
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels, fontsize=10.5, fontweight="bold")
    ax2.grid(False)
    ax2.set_xticks(np.arange(-0.5, len(MODELS), 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax2.grid(which="minor", color="#ffffff", linestyle="-", linewidth=2.0)
    ax2.tick_params(which="minor", bottom=False, left=False)
    ax2.set_title("(b) Decision Concordance vs. FP32 Baseline (%)\n[Red Borders Indicate < 99% Rejections]",
                  fontsize=12.0, fontweight="bold", pad=12)

    cb2 = fig.colorbar(im2, ax=ax2, pad=0.03)
    cb2.set_label("Concordance (%)", fontsize=10.5, fontweight="bold")
    cb2.ax.tick_params(labelsize=9.0)
    cb2.ax.axhline(99.0, color="#b33939", linestyle="--", linewidth=1.6)

    fig.tight_layout()
    save_figure(fig, save_paths)
    return fig


def _plot_grouped_bar_pair(
    grouped_records: Dict[Tuple[str, str], List[Dict[str, Any]]],
    measures: List[Tuple[str, str, Tuple[str, ...], bool]],
    save_paths: Optional[Sequence[Union[str, Path]]] = None,
    figsize: Tuple[float, float] = (14.0, 4.8),
) -> plt.Figure:
    """Plot exactly 2 resource measurements side by side with a shared top legend."""
    apply_publication_style()

    def _nested_number(payload, *keys):
        cur = payload
        for k in keys:
            if not isinstance(cur, dict):
                return np.nan
            cur = cur.get(k)
        try:
            return float(cur) if cur is not None else np.nan
        except (TypeError, ValueError):
            return np.nan

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    backends = list(RUNTIME_COLORS.keys())
    x = np.arange(len(MODELS))
    n_backends = len(backends)
    bar_w = 0.88 / n_backends

    for ax, (title, ylabel, keys, is_log) in zip(axes, measures):
        for b_idx, backend in enumerate(backends):
            means = []
            stds = []
            for model in MODELS:
                recs = grouped_records.get((model, backend), [])
                vals = [_nested_number(rec, *keys) for rec in recs]
                vals = [v for v in vals if np.isfinite(v)]
                if vals:
                    means.append(float(np.mean(vals)))
                    stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
                else:
                    means.append(0.0 if "cuda_memory" in keys else np.nan)
                    stds.append(0.0)

            x_pos = x + (b_idx - (n_backends - 1) / 2) * bar_w
            ax.bar(
                x_pos, means, bar_w, yerr=stds, capsize=2.0,
                label=backend, color=RUNTIME_COLORS[backend],
                edgecolor="#222222", linewidth=0.6,
                error_kw={"elinewidth": 0.9, "ecolor": "#333333"},
            )

        ax.set_title(title, fontsize=12.0, fontweight="bold", pad=8)
        ax.set_ylabel(ylabel, fontsize=10.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_DISPLAY[m] for m in MODELS], fontsize=10.5, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.tick_params(axis="y", labelsize=9.5)

        if is_log:
            ax.set_yscale("log")
            ax.set_ylim(bottom=1.0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06),
        ncol=len(backends), fontsize=9.0, framealpha=0.95, edgecolor="#cccccc"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    save_figure(fig, save_paths)
    return fig


def plot_power_energy(
    grouped_records: Dict[Tuple[str, str], List[Dict[str, Any]]],
    save_paths: Optional[Sequence[Union[str, Path]]] = None,
) -> plt.Figure:
    """Plot Dynamic Input Power (W) and Total Energy per Inference (mJ) side-by-side."""
    measures = [
        ("Dynamic Input Power", "Power (W)", ("power", "dynamic_input_mean_w"), False),
        ("Total Energy per Inference", "Energy (mJ, log scale)", ("power", "total_energy_per_inference_mj"), True),
    ]
    return _plot_grouped_bar_pair(grouped_records, measures, save_paths=save_paths)


def plot_memory_footprint(
    grouped_records: Dict[Tuple[str, str], List[Dict[str, Any]]],
    save_paths: Optional[Sequence[Union[str, Path]]] = None,
) -> plt.Figure:
    """Plot System Shared RAM Peak Delta (MB) and Process RAM Peak Delta (MB) side-by-side."""
    measures = [
        ("System Shared RAM Peak Delta", "Shared RAM (MB)", ("system_shared_memory", "active_peak_delta_mb"), False),
        ("Process RAM Peak Delta", "RAM Delta (MB)", ("process_ram", "peak_delta_mb"), False),
    ]
    return _plot_grouped_bar_pair(grouped_records, measures, save_paths=save_paths)


def plot_efficiency_throughput(
    grouped_records: Dict[Tuple[str, str], List[Dict[str, Any]]],
    save_paths: Optional[Sequence[Union[str, Path]]] = None,
) -> plt.Figure:
    """Plot CUDA Memory Peak Delta (MB) and Throughput per Input Watt side-by-side."""
    measures = [
        ("CUDA Memory Peak Delta", "CUDA RAM (MB)", ("cuda_memory", "peak_delta_from_context_baseline_mb"), False),
        ("Throughput per Input Watt", "Inferences / s / Watt", ("power", "throughput_per_input_watt"), False),
    ]
    return _plot_grouped_bar_pair(grouped_records, measures, save_paths=save_paths)


def plot_resource_heatmaps(
    grouped_records: Dict[Tuple[str, str], List[Dict[str, Any]]],
    save_paths: Optional[Sequence[Union[str, Path]]] = None,
) -> plt.Figure:
    """
    Plot grouped bar charts for each sustained resource measurement across models and runtimes.
    """
    apply_publication_style()

    def _nested_number(payload, *keys):
        cur = payload
        for k in keys:
            if not isinstance(cur, dict):
                return np.nan
            cur = cur.get(k)
        try:
            return float(cur) if cur is not None else np.nan
        except (TypeError, ValueError):
            return np.nan

    measures = [
        ("Dynamic Input Power", "Power (W)", ("power", "dynamic_input_mean_w"), False),
        ("Total Energy per Inference", "Energy (mJ, log scale)", ("power", "total_energy_per_inference_mj"), True),
        ("Process RAM Peak Delta", "RAM Delta (MB)", ("process_ram", "peak_delta_mb"), False),
        ("System Shared RAM Peak Delta", "Shared RAM (MB)", ("system_shared_memory", "active_peak_delta_mb"), False),
        ("CUDA Memory Peak Delta", "CUDA RAM (MB)", ("cuda_memory", "peak_delta_from_context_baseline_mb"), False),
        ("Throughput per Input Watt", "Inferences / s / Watt", ("power", "throughput_per_input_watt"), False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18.0, 10.5))
    backends = list(RUNTIME_COLORS.keys())
    x = np.arange(len(MODELS))
    n_backends = len(backends)
    bar_w = 0.88 / n_backends

    for ax, (title, ylabel, keys, is_log) in zip(axes.flat, measures):
        for b_idx, backend in enumerate(backends):
            means = []
            stds = []
            for model in MODELS:
                recs = grouped_records.get((model, backend), [])
                vals = [_nested_number(rec, *keys) for rec in recs]
                vals = [v for v in vals if np.isfinite(v)]
                if vals:
                    means.append(float(np.mean(vals)))
                    stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
                else:
                    means.append(0.0 if "cuda_memory" in keys else np.nan)
                    stds.append(0.0)

            x_pos = x + (b_idx - (n_backends - 1) / 2) * bar_w
            ax.bar(
                x_pos, means, bar_w, yerr=stds, capsize=2.0,
                label=backend, color=RUNTIME_COLORS[backend],
                edgecolor="#222222", linewidth=0.6,
                error_kw={"elinewidth": 0.9, "ecolor": "#333333"},
            )

        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_ylabel(ylabel, fontsize=10.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_DISPLAY[m] for m in MODELS], fontsize=10.5, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.tick_params(axis="y", labelsize=9.5)

        if is_log:
            ax.set_yscale("log")
            ax.set_ylim(bottom=1.0)

    # Place a single clean shared legend across the top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.025),
        ncol=len(backends), fontsize=9.5, framealpha=0.95, edgecolor="#cccccc"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    save_figure(fig, save_paths)
    return fig


def plot_thermal_utilization(
    grouped_records: Dict[Tuple[str, str], List[Dict[str, Any]]],
    save_paths: Optional[Sequence[Union[str, Path]]] = None,
) -> plt.Figure:
    """
    Plot grouped bar charts for CPU utilization, GPU utilization, and maximum GPU temperature.
    """
    apply_publication_style()

    def _nested_number(payload, *keys):
        cur = payload
        for k in keys:
            if not isinstance(cur, dict):
                return np.nan
            cur = cur.get(k)
        try:
            return float(cur) if cur is not None else np.nan
        except (TypeError, ValueError):
            return np.nan

    measures = [
        ("Mean CPU Utilization", "Active CPU (%)", ("utilization", "cpu_active_mean_percent"), (0, 108)),
        ("Mean GPU Utilization", "Active GPU (%)", ("utilization", "gpu_active_mean_percent"), (0, 100)),
        ("Maximum GPU Temperature", "Max Temp ($^\\circ$C)", ("temperatures", "gpu", "max_c"), (38, 58)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.2))
    backends = list(RUNTIME_COLORS.keys())
    x = np.arange(len(MODELS))
    n_backends = len(backends)
    bar_w = 0.88 / n_backends

    for ax, (title, ylabel, keys, ylim) in zip(axes, measures):
        for b_idx, backend in enumerate(backends):
            means = []
            stds = []
            for model in MODELS:
                recs = grouped_records.get((model, backend), [])
                vals = [_nested_number(rec, *keys) for rec in recs]
                vals = [v for v in vals if np.isfinite(v)]
                if vals:
                    means.append(float(np.mean(vals)))
                    stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
                else:
                    means.append(np.nan)
                    stds.append(0.0)

            x_pos = x + (b_idx - (n_backends - 1) / 2) * bar_w
            ax.bar(
                x_pos, means, bar_w, yerr=stds, capsize=2.0,
                label=backend, color=RUNTIME_COLORS[backend],
                edgecolor="#222222", linewidth=0.6,
                error_kw={"elinewidth": 0.9, "ecolor": "#333333"},
            )

        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_ylabel(ylabel, fontsize=10.5, fontweight="bold")
        ax.set_ylim(ylim)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_DISPLAY[m] for m in MODELS], fontsize=10.5, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.tick_params(axis="y", labelsize=9.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05),
        ncol=len(backends), fontsize=9.5, framealpha=0.95, edgecolor="#cccccc"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    save_figure(fig, save_paths)
    return fig


def plot_native_gpu_precision(
    native_records: Dict[str, Dict[str, Any]],
    save_paths: Optional[Sequence[Union[str, Path]]] = None,
) -> plt.Figure:
    """Plot Native Keras GPU FP32 vs Mixed-FP16 evaluation on Jetson Nano."""
    apply_publication_style()

    fp32, fp16, rejections = [], [], []
    for model in MODELS:
        rec = native_records.get(model, {})
        val32 = rec.get("fp32", {}).get("latency_mean_ms", np.nan)
        val16 = rec.get("fp16", {}).get("latency_mean_ms", np.nan)
        disc16 = rec.get("fp16", {}).get("disagreement_rate", 0.0)
        fp32.append(val32)
        fp16.append(val16)
        rejections.append(disc16 > 0.01)

    x = np.arange(len(MODELS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    bars32 = ax.bar(x - width / 2, fp32, width, label="Native Keras FP32", color="#1b7a6e",
                    edgecolor="#222222", linewidth=0.8)
    bars16 = ax.bar(x + width / 2, fp16, width, label="Native Keras Mixed-FP16", color="#b33939",
                    edgecolor="#222222", linewidth=0.8)

    for bar, val in zip(bars32, fp32):
        if np.isfinite(val):
            ax.text(bar.get_x() + bar.get_width() / 2, val * 1.14, f"{val:.1f} ms",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#111111")
    for idx, (bar, val32, val16) in enumerate(zip(bars16, fp32, fp16)):
        if np.isfinite(val16):
            ax.text(bar.get_x() + bar.get_width() / 2, val16 * 1.14, f"{val16:.1f} ms",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#111111")
            if np.isfinite(val32) and val32 > 0:
                slowdown = ((val16 - val32) / val32) * 100.0
                ax.text(
                    bar.get_x() + bar.get_width() / 2, val16 * 1.55, f"+{slowdown:.1f}%\n(penalty)",
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#b33939",
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="#fdf0ed", edgecolor="#f5c2b7", lw=0.8)
                )

    ax.set_yscale("log")
    ax.set_ylabel("CUDA-event Latency (ms, log scale)", fontsize=11.5, fontweight="bold")
    ax.set_ylim(1.0, 7000)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY[m] for m in MODELS], fontsize=11.5, fontweight="bold")
    ax.tick_params(axis="y", labelsize=10.5)
    ax.grid(axis="y", which="both", linestyle="--", alpha=0.35)
    ax.legend(fontsize=10.0, loc="upper left", framealpha=0.95, edgecolor="#cccccc")
    ax.set_title("Jetson Nano Native GPU: FP32 vs Mixed-FP16 (Tegra X1, CC 5.3)\n[Software Casting Overhead Without Tensor Cores]",
                 fontsize=12.0, fontweight="bold", pad=14)
    fig.tight_layout()

    save_figure(fig, save_paths)
    return fig


def plot_speedup_fidelity(
    points: Sequence[Tuple[str, str, float, float, float]],
    save_paths: Optional[Sequence[Union[str, Path]]] = None,
) -> plt.Figure:
    """Plot speedup and ROC-AUC fidelity scatter comparison."""
    apply_publication_style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.6))
    x = np.arange(len(MODELS))
    width = 0.34
    colors = {"FP32": "#2a9d8f", "FP16": "#e76f51"}

    for precision, offset in (("FP32", -width / 2), ("FP16", width / 2)):
        by_model = {point[0]: point[2] for point in points if point[1] == precision}
        values = [by_model.get(MODEL_DISPLAY[m], np.nan) for m in MODELS]
        left.bar(x + offset, values, width, color=colors[precision], label=precision,
                 edgecolor="#222222", linewidth=0.8)
    left.set_xticks(x)
    left.set_xticklabels([MODEL_DISPLAY[m] for m in MODELS], fontsize=10.5, fontweight="bold")
    left.set_ylabel("Speedup over Keras CPU (×)", fontsize=11, fontweight="bold")
    left.grid(axis="y", linestyle="--", alpha=0.35)
    left.legend(fontsize=9.5, framealpha=0.95)

    for precision in ("FP32", "FP16"):
        selected = [p for p in points if p[1] == precision]
        right.scatter(
            [p[4] for p in selected], [p[3] for p in selected],
            s=90, color=colors[precision], edgecolor="black", linewidth=1.0, label=precision
        )

    right.set_xscale("log")
    right.set_xlim(0.3, 100)
    right.set_xlabel("CUDA-event Latency (ms, log scale)", fontsize=11, fontweight="bold")
    right.set_ylabel("ROC-AUC", fontsize=11, fontweight="bold")
    right.grid(linestyle="--", alpha=0.35)
    right.legend(fontsize=9.5, loc="lower left", framealpha=0.95)
    fig.tight_layout()

    save_figure(fig, save_paths)
    return fig


def plot_backend_compatibility(
    matrix: np.ndarray,
    save_paths: Optional[Sequence[Union[str, Path]]] = None,
) -> plt.Figure:
    """Plot TensorRT backend compatibility matrix."""
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    from matplotlib.colors import BoundaryNorm, ListedColormap
    cmap = ListedColormap(["#d62828", "#adb5bd", "#2a9d8f"])
    ax.imshow(matrix, aspect="auto", cmap=cmap, norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3))
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([MODEL_DISPLAY[m] for m in MODELS], fontsize=10.5, fontweight="bold")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["TensorRT FP32", "TensorRT FP16"], fontsize=10.5, fontweight="bold")
    text = {0: "Rejected\n(discordance)", 1: "Unavailable", 2: "Validated"}
    for row in range(2):
        for col in range(len(MODELS)):
            ax.text(col, row, text[matrix[row, col]], ha="center", va="center",
                    fontsize=9.5, fontweight="bold",
                    color="white" if matrix[row, col] != 1 else "#222222")
    ax.tick_params(length=0)
    fig.tight_layout()

    save_figure(fig, save_paths)
    return fig
