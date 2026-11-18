"""
JSC Edge Deployment & Post-Audit Inference Extension Figures
============================================================

Publication-grade `#ffffff` figures illustrating:
1. INT8 quantization rescue via input-clipping saturation, INT8 reference-bank
   recalibration, and M=32 prototype compression across TFLite INT8 and ONNX INT8.
2. Embedding anomaly-scorer latency–memory–accuracy Pareto frontier (4,200x speedup
   and 75x bank compression via k-means prototype banks).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import apply_publication_style, save_figure, style_axis


def plot_jsc_fig01_int8_recalibration_rescue(
    int8_df: pd.DataFrame,
    save_paths: Optional[Sequence[Union[str, Path]]] = None,
) -> plt.Figure:
    """Plot 2-panel INT8 quantization rescue across 5 CARLA backbones and 2 runtimes."""
    apply_publication_style()
    fig, (ax_f1, ax_disc) = plt.subplots(1, 2, figsize=(13.8, 5.4), gridspec_kw={"wspace": 0.24})
    fig.patch.set_facecolor("#ffffff")

    order_pairs = [
        ("conv1d", "tflite_int8", "Conv1D\n(TFLite)"),
        ("conv1d", "onnx_int8", "Conv1D\n(ONNX)"),
        ("mlp", "tflite_int8", "MLP\n(TFLite)"),
        ("mlp", "onnx_int8", "MLP\n(ONNX)"),
        ("transformer", "tflite_int8", "Transf.\n(TFLite)"),
        ("transformer", "onnx_int8", "Transf.\n(ONNX)"),
        ("gru", "tflite_int8", "GRU\n(TFLite)"),
        ("lstm", "onnx_int8", "LSTM\n(ONNX)"),
    ]

    strategies = [
        ("1_naive_fp32_bank_orig_scorer", "Naive FP32 Bank (All-Pos Collapse)", "#b33939"),
        ("3_int8_recal_knn5_zero_label_p95", "INT8 Recal. k-NN (k=5, Zero-p95)", "#1f4e79"),
        ("5_int8_recal_proto32_zero_label_p95", "INT8 Recal. Proto-32 (Zero-p95)", "#2e7d6e"),
        ("4_int8_recal_proto32_val_opt", "INT8 Recal. Proto-32 (Val-Opt)", "#c4842b"),
    ]

    x = np.arange(len(order_pairs), dtype=float)
    width = 0.19

    for idx, (st_key, st_label, color) in enumerate(strategies):
        f1_vals = []
        for arch, bk, _ in order_pairs:
            sub = int8_df[
                (int8_df["architecture"] == arch)
                & (int8_df["backend"] == bk)
                & (int8_df["strategy"] == st_key)
            ]
            f1_vals.append(float(sub.iloc[0]["f1"]) if not sub.empty else 0.0)
        offset = (idx - 1.5) * width
        ax_f1.bar(
            x + offset,
            f1_vals,
            width=width * 0.92,
            color=color,
            edgecolor="#ffffff",
            linewidth=0.6,
            label=st_label,
            zorder=3,
        )

    ax_f1.axhline(
        0.7421,
        color="#b33939",
        ls="--",
        lw=1.2,
        alpha=0.75,
        label="Trivial All-Positive Floor (F1 = 0.742)",
    )
    ax_f1.set_xticks(x)
    ax_f1.set_xticklabels([lbl for _, _, lbl in order_pairs], fontsize=8.8)
    ax_f1.set_ylim(0.55, 0.96)
    ax_f1.set_ylabel("Test F1 Score", fontweight="bold")
    ax_f1.set_title("(a) INT8 Quantization F1 Rescue Across Edge Runtimes", loc="left", pad=10)
    ax_f1.grid(True, axis="y", alpha=0.45, zorder=0)
    style_axis(ax_f1)
    ax_f1.legend(loc="upper right", fontsize=8.0, ncol=1, framealpha=0.96)

    # Panel (b): FP32-vs-INT8 Decision Discordance (%) & Memory Compression
    disc_naive, disc_knn, disc_p32, mem_naive, mem_p32 = [], [], [], [], []
    for arch, bk, _ in order_pairs:
        sub = int8_df[(int8_df["architecture"] == arch) & (int8_df["backend"] == bk)]
        r1 = sub[sub["strategy"] == "1_naive_fp32_bank_orig_scorer"].iloc[0]
        r3 = sub[sub["strategy"] == "3_int8_recal_knn5_zero_label_p95"].iloc[0]
        r5 = sub[sub["strategy"] == "5_int8_recal_proto32_zero_label_p95"].iloc[0]
        disc_naive.append(float(r1["discordance_vs_fp32_pct"]))
        disc_knn.append(float(r3["discordance_vs_fp32_pct"]))
        disc_p32.append(float(r5["discordance_vs_fp32_pct"]))
        mem_naive.append(float(r1["total_detector_mb"]))
        mem_p32.append(float(r5["total_detector_mb"]))

    w2 = 0.25
    ax_disc.bar(
        x - w2,
        disc_naive,
        width=w2 * 0.92,
        color="#b33939",
        label="Naive FP32 Bank Discordance",
        zorder=3,
    )
    ax_disc.bar(
        x,
        disc_knn,
        width=w2 * 0.92,
        color="#1f4e79",
        label="Recalibrated INT8 k-NN (k=5)",
        zorder=3,
    )
    bars_p32 = ax_disc.bar(
        x + w2,
        disc_p32,
        width=w2 * 0.92,
        color="#2e7d6e",
        label="Recalibrated INT8 Proto-32",
        zorder=3,
    )

    for idx_b, bar in enumerate(bars_p32):
        ratio = mem_naive[idx_b] / max(mem_p32[idx_b], 1e-6)
        ax_disc.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 1.3,
            f"{ratio:.1f}x\nsmaller",
            ha="center",
            va="bottom",
            fontsize=7.3,
            fontweight="bold",
            color="#2e7d6e",
        )

    ax_disc.set_xticks(x)
    ax_disc.set_xticklabels([lbl for _, _, lbl in order_pairs], fontsize=8.8)
    ax_disc.set_ylim(0.0, 69.0)
    ax_disc.set_ylabel("Decision Discordance vs. FP32 [%]", fontweight="bold")
    ax_disc.set_title("(b) FP32-to-INT8 Discordance Collapse & Footprint Ratio", loc="left", pad=10)
    ax_disc.grid(True, axis="y", alpha=0.45, zorder=0)
    style_axis(ax_disc)
    ax_disc.legend(loc="upper right", fontsize=8.3, framealpha=0.96)

    save_figure(fig, save_paths)
    return fig


def plot_jsc_fig02_prototype_scorer_pareto(
    conv1d_pareto_df: pd.DataFrame,
    mlp_pareto_df: pd.DataFrame,
    save_paths: Optional[Sequence[Union[str, Path]]] = None,
) -> plt.Figure:
    """Plot 2-panel latency–accuracy Pareto and stacked memory footprint for CARLA scorers."""
    apply_publication_style()
    fig, (ax_lat, ax_mem) = plt.subplots(1, 2, figsize=(14.0, 5.3), gridspec_kw={"wspace": 0.28})
    fig.patch.set_facecolor("#ffffff")

    label_map = {
        "legacy_sklearn_knn_N2419": "Sklearn k-NN (N=2419)",
        "fast_blas_knn_N2419": "BLAS k-NN (N=2419)",
        "proto_knn_M128": "Proto M=128",
        "proto_knn_M64": "Proto M=64",
        "proto_knn_M32": "Proto M=32",
        "proto_knn_M16": "Proto M=16",
        "centroid_M1": "Centroid M=1",
        "ridge_mahalanobis_1e6": "Ridge Mahalanobis",
        "ledoit_wolf_precision": "Ledoit-Wolf",
    }

    for df_arch, arch_name, color, marker in [
        (conv1d_pareto_df, "CARLA-Conv1D", "#1f4e79", "o"),
        (mlp_pareto_df, "CARLA-MLP", "#b85d38", "s"),
    ]:
        proto_order = [
            "legacy_sklearn_knn_N2419",
            "fast_blas_knn_N2419",
            "proto_knn_M128",
            "proto_knn_M64",
            "proto_knn_M32",
            "proto_knn_M16",
            "centroid_M1",
        ]
        sub_p = df_arch.set_index("scorer_variant").loc[proto_order].reset_index()
        ax_lat.plot(
            sub_p["scorer_latency_p50_ms"],
            sub_p["f1_unsup_p95"],
            color=color,
            marker=marker,
            lw=2.1,
            ms=7.0,
            label=f"{arch_name} (k-NN / Prototype Trajectory)",
            zorder=4,
        )
        if arch_name == "CARLA-Conv1D":
            offsets = {
                "legacy_sklearn_knn_N2419": (-6, -15, "right"),
                "fast_blas_knn_N2419": (-18, 9, "center"),
                "proto_knn_M128": (8, 5, "left"),
                "proto_knn_M64": (8, -3, "left"),
                "proto_knn_M32": (9, 1, "left"),
                "centroid_M1": (8, -4, "left"),
            }
            for _, row in sub_p.iterrows():
                vk = str(row["scorer_variant"])
                if vk in offsets:
                    dx, dy, ha_align = offsets[vk]
                    ax_lat.annotate(
                        label_map[vk],
                        xy=(float(row["scorer_latency_p50_ms"]), float(row["f1_unsup_p95"])),
                        xytext=(dx, dy),
                        textcoords="offset points",
                        ha=ha_align,
                        fontsize=7.9,
                        fontweight="bold" if "M128" in vk or "M64" in vk else "normal",
                        color="#1f4e79",
                    )

    ax_lat.set_xscale("log")
    ax_lat.set_xlim(0.008, 650.0)
    ax_lat.set_ylim(0.75, 0.915)
    ax_lat.set_xlabel("Median Scorer Latency $p_{50}$ [ms] (log scale)", fontweight="bold")
    ax_lat.set_ylabel("Zero-Label F1 Score (p95)", fontweight="bold")
    ax_lat.set_title("(a) 4,200x Scorer Speedup via Prototype Compression", loc="left", pad=10)
    ax_lat.grid(True, which="both", alpha=0.45, zorder=0)
    style_axis(ax_lat)
    ax_lat.legend(loc="lower right", fontsize=8.5, framealpha=0.96)

    # Panel (b): Stacked Detector Memory Footprint on CARLA-Conv1D
    mem_order = [
        "fast_blas_knn_N2419",
        "ridge_mahalanobis_1e6",
        "proto_knn_M128",
        "proto_knn_M64",
        "proto_knn_M32",
        "proto_knn_M16",
        "centroid_M1",
    ]
    sub_m = conv1d_pareto_df.set_index("scorer_variant").loc[mem_order].reset_index()
    graph_mb = sub_m["total_detector_mb"].to_numpy(dtype=float) - sub_m["bank_memory_mb"].to_numpy(
        dtype=float
    )
    bank_mb = sub_m["bank_memory_mb"].to_numpy(dtype=float)
    y_pos = np.arange(len(mem_order), dtype=float)

    ax_mem.barh(
        y_pos,
        graph_mb,
        height=0.62,
        color="#4a7c9f",
        label="Encoder + Projection Graph [MB]",
        zorder=3,
    )
    ax_mem.barh(
        y_pos,
        bank_mb,
        left=graph_mb,
        height=0.62,
        color="#c4842b",
        label="Reference Embedding Bank [MB]",
        zorder=3,
    )
    for idx_r, (tot, f1v) in enumerate(zip(sub_m["total_detector_mb"], sub_m["f1_unsup_p95"])):
        ax_mem.text(
            float(tot) + 0.10,
            y_pos[idx_r],
            f"{float(tot):.2f} MB  (F1={float(f1v):.3f})",
            va="center",
            ha="left",
            fontsize=8.3,
            color="#111111",
        )

    ax_mem.set_yticks(y_pos)
    ax_mem.set_yticklabels([label_map[k] for k in mem_order], fontsize=8.8)
    ax_mem.invert_yaxis()
    ax_mem.set_xlim(0.0, 8.2)
    ax_mem.set_xlabel("Total Deployed Detector Memory [MB]", fontweight="bold")
    ax_mem.set_title("(b) CARLA-Conv1D Detector Footprint Breakdown", loc="left", pad=10)
    ax_mem.grid(True, axis="x", alpha=0.45, zorder=0)
    style_axis(ax_mem)
    ax_mem.legend(loc="lower right", fontsize=8.4, framealpha=0.96)

    save_figure(fig, save_paths)
    return fig
