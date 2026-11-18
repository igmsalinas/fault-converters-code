"""
Model Evaluation and Anomaly Detection Metrics Visualization Module
===================================================================

Provides publication-grade visualizations for anomaly detection metrics (F1, ROC-AUC,
PR-AUC with bootstrap confidence intervals) and negative sampler ablation studies.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import matplotlib.pyplot as plt

from .style import apply_publication_style, save_figure, PUBLICATION_PALETTE

MODELS = ["conv1d", "gru", "lstm", "mlp", "transformer"]
MODEL_DISPLAY = {
    "conv1d": "Conv1D",
    "gru": "GRU",
    "lstm": "LSTM",
    "mlp": "MLP",
    "transformer": "Transformer",
}


def plot_carla_metrics_comparison(
    classification: Dict[str, Any],
    save_paths: Optional[Sequence[Union[str, Path]]] = None,
) -> plt.Figure:
    """
    Plot F1 Score, ROC-AUC, and PR-AUC with 95% bootstrap CI across encoder architectures.
    """
    apply_publication_style()

    labels = [MODEL_DISPLAY[m] for m in MODELS]
    metrics = [("F1 Score", "f1"), ("ROC-AUC", "auc_roc"), ("PR-AUC", "auc_pr")]
    x = np.arange(len(labels))
    width = 0.25
    palette = [PUBLICATION_PALETTE[0], PUBLICATION_PALETTE[1], PUBLICATION_PALETTE[2]]

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for index, ((label, key), color) in enumerate(zip(metrics, palette)):
        values = [classification[m][key] for m in MODELS]
        interval_key = {"f1": "f1", "auc_roc": "auc_roc", "auc_pr": "auc_pr"}[key]
        intervals = [
            (classification[m].get("bootstrap_95_ci") or {}).get(interval_key)
            for m in MODELS
        ]
        if all(interval is not None for interval in intervals):
            errors = np.asarray([
                [max(0.0, value - interval["lower"])
                 for value, interval in zip(values, intervals)],
                [max(0.0, interval["upper"] - value)
                 for value, interval in zip(values, intervals)],
            ])
        else:
            errors = None

        bars = ax.bar(
            x + (index - 1) * width, values, width, label=label,
            yerr=errors, capsize=4.5, color=color, edgecolor="#222222", linewidth=0.8,
            error_kw={"elinewidth": 1.4, "ecolor": "#222222", "capthick": 1.3},
        )
        for b_idx, (bar, value) in enumerate(zip(bars, values)):
            upper_err = errors[1][b_idx] if errors is not None else 0.0
            y_pos = value + upper_err + 0.009
            ax.text(
                bar.get_x() + bar.get_width() / 2, y_pos, f"{value:.3f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                color="#111111",
            )

    ax.set_ylabel("Evaluation Score", fontsize=11.5, fontweight="bold")
    ax.set_ylim(0.75, 1.04)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11.5, fontweight="bold")
    ax.tick_params(axis="y", labelsize=10.5)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(ncol=3, loc="upper left", fontsize=10.0, framealpha=0.95, edgecolor="#cccccc")
    ax.set_title("CARLA Encoder Performance Comparison (95% Bootstrap CI)",
                 fontsize=12.5, fontweight="bold", pad=14)
    fig.tight_layout()

    save_figure(fig, save_paths)
    return fig


def plot_negative_sampler_ablation(
    summary: Dict[str, Any],
    individual_runs: Dict[str, Dict[str, List[float]]],
    seeds: Sequence[int],
    save_paths: Optional[Sequence[Union[str, Path]]] = None,
) -> plt.Figure:
    """
    Plot grouped bar comparison of Negative Sampler Ablation
    (Physics-Informed Small-Signal Perturbations vs Heuristic Random Synthesis).
    """
    apply_publication_style()

    samplers_data = summary.get("samplers", summary)
    metric_defs = [
        ("F1 Score", "f1"),
        ("Recall (Sensitivity)", "recall"),
        ("ROC-AUC", "auc_roc"),
        ("PR-AUC", "auc_pr"),
    ]

    labels = [m[0] for m in metric_defs]
    keys = [m[1] for m in metric_defs]
    x = np.arange(len(keys))
    bar_width = 0.35

    physics_color = "#1f4e79"
    heuristic_color = "#b85d38"

    fig, ax = plt.subplots(figsize=(9.8, 5.4))

    # 1. Physics-Informed Bars
    physics_means = [samplers_data.get("physics", {}).get(k, {}).get("mean", 0.0) for k in keys]
    physics_stds = [samplers_data.get("physics", {}).get(k, {}).get("sample_std", 0.0) for k in keys]

    # 2. Heuristic Bars
    heuristic_means = [samplers_data.get("heuristic", {}).get(k, {}).get("mean", 0.0) for k in keys]
    heuristic_stds = [samplers_data.get("heuristic", {}).get(k, {}).get("sample_std", 0.0) for k in keys]

    bars_phys = ax.bar(
        x - bar_width / 2, physics_means, bar_width, yerr=physics_stds,
        label="Physics-Informed Synthesis ($H(s)$ Component Shift)",
        color=physics_color, edgecolor="#111111", linewidth=0.9, capsize=3.5,
        error_kw={"elinewidth": 1.4, "ecolor": "#111111", "capthick": 1.2},
    )

    bars_heur = ax.bar(
        x + bar_width / 2, heuristic_means, bar_width, yerr=heuristic_stds,
        label="Heuristic Synthesis (Random Gaussian/Freq Noise)",
        color=heuristic_color, edgecolor="#111111", linewidth=0.9, capsize=3.5,
        error_kw={"elinewidth": 1.4, "ecolor": "#111111", "capthick": 1.2},
    )

    # Add numeric callouts and seed dots
    for idx, (m_key, p_bar, h_bar, p_m, p_s, h_m, h_s) in enumerate(
        zip(keys, bars_phys, bars_heur, physics_means, physics_stds, heuristic_means, heuristic_stds)
    ):
        # Physics label
        ax.text(
            p_bar.get_x() + p_bar.get_width() / 2, p_m + p_s + 0.008,
            f"{p_m:.3f}", ha="center", va="bottom",
            fontsize=9.0, fontweight="bold", color=physics_color,
        )
        # Heuristic label
        ax.text(
            h_bar.get_x() + h_bar.get_width() / 2, h_m + h_s + 0.008,
            f"{h_m:.3f}", ha="center", va="bottom",
            fontsize=9.0, fontweight="bold", color=heuristic_color,
        )

        # Plot individual seed dots and connect paired runs
        p_vals = individual_runs.get("physics", {}).get(m_key, [])
        h_vals = individual_runs.get("heuristic", {}).get(m_key, [])
        p_x = p_bar.get_x() + p_bar.get_width() / 2
        h_x = h_bar.get_x() + h_bar.get_width() / 2

        for p_v, h_v in zip(p_vals, h_vals):
            ax.plot([p_x, h_x], [p_v, h_v], color="#888888", linestyle="--", linewidth=1.0, alpha=0.6, zorder=4)

        if p_vals:
            ax.scatter(
                np.full(len(p_vals), p_x),
                p_vals, color="#ffffff", edgecolor=physics_color, s=32, zorder=5, lw=1.3
            )
        if h_vals:
            ax.scatter(
                np.full(len(h_vals), h_x),
                h_vals, color="#ffffff", edgecolor=heuristic_color, s=32, zorder=5, lw=1.3
            )

        # Improvement callout badge
        delta = p_m - h_m
        if delta > 0.001:
            ax.text(
                idx, max(p_m + p_s, h_m + h_s) + 0.038,
                f"+{delta*100:.1f}%", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color="#1d3557",
                bbox=dict(boxstyle="Round,pad=0.2", facecolor="#e8f4f8", edgecolor="#2a9d8f", lw=1.0)
            )

    ax.set_ylabel("Held-out Test Metric", fontsize=11.5, fontweight="bold")
    ax.set_ylim(0.72, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11.0, fontweight="bold")
    ax.tick_params(axis="y", labelsize=10.5)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.95, edgecolor="#cccccc")
    ax.set_title(
        "CARLA Negative Sampler Ablation: Physics-Informed vs. Heuristic Synthesis (3 Seeds)",
        fontsize=12.0, fontweight="bold", pad=14,
    )
    fig.tight_layout()

    save_figure(fig, save_paths)
    return fig
