"""Tests for the modular visualization suite in src.visualization."""

from pathlib import Path
import numpy as np
import pytest
import matplotlib.pyplot as plt

from src.visualization.style import (
    apply_publication_style,
    apply_deck_style,
    adaptive_contrast,
    PUBLICATION_PALETTE,
    RUNTIME_COLORS,
)
from src.visualization.schematic import draw_buck_converter_schematic
from src.visualization.frequency_response import (
    plot_bode_deviation,
    plot_transfer_function_comparison,
)
from src.visualization.architecture import (
    plot_carla_architecture,
    plot_hero_autoencoder,
    HERO_MODELS,
)
from src.visualization.deployment import (
    plot_precision_matrix,
    plot_latency_comparison,
    plot_resource_heatmaps,
    plot_thermal_utilization,
    plot_native_gpu_precision,
    plot_speedup_fidelity,
    plot_backend_compatibility,
)
from src.visualization.metrics import (
    plot_carla_metrics_comparison,
    plot_negative_sampler_ablation,
)


def test_style_application():
    apply_publication_style()
    assert plt.rcParams["figure.facecolor"] == "#ffffff"
    assert plt.rcParams["axes.facecolor"] == "#ffffff"

    apply_deck_style()
    assert plt.rcParams["figure.facecolor"] == "#f7f4ef"

    apply_publication_style()


def test_adaptive_contrast():
    # Value near min -> dark background in magma/viridis -> returns white text
    assert adaptive_contrast(0.1, 0.0, 1.0, light_text="#ffffff", dark_text="#000000") == "#ffffff"
    # Value near max -> bright background -> returns dark text
    assert adaptive_contrast(0.9, 0.0, 1.0, light_text="#ffffff", dark_text="#000000") == "#000000"


def test_schematic_generation(tmp_path: Path):
    out_png = tmp_path / "schematic.png"
    out_pdf = tmp_path / "schematic.pdf"

    fig = draw_buck_converter_schematic(save_paths=[out_png, out_pdf])
    assert fig is not None
    assert out_png.exists()
    assert out_pdf.exists()
    assert out_png.stat().st_size > 0
    assert out_pdf.stat().st_size > 0


def test_frequency_response_plotting(tmp_path: Path):
    # Create mock frequency response files
    freq = np.logspace(2, 5, 101)
    amp_normal = -20 * np.log10(1 + (freq / 1000)**2)
    phase_normal = -np.arctan(freq / 1000) * (180 / np.pi)

    variation_paths = {}
    for var in [-20.0, -10.0, 0.0, 10.0, 20.0]:
        amp = amp_normal + var * 0.1
        phase = phase_normal + var * 0.2
        data = np.column_stack([freq, amp, phase])
        fpath = tmp_path / f"tf_{var:+.0f}.txt"
        np.savetxt(fpath, data, header="Frequency Amplitude Phase")
        variation_paths[var] = fpath

    out_png = tmp_path / "bode_cout.png"
    fig = plot_bode_deviation("Cout", variation_paths, save_paths=[out_png])
    assert fig is not None
    assert out_png.exists()


def test_architecture_diagrams(tmp_path: Path):
    carla_png = tmp_path / "arch_carla.png"
    carla_pdf = tmp_path / "arch_carla.pdf"
    fig = plot_carla_architecture(save_paths=[carla_png, carla_pdf])
    assert fig is not None
    assert carla_png.exists()
    assert carla_pdf.exists()

    for model in HERO_MODELS:
        hero_png = tmp_path / f"arch_hero_{model}.png"
        hfig = plot_hero_autoencoder(model, save_paths=[hero_png])
        assert hfig is not None
        assert hero_png.exists()


def test_deployment_visualizations(tmp_path: Path):
    models = ["conv1d", "gru", "lstm", "mlp", "transformer"]
    backends = ["Keras FP32 (CPU)", "TensorRT FP32 (GPU)", "TensorRT FP16 (GPU)"]

    mock_deployment = {
        m: {
            b: {
                "performance": {"latency_mean_ms": 5.0 + i * 2.0},
                "evaluation": {"auc_roc": 0.95, "f1": 0.92},
            }
            for i, b in enumerate(backends)
        }
        for m in models
    }
    mock_rejected = {
        "transformer": {"TensorRT FP16 (GPU)": {"anomaly_disagreement_rate_vs_baseline": 0.262}},
        "conv1d": {},
        "gru": {},
        "lstm": {},
        "mlp": {},
    }

    out_matrix = tmp_path / "precision_matrix.pdf"
    fig_mat = plot_precision_matrix(mock_deployment, mock_rejected, save_paths=[out_matrix])
    assert fig_mat is not None
    assert out_matrix.exists()

    out_latency = tmp_path / "latency.pdf"
    fig_lat = plot_latency_comparison(mock_deployment, mock_rejected, save_paths=[out_latency])
    assert fig_lat is not None
    assert out_latency.exists()

    mock_grouped = {
        (m, b): [{
            "power": {"dynamic_input_mean_w": 2.0, "total_energy_per_inference_mj": 15.0, "throughput_per_input_watt": 50.0},
            "process_ram": {"peak_delta_mb": 300.0},
            "system_shared_memory": {"active_peak_delta_mb": 200.0},
            "cuda_memory": {"peak_delta_from_context_baseline_mb": 100.0},
            "utilization": {"cpu_active_mean_percent": 90.0, "gpu_active_mean_percent": 5.0},
            "temperatures": {"gpu": {"max_c": 48.0}},
        }]
        for m in models
        for b in list(RUNTIME_COLORS.keys())
    }
    out_res = tmp_path / "resources.pdf"
    fig_res = plot_resource_heatmaps(mock_grouped, save_paths=[out_res])
    assert fig_res is not None
    assert out_res.exists()

    out_therm = tmp_path / "thermal.pdf"
    fig_therm = plot_thermal_utilization(mock_grouped, save_paths=[out_therm])
    assert fig_therm is not None
    assert out_therm.exists()


def test_metrics_visualizations(tmp_path: Path):
    models = ["conv1d", "gru", "lstm", "mlp", "transformer"]
    mock_classification = {
        m: {
            "f1": 0.90 + i * 0.01,
            "auc_roc": 0.94 + i * 0.01,
            "auc_pr": 0.92 + i * 0.01,
            "bootstrap_95_ci": {
                "f1": {"lower": 0.88, "upper": 0.92},
                "auc_roc": {"lower": 0.92, "upper": 0.96},
                "auc_pr": {"lower": 0.90, "upper": 0.94},
            },
        }
        for i, m in enumerate(models)
    }

    out_metrics = tmp_path / "metrics.pdf"
    fig_m = plot_carla_metrics_comparison(mock_classification, save_paths=[out_metrics])
    assert fig_m is not None
    assert out_metrics.exists()

    mock_summary = {
        "samplers": {
            "physics": {
                "f1": {"mean": 0.884, "sample_std": 0.005},
                "recall": {"mean": 0.838, "sample_std": 0.020},
                "auc_roc": {"mean": 0.920, "sample_std": 0.001},
                "auc_pr": {"mean": 0.958, "sample_std": 0.001},
            },
            "heuristic": {
                "f1": {"mean": 0.866, "sample_std": 0.015},
                "recall": {"mean": 0.804, "sample_std": 0.028},
                "auc_roc": {"mean": 0.918, "sample_std": 0.005},
                "auc_pr": {"mean": 0.954, "sample_std": 0.003},
            },
        }
    }
    mock_runs = {
        "physics": {"f1": [0.88, 0.89, 0.885], "recall": [0.84, 0.83, 0.845], "auc_roc": [0.92, 0.921, 0.919], "auc_pr": [0.958, 0.959, 0.957]},
        "heuristic": {"f1": [0.86, 0.87, 0.868], "recall": [0.80, 0.81, 0.802], "auc_roc": [0.917, 0.919, 0.918], "auc_pr": [0.954, 0.955, 0.953]},
    }
    out_abl = tmp_path / "ablation.pdf"
    fig_abl = plot_negative_sampler_ablation(mock_summary, mock_runs, [1, 2, 3], save_paths=[out_abl])
    assert fig_abl is not None
    assert out_abl.exists()
