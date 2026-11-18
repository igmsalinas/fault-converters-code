"""TDD test suite for from-scratch publication figures and tables across both papers."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest

matplotlib.use("Agg")

from src.visualization.jsc_extension_figures import (
    plot_jsc_fig01_int8_recalibration_rescue,
    plot_jsc_fig02_prototype_scorer_pareto,
)
from src.visualization.paper_tables import (
    PublicationTableExporter,
    TableSpec,
    build_jsc_extension_tables,
    build_variable_load_paper_tables,
)
from src.visualization.variable_load_figures import (
    plot_fig01_system_and_carla_framework,
    plot_fig02_physics_ssa_fidelity_and_bode,
    plot_fig03_carla_transformer_architecture,
    plot_fig04_ds01_vs_ds02_and_fault_recall,
    plot_fig05_pr_curves_and_10seed_ablation,
    plot_fig06_encoder_backbones_and_scorer_matrix,
    plot_fig07_pca_cliff_and_7level_fra_noise,
    plot_fig08_learning_curves_and_lofo_generalization,
    plot_fig09_attention_and_jacobian_saliency,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MLWA_DS02_DIR = REPO_ROOT / "papers" / "results" / "mlwa_study_dataset02"
MLWA_DS01_DIR = REPO_ROOT / "papers" / "results" / "mlwa_study"
JSC_STUDY_DIR = REPO_ROOT / "papers" / "results" / "jsc_inference_study"
JETSON_CSV = REPO_ROOT / "papers" / "results" / "jetson_deployment_results.csv"


def test_publication_table_exporter_formats_and_latex_escaping(tmp_path: Path) -> None:
    """Exporter must emit valid CSV, Markdown, and balanced booktabs LaTeX."""
    spec = TableSpec(
        table_id="table_unit_demo",
        caption="Unit test table for FRA & SSA metrics (97.5% quantile).",
        label="tab:unit_demo",
        columns=["Detector_Name", "$\\mathrm{F1}_{\\mathrm{p97.5}}$", "ESR_C Recall (%)"],
        alignments=["l", "c", "r"],
        rows=[
            ["Raw k-NN (k=5)", "0.7112", "3.6%"],
            ["**CARLA-Transformer (SSA)**", "**0.9417 ± 0.0128**", "**89.6%**"],
        ],
        notes="Evaluated on dataset_02 under R_out in [2, 8] Ohm.",
    )
    exporter = PublicationTableExporter()
    paths = exporter.export_all_formats(spec, tmp_path)

    assert paths["csv"].exists() and paths["csv"].stat().st_size > 40
    assert paths["md"].exists() and paths["md"].stat().st_size > 80
    assert paths["tex"].exists() and paths["tex"].stat().st_size > 120

    df = pd.read_csv(paths["csv"])
    assert len(df) == 2
    assert "CARLA-Transformer (SSA)" in df.iloc[1, 0]

    tex = paths["tex"].read_text(encoding="utf-8")
    assert r"\toprule" in tex and r"\midrule" in tex and r"\bottomrule" in tex
    assert r"\textbf{CARLA-Transformer (SSA)}" in tex
    assert r"\pm" in tex
    assert r"\%" in tex
    # Braces in generated LaTeX must be balanced
    assert tex.count("{") == tex.count("}")


def test_build_variable_load_and_jsc_tables_from_real_artifacts(tmp_path: Path) -> None:
    """All 7 variable-load tables and 3 JSC tables must build from persisted JSON/CSVs."""
    vl_specs = build_variable_load_paper_tables(MLWA_DS02_DIR, MLWA_DS01_DIR)
    assert len(vl_specs) == 7

    exporter = PublicationTableExporter()
    for spec in vl_specs:
        out = exporter.export_all_formats(spec, tmp_path / "vl_tables")
        tex = out["tex"].read_text(encoding="utf-8")
        assert tex.count("{") == tex.count("}"), f"Unbalanced braces in {spec.table_id}"
        assert len(spec.rows) >= 3

    # Spot-check exact anchor numbers in Table 2 (main benchmark)
    t2_md = (tmp_path / "vl_tables" / "table02_variable_load_benchmark.md").read_text(
        encoding="utf-8"
    )
    assert "0.9417" in t2_md
    assert "0.9126" in t2_md
    assert "89.6%" in t2_md
    assert "92.4%" in t2_md

    jsc_specs = build_jsc_extension_tables(JSC_STUDY_DIR, JETSON_CSV)
    assert len(jsc_specs) == 3
    for spec in jsc_specs:
        out = exporter.export_all_formats(spec, tmp_path / "jsc_tables")
        tex = out["tex"].read_text(encoding="utf-8")
        assert tex.count("{") == tex.count("}"), f"Unbalanced braces in {spec.table_id}"


def _assert_saved_white_pub_figure(fig: plt.Figure, out_base: Path) -> None:
    assert fig is not None
    # Publication background must be crisp white (#ffffff), never slide beige
    rgba = fig.patch.get_facecolor()
    assert rgba[0] > 0.99 and rgba[1] > 0.99 and rgba[2] > 0.99
    pdf_p = out_base.with_suffix(".pdf")
    png_p = out_base.with_suffix(".png")
    assert pdf_p.exists() and pdf_p.stat().st_size > 4_000
    assert png_p.exists() and png_p.stat().st_size > 20_000
    plt.close(fig)


def test_all_nine_variable_load_figures_render_cleanly(tmp_path: Path) -> None:
    """Each of the 9 variable-load figures must render from artifacts in crisp white style."""
    import json
    import numpy as np

    f1 = plot_fig01_system_and_carla_framework(
        save_paths=[tmp_path / "fig01.pdf", tmp_path / "fig01.png"]
    )
    _assert_saved_white_pub_figure(f1, tmp_path / "fig01")

    f3_arch = plot_fig03_carla_transformer_architecture(
        save_paths=[tmp_path / "fig03.pdf", tmp_path / "fig03.png"]
    )
    _assert_saved_white_pub_figure(f3_arch, tmp_path / "fig03")

    freqs = np.geomspace(100.0, 50_000.0, 101)
    dummy_bode = {
        "frequencies_hz": freqs.tolist(),
        "healthy_mag_p025_db": (34.0 - 20.0 * np.log10(np.maximum(freqs / 1125.0, 1.0))).tolist(),
        "healthy_mag_median_db": (34.0 - 18.0 * np.log10(np.maximum(freqs / 1125.0, 1.0))).tolist(),
        "healthy_mag_p975_db": (36.0 - 16.0 * np.log10(np.maximum(freqs / 1125.0, 1.0))).tolist(),
        "healthy_phase_p025_deg": np.linspace(-5.0, -175.0, 101).tolist(),
        "healthy_phase_median_deg": np.linspace(-3.0, -160.0, 101).tolist(),
        "healthy_phase_p975_deg": np.linspace(-1.0, -140.0, 101).tolist(),
        "fault_examples": {
            "Cout": {
                "label": r"Single $C_{\mathrm{out}}$ ($-42\%$)",
                "mag_db": (34.0 - 17.0 * np.log10(np.maximum(freqs / 1450.0, 1.0))).tolist(),
                "phase_deg": np.linspace(-2.0, -170.0, 101).tolist(),
            },
            "Esr_C": {
                "label": r"Single $\mathrm{ESR}_C$ ($+150\%$)",
                "mag_db": (34.0 - 15.0 * np.log10(np.maximum(freqs / 1125.0, 1.0))).tolist(),
                "phase_deg": np.linspace(-3.0, -120.0, 101).tolist(),
            },
            "Esr_L": {
                "label": r"Single $\mathrm{ESR}_L$ ($+150\%$)",
                "mag_db": (33.3 - 18.0 * np.log10(np.maximum(freqs / 1125.0, 1.0))).tolist(),
                "phase_deg": np.linspace(-4.0, -162.0, 101).tolist(),
            },
            "Rds_1": {
                "label": r"Single $R_{\mathrm{ds},1}$ ($+350\%$)",
                "mag_db": (33.6 - 18.0 * np.log10(np.maximum(freqs / 1125.0, 1.0))).tolist(),
                "phase_deg": np.linspace(-3.5, -161.0, 101).tolist(),
            },
        },
    }
    phys_ds02 = json.loads((MLWA_DS02_DIR / "data_physics_audit.json").read_text())
    phys_ds01 = json.loads((MLWA_DS01_DIR / "data_physics_audit.json").read_text())
    f2 = plot_fig02_physics_ssa_fidelity_and_bode(
        bode_payload=dummy_bode,
        physics_audit_ds02=phys_ds02,
        physics_audit_ds01=phys_ds01,
        save_paths=[tmp_path / "fig02.pdf", tmp_path / "fig02.png"],
    )
    _assert_saved_white_pub_figure(f2, tmp_path / "fig02")

    ds01_zero = json.loads((MLWA_DS01_DIR / "zero_label_summary.json").read_text())
    ds02_base = pd.read_csv(MLWA_DS02_DIR / "zero_label_baselines.csv")
    ablation_10 = json.loads((MLWA_DS02_DIR / "ablation_transformer_10seed_zero_label.json").read_text())
    enc_5 = json.loads((MLWA_DS02_DIR / "encoder_comparison_ssa_fixed_summary.json").read_text())

    f4 = plot_fig04_ds01_vs_ds02_and_fault_recall(
        ds01_zero_summary=ds01_zero,
        ds02_baselines_df=ds02_base,
        ablation_10seed=ablation_10,
        encoder_5seed=enc_5,
        save_paths=[tmp_path / "fig04.pdf", tmp_path / "fig04.png"],
    )
    _assert_saved_white_pub_figure(f4, tmp_path / "fig04")

    dummy_pr = {
        "CARLA-Transformer (SSA)": {
            "recall": [0.0, 0.88, 0.95, 1.0],
            "precision_pi05": [1.0, 0.98, 0.55, 0.05],
            "ap_pi05": 0.9126,
            "std_pi05": 0.0308,
        },
        "PCA Reconstruction (d=10)": {
            "recall": [0.0, 0.60, 0.90, 1.0],
            "precision_pi05": [1.0, 0.80, 0.50, 0.05],
            "ap_pi05": 0.8538,
        },
        "Mahalanobis (Ledoit-Wolf)": {
            "recall": [0.0, 0.55, 0.85, 1.0],
            "precision_pi05": [1.0, 0.72, 0.40, 0.05],
            "ap_pi05": 0.7874,
        },
        "AE-Transformer (MSE)": {
            "recall": [0.0, 0.50, 0.80, 1.0],
            "precision_pi05": [1.0, 0.60, 0.20, 0.05],
            "ap_pi05": 0.6640,
            "std_pi05": 0.0153,
        },
        "CARLA-Transformer (Heuristic)": {
            "recall": [0.0, 0.40, 0.75, 1.0],
            "precision_pi05": [1.0, 0.50, 0.15, 0.05],
            "ap_pi05": 0.5272,
            "std_pi05": 0.0529,
        },
        "Raw k-NN (k=5)": {
            "recall": [0.0, 0.55, 0.80, 1.0],
            "precision_pi05": [1.0, 0.58, 0.18, 0.05],
            "ap_pi05": 0.6268,
        },
    }
    f5 = plot_fig05_pr_curves_and_10seed_ablation(
        pr_curves_payload=dummy_pr,
        ablation_10seed=ablation_10,
        ds02_baselines_df=ds02_base,
        save_paths=[tmp_path / "fig05.pdf", tmp_path / "fig05.png"],
    )
    _assert_saved_white_pub_figure(f5, tmp_path / "fig05")

    latent_ctrl = json.loads((MLWA_DS02_DIR / "latent_controls_summary.json").read_text())
    f6 = plot_fig06_encoder_backbones_and_scorer_matrix(
        encoder_5seed=enc_5,
        latent_controls=latent_ctrl,
        save_paths=[tmp_path / "fig06.pdf", tmp_path / "fig06.png"],
    )
    _assert_saved_white_pub_figure(f6, tmp_path / "fig06")

    pca_frag = json.loads((MLWA_DS02_DIR / "pca_subspace_fragility_audit.json").read_text())
    ablation_full = json.loads((MLWA_DS02_DIR / "ablation_transformer_10seed_summary.json").read_text())
    fra_sweep = json.loads((MLWA_DS02_DIR / "fra_noise_sweep_dataset02_summary.json").read_text())
    f7 = plot_fig07_pca_cliff_and_7level_fra_noise(
        pca_fragility=pca_frag,
        ablation_10seed_full=ablation_full,
        fra_noise_sweep=fra_sweep,
        save_paths=[tmp_path / "fig07.pdf", tmp_path / "fig07.png"],
    )
    _assert_saved_white_pub_figure(f7, tmp_path / "fig07")

    lc_sum = json.loads((MLWA_DS02_DIR / "learning_curves_summary.json").read_text())
    lofo_5 = json.loads((MLWA_DS02_DIR / "lofo_transformer_ssa_fixed_5seed_summary.json").read_text())
    f8 = plot_fig08_learning_curves_and_lofo_generalization(
        learning_curves=lc_sum,
        lofo_5seed=lofo_5,
        save_paths=[tmp_path / "fig08.pdf", tmp_path / "fig08.png"],
    )
    _assert_saved_white_pub_figure(f8, tmp_path / "fig08")

    attn_aud = json.loads((MLWA_DS02_DIR / "transformer_attention_audit.json").read_text())
    f9 = plot_fig09_attention_and_jacobian_saliency(
        attn_audit=attn_aud,
        save_paths=[tmp_path / "fig09.pdf", tmp_path / "fig09.png"],
    )
    _assert_saved_white_pub_figure(f9, tmp_path / "fig09")


def test_jsc_extension_figures_render_cleanly(tmp_path: Path) -> None:
    """Both JSC post-audit quantization rescue and prototype Pareto figures must render."""
    int8_df = pd.read_csv(JSC_STUDY_DIR / "int8_recalibration_rescue.csv")
    conv_df = pd.read_csv(JSC_STUDY_DIR / "scorer_pareto_conv1d.csv")
    mlp_df = pd.read_csv(JSC_STUDY_DIR / "scorer_pareto_mlp.csv")

    jf1 = plot_jsc_fig01_int8_recalibration_rescue(
        int8_df=int8_df,
        save_paths=[tmp_path / "jsc_fig01.pdf", tmp_path / "jsc_fig01.png"],
    )
    _assert_saved_white_pub_figure(jf1, tmp_path / "jsc_fig01")

    jf2 = plot_jsc_fig02_prototype_scorer_pareto(
        conv1d_pareto_df=conv_df,
        mlp_pareto_df=mlp_df,
        save_paths=[tmp_path / "jsc_fig02.pdf", tmp_path / "jsc_fig02.png"],
    )
    _assert_saved_white_pub_figure(jf2, tmp_path / "jsc_fig02")
