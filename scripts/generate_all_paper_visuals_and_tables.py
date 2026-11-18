#!/usr/bin/env python3
"""Master Publication Visuals & Multi-Format Tables Generator.

Programmatically generates from scratch all publication-grade `#ffffff` vector/300-DPI
figures (`.pdf` + `.png`) and standalone tables (`.csv`, `.md`, `booktabs` `.tex`)
for:
  1. Variable-Load FRA Power-Converter Anomaly Detection (`papers/variable_load_fra_ad/`)
     - 8 figures (`fig01` .. `fig08`) + mirror in `docs/plots/`
     - 7 standalone tables (`table01` .. `table07`)
  2. JSC Post-Audit Edge Inference & INT8 Quantization Study (`papers/results/jsc_inference_study/`)
     - 2 figures (`jsc_fig01` .. `jsc_fig02`) + mirror in `docs/plots/`
     - 3 standalone tables (`jsc_table01` .. `jsc_table03`)

Frozen directories (`experiments/dataset_01_run_06_corrected/`,
`papers/jsc_edge_deployment/`, `papers/mmas_anomaly_detection/`) are strictly read-only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.baselines import (
    MahalanobisDetector,
    PCAReconstructionDetector,
    RawKNNDetector,
)
from src.evaluation.fault_analysis import compute_prevalence_adjusted_prauc
from src.inference.scoring import EmbeddingAnomalyScorer
from src.visualization import (
    PublicationTableExporter,
    build_jsc_extension_tables,
    build_variable_load_paper_tables,
    plot_fig01_system_and_carla_framework,
    plot_fig02_physics_ssa_fidelity_and_bode,
    plot_fig03_carla_transformer_architecture,
    plot_fig04_ds01_vs_ds02_and_fault_recall,
    plot_fig05_pr_curves_and_10seed_ablation,
    plot_fig06_encoder_backbones_and_scorer_matrix,
    plot_fig07_pca_cliff_and_7level_fra_noise,
    plot_fig08_learning_curves_and_lofo_generalization,
    plot_fig09_attention_and_jacobian_saliency,
    plot_jsc_fig01_int8_recalibration_rescue,
    plot_jsc_fig02_prototype_scorer_pareto,
)


def _compute_or_load_empirical_payloads(
    ds02_results_dir: Path,
    refresh: bool = False,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Extract empirical PSIM Bode quantiles and 10-seed PR curves from `dataset_02`."""
    bode_path = ds02_results_dir / "bode_plot_payload.json"
    pr_path = ds02_results_dir / "pr_curves_pi05_payload.json"
    if bode_path.exists() and pr_path.exists() and not refresh:
        return (
            json.loads(bode_path.read_text(encoding="utf-8")),
            json.loads(pr_path.read_text(encoding="utf-8")),
        )

    # Use CPU for one-shot evaluation of 30 small (N=310) checkpoints to avoid
    # 30x CUDA XLA JIT dot_merger compilations (~300 s overhead vs ~6 s on CPU).
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

    import keras
    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    from scripts.run_dataset02_mlwa_benchmark import (
        _active_fault_components,
        _get_embeddings_batched,
        _train_or_load_model,
        prepare_or_load_stratified_dataset02,
    )
    from src.data.physics_anomaly import BuckConverter

    data_dir = ROOT / "data" / "buck" / "buck_data_v2" / "dataset_02"
    frozen_exp = ROOT / "experiments" / "dataset_01_run_06_corrected"
    study_exp = ROOT / "experiments" / "dataset_02_mlwa_study"
    cache_dir = ROOT / "cache"

    dataset = prepare_or_load_stratified_dataset02(data_dir, study_exp, cache_dir)
    splits = dataset._splits
    frequencies = np.asarray(dataset.frequencies, dtype=np.float64)
    healthy_raw = dataset._raw_data[splits["train_idx"]]  # (300, 101, 2): [mag_dB, phase_deg]

    fault_examples: Dict[str, Dict[str, Any]] = {}
    comp_tex = {
        "Cout": r"$C_{\mathrm{out}}$",
        "Esr_C": r"$\mathrm{ESR}_C$",
        "Esr_L": r"$\mathrm{ESR}_L$",
        "Rds_1": r"$R_{\mathrm{ds},1}$",
    }
    test_idx = splits["test_idx"]
    for comp, tex_name in comp_tex.items():
        cands = []
        for idx in test_idx:
            v = dataset._metadata[int(idx)].variations
            if _active_fault_components(v) == [comp]:
                r_mult = 1.0 + float(v.get("Rout", 0.0)) / 100.0
                cands.append((abs(r_mult - 1.0), int(idx), float(v.get(comp, 0.0))))
        if cands:
            cands.sort(key=lambda t: t[0])
            _, chosen_idx, pct_val = cands[0]
            curve = dataset._raw_data[chosen_idx]
            fault_examples[comp] = {
                "sample_index": int(chosen_idx),
                "variation_pct": float(pct_val),
                "label": f"Single {tex_name} (${pct_val:+.0f}\\%$)",
                "mag_db": curve[:, 0].astype(float).tolist(),
                "phase_deg": curve[:, 1].astype(float).tolist(),
            }

    bode_payload: Dict[str, Any] = {
        "frequencies_hz": frequencies.tolist(),
        "healthy_mag_p025_db": np.percentile(healthy_raw[:, :, 0], 2.5, axis=0).tolist(),
        "healthy_mag_median_db": np.median(healthy_raw[:, :, 0], axis=0).tolist(),
        "healthy_mag_p975_db": np.percentile(healthy_raw[:, :, 0], 97.5, axis=0).tolist(),
        "healthy_phase_p025_deg": np.percentile(healthy_raw[:, :, 1], 2.5, axis=0).tolist(),
        "healthy_phase_median_deg": np.median(healthy_raw[:, :, 1], axis=0).tolist(),
        "healthy_phase_p975_deg": np.percentile(healthy_raw[:, :, 1], 97.5, axis=0).tolist(),
        "fault_examples": fault_examples,
    }
    bode_path.write_text(json.dumps(bode_payload, indent=2) + "\n", encoding="utf-8")

    # Compute exact prevalence-adjusted PR curves (pi = 0.05) across 10 checkpoints + classical
    tf_cfg = json.loads((frozen_exp / "carla_transformer" / "model_config.json").read_text(encoding="utf-8"))
    ablation_zero = json.loads((ds02_results_dir / "ablation_transformer_10seed_zero_label.json").read_text(encoding="utf-8"))
    seeds_10 = list(range(200042, 200052))
    test_labels = np.asarray(dataset.test_labels, dtype=int)
    ssa_spec = BuckConverter(include_parasitic_drop=True, rds_low=0.0)

    ssa_scores_10: List[np.ndarray] = []
    mse_scores_10: List[np.ndarray] = []
    heur_scores_10: List[np.ndarray] = []

    for seed in seeds_10:
        rec_mdl, _ = _train_or_load_model(
            arch_cfg=tf_cfg,
            regime="tf_recon_only",
            seed=seed,
            run_dir=study_exp / "ablation_transformer" / "recon_only" / f"seed_{seed}",
            dataset=dataset,
            epochs=150,
            batch_size=64,
            contrastive_weight_override=0.0,
            use_physics=False,
        )
        mse_scores_10.append(np.asarray(rec_mdl.compute_reconstruction_error(dataset.test_data), dtype=np.float64))
        keras.backend.clear_session()

        heur_mdl, _ = _train_or_load_model(
            arch_cfg=tf_cfg,
            regime="tf_heuristic",
            seed=seed,
            run_dir=study_exp / "ablation_transformer" / "heuristic" / f"seed_{seed}",
            dataset=dataset,
            epochs=150,
            batch_size=64,
            use_physics=False,
        )
        tr_h = _get_embeddings_batched(heur_mdl, dataset.train_data, key="projection")
        te_h = _get_embeddings_batched(heur_mdl, dataset.test_data, key="projection")
        sc_h = EmbeddingAnomalyScorer(tr_h, method="knn", k=5, normalize=True)
        heur_scores_10.append(np.asarray(sc_h.score(te_h), dtype=np.float64))
        keras.backend.clear_session()

        ssa_mdl, _ = _train_or_load_model(
            arch_cfg=tf_cfg,
            regime="tf_physics_ssa_fixed_load",
            seed=seed,
            run_dir=study_exp / "ablation_transformer" / "physics_ssa_fixed_load" / f"seed_{seed}",
            dataset=dataset,
            epochs=150,
            batch_size=64,
            converter_spec=ssa_spec,
            use_physics=True,
            adapt_load=False,
            strict_margin=False,
        )
        tr_s = _get_embeddings_batched(ssa_mdl, dataset.train_data, key="projection")
        te_s = _get_embeddings_batched(ssa_mdl, dataset.test_data, key="projection")
        sc_s = EmbeddingAnomalyScorer(tr_s, method="knn", k=5, normalize=True)
        ssa_scores_10.append(np.asarray(sc_s.score(te_s), dtype=np.float64))
        keras.backend.clear_session()

    pca10 = PCAReconstructionDetector(n_components=10).fit(dataset.train_data)
    maha_lw = MahalanobisDetector(ridge=None).fit(dataset.train_data)
    knn5_raw = RawKNNDetector(n_neighbors=5).fit(dataset.train_data)

    def _prevalence_pr_curve(scores_1d: np.ndarray, pi: float = 0.05) -> Dict[str, Any]:
        fpr, tpr, _ = roc_curve(test_labels, scores_1d)
        denom = pi * tpr + (1.0 - pi) * fpr
        prec_pi = np.where(denom > 0.0, (pi * tpr) / np.maximum(denom, 1e-15), 1.0)
        prec_pi[0] = 1.0
        ap_pi05 = float(compute_prevalence_adjusted_prauc(test_labels, scores_1d, target_prevalence=pi))
        return {
            "recall": tpr.astype(float).tolist(),
            "precision_pi05": prec_pi.astype(float).tolist(),
            "ap_pi05": ap_pi05,
        }

    def _median_seed_curve(s_list: Sequence[np.ndarray], reg_stats: Dict[str, Any]) -> Dict[str, Any]:
        aps = [_prevalence_pr_curve(s, 0.05)["ap_pi05"] for s in s_list]
        med_idx = int(np.argsort(aps)[len(aps) // 2])
        c = _prevalence_pr_curve(s_list[med_idx], 0.05)
        c["ap_pi05"] = float(reg_stats["auc_pr_pi05"]["mean"])
        c["std_pi05"] = float(reg_stats["auc_pr_pi05"]["sample_std"])
        return c

    pr_payload: Dict[str, Any] = {
        "CARLA-Transformer (SSA)": _median_seed_curve(
            ssa_scores_10, ablation_zero["regimes"]["tf_physics_ssa_fixed_load"]
        ),
        "PCA Reconstruction (d=10)": _prevalence_pr_curve(pca10.score(dataset.test_data), 0.05),
        "Mahalanobis (Ledoit-Wolf)": _prevalence_pr_curve(maha_lw.score(dataset.test_data), 0.05),
        "AE-Transformer (MSE)": _median_seed_curve(
            mse_scores_10, ablation_zero["regimes"]["tf_recon_only_mse"]
        ),
        "CARLA-Transformer (Heuristic)": _median_seed_curve(
            heur_scores_10, ablation_zero["regimes"]["tf_heuristic"]
        ),
        "Raw k-NN (k=5)": _prevalence_pr_curve(knn5_raw.score(dataset.test_data), 0.05),
    }
    pr_path.write_text(json.dumps(pr_payload, indent=2) + "\n", encoding="utf-8")
    return bode_payload, pr_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate all publication figures and tables.")
    parser.add_argument(
        "--refresh-payloads",
        action="store_true",
        help="Re-extract Bode and PR payloads from dataset_02 and checkpoints.",
    )
    args = parser.parse_args()

    ds02_results = ROOT / "papers" / "results" / "mlwa_study_dataset02"
    ds01_results = ROOT / "papers" / "results" / "mlwa_study"
    jsc_results = ROOT / "papers" / "results" / "jsc_inference_study"
    jetson_csv = ROOT / "papers" / "results" / "jetson_deployment_results.csv"

    vl_paper_dir = ROOT / "papers" / "variable_load_fra_ad"
    vl_fig_dir = vl_paper_dir / "figures"
    vl_tab_dir = vl_paper_dir / "tables"
    jsc_fig_dir = jsc_results / "figures"
    jsc_tab_dir = jsc_results / "tables"
    docs_plots_dir = ROOT / "docs" / "plots"

    for d in (vl_fig_dir, vl_tab_dir, jsc_fig_dir, jsc_tab_dir, docs_plots_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Remove legacy / superseded figure filenames before writing clean 9-figure set
    for old_file in vl_fig_dir.glob("mlwa_ds02_*"):
        old_file.unlink()
    for stale_pattern in (
        "fig03_ds01_*",
        "fig04_pr_*",
        "fig05_encoder_*",
        "fig06_pca_*",
        "fig07_learning_*",
        "fig08_attention_*",
    ):
        for folder in (vl_fig_dir, docs_plots_dir):
            for old_file in folder.glob(stale_pattern):
                old_file.unlink()

    print("[1/4] Exporting 7 Variable-Load & 3 JSC standalone multi-format tables (.csv/.md/.tex)...")
    exporter = PublicationTableExporter()
    vl_tables = build_variable_load_paper_tables(ds02_results, ds01_results)
    for spec in vl_tables:
        exporter.export_all_formats(spec, vl_tab_dir)
        print(f"  -> {spec.table_id}.{{csv,md,tex}}")

    jsc_tables = build_jsc_extension_tables(jsc_results, jetson_csv)
    for spec in jsc_tables:
        exporter.export_all_formats(spec, jsc_tab_dir)
        print(f"  -> {spec.table_id}.{{csv,md,tex}}")

    print("[2/4] Loading / computing empirical PSIM Bode & 10-seed PR curve payloads...")
    bode_payload, pr_payload = _compute_or_load_empirical_payloads(
        ds02_results, refresh=args.refresh_payloads
    )

    print("[3/4] Rendering 9 white-background Variable-Load IEEE/Elsevier figures (.pdf + .png)...")
    phys_ds02 = json.loads((ds02_results / "data_physics_audit.json").read_text(encoding="utf-8"))
    phys_ds01 = json.loads((ds01_results / "data_physics_audit.json").read_text(encoding="utf-8"))
    ds01_zero = json.loads((ds01_results / "zero_label_summary.json").read_text(encoding="utf-8"))
    ds02_base = pd.read_csv(ds02_results / "zero_label_baselines.csv")
    ablation_10 = json.loads((ds02_results / "ablation_transformer_10seed_zero_label.json").read_text(encoding="utf-8"))
    ablation_full = json.loads((ds02_results / "ablation_transformer_10seed_summary.json").read_text(encoding="utf-8"))
    enc_5 = json.loads((ds02_results / "encoder_comparison_ssa_fixed_summary.json").read_text(encoding="utf-8"))
    latent_ctrl = json.loads((ds02_results / "latent_controls_summary.json").read_text(encoding="utf-8"))
    pca_frag = json.loads((ds02_results / "pca_subspace_fragility_audit.json").read_text(encoding="utf-8"))
    fra_sweep = json.loads((ds02_results / "fra_noise_sweep_dataset02_summary.json").read_text(encoding="utf-8"))
    lc_sum = json.loads((ds02_results / "learning_curves_summary.json").read_text(encoding="utf-8"))
    lofo_5 = json.loads((ds02_results / "lofo_transformer_ssa_fixed_5seed_summary.json").read_text(encoding="utf-8"))
    attn_aud = json.loads((ds02_results / "transformer_attention_audit.json").read_text(encoding="utf-8"))

    def _dual_paths(stem: str, primary_dir: Path) -> List[Path]:
        return [
            primary_dir / f"{stem}.pdf",
            primary_dir / f"{stem}.png",
            docs_plots_dir / f"{stem}.pdf",
            docs_plots_dir / f"{stem}.png",
        ]

    vl_jobs = [
        ("fig01_system_and_carla_framework", lambda p: plot_fig01_system_and_carla_framework(save_paths=p)),
        (
            "fig02_physics_ssa_fidelity_and_bode",
            lambda p: plot_fig02_physics_ssa_fidelity_and_bode(bode_payload, phys_ds02, phys_ds01, save_paths=p),
        ),
        (
            "fig03_carla_transformer_architecture",
            lambda p: plot_fig03_carla_transformer_architecture(save_paths=p),
        ),
        (
            "fig04_ds01_vs_ds02_and_fault_recall",
            lambda p: plot_fig04_ds01_vs_ds02_and_fault_recall(ds01_zero, ds02_base, ablation_10, enc_5, save_paths=p),
        ),
        (
            "fig05_pr_curves_and_10seed_ablation",
            lambda p: plot_fig05_pr_curves_and_10seed_ablation(pr_payload, ablation_10, ds02_base, save_paths=p),
        ),
        (
            "fig06_encoder_backbones_and_scorer_matrix",
            lambda p: plot_fig06_encoder_backbones_and_scorer_matrix(enc_5, latent_ctrl, save_paths=p),
        ),
        (
            "fig07_pca_cliff_and_7level_fra_noise",
            lambda p: plot_fig07_pca_cliff_and_7level_fra_noise(pca_frag, ablation_full, fra_sweep, save_paths=p),
        ),
        (
            "fig08_learning_curves_and_lofo_generalization",
            lambda p: plot_fig08_learning_curves_and_lofo_generalization(lc_sum, lofo_5, save_paths=p),
        ),
        (
            "fig09_attention_and_jacobian_saliency",
            lambda p: plot_fig09_attention_and_jacobian_saliency(attn_aud, save_paths=p),
        ),
    ]
    for stem, builder in vl_jobs:
        fig = builder(_dual_paths(stem, vl_fig_dir))
        plt.close(fig)
        print(f"  -> {stem}.{{pdf,png}}")

    print("[4/4] Rendering 2 white-background JSC Extension figures (.pdf + .png)...")
    int8_df = pd.read_csv(jsc_results / "int8_recalibration_rescue.csv")
    conv_p = pd.read_csv(jsc_results / "scorer_pareto_conv1d.csv")
    mlp_p = pd.read_csv(jsc_results / "scorer_pareto_mlp.csv")

    jf1 = plot_jsc_fig01_int8_recalibration_rescue(
        int8_df=int8_df,
        save_paths=_dual_paths("jsc_fig01_int8_recalibration_rescue", jsc_fig_dir),
    )
    plt.close(jf1)
    print("  -> jsc_fig01_int8_recalibration_rescue.{pdf,png}")

    jf2 = plot_jsc_fig02_prototype_scorer_pareto(
        conv1d_pareto_df=conv_p,
        mlp_pareto_df=mlp_p,
        save_paths=_dual_paths("jsc_fig02_prototype_scorer_pareto", jsc_fig_dir),
    )
    plt.close(jf2)
    print("  -> jsc_fig02_prototype_scorer_pareto.{pdf,png}")

    print("\n[OK] All 10 publication figures (.pdf + .png) and 10 multi-format tables (.csv + .md + .tex) generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
