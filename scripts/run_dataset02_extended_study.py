#!/usr/bin/env python3
"""Extended Rigorous MLWA Study on Variable-Load PSIM `dataset_02`.

Executes the three experimental extensions prior to MLWA manuscript drafting:
1. 10-Seed Paired Contrasts (`seeds = 200042..200051`, exact sign-flip p_min = 0.00195):
   - `tf_recon_only_mse`, `tf_recon_only_latent`, `tf_heuristic`,
     `tf_physics_ssa_fixed_load`, and `tf_physics_ssa_load_adapt`,
     plus paired sign-flip tests against deterministic classical baselines.
2. Representation vs. Scorer Controls (`latent` & `projection` spaces, 10 seeds)
   and Raw-Space PCA Subspace Dimension Sweep (`d in {3, 5, 8, 10, 12, 15, 20, 30}`).
3. Empirical Multi-Head Self-Attention & Jacobian Saliency Audit across 10 seeds
   (testing objectively whether Transformer self-attention and score gradients
   concentrate on the ~1.13 kHz LC resonance peak and fault-specific bands).
4. Leak-Free Sample-Complexity Learning Curves vs `N_train in {50, 100, 150, 200, 250, 300}`
   across 5 seeds (`Raw k-NN`, `Mahalanobis LW`, `PCA d=10`, `CARLA-Transformer SSA`).
5. Publication-ready vector PDF & PNG figures under `docs/plots/`.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import keras
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_dataset02_mlwa_benchmark import (
    _active_fault_components,
    _evaluate_deep_model,
    _f1,
    _fault_stratum,
    _get_embeddings_batched,
    _model_weights_sha256,
    _summarize_predictions,
    _train_or_load_model,
    _write_csv,
    prepare_or_load_stratified_dataset02,
)
from src.data.physics_anomaly import BuckConverter
from src.data.preprocessor import DataPreprocessor
from src.evaluation.attribution import (
    DEFAULT_FREQUENCY_BANDS_HZ,
    compute_knn_score_frequency_saliency,
    extract_transformer_attention_maps,
    summarize_frequency_attention,
)
from src.evaluation.baselines import (
    MahalanobisDetector,
    PCAReconstructionDetector,
    RawKNNDetector,
)
from src.evaluation.fault_analysis import (
    inject_fra_measurement_noise,
    paired_sign_flip_test,
)
from src.inference.scoring import EmbeddingAnomalyScorer
from src.models.contrastive_ae import ContrastiveAutoencoder
from src.training.carla_trainer import CARLAConfig, CARLATrainer


def _agg(vals: List[float]) -> Dict[str, Any]:
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "sample_std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        "values": [float(v) for v in arr],
    }


def _eval_zero_label_scores(
    name: str,
    family: str,
    val_norm_scores: np.ndarray,
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    test_variations: List[Dict[str, float]],
) -> Dict[str, Any]:
    th975 = float(np.percentile(val_norm_scores, 97.5))
    th95 = float(np.percentile(val_norm_scores, 95.0))
    r975, _ = _summarize_predictions(
        name, f"{family}_p97.5", test_scores, th975, test_labels, test_variations
    )
    r95, _ = _summarize_predictions(
        name, f"{family}_p95", test_scores, th95, test_labels, test_variations
    )
    return {
        "detector": name,
        "family": family,
        "zero_p975_f1": float(r975["f1"]),
        "zero_p975_precision": float(r975["precision"]),
        "zero_p975_recall": float(r975["recall"]),
        "zero_p95_f1": float(r95["f1"]),
        "zero_p95_precision": float(r95["precision"]),
        "zero_p95_recall": float(r95["recall"]),
        "auc_roc": float(r975["auc_roc"]),
        "auc_pr_test": float(r975["auc_pr_test"]),
        "auc_pr_pi10": float(r975["auc_pr_pi10"]),
        "auc_pr_pi05": float(r975["auc_pr_pi05"]),
        "recall_passive_single": float(r975["recall_passive_single"]),
        "recall_cout_single": float(r975["recall_cout_single"]),
        "recall_esrc_single": float(r975["recall_esrc_single"]),
        "recall_lout_single": float(r975["recall_lout_single"]),
        "recall_esrl_single": float(r975["recall_esrl_single"]),
        "recall_rds1_single": float(r975["recall_rds1_single"]),
        "recall_multi_fault": float(r975["recall_multi_fault"]),
    }


def _train_or_load_subsampled_transformer(
    arch_cfg: Dict[str, Any],
    n_train: int,
    seed: int,
    run_dir: Path,
    train_sub: np.ndarray,
    val_norm: np.ndarray,
    frequencies: np.ndarray,
    scaler: Any,
    use_real_imag: bool,
    converter_spec: BuckConverter,
    epochs: int = 150,
) -> Tuple[ContrastiveAutoencoder, Dict[str, Any]]:
    best_ckpt = run_dir / "checkpoints" / "best"
    prov_file = run_dir / "training_provenance.json"
    if (best_ckpt / "autoencoder.weights.h5").exists() and prov_file.exists():
        prov = json.loads(prov_file.read_text())
        model = ContrastiveAutoencoder.load(str(best_ckpt), config=arch_cfg)
        return model, prov

    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    batch_size = min(64, max(16, len(train_sub) // 2))
    config = CARLAConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=float(arch_cfg["learning_rate"]),
        reconstruction_weight=float(arch_cfg["reconstruction_weight"]),
        contrastive_weight=float(arch_cfg["contrastive_weight"]),
        temperature=float(arch_cfg["temperature"]),
        anomaly_ratio=1.0,
        seed=seed,
        use_physics_anomalies=True,
        resume_from_latest=False,
        scoring_method="knn",
        k_neighbors=5,
        adapt_load=False,
        strict_margin=False,
        checkpoint_dir=str(run_dir / "checkpoints"),
        log_dir=str(run_dir / "logs"),
        early_stopping=True,
        patience=25,
        save_best_only=True,
    )
    trainer = CARLATrainer(config, experiment_name=f"ds02_lc_n{n_train}_{seed}")
    trainer.create_model(
        input_shape=(train_sub.shape[1], train_sub.shape[2]),
        latent_dim=int(arch_cfg["latent_dim"]),
        projection_dim=int(arch_cfg["projection_dim"]),
        encoder_type=str(arch_cfg["encoder_type"]),
        encoder_filters=arch_cfg["encoder_filters"],
        encoder_units=arch_cfg.get("encoder_units", [256, 128, 64]),
        kernel_size=int(arch_cfg["kernel_size"]),
        num_heads=int(arch_cfg["num_heads"]),
        dropout_rate=float(arch_cfg["dropout_rate"]),
    )
    init_sha = _model_weights_sha256(trainer.model)
    trainer.set_physics_context(
        frequencies=frequencies,
        scaler=scaler,
        use_real_imag=use_real_imag,
        converter_spec=converter_spec,
    )
    trainer.setup_training()

    t0 = time.perf_counter()
    history = trainer.train(train_sub, val_norm, verbose=0)
    elapsed = time.perf_counter() - t0

    prov = {
        "n_train": int(n_train),
        "seed": int(seed),
        "batch_size": int(batch_size),
        "initial_model_weights_sha256": init_sha,
        "final_best_weights_sha256": _model_weights_sha256(trainer.model),
        "training_seconds": elapsed,
        "epochs_trained": len(history["loss"]),
    }
    prov_file.write_text(json.dumps(prov, indent=2) + "\n")
    print(f"  [LC N={n_train}] seed={seed} trained in {elapsed:.1f}s ({len(history['loss'])} ep)")
    return trainer.model, prov


def generate_extended_figures(
    out_dir: Path,
    plots_dir: Path,
    frequencies: np.ndarray,
    pr_curves_payload: Dict[str, Any],
    summary_10seed: Dict[str, Any],
    lc_summary: Dict[str, Any],
    attn_audit: Dict[str, Any],
    pca_fragility: Dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    # --- Figure A: Prevalence-adjusted PR curves (pi=5%) + 10-seed Zero-Label F1 / PR-AUC@5% ---
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)

    ax = axes[0]
    style_map = [
        ("CARLA-Transformer (SSA)", "#0072B2", "-", 2.2),
        ("PCA Reconstruction (d=10)", "#009E73", "--", 1.8),
        ("Mahalanobis (Ledoit-Wolf)", "#E69F00", "-.", 1.7),
        ("AE-Transformer (MSE)", "#CC79A7", ":", 1.7),
        ("CARLA-Transformer (Heuristic)", "#D55E00", ":", 1.7),
        ("Raw k-NN (k=5)", "#555555", "--", 1.5),
    ]
    for label, color, ls, lw in style_map:
        if label not in pr_curves_payload:
            continue
        item = pr_curves_payload[label]
        ax.plot(
            item["recall"],
            item["precision_pi05"],
            color=color,
            linestyle=ls,
            linewidth=lw,
            label=f"{label} (AP={item['ap_label']})",
        )
    ax.axhline(0.05, color="#999999", linestyle=":", linewidth=1.0, label="Random (pi=0.05)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision at realistic fault prevalence (pi = 5%)")
    ax.set_title("(a) Precision-Recall under Variable Load (dataset_02)")
    ax.set_xlim(0.0, 1.01)
    ax.set_ylim(0.0, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", frameon=True)

    ax2 = axes[1]
    reg_order = [
        ("tf_physics_ssa_fixed_load", "CARLA-TF\n(SSA)"),
        ("tf_physics_ssa_load_adapt", "CARLA-TF\n(SSA+Load)"),
        ("tf_recon_only_mse", "AE-TF\n(MSE)"),
        ("tf_recon_only_latent", "AE-TF\n(Latent kNN)"),
        ("tf_heuristic", "CARLA-TF\n(Heuristic)"),
    ]
    x = np.arange(len(reg_order))
    w = 0.36
    f1_means = [summary_10seed["regimes"][k]["zero_knn5_p975_f1"]["mean"] for k, _ in reg_order]
    f1_stds = [summary_10seed["regimes"][k]["zero_knn5_p975_f1"]["sample_std"] for k, _ in reg_order]
    pr_means = [summary_10seed["regimes"][k]["zero_knn5_auc_pr_pi05"]["mean"] for k, _ in reg_order]
    pr_stds = [summary_10seed["regimes"][k]["zero_knn5_auc_pr_pi05"]["sample_std"] for k, _ in reg_order]

    ax2.bar(x - w / 2, f1_means, w, yerr=f1_stds, capsize=3, color="#0072B2", alpha=0.88, label="Zero-Label F1 (p97.5)")
    ax2.bar(x + w / 2, pr_means, w, yerr=pr_stds, capsize=3, color="#E69F00", alpha=0.88, label="PR-AUC (pi=5%)")
    ax2.axhline(
        summary_10seed["classical_references"]["pca_d10"]["zero_p975_f1"],
        color="#009E73",
        linestyle="--",
        linewidth=1.2,
        label="PCA (d=10) Zero-Label F1",
    )
    ax2.axhline(
        summary_10seed["classical_references"]["pca_d10"]["auc_pr_pi05"],
        color="#009E73",
        linestyle=":",
        linewidth=1.2,
        label="PCA (d=10) PR-AUC@5%",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels([lbl for _, lbl in reg_order])
    ax2.set_ylim(0.45, 1.02)
    ax2.set_ylabel("Score (mean +/- SD across 10 seeds)")
    ax2.set_title("(b) 10-Seed Paired Ablation (p = 0.002 vs non-physics)")
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.legend(loc="lower left", fontsize=7.5, frameon=True)

    for ext in ("pdf", "png"):
        fig.savefig(plots_dir / f"mlwa_ds02_pr_and_ablation10.{ext}", dpi=300)
    plt.close(fig)

    # --- Figure B: Sample-complexity learning curves vs N_train ---
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), constrained_layout=True)
    ntrain_grid = lc_summary["n_train_values"]
    series_cfg = [
        ("carla_transformer_ssa", "CARLA-Transformer (SSA)", "#0072B2", "o", "-"),
        ("pca_d10", "PCA Reconstruction (d=10)", "#009E73", "s", "--"),
        ("mahalanobis_lw", "Mahalanobis (Ledoit-Wolf)", "#E69F00", "^", "-."),
        ("raw_knn5", "Raw k-NN (k=5)", "#555555", "d", ":"),
    ]
    for metric_key, ax_i, ylabel, title in [
        ("zero_p975_f1", axes[0], "Zero-Label F1 (healthy val p97.5)", "(a) Zero-Label F1 vs Healthy Sample Size"),
        ("auc_pr_pi05", axes[1], "Prevalence-Adjusted PR-AUC (pi = 5%)", "(b) PR-AUC (pi = 5%) vs Healthy Sample Size"),
    ]:
        for key, label, color, marker, ls in series_cfg:
            means = [lc_summary["detectors"][key][str(n)][metric_key]["mean"] for n in ntrain_grid]
            stds = [lc_summary["detectors"][key][str(n)][metric_key]["sample_std"] for n in ntrain_grid]
            m_arr = np.asarray(means)
            s_arr = np.asarray(stds)
            ax_i.plot(ntrain_grid, m_arr, marker=marker, linestyle=ls, color=color, linewidth=1.9, label=label)
            ax_i.fill_between(ntrain_grid, m_arr - s_arr, m_arr + s_arr, color=color, alpha=0.14)
        ax_i.set_xlabel("Number of healthy training sweeps (N_train)")
        ax_i.set_ylabel(ylabel)
        ax_i.set_title(title)
        ax_i.set_xticks(ntrain_grid)
        ax_i.set_ylim(0.45, 1.0)
        ax_i.grid(True, alpha=0.25)
        ax_i.legend(loc="lower right", frameon=True)

    for ext in ("pdf", "png"):
        fig.savefig(plots_dir / f"mlwa_ds02_learning_curves.{ext}", dpi=300)
    plt.close(fig)

    # --- Figure C: Empirical Self-Attention Profile vs Jacobian Score Saliency ---
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), constrained_layout=True)
    f_khz = frequencies / 1000.0

    ax = axes[0]
    for reg_key, label, color, ls in [
        ("tf_physics_ssa_fixed_load", "CARLA-TF (SSA, 10 seeds)", "#0072B2", "-"),
        ("tf_recon_only", "AE-TF (Recon-only, 10 seeds)", "#009E73", "--"),
        ("tf_heuristic", "CARLA-TF (Heuristic, 10 seeds)", "#D55E00", ":"),
    ]:
        prof = np.asarray(attn_audit["attention_by_regime"][reg_key]["mean_key_profile_across_seeds"])
        std_p = np.asarray(attn_audit["attention_by_regime"][reg_key]["std_key_profile_across_seeds"])
        ax.plot(f_khz, prof * 100.0, color=color, linestyle=ls, linewidth=1.8, label=label)
        ax.fill_between(f_khz, (prof - std_p) * 100.0, (prof + std_p) * 100.0, color=color, alpha=0.14)
    ax.axhline(100.0 / len(frequencies), color="#888888", linestyle="--", linewidth=1.0, label="Uniform (0.99%/bin)")
    ax.axvspan(0.7, 2.0, color="#F0E442", alpha=0.22, label="LC resonance band")
    ax.set_xscale("log")
    ax.set_xlabel("Frequency [kHz]")
    ax.set_ylabel("Incoming key-attention share [% per bin]")
    ax.set_title("(a) Encoder Self-Attention Key Marginal")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", fontsize=7.5, frameon=True)

    ax2 = axes[1]
    strata_plot = [
        ("single_Cout", "Fault: C_out", "#0072B2", "-"),
        ("single_Lout", "Fault: L_out", "#56B4E9", "--"),
        ("single_Esr_C", "Fault: ESR_C", "#009E73", "-."),
        ("single_Esr_L", "Fault: ESR_L", "#E69F00", "-"),
        ("single_Rds_1", "Fault: R_ds1", "#D55E00", ":"),
    ]
    sal_dict = attn_audit["saliency_by_stratum_ssa_fixed_load"]
    for st_key, label, color, ls in strata_plot:
        if st_key not in sal_dict:
            continue
        prof = np.asarray(sal_dict[st_key]["normalized_mean_profile"])
        ax2.plot(f_khz, prof * 100.0, color=color, linestyle=ls, linewidth=1.8, label=label)
    ax2.axvspan(0.7, 2.0, color="#F0E442", alpha=0.22, label="LC resonance band")
    ax2.set_xscale("log")
    ax2.set_xlabel("Frequency [kHz]")
    ax2.set_ylabel("Normalized Jacobian saliency ||d s / d x_f|| [% per bin]")
    ax2.set_title("(b) CARLA-TF (SSA) Score Sensitivity by Fault Mode")
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(loc="upper right", fontsize=7.5, frameon=True)

    for ext in ("pdf", "png"):
        fig.savefig(plots_dir / f"mlwa_ds02_attention_and_saliency.{ext}", dpi=300)
    plt.close(fig)

    # --- Figure D: PCA Subspace Cliff & FRA Instrument Noise Fragility vs CARLA-TF (SSA) ---
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), constrained_layout=True)
    sub_rows = [r for r in pca_fragility["sweep"] if int(r["n_components"]) <= 20]
    d_vals = [int(r["n_components"]) for r in sub_rows]

    tf_reg = summary_10seed["regimes"]["tf_physics_ssa_fixed_load"]
    for ax_i, m_clean, m_cn, m_nn, tf_clean_key, tf_noisy_key, ylabel, title in [
        (
            axes[0],
            "clean_zero_p975_f1",
            "fra015_cleanfit_matched_p975_f1",
            "fra015_noisyfit_p975_f1",
            "zero_knn5_p975_f1",
            "fra015_matched_p975_f1",
            "Zero-Label F1 (healthy val p97.5)",
            "(a) PCA Subspace Cliff & FRA Noise Collapse (F1)",
        ),
        (
            axes[1],
            "clean_auc_pr_pi05",
            "fra015_cleanfit_auc_pr_pi05",
            "fra015_noisyfit_auc_pr_pi05",
            "zero_knn5_auc_pr_pi05",
            None,
            "Prevalence-Adjusted PR-AUC (pi = 5%)",
            "(b) PCA Subspace Sensitivity (PR-AUC @ pi = 5%)",
        ),
    ]:
        ax_i.plot(
            d_vals,
            [r[m_clean] for r in sub_rows],
            marker="s",
            color="#009E73",
            linewidth=1.9,
            label="PCA (Noiseless PSIM)",
        )
        ax_i.plot(
            d_vals,
            [r[m_cn] for r in sub_rows],
            marker="^",
            linestyle="--",
            color="#D55E00",
            linewidth=1.8,
            label="PCA Clean-Fit + FRA 0.15 dB",
        )
        ax_i.plot(
            d_vals,
            [r[m_nn] for r in sub_rows],
            marker="o",
            linestyle="-.",
            color="#E69F00",
            linewidth=1.7,
            label="PCA Noisy-Refit + FRA 0.15 dB",
        )

        m_tf = tf_reg[tf_clean_key]["mean"]
        s_tf = tf_reg[tf_clean_key]["sample_std"]
        ax_i.axhline(m_tf, color="#0072B2", linewidth=2.0, label="CARLA-TF SSA (Clean, 10 seeds)")
        ax_i.axhspan(m_tf - s_tf, m_tf + s_tf, color="#0072B2", alpha=0.13)

        if tf_noisy_key is not None:
            mn_tf = tf_reg[tf_noisy_key]["mean"]
            sn_tf = tf_reg[tf_noisy_key]["sample_std"]
            ax_i.axhline(
                mn_tf,
                color="#0072B2",
                linestyle="--",
                linewidth=1.8,
                label="CARLA-TF SSA (Clean-Fit + FRA 0.15 dB)",
            )
            ax_i.axhspan(mn_tf - sn_tf, mn_tf + sn_tf, color="#56B4E9", alpha=0.14)

        ax_i.axvline(3, color="#666666", linestyle=":", linewidth=1.1)
        ax_i.axvline(4, color="#666666", linestyle=":", linewidth=1.1)
        ax_i.text(3.15, 0.38, "95%/99%\ncum. var\n(d=3,4)", fontsize=7.3, color="#444444")
        ax_i.set_xlabel("PCA subspace dimension (d)")
        ax_i.set_ylabel(ylabel)
        ax_i.set_title(title)
        ax_i.set_xticks([2, 4, 6, 8, 10, 12, 15, 20])
        ax_i.set_ylim(0.32, 1.01)
        ax_i.grid(True, alpha=0.25)
        ax_i.legend(loc="lower right", fontsize=7.2, frameon=True)

    for ext in ("pdf", "png"):
        fig.savefig(plots_dir / f"mlwa_ds02_pca_subspace_fragility.{ext}", dpi=300)
    plt.close(fig)
    print(f"Saved extended publication plots to {plots_dir}")


def main() -> None:
    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    data_dir = ROOT / "data" / "buck" / "buck_data_v2" / "dataset_02"
    frozen_exp = ROOT / "experiments" / "dataset_01_run_06_corrected"
    study_exp = ROOT / "experiments" / "dataset_02_mlwa_study"
    eval_cache_dir = study_exp / "eval_cache"
    eval_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = ROOT / "cache"
    out_dir = ROOT / "papers" / "results" / "mlwa_study_dataset02"
    plots_dir = ROOT / "docs" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds_10 = list(range(200042, 200052))  # 10 paired seeds
    epochs = 150
    batch_size = 64

    tf_cfg = json.loads((frozen_exp / "carla_transformer" / "model_config.json").read_text())
    conv_cfg = json.loads((frozen_exp / "carla_conv1d" / "model_config.json").read_text())

    dataset = prepare_or_load_stratified_dataset02(data_dir, study_exp, cache_dir)
    splits = dataset._splits
    frequencies = np.asarray(dataset.frequencies, dtype=np.float64)
    test_labels = np.asarray(dataset.test_labels, dtype=int)
    test_variations = [dataset._metadata[int(i)].variations for i in splits["test_idx"]]
    test_strata = [_fault_stratum(v) if lbl == 1 else "healthy" for lbl, v in zip(test_labels, test_variations)]
    fv_labels = np.asarray(dataset.fit_val_labels, dtype=int)
    fv_norm_mask = fv_labels == 0

    noisy_tr_015 = dataset.preprocessor.transform(
        inject_fra_measurement_noise(dataset._raw_data[splits["train_idx"]], 0.15, 0.75, seed=301)
    )
    noisy_fv_norm_015 = dataset.preprocessor.transform(
        inject_fra_measurement_noise(
            dataset._raw_data[splits["fit_val_idx"]][fv_norm_mask], 0.15, 0.75, seed=302
        )
    )
    noisy_test_015 = dataset.preprocessor.transform(
        inject_fra_measurement_noise(dataset._raw_data[splits["test_idx"]], 0.15, 0.75, seed=303)
    )
    ssa_spec = BuckConverter(include_parasitic_drop=True, rds_low=0.0)

    # =========================================================================
    # BLOCK 0: Raw-space PCA dimension sweep + FRA Noise Fragility + References
    # =========================================================================
    print("=== [Block 0] Raw-Space PCA Subspace Sweep & Classical References ===")
    from sklearn.decomposition import PCA

    pca_full = PCA().fit(dataset.train_data.reshape(len(dataset.train_data), -1))
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    pca_sweep_rows: List[Dict[str, Any]] = []
    pca_frag_rows: List[Dict[str, Any]] = []
    for d_comp in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30):
        det = PCAReconstructionDetector(n_components=d_comp, seed=42).fit(dataset.train_data)
        s_val_norm = det.score(dataset.fit_val_data[fv_norm_mask])
        s_test = det.score(dataset.test_data)
        row = _eval_zero_label_scores(
            f"PCA (d={d_comp})", "raw_pca_sweep", s_val_norm, s_test, test_labels, test_variations
        )
        row["n_components"] = d_comp
        row["cumulative_explained_variance"] = float(cum_var[d_comp - 1])

        # Clean-fit PCA under 0.15 dB / 0.75 deg FRA instrument noise (matched calibration)
        row_cn = _eval_zero_label_scores(
            f"PCA_cleanfit_fra015 (d={d_comp})",
            "raw_pca_fra015_cleanfit",
            det.score(noisy_fv_norm_015),
            det.score(noisy_test_015),
            test_labels,
            test_variations,
        )
        # Noisy-refit PCA under 0.15 dB / 0.75 deg FRA instrument noise
        det_nf = PCAReconstructionDetector(n_components=d_comp, seed=42).fit(noisy_tr_015)
        row_nn = _eval_zero_label_scores(
            f"PCA_noisyfit_fra015 (d={d_comp})",
            "raw_pca_fra015_noisyfit",
            det_nf.score(noisy_fv_norm_015),
            det_nf.score(noisy_test_015),
            test_labels,
            test_variations,
        )
        row["fra015_matched_p975_f1"] = float(row_cn["zero_p975_f1"])
        row["fra015_cleanfit_auc_pr_pi05"] = float(row_cn["auc_pr_pi05"])
        row["fra015_noisyfit_p975_f1"] = float(row_nn["zero_p975_f1"])
        row["fra015_noisyfit_auc_pr_pi05"] = float(row_nn["auc_pr_pi05"])
        pca_sweep_rows.append(row)
        pca_frag_rows.append(
            {
                "n_components": d_comp,
                "cumulative_explained_variance": float(cum_var[d_comp - 1]),
                "eigenvalue_variance_ratio": float(pca_full.explained_variance_ratio_[d_comp - 1]),
                "clean_zero_p975_f1": float(row["zero_p975_f1"]),
                "clean_auc_pr_pi05": float(row["auc_pr_pi05"]),
                "fra015_cleanfit_matched_p975_f1": float(row_cn["zero_p975_f1"]),
                "fra015_cleanfit_auc_pr_pi05": float(row_cn["auc_pr_pi05"]),
                "fra015_noisyfit_p975_f1": float(row_nn["zero_p975_f1"]),
                "fra015_noisyfit_auc_pr_pi05": float(row_nn["auc_pr_pi05"]),
            }
        )
    _write_csv(out_dir / "raw_pca_dimension_sweep.csv", pca_sweep_rows)
    pca_fragility = {
        "zero_label_variance_heuristics": {
            f"{thr*100:.2f}%": int(np.searchsorted(cum_var, thr) + 1)
            for thr in (0.90, 0.95, 0.99, 0.995, 0.999)
        },
        "sweep": pca_frag_rows,
    }
    (out_dir / "pca_subspace_fragility_audit.json").write_text(
        json.dumps(pca_fragility, indent=2) + "\n"
    )

    pca8 = PCAReconstructionDetector(n_components=8, seed=42).fit(dataset.train_data)
    pca10 = PCAReconstructionDetector(n_components=10, seed=42).fit(dataset.train_data)
    maha_lw = MahalanobisDetector(ridge=None).fit(dataset.train_data)
    knn5_raw = RawKNNDetector(n_neighbors=5).fit(dataset.train_data)

    def _ref_with_fra015(name: str, det_obj: Any) -> Dict[str, Any]:
        base = _eval_zero_label_scores(
            name,
            "raw_classical",
            det_obj.score(dataset.fit_val_data[fv_norm_mask]),
            det_obj.score(dataset.test_data),
            test_labels,
            test_variations,
        )
        noisy_ev = _eval_zero_label_scores(
            f"{name}_fra015",
            "raw_classical_fra015",
            det_obj.score(noisy_fv_norm_015),
            det_obj.score(noisy_test_015),
            test_labels,
            test_variations,
        )
        base["fra015_matched_p975_f1"] = float(noisy_ev["zero_p975_f1"])
        base["fra015_auc_pr_pi05"] = float(noisy_ev["auc_pr_pi05"])
        return base

    classical_refs = {
        "pca_d8_oracle": _ref_with_fra015("PCA Reconstruction (d=8, clean oracle)", pca8),
        "pca_d10": _ref_with_fra015("PCA Reconstruction (d=10)", pca10),
        "mahalanobis_lw": _ref_with_fra015("Mahalanobis (Ledoit-Wolf)", maha_lw),
        "raw_knn5": _ref_with_fra015("Raw k-NN (k=5)", knn5_raw),
    }

    # =========================================================================
    # BLOCK 1 & 2 & 3: 10-Seed Transformer Training + Controls + Attention Audit
    # =========================================================================
    print("\n=== [Block 1-3] 10-Seed Paired Contrasts, Latent Controls & Attention Audit ===")
    ablation_rows: List[Dict[str, Any]] = []
    control_rows: List[Dict[str, Any]] = []

    attn_key_profiles: Dict[str, List[np.ndarray]] = {
        "tf_physics_ssa_fixed_load": [],
        "tf_physics_ssa_load_adapt": [],
        "tf_heuristic": [],
        "tf_recon_only": [],
    }
    attn_summaries_by_seed: Dict[str, List[Dict[str, Any]]] = {k: [] for k in attn_key_profiles}
    saliency_ssa_fixed_all: List[np.ndarray] = []
    saliency_by_regime: Dict[str, List[np.ndarray]] = {
        "tf_physics_ssa_fixed_load": [],
        "tf_heuristic": [],
        "tf_recon_only": [],
    }
    ssa_seed_test_scores_zero: List[np.ndarray] = []
    mse_seed_test_scores_zero: List[np.ndarray] = []
    heur_seed_test_scores_zero: List[np.ndarray] = []

    def _run_learned_space_controls(
        mdl: ContrastiveAutoencoder, regime_label: str, seed_val: int
    ) -> List[Dict[str, Any]]:
        rows_out: List[Dict[str, Any]] = []
        for space_key, norm_flag in [("latent", False), ("projection", True)]:
            tr_e = _get_embeddings_batched(mdl, dataset.train_data, key=space_key)
            fv_e = _get_embeddings_batched(mdl, dataset.fit_val_data[fv_norm_mask], key=space_key)
            te_e = _get_embeddings_batched(mdl, dataset.test_data, key=space_key)

            # 1. k-NN (k=5)
            sc_knn = EmbeddingAnomalyScorer(tr_e, method="knn", k=5, normalize=norm_flag)
            r_knn = _eval_zero_label_scores(
                f"{regime_label}__{space_key}__knn5",
                "learned_control",
                sc_knn.score(fv_e),
                sc_knn.score(te_e),
                test_labels,
                test_variations,
            )
            rows_out.append({"regime": regime_label, "space": space_key, "scorer": "knn_k5", "seed": seed_val, **r_knn})

            # 2. Spectral-floored Ledoit-Wolf
            sc_lw = EmbeddingAnomalyScorer(tr_e, method="ledoit_wolf", k=5, normalize=norm_flag)
            r_lw = _eval_zero_label_scores(
                f"{regime_label}__{space_key}__spectral_lw",
                "learned_control",
                sc_lw.score(fv_e),
                sc_lw.score(te_e),
                test_labels,
                test_variations,
            )
            rows_out.append({"regime": regime_label, "space": space_key, "scorer": "spectral_ledoit_wolf", "seed": seed_val, **r_lw})

            # 3. Standard sklearn Ledoit-Wolf Mahalanobis
            det_mlw = MahalanobisDetector(ridge=None).fit(tr_e)
            r_mlw = _eval_zero_label_scores(
                f"{regime_label}__{space_key}__sklearn_lw",
                "learned_control",
                det_mlw.score(fv_e),
                det_mlw.score(te_e),
                test_labels,
                test_variations,
            )
            rows_out.append({"regime": regime_label, "space": space_key, "scorer": "sklearn_ledoit_wolf", "seed": seed_val, **r_mlw})

            # 4. Linear PCA reconstruction in embedding space (d=5 and d=10)
            for d_emb in (5, 10):
                det_pca = PCAReconstructionDetector(n_components=d_emb, seed=42).fit(tr_e)
                r_pca = _eval_zero_label_scores(
                    f"{regime_label}__{space_key}__pca_d{d_emb}",
                    "learned_control",
                    det_pca.score(fv_e),
                    det_pca.score(te_e),
                    test_labels,
                    test_variations,
                )
                rows_out.append({"regime": regime_label, "space": space_key, "scorer": f"pca_d{d_emb}", "seed": seed_val, **r_pca})
        return rows_out

    for seed in seeds_10:
        cache_meta_file = eval_cache_dir / f"seed_{seed}_meta.json"
        cache_npz_file = eval_cache_dir / f"seed_{seed}_arrays.npz"

        if cache_meta_file.exists() and cache_npz_file.exists():
            meta_b = json.loads(cache_meta_file.read_text())
            arr_b = np.load(cache_npz_file)
            ablation_rows.extend(meta_b["ablation_rows"])
            control_rows.extend(meta_b["control_rows"])
            for rk in ("tf_recon_only", "tf_heuristic", "tf_physics_ssa_fixed_load", "tf_physics_ssa_load_adapt"):
                attn_summaries_by_seed[rk].append(meta_b["attn_summaries"][rk])
                attn_key_profiles[rk].append(np.asarray(arr_b[f"attn_key_{rk}"], dtype=np.float64))
            saliency_by_regime["tf_recon_only"].append(np.asarray(arr_b["sal_rec"], dtype=np.float64))
            saliency_by_regime["tf_heuristic"].append(np.asarray(arr_b["sal_h"], dtype=np.float64))
            saliency_by_regime["tf_physics_ssa_fixed_load"].append(np.asarray(arr_b["sal_f"], dtype=np.float64))
            saliency_ssa_fixed_all.append(np.asarray(arr_b["sal_f"], dtype=np.float64))
            mse_seed_test_scores_zero.append(np.asarray(arr_b["mse_scores"], dtype=np.float64))
            heur_seed_test_scores_zero.append(np.asarray(arr_b["heur_scores"], dtype=np.float64))
            ssa_seed_test_scores_zero.append(np.asarray(arr_b["ssa_scores"], dtype=np.float64))
            print(f"  [Cached] Loaded full evaluation & attribution bundle for seed={seed}")
            continue

        seed_ablation_rows: List[Dict[str, Any]] = []
        seed_control_rows: List[Dict[str, Any]] = []

        # 1. Recon-only Transformer
        rec_mdl, rec_prov = _train_or_load_model(
            arch_cfg=tf_cfg,
            regime="tf_recon_only",
            seed=seed,
            run_dir=study_exp / "ablation_transformer" / "recon_only" / f"seed_{seed}",
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            contrastive_weight_override=0.0,
            use_physics=False,
        )
        mse_scores_s: Optional[np.ndarray] = None
        for sub_name, space in [
            ("tf_recon_only_mse", "reconstruction_mse"),
            ("tf_recon_only_latent", "latent"),
        ]:
            ev = _evaluate_deep_model(
                rec_mdl, dataset, space, test_variations,
                noisy_test_015, noisy_fv_norm_015, noisy_tr_015,
            )
            if sub_name == "tf_recon_only_mse":
                mse_scores_s = np.asarray(ev["scores"], dtype=np.float64)
            ev.pop("scores")
            ev.pop("predictions")
            seed_ablation_rows.append({"regime": sub_name, "seed": seed, "epochs_trained": rec_prov["epochs_trained"], **ev})

        seed_control_rows.extend(_run_learned_space_controls(rec_mdl, "tf_recon_only", seed))
        maps_rec = extract_transformer_attention_maps(rec_mdl, dataset.test_data)
        sum_rec = summarize_frequency_attention(maps_rec, frequencies)
        sal_rec = compute_knn_score_frequency_saliency(
            rec_mdl, dataset.test_data, dataset.train_data, space="latent", k=5
        )
        keras.backend.clear_session()

        # 2. Heuristic CARLA-Transformer
        heur_mdl, heur_prov = _train_or_load_model(
            arch_cfg=tf_cfg,
            regime="tf_heuristic",
            seed=seed,
            run_dir=study_exp / "ablation_transformer" / "heuristic" / f"seed_{seed}",
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            use_physics=False,
        )
        ev_h = _evaluate_deep_model(
            heur_mdl, dataset, "projection", test_variations,
            noisy_test_015, noisy_fv_norm_015, noisy_tr_015,
        )
        tr_h = _get_embeddings_batched(heur_mdl, dataset.train_data, key="projection")
        te_h = _get_embeddings_batched(heur_mdl, dataset.test_data, key="projection")
        heur_scores_s = EmbeddingAnomalyScorer(tr_h, method="knn", k=5, normalize=True).score(te_h)
        ev_h.pop("scores")
        ev_h.pop("predictions")
        seed_ablation_rows.append({"regime": "tf_heuristic", "seed": seed, "epochs_trained": heur_prov["epochs_trained"], **ev_h})

        seed_control_rows.extend(_run_learned_space_controls(heur_mdl, "tf_heuristic", seed))
        maps_h = extract_transformer_attention_maps(heur_mdl, dataset.test_data)
        sum_h = summarize_frequency_attention(maps_h, frequencies)
        sal_h = compute_knn_score_frequency_saliency(
            heur_mdl, dataset.test_data, dataset.train_data, space="projection", k=5
        )
        keras.backend.clear_session()

        # 3. Physics SSA Fixed-Load CARLA-Transformer (primary clean model)
        fix_mdl, fix_prov = _train_or_load_model(
            arch_cfg=tf_cfg,
            regime="tf_physics_ssa_fixed_load",
            seed=seed,
            run_dir=study_exp / "ablation_transformer" / "physics_ssa_fixed_load" / f"seed_{seed}",
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            converter_spec=ssa_spec,
            use_physics=True,
            adapt_load=False,
            strict_margin=False,
        )
        ev_f = _evaluate_deep_model(
            fix_mdl, dataset, "projection", test_variations,
            noisy_test_015, noisy_fv_norm_015, noisy_tr_015,
        )
        tr_f = _get_embeddings_batched(fix_mdl, dataset.train_data, key="projection")
        te_f = _get_embeddings_batched(fix_mdl, dataset.test_data, key="projection")
        ssa_scores_s = EmbeddingAnomalyScorer(tr_f, method="knn", k=5, normalize=True).score(te_f)
        ev_f.pop("scores")
        ev_f.pop("predictions")
        seed_ablation_rows.append(
            {"regime": "tf_physics_ssa_fixed_load", "seed": seed, "epochs_trained": fix_prov["epochs_trained"], **ev_f}
        )

        seed_control_rows.extend(_run_learned_space_controls(fix_mdl, "tf_physics_ssa_fixed_load", seed))
        maps_f = extract_transformer_attention_maps(fix_mdl, dataset.test_data)
        sum_f = summarize_frequency_attention(maps_f, frequencies)
        sal_f = compute_knn_score_frequency_saliency(
            fix_mdl, dataset.test_data, dataset.train_data, space="projection", k=5
        )
        keras.backend.clear_session()

        # 4. Physics SSA Load-Adaptive CARLA-Transformer
        la_mdl, la_prov = _train_or_load_model(
            arch_cfg=tf_cfg,
            regime="enc_carla_transformer",
            seed=seed,
            run_dir=study_exp / "encoders" / "carla_transformer" / f"seed_{seed}",
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            converter_spec=ssa_spec,
            use_physics=True,
            adapt_load=True,
            strict_margin=True,
        )
        ev_la = _evaluate_deep_model(
            la_mdl, dataset, "projection", test_variations,
            noisy_test_015, noisy_fv_norm_015, noisy_tr_015,
        )
        ev_la.pop("scores")
        ev_la.pop("predictions")
        seed_ablation_rows.append(
            {"regime": "tf_physics_ssa_load_adapt", "seed": seed, "epochs_trained": la_prov["epochs_trained"], **ev_la}
        )
        maps_la = extract_transformer_attention_maps(la_mdl, dataset.test_data)
        sum_la = summarize_frequency_attention(maps_la, frequencies)
        keras.backend.clear_session()

        assert mse_scores_s is not None
        np.savez_compressed(
            cache_npz_file,
            attn_key_tf_recon_only=np.asarray(sum_rec["mean_key_attention"], dtype=np.float64),
            attn_key_tf_heuristic=np.asarray(sum_h["mean_key_attention"], dtype=np.float64),
            attn_key_tf_physics_ssa_fixed_load=np.asarray(sum_f["mean_key_attention"], dtype=np.float64),
            attn_key_tf_physics_ssa_load_adapt=np.asarray(sum_la["mean_key_attention"], dtype=np.float64),
            sal_rec=sal_rec,
            sal_h=sal_h,
            sal_f=sal_f,
            mse_scores=mse_scores_s,
            heur_scores=heur_scores_s,
            ssa_scores=ssa_scores_s,
        )
        cache_meta_file.write_text(
            json.dumps(
                {
                    "seed": seed,
                    "ablation_rows": seed_ablation_rows,
                    "control_rows": seed_control_rows,
                    "attn_summaries": {
                        "tf_recon_only": sum_rec,
                        "tf_heuristic": sum_h,
                        "tf_physics_ssa_fixed_load": sum_f,
                        "tf_physics_ssa_load_adapt": sum_la,
                    },
                },
                indent=2,
            )
            + "\n"
        )

        ablation_rows.extend(seed_ablation_rows)
        control_rows.extend(seed_control_rows)
        attn_summaries_by_seed["tf_recon_only"].append(sum_rec)
        attn_key_profiles["tf_recon_only"].append(np.asarray(sum_rec["mean_key_attention"]))
        attn_summaries_by_seed["tf_heuristic"].append(sum_h)
        attn_key_profiles["tf_heuristic"].append(np.asarray(sum_h["mean_key_attention"]))
        attn_summaries_by_seed["tf_physics_ssa_fixed_load"].append(sum_f)
        attn_key_profiles["tf_physics_ssa_fixed_load"].append(np.asarray(sum_f["mean_key_attention"]))
        attn_summaries_by_seed["tf_physics_ssa_load_adapt"].append(sum_la)
        attn_key_profiles["tf_physics_ssa_load_adapt"].append(np.asarray(sum_la["mean_key_attention"]))
        saliency_by_regime["tf_recon_only"].append(sal_rec)
        saliency_by_regime["tf_heuristic"].append(sal_h)
        saliency_by_regime["tf_physics_ssa_fixed_load"].append(sal_f)
        saliency_ssa_fixed_all.append(sal_f)
        mse_seed_test_scores_zero.append(mse_scores_s)
        heur_seed_test_scores_zero.append(heur_scores_s)
        ssa_seed_test_scores_zero.append(ssa_scores_s)

        # Incremental checkpoint flush
        _write_csv(out_dir / "ablation_transformer_10seed.csv", ablation_rows)
        _write_csv(out_dir / "latent_controls_10seed.csv", control_rows)
        print(f"  [Done] Evaluated & cached seed={seed}")

    # Also compute Conv1D Jacobian saliency across its 5 existing seeds for architectural comparison
    conv_saliencies: List[np.ndarray] = []
    for s5 in range(200042, 200047):
        c_dir = study_exp / "encoders" / "carla_conv1d" / f"seed_{s5}" / "checkpoints" / "best"
        if (c_dir / "autoencoder.weights.h5").exists():
            c_mdl = ContrastiveAutoencoder.load(str(c_dir), config=conv_cfg)
            conv_saliencies.append(
                compute_knn_score_frequency_saliency(
                    c_mdl, dataset.test_data, dataset.train_data, space="projection", k=5
                )
            )
            keras.backend.clear_session()

    # Summarize 10-seed ablation + paired statistical tests
    regimes_10 = [
        "tf_recon_only_mse",
        "tf_recon_only_latent",
        "tf_heuristic",
        "tf_physics_ssa_fixed_load",
        "tf_physics_ssa_load_adapt",
    ]
    metrics_10 = [
        "f1",
        "precision",
        "recall",
        "auc_roc",
        "auc_pr_pi10",
        "auc_pr_pi05",
        "zero_knn5_p975_f1",
        "zero_knn5_p95_f1",
        "zero_knn5_auc_roc",
        "zero_knn5_auc_pr_pi05",
        "fra015_matched_p975_f1",
        "recall_passive_single",
        "recall_cout_single",
        "recall_esrc_single",
        "recall_lout_single",
        "recall_esrl_single",
        "recall_rds1_single",
    ]
    by_rs = {(r["regime"], r["seed"]): r for r in ablation_rows}
    summary_10seed: Dict[str, Any] = {
        "seeds": seeds_10,
        "n_seeds": len(seeds_10),
        "min_possible_two_sided_p": float(2.0 / (2 ** len(seeds_10))),
        "regimes": {},
        "paired_comparisons_deep": {},
        "paired_vs_classical_baselines": {},
        "classical_references": classical_refs,
    }
    for reg in regimes_10:
        rr = [by_rs[(reg, s)] for s in seeds_10]
        summary_10seed["regimes"][reg] = {m: _agg([float(x[m]) for x in rr]) for m in metrics_10}

    for reg_a, reg_b in [
        ("tf_physics_ssa_fixed_load", "tf_recon_only_mse"),
        ("tf_physics_ssa_fixed_load", "tf_recon_only_latent"),
        ("tf_physics_ssa_fixed_load", "tf_heuristic"),
        ("tf_physics_ssa_load_adapt", "tf_physics_ssa_fixed_load"),
        ("tf_physics_ssa_load_adapt", "tf_recon_only_mse"),
        ("tf_physics_ssa_load_adapt", "tf_heuristic"),
    ]:
        summary_10seed["paired_comparisons_deep"][f"{reg_a}_vs_{reg_b}"] = {
            m: paired_sign_flip_test(
                [by_rs[(reg_a, s)][m] for s in seeds_10],
                [by_rs[(reg_b, s)][m] for s in seeds_10],
            )
            for m in ("f1", "zero_knn5_p975_f1", "zero_knn5_p95_f1", "zero_knn5_auc_roc", "zero_knn5_auc_pr_pi05", "fra015_matched_p975_f1")
        }

    # Test 10-seed CARLA-Transformer (SSA) against deterministic raw classical baselines
    for base_key, base_dict in classical_refs.items():
        summary_10seed["paired_vs_classical_baselines"][f"tf_physics_ssa_fixed_load_vs_{base_key}"] = {
            "zero_p975_f1": paired_sign_flip_test(
                [by_rs[("tf_physics_ssa_fixed_load", s)]["zero_knn5_p975_f1"] for s in seeds_10],
                [base_dict["zero_p975_f1"]] * len(seeds_10),
            ),
            "zero_p95_f1": paired_sign_flip_test(
                [by_rs[("tf_physics_ssa_fixed_load", s)]["zero_knn5_p95_f1"] for s in seeds_10],
                [base_dict["zero_p95_f1"]] * len(seeds_10),
            ),
            "auc_pr_pi05": paired_sign_flip_test(
                [by_rs[("tf_physics_ssa_fixed_load", s)]["zero_knn5_auc_pr_pi05"] for s in seeds_10],
                [base_dict["auc_pr_pi05"]] * len(seeds_10),
            ),
            "auc_roc": paired_sign_flip_test(
                [by_rs[("tf_physics_ssa_fixed_load", s)]["zero_knn5_auc_roc"] for s in seeds_10],
                [base_dict["auc_roc"]] * len(seeds_10),
            ),
            "fra015_matched_p975_f1": paired_sign_flip_test(
                [by_rs[("tf_physics_ssa_fixed_load", s)]["fra015_matched_p975_f1"] for s in seeds_10],
                [base_dict["fra015_matched_p975_f1"]] * len(seeds_10),
            ),
        }
    (out_dir / "ablation_transformer_10seed_summary.json").write_text(
        json.dumps(summary_10seed, indent=2) + "\n"
    )

    # Summarize learned-space controls
    controls_summary: Dict[str, Any] = {}
    ctrl_groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in control_rows:
        key = (row["regime"], row["space"], row["scorer"])
        ctrl_groups.setdefault(key, []).append(row)
    for (reg, spc, scr), rr in ctrl_groups.items():
        controls_summary[f"{reg}__{spc}__{scr}"] = {
            m: _agg([float(x[m]) for x in rr])
            for m in ("zero_p975_f1", "zero_p95_f1", "auc_roc", "auc_pr_pi05", "recall_passive_single", "recall_esrc_single", "recall_esrl_single", "recall_rds1_single")
        }
    (out_dir / "latent_controls_summary.json").write_text(
        json.dumps(controls_summary, indent=2) + "\n"
    )

    # Summarize Attention & Jacobian Saliency Audit
    attn_audit: Dict[str, Any] = {
        "frequencies_hz": frequencies.tolist(),
        "bands_hz": DEFAULT_FREQUENCY_BANDS_HZ,
        "attention_by_regime": {},
        "saliency_by_regime": {},
        "saliency_by_stratum_ssa_fixed_load": {},
    }
    freq_profile_rows: List[Dict[str, Any]] = []
    for reg_k, prof_list in attn_key_profiles.items():
        mat = np.stack(prof_list, axis=0)  # (10, 101)
        sums = attn_summaries_by_seed[reg_k]
        band_enrichment = {
            bname: _agg([float(s["bands"][bname]["enrichment_ratio"]) for s in sums])
            for bname in DEFAULT_FREQUENCY_BANDS_HZ
        }
        cross_dc_to_res = _agg(
            [float(s["cross_band_enrichment"]["dc_low"]["resonance"]) for s in sums]
        )
        cross_hf_to_res = _agg(
            [float(s["cross_band_enrichment"]["hf_esr"]["resonance"]) for s in sums]
        )
        attn_audit["attention_by_regime"][reg_k] = {
            "normalized_entropy": _agg([float(s["normalized_entropy_mean"]) for s in sums]),
            "peak_freq_hz": _agg([float(s["peak_attended_freq_hz"]) for s in sums]),
            "band_enrichment": band_enrichment,
            "cross_band_dc_to_resonance_enrichment": cross_dc_to_res,
            "cross_band_hf_to_resonance_enrichment": cross_hf_to_res,
            "mean_key_profile_across_seeds": np.mean(mat, axis=0).tolist(),
            "std_key_profile_across_seeds": np.std(mat, axis=0, ddof=1).tolist(),
        }

    def _summarize_saliency_stack(sal_list: List[np.ndarray]) -> Dict[str, Any]:
        # Each entry is (N_test, 101); normalize per-seed mean profile to sum to 1
        norm_seed_profs = []
        band_enrich: Dict[str, List[float]] = {b: [] for b in DEFAULT_FREQUENCY_BANDS_HZ}
        for sal in sal_list:
            p = np.mean(sal, axis=0, dtype=np.float64)
            p = p / max(float(np.sum(p)), 1e-12)
            norm_seed_profs.append(p)
            for bname, (f_lo, f_hi) in DEFAULT_FREQUENCY_BANDS_HZ.items():
                m = (frequencies >= f_lo) & (frequencies <= f_hi)
                band_enrich[bname].append(float(np.sum(p[m]) / np.mean(m)))
        p_mat = np.stack(norm_seed_profs, axis=0)
        mean_p = np.mean(p_mat, axis=0)
        peak_i = int(np.argmax(mean_p))
        return {
            "peak_freq_hz": float(frequencies[peak_i]),
            "band_enrichment": {b: _agg(vals) for b, vals in band_enrich.items()},
            "normalized_mean_profile": mean_p.tolist(),
            "normalized_std_profile": (
                np.std(p_mat, axis=0, ddof=1).tolist() if len(sal_list) > 1 else [0.0] * len(mean_p)
            ),
        }

    for reg_k, s_list in saliency_by_regime.items():
        attn_audit["saliency_by_regime"][reg_k] = _summarize_saliency_stack(s_list)
    if conv_saliencies:
        attn_audit["saliency_by_regime"]["conv1d_physics_ssa_load_adapt"] = _summarize_saliency_stack(
            conv_saliencies
        )

    # Stratify CARLA-Transformer (SSA fixed_load) Jacobian saliency by physical fault mode
    sal_ssa_stack = np.stack(saliency_ssa_fixed_all, axis=0)  # (10, N_test, 101)
    for st_name in ("healthy", "single_Cout", "single_Lout", "single_Esr_C", "single_Esr_L", "single_Rds_1", "multi_fault"):
        mask = np.array([s == st_name for s in test_strata], dtype=bool)
        if not np.any(mask):
            continue
        sub_list = [sal_ssa_stack[i, mask, :] for i in range(sal_ssa_stack.shape[0])]
        attn_audit["saliency_by_stratum_ssa_fixed_load"][st_name] = _summarize_saliency_stack(sub_list)

    for idx_f, f_hz in enumerate(frequencies):
        freq_profile_rows.append(
            {
                "bin_index": idx_f,
                "frequency_hz": float(f_hz),
                "attn_key_ssa_fixed": attn_audit["attention_by_regime"]["tf_physics_ssa_fixed_load"]["mean_key_profile_across_seeds"][idx_f],
                "attn_key_recon_only": attn_audit["attention_by_regime"]["tf_recon_only"]["mean_key_profile_across_seeds"][idx_f],
                "attn_key_heuristic": attn_audit["attention_by_regime"]["tf_heuristic"]["mean_key_profile_across_seeds"][idx_f],
                "saliency_ssa_fixed": attn_audit["saliency_by_regime"]["tf_physics_ssa_fixed_load"]["normalized_mean_profile"][idx_f],
                "saliency_recon_only": attn_audit["saliency_by_regime"]["tf_recon_only"]["normalized_mean_profile"][idx_f],
                "saliency_conv1d_ssa": (
                    attn_audit["saliency_by_regime"]["conv1d_physics_ssa_load_adapt"]["normalized_mean_profile"][idx_f]
                    if "conv1d_physics_ssa_load_adapt" in attn_audit["saliency_by_regime"]
                    else 0.0
                ),
            }
        )
    _write_csv(out_dir / "transformer_frequency_profiles.csv", freq_profile_rows)
    (out_dir / "transformer_attention_audit.json").write_text(json.dumps(attn_audit, indent=2) + "\n")

    # =========================================================================
    # BLOCK 4: Leak-Free Learning Curves vs N_train (5 seeds)
    # =========================================================================
    print("\n=== [Block 4] Leak-Free Sample-Complexity Learning Curves vs N_train ===")
    lc_seeds = list(range(200042, 200047))
    ntrain_values = [50, 100, 150, 200, 250, 300]
    raw_train_all = dataset._raw_data[splits["train_idx"]]
    raw_val_norm = dataset._raw_data[splits["fit_val_idx"]][fv_norm_mask]
    raw_test_all = dataset._raw_data[splits["test_idx"]]

    lc_rows: List[Dict[str, Any]] = []
    for n_tr in ntrain_values:
        for seed in lc_seeds:
            if n_tr == len(raw_train_all):
                prep = dataset.preprocessor
                tr_sub = dataset.train_data
                fv_norm_sub = dataset.fit_val_data[fv_norm_mask]
                te_sub = dataset.test_data
            else:
                rng = np.random.default_rng(seed * 31 + n_tr)
                sub_idx = np.sort(rng.choice(len(raw_train_all), size=n_tr, replace=False))
                prep = DataPreprocessor(
                    normalization_method="standard",
                    use_real_imag=dataset.preprocessor.use_real_imag,
                )
                tr_sub = prep.fit_transform(raw_train_all[sub_idx])
                fv_norm_sub = prep.transform(raw_val_norm)
                te_sub = prep.transform(raw_test_all)

            # Classical baselines on exact leak-free preprocessed subsample
            for det_key, det_obj in [
                ("raw_knn5", RawKNNDetector(n_neighbors=5)),
                ("mahalanobis_lw", MahalanobisDetector(ridge=None)),
                ("pca_d10", PCAReconstructionDetector(n_components=10, seed=seed)),
            ]:
                det_obj.fit(tr_sub)
                ev_c = _eval_zero_label_scores(
                    det_key, "learning_curve", det_obj.score(fv_norm_sub), det_obj.score(te_sub), test_labels, test_variations
                )
                lc_rows.append({"detector_key": det_key, "n_train": n_tr, "seed": seed, **ev_c})

            # CARLA-Transformer (SSA fixed_load)
            if n_tr == len(raw_train_all):
                r_full = by_rs[("tf_physics_ssa_fixed_load", seed)]
                lc_rows.append(
                    {
                        "detector_key": "carla_transformer_ssa",
                        "n_train": n_tr,
                        "seed": seed,
                        "detector": "carla_transformer_ssa",
                        "family": "learning_curve",
                        "zero_p975_f1": float(r_full["zero_knn5_p975_f1"]),
                        "zero_p975_precision": float(r_full["zero_knn5_p975_precision"]),
                        "zero_p975_recall": float(r_full["zero_knn5_p975_recall"]),
                        "zero_p95_f1": float(r_full["zero_knn5_p95_f1"]),
                        "zero_p95_precision": np.nan,
                        "zero_p95_recall": np.nan,
                        "auc_roc": float(r_full["zero_knn5_auc_roc"]),
                        "auc_pr_test": float(r_full["auc_pr_test"]),
                        "auc_pr_pi10": float(r_full["auc_pr_pi10"]),
                        "auc_pr_pi05": float(r_full["zero_knn5_auc_pr_pi05"]),
                        "recall_passive_single": float(r_full["recall_passive_single"]),
                        "recall_cout_single": float(r_full["recall_cout_single"]),
                        "recall_esrc_single": float(r_full["recall_esrc_single"]),
                        "recall_lout_single": float(r_full["recall_lout_single"]),
                        "recall_esrl_single": float(r_full["recall_esrl_single"]),
                        "recall_rds1_single": float(r_full["recall_rds1_single"]),
                        "recall_multi_fault": float(r_full["recall_multi_fault"]),
                    }
                )
            else:
                lc_cache_file = eval_cache_dir / f"lc_n{n_tr}_s{seed}.json"
                if lc_cache_file.exists():
                    lc_rows.append(json.loads(lc_cache_file.read_text()))
                else:
                    lc_mdl, _ = _train_or_load_subsampled_transformer(
                        arch_cfg=tf_cfg,
                        n_train=n_tr,
                        seed=seed,
                        run_dir=study_exp / "learning_curves" / f"ntrain_{n_tr}" / f"seed_{seed}",
                        train_sub=tr_sub,
                        val_norm=fv_norm_sub,
                        frequencies=frequencies,
                        scaler=prep.scaler,
                        use_real_imag=prep.use_real_imag,
                        converter_spec=ssa_spec,
                        epochs=epochs,
                    )
                    tr_e = _get_embeddings_batched(lc_mdl, tr_sub, key="projection")
                    fv_e = _get_embeddings_batched(lc_mdl, fv_norm_sub, key="projection")
                    te_e = _get_embeddings_batched(lc_mdl, te_sub, key="projection")
                    sc = EmbeddingAnomalyScorer(tr_e, method="knn", k=5, normalize=True)
                    ev_tf = _eval_zero_label_scores(
                        "carla_transformer_ssa",
                        "learning_curve",
                        sc.score(fv_e),
                        sc.score(te_e),
                        test_labels,
                        test_variations,
                    )
                    row_lc = {"detector_key": "carla_transformer_ssa", "n_train": n_tr, "seed": seed, **ev_tf}
                    lc_cache_file.write_text(json.dumps(row_lc, indent=2) + "\n")
                    lc_rows.append(row_lc)
                    keras.backend.clear_session()
            _write_csv(out_dir / "learning_curves_ntrain.csv", lc_rows)

    lc_summary: Dict[str, Any] = {"n_train_values": ntrain_values, "seeds": lc_seeds, "detectors": {}}
    for dkey in ("carla_transformer_ssa", "pca_d10", "mahalanobis_lw", "raw_knn5"):
        lc_summary["detectors"][dkey] = {}
        for n_tr in ntrain_values:
            sub = [r for r in lc_rows if r["detector_key"] == dkey and int(r["n_train"]) == n_tr]
            lc_summary["detectors"][dkey][str(n_tr)] = {
                m: _agg([float(x[m]) for x in sub])
                for m in ("zero_p975_f1", "zero_p95_f1", "auc_roc", "auc_pr_pi05")
            }
    (out_dir / "learning_curves_summary.json").write_text(json.dumps(lc_summary, indent=2) + "\n")

    # Build prevalence-adjusted PR curves at pi = 5% for Figure A
    def _prevalence_pr_curve(scores_1d: np.ndarray, pi: float = 0.05) -> Dict[str, Any]:
        from sklearn.metrics import roc_curve
        from src.evaluation.fault_analysis import compute_prevalence_adjusted_prauc

        fpr, tpr, _ = roc_curve(test_labels, scores_1d)
        denom = pi * tpr + (1.0 - pi) * fpr
        prec_pi = np.where(denom > 0.0, (pi * tpr) / np.maximum(denom, 1e-15), 1.0)
        prec_pi[0] = 1.0
        ap_pi05 = float(compute_prevalence_adjusted_prauc(test_labels, scores_1d, target_prevalence=pi))
        return {
            "recall": tpr.tolist(),
            "precision_pi05": prec_pi.tolist(),
            "ap_pi05": ap_pi05,
            "ap_label": f"{ap_pi05:.3f}",
        }

    def _median_seed_curve(s_list: List[np.ndarray], reg_stats: Dict[str, Any], m_key: str) -> Dict[str, Any]:
        aps = [_prevalence_pr_curve(s, 0.05)["ap_pi05"] for s in s_list]
        med_idx = int(np.argsort(aps)[len(aps) // 2])
        c = _prevalence_pr_curve(s_list[med_idx], 0.05)
        c["ap_pi05"] = float(reg_stats[m_key]["mean"])
        c["ap_label"] = f"{reg_stats[m_key]['mean']:.3f}+/-{reg_stats[m_key]['sample_std']:.3f}"
        return c

    pr_payload = {
        "CARLA-Transformer (SSA)": _median_seed_curve(
            ssa_seed_test_scores_zero,
            summary_10seed["regimes"]["tf_physics_ssa_fixed_load"],
            "zero_knn5_auc_pr_pi05",
        ),
        "PCA Reconstruction (d=10)": _prevalence_pr_curve(pca10.score(dataset.test_data), 0.05),
        "Mahalanobis (Ledoit-Wolf)": _prevalence_pr_curve(maha_lw.score(dataset.test_data), 0.05),
        "AE-Transformer (MSE)": _median_seed_curve(
            mse_seed_test_scores_zero,
            summary_10seed["regimes"]["tf_recon_only_mse"],
            "auc_pr_pi05",
        ),
        "CARLA-Transformer (Heuristic)": _median_seed_curve(
            heur_seed_test_scores_zero,
            summary_10seed["regimes"]["tf_heuristic"],
            "zero_knn5_auc_pr_pi05",
        ),
        "Raw k-NN (k=5)": _prevalence_pr_curve(knn5_raw.score(dataset.test_data), 0.05),
    }

    generate_extended_figures(
        out_dir=out_dir,
        plots_dir=plots_dir,
        frequencies=frequencies,
        pr_curves_payload=pr_payload,
        summary_10seed=summary_10seed,
        lc_summary=lc_summary,
        attn_audit=attn_audit,
        pca_fragility=pca_fragility,
    )
    print("\nCompleted full extended dataset_02 study successfully.")


if __name__ == "__main__":
    main()
