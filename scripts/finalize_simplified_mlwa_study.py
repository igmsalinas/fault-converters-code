#!/usr/bin/env python3
"""Finalize 100% Methodologically Unified Simplified Study on `dataset_02`.

Ensures every experiment, table, and figure in the variable-load FRA manuscript uses
the single streamlined method (`CARLA` with exact SSA nominal-load transfer ratios,
`adapt_load=False, strict_margin=False`, and strict zero-anomaly-label `p97.5`
threshold calibration):

1. Exact 10-seed Zero-Label (`p97.5` & `p95`) per-fault recall breakdown for
   `tf_physics_ssa_fixed_load`, `tf_recon_only_mse`, `tf_recon_only_latent`, and `tf_heuristic`.
2. 5-Architecture Backbone Comparison (`Transformer`, `Conv1D`, `MLP`, `LSTM`, `GRU`)
   across 5 paired seeds (`200042..200046`) with `adapt_load=False, strict_margin=False`.
3. 5-Mode Strict Leave-One-Fault-Out (`LOFO`) across 5 seeds (`200042..200046`)
   for `CARLA-Transformer (SSA)` (`adapt_load=False, strict_margin=False`) under
   Zero-Label `p97.5`.
4. FRA Instrument-Noise Sweep (`0.00` to `0.60 dB` / `0.00` to `3.00 deg`) on
   `dataset_02` across 10 seeds of `CARLA-Transformer (SSA)` vs classical baselines.
5. Simplified publication figures:
   - `mlwa_ds02_pr_and_ablation10.{pdf,png}` (4-bar simplified ablation, no `SSA+Load`)
   - `mlwa_ds02_bode_and_fault_recall.{pdf,png}` (PSIM variable-load Bode confound + Zero-Label per-fault recall)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import keras
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_dataset02_extended_study import _agg, _eval_zero_label_scores
from scripts.run_dataset02_mlwa_benchmark import (
    FAULT_MODE_TO_DATASET_COMP,
    _active_fault_components,
    _evaluate_deep_model,
    _get_embeddings_batched,
    _train_or_load_model,
    _write_csv,
    prepare_or_load_stratified_dataset02,
)
from src.data.physics_anomaly import BuckConverter, available_fault_modes
from src.evaluation.baselines import (
    IsolationForestDetector,
    MahalanobisDetector,
    PCAReconstructionDetector,
    RawKNNDetector,
)
from src.evaluation.fault_analysis import (
    compute_prevalence_adjusted_prauc,
    inject_fra_measurement_noise,
    paired_sign_flip_test,
)
from src.inference.scoring import EmbeddingAnomalyScorer


ZERO_METRICS = (
    "zero_p975_f1",
    "zero_p975_precision",
    "zero_p975_recall",
    "zero_p95_f1",
    "auc_roc",
    "auc_pr_test",
    "auc_pr_pi10",
    "auc_pr_pi05",
    "recall_passive_single",
    "recall_cout_single",
    "recall_esrc_single",
    "recall_lout_single",
    "recall_esrl_single",
    "recall_rds1_single",
    "recall_multi_fault",
)


def _zero_eval_model(
    model: Any,
    name: str,
    family: str,
    space: str,
    train_data: np.ndarray,
    fv_norm_data: np.ndarray,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    test_variations: List[Dict[str, float]],
) -> tuple[Dict[str, Any], np.ndarray]:
    if space == "reconstruction_mse":
        s_val = model.compute_reconstruction_error(fv_norm_data)
        s_te = model.compute_reconstruction_error(test_data)
    else:
        norm = space == "projection"
        tr_e = _get_embeddings_batched(model, train_data, key=space)
        fv_e = _get_embeddings_batched(model, fv_norm_data, key=space)
        te_e = _get_embeddings_batched(model, test_data, key=space)
        sc = EmbeddingAnomalyScorer(tr_e, method="knn", k=5, normalize=norm)
        s_val = sc.score(fv_e)
        s_te = sc.score(te_e)
    ev = _eval_zero_label_scores(name, family, s_val, s_te, test_labels, test_variations)
    return ev, np.asarray(s_te, dtype=np.float64)


def _generate_figures(
    plots_dir: Path,
    dataset: Any,
    frequencies: np.ndarray,
    pr_payload: Dict[str, Any],
    ablation_zero_summary: Dict[str, Any],
    enc_zero_summary: Dict[str, Any],
    classical_zero: Dict[str, Dict[str, Any]],
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

    # -------------------------------------------------------------------------
    # Figure 1: PSIM Variable-Load Bode Confound + Zero-Label Per-Fault Recall
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(11.2, 4.3), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.12, 1.0], hspace=0.14)
    ax_mag = fig.add_subplot(gs[0, 0])
    ax_phs = fig.add_subplot(gs[1, 0], sharex=ax_mag)
    ax_rec = fig.add_subplot(gs[:, 1])

    f_khz = frequencies / 1000.0
    splits = dataset._splits
    healthy_raw = dataset._raw_data[splits["train_idx"]]  # (300, 101, 2): [mag_dB, phase_deg]

    mag_p02 = np.percentile(healthy_raw[:, :, 0], 2.5, axis=0)
    mag_p98 = np.percentile(healthy_raw[:, :, 0], 97.5, axis=0)
    mag_med = np.median(healthy_raw[:, :, 0], axis=0)
    phs_p02 = np.percentile(healthy_raw[:, :, 1], 2.5, axis=0)
    phs_p98 = np.percentile(healthy_raw[:, :, 1], 97.5, axis=0)
    phs_med = np.median(healthy_raw[:, :, 1], axis=0)

    ax_mag.fill_between(
        f_khz, mag_p02, mag_p98, color="#B0C4DE", alpha=0.55, label="Healthy 95% env. (R_out in [2, 8] ohm)"
    )
    ax_mag.plot(f_khz, mag_med, color="#1F3A60", linewidth=1.6, label="Healthy median")
    ax_phs.fill_between(f_khz, phs_p02, phs_p98, color="#B0C4DE", alpha=0.55)
    ax_phs.plot(f_khz, phs_med, color="#1F3A60", linewidth=1.6)

    # Pick representative pure single-fault curves near median load from test set
    test_idx = splits["test_idx"]
    fault_styles = [
        ("Cout", "Fault: C_out (-50%)", "#0072B2", "-"),
        ("Esr_C", "Fault: ESR_C (+450%)", "#009E73", "-."),
        ("Esr_L", "Fault: ESR_L (+300%)", "#E69F00", "--"),
        ("Rds_1", "Fault: R_ds1 (+800%)", "#D55E00", ":"),
    ]
    for comp, _, color, ls in fault_styles:
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
            sign_str = f"{pct_val:+.0f}%"
            ax_mag.plot(
                f_khz, curve[:, 0], color=color, linestyle=ls, linewidth=1.6, label=f"Single {comp} ({sign_str})"
            )
            ax_phs.plot(f_khz, curve[:, 1], color=color, linestyle=ls, linewidth=1.6)

    ax_mag.axvspan(0.7, 2.0, color="#F0E442", alpha=0.18)
    ax_phs.axvspan(0.7, 2.0, color="#F0E442", alpha=0.18)
    ax_mag.set_xscale("log")
    ax_mag.set_ylabel("|G_vd(jw)| [dB]")
    ax_mag.set_title("(a) Switched PSIM Bode Signatures under 4x Variable Load")
    ax_mag.grid(True, which="both", alpha=0.25)
    ax_mag.legend(loc="lower left", fontsize=7.1, ncol=2, frameon=True)
    plt.setp(ax_mag.get_xticklabels(), visible=False)

    ax_phs.set_xscale("log")
    ax_phs.set_xlabel("Frequency [kHz]")
    ax_phs.set_ylabel("Phase [deg]")
    ax_phs.grid(True, which="both", alpha=0.25)

    # Panel (b): Zero-Label (p97.5) per-fault recall comparison
    fault_cols = [
        ("recall_cout_single", "C_out\n(n=23)"),
        ("recall_lout_single", "L_out\n(n=22)"),
        ("recall_esrc_single", "ESR_C\n(n=28)"),
        ("recall_esrl_single", "ESR_L\n(n=21)"),
        ("recall_rds1_single", "R_ds1\n(n=19)"),
        ("recall_multi_fault", "Multi\n(n=97)"),
    ]
    x = np.arange(len(fault_cols))
    bars_spec = [
        (
            "CARLA-Transformer (SSA)",
            [ablation_zero_summary["regimes"]["tf_physics_ssa_fixed_load"][k]["mean"] * 100.0 for k, _ in fault_cols],
            [ablation_zero_summary["regimes"]["tf_physics_ssa_fixed_load"][k]["sample_std"] * 100.0 for k, _ in fault_cols],
            "#0072B2",
        ),
        (
            "CARLA-Conv1D (SSA)",
            [enc_zero_summary["architectures"]["Conv1D"][k]["mean"] * 100.0 for k, _ in fault_cols],
            [enc_zero_summary["architectures"]["Conv1D"][k]["sample_std"] * 100.0 for k, _ in fault_cols],
            "#56B4E9",
        ),
        (
            "Mahalanobis (Ledoit-Wolf)",
            [classical_zero["mahalanobis_lw"][k] * 100.0 for k, _ in fault_cols],
            None,
            "#E69F00",
        ),
        (
            "Raw k-NN (k=5)",
            [classical_zero["raw_knn5"][k] * 100.0 for k, _ in fault_cols],
            None,
            "#666666",
        ),
    ]
    bw = 0.19
    offsets = [-1.5 * bw, -0.5 * bw, 0.5 * bw, 1.5 * bw]
    for (lbl, vals, errs, col), off in zip(bars_spec, offsets):
        ax_rec.bar(
            x + off,
            vals,
            bw,
            yerr=errs,
            capsize=2.2 if errs is not None else 0,
            color=col,
            alpha=0.90,
            label=lbl,
        )
    ax_rec.set_xticks(x)
    ax_rec.set_xticklabels([lbl for _, lbl in fault_cols])
    ax_rec.set_ylim(0.0, 108.0)
    ax_rec.set_ylabel("Zero-Label Recall at p97.5 [%]")
    ax_rec.set_title("(b) Per-Fault Recall Disentanglement (Zero-Label p97.5)")
    ax_rec.grid(True, axis="y", alpha=0.25)
    ax_rec.legend(loc="upper right", fontsize=7.5, frameon=True)

    for ext in ("pdf", "png"):
        fig.savefig(plots_dir / f"mlwa_ds02_bode_and_fault_recall.{ext}", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 2 (Simplified Figure A): PR Curves (pi=5%) + 4-Bar 10-Seed Ablation
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), constrained_layout=True)

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
        if label not in pr_payload:
            continue
        item = pr_payload[label]
        ax.plot(
            item["recall"],
            item["precision_pi05"],
            color=color,
            linestyle=ls,
            linewidth=lw,
            label=f"{label} (AP={item['ap_label']})",
        )
    ax.axhline(0.05, color="#999999", linestyle=":", linewidth=1.0, label="Random (pi = 0.05)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision at realistic fault prevalence (pi = 5%)")
    ax.set_title("(a) Precision-Recall under Variable Load (dataset_02)")
    ax.set_xlim(0.0, 1.01)
    ax.set_ylim(0.0, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", frameon=True, fontsize=7.5)

    ax2 = axes[1]
    reg_order = [
        ("tf_physics_ssa_fixed_load", "CARLA-TF\n(SSA)"),
        ("tf_recon_only_mse", "AE-TF\n(MSE)"),
        ("tf_recon_only_latent", "AE-TF\n(Latent kNN)"),
        ("tf_heuristic", "CARLA-TF\n(Heuristic)"),
    ]
    x = np.arange(len(reg_order))
    w = 0.35
    f1_means = [ablation_zero_summary["regimes"][k]["zero_p975_f1"]["mean"] for k, _ in reg_order]
    f1_stds = [ablation_zero_summary["regimes"][k]["zero_p975_f1"]["sample_std"] for k, _ in reg_order]
    pr_means = [ablation_zero_summary["regimes"][k]["auc_pr_pi05"]["mean"] for k, _ in reg_order]
    pr_stds = [ablation_zero_summary["regimes"][k]["auc_pr_pi05"]["sample_std"] for k, _ in reg_order]

    ax2.bar(
        x - w / 2,
        f1_means,
        w,
        yerr=f1_stds,
        capsize=3,
        color="#0072B2",
        alpha=0.88,
        label="Zero-Label F1 (p97.5)",
    )
    ax2.bar(
        x + w / 2,
        pr_means,
        w,
        yerr=pr_stds,
        capsize=3,
        color="#E69F00",
        alpha=0.88,
        label="PR-AUC (pi = 5%)",
    )
    ax2.axhline(
        classical_zero["mahalanobis_lw"]["zero_p975_f1"],
        color="#009E73",
        linestyle="--",
        linewidth=1.2,
        label="Mahalanobis LW Zero-Label F1",
    )
    ax2.axhline(
        classical_zero["mahalanobis_lw"]["auc_pr_pi05"],
        color="#009E73",
        linestyle=":",
        linewidth=1.2,
        label="Mahalanobis LW PR-AUC@5%",
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
    print(f"Saved simplified publication plots to {plots_dir}")


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
    cache_dir = ROOT / "cache"
    out_dir = ROOT / "papers" / "results" / "mlwa_study_dataset02"
    plots_dir = ROOT / "docs" / "plots"

    seeds_10 = list(range(200042, 200052))
    seeds_5 = list(range(200042, 200047))
    epochs = 150
    batch_size = 64

    tf_cfg = json.loads((frozen_exp / "carla_transformer" / "model_config.json").read_text())
    dataset = prepare_or_load_stratified_dataset02(data_dir, study_exp, cache_dir)
    splits = dataset._splits
    frequencies = np.asarray(dataset.frequencies, dtype=np.float64)
    test_labels = np.asarray(dataset.test_labels, dtype=int)
    test_variations = [dataset._metadata[int(i)].variations for i in splits["test_idx"]]
    fv_labels = np.asarray(dataset.fit_val_labels, dtype=int)
    fv_norm_mask = fv_labels == 0
    fv_norm_data = dataset.fit_val_data[fv_norm_mask]

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
    # 1. Exact 10-Seed Zero-Label Per-Fault Breakdown + FRA Noise Sweep Models
    # =========================================================================
    print("=== [1] 10-Seed Pure Zero-Label Breakdown & Checkpoint Loading ===")
    ablation_zero_rows: List[Dict[str, Any]] = []
    ssa_models_10: List[Any] = []
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
            epochs=epochs,
            batch_size=batch_size,
            contrastive_weight_override=0.0,
            use_physics=False,
        )
        ev_mse, s_mse = _zero_eval_model(
            rec_mdl, "tf_recon_only_mse", "ablation_zero", "reconstruction_mse",
            dataset.train_data, fv_norm_data, dataset.test_data, test_labels, test_variations,
        )
        mse_scores_10.append(s_mse)
        ablation_zero_rows.append({"regime": "tf_recon_only_mse", "seed": seed, **ev_mse})

        ev_lat, _ = _zero_eval_model(
            rec_mdl, "tf_recon_only_latent", "ablation_zero", "latent",
            dataset.train_data, fv_norm_data, dataset.test_data, test_labels, test_variations,
        )
        ablation_zero_rows.append({"regime": "tf_recon_only_latent", "seed": seed, **ev_lat})
        keras.backend.clear_session()

        heur_mdl, _ = _train_or_load_model(
            arch_cfg=tf_cfg,
            regime="tf_heuristic",
            seed=seed,
            run_dir=study_exp / "ablation_transformer" / "heuristic" / f"seed_{seed}",
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            use_physics=False,
        )
        ev_h, s_h = _zero_eval_model(
            heur_mdl, "tf_heuristic", "ablation_zero", "projection",
            dataset.train_data, fv_norm_data, dataset.test_data, test_labels, test_variations,
        )
        heur_scores_10.append(s_h)
        ablation_zero_rows.append({"regime": "tf_heuristic", "seed": seed, **ev_h})
        keras.backend.clear_session()

        fix_mdl, _ = _train_or_load_model(
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
        ev_f, s_f = _zero_eval_model(
            fix_mdl, "tf_physics_ssa_fixed_load", "ablation_zero", "projection",
            dataset.train_data, fv_norm_data, dataset.test_data, test_labels, test_variations,
        )
        ssa_scores_10.append(s_f)
        ablation_zero_rows.append({"regime": "tf_physics_ssa_fixed_load", "seed": seed, **ev_f})
        ssa_models_10.append(fix_mdl)

    _write_csv(out_dir / "ablation_transformer_10seed_zero_label.csv", ablation_zero_rows)
    by_rz = {(r["regime"], r["seed"]): r for r in ablation_zero_rows}
    ablation_zero_summary: Dict[str, Any] = {"seeds": seeds_10, "regimes": {}, "paired_tests": {}}
    for reg in ("tf_physics_ssa_fixed_load", "tf_recon_only_mse", "tf_recon_only_latent", "tf_heuristic"):
        rr = [by_rz[(reg, s)] for s in seeds_10]
        ablation_zero_summary["regimes"][reg] = {m: _agg([float(x[m]) for x in rr]) for m in ZERO_METRICS}
    for reg_b in ("tf_recon_only_mse", "tf_recon_only_latent", "tf_heuristic"):
        ablation_zero_summary["paired_tests"][f"tf_physics_ssa_fixed_load_vs_{reg_b}"] = {
            m: paired_sign_flip_test(
                [by_rz[("tf_physics_ssa_fixed_load", s)][m] for s in seeds_10],
                [by_rz[(reg_b, s)][m] for s in seeds_10],
            )
            for m in ("zero_p975_f1", "zero_p95_f1", "auc_roc", "auc_pr_pi05", "recall_esrc_single", "recall_esrl_single", "recall_rds1_single")
        }
    (out_dir / "ablation_transformer_10seed_zero_label.json").write_text(
        json.dumps(ablation_zero_summary, indent=2) + "\n"
    )

    # =========================================================================
    # 2. FRA Instrument-Noise Sweep on dataset_02 (0.00 to 0.60 dB, 10 seeds)
    # =========================================================================
    print("\n=== [2] FRA Instrument-Noise Sweep on dataset_02 (10 seeds) ===")
    noise_grid = [
        (0.00, 0.00),
        (0.05, 0.25),
        (0.10, 0.50),
        (0.15, 0.75),
        (0.25, 1.25),
        (0.40, 2.00),
        (0.60, 3.00),
    ]
    pca8 = PCAReconstructionDetector(n_components=8, seed=42).fit(dataset.train_data)
    pca10 = PCAReconstructionDetector(n_components=10, seed=42).fit(dataset.train_data)
    maha_lw = MahalanobisDetector(ridge=None).fit(dataset.train_data)
    knn5_raw = RawKNNDetector(n_neighbors=5).fit(dataset.train_data)
    iforest = IsolationForestDetector(n_estimators=200, seed=42).fit(dataset.train_data)

    raw_tr = dataset._raw_data[splits["train_idx"]]
    raw_fvn = dataset._raw_data[splits["fit_val_idx"]][fv_norm_mask]
    raw_te = dataset._raw_data[splits["test_idx"]]

    fra_rows: List[Dict[str, Any]] = []
    fra_summary: Dict[str, Any] = {"noise_levels": [], "detectors": {}}

    for idx_n, (sig_db, sig_deg) in enumerate(noise_grid):
        lbl_n = f"{sig_db:.2f}dB_{sig_deg:.2f}deg"
        fra_summary["noise_levels"].append({"sigma_db": sig_db, "sigma_deg": sig_deg, "key": lbl_n})
        if sig_db == 0.0:
            tr_n = dataset.train_data
            fvn_n = fv_norm_data
            te_n = dataset.test_data
        else:
            tr_n = dataset.preprocessor.transform(
                inject_fra_measurement_noise(raw_tr, sig_db, sig_deg, seed=500 + idx_n * 10 + 1)
            )
            fvn_n = dataset.preprocessor.transform(
                inject_fra_measurement_noise(raw_fvn, sig_db, sig_deg, seed=500 + idx_n * 10 + 2)
            )
            te_n = dataset.preprocessor.transform(
                inject_fra_measurement_noise(raw_te, sig_db, sig_deg, seed=500 + idx_n * 10 + 3)
            )

        for cname, cdet in [
            ("pca_d8_cleanfit", pca8),
            ("pca_d10_cleanfit", pca10),
            ("mahalanobis_lw_cleanfit", maha_lw),
            ("raw_knn5_cleanfit", knn5_raw),
        ]:
            ev_c = _eval_zero_label_scores(
                cname, "fra_sweep", cdet.score(fvn_n), cdet.score(te_n), test_labels, test_variations
            )
            fra_rows.append(
                {"detector_key": cname, "seed": 0, "sigma_db": sig_db, "sigma_deg": sig_deg, **ev_c}
            )
            fra_summary["detectors"].setdefault(cname, {})[lbl_n] = {
                "zero_p975_f1": float(ev_c["zero_p975_f1"]),
                "auc_pr_pi05": float(ev_c["auc_pr_pi05"]),
                "auc_roc": float(ev_c["auc_roc"]),
            }

        # Also noisy-refit PCA (d=8 and d=10) when sigma_db > 0
        for d_val in (8, 10):
            det_nf = PCAReconstructionDetector(n_components=d_val, seed=42).fit(tr_n)
            ev_nf = _eval_zero_label_scores(
                f"pca_d{d_val}_noisyrefit",
                "fra_sweep",
                det_nf.score(fvn_n),
                det_nf.score(te_n),
                test_labels,
                test_variations,
            )
            fra_rows.append(
                {
                    "detector_key": f"pca_d{d_val}_noisyrefit",
                    "seed": 0,
                    "sigma_db": sig_db,
                    "sigma_deg": sig_deg,
                    **ev_nf,
                }
            )
            fra_summary["detectors"].setdefault(f"pca_d{d_val}_noisyrefit", {})[lbl_n] = {
                "zero_p975_f1": float(ev_nf["zero_p975_f1"]),
                "auc_pr_pi05": float(ev_nf["auc_pr_pi05"]),
                "auc_roc": float(ev_nf["auc_roc"]),
            }

        tf_seed_evs: List[Dict[str, Any]] = []
        for seed, mdl in zip(seeds_10, ssa_models_10):
            ev_tf, _ = _zero_eval_model(
                mdl, "carla_transformer_ssa", "fra_sweep", "projection",
                tr_n, fvn_n, te_n, test_labels, test_variations,
            )
            tf_seed_evs.append(ev_tf)
            fra_rows.append(
                {
                    "detector_key": "carla_transformer_ssa",
                    "seed": seed,
                    "sigma_db": sig_db,
                    "sigma_deg": sig_deg,
                    **ev_tf,
                }
            )
        fra_summary["detectors"].setdefault("carla_transformer_ssa", {})[lbl_n] = {
            m: _agg([float(x[m]) for x in tf_seed_evs])
            for m in ("zero_p975_f1", "auc_pr_pi05", "auc_roc")
        }

    _write_csv(out_dir / "fra_noise_sweep_dataset02.csv", fra_rows)
    (out_dir / "fra_noise_sweep_dataset02_summary.json").write_text(
        json.dumps(fra_summary, indent=2) + "\n"
    )
    del ssa_models_10
    keras.backend.clear_session()

    # =========================================================================
    # 3. 5-Architecture Backbone Comparison (`adapt_load=False`, 5 seeds)
    # =========================================================================
    print("\n=== [3] 5-Architecture Backbone Comparison (`adapt_load=False`, 5 seeds) ===")
    enc_rows: List[Dict[str, Any]] = []
    for subdir, arch_label in [
        ("carla_transformer", "Transformer"),
        ("carla_conv1d", "Conv1D"),
        ("carla_lstm", "LSTM"),
        ("carla_gru", "GRU"),
        ("carla_mlp", "MLP"),
    ]:
        acfg = json.loads((frozen_exp / subdir / "model_config.json").read_text())
        for seed in seeds_5:
            if subdir == "carla_transformer":
                rdir = study_exp / "ablation_transformer" / "physics_ssa_fixed_load" / f"seed_{seed}"
                reg_name = "tf_physics_ssa_fixed_load"
            elif subdir == "carla_conv1d":
                rdir = study_exp / "ablation" / "physics_ssa_fixed_load" / f"seed_{seed}"
                reg_name = "physics_ssa_fixed_load"
            else:
                rdir = study_exp / "encoders_ssa_fixed" / subdir / f"seed_{seed}"
                reg_name = f"enc_ssa_fixed_{subdir}"

            mdl, prov = _train_or_load_model(
                arch_cfg=acfg,
                regime=reg_name,
                seed=seed,
                run_dir=rdir,
                dataset=dataset,
                epochs=epochs,
                batch_size=batch_size,
                converter_spec=ssa_spec,
                use_physics=True,
                adapt_load=False,
                strict_margin=False,
            )
            ev_z, _ = _zero_eval_model(
                mdl, f"CARLA-{arch_label} (SSA)", "encoder_ssa_fixed_zero", "projection",
                dataset.train_data, fv_norm_data, dataset.test_data, test_labels, test_variations,
            )
            ev_cal = _evaluate_deep_model(
                mdl, dataset, "projection", test_variations,
                noisy_test_015, noisy_fv_norm_015, noisy_tr_015,
            )
            enc_rows.append(
                {
                    "architecture": arch_label,
                    "seed": seed,
                    "epochs_trained": prov["epochs_trained"],
                    "training_seconds": prov.get("training_seconds", 0.0),
                    "val_calibrated_f1": float(ev_cal["f1"]),
                    "fra015_matched_p975_f1": float(ev_cal["fra015_matched_p975_f1"]),
                    **ev_z,
                }
            )
            keras.backend.clear_session()
            _write_csv(out_dir / "encoder_comparison_ssa_fixed.csv", enc_rows)

    by_as = {(r["architecture"], r["seed"]): r for r in enc_rows}
    enc_summary: Dict[str, Any] = {"seeds": seeds_5, "architectures": {}, "paired_vs_transformer": {}}
    enc_metrics = list(ZERO_METRICS) + ["val_calibrated_f1", "fra015_matched_p975_f1"]
    for arch_label in ("Transformer", "Conv1D", "LSTM", "GRU", "MLP"):
        rr = [by_as[(arch_label, s)] for s in seeds_5]
        enc_summary["architectures"][arch_label] = {m: _agg([float(x[m]) for x in rr]) for m in enc_metrics}
        if arch_label != "Transformer":
            enc_summary["paired_vs_transformer"][f"Transformer_vs_{arch_label}"] = {
                m: paired_sign_flip_test(
                    [by_as[("Transformer", s)][m] for s in seeds_5],
                    [by_as[(arch_label, s)][m] for s in seeds_5],
                )
                for m in ("zero_p975_f1", "auc_pr_pi05", "auc_roc", "val_calibrated_f1")
            }
    (out_dir / "encoder_comparison_ssa_fixed_summary.json").write_text(
        json.dumps(enc_summary, indent=2) + "\n"
    )

    # =========================================================================
    # 4. 5-Mode Strict LOFO across 5 Seeds (`adapt_load=False`, Zero-Label p97.5)
    # =========================================================================
    print("\n=== [4] 5-Mode Strict LOFO (`adapt_load=False`, 5 seeds, Zero-Label p97.5) ===")
    all_modes = available_fault_modes()
    healthy_test_mask = test_labels == 0
    lofo_rows: List[Dict[str, Any]] = []

    # Pre-collect 5-seed in-distribution Zero-Label p97.5 scores & recalls
    indist_by_seed: Dict[int, Dict[str, Any]] = {}
    for idx_s, seed in enumerate(seeds_5):
        indist_by_seed[seed] = {
            "ev": by_rz[("tf_physics_ssa_fixed_load", seed)],
            "scores": ssa_scores_10[idx_s],
        }

    for held_mode in all_modes:
        held_comp = FAULT_MODE_TO_DATASET_COMP[held_mode]
        kept_modes = [m for m in all_modes if m != held_mode]
        single_held_mask = np.array(
            [_active_fault_components(v) == [held_comp] for v in test_variations], dtype=bool
        )
        any_held_mask = np.array(
            [held_comp in _active_fault_components(v) for v in test_variations], dtype=bool
        )
        sub_single = healthy_test_mask | single_held_mask
        sub_any = healthy_test_mask | any_held_mask
        comp_recall_key = {
            "Cout": "recall_cout_single",
            "Esr_C": "recall_esrc_single",
            "Lout": "recall_lout_single",
            "Esr_L": "recall_esrl_single",
            "Rds_1": "recall_rds1_single",
        }[held_comp]

        for seed in seeds_5:
            lofo_mdl, lofo_prov = _train_or_load_model(
                arch_cfg=tf_cfg,
                regime=f"tf_ssa_fixed_lofo_no_{held_mode}",
                seed=seed,
                run_dir=study_exp / "lofo_transformer_ssa_fixed" / f"no_{held_mode}" / f"seed_{seed}",
                dataset=dataset,
                epochs=epochs,
                batch_size=batch_size,
                converter_spec=ssa_spec,
                fault_modes=kept_modes,
                use_physics=True,
                adapt_load=False,
                strict_margin=False,
            )
            ev_lofo, s_lofo = _zero_eval_model(
                lofo_mdl, f"lofo_no_{held_mode}", "lofo_zero", "projection",
                dataset.train_data, fv_norm_data, dataset.test_data, test_labels, test_variations,
            )
            ind_ev = indist_by_seed[seed]["ev"]
            ind_s = indist_by_seed[seed]["scores"]

            lofo_rows.append(
                {
                    "held_out_fault_mode": held_mode,
                    "dataset_component": held_comp,
                    "seed": seed,
                    "n_test_single_fault": int(np.sum(single_held_mask)),
                    "n_test_any_fault": int(np.sum(any_held_mask)),
                    "lofo_zero_p975_recall_single": float(ev_lofo[comp_recall_key]),
                    "indist_zero_p975_recall_single": float(ind_ev[comp_recall_key]),
                    "lofo_single_auc_roc": float(roc_auc_score(test_labels[sub_single], s_lofo[sub_single])),
                    "indist_single_auc_roc": float(roc_auc_score(test_labels[sub_single], ind_s[sub_single])),
                    "lofo_any_auc_roc": float(roc_auc_score(test_labels[sub_any], s_lofo[sub_any])),
                    "indist_any_auc_roc": float(roc_auc_score(test_labels[sub_any], ind_s[sub_any])),
                    "lofo_zero_p975_overall_f1": float(ev_lofo["zero_p975_f1"]),
                    "indist_zero_p975_overall_f1": float(ind_ev["zero_p975_f1"]),
                    "lofo_auc_pr_pi05": float(ev_lofo["auc_pr_pi05"]),
                    "epochs_trained": int(lofo_prov["epochs_trained"]),
                }
            )
            keras.backend.clear_session()
            _write_csv(out_dir / "lofo_transformer_ssa_fixed_5seed.csv", lofo_rows)

    lofo_summary: Dict[str, Any] = {"seeds": seeds_5, "modes": {}}
    for held_mode in all_modes:
        held_comp = FAULT_MODE_TO_DATASET_COMP[held_mode]
        sub = [r for r in lofo_rows if r["held_out_fault_mode"] == held_mode]
        lofo_summary["modes"][held_comp] = {
            "held_out_fault_mode": held_mode,
            "n_test_single_fault": int(sub[0]["n_test_single_fault"]),
            "n_test_any_fault": int(sub[0]["n_test_any_fault"]),
            **{
                m: _agg([float(x[m]) for x in sub])
                for m in (
                    "lofo_zero_p975_recall_single",
                    "indist_zero_p975_recall_single",
                    "lofo_single_auc_roc",
                    "indist_single_auc_roc",
                    "lofo_any_auc_roc",
                    "indist_any_auc_roc",
                    "lofo_zero_p975_overall_f1",
                    "lofo_auc_pr_pi05",
                )
            },
        }
    (out_dir / "lofo_transformer_ssa_fixed_5seed_summary.json").write_text(
        json.dumps(lofo_summary, indent=2) + "\n"
    )

    # =========================================================================
    # 5. Generate Simplified Publication Figures
    # =========================================================================
    def _prevalence_pr_curve(scores_1d: np.ndarray, pi: float = 0.05) -> Dict[str, Any]:
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

    def _median_seed_curve(s_list: Sequence[np.ndarray], reg_stats: Dict[str, Any]) -> Dict[str, Any]:
        aps = [_prevalence_pr_curve(s, 0.05)["ap_pi05"] for s in s_list]
        med_idx = int(np.argsort(aps)[len(aps) // 2])
        c = _prevalence_pr_curve(s_list[med_idx], 0.05)
        c["ap_pi05"] = float(reg_stats["auc_pr_pi05"]["mean"])
        c["ap_label"] = f"{reg_stats['auc_pr_pi05']['mean']:.3f}+/-{reg_stats['auc_pr_pi05']['sample_std']:.3f}"
        return c

    classical_zero = {
        "pca_d10": _eval_zero_label_scores(
            "PCA (d=10)", "classical", pca10.score(fv_norm_data), pca10.score(dataset.test_data), test_labels, test_variations
        ),
        "mahalanobis_lw": _eval_zero_label_scores(
            "Mahalanobis LW", "classical", maha_lw.score(fv_norm_data), maha_lw.score(dataset.test_data), test_labels, test_variations
        ),
        "raw_knn5": _eval_zero_label_scores(
            "Raw k-NN (k=5)", "classical", knn5_raw.score(fv_norm_data), knn5_raw.score(dataset.test_data), test_labels, test_variations
        ),
        "iforest": _eval_zero_label_scores(
            "Isolation Forest", "classical", iforest.score(fv_norm_data), iforest.score(dataset.test_data), test_labels, test_variations
        ),
    }

    pr_payload = {
        "CARLA-Transformer (SSA)": _median_seed_curve(
            ssa_scores_10, ablation_zero_summary["regimes"]["tf_physics_ssa_fixed_load"]
        ),
        "PCA Reconstruction (d=10)": _prevalence_pr_curve(pca10.score(dataset.test_data), 0.05),
        "Mahalanobis (Ledoit-Wolf)": _prevalence_pr_curve(maha_lw.score(dataset.test_data), 0.05),
        "AE-Transformer (MSE)": _median_seed_curve(
            mse_scores_10, ablation_zero_summary["regimes"]["tf_recon_only_mse"]
        ),
        "CARLA-Transformer (Heuristic)": _median_seed_curve(
            heur_scores_10, ablation_zero_summary["regimes"]["tf_heuristic"]
        ),
        "Raw k-NN (k=5)": _prevalence_pr_curve(knn5_raw.score(dataset.test_data), 0.05),
    }

    _generate_figures(
        plots_dir=plots_dir,
        dataset=dataset,
        frequencies=frequencies,
        pr_payload=pr_payload,
        ablation_zero_summary=ablation_zero_summary,
        enc_zero_summary=enc_summary,
        classical_zero=classical_zero,
    )
    print("\nCompleted simplified unified dataset_02 study and figures.")


if __name__ == "__main__":
    main()
