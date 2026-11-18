#!/usr/bin/env python3
"""5-Seed Paired Ablation & 5-Mode Strict LOFO for CARLA-Transformer on dataset_02.

Tests whether CARLA-Transformer's breakthrough on variable-load `dataset_02`
(F1 = 0.9493 +/- 0.0046, Zero-Label p97.5 F1 = 0.9428 +/- 0.0119, PR-AUC@5% = 0.9271)
requires physics-informed SSA contrastive learning (`physics_ssa_fixed_load` /
`physics_ssa_load_adapt`) vs `recon_only_mse`, `recon_only_latent`, and `heuristic`.
Reuses the 5 already-trained `encoders/carla_transformer/seed_*` checkpoints for
`physics_ssa_load_adapt`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_dataset02_mlwa_benchmark import (
    FAULT_MODE_TO_DATASET_COMP,
    _active_fault_components,
    _evaluate_deep_model,
    _train_or_load_model,
    _write_csv,
    prepare_or_load_stratified_dataset02,
)
from src.data.physics_anomaly import BuckConverter, available_fault_modes
from src.evaluation.fault_analysis import (
    compute_per_fault_recall,
    inject_fra_measurement_noise,
    paired_sign_flip_test,
)


def main() -> None:
    data_dir = ROOT / "data" / "buck" / "buck_data_v2" / "dataset_02"
    frozen_exp = ROOT / "experiments" / "dataset_01_run_06_corrected"
    study_exp = ROOT / "experiments" / "dataset_02_mlwa_study"
    cache_dir = ROOT / "cache"
    out_dir = ROOT / "papers" / "results" / "mlwa_study_dataset02"
    seeds = [200042, 200043, 200044, 200045, 200046]
    epochs = 150
    batch_size = 64

    tf_cfg = json.loads((frozen_exp / "carla_transformer" / "model_config.json").read_text())
    dataset = prepare_or_load_stratified_dataset02(data_dir, study_exp, cache_dir)
    splits = dataset._splits
    test_variations = [dataset._metadata[int(i)].variations for i in splits["test_idx"]]
    fv_norm_mask = np.asarray(dataset.fit_val_labels, dtype=int) == 0

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
    rows: List[Dict[str, Any]] = []
    cached_la_seed0 = None

    print("=== [Transformer Ablation on dataset_02] ===")
    for seed in seeds:
        # 1. Recon-only Transformer (MSE & Latent)
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
        for sub_name, space in [
            ("tf_recon_only_mse", "reconstruction_mse"),
            ("tf_recon_only_latent", "latent"),
        ]:
            ev = _evaluate_deep_model(
                rec_mdl, dataset, space, test_variations,
                noisy_test_015, noisy_fv_norm_015, noisy_tr_015,
            )
            ev.pop("scores")
            ev.pop("predictions")
            rows.append(
                {"regime": sub_name, "seed": seed, "epochs_trained": rec_prov["epochs_trained"], **ev}
            )

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
        ev_h.pop("scores")
        ev_h.pop("predictions")
        rows.append(
            {"regime": "tf_heuristic", "seed": seed, "epochs_trained": heur_prov["epochs_trained"], **ev_h}
        )

        # 3. Physics SSA Fixed-Load CARLA-Transformer
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
        ev_f.pop("scores")
        ev_f.pop("predictions")
        rows.append(
            {
                "regime": "tf_physics_ssa_fixed_load",
                "seed": seed,
                "epochs_trained": fix_prov["epochs_trained"],
                **ev_f,
            }
        )

        # 4. Physics SSA Load-Adaptive CARLA-Transformer (reuses encoders/carla_transformer/seed_*)
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
        if seed == seeds[0]:
            cached_la_seed0 = la_mdl
        ev_la = _evaluate_deep_model(
            la_mdl, dataset, "projection", test_variations,
            noisy_test_015, noisy_fv_norm_015, noisy_tr_015,
        )
        ev_la.pop("scores")
        ev_la.pop("predictions")
        rows.append(
            {
                "regime": "tf_physics_ssa_load_adapt",
                "seed": seed,
                "epochs_trained": la_prov["epochs_trained"],
                **ev_la,
            }
        )

        # 5. Physics SSA Load-Adaptive + FRA Denoising CARLA-Transformer
        fra_mdl, fra_prov = _train_or_load_model(
            arch_cfg=tf_cfg,
            regime="tf_physics_ssa_load_adapt_fra",
            seed=seed,
            run_dir=study_exp / "ablation_transformer" / "physics_ssa_load_adapt_fra" / f"seed_{seed}",
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            converter_spec=ssa_spec,
            use_physics=True,
            adapt_load=True,
            strict_margin=True,
            fra_jitter_db=0.25,
            fra_jitter_deg=1.25,
        )
        ev_fra = _evaluate_deep_model(
            fra_mdl, dataset, "projection", test_variations,
            noisy_test_015, noisy_fv_norm_015, noisy_tr_015,
        )
        ev_fra.pop("scores")
        ev_fra.pop("predictions")
        rows.append(
            {
                "regime": "tf_physics_ssa_load_adapt_fra",
                "seed": seed,
                "epochs_trained": fra_prov["epochs_trained"],
                **ev_fra,
            }
        )
        _write_csv(out_dir / "ablation_transformer_multiseed.csv", rows)

    regimes = [
        "tf_recon_only_mse",
        "tf_recon_only_latent",
        "tf_heuristic",
        "tf_physics_ssa_fixed_load",
        "tf_physics_ssa_load_adapt",
        "tf_physics_ssa_load_adapt_fra",
    ]
    metrics = [
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
    by_rs = {(r["regime"], r["seed"]): r for r in rows}
    summary: Dict[str, Any] = {"seeds": seeds, "regimes": {}, "paired_comparisons": {}}
    for reg in regimes:
        rr = [by_rs[(reg, s)] for s in seeds]
        summary["regimes"][reg] = {
            m: {
                "mean": float(np.mean([x[m] for x in rr])),
                "sample_std": float(np.std([x[m] for x in rr], ddof=1)),
                "values": [float(x[m]) for x in rr],
            }
            for m in metrics
        }
    for reg_a, reg_b in [
        ("tf_physics_ssa_load_adapt", "tf_physics_ssa_fixed_load"),
        ("tf_physics_ssa_load_adapt", "tf_heuristic"),
        ("tf_physics_ssa_load_adapt", "tf_recon_only_mse"),
        ("tf_physics_ssa_load_adapt", "tf_recon_only_latent"),
        ("tf_physics_ssa_load_adapt_fra", "tf_physics_ssa_load_adapt"),
    ]:
        summary["paired_comparisons"][f"{reg_a}_vs_{reg_b}"] = {
            m: paired_sign_flip_test(
                [by_rs[(reg_a, s)][m] for s in seeds],
                [by_rs[(reg_b, s)][m] for s in seeds],
            )
            for m in ("f1", "zero_knn5_p975_f1", "auc_roc", "auc_pr_pi05", "fra015_matched_p975_f1")
        }
    (out_dir / "ablation_transformer_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # 5-Mode Strict LOFO with CARLA-Transformer
    print("\n=== [Transformer 5-Mode Strict LOFO on dataset_02] ===")
    lofo_seed = seeds[0]
    all_modes = available_fault_modes()
    fv_variations = [dataset._metadata[int(i)].variations for i in splits["fit_val_idx"]]
    assert cached_la_seed0 is not None
    indist_ev = _evaluate_deep_model(
        cached_la_seed0, dataset, "projection", test_variations,
        noisy_test_015, noisy_fv_norm_015, noisy_tr_015,
    )
    indist_pfr = compute_per_fault_recall(indist_ev["predictions"], test_variations)

    lofo_rows: List[Dict[str, Any]] = []
    for held_mode in all_modes:
        held_comp = FAULT_MODE_TO_DATASET_COMP[held_mode]
        kept_modes = [m for m in all_modes if m != held_mode]
        fv_keep_mask = np.array(
            [
                (int(lbl) == 0) or (held_comp not in _active_fault_components(v))
                for lbl, v in zip(dataset.fit_val_labels, fv_variations)
            ],
            dtype=bool,
        )
        lofo_mdl, lofo_prov = _train_or_load_model(
            arch_cfg=tf_cfg,
            regime=f"tf_lofo_no_{held_mode}",
            seed=lofo_seed,
            run_dir=study_exp / "lofo_transformer" / f"no_{held_mode}_seed_{lofo_seed}",
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            converter_spec=ssa_spec,
            fault_modes=kept_modes,
            use_physics=True,
            adapt_load=True,
            strict_margin=True,
        )
        ev_lofo = _evaluate_deep_model(
            lofo_mdl, dataset, "projection", test_variations,
            noisy_test_015, noisy_fv_norm_015, noisy_tr_015,
            fit_val_mask=fv_keep_mask,
        )
        pfr_lofo = compute_per_fault_recall(ev_lofo["predictions"], test_variations)

        single_held_mask = np.array(
            [_active_fault_components(v) == [held_comp] for v in test_variations], dtype=bool
        )
        any_held_mask = np.array(
            [held_comp in _active_fault_components(v) for v in test_variations], dtype=bool
        )
        healthy_test_mask = np.asarray(dataset.test_labels, dtype=int) == 0
        y_int = np.asarray(dataset.test_labels, dtype=int)

        sub_single = healthy_test_mask | single_held_mask
        sub_any = healthy_test_mask | any_held_mask
        lofo_rows.append(
            {
                "held_out_fault_mode": held_mode,
                "dataset_component": held_comp,
                "n_test_single_fault": int(pfr_lofo[held_comp]["n_single"]),
                "n_test_any_fault": int(pfr_lofo[held_comp]["n_any"]),
                "lofo_recall_single": float(pfr_lofo[held_comp]["recall_single"]),
                "indist_recall_single": float(indist_pfr[held_comp]["recall_single"]),
                "lofo_recall_any": float(pfr_lofo[held_comp]["recall_any"]),
                "indist_recall_any": float(indist_pfr[held_comp]["recall_any"]),
                "lofo_single_auc_roc": float(
                    roc_auc_score(y_int[sub_single], ev_lofo["scores"][sub_single])
                ),
                "indist_single_auc_roc": float(
                    roc_auc_score(y_int[sub_single], indist_ev["scores"][sub_single])
                ),
                "lofo_any_auc_roc": float(
                    roc_auc_score(y_int[sub_any], ev_lofo["scores"][sub_any])
                ),
                "indist_any_auc_roc": float(
                    roc_auc_score(y_int[sub_any], indist_ev["scores"][sub_any])
                ),
                "lofo_overall_f1": float(ev_lofo["f1"]),
                "lofo_zero_knn5_p975_f1": float(ev_lofo["zero_knn5_p975_f1"]),
                "epochs_trained": int(lofo_prov["epochs_trained"]),
            }
        )
        _write_csv(out_dir / "lofo_transformer_results.csv", lofo_rows)
    print("Completed Transformer ablation & 5-mode LOFO on dataset_02.")


if __name__ == "__main__":
    main()
