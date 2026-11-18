#!/usr/bin/env python3
"""Train and evaluate the FRA-Denoising Physics-Informed CARLA variant across 5 seeds.

Compares ``physics_exact_ssa_fra_denoising`` against clean-trained ``physics_exact_ssa``
and classical baselines (``Raw k-NN k=1,5``, ``Mahalanobis Ledoit-Wolf``) across
clean and swept-sine FRA noisy regimes (0.00 dB to 0.60 dB), under both:
- Fixed benchtop threshold deployment (fit_val_f1 and zero-anomaly unsupervised p97.5), and
- Noise-matched recalibrated threshold & threshold-free ranking (ROC-AUC, PR-AUC@5%).
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import keras
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src  # noqa: F401
from scripts.run_mlwa_technical_study import (
    PERCENTILES,
    _active_fault_components,
    _evaluate_trained_model,
    _f1,
    _get_embeddings_batched,
    _load_frozen_dataset,
    _model_weights_sha256,
    _select_embedding_detector,
    _write_csv,
)
from src.data.physics_anomaly import BuckConverter
from src.evaluation.baselines import MahalanobisDetector, RawKNNDetector
from src.evaluation.fault_analysis import (
    compute_prevalence_adjusted_prauc,
    inject_fra_measurement_noise,
    paired_sign_flip_test,
)
from src.losses.contrastive import anomaly_score_from_embeddings
from src.models.contrastive_ae import ContrastiveAutoencoder
from src.training.carla_trainer import CARLAConfig, CARLATrainer


def _train_or_load_fra_variant(
    selected: Dict[str, Any],
    seed: int,
    run_dir: Path,
    dataset,
    epochs: int,
    batch_size: int,
    fra_jitter_db: float,
    fra_jitter_deg: float,
    fra_clean_prob: float,
) -> tuple[ContrastiveAutoencoder, Dict[str, Any]]:
    best_ckpt = run_dir / "checkpoints" / "best"
    prov_file = run_dir / "training_provenance.json"
    if (best_ckpt / "autoencoder.weights.h5").exists() and prov_file.exists():
        prov = json.loads(prov_file.read_text())
        model = ContrastiveAutoencoder.load(str(best_ckpt), config=selected)
        return model, prov

    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    config = CARLAConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=float(selected["learning_rate"]),
        reconstruction_weight=float(selected["reconstruction_weight"]),
        contrastive_weight=float(selected["contrastive_weight"]),
        temperature=float(selected["temperature"]),
        anomaly_ratio=1.0,
        seed=seed,
        use_physics_anomalies=True,
        resume_from_latest=False,
        scoring_method=str(selected["scoring_method"]),
        k_neighbors=int(selected["k_neighbors"]),
        fra_jitter_db=fra_jitter_db,
        fra_jitter_deg=fra_jitter_deg,
        fra_clean_prob=fra_clean_prob,
        denoising_reconstruction=True,
        checkpoint_dir=str(run_dir / "checkpoints"),
        log_dir=str(run_dir / "logs"),
        early_stopping=True,
        patience=25,
        save_best_only=True,
    )
    trainer = CARLATrainer(config, experiment_name=f"physics_ssa_fra_{seed}")
    trainer.create_model(
        input_shape=dataset.input_shape,
        latent_dim=int(selected["latent_dim"]),
        projection_dim=int(selected["projection_dim"]),
        encoder_type=str(selected["encoder_type"]),
        encoder_filters=selected["encoder_filters"],
        encoder_units=selected.get("encoder_units", [256, 128, 64]),
        kernel_size=int(selected["kernel_size"]),
        num_heads=int(selected["num_heads"]),
        dropout_rate=float(selected["dropout_rate"]),
    )
    init_sha = _model_weights_sha256(trainer.model)
    trainer.set_physics_context(
        frequencies=dataset.frequencies,
        scaler=dataset.preprocessor.scaler,
        use_real_imag=dataset.preprocessor.use_real_imag,
        converter_spec=BuckConverter(include_parasitic_drop=True, rds_low=0.0),
    )
    trainer.setup_training()

    t0 = time.perf_counter()
    history = trainer.train(dataset.train_data, dataset.val_data, verbose=0)
    elapsed = time.perf_counter() - t0

    prov = {
        "regime": "physics_exact_ssa_fra_denoising",
        "seed": seed,
        "fra_jitter_db": fra_jitter_db,
        "fra_jitter_deg": fra_jitter_deg,
        "fra_clean_prob": fra_clean_prob,
        "initial_model_weights_sha256": init_sha,
        "final_best_weights_sha256": _model_weights_sha256(trainer.model),
        "training_seconds": elapsed,
        "epochs_trained": len(history["loss"]),
    }
    prov_file.write_text(json.dumps(prov, indent=2) + "\n")
    print(
        f"  Trained physics_exact_ssa_fra_denoising (seed={seed}) in {elapsed:.1f}s "
        f"({len(history['loss'])} epochs, init={init_sha[:10]})"
    )
    return trainer.model, prov


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-experiment", default="experiments/dataset_01_run_06_corrected")
    parser.add_argument("--study-experiment", default="experiments/dataset_01_mlwa_study")
    parser.add_argument("--data-dir", default="data/buck/buck_data/dataset_01")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--out-dir", default="papers/results/mlwa_study")
    parser.add_argument("--seeds", type=int, nargs="+", default=[200042, 200043, 200044, 200045, 200046])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--fra-jitter-db", type=float, default=0.35)
    parser.add_argument("--fra-jitter-deg", type=float, default=1.75)
    parser.add_argument("--fra-clean-prob", type=float, default=0.25)
    args = parser.parse_args()

    frozen_exp = ROOT / args.frozen_experiment
    study_exp = ROOT / args.study_experiment
    data_dir = ROOT / args.data_dir
    cache_dir = ROOT / args.cache_dir
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = _load_frozen_dataset(data_dir, frozen_exp / "carla_conv1d", cache_dir)
    selected = json.loads((frozen_exp / "carla_conv1d" / "model_config.json").read_text())
    splits = dataset._splits
    test_variations = [dataset._metadata[int(i)].variations for i in splits["test_idx"]]
    active5_mask = np.array(
        [_active_fault_components(v) != ["Rds_2"] for v in test_variations],
        dtype=bool,
    )
    test_labels = np.asarray(dataset.test_labels, dtype=int)
    fit_val_labels = np.asarray(dataset.fit_val_labels, dtype=int)
    fv_normal_mask = fit_val_labels == 0

    raw_test = dataset._raw_data[splits["test_idx"]]
    raw_fv = dataset._raw_data[splits["fit_val_idx"]]
    noisy_test_proc_015 = dataset.preprocessor.transform(
        inject_fra_measurement_noise(raw_test, 0.15, 0.75, seed=100)
    )
    noisy_test_proc_030 = dataset.preprocessor.transform(
        inject_fra_measurement_noise(raw_test, 0.30, 1.50, seed=100)
    )

    seeds = list(args.seeds)
    fra_models: Dict[int, ContrastiveAutoencoder] = {}
    clean_ssa_models: Dict[int, ContrastiveAutoencoder] = {}
    multiseed_rows: List[Dict[str, Any]] = []

    print(f"\n=== Training & Evaluating FRA-Denoising CARLA across {len(seeds)} seeds ===")
    for seed in seeds:
        model_fra, prov_fra = _train_or_load_fra_variant(
            selected=selected,
            seed=seed,
            run_dir=study_exp / "ablation" / "physics_exact_ssa_fra_denoising" / f"seed_{seed}",
            dataset=dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            fra_jitter_db=args.fra_jitter_db,
            fra_jitter_deg=args.fra_jitter_deg,
            fra_clean_prob=args.fra_clean_prob,
        )
        fra_models[seed] = model_fra

        ev_fra = _evaluate_trained_model(
            model_fra, dataset, "projection", active5_mask, test_variations,
            noisy_test_proc_015, noisy_test_proc_030, batch_size=args.batch_size
        )
        ev_fra.pop("scores")
        ev_fra.pop("predictions")
        multiseed_rows.append(
            {
                "regime": "physics_exact_ssa_fra_denoising",
                "seed": seed,
                "initial_weights_sha256": prov_fra["initial_model_weights_sha256"],
                "epochs_trained": prov_fra["epochs_trained"],
                **ev_fra,
            }
        )
        _write_csv(out_dir / "fra_denoising_multiseed.csv", multiseed_rows)

        clean_ckpt = study_exp / "ablation" / "physics_exact_ssa" / f"seed_{seed}" / "checkpoints" / "best"
        clean_ssa_models[seed] = ContrastiveAutoencoder.load(str(clean_ckpt), config=selected)

    # Full FRA noise sweep comparing Classical vs Clean CARLA (5 seeds) vs FRA-Denoising CARLA (5 seeds)
    print("\n=== Running Full Multi-Seed FRA Noise Sweep (0.00 dB -> 0.60 dB) ===")
    knn1 = RawKNNDetector(n_neighbors=1).fit(dataset.train_data)
    knn1.calibrate_threshold(dataset.fit_val_data, fit_val_labels)
    knn1_u975 = float(np.percentile(knn1.score(dataset.fit_val_data[fv_normal_mask]), 97.5))

    knn5 = RawKNNDetector(n_neighbors=5).fit(dataset.train_data)
    knn5.calibrate_threshold(dataset.fit_val_data, fit_val_labels)
    knn5_u975 = float(np.percentile(knn5.score(dataset.fit_val_data[fv_normal_mask]), 97.5))

    maha = MahalanobisDetector(ridge=None).fit(dataset.train_data)
    maha.calibrate_threshold(dataset.fit_val_data, fit_val_labels)
    maha_u975 = float(np.percentile(maha.score(dataset.fit_val_data[fv_normal_mask]), 97.5))

    # Precompute clean reference banks and clean calibrated thresholds for all 5 seeds of both CARLA variants
    carla_banks: Dict[str, Dict[int, Dict[str, Any]]] = {
        "CARLA-SSA (clean-trained)": {},
        "CARLA-SSA-FRA (denoising)": {},
    }
    for label_key, mdict in [
        ("CARLA-SSA (clean-trained)", clean_ssa_models),
        ("CARLA-SSA-FRA (denoising)", fra_models),
    ]:
        for seed, mdl in mdict.items():
            ref_emb = _get_embeddings_batched(mdl, dataset.train_data, key="projection", batch_size=args.batch_size)
            det = _select_embedding_detector(
                lambda x, m=mdl: _get_embeddings_batched(m, x, key="projection", batch_size=args.batch_size),
                ref_emb,
                dataset.fit_val_data,
                fit_val_labels,
                normalize=True,
                batch_size=args.batch_size,
            )
            fv_norm_emb = _get_embeddings_batched(
                mdl, dataset.fit_val_data[fv_normal_mask], key="projection", batch_size=args.batch_size
            )
            fv_norm_scores = anomaly_score_from_embeddings(
                fv_norm_emb, ref_emb, method=det["scoring_method"], k=det["k_neighbors"], normalize=True
            )
            carla_banks[label_key][seed] = {
                "model": mdl,
                "ref_emb": ref_emb,
                "det": det,
                "th_clean": float(det["threshold"]),
                "th_u975": float(np.percentile(fv_norm_scores, 97.5)),
            }

    noise_levels = [
        (0.00, 0.00),
        (0.05, 0.25),
        (0.15, 0.75),
        (0.30, 1.50),
        (0.60, 3.00),
    ]
    mc_seeds = (100, 101, 102, 103, 104)
    sweep_rows: List[Dict[str, Any]] = []

    def _recal_f1(scores_fv: np.ndarray, scores_te: np.ndarray) -> tuple[float, float]:
        best_th = 0.0
        best_f1 = -1.0
        for pct in PERCENTILES:
            th_c = float(np.percentile(scores_fv[fv_normal_mask], pct))
            f1_c = _f1(fit_val_labels, (scores_fv > th_c).astype(int))
            if f1_c > best_f1:
                best_f1 = f1_c
                best_th = th_c
        preds = (scores_te > best_th).astype(int)
        return _f1(test_labels, preds), _f1(test_labels[active5_mask], preds[active5_mask])

    for sigma_db, sigma_deg in noise_levels:
        draws = [0] if (sigma_db == 0.0 and sigma_deg == 0.0) else list(mc_seeds)
        proc_tests = [
            dataset.preprocessor.transform(inject_fra_measurement_noise(raw_test, sigma_db, sigma_deg, seed=s))
            for s in draws
        ]
        proc_fvs = [
            dataset.preprocessor.transform(inject_fra_measurement_noise(raw_fv, sigma_db, sigma_deg, seed=s + 1000))
            for s in draws
        ]

        # Classical baselines
        for cname, cdet, cu975 in [
            ("Raw k-NN (k=1)", knn1, knn1_u975),
            ("Raw k-NN (k=5)", knn5, knn5_u975),
            ("Mahalanobis (Ledoit-Wolf)", maha, maha_u975),
        ]:
            f1_fix, a5_fix, u975_a5_fix, f1_rec, a5_rec, auc_f, auc_a5, pr05 = [], [], [], [], [], [], [], []
            for pt, pfv in zip(proc_tests, proc_fvs):
                st = cdet.score(pt)
                sfv = cdet.score(pfv)
                p_fix = (st > float(cdet.threshold_)).astype(int)
                p_u975 = (st > cu975).astype(int)
                f1_fix.append(_f1(test_labels, p_fix))
                a5_fix.append(_f1(test_labels[active5_mask], p_fix[active5_mask]))
                u975_a5_fix.append(_f1(test_labels[active5_mask], p_u975[active5_mask]))
                rf, ra5 = _recal_f1(sfv, st)
                f1_rec.append(rf)
                a5_rec.append(ra5)
                auc_f.append(float(roc_auc_score(test_labels, st)))
                auc_a5.append(float(roc_auc_score(test_labels[active5_mask], st[active5_mask])))
                pr05.append(compute_prevalence_adjusted_prauc(test_labels, st, 0.05))
            sweep_rows.append(
                {
                    "detector": cname,
                    "sigma_mag_db": sigma_db,
                    "sigma_phase_deg": sigma_deg,
                    "f1_fixed_th_mean": float(np.mean(f1_fix)),
                    "active5_f1_fixed_th_mean": float(np.mean(a5_fix)),
                    "unsup_p975_active5_f1_fixed_mean": float(np.mean(u975_a5_fix)),
                    "f1_recal_th_mean": float(np.mean(f1_rec)),
                    "active5_f1_recal_th_mean": float(np.mean(a5_rec)),
                    "auc_roc_mean": float(np.mean(auc_f)),
                    "active5_auc_roc_mean": float(np.mean(auc_a5)),
                    "auc_pr_pi05_mean": float(np.mean(pr05)),
                }
            )

        # CARLA multi-seed models
        for label_key, seed_dict in carla_banks.items():
            f1_fix, a5_fix, u975_a5_fix, f1_rec, a5_rec, auc_f, auc_a5, pr05 = [], [], [], [], [], [], [], []
            for seed, info in seed_dict.items():
                mdl = info["model"]
                ref_emb = info["ref_emb"]
                det = info["det"]
                for pt, pfv in zip(proc_tests, proc_fvs):
                    te_emb = _get_embeddings_batched(mdl, pt, key="projection", batch_size=args.batch_size)
                    fv_emb = _get_embeddings_batched(mdl, pfv, key="projection", batch_size=args.batch_size)
                    st = anomaly_score_from_embeddings(
                        te_emb, ref_emb, method=det["scoring_method"], k=det["k_neighbors"], normalize=True
                    )
                    sfv = anomaly_score_from_embeddings(
                        fv_emb, ref_emb, method=det["scoring_method"], k=det["k_neighbors"], normalize=True
                    )
                    p_fix = (st > info["th_clean"]).astype(int)
                    p_u975 = (st > info["th_u975"]).astype(int)
                    f1_fix.append(_f1(test_labels, p_fix))
                    a5_fix.append(_f1(test_labels[active5_mask], p_fix[active5_mask]))
                    u975_a5_fix.append(_f1(test_labels[active5_mask], p_u975[active5_mask]))
                    rf, ra5 = _recal_f1(sfv, st)
                    f1_rec.append(rf)
                    a5_rec.append(ra5)
                    auc_f.append(float(roc_auc_score(test_labels, st)))
                    auc_a5.append(float(roc_auc_score(test_labels[active5_mask], st[active5_mask])))
                    pr05.append(compute_prevalence_adjusted_prauc(test_labels, st, 0.05))
            sweep_rows.append(
                {
                    "detector": label_key,
                    "sigma_mag_db": sigma_db,
                    "sigma_phase_deg": sigma_deg,
                    "f1_fixed_th_mean": float(np.mean(f1_fix)),
                    "active5_f1_fixed_th_mean": float(np.mean(a5_fix)),
                    "unsup_p975_active5_f1_fixed_mean": float(np.mean(u975_a5_fix)),
                    "f1_recal_th_mean": float(np.mean(f1_rec)),
                    "active5_f1_recal_th_mean": float(np.mean(a5_rec)),
                    "auc_roc_mean": float(np.mean(auc_f)),
                    "active5_auc_roc_mean": float(np.mean(auc_a5)),
                    "auc_pr_pi05_mean": float(np.mean(pr05)),
                }
            )

    _write_csv(out_dir / "fra_denoising_noise_sweep.csv", sweep_rows)

    # Paired 5-seed summary against clean-trained physics_exact_ssa
    prev_summary = json.loads((out_dir / "ablation_summary.json").read_text())
    clean_ssa_stats = prev_summary["regimes"]["physics_exact_ssa"]

    metrics_to_summarize = [
        "f1",
        "active5_f1",
        "unsup_p975_f1",
        "unsup_p975_active5_f1",
        "auc_roc",
        "active5_auc_roc",
        "auc_pr_pi05",
        "recall_rds1_single",
        "fra_015db_f1",
        "fra_015db_active5_f1",
        "fra_030db_f1",
        "fra_030db_active5_f1",
    ]
    summary = {
        "seeds": seeds,
        "fra_training_params": {
            "fra_jitter_db": args.fra_jitter_db,
            "fra_jitter_deg": args.fra_jitter_deg,
            "fra_clean_prob": args.fra_clean_prob,
            "denoising_reconstruction": True,
        },
        "physics_exact_ssa_fra_denoising": {
            m: {
                "mean": float(np.mean([r[m] for r in multiseed_rows])),
                "sample_std": float(np.std([r[m] for r in multiseed_rows], ddof=1))
                if len(multiseed_rows) > 1
                else 0.0,
                "values": [float(r[m]) for r in multiseed_rows],
            }
            for m in metrics_to_summarize
        },
        "paired_vs_clean_physics_exact_ssa": {
            m: paired_sign_flip_test(
                [r[m] for r in multiseed_rows],
                clean_ssa_stats[m]["values"],
            )
            for m in (
                "f1",
                "active5_f1",
                "unsup_p975_f1",
                "unsup_p975_active5_f1",
                "auc_roc",
                "active5_auc_roc",
                "auc_pr_pi05",
                "recall_rds1_single",
                "fra_015db_f1",
                "fra_030db_f1",
            )
            if m in clean_ssa_stats
        },
    }
    (out_dir / "fra_denoising_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("  Completed FRA-Denoising multi-seed study and noise sweep.")


if __name__ == "__main__":
    main()
