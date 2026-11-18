#!/usr/bin/env python3
"""Paired physics-versus-heuristic negative-sampler ablation.

The selected Conv1D architecture and optimization hyperparameters are read from
the completed main experiment.  For each paired seed, only the negative sampler
changes.  Detector/scorer/threshold selection uses the persisted fit-validation
partition; the held-out test partition is evaluated once per trained replicate.
"""

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import keras
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Importing ``src`` first installs its JetPack 4.6 NumPy/SciPy compatibility
# aliases before Ubuntu 18.04's system scikit-learn imports SciPy.
import src  # noqa: F401
from sklearn.metrics import f1_score
from src.data.dataset import PowerConverterDataset
from src.data.preprocessor import DataPreprocessor
from src.evaluation.metrics import compute_classification_metrics
from src.training.carla_trainer import CARLAConfig, CARLATrainer


METHODS = ("knn", "centroid", "cosine", "mahalanobis")
K_VALUES = (1, 3, 5, 10, 20, 50)
PERCENTILES = (70.0, 80.0, 90.0, 95.0, 98.0, 99.0, 99.5, 99.9)
METRICS = ("accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr")


def _git_commit(root: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            universal_newlines=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    try:
        return float(f1_score(labels, predictions, zero_division=0))
    except TypeError:
        return float(f1_score(labels, predictions))


def _model_weights_sha256(model) -> str:
    """Hash the trainable detector state, including the projection head."""
    digest = hashlib.sha256()
    arrays = list(model.autoencoder.get_weights()) + list(
        model.projection_head.get_weights()
    )
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def select_detector(trainer: CARLATrainer, data: np.ndarray,
                    labels: np.ndarray, batch_size: int) -> dict:
    """Select scorer and threshold exclusively on fit-validation labels."""
    best = None
    normal = labels == 0
    if not np.any(normal) or np.all(normal):
        raise ValueError("Fit-validation selection requires both classes")
    for method in METHODS:
        k_values = K_VALUES if method == "knn" else (trainer.config.k_neighbors,)
        for k in k_values:
            if k > len(trainer.reference_embeddings):
                continue
            scores = trainer.compute_anomaly_scores(
                data, method=method, k=k, batch_size=batch_size
            )
            for percentile in PERCENTILES:
                threshold = float(np.percentile(scores[normal], percentile))
                score = _f1(labels, scores > threshold)
                candidate = {
                    "scoring_method": method,
                    "k_neighbors": int(k),
                    "normal_percentile": percentile,
                    "threshold": threshold,
                    "fit_validation_f1": score,
                }
                if best is None or score > best["fit_validation_f1"]:
                    best = candidate
    if best is None:
        raise RuntimeError("No valid detector configuration was evaluated")
    return best


def _load_dataset(data_dir: Path, model_dir: Path, cache_dir: Path):
    preprocessor = DataPreprocessor.load(str(model_dir / "preprocessor"))
    dataset = PowerConverterDataset(
        data_dir=str(data_dir), cache_dir=str(cache_dir), preprocessor=preprocessor
    )
    dataset.load()
    dataset.load_split_indices(str(model_dir / "split_indices.npz"))
    dataset.preprocess(fit=False)
    return dataset


def _create_trainer(selected: dict, sampler: str, seed: int, run_dir: Path,
                    dataset: PowerConverterDataset, epochs: int,
                    batch_size: int) -> CARLATrainer:
    config = CARLAConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=float(selected["learning_rate"]),
        reconstruction_weight=float(selected["reconstruction_weight"]),
        contrastive_weight=float(selected["contrastive_weight"]),
        temperature=float(selected["temperature"]),
        anomaly_ratio=1.0,
        seed=seed,
        use_physics_anomalies=(sampler == "physics"),
        resume_from_latest=False,
        scoring_method=str(selected["scoring_method"]),
        k_neighbors=int(selected["k_neighbors"]),
        checkpoint_dir=str(run_dir / "checkpoints"),
        log_dir=str(run_dir / "logs"),
        early_stopping=True,
        patience=20,
        save_best_only=True,
    )
    trainer = CARLATrainer(config, experiment_name=f"ablation_{sampler}_{seed}")
    trainer.create_model(
        input_shape=dataset.input_shape,
        latent_dim=int(selected["latent_dim"]),
        projection_dim=int(selected["projection_dim"]),
        encoder_type=str(selected["encoder_type"]),
        encoder_filters=selected["encoder_filters"],
        encoder_units=selected.get("encoder_units", [64, 32]),
        kernel_size=int(selected["kernel_size"]),
        num_heads=int(selected["num_heads"]),
        dropout_rate=float(selected["dropout_rate"]),
    )
    trainer.set_physics_context(
        frequencies=dataset.frequencies,
        scaler=dataset.preprocessor.scaler,
        use_real_imag=dataset.preprocessor.use_real_imag,
    )
    trainer.setup_training()
    return trainer


def run_replicate(selected: dict, sampler: str, seed: int, run_dir: Path,
                  dataset: PowerConverterDataset, epochs: int,
                  batch_size: int, source_commit: Optional[str],
                  train_only: bool = False) -> dict:
    training_record_path = run_dir / "ablation_training_provenance.json"
    metrics_path = run_dir / "metrics.json"
    if train_only and training_record_path.exists():
        return json.loads(training_record_path.read_text())
    if not train_only and metrics_path.exists():
        return json.loads(metrics_path.read_text())

    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    trainer = _create_trainer(
        selected, sampler, seed, run_dir, dataset, epochs, batch_size
    )
    initial_weights_sha256 = _model_weights_sha256(trainer.model)
    started = time.perf_counter()
    history = trainer.train(dataset.train_data, dataset.val_data, verbose=1)
    training_seconds = time.perf_counter() - started

    training_record = {
        "sampler": sampler,
        "seed": seed,
        "architecture": selected["encoder_type"],
        "selected_hyperparameters": selected,
        "training_seconds": training_seconds,
        "epochs_trained": len(history["loss"]),
        "train_rows": int(len(dataset.train_data)),
        "fit_validation_rows": int(len(dataset.fit_val_data)),
        "test_rows": int(len(dataset.test_data)),
        "source_commit": source_commit,
        "random_seed_api": "keras.utils.set_random_seed",
        "training_host": platform.node(),
        "training_machine": platform.machine(),
        "tensorflow_version": tf.__version__,
        "tensorflow_gpu_details": [
            tf.config.experimental.get_device_details(device)
            for device in tf.config.list_physical_devices("GPU")
        ],
        "initial_model_weights_sha256": initial_weights_sha256,
        "selected_final_model_weights_sha256": _model_weights_sha256(trainer.model),
        "test_inference_performed": False,
    }
    (run_dir / "model_config.json").write_text(
        json.dumps({"name": "carla_conv1d", "type": "carla",
                    "input_shape": list(dataset.input_shape), **selected}, indent=2)
        + "\n"
    )
    dataset.preprocessor.save(str(run_dir / "preprocessor"))
    shutil.copy2(dataset.model_split_path, run_dir / "split_indices.npz")
    training_record_path.write_text(json.dumps(training_record, indent=2) + "\n")
    if train_only:
        return training_record

    detector = select_detector(
        trainer, dataset.fit_val_data, dataset.fit_val_labels, batch_size
    )
    trainer.config.scoring_method = detector["scoring_method"]
    trainer.config.k_neighbors = detector["k_neighbors"]
    test_scores = trainer.compute_anomaly_scores(
        dataset.test_data,
        method=detector["scoring_method"],
        k=detector["k_neighbors"],
        batch_size=batch_size,
    )
    test_predictions = test_scores > detector["threshold"]
    metrics = compute_classification_metrics(
        test_predictions, dataset.test_labels, scores=test_scores
    ).to_dict()
    record = {
        "sampler": sampler,
        "seed": seed,
        "architecture": selected["encoder_type"],
        "selected_hyperparameters": selected,
        "detector_selection": detector,
        **training_record,
        "test_inference_performed": True,
        **metrics,
    }
    (run_dir / "threshold.json").write_text(json.dumps(detector, indent=2) + "\n")
    metrics_path.write_text(json.dumps(record, indent=2) + "\n")
    np.savez_compressed(
        run_dir / "test_outputs.npz",
        scores=test_scores,
        predictions=test_predictions.astype(np.int8),
        labels=dataset.test_labels.astype(np.int8),
    )
    return record


def write_summary(output_dir: Path, records: List[Dict], seeds: List[int]) -> None:
    rows = [{key: record.get(key) for key in ("sampler", "seed", *METRICS)}
            for record in records]
    with (output_dir / "replicate_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary = {"seeds": seeds, "replicates": len(seeds), "samplers": {},
               "paired_physics_minus_heuristic": {}}
    by_key = {(record["sampler"], record["seed"]): record for record in records}
    for sampler in ("physics", "heuristic"):
        selected = [by_key[(sampler, seed)] for seed in seeds]
        summary["samplers"][sampler] = {
            metric: {
                "mean": float(np.mean([row[metric] for row in selected])),
                "sample_std": float(np.std(
                    [row[metric] for row in selected], ddof=1
                )) if len(selected) > 1 else 0.0,
            }
            for metric in METRICS
        }
    for metric in METRICS:
        differences = [
            by_key[("physics", seed)][metric]
            - by_key[("heuristic", seed)][metric]
            for seed in seeds
        ]
        summary["paired_physics_minus_heuristic"][metric] = {
            "values": differences,
            "mean": float(np.mean(differences)),
            "sample_std": float(np.std(differences, ddof=1))
            if len(differences) > 1 else 0.0,
        }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment", default="experiments/dataset_01_run_06_corrected"
    )
    parser.add_argument("--data-dir", default="data/buck/buck_data/dataset_01")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--seeds", type=int, nargs="+", default=[200042, 200043, 200044])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--train-only", action="store_true",
        help="Train and preserve paired checkpoints without fit/test inference.",
    )
    mode.add_argument(
        "--summarize-only", action="store_true",
        help="Aggregate target-generated metrics.json files without training.",
    )
    args = parser.parse_args()

    experiment = Path(args.experiment)
    main_model = experiment / "carla_conv1d"
    config_path = main_model / "model_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            "The completed Conv1D main model is required before the ablation"
        )
    dataset = _load_dataset(Path(args.data_dir), main_model, Path(args.cache_dir))
    # Retain the exact persisted split path so each replicate can copy it.
    dataset.model_split_path = main_model / "split_indices.npz"
    selected = json.loads(config_path.read_text())
    if selected.get("encoder_type") != "conv1d":
        raise ValueError("The ablation is predeclared for the selected Conv1D model")

    output_dir = experiment / "ablations" / "negative_sampler"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        records = []
        for seed in args.seeds:
            for sampler in ("physics", "heuristic"):
                metrics_path = output_dir / sampler / f"seed_{seed}" / "metrics.json"
                if not metrics_path.exists():
                    raise FileNotFoundError(
                        f"Missing target-generated ablation metrics: {metrics_path}"
                    )
                record = json.loads(metrics_path.read_text())
                record.update({"sampler": sampler, "seed": seed})
                records.append(record)
        write_summary(output_dir, records, args.seeds)
        return

    source_commit = _git_commit(ROOT)
    records = []
    for seed in args.seeds:
        for sampler in ("physics", "heuristic"):
            run_dir = output_dir / sampler / f"seed_{seed}"
            records.append(run_replicate(
                selected, sampler, seed, run_dir, dataset,
                args.epochs, args.batch_size, source_commit,
                train_only=args.train_only,
            ))
    if not args.train_only:
        write_summary(output_dir, records, args.seeds)


if __name__ == "__main__":
    main()
