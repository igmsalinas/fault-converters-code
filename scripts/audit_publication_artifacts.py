#!/usr/bin/env python3
"""Audit final manuscripts against citations, figures, and experiment artifacts."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "dataset_01_run_06_corrected"
MODELS = ("conv1d", "gru", "lstm", "mlp", "transformer")
PAPERS = {
    "mmas": ROOT / "papers" / "mmas_anomaly_detection",
    "jsc": ROOT / "papers" / "jsc_edge_deployment",
}


def _effective_trial_key(config: dict) -> str:
    encoder_type = config.get("encoder_type")
    inactive = {
        "conv1d": {"encoder_units", "num_heads"},
        "lstm": {"encoder_filters", "kernel_size", "num_heads"},
        "gru": {"encoder_filters", "kernel_size", "num_heads"},
        "mlp": {"encoder_filters", "kernel_size", "num_heads"},
        "transformer": {"encoder_units", "kernel_size"},
    }.get(encoder_type, set())
    effective = {key: value for key, value in config.items() if key not in inactive}
    return json.dumps(effective, sort_keys=True)


def _expected_timing_source(backend: str) -> str:
    """Return the required primary clock for a runtime label or backend key."""
    lowered = backend.lower()
    return (
        "cuda_event"
        if "tensorrt" in lowered or "keras-gpu" in lowered or "keras gpu" in lowered
        else "perf_counter"
    )


def _citation_keys(tex: str) -> set[str]:
    keys = set()
    for group in re.findall(r"\\cite\w*\{([^}]+)\}", tex):
        keys.update(key.strip() for key in group.split(",") if key.strip())
    return keys


def _bib_keys(bib: str) -> set[str]:
    return set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib, re.IGNORECASE))


def _audit_paper(name: str, directory: Path) -> list[str]:
    errors = []
    tex_path = directory / "main.tex"
    bib_path = directory / "references.bib"
    if not tex_path.exists() or not bib_path.exists():
        return [f"{name}: missing main.tex or references.bib"]
    tex = tex_path.read_text()
    bib = bib_path.read_text()

    missing_citations = sorted(_citation_keys(tex) - _bib_keys(bib))
    if missing_citations:
        errors.append(f"{name}: unresolved citation keys: {missing_citations}")
    duplicate_bib = sorted({
        key for key in _bib_keys(bib)
        if len(re.findall(r"@\w+\s*\{\s*" + re.escape(key) + r"\s*,", bib,
                          re.IGNORECASE)) > 1
    })
    if duplicate_bib:
        errors.append(f"{name}: duplicate bibliography keys: {duplicate_bib}")

    for figure in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex):
        path = Path(figure)
        candidate = directory / path
        if not candidate.exists():
            candidate = directory / "figures" / path
        if not candidate.exists():
            errors.append(f"{name}: unresolved figure: {figure}")

    begin_counts = {}
    end_counts = {}
    for environment in re.findall(r"\\begin\{([^}]+)\}", tex):
        begin_counts[environment] = begin_counts.get(environment, 0) + 1
    for environment in re.findall(r"\\end\{([^}]+)\}", tex):
        end_counts[environment] = end_counts.get(environment, 0) + 1
    for environment in sorted(set(begin_counts) | set(end_counts)):
        if begin_counts.get(environment, 0) != end_counts.get(environment, 0):
            errors.append(
                f"{name}: unbalanced {environment} environment "
                f"({begin_counts.get(environment, 0)} begins, "
                f"{end_counts.get(environment, 0)} ends)"
            )

    for marker in ("TODO", "FIXME", "VERIFY", "PLACEHOLDER"):
        pattern = re.compile(rf"\b{re.escape(marker)}\b", re.IGNORECASE)
        if pattern.search(tex) or pattern.search(bib):
            errors.append(f"{name}: forbidden stale/placeholder marker: {marker}")
    if "dataset_01_run_05" in tex.lower() or "dataset_01_run_05" in bib.lower():
        errors.append(f"{name}: forbidden stale/placeholder marker: dataset_01_run_05")
    if name == "mmas":
        for false_method in ("Keras Tuner", "Bayesian optimization"):
            if false_method.lower() in tex.lower():
                errors.append(f"{name}: obsolete search-method claim: {false_method}")
    if name == "jsc":
        if "IInt8EntropyCalibrator2" in tex:
            errors.append("jsc: obsolete TensorRT calibrator claim")
        for line_number, line in enumerate(tex.splitlines(), start=1):
            if "4070" in line and "train" not in line.lower():
                errors.append(
                    f"jsc:{line_number}: RTX 4070 may only be identified as training hardware"
                )
    return errors


def _audit_experiment() -> list[str]:
    errors = []
    all_calibration_hashes = set()
    capability_path = EXPERIMENT / "jetson_target_capabilities.json"
    if not capability_path.exists():
        errors.append("missing Jetson target-capability record")
    else:
        capabilities = json.loads(capability_path.read_text())
        if capabilities.get("machine") != "aarch64":
            errors.append("target capabilities were not recorded on aarch64")
        if not capabilities.get("platform_has_fast_fp16"):
            errors.append("Jetson target does not report fast FP16")
        if capabilities.get("platform_has_fast_int8"):
            errors.append("Jetson capability record unexpectedly reports fast INT8")
        if not capabilities.get("fixed_clock_lock_verified"):
            errors.append("Jetson target record does not verify fixed CPU/GPU clocks")
        if "MAXN" not in (capabilities.get("power_mode_query") or ""):
            errors.append("Jetson target record does not verify MAXN mode")
        recurrent = set(capabilities.get("tensorrt_rnn_operations") or [])
        if not {"GRU", "LSTM"}.issubset(recurrent):
            errors.append("TensorRT target record lacks GRU/LSTM API operations")
    for model in MODELS:
        model_dir = EXPERIMENT / f"carla_{model}"
        required = (
            "training_provenance.json", "split_indices.npz", "model_config.json",
            "threshold.json", "metrics.json", "evaluation_provenance.json",
            "deployment/model.onnx",
            "deployment/precision_export_status.json",
            "deployment/unified_deployment_report.json",
            "deployment/native_gpu_fp32.json", "deployment/native_gpu_fp16.json",
            "deployment/model_provenance.json",
        )
        for relative in required:
            path = model_dir / relative
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"{model}: missing/empty {relative}")
        report_path = model_dir / "deployment" / "unified_deployment_report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            for backend, record in report.items():
                expected_source = _expected_timing_source(backend)
                for section_name in ("performance", "evaluation"):
                    section = record.get(section_name) or {}
                    if section and section.get("timing_source") != expected_source:
                        errors.append(
                            f"{model}: {backend} {section_name} uses "
                            f"{section.get('timing_source')!r}, expected {expected_source}"
                        )
                evaluation = record.get("evaluation") or {}
                if evaluation and evaluation.get("detector_timing_source") != "perf_counter":
                    errors.append(
                        f"{model}: {backend} detector latency is not wall-clock timed"
                    )
        provenance_path = model_dir / "training_provenance.json"
        if provenance_path.exists():
            provenance = json.loads(provenance_path.read_text())
            gpu_details = provenance.get("tensorflow_gpu_details") or []
            if not any("RTX 4070" in item.get("device_name", "") for item in gpu_details):
                errors.append(f"{model}: training provenance does not identify RTX 4070")
            if provenance.get("git_tracked_dirty"):
                errors.append(f"{model}: training began from tracked-dirty source")
            if provenance.get("random_seed_api") != "keras.utils.set_random_seed":
                errors.append(f"{model}: Keras RNG was not reset through the unified API")
            fault_modes = provenance.get("physics_fault_modes") or []
            fault_components = {item.get("component") for item in fault_modes}
            expected_components = {
                "esr_cap", "capacitance", "inductance", "esr_ind", "rds"
            }
            if fault_components != expected_components:
                errors.append(
                    f"{model}: physics-negative components do not match the "
                    "publication protocol"
                )
            if "load" in fault_components:
                errors.append(f"{model}: healthy load variation used as a fault negative")
        search_root = (
            model_dir / "hp_tuning" / f"carla_{model}_hp_search"
        )
        progress_path = search_root / "search_progress.json"
        search_results_path = model_dir / "search_results.json"
        if not progress_path.exists() or not search_results_path.exists():
            errors.append(f"{model}: missing hyperparameter-search records")
        else:
            progress = json.loads(progress_path.read_text())
            search_results = json.loads(search_results_path.read_text())
            trials = progress.get("trials") or []
            trial_ids = sorted(trial.get("trial_id") for trial in trials)
            if trial_ids != list(range(50)):
                errors.append(
                    f"{model}: completed trial IDs are not exactly 0 through 49"
                )
            effective_keys = [
                _effective_trial_key(trial.get("config") or {}) for trial in trials
            ]
            if len(set(effective_keys)) != 50:
                errors.append(
                    f"{model}: search does not contain 50 unique effective configurations"
                )
            failed_trials = [
                trial.get("trial_id") for trial in trials
                if trial.get("failure_reason")
                or not all(
                    math.isfinite(float(trial.get(metric, float("nan"))))
                    for metric in ("val_loss", "roc_auc", "pr_auc", "f1_score")
                )
            ]
            if failed_trials:
                errors.append(f"{model}: failed/non-finite trials: {failed_trials}")
            for trial_id in range(50):
                trial_dir = search_root / f"trial_{trial_id}"
                for relative in (
                    "trial_config.json",
                    "checkpoints/best/autoencoder.weights.h5",
                    "checkpoints/final/autoencoder.weights.h5",
                    "checkpoints/latest/autoencoder.weights.h5",
                ):
                    path = trial_dir / relative
                    if not path.is_file() or path.stat().st_size == 0:
                        errors.append(
                            f"{model}: trial {trial_id} missing/empty {relative}"
                        )
            if search_results.get("n_trials") != 50 or len(
                search_results.get("all_trials") or []
            ) != 50:
                errors.append(f"{model}: final search summary does not contain 50 trials")
            selected = search_results.get("best_config") or {}
            final_config_path = model_dir / "model_config.json"
            if final_config_path.exists():
                final_config = json.loads(final_config_path.read_text())
                mismatched = [
                    key for key, value in selected.items()
                    if final_config.get(key) != value
                ]
                if mismatched:
                    errors.append(
                        f"{model}: selected/final model config mismatch: {mismatched}"
                    )
        evaluation_provenance_path = model_dir / "evaluation_provenance.json"
        if evaluation_provenance_path.exists():
            evaluation_provenance = json.loads(
                evaluation_provenance_path.read_text()
            )
            if evaluation_provenance.get("machine") != "aarch64":
                errors.append(f"{model}: final detector evaluation was not on Jetson")
            if evaluation_provenance.get("test_rows") != 1268:
                errors.append(f"{model}: final detector evaluation test size is not 1268")
        deployment_dir = model_dir / "deployment"
        optimization_host_path = deployment_dir / "optimization_host.json"
        if not optimization_host_path.exists():
            errors.append(f"{model}: missing FP32 serialization host record")
        else:
            optimization_host = json.loads(optimization_host_path.read_text())
            if optimization_host.get("machine") != "aarch64":
                errors.append(f"{model}: FP32 ONNX was not serialized on the Jetson")
        precision_status_path = deployment_dir / "precision_export_status.json"
        if precision_status_path.exists():
            precision_records = json.loads(precision_status_path.read_text())
            calibration_hashes = set()
            for record in precision_records:
                if record.get("conversion_machine") != "aarch64":
                    errors.append(
                        f"{model}: {record.get('backend')} {record.get('precision')} "
                        "precision artifact was not generated on the Jetson"
                    )
                if record.get("precision") == "int8" and record.get("backend") in {
                    "tflite-cpu", "onnx"
                } and record.get("status") == "exported":
                    calibration_hash = record.get("calibration_sha256")
                    if not calibration_hash:
                        errors.append(
                            f"{model}: {record.get('backend')} INT8 lacks calibration hash"
                        )
                    else:
                        calibration_hashes.add(calibration_hash)
                    if (
                        record.get("backend") == "onnx"
                        and not record.get("qdq_compute_coverage")
                    ):
                        errors.append(
                            f"{model}: ONNX INT8 lacks per-operator Q/DQ coverage"
                        )
            if len(calibration_hashes) > 1:
                errors.append(f"{model}: TFLite and ONNX used different calibration data")
            all_calibration_hashes.update(calibration_hashes)
        consistency_path = deployment_dir / "cross_backend_consistency.json"
        executable_resources = {
            "keras-cpu", "keras-gpu-fp32", "keras-gpu-fp16"
        }
        if not consistency_path.exists():
            errors.append(f"{model}: missing cross-backend consistency report")
        else:
            consistency_records = json.loads(consistency_path.read_text())
            for record in consistency_records:
                if not record.get("preflight_passed"):
                    continue
                label = record.get("model_name", "")
                if label.startswith("TFLite "):
                    resource_name = "tflite_" + label.split()[1].lower()
                elif label.startswith("ONNX "):
                    resource_name = "onnx_" + label.split()[1].lower()
                elif label.startswith("TensorRT "):
                    resource_name = "tensorrt_" + label.split()[1].lower()
                else:
                    errors.append(f"{model}: unknown consistency backend {label}")
                    continue
                executable_resources.add(resource_name)
                if record.get("evaluation_completed"):
                    if record.get("full_test_consistency_samples") != 1268:
                        errors.append(
                            f"{model}: {label} lacks 1268-sample discordance"
                        )
                elif not record.get("runtime_evaluation_failed"):
                    errors.append(
                        f"{model}: {label} has neither full evaluation nor failure record"
                    )
        for resource_name in sorted(executable_resources):
            repetitions = list(deployment_dir.glob(
                f"jetson_resources_{resource_name}_repeat*.json"
            ))
            repetition_numbers = []
            for path in repetitions:
                match = re.search(r"_repeat(\d+)\.json$", path.name)
                if match:
                    repetition_numbers.append(int(match.group(1)))
            if sorted(repetition_numbers) != [1, 2, 3]:
                errors.append(
                    f"{model}: {resource_name} resource repetitions are "
                    f"{sorted(repetition_numbers)} instead of [1, 2, 3]"
                )
            for path in repetitions:
                resource = json.loads(path.read_text())
                backend = resource.get("backend", "")
                expected_source = _expected_timing_source(backend)
                if (resource.get("timing") or {}).get("timing_source") != expected_source:
                    errors.append(
                        f"{model}: {path.name} primary timing is not {expected_source}"
                    )
                if (resource.get("detector_timing") or {}).get(
                    "timing_source"
                ) != "perf_counter":
                    errors.append(
                        f"{model}: {path.name} detector timing is not perf_counter"
                    )
                protocol = resource.get("protocol") or {}
                if protocol.get("batch_size") != 1 or protocol.get(
                    "warmup_runs"
                ) != 15 or protocol.get("timed_runs") != 150:
                    errors.append(f"{model}: {path.name} has a nonstandard timing protocol")
                if (resource.get("device") or {}).get("machine") != "aarch64":
                    errors.append(f"{model}: {path.name} was not measured on the Jetson")
                power = resource.get("power") or {}
                if not power.get("sensor_available"):
                    errors.append(f"{model}: {path.name} lacks PMLib power measurements")
                for phase in ("idle", "active"):
                    measurement = power.get(phase) or {}
                    rails = measurement.get("rails") or {}
                    if measurement.get("sample_count", 0) < 2 or not {
                        "POM_5V_IN", "POM_5V_GPU", "POM_5V_CPU"
                    }.issubset(rails):
                        errors.append(
                            f"{model}: {path.name} has incomplete PMLib {phase} data"
                        )
                if expected_source == "cuda_event" and not (
                    resource.get("cuda_memory") or {}
                ).get("available"):
                    errors.append(f"{model}: {path.name} lacks CUDA memory measurements")
        for precision in ("fp32", "fp16"):
            native_path = deployment_dir / f"native_gpu_{precision}.json"
            if native_path.exists():
                native = json.loads(native_path.read_text())
                consistency = native.get("consistency") or {}
                if consistency.get("full_test_consistency_samples") != 1268:
                    errors.append(
                        f"{model}: native GPU {precision} lacks full-test discordance"
                    )
                evaluation = native.get("evaluation") or {}
                if evaluation.get("timing_source") != "cuda_event":
                    errors.append(
                        f"{model}: native GPU {precision} is not CUDA-event timed"
                    )
                if evaluation.get("detector_timing_source") != "perf_counter":
                    errors.append(
                        f"{model}: native GPU {precision} detector is not wall-clock timed"
                    )
        host_reports = list(deployment_dir.glob("model_provenance_*.json"))
        if len(host_reports) < 2:
            errors.append(f"{model}: missing cross-host loaded-weight provenance")

    if len(all_calibration_hashes) != 1:
        errors.append(
            "INT8 conversions did not share one calibration tensor set across models"
        )

    engine_status = EXPERIMENT / "engine_build_status.tsv"
    if not engine_status.exists():
        errors.append("missing TensorRT engine-build status matrix")
    else:
        rows = [line for line in engine_status.read_text().splitlines()[1:] if line]
        if len(rows) != len(MODELS) * 3:
            errors.append(
                "TensorRT engine-build status does not contain all 15 outcomes"
            )
        for row in rows:
            try:
                model_name, precision, status = row.split("\t")
            except ValueError:
                errors.append(f"malformed TensorRT engine-build status row: {row}")
                continue
            deployment_dir = EXPERIMENT / model_name / "deployment"
            stem = f"model_{precision.lower()}.engine"
            if status == "built":
                for filename in (stem, stem + ".layers.json", stem + ".source.sha256"):
                    path = deployment_dir / filename
                    if not path.is_file() or path.stat().st_size == 0:
                        errors.append(
                            f"{model_name} {precision}: built engine lacks {filename}"
                        )
            elif status == "build_failed":
                log_path = deployment_dir / f"build_{precision.lower()}.log"
                if not log_path.is_file() or log_path.stat().st_size == 0:
                    errors.append(
                        f"{model_name} {precision}: build failure lacks builder log"
                    )
            elif status != "unsupported":
                errors.append(
                    f"{model_name} {precision}: unknown engine status {status}"
                )

    ablation_root = EXPERIMENT / "ablations" / "negative_sampler"
    if not (ablation_root / "summary.json").exists():
        errors.append("missing paired negative-sampler ablation summary")
    for sampler in ("physics", "heuristic"):
        for seed in (200042, 200043, 200044):
            run_dir = ablation_root / sampler / f"seed_{seed}"
            training_path = run_dir / "ablation_training_provenance.json"
            evaluation_path = run_dir / "evaluation_provenance.json"
            metrics_path = run_dir / "metrics.json"
            if not training_path.exists():
                errors.append(f"ablation {sampler}/{seed}: missing training provenance")
            else:
                training = json.loads(training_path.read_text())
                gpu_details = training.get("tensorflow_gpu_details") or []
                if not any(
                    "RTX 4070" in item.get("device_name", "")
                    for item in gpu_details
                ):
                    errors.append(
                        f"ablation {sampler}/{seed}: training did not record RTX 4070"
                    )
                if not training.get("initial_model_weights_sha256"):
                    errors.append(
                        f"ablation {sampler}/{seed}: missing initial-weight hash"
                    )
                if training.get("random_seed_api") != "keras.utils.set_random_seed":
                    errors.append(
                        f"ablation {sampler}/{seed}: unified Keras RNG reset not recorded"
                    )
            if not evaluation_path.exists() or not metrics_path.exists():
                errors.append(
                    f"ablation {sampler}/{seed}: missing target evaluation artifacts"
                )
            else:
                evaluation = json.loads(evaluation_path.read_text())
                if evaluation.get("machine") != "aarch64":
                    errors.append(
                        f"ablation {sampler}/{seed}: test inference was not on Jetson"
                    )
    for seed in (200042, 200043, 200044):
        paired_hashes = []
        for sampler in ("physics", "heuristic"):
            path = (
                ablation_root / sampler / f"seed_{seed}"
                / "ablation_training_provenance.json"
            )
            if path.exists():
                paired_hashes.append(
                    json.loads(path.read_text()).get(
                        "initial_model_weights_sha256"
                    )
                )
        if len(paired_hashes) == 2 and (
            not all(paired_hashes) or len(set(paired_hashes)) != 1
        ):
            errors.append(
                f"ablation seed {seed}: paired initial model weights differ"
            )

    for relative in (
        "papers/results/latest_carla_metrics.csv",
        "papers/results/jetson_deployment_results.csv",
        "papers/results/jetson_resource_measurements.csv",
        "papers/results/jetson_resource_summary.csv",
        "papers/results/native_gpu_precision_results.csv",
        "papers/results/tensorrt_layer_precision_summary.csv",
    ):
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing final publication table: {relative}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--papers-only", action="store_true",
        help="Skip the final experiment-artifact completeness checks.",
    )
    args = parser.parse_args()
    errors = []
    for name, directory in PAPERS.items():
        errors.extend(_audit_paper(name, directory))
    if not args.papers_only:
        errors.extend(_audit_experiment())
    if errors:
        for error in errors:
            print("ERROR:", error)
        raise SystemExit("Publication audit failed with %d error(s)" % len(errors))
    print("Publication manuscripts and final artifacts passed all audits.")


if __name__ == "__main__":
    main()
