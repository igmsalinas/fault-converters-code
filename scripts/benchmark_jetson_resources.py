#!/usr/bin/env python3
"""Measure Jetson latency, RAM, CUDA memory, utilization, and PMLib power.

The script intentionally runs one backend per process.  CPU runs set
``CUDA_VISIBLE_DEVICES=-1`` before TensorFlow is imported; TensorRT runs use the
same CUDA-event timing implementation as the deployment evaluation pipeline.
``tegrastats`` is sampled concurrently for board RAM/utilization/temperature.
PMLib brackets separate idle and sustained-active windows around its root daemon,
which reads the Nano INA3221 rails. TensorRT memory is measured directly with
``pycuda.driver.mem_get_info`` rather than treating unified LPDDR as discrete VRAM.
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


RAM_RE = re.compile(r"RAM\s+(\d+)/(\d+)MB")
SWAP_RE = re.compile(r"SWAP\s+(\d+)/(\d+)MB")
CPU_RE = re.compile(r"CPU\s+\[([^]]+)\]")
EMC_RE = re.compile(r"EMC_FREQ\s+(\d+)%")
GPU_RE = re.compile(r"GR3D_FREQ\s+(\d+)%")
TEMP_RE = re.compile(r"([A-Za-z_]+)@([0-9.]+)C")
POWER_PATTERNS = (
    re.compile(r"POM_5V_IN\s+(\d+)/(\d+)"),
    re.compile(r"VDD_IN\s+(\d+)mW"),
)


def parse_tegrastats(line):
    result = {"raw": line.rstrip()}
    match = RAM_RE.search(line)
    if match:
        result["ram_used_mb"] = float(match.group(1))
        result["ram_total_mb"] = float(match.group(2))
    match = SWAP_RE.search(line)
    if match:
        result["swap_used_mb"] = float(match.group(1))
    match = CPU_RE.search(line)
    if match:
        percentages = [float(value) for value in re.findall(r"(\d+)%", match.group(1))]
        if percentages:
            result["cpu_utilization_percent"] = sum(percentages) / len(percentages)
    match = EMC_RE.search(line)
    if match:
        result["emc_utilization_percent"] = float(match.group(1))
    match = GPU_RE.search(line)
    if match:
        result["gpu_utilization_percent"] = float(match.group(1))
    temperatures = {name.lower(): float(value) for name, value in TEMP_RE.findall(line)}
    if temperatures:
        result["temperatures_c"] = temperatures
    for pattern in POWER_PATTERNS:
        match = pattern.search(line)
        if match:
            result["input_power_w"] = float(match.group(1)) / 1000.0
            break
    return result


class TegrastatsMonitor(object):
    def __init__(self, interval_ms):
        self.interval_ms = interval_ms
        self.phase = "initializing"
        self.samples = []
        self.process = None
        self.thread = None

    def start(self):
        self.process = subprocess.Popen(
            ["tegrastats", "--interval", str(self.interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )

        def read_output():
            for line in self.process.stdout:
                sample = parse_tegrastats(line)
                sample["timestamp_monotonic"] = time.monotonic()
                sample["phase"] = self.phase
                self.samples.append(sample)

        self.thread = threading.Thread(target=read_output)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.thread is not None:
            self.thread.join(timeout=3)


def summarize(samples, field, reducer="mean"):
    values = [sample[field] for sample in samples if field in sample]
    if not values:
        return None
    if reducer == "max":
        return float(max(values))
    return float(sum(values) / len(values))


def temperature_summary(samples):
    names = sorted({name for sample in samples for name in sample.get("temperatures_c", {})})
    return {
        name: {
            "mean_c": float(np.mean([
                sample["temperatures_c"][name]
                for sample in samples if name in sample.get("temperatures_c", {})
            ])),
            "max_c": float(np.max([
                sample["temperatures_c"][name]
                for sample in samples if name in sample.get("temperatures_c", {})
            ])),
        }
        for name in names
    }


def get_process_memory_mb():
    """Return current RSS and lifetime peak RSS from procfs."""
    values = {}
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                values["rss_mb"] = float(line.split()[1]) / 1024.0
            elif line.startswith("VmHWM:"):
                values["peak_rss_mb"] = float(line.split()[1]) / 1024.0
    return values


def device_provenance():
    """Capture the physical board, software stack, and exact source revision."""
    def command(*args):
        try:
            return subprocess.check_output(args, stderr=subprocess.STDOUT).decode().strip()
        except Exception:
            return None

    l4t_release = None
    release_path = Path("/etc/nv_tegra_release")
    if release_path.exists():
        l4t_release = release_path.read_text().splitlines()[0]
    return {
        "hostname": platform.node(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "l4t_release": l4t_release,
        "git_commit": command("git", "-C", str(ROOT), "rev-parse", "HEAD"),
        "git_dirty": bool(command("git", "-C", str(ROOT), "status", "--porcelain")),
        "tensorrt_version": command(
            "dpkg-query", "-W", "-f=${Version}", "python3-libnvinfer"
        ),
    }


def precision_from_path(backend, model_path):
    if backend.endswith("fp16"):
        return "fp16"
    if backend.startswith("keras-"):
        return "fp32"
    stem = model_path.stem.lower() if model_path else ""
    if "int8" in stem:
        return "int8"
    if "fp16" in stem or "float16" in stem:
        return "fp16"
    return "fp32"


class CudaMemoryProbe(object):
    """Sample CUDA allocatable memory from the process's primary context."""
    def __init__(self, enabled):
        self.enabled = enabled
        self.cuda = None
        self.samples = []
        if enabled:
            import pycuda.driver as cuda
            import pycuda.autoprimaryctx  # noqa: F401
            self.cuda = cuda
            self.snapshot("context_baseline")

    def snapshot(self, phase):
        if not self.enabled:
            return None
        free_bytes, total_bytes = self.cuda.mem_get_info()
        sample = {
            "phase": phase,
            "free_mb": float(free_bytes) / (1024.0 * 1024.0),
            "used_mb": float(total_bytes - free_bytes) / (1024.0 * 1024.0),
            "total_mb": float(total_bytes) / (1024.0 * 1024.0),
        }
        self.samples.append(sample)
        return sample

    def summary(self):
        if not self.samples:
            return {
                "available": False,
                "method": "pycuda.driver.mem_get_info",
                "samples": [],
            }
        baseline = self.samples[0]["used_mb"]
        peak = max(sample["used_mb"] for sample in self.samples)
        return {
            "available": True,
            "method": "pycuda.driver.mem_get_info",
            "context_baseline_used_mb": baseline,
            "peak_used_mb": peak,
            "peak_delta_from_context_baseline_mb": max(0.0, peak - baseline),
            "total_cuda_addressable_mb": self.samples[0]["total_mb"],
            "samples": self.samples,
        }


class PMLibPowerMeter(object):
    """Client wrapper for the PMLib INA3221 sampling daemon."""
    RAIL_NAMES = ("POM_5V_IN", "POM_5V_GPU", "POM_5V_CPU")

    def __init__(self, client_dir, server_ip, server_port, device_name):
        client_dir = str(Path(client_dir).resolve())
        if client_dir not in sys.path:
            sys.path.insert(0, client_dir)
        library_path = os.environ.get("LD_LIBRARY_PATH", "")
        paths = [path for path in library_path.split(":") if path]
        if client_dir not in paths:
            os.environ["LD_LIBRARY_PATH"] = ":".join([client_dir] + paths)
        import pmlib
        self.pmlib_module = pmlib
        self.server_ip = server_ip
        self.server_port = server_port
        self.device_name = device_name
        self.client = None

    def start(self):
        self.client = self.pmlib_module.PMLib(self.server_ip, self.server_port)
        self.client.create_lines("0-2")
        # aggregate=0 preserves the three physical rails. PMLib prepends a
        # synthetic sum row; it must not be used as board input power because
        # POM_5V_IN already measures the total module input.
        self.client.create_counter(self.device_name, 0, 0)
        self.client.start_counter()

    def stop(self):
        if self.client is None:
            raise RuntimeError("PMLib counter was not started")
        try:
            self.client.stop_counter()
            # PMLib's bundled ``get_counter_data`` builds a nested ctypes array
            # whose itemsize is rejected by NumPy 1.19 on JetPack 4.6. Read the
            # same native buffers through their pointers with an explicit flat
            # shape, then reshape after copying them into Python-owned memory.
            self.client._get_counter_data()
            measures = self.client.counter.measures.contents
            energy = measures.energy
            line_count = int(energy.lines_len)
            value_count = int(energy.watts_size)
            if line_count <= 0 or value_count % line_count != 0:
                raise RuntimeError(
                    "Invalid PMLib measurement dimensions: %d values, %d lines"
                    % (value_count, line_count)
                )
            sample_count = value_count // line_count
            timing = np.ctypeslib.as_array(measures.timing, shape=(2,)).copy()
            times = np.linspace(timing[0], timing[1], sample_count)
            physical_watts = np.ctypeslib.as_array(
                energy.watts, shape=(value_count,)
            ).copy().reshape((line_count, sample_count))
            watts = np.concatenate(
                (np.sum(physical_watts, axis=0, keepdims=True), physical_watts),
                axis=0,
            )
            if watts.shape[0] != 4 or len(times) < 2:
                raise RuntimeError(
                    "Expected PMLib sum + three Nano rails and at least two samples; "
                    "received shape %s with %d timestamps" % (watts.shape, len(times))
                )
            duration = float(times[-1] - times[0])
            rails = {}
            for index, name in enumerate(self.RAIL_NAMES, start=1):
                values = watts[index]
                rails[name] = {
                    "mean_w": float(np.mean(values)),
                    "std_w": float(np.std(values)),
                    "min_w": float(np.min(values)),
                    "max_w": float(np.max(values)),
                    "energy_j": float(np.trapz(values, times)),
                }
            return {
                "source": "PMLib INA3221 daemon",
                "sample_count": int(len(times)),
                "sample_period_mean_s": float(np.mean(np.diff(times))),
                "measured_duration_s": duration,
                "rails": rails,
            }
        finally:
            self.client.finalize_counter()
            self.client = None


def load_input_shape(model_dir):
    with (model_dir / "model_config.json").open() as handle:
        config = json.load(handle)
    shape = config.get("input_shape") or config.get("model_config", {}).get("input_shape")
    if not shape:
        shape = [101, 2]
    return tuple(int(value) for value in shape)


def build_runner(backend, model_dir, model_path):
    gpu_backends = ("tensorrt", "keras-gpu-fp32", "keras-gpu-fp16")
    if backend not in gpu_backends:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from src.deployment.runners import (
        create_keras_runner,
        create_onnx_runner,
        create_tensorrt_runner,
        create_tflite_runner,
    )
    if backend.startswith("keras-"):
        import keras
        from src.inference.predictor import AnomalyPredictor
        from src.inference.scoring import EmbeddingAnomalyScorer
        from src.deployment.utils import build_carla_embedding_model
        from src.utils.keras_ops_compat import set_mixed_precision_policy
        policy = "mixed_float16" if backend.endswith("fp16") else "float32"
        set_mixed_precision_policy(policy)
        predictor = AnomalyPredictor(model_dir=str(model_dir))
        model_runner = create_keras_runner(build_carla_embedding_model(predictor.model))
        reference = predictor.reference_embeddings
        if reference is None:
            deployment_reference = model_dir / "deployment" / "reference_embeddings.npy"
            if deployment_reference.exists():
                reference = np.load(str(deployment_reference))
        if reference is None:
            raise RuntimeError("Keras resource benchmark requires a CARLA reference bank")
        scorer = EmbeddingAnomalyScorer(
            reference,
            method=predictor.scoring_method,
            k=int(predictor.k_neighbors),
        )
    elif backend == "tflite":
        model_runner = create_tflite_runner(str(model_path))
    elif backend == "onnx":
        runner = create_onnx_runner(str(model_path))
        if runner is None:
            raise RuntimeError("ONNX Runtime CPU runner could not be initialized")
        model_runner = runner
    else:
        runner = create_tensorrt_runner(str(model_path))
        if runner is None:
            raise RuntimeError("TensorRT runner could not be initialized")
        model_runner = runner

    if not backend.startswith("keras-"):
        from src.inference.scoring import EmbeddingAnomalyScorer

        with (model_dir / "deployment" / "detector_config.json").open() as handle:
            detector_config = json.load(handle)
        reference = np.load(model_dir / "deployment" / "reference_embeddings.npy")
        scorer = EmbeddingAnomalyScorer(
            reference,
            method=detector_config["scoring_method"],
            k=int(detector_config["k_neighbors"]),
        )

    def detector_runner(x):
        return scorer.score(model_runner(x))[:, np.newaxis]

    detector_runner.output_shape = (1,)
    return model_runner, detector_runner


def run(args):
    model_dir = Path(args.model_dir)
    model_path = Path(args.model_file) if args.model_file else None
    if args.engine:
        model_path = Path(args.engine)
    if not args.backend.startswith("keras-") and model_path is None:
        raise ValueError("--model-file is required for converted backends")

    power_meter = None
    if args.pmlib_client_dir:
        power_meter = PMLibPowerMeter(
            args.pmlib_client_dir,
            args.pmlib_server_ip,
            args.pmlib_server_port,
            args.pmlib_device,
        )
    elif args.require_power:
        raise ValueError("--pmlib-client-dir is required when --require-power is set")

    monitor = TegrastatsMonitor(args.interval_ms)
    idle_power_measurement = None
    active_power_measurement = None
    process_samples = []
    cuda_probe = None
    monitor.start()
    try:
        monitor.phase = "idle"
        if power_meter is not None:
            power_meter.start()
        try:
            time.sleep(args.idle_seconds)
        finally:
            if power_meter is not None:
                idle_power_measurement = power_meter.stop()

        baseline_process = get_process_memory_mb()
        baseline_process["phase"] = "pre_load_baseline"
        process_samples.append(baseline_process)
        cuda_probe = CudaMemoryProbe(
            args.backend == "tensorrt" or args.backend.startswith("keras-gpu-")
        )
        if cuda_probe.enabled:
            context_process = get_process_memory_mb()
            context_process["phase"] = "cuda_context"
            process_samples.append(context_process)

        monitor.phase = "loading"
        model_runner, detector_runner = build_runner(args.backend, model_dir, model_path)
        if cuda_probe.enabled:
            cuda_probe.snapshot("engine_loaded")
        loaded_process = get_process_memory_mb()
        loaded_process["phase"] = "runner_loaded"
        process_samples.append(loaded_process)
        sample_path = model_dir / "deployment" / "benchmark_sample.npy"
        if not sample_path.exists():
            raise FileNotFoundError(
                "Missing held-out benchmark sample; rerun deployment optimization"
            )
        sample = np.asarray(np.load(str(sample_path)), dtype=np.float32)
        expected_shape = (1,) + load_input_shape(model_dir)
        if sample.shape != expected_shape or not np.all(np.isfinite(sample)):
            raise ValueError(
                "Invalid benchmark sample shape/content: {} (expected {})".format(
                    sample.shape, expected_shape
                )
            )

        from src.deployment.runners import measure_inference_latency
        monitor.phase = "warmup_and_timing"
        model_timing = measure_inference_latency(
            model_runner, sample, num_warmup=args.warmup, num_runs=args.runs
        )
        detector_timing = measure_inference_latency(
            detector_runner, sample, num_warmup=args.warmup, num_runs=args.runs
        )
        if cuda_probe.enabled:
            cuda_probe.snapshot("timing_complete")
        timed_process = get_process_memory_mb()
        timed_process["phase"] = "timing_complete"
        process_samples.append(timed_process)

        monitor.phase = "active"
        if power_meter is not None:
            power_meter.start()
        active_started = time.perf_counter()
        inference_count = 0
        try:
            while time.perf_counter() - active_started < args.active_seconds:
                detector_runner(sample)
                inference_count += 1
                if cuda_probe.enabled and inference_count % 25 == 0:
                    cuda_probe.snapshot("active")
            active_elapsed = time.perf_counter() - active_started
        finally:
            if power_meter is not None:
                active_power_measurement = power_meter.stop()
        if cuda_probe.enabled:
            cuda_probe.snapshot("active_complete")
        active_process = get_process_memory_mb()
        active_process["phase"] = "active_complete"
        process_samples.append(active_process)
        monitor.phase = "complete"
        time.sleep(max(0.2, args.interval_ms / 1000.0))
    finally:
        monitor.stop()

    idle = [sample for sample in monitor.samples if sample["phase"] == "idle"]
    active = [sample for sample in monitor.samples if sample["phase"] == "active"]
    idle_ram = summarize(idle, "ram_used_mb")
    active_ram_mean = summarize(active, "ram_used_mb")
    active_ram_peak = summarize(active, "ram_used_mb", "max")
    throughput = inference_count / active_elapsed
    power_result = {
        "sensor_available": False,
        "source": None,
        "idle": idle_power_measurement,
        "active": active_power_measurement,
    }
    if idle_power_measurement is not None and active_power_measurement is not None:
        idle_rails = idle_power_measurement["rails"]
        active_rails = active_power_measurement["rails"]
        idle_input_w = idle_rails["POM_5V_IN"]["mean_w"]
        active_input_w = active_rails["POM_5V_IN"]["mean_w"]
        dynamic_input_w = active_input_w - idle_input_w
        power_result.update({
            "sensor_available": True,
            "source": "PMLib INA3221 daemon",
            "idle_input_mean_w": idle_input_w,
            "active_input_mean_w": active_input_w,
            "dynamic_input_mean_w": dynamic_input_w,
            "total_energy_per_inference_mj": active_input_w / throughput * 1000.0,
            "dynamic_energy_per_inference_mj": dynamic_input_w / throughput * 1000.0,
            "gpu_rail_active_mean_w": active_rails["POM_5V_GPU"]["mean_w"],
            "gpu_rail_energy_per_inference_mj": (
                active_rails["POM_5V_GPU"]["mean_w"] / throughput * 1000.0
            ),
            "cpu_rail_active_mean_w": active_rails["POM_5V_CPU"]["mean_w"],
            "cpu_rail_energy_per_inference_mj": (
                active_rails["POM_5V_CPU"]["mean_w"] / throughput * 1000.0
            ),
            "throughput_per_input_watt": throughput / active_input_w,
            "note": (
                "POM_5V_IN is total module input power. GPU/CPU rails are reported "
                "separately and are not added to it."
            ),
        })
    elif args.require_power:
        raise RuntimeError("PMLib did not return both idle and active power windows")

    baseline_rss = process_samples[0]["rss_mb"]
    process_peak = max(sample["peak_rss_mb"] for sample in process_samples)

    result = {
        "device": device_provenance(),
        "model_dir": str(model_dir),
        "backend": args.backend,
        "precision": precision_from_path(args.backend, model_path),
        "model_file": str(model_path) if model_path else None,
        "protocol": {
            "batch_size": 1,
            "warmup_runs": args.warmup,
            "timed_runs": args.runs,
            "resource_sample_interval_ms": args.interval_ms,
            "idle_seconds": args.idle_seconds,
            "sustained_active_seconds": args.active_seconds,
        },
        "timing": model_timing,
        "detector_timing": detector_timing,
        "sustained_throughput_inferences_per_second": throughput,
        "sustained_inference_count": inference_count,
        "system_shared_memory": {
            "idle_mean_mb": idle_ram,
            "active_mean_mb": active_ram_mean,
            "active_peak_mb": active_ram_peak,
            "active_mean_delta_mb": (
                active_ram_mean - idle_ram
                if active_ram_mean is not None and idle_ram is not None else None
            ),
            "active_peak_delta_mb": (
                active_ram_peak - idle_ram
                if active_ram_peak is not None and idle_ram is not None else None
            ),
        },
        "process_ram": {
            "method": "/proc/self/status VmRSS/VmHWM",
            "baseline_rss_mb": baseline_rss,
            "peak_rss_mb": process_peak,
            "peak_delta_mb": max(0.0, process_peak - baseline_rss),
            "samples": process_samples,
        },
        "cuda_memory": cuda_probe.summary(),
        "utilization": {
            "cpu_active_mean_percent": summarize(active, "cpu_utilization_percent"),
            "gpu_active_mean_percent": summarize(active, "gpu_utilization_percent"),
            "emc_active_mean_percent": summarize(active, "emc_utilization_percent"),
        },
        "temperatures": temperature_summary(active),
        "power": power_result,
        "tegrastats_sample_count": len(monitor.samples),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument(
        "--backend",
        choices=(
            "keras-cpu", "keras-gpu-fp32", "keras-gpu-fp16",
            "tflite", "onnx", "tensorrt",
        ),
        required=True,
    )
    parser.add_argument("--model-file")
    parser.add_argument("--engine", help=argparse.SUPPRESS)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--runs", type=int, default=150)
    parser.add_argument("--interval-ms", type=int, default=100)
    parser.add_argument("--idle-seconds", type=float, default=5.0)
    parser.add_argument("--active-seconds", type=float, default=15.0)
    parser.add_argument("--pmlib-client-dir")
    parser.add_argument("--pmlib-server-ip", default="127.0.0.1")
    parser.add_argument("--pmlib-server-port", type=int, default=6526)
    parser.add_argument("--pmlib-device", default="Jetson-Nano")
    parser.add_argument("--require-power", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
