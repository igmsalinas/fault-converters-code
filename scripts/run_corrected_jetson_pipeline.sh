#!/usr/bin/env bash
# Reproducible JetPack 4.6 deployment campaign for the corrected CARLA study.

set -euo pipefail

if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "This campaign must run on the target Jetson Nano (aarch64)." >&2
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-.venv-jp46/bin/python}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-dataset_01_run_06_corrected}"
EXPERIMENT="experiments/${EXPERIMENT_NAME}"
DATA_DIR="${DATA_DIR:-data/buck/buck_data/dataset_01}"
REPETITIONS="${REPETITIONS:-3}"
PMLIB_CLIENT_DIR="${PMLIB_CLIENT_DIR:-/home/ignaciouc3m/projects/pmlib/client}"
CLOCKS_RECORD="${CLOCKS_RECORD:-${EXPERIMENT}/jetson_clocks_show.txt}"
MODELS=(conv1d gru lstm mlp transformer)

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "JetPack Python environment not found: ${PYTHON_BIN}" >&2
    exit 1
fi
if ! pgrep -x pmlib_server >/dev/null; then
    echo "PMLib daemon is not running; start it from ../pmlib/server first." >&2
    exit 1
fi

for model in "${MODELS[@]}"; do
    test -f "${EXPERIMENT}/carla_${model}/model_config.json"
done
if [[ ! -s "${CLOCKS_RECORD}" ]]; then
    echo "Missing privileged clock record: ${CLOCKS_RECORD}" >&2
    echo "After setting MAXN and jetson_clocks, run:" >&2
    echo "  { date -Is; sudo jetson_clocks --show; } > ${CLOCKS_RECORD}" >&2
    exit 1
fi

EXPERIMENT="${EXPERIMENT}" CLOCKS_RECORD="${CLOCKS_RECORD}" "${PYTHON_BIN}" - <<'PY'
import json
import os
import platform
import re
import subprocess
import time
from pathlib import Path

import onnx
import onnxruntime
import tensorflow
import tf2onnx
import tensorrt

print("Target conversion stack:", tensorflow.__version__, onnx.__version__,
      onnxruntime.__version__, tf2onnx.__version__)
logger = tensorrt.Logger(tensorrt.Logger.ERROR)
builder = tensorrt.Builder(logger)
network = builder.create_network()

def read_int(path):
    try:
        return int(Path(path).read_text().strip())
    except (IOError, OSError, ValueError):
        return None

cpu_clock = {
    "unit": "kHz",
    "current": read_int("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"),
    "minimum": read_int("/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq"),
    "maximum": read_int("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"),
}
gpu_root = "/sys/devices/57000000.gpu/devfreq/57000000.gpu"
gpu_clock = {
    "unit": "Hz",
    "current": read_int(gpu_root + "/cur_freq"),
    "minimum": read_int(gpu_root + "/min_freq"),
    "maximum": read_int(gpu_root + "/max_freq"),
}
sysfs_cpu_gpu_lock_verified = all(
    values["current"] is not None
    and values["current"] == values["minimum"] == values["maximum"]
    for values in (cpu_clock, gpu_clock)
)
clocks_record_path = Path(os.environ["CLOCKS_RECORD"])
clocks_show = clocks_record_path.read_text()
clocks_record_age_seconds = time.time() - clocks_record_path.stat().st_mtime
cpu_matches = re.findall(
    r"cpu\d+: Online=1 .*?MinFreq=(\d+) MaxFreq=(\d+) CurrentFreq=(\d+)",
    clocks_show,
)
gpu_match = re.search(
    r"GPU MinFreq=(\d+) MaxFreq=(\d+) CurrentFreq=(\d+)", clocks_show
)
emc_match = re.search(
    r"EMC MinFreq=(\d+) MaxFreq=(\d+) CurrentFreq=(\d+) FreqOverride=(\d+)",
    clocks_show,
)
fan_match = re.search(r"Fan: PWM=(\d+)", clocks_show)
privileged_clock_checks = {
    "record_age_seconds": clocks_record_age_seconds,
    "record_fresh_within_10_minutes": 0 <= clocks_record_age_seconds <= 600,
    "all_four_cpus_fixed": len(cpu_matches) == 4 and all(
        minimum == maximum == current
        for minimum, maximum, current in cpu_matches
    ),
    "gpu_fixed": bool(
        gpu_match and gpu_match.group(1) == gpu_match.group(2) == gpu_match.group(3)
    ),
    # Tegra210 retains the nominal EMC minimum while jetson_clocks asserts an
    # override. The correct lock condition is current=max with FreqOverride=1.
    "emc_override_fixed_at_max": bool(
        emc_match
        and emc_match.group(3) == emc_match.group(2)
        and emc_match.group(4) == "1"
    ),
    "fan_pwm": int(fan_match.group(1)) if fan_match else None,
}
clock_lock_verified = bool(
    sysfs_cpu_gpu_lock_verified
    and privileged_clock_checks["record_fresh_within_10_minutes"]
    and privileged_clock_checks["all_four_cpus_fixed"]
    and privileged_clock_checks["gpu_fixed"]
    and privileged_clock_checks["emc_override_fixed_at_max"]
)
try:
    power_mode = subprocess.check_output(
        ["nvpmodel", "-q"], stderr=subprocess.STDOUT
    ).decode().strip()
except (OSError, subprocess.CalledProcessError):
    power_mode = None
capabilities = {
    "hostname": platform.node(),
    "machine": platform.machine(),
    "platform": platform.platform(),
    "tensorflow_version": tensorflow.__version__,
    "onnx_version": onnx.__version__,
    "onnxruntime_version": onnxruntime.__version__,
    "tf2onnx_version": tf2onnx.__version__,
    "tensorrt_version": tensorrt.__version__,
    "tensorrt_add_rnn_v2_available": hasattr(network, "add_rnn_v2"),
    "tensorrt_rnn_operations": [
        name for name in ("GRU", "LSTM", "RELU", "TANH")
        if hasattr(tensorrt.RNNOperation, name)
    ],
    "platform_has_fast_fp16": bool(builder.platform_has_fast_fp16),
    "platform_has_fast_int8": bool(builder.platform_has_fast_int8),
    "power_mode_query": power_mode,
    "cpu_clock": cpu_clock,
    "gpu_clock": gpu_clock,
    "jetson_clocks_show": clocks_show,
    "privileged_clock_checks": privileged_clock_checks,
    "fixed_clock_lock_verified": clock_lock_verified,
    "tensorflow_gpus": [
        tensorflow.config.experimental.get_device_details(device)
        for device in tensorflow.config.list_physical_devices("GPU")
    ],
}
output = Path(os.environ["EXPERIMENT"]) / "jetson_target_capabilities.json"
with output.open("w") as handle:
    json.dump(capabilities, handle, indent=2)
print(json.dumps(capabilities, indent=2))
if not clock_lock_verified:
    raise RuntimeError(
        "CPU/GPU clocks are not locked; run sudo jetson_clocks before the campaign"
    )
if power_mode is None or "MAXN" not in power_mode:
    raise RuntimeError("Jetson Nano is not in MAXN power mode")
PY

echo "[preflight] Verifying two reusable PMLib power windows"
"${PYTHON_BIN}" scripts/check_pmlib_counter.py \
    --client-dir "${PMLIB_CLIENT_DIR}" \
    --windows 2

echo "[1/8] Selecting final detectors and evaluating trained FP32 models on the Nano"
FAULT_CONVERTERS_PYTHON="${PYTHON_BIN}" bash scripts/evaluate_all.sh \
    --experiment-name "${EXPERIMENT_NAME}" \
    --data-dir "${DATA_DIR}"
ABLATION_ROOT="${EXPERIMENT}/ablations/negative_sampler"
if [[ -d "${ABLATION_ROOT}" ]]; then
    echo "[1/8] Evaluating paired negative-sampler ablations on the Nano"
    for sampler in physics heuristic; do
        for ablation_dir in "${ABLATION_ROOT}/${sampler}"/seed_*; do
            [[ -f "${ablation_dir}/model_config.json" ]] || continue
            "${PYTHON_BIN}" -m src.evaluate \
                --model-dir "${ablation_dir}" \
                --data-dir "${DATA_DIR}"
        done
    done
    "${PYTHON_BIN}" scripts/run_negative_sampler_ablation.py \
        --experiment "${EXPERIMENT}" \
        --data-dir "${DATA_DIR}" \
        --summarize-only
fi

echo "[2/8] Serializing FP32 ONNX graphs and detector assets on the Nano"
for model in "${MODELS[@]}"; do
    "${PYTHON_BIN}" -m src.deployment_optimization \
        --model-dir "${EXPERIMENT}/carla_${model}" \
        --data-dir "${DATA_DIR}" \
        --portable-export-only
done

echo "[3/8] Exporting target-compatible TFLite and ONNX FP32/FP16/INT8 artifacts"
"${PYTHON_BIN}" scripts/export_precision_variants.py \
    --experiment "${EXPERIMENT}" \
    --data-dir "${DATA_DIR}"

echo "[4/8] Building all available TensorRT engines on the Nano"
bash scripts/build_jetson_engines.sh --experiment-name "${EXPERIMENT_NAME}"

echo "[5/8] Auditing loaded detector checkpoints and reference banks"
model_dirs=()
for model in "${MODELS[@]}"; do
    model_dirs+=("${EXPERIMENT}/carla_${model}")
done
"${PYTHON_BIN}" scripts/audit_model_provenance.py "${model_dirs[@]}"

echo "[6/8] Evaluating every available TFLite, ONNX, and TensorRT path"
FAULT_CONVERTERS_PYTHON="${PYTHON_BIN}" bash scripts/evaluate_inference.sh \
    --experiment-name "${EXPERIMENT_NAME}" \
    --data-dir "${DATA_DIR}"

echo "[7/8] Evaluating native Jetson TensorFlow GPU FP32 and mixed FP16"
for model in "${MODELS[@]}"; do
    model_dir="${EXPERIMENT}/carla_${model}"
    for precision in fp32 fp16; do
        "${PYTHON_BIN}" scripts/benchmark_native_gpu_precisions.py \
            --model-dir "${model_dir}" \
            --data-dir "${DATA_DIR}" \
            --precision "${precision}" \
            --output "${model_dir}/deployment/native_gpu_${precision}.json"
    done
done

echo "[8/8] Running isolated timing, memory, utilization, and PMLib power studies"
PYTHON_BIN="${PYTHON_BIN}" EXPERIMENT="${EXPERIMENT}" \
PMLIB_CLIENT_DIR="${PMLIB_CLIENT_DIR}" \
REPETITIONS="${REPETITIONS}" bash scripts/run_jetson_resource_suite.sh all

echo "Corrected Jetson Nano campaign completed."
