#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv-jp46/bin/python}"
EXPERIMENT="${EXPERIMENT:-experiments/dataset_01_run_06_corrected}"
PMLIB_CLIENT_DIR="${PMLIB_CLIENT_DIR:-/home/ignaciouc3m/projects/pmlib/client}"
REPETITIONS="${REPETITIONS:-3}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-5}"
MODE="${1:-all}"

case "${MODE}" in
    all) MODELS=(conv1d gru lstm mlp transformer); INCLUDE_CPU=1; INCLUDE_TRT_BASE=1 ;;
    recurrent) MODELS=(gru lstm); INCLUDE_CPU=1; INCLUDE_TRT_BASE=1 ;;
    gpu) MODELS=(conv1d gru lstm mlp transformer); INCLUDE_CPU=0; INCLUDE_TRT_BASE=1 ;;
    portable) MODELS=(conv1d gru lstm mlp transformer); INCLUDE_CPU=0; INCLUDE_TRT_BASE=0 ;;
    *) echo "Usage: $0 [all|recurrent|gpu|portable]" >&2; exit 2 ;;
esac

run_resource_benchmark() {
    local model="$1"
    local backend="$2"
    local precision="${3:-}"
    local model_dir="${EXPERIMENT}/carla_${model}"
    local repetition="${4:-1}"
    local output="${model_dir}/deployment/jetson_resources_${backend}"
    local model_args=()

    if [[ -n "${precision}" ]]; then
        output="${output}_${precision}"
        case "${backend}" in
            tensorrt) model_args=(--model-file "${model_dir}/deployment/model_${precision}.engine") ;;
            onnx)
                if [[ "${precision}" == "fp32" ]]; then
                    model_args=(--model-file "${model_dir}/deployment/model.onnx")
                else
                    model_args=(--model-file "${model_dir}/deployment/model_${precision}.onnx")
                fi
                ;;
            tflite)
                if [[ "${precision}" == "fp16" ]]; then
                    model_args=(--model-file "${model_dir}/deployment/model_float16.tflite")
                else
                    model_args=(--model-file "${model_dir}/deployment/model_${precision}.tflite")
                fi
                ;;
        esac
        if [[ ! -f "${model_args[1]}" ]]; then
            echo "Skipping unsupported or unavailable path: ${model} ${backend} ${precision}"
            return
        fi
    fi

    output="${output}_repeat${repetition}"
    if [[ -s "${output}.json" ]]; then
        echo "Keeping completed resource study: ${model} ${backend} ${precision}"
        return
    fi

    if ! "${PYTHON_BIN}" scripts/benchmark_jetson_resources.py \
            --model-dir "${model_dir}" \
            --backend "${backend}" \
            "${model_args[@]}" \
            --pmlib-client-dir "${PMLIB_CLIENT_DIR}" \
            --require-power \
            --output "${output}.json" >/dev/null; then
        echo "Resource study failed at runtime: ${model} ${backend} ${precision}" >&2
    fi
    sleep "${COOLDOWN_SECONDS}"
}

# Each runtime is isolated in its own process. Resource measurements are kept
# for every executable path, including accuracy-affected paths; the separate
# full-test fidelity status prevents rejected paths being presented as valid.
for repetition in $(seq 1 "${REPETITIONS}"); do
for model in "${MODELS[@]}"; do
    if [[ "${INCLUDE_CPU}" -eq 1 ]]; then
        run_resource_benchmark "${model}" keras-cpu "" "${repetition}"
    fi
    run_resource_benchmark "${model}" keras-gpu-fp32 "" "${repetition}"
    run_resource_benchmark "${model}" keras-gpu-fp16 "" "${repetition}"
    if [[ "${INCLUDE_TRT_BASE}" -eq 1 ]]; then
        run_resource_benchmark "${model}" tensorrt fp32 "${repetition}"
        run_resource_benchmark "${model}" tensorrt fp16 "${repetition}"
    fi
    run_resource_benchmark "${model}" tensorrt int8 "${repetition}"
    run_resource_benchmark "${model}" tflite fp32 "${repetition}"
    run_resource_benchmark "${model}" tflite fp16 "${repetition}"
    run_resource_benchmark "${model}" tflite int8 "${repetition}"
    run_resource_benchmark "${model}" onnx fp32 "${repetition}"
    run_resource_benchmark "${model}" onnx fp16 "${repetition}"
    run_resource_benchmark "${model}" onnx int8 "${repetition}"
done
done
