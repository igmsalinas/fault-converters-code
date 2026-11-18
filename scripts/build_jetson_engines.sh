#!/usr/bin/env bash
# Build device-specific TensorRT engines for CARLA ONNX exports on JetPack 4.6.

set -euo pipefail

EXPERIMENT_NAME="dataset_01_run_06_corrected"
MODEL_DIR=""
WORKSPACE_MB="512"
FORCE=0
FAILED=()
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment-name) EXPERIMENT_NAME="$2"; shift 2 ;;
        --experiment-name=*) EXPERIMENT_NAME="${1#*=}"; shift ;;
        --model-dir) MODEL_DIR="$2"; shift 2 ;;
        --model-dir=*) MODEL_DIR="${1#*=}"; shift ;;
        --workspace-mb) WORKSPACE_MB="$2"; shift 2 ;;
        --workspace-mb=*) WORKSPACE_MB="${1#*=}"; shift ;;
        --force) FORCE=1; shift ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "TensorRT engines must be built on the target Jetson (aarch64)." >&2
    exit 1
fi
if [[ ! -x "$TRTEXEC" ]]; then
    echo "trtexec not found or not executable: $TRTEXEC" >&2
    exit 1
fi

build_engine() {
    local target_dir="$1"
    local precision="$2"
    local deployment_dir="$target_dir/deployment"
    local onnx_path="$deployment_dir/model.onnx"
    local engine_path="$deployment_dir/model_${precision,,}.engine"
    local layer_info_path="${engine_path}.layers.json"
    local source_hash_path="$engine_path.source.sha256"
    local log_path="$deployment_dir/build_${precision,,}.log"
    local precision_flag=()

    if [[ "$precision" == "INT8" ]]; then
        onnx_path="$deployment_dir/model_int8.onnx"
        precision_flag=(--int8)
    elif [[ "$precision" == "FP16" ]]; then
        precision_flag=(--fp16)
    fi

    if [[ ! -s "$onnx_path" ]]; then
        echo "Missing ONNX export: $onnx_path" >&2
        return 1
    fi
    local source_hash
    source_hash="$(sha256sum "$onnx_path" | awk '{print $1}')"
    if [[ -s "$engine_path" && -s "$layer_info_path" \
            && -s "$source_hash_path" && "$FORCE" -eq 0 ]] \
            && [[ "$(tr -d '[:space:]' < "$source_hash_path")" == "$source_hash" ]]; then
        echo "Keeping existing engine: $engine_path"
        return 0
    fi
    local temporary_engine="${engine_path}.building.$$"
    local temporary_layer_info="${layer_info_path}.building.$$"
    echo "Building $(basename "$target_dir") $precision engine on this Jetson..."
    if "$TRTEXEC" \
        --onnx="$onnx_path" \
        --saveEngine="$temporary_engine" \
        --workspace="$WORKSPACE_MB" \
        --buildOnly \
        --profilingVerbosity=detailed \
        --dumpLayerInfo \
        --exportLayerInfo="$temporary_layer_info" \
        "${precision_flag[@]}" 2>&1 | tee "$log_path" \
        && [[ -s "$temporary_engine" && -s "$temporary_layer_info" ]]; then
        mv "$temporary_engine" "$engine_path"
        mv "$temporary_layer_info" "$layer_info_path"
        printf '%s\n' "$source_hash" > "$source_hash_path"
        return 0
    fi
    # Exact generated targets only: never let a partial or source-stale plan
    # enter the evaluation discovery step or a subsequent resume.
    rm -f "$temporary_engine" "$temporary_layer_info" "$engine_path" \
        "$layer_info_path" "$source_hash_path"
    return 1
}

build_model() {
    local target_dir="$1"
    local model_name
    model_name="$(basename "$target_dir")"
    for precision in FP32 FP16 INT8; do
        if [[ "$precision" == "INT8" && ! -s "$target_dir/deployment/model_int8.onnx" ]]; then
            echo "Skipping unavailable calibrated INT8 ONNX graph: $target_dir"
            printf '%s\t%s\t%s\n' "$model_name" "$precision" "unsupported" >> "$STATUS_PATH"
        elif build_engine "$target_dir" "$precision"; then
            printf '%s\t%s\t%s\n' "$model_name" "$precision" "built" >> "$STATUS_PATH"
        else
            printf '%s\t%s\t%s\n' "$model_name" "$precision" "build_failed" >> "$STATUS_PATH"
            FAILED+=("${model_name}:${precision}")
        fi
    done
}

if [[ -n "$MODEL_DIR" ]]; then
    STATUS_PATH="$MODEL_DIR/deployment/engine_build_status.tsv"
    printf 'model\tprecision\tstatus\n' > "$STATUS_PATH"
    build_model "$MODEL_DIR"
else
    experiment_dir="experiments/$EXPERIMENT_NAME"
    if [[ ! -d "$experiment_dir" ]]; then
        echo "Experiment directory not found: $experiment_dir" >&2
        exit 1
    fi
    STATUS_PATH="$experiment_dir/engine_build_status.tsv"
    printf 'model\tprecision\tstatus\n' > "$STATUS_PATH"
    for target_dir in "$experiment_dir"/carla_*/; do
        [[ -f "$target_dir/model_config.json" ]] || continue
        build_model "${target_dir%/}"
    done
fi

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "TensorRT build failures retained for compatibility reporting: ${FAILED[*]}" >&2
else
    echo "All available FP32, FP16, and INT8 TensorRT engines were created successfully."
fi
