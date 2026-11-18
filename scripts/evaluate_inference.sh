#!/usr/bin/env bash
# ==============================================================================
# Run edge compute inference evaluation and benchmarking
# ==============================================================================

set -euo pipefail

# Default values
MODEL_DIR=""
EXPERIMENT_NAME=""
DATA_DIR="data/buck/buck_data"
MAX_FILES=""

# JetPack 4.6 cannot use the project's Python 3.12/uv environment. Prefer the
# pinned on-device Python 3.6 environment when it exists; callers may override
# either platform with FAULT_CONVERTERS_PYTHON.
if [ -n "${FAULT_CONVERTERS_PYTHON:-}" ]; then
    PYTHON_BIN="$FAULT_CONVERTERS_PYTHON"
elif [ "$(uname -m)" = "aarch64" ] && [ -x ".venv-jp46/bin/python" ]; then
    PYTHON_BIN=".venv-jp46/bin/python"
elif command -v uv >/dev/null 2>&1; then
    PYTHON_BIN="uv run python"
else
    PYTHON_BIN="python3"
fi

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-dir) MODEL_DIR="$2"; shift 2 ;;
        --model-dir=*) MODEL_DIR="${1#*=}"; shift ;;
        --experiment-name) EXPERIMENT_NAME="$2"; shift 2 ;;
        --experiment-name=*) EXPERIMENT_NAME="${1#*=}"; shift ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --data-dir=*) DATA_DIR="${1#*=}"; shift ;;
        --max-files) MAX_FILES="$2"; shift 2 ;;
        --max-files=*) MAX_FILES="${1#*=}"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

run_evaluation() {
    local target_dir="$1"
    local extra_args=()
    if [ -n "$MAX_FILES" ]; then
        extra_args=(--max-files "$MAX_FILES")
    fi
    echo "--------------------------------------------------------"
    echo "Evaluating Edge Inference: $(basename "$target_dir")"
    echo "--------------------------------------------------------"
    # Word splitting is intentional for the local "uv run python" command.
    # shellcheck disable=SC2086
    $PYTHON_BIN -m src.deployment_evaluation \
      --model-dir "$target_dir" \
      --data-dir "$DATA_DIR" \
      "${extra_args[@]}"
}

echo "Starting Edge Compute Evaluation Pipeline..."

if [ -n "$MODEL_DIR" ]; then
    run_evaluation "$MODEL_DIR"
elif [ -n "$EXPERIMENT_NAME" ]; then
    EXP_DIR="experiments/$EXPERIMENT_NAME"
    if [ ! -d "$EXP_DIR" ]; then
        echo "Error: Experiment directory not found: $EXP_DIR"
        exit 1
    fi
    for m_dir in "$EXP_DIR"/*/; do
        if [[ -d "$m_dir" && -f "$m_dir/model_config.json" ]]; then
            run_evaluation "${m_dir%/}"
        fi
    done
else
    # Default fallback to conv1d_ae
    run_evaluation "experiments/conv1d_ae"
fi

echo "Evaluation pipeline complete!"
