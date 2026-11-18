#!/usr/bin/env bash
# Evaluates all available trained models in the experiments directory

set -uo pipefail

EXPERIMENTS_DIR="experiments"
EXPERIMENT_NAME=""
DATA_DIR="data/buck/buck_data/dataset_01"

if [ -n "${FAULT_CONVERTERS_PYTHON:-}" ]; then
    PYTHON_CMD=("$FAULT_CONVERTERS_PYTHON")
elif command -v uv >/dev/null 2>&1; then
    PYTHON_CMD=(uv run python)
else
    PYTHON_CMD=(python3)
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment-name) EXPERIMENT_NAME="$2"; shift 2 ;;
        --experiment-name=*) EXPERIMENT_NAME="${1#*=}"; shift ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --data-dir=*) DATA_DIR="${1#*=}"; shift ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

if [ -n "$EXPERIMENT_NAME" ]; then
    EXPERIMENTS_DIR="$EXPERIMENTS_DIR/$EXPERIMENT_NAME"
fi

if [ ! -d "$EXPERIMENTS_DIR" ]; then
    echo "Experiments directory not found: $EXPERIMENTS_DIR"
    exit 1
fi

echo "Evaluating all models in $EXPERIMENTS_DIR..."
FAILED=()

for model_dir in "$EXPERIMENTS_DIR"/*/; do
    if [ -d "$model_dir" ]; then
        model_name=$(basename "$model_dir")
        
        # Check if the model has a model_config.json to confirm it is actually trained
        if [ ! -f "$model_dir/model_config.json" ]; then
            continue
        fi
        
        echo "--------------------------------------------------------"
        echo "Evaluating Model: $model_name"
        echo "--------------------------------------------------------"

        # Always refresh target-side selection/evaluation. A result filename
        # alone cannot prove that it belongs to the latest checkpoint, source
        # revision, or Jetson runtime.
        if "${PYTHON_CMD[@]}" -m src.evaluate \
            --model-dir "${model_dir%/}" \
            --data-dir "$DATA_DIR"; then
            printf '\nDetailed results saved for %s.\n' "$model_name"
        else
            FAILED+=("$model_name")
            printf '\nEvaluation failed for %s; continuing with the remaining models.\n' \
                "$model_name" >&2
        fi
    fi
done

if [ "${#FAILED[@]}" -gt 0 ]; then
    printf 'Evaluation failures: %s\n' "${FAILED[*]}" >&2
    exit 1
fi

echo "All evaluations complete!"
