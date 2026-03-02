#!/bin/bash

set -euo pipefail

CUDA_VISIBLE_DEVICES=4,5,6,7

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PY_SCRIPT="${REPO_ROOT}/scripts/train/train_branch_policy_dpo.py"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"

LOG_DIR="${REPO_ROOT}/scripts/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_branch_policy_dpo_${TIMESTAMP}.log"

DPO_DATA_DIR="${DPO_DATA_DIR:-${REPO_ROOT}/data/train/traversal_dpo_1500sample}"
TRAIN_JSONL="${TRAIN_JSONL:-${DPO_DATA_DIR}/train_dpo.jsonl}"
EVAL_JSONL="${EVAL_JSONL:-${DPO_DATA_DIR}/eval_dpo.jsonl}"
MODEL_PATH="${MODEL_PATH:-/data2/da02/models/Qwen3-4B-Instruct-2507}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/train/branch_policy_dpo_${TIMESTAMP}}"

NUM_EPOCHS="${NUM_EPOCHS:-1}"
LR="${LR:-5e-6}"
TRAIN_BS="${TRAIN_BS:-4}"
EVAL_BS="${EVAL_BS:-1}"
GRAD_ACC="${GRAD_ACC:-8}"
MAX_LEN="${MAX_LEN:-1024}"
BETA="${BETA:-0.1}"
SEED="${SEED:-42}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29600}"
ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-}"
LAUNCHER="${LAUNCHER:-torchrun}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NUM_PROCESSES="${NUM_PROCESSES:-$(python - <<'PY'
import os
devices = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip() != ""]
print(len(devices) if devices else 1)
PY
)}"
else
    NUM_PROCESSES="${NUM_PROCESSES:-1}"
fi

TRAIN_ARGS=(
    python "${PY_SCRIPT}"
    --train_jsonl "${TRAIN_JSONL}"
    --eval_jsonl "${EVAL_JSONL}"
    --model_name_or_path "${MODEL_PATH}"
    --output_dir "${OUTPUT_DIR}"
    --num_train_epochs "${NUM_EPOCHS}"
    --learning_rate "${LR}"
    --per_device_train_batch_size "${TRAIN_BS}"
    --per_device_eval_batch_size "${EVAL_BS}"
    --gradient_accumulation_steps "${GRAD_ACC}"
    --max_length "${MAX_LEN}"
    --beta "${BETA}"
    --seed "${SEED}"
    --load_in_4bit
    --bf16
)

if [[ "${NUM_PROCESSES}" -gt 1 ]]; then
    if [[ "${LAUNCHER}" == "accelerate" ]]; then
        LAUNCH_CMD=(accelerate launch --num_processes "${NUM_PROCESSES}" --main_process_port "${MAIN_PROCESS_PORT}")
        if [[ -n "${ACCELERATE_CONFIG_FILE}" ]]; then
            # Intent: allow user-provided accelerate/deepspeed config while keeping a sane default launch path.
            LAUNCH_CMD+=(--config_file "${ACCELERATE_CONFIG_FILE}")
        fi
        CMD=("${LAUNCH_CMD[@]}" "${PY_SCRIPT}" "${TRAIN_ARGS[@]:2}")
    else
        # Intent: prefer torchrun for multi-GPU QLoRA DPO to avoid accelerate rank-device ordinal issues.
        CMD=(torchrun --nproc_per_node "${NUM_PROCESSES}" --master_port "${MAIN_PROCESS_PORT}" "${PY_SCRIPT}" "${TRAIN_ARGS[@]:2}")
    fi
else
    CMD=("${TRAIN_ARGS[@]}")
fi

printf -v CMD_STR '%q ' "${CMD[@]}"
echo "[INFO] ${CMD_STR}" | tee -a "${LOG_FILE}"
"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
echo "[INFO] done. output_dir=${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
