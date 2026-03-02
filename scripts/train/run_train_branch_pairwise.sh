#!/bin/bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:4,5,6,7}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PY_SCRIPT="${REPO_ROOT}/scripts/train/train_branch_pairwise.py"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"

LOG_DIR="${REPO_ROOT}/scripts/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_branch_pairwise_${TIMESTAMP}.log"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/train/tree_traversal_rl}"
SUBSETS="${SUBSETS:-all}"
MODEL_PATH="${MODEL_PATH:-/data2/da02/models/Qwen3-4B-Instruct-2507}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/train/branch_pairwise_${TIMESTAMP}}"

NUM_EPOCHS="${NUM_EPOCHS:-1}"
LR="${LR:-1e-5}"
TRAIN_BS="${TRAIN_BS:-2}"
EVAL_BS="${EVAL_BS:-4}"
GRAD_ACC="${GRAD_ACC:-8}"
MAX_LEN="${MAX_LEN:-1024}"
MAX_PAIRS_PER_STEP="${MAX_PAIRS_PER_STEP:-16}"
MIN_REWARD_GAP="${MIN_REWARD_GAP:-0.05}"
MAX_STEPS_PER_SUBSET="${MAX_STEPS_PER_SUBSET:--1}"
SEED="${SEED:-42}"

# Intent: keep first training run stable with conservative defaults and query-level split.
CMD=(
    python "${PY_SCRIPT}"
    --data_root "${DATA_ROOT}"
    --subsets "${SUBSETS}"
    --model_name_or_path "${MODEL_PATH}"
    --output_dir "${OUTPUT_DIR}"
    --num_train_epochs "${NUM_EPOCHS}"
    --learning_rate "${LR}"
    --per_device_train_batch_size "${TRAIN_BS}"
    --per_device_eval_batch_size "${EVAL_BS}"
    --gradient_accumulation_steps "${GRAD_ACC}"
    --max_length "${MAX_LEN}"
    --max_pairs_per_step "${MAX_PAIRS_PER_STEP}"
    --min_reward_gap "${MIN_REWARD_GAP}"
    --max_steps_per_subset "${MAX_STEPS_PER_SUBSET}"
    --seed "${SEED}"
    --bf16
)

printf -v CMD_STR '%q ' "${CMD[@]}"
echo "[INFO] ${CMD_STR}" | tee -a "${LOG_FILE}"
"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
echo "[INFO] done. output_dir=${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
