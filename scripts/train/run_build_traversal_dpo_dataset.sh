#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PY_SCRIPT="${REPO_ROOT}/scripts/train/build_dataset/build_traversal_dpo_dataset.py"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"

LOG_DIR="${REPO_ROOT}/scripts/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/build_traversal_dpo_dataset_${TIMESTAMP}.log"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/train/tree_traversal_rl}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/data/train/traversal_dpo_${TIMESTAMP}}"
SUBSETS="${SUBSETS:-biology,earth_science,economics,psychology,robotics,sustainable_living,stackoverflow,pony}"
EVAL_RATIO="${EVAL_RATIO:-0.1}"
MAX_PAIRS_PER_STEP_DPO="${MAX_PAIRS_PER_STEP_DPO:-1}"
MAX_CHOSEN_COMBOS="${MAX_CHOSEN_COMBOS:-4}"
MAX_STEPS_PER_SUBSET="${MAX_STEPS_PER_SUBSET:--1}"
MAX_QUERIES_PER_SUBSET="${MAX_QUERIES_PER_SUBSET:-1500}"
PROMPT_TEMPLATE_FILE="${PROMPT_TEMPLATE_FILE:-scripts/train/prompts/evidence_support_v1.txt}"
SEED="${SEED:-42}"

if ! find "${DATA_ROOT}" -mindepth 2 -maxdepth 2 -type f -name "branch_steps.jsonl" | grep -q .; then
    CANDIDATE_ROOT="$(find "${DATA_ROOT}" -mindepth 1 -maxdepth 1 -type d -name "core_*" | sort | tail -n 1 || true)"
    if [[ -n "${CANDIDATE_ROOT}" ]] && find "${CANDIDATE_ROOT}" -mindepth 2 -maxdepth 2 -type f -name "branch_steps.jsonl" | grep -q .; then
        # Intent: auto-select the latest core_* layout when branch files are stored one level deeper.
        DATA_ROOT="${CANDIDATE_ROOT}"
        echo "[INFO] DATA_ROOT auto-resolved to ${DATA_ROOT}" | tee -a "${LOG_FILE}"
    fi
fi

# Intent: cap pair expansion per step so tie-heavy queries do not dominate DPO training distribution.
CMD=(
    python "${PY_SCRIPT}"
    --data_root "${DATA_ROOT}"
    --out_dir "${OUT_DIR}"
    --subsets "${SUBSETS}"
    --eval_ratio "${EVAL_RATIO}"
    --max_pairs_per_step_dpo "${MAX_PAIRS_PER_STEP_DPO}"
    --max_chosen_combos "${MAX_CHOSEN_COMBOS}"
    --max_steps_per_subset "${MAX_STEPS_PER_SUBSET}"
    --max_queries_per_subset "${MAX_QUERIES_PER_SUBSET}"
    --seed "${SEED}"
)

if [[ -n "${PROMPT_TEMPLATE_FILE}" ]]; then
    # Intent: allow prompt customization from shell without modifying Python code or src/prompts.py.
    CMD+=(--prompt_template_file "${PROMPT_TEMPLATE_FILE}")
fi

printf -v CMD_STR '%q ' "${CMD[@]}"
echo "[INFO] ${CMD_STR}" | tee -a "${LOG_FILE}"
"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
echo "[INFO] done. out_dir=${OUT_DIR}" | tee -a "${LOG_FILE}"
