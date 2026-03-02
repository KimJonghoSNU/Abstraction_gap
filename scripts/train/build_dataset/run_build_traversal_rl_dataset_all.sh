#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PY_SCRIPT="${REPO_ROOT}/scripts/train/build_dataset/build_traversal_rl_dataset.py"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"

LOG_DIR="${REPO_ROOT}/scripts/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/build_traversal_rl_dataset_all_${TIMESTAMP}.log"

MAX_DEPTH="${MAX_DEPTH:-auto}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/data/train/tree_traversal_rl/all_${TIMESTAMP}}"
BASELINE_SCRIPT="${BASELINE_SCRIPT:-${REPO_ROOT}/src/bash/baselines/run_baseline1_tree_only.sh}"

# Intent: use unified settings across all subsets for integrated policy training data.
CMD=(
    python "${PY_SCRIPT}"
    --subsets all
    --tree_map_source baseline_script
    --baseline_run_script "${BASELINE_SCRIPT}"
    --max_depth "${MAX_DEPTH}"
    --beam_size 2
    --neg_penalty -0.1
    --empty_penalty -1.0
    --max_samples "${MAX_SAMPLES}"
    --out_dir "${OUT_DIR}"
)

printf -v CMD_STR '%q ' "${CMD[@]}"
echo "[INFO] ${CMD_STR}" | tee -a "${LOG_FILE}"
"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
echo "[INFO] done. out_dir=${OUT_DIR}" | tee -a "${LOG_FILE}"
