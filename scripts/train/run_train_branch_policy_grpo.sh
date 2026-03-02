#!/bin/bash

set -euo pipefail

# Intent: default to 4 GPUs but allow caller override (e.g., export CUDA_VISIBLE_DEVICES=4,5,6,7).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
# Intent: reduce CUDA allocator fragmentation risk for long-context GRPO runs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PY_SCRIPT="${REPO_ROOT}/scripts/train/train_branch_policy_grpo.py"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"

LOG_DIR="${REPO_ROOT}/scripts/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_branch_policy_grpo_${TIMESTAMP}.log"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/train/tree_traversal_rl}"
SUBSETS="${SUBSETS:-biology,earth_science,economics,psychology,robotics,sustainable_living,stackoverflow,pony}"
# Intent: default to base model for fresh GRPO training instead of continual tuning from DPO adapter.
MODEL_PATH="${MODEL_PATH:-/data2/da02/models/Qwen3-4B-Instruct-2507}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/train/branch_policy_grpo}"
BASELINE_SCRIPT="${BASELINE_SCRIPT:-${REPO_ROOT}/src/bash/baselines/run_baseline1_tree_only.sh}"
DOCS_ROOT="${DOCS_ROOT:-${REPO_ROOT}/data/BRIGHT/documents}"
PROMPT_TEMPLATE_FILE="${PROMPT_TEMPLATE_FILE:-scripts/train/prompts/evidence_support_v1.txt}"

NUM_PROCESSES="${NUM_PROCESSES:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LR="${LR:-5e-7}"
TRAIN_BS="${TRAIN_BS:-1}"
EVAL_BS="${EVAL_BS:-1}"
GRAD_ACC="${GRAD_ACC:-16}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-256}"
TEMPERATURE="${TEMPERATURE:-0.8}"
KL_BETA="${KL_BETA:-0.01}"
MAX_STEPS_PER_SUBSET="${MAX_STEPS_PER_SUBSET:--1}"
CANDIDATE_CAP="${CANDIDATE_CAP:-20}"
GOLD_MIN_KEEP="${GOLD_MIN_KEEP:-1}"
EVAL_RATIO="${EVAL_RATIO:-0.0}"
MAX_DESC_CHAR_LEN="${MAX_DESC_CHAR_LEN:-1200}"
MAX_PROMPT_PROTO_SIZE="${MAX_PROMPT_PROTO_SIZE:-0}"
USE_VLLM="${USE_VLLM:-1}"
VLLM_SERVER_HOST="${VLLM_SERVER_HOST:-127.0.0.1}"
VLLM_SERVER_PORT="${VLLM_SERVER_PORT:-8000}"
VLLM_SERVER_BASE_URL="${VLLM_SERVER_BASE_URL:-}"
VLLM_GROUP_PORT="${VLLM_GROUP_PORT:-52345}"
SERVER_COMPATIBLE_MODE="${SERVER_COMPATIBLE_MODE:-1}"
SKIP_VLLM_GROUP_PORT_CHECK="${SKIP_VLLM_GROUP_PORT_CHECK:-0}"
SEED="${SEED:-42}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29610}"
LAUNCHER="${LAUNCHER:-accelerate}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${REPO_ROOT}/scripts/train/zero2.yaml}"


GLOBAL_BATCH_SIZE=$((TRAIN_BS * NUM_PROCESSES))
if (( GLOBAL_BATCH_SIZE <= 0 )); then
    echo "[ERROR] invalid global batch size: TRAIN_BS=${TRAIN_BS}, NUM_PROCESSES=${NUM_PROCESSES}" >&2
    exit 1
fi
if (( NUM_GENERATIONS <= 0 )); then
    echo "[ERROR] invalid NUM_GENERATIONS=${NUM_GENERATIONS}" >&2
    exit 1
fi

gcd() {
    local a="$1"
    local b="$2"
    local t
    while (( b != 0 )); do
        t=$((a % b))
        a="$b"
        b="$t"
    done
    echo "$a"
}

lcm() {
    local a="$1"
    local b="$2"
    local g
    g="$(gcd "${a}" "${b}")"
    echo $(( (a / g) * b ))
}

REQUIRED_UNIT="$(lcm "${GLOBAL_BATCH_SIZE}" "${NUM_GENERATIONS}")"
if [[ -z "${GENERATION_BATCH_SIZE}" ]]; then
    # Intent: satisfy both constraints: divisible by global batch size and num_generations.
    GENERATION_BATCH_SIZE="${REQUIRED_UNIT}"
fi
if (( GENERATION_BATCH_SIZE % REQUIRED_UNIT != 0 )); then
    ADJUSTED_GENERATION_BATCH_SIZE=$(( (GENERATION_BATCH_SIZE / REQUIRED_UNIT + 1) * REQUIRED_UNIT ))
    echo "[WARN] generation_batch_size=${GENERATION_BATCH_SIZE} violates GRPO divisibility (global_batch_size=${GLOBAL_BATCH_SIZE}, num_generations=${NUM_GENERATIONS}); using ${ADJUSTED_GENERATION_BATCH_SIZE}."
    GENERATION_BATCH_SIZE="${ADJUSTED_GENERATION_BATCH_SIZE}"
fi

# Intent: optimize ranking->top2 traversal actions directly with subtree nDCG rewards from generated completions.
CMD=(
    python "${PY_SCRIPT}"
    --data_root "${DATA_ROOT}"
    --subsets "${SUBSETS}"
    --model_name_or_path "${MODEL_PATH}"
    --output_dir "${OUTPUT_DIR}"
    --tree_map_source baseline_script
    --baseline_run_script "${BASELINE_SCRIPT}"
    --docs_root "${DOCS_ROOT}"
    --num_train_epochs "${NUM_EPOCHS}"
    --learning_rate "${LR}"
    --per_device_train_batch_size "${TRAIN_BS}"
    --per_device_eval_batch_size "${EVAL_BS}"
    --gradient_accumulation_steps "${GRAD_ACC}"
    --num_generations "${NUM_GENERATIONS}"
    --generation_batch_size "${GENERATION_BATCH_SIZE}"
    --max_completion_length "${MAX_COMPLETION_LENGTH}"
    --eval_ratio "${EVAL_RATIO}"
    --temperature "${TEMPERATURE}"
    --kl_beta "${KL_BETA}"
    --max_steps_per_subset "${MAX_STEPS_PER_SUBSET}"
    --candidate_cap "${CANDIDATE_CAP}"
    --gold_min_keep "${GOLD_MIN_KEEP}"
    --max_desc_char_len "${MAX_DESC_CHAR_LEN}"
    --max_prompt_proto_size "${MAX_PROMPT_PROTO_SIZE}"
    --seed "${SEED}"
    --bf16
)

if [[ "${SERVER_COMPATIBLE_MODE}" == "1" ]]; then
    # Intent: force full-weight bf16 path to avoid LoRA/4bit sync instability with external vLLM generation server.
    CMD+=(--server_compatible_mode)
else
    CMD+=(--load_in_4bit)
fi

if [[ -n "${PROMPT_TEMPLATE_FILE}" ]]; then
    # Intent: keep GRPO prompt source consistent with DPO dataset prompt customization when requested.
    CMD+=(--prompt_template_file "${PROMPT_TEMPLATE_FILE}")
fi

if [[ "${USE_VLLM}" == "1" ]]; then
    # Intent: route GRPO generation to external vLLM server when available.
    CMD+=(--use_vllm)
    CMD+=(--vllm_server_host "${VLLM_SERVER_HOST}")
    CMD+=(--vllm_group_port "${VLLM_GROUP_PORT}")
    if [[ -n "${VLLM_SERVER_BASE_URL}" ]]; then
        CMD+=(--vllm_server_base_url "${VLLM_SERVER_BASE_URL}")
    else
        CMD+=(--vllm_server_port "${VLLM_SERVER_PORT}")
    fi
fi

printf -v CMD_STR '%q ' "${CMD[@]}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee -a "${LOG_FILE}"
echo "[INFO] PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}" | tee -a "${LOG_FILE}"
echo "[INFO] PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF}" | tee -a "${LOG_FILE}"
echo "[INFO] MODEL_PATH=${MODEL_PATH}" | tee -a "${LOG_FILE}"
echo "[INFO] DATA_ROOT=${DATA_ROOT} | SUBSETS=${SUBSETS}" | tee -a "${LOG_FILE}"
echo "[INFO] MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH} | NUM_GENERATIONS=${NUM_GENERATIONS} | GENERATION_BATCH_SIZE=${GENERATION_BATCH_SIZE} | REQUIRED_UNIT=${REQUIRED_UNIT} | CANDIDATE_CAP=${CANDIDATE_CAP} | GOLD_MIN_KEEP=${GOLD_MIN_KEEP} | MAX_DESC_CHAR_LEN=${MAX_DESC_CHAR_LEN}" | tee -a "${LOG_FILE}"
echo "[INFO] USE_VLLM=${USE_VLLM} | VLLM_SERVER_HOST=${VLLM_SERVER_HOST} | VLLM_SERVER_PORT=${VLLM_SERVER_PORT} | VLLM_SERVER_BASE_URL=${VLLM_SERVER_BASE_URL} | VLLM_GROUP_PORT=${VLLM_GROUP_PORT}" | tee -a "${LOG_FILE}"
echo "[INFO] SERVER_COMPATIBLE_MODE=${SERVER_COMPATIBLE_MODE}" | tee -a "${LOG_FILE}"
echo "[INFO] SKIP_VLLM_GROUP_PORT_CHECK=${SKIP_VLLM_GROUP_PORT_CHECK}" | tee -a "${LOG_FILE}"
echo "[INFO] NUM_PROCESSES=${NUM_PROCESSES} | LAUNCHER=${LAUNCHER} | ACCELERATE_CONFIG=${ACCELERATE_CONFIG}" | tee -a "${LOG_FILE}"

if [[ "${USE_VLLM}" == "1" && "${SKIP_VLLM_GROUP_PORT_CHECK}" != "1" ]]; then
    # Intent: fail fast when TRL fixed vLLM communicator port(51216) is already occupied, which otherwise crashes after long data loading.
    if ! python - <<'PY'
import psutil
import sys

group_port = 51216
listeners = []
for conn in psutil.net_connections(kind="tcp"):
    if conn.status != "LISTEN" or conn.laddr is None:
        continue
    if int(conn.laddr.port) == group_port:
        listeners.append(conn)

if listeners:
    pids = sorted({int(c.pid) for c in listeners if c.pid is not None})
    print(
        f"[ERROR] TRL GRPO communicator port {group_port} is already in LISTEN state by pid(s)={pids}.",
        file=sys.stderr,
    )
    print(
        "[ERROR] Restart vLLM server with a non-conflicting internal base port, "
        "e.g. `export VLLM_PORT=27000; bash scripts/train/run_server.sh`.",
        file=sys.stderr,
    )
    sys.exit(1)
print(f"[INFO] preflight ok: communicator port {group_port} is free.")
PY
    then
        exit 1
    fi
fi

if [[ "${NUM_PROCESSES}" -gt 1 ]]; then
    if [[ "${LAUNCHER}" == "torchrun" ]]; then
        CMD=(torchrun --nproc_per_node "${NUM_PROCESSES}" --master_port "${MAIN_PROCESS_PORT}" "${PY_SCRIPT}" "${CMD[@]:2}")
    elif [[ "${LAUNCHER}" == "accelerate" ]]; then
        if [[ -f "${ACCELERATE_CONFIG}" ]]; then
            # Intent: mirror TongSearch GRPO launcher path with accelerate+DeepSpeed Zero2 config.
            CMD=(
                accelerate launch
                --config_file "${ACCELERATE_CONFIG}"
                --num_processes "${NUM_PROCESSES}"
                --main_process_port "${MAIN_PROCESS_PORT}"
                "${PY_SCRIPT}"
                "${CMD[@]:2}"
            )
        else
            # Intent: fail-soft to plain accelerate launch when config file path is not present.
            echo "[WARN] ACCELERATE_CONFIG not found: ${ACCELERATE_CONFIG}. Falling back to accelerate defaults."
            CMD=(accelerate launch --num_processes "${NUM_PROCESSES}" --main_process_port "${MAIN_PROCESS_PORT}" "${PY_SCRIPT}" "${CMD[@]:2}")
        fi
    else
        echo "[ERROR] Unknown LAUNCHER=${LAUNCHER}. Expected one of: torchrun, accelerate" >&2
        exit 1
    fi
fi

printf -v CMD_STR '%q ' "${CMD[@]}"
echo "[INFO] ${CMD_STR}" | tee -a "${LOG_FILE}"
"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
echo "[INFO] done. output_dir=${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
