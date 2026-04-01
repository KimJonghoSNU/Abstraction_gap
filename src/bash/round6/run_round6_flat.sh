#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_DIR="$(cd "${SRC_DIR}/.." && pwd)"
LOG_DIR="${REPO_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run_round6_flat_$(date '+%Y_%m_%d').log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

cd "${SRC_DIR}"

log "Starting run_round6_flat.sh script"

# Intent: keep this launcher isolated from the legacy expandable launcher so the flat-descendant ablation is easy to compare.
RETRIEVER_MODEL_PATH="${RETRIEVER_MODEL_PATH:-/data2/pretrained_models/reason-embed-qwen3-8b-0928}"
NODE_EMB_BASE="${NODE_EMB_BASE:-../trees/BRIGHT}"
ROUND5_DISABLE_CALIBRATION="${ROUND5_DISABLE_CALIBRATION:-1}"
ROUND5_SELECTOR_MODES="${ROUND5_SELECTOR_MODES:-meanscore_global}"
ROUND6_EXPANDABLE_MODE="${ROUND6_EXPANDABLE_MODE:-ended_reseat}"
ROUND6_EXPANDABLE_CANDIDATE_MODE="${ROUND6_EXPANDABLE_CANDIDATE_MODE:-descendant_flat}"
ROUND6_EXPANDABLE_ENDED_SCOPES="${ROUND6_EXPANDABLE_ENDED_SCOPES:-whole_tree_flat goexplore_direct_child}"
ROUND5_REWRITE_PROMPT_NAME="${ROUND5_REWRITE_PROMPT_NAME:-agent_executor_v1_icl2}"
MAX_BEAM_SIZE="${MAX_BEAM_SIZE:-10}"
NUM_EVAL_SAMPLES="${NUM_EVAL_SAMPLES:-1000}"
NUM_ITERS="${NUM_ITERS:-10}"
FLAT_TOPK="${FLAT_TOPK:-1000}"
REWRITE_CONTEXT_TOPK="${REWRITE_CONTEXT_TOPK:-10}"
ROUND5_MRR_POOL_K="${ROUND5_MRR_POOL_K:-100}"

COMMON_PARAMS=(
    --reasoning_in_traversal_prompt -1
    --load_existing
    --num_iters "${NUM_ITERS}"
    --num_eval_samples "${NUM_EVAL_SAMPLES}"
    --max_beam_size "${MAX_BEAM_SIZE}"

    --llm_api_backend vllm
    --llm /data2/da02/models/Qwen3-4B-Instruct-2507
    --llm_api_staggering_delay 0.02
    --llm_api_timeout 60
    --llm_api_max_retries 3

    --retriever_model_path "${RETRIEVER_MODEL_PATH}"
    --flat_topk "${FLAT_TOPK}"
    --rewrite_context_topk "${REWRITE_CONTEXT_TOPK}"
    --round5_mrr_pool_k "${ROUND5_MRR_POOL_K}"
)

if [[ "${ROUND5_DISABLE_CALIBRATION}" == "1" ]]; then
    # Intent: keep the flat-descendant comparison focused on retriever-score branch control rather than calibration effects.
    COMMON_PARAMS+=(--disable_calibration)
fi

RUN_SUBSETS=(
    "biology"
    "psychology"
    "economics"
    "earth_science"
    "robotics"
    "sustainable_living"
    "stackoverflow"
    "theoremqa_questions"
    "theoremqa_theorems"
    "pony"
    "aops"
    "leetcode"
)

declare -A TREE_VERSION_MAP=(
    ["aops"]="top-down"
    ["biology"]="bottom-up"
    ["earth_science"]="bottom-up"
    ["economics"]="bottom-up"
    ["leetcode"]="top-down"
    ["pony"]="bottom-up"
    ["psychology"]="bottom-up"
    ["robotics"]="bottom-up"
    ["stackoverflow"]="bottom-up"
    ["sustainable_living"]="bottom-up"
    ["theoremqa_questions"]="top-down"
    ["theoremqa_theorems"]="top-down"
)

QUERY_SOURCES=(
    "original"
)

read -r -a SELECTOR_MODES <<< "${ROUND5_SELECTOR_MODES}"
read -r -a ENDED_SCOPES <<< "${ROUND6_EXPANDABLE_ENDED_SCOPES}"
for selector_mode in "${SELECTOR_MODES[@]}"; do
    for ended_scope in "${ENDED_SCOPES[@]}"; do
        for query_source in "${QUERY_SOURCES[@]}"; do
            for subset in "${RUN_SUBSETS[@]}"; do
                tree_version="${TREE_VERSION_MAP[$subset]}"
                if [[ -z "${tree_version}" ]]; then
                    log "Missing tree version for subset=${subset}"
                    exit 1
                fi

                NODE_EMB_PATH="${NODE_EMB_BASE}/${subset}/node_embs.reasonembed8b.npy"
                if [[ ! -f "${NODE_EMB_PATH}" ]]; then
                    log "Missing node embedding file for subset=${subset}: ${NODE_EMB_PATH}"
                    exit 1
                fi

                # Intent: surface both descendant-flat active selection and ended-scope choice directly in the result suffix.
                suffix="round6_mrr_selector_accum_${selector_mode}_expandable_${ROUND6_EXPANDABLE_MODE}_${ROUND6_EXPANDABLE_CANDIDATE_MODE}_${ended_scope}"

                final_args=()
                i=0
                while (( i < ${#COMMON_PARAMS[@]} )); do
                    key="${COMMON_PARAMS[i]}"
                    final_args+=("${key}")
                    if (( i + 1 < ${#COMMON_PARAMS[@]} )) && [[ "${COMMON_PARAMS[i + 1]}" != --* ]]; then
                        final_args+=("${COMMON_PARAMS[i + 1]}")
                        ((i+=2))
                        continue
                    fi
                    ((i++))
                done

                final_args+=(--round5_selector_mode "${selector_mode}")
                final_args+=(--round6_expandable_mode "${ROUND6_EXPANDABLE_MODE}")
                final_args+=(--round6_expandable_candidate_mode "${ROUND6_EXPANDABLE_CANDIDATE_MODE}")
                final_args+=(--round6_expandable_ended_scope "${ended_scope}")
                final_args+=(--subset "${subset}")
                final_args+=(--tree_version "${tree_version}")
                final_args+=(--node_emb_path "${NODE_EMB_PATH}")
                final_args+=(--query_source "${query_source}")
                final_args+=(--suffix "${suffix}")
                if [[ -n "${ROUND5_REWRITE_PROMPT_NAME}" ]]; then
                    # Intent: keep prompt behavior matched to round6 legacy so this launcher isolates controller changes.
                    final_args+=(--rewrite_prompt_name "${ROUND5_REWRITE_PROMPT_NAME}")
                fi

                cmd=(python run_round6.py "${final_args[@]}")
                printf -v cmd_str '%q ' "${cmd[@]}"
                log "Executing: ${cmd_str}"

                "${cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"
                if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
                    log "Error in selector_mode=${selector_mode} ended_scope=${ended_scope} subset=${subset} query_source=${query_source}"
                    exit 1
                fi

                log "Completed selector_mode=${selector_mode} ended_scope=${ended_scope} subset=${subset} query_source=${query_source}"
                log "---"
            done
        done
    done
done

log "All runs completed successfully"
