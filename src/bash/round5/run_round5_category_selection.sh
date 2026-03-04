#!/bin/bash

# Resolve repository root so path references are stable regardless of current working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

mkdir -p "${REPO_ROOT}/src/logs"
LOG_FILE="${REPO_ROOT}/src/logs/run_round5_category_selection_$(date '+%Y_%m_%d').log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting run_round5_category_selection.sh script"

VLLM_BASE_URL_FILES=(
    "${REPO_ROOT}/scripts/logs/vllm_base_url.txt"
    "${REPO_ROOT}/logs/vllm_base_url.txt"
)
for url_file in "${VLLM_BASE_URL_FILES[@]}"; do
    if [[ -f "${url_file}" ]]; then
        # Intent: endpoint count should follow active vLLM cluster topology without hardcoding worker count.
        export VLLM_BASE_URL="$(cat "${url_file}")"
        log "Using VLLM_BASE_URL from ${url_file}: ${VLLM_BASE_URL}"
        break
    fi
done
if [[ -z "${VLLM_BASE_URL:-}" ]]; then
    log "VLLM base URL file not found; falling back to existing VLLM_BASE_URL env or run_round5.py defaults."
fi

# Keep all relative paths stable by executing from the src directory.
cd "${REPO_ROOT}/src" || exit 1

RETRIEVER_MODEL_PATH="/data4/jaeyoung/models/Diver-Retriever-4B"
NODE_EMB_BASE="../trees/BRIGHT"
# Intent: this script is for category-selection runs; default mode is category.
ROUND5_MODE="${ROUND5_MODE:-category}"
ROUND5_CATEGORY_K="${ROUND5_CATEGORY_K:-3}"
ROUND5_CATEGORY_GENERATOR_PROMPT_NAME="${ROUND5_CATEGORY_GENERATOR_PROMPT_NAME:-round5_category_generator_v1}"
ROUND5_CATEGORY_REWRITE_PROMPT_NAME="${ROUND5_CATEGORY_REWRITE_PROMPT_NAME:-round5_agent_executor_category_v1}"

COMMON_PARAMS=(
    --reasoning_in_traversal_prompt -1
    --load_existing
    --num_iters 10
    --num_eval_samples 1000
    --max_beam_size 5

    --llm_api_backend vllm
    --llm /data2/Qwen3-30B-A3B-Instruct-2507
    --llm_api_staggering_delay 0.02
    --llm_api_timeout 60
    --llm_api_max_retries 3

    --retriever_model_path "$RETRIEVER_MODEL_PATH"
    --flat_topk 1000
    --rewrite_context_topk 5
    --round5_mrr_pool_k 100
    --round5_mode "$ROUND5_MODE"
)

if [[ "$ROUND5_MODE" == "category" ]]; then
    COMMON_PARAMS+=(
        --round5_category_k "$ROUND5_CATEGORY_K"
        --round5_category_generator_prompt_name "$ROUND5_CATEGORY_GENERATOR_PROMPT_NAME"
        --round5_category_rewrite_prompt_name "$ROUND5_CATEGORY_REWRITE_PROMPT_NAME"
    )
fi

RUN_SUBSETS=(
    "biology"
    "psychology"
    "economics"
    "earth_science"
    "robotics"
    # "sustainable_living"
    # "stackoverflow"
    # "theoremqa_questions"
    # "theoremqa_theorems"
    # "pony"
)

# subset -> tree_version
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
    # "gpt4"
)

for query_source in "${QUERY_SOURCES[@]}"; do
    for subset in "${RUN_SUBSETS[@]}"; do
        tree_version="${TREE_VERSION_MAP[$subset]}"
        if [[ -z "$tree_version" ]]; then
            log "Missing tree version for subset=${subset}"
            exit 1
        fi

        NODE_EMB_PATH="${NODE_EMB_BASE}/${subset}/node_embs.diver.npy"
        if [[ ! -f "$NODE_EMB_PATH" ]]; then
            log "Missing node embedding file for subset=${subset}: ${NODE_EMB_PATH}"
            exit 1
        fi

        # Intent: keep suffix minimal because run_round5.py is fixed to agent_executor_v1.
        # suffix="round5_mrr_selector"
        suffix="round5_mrr_selector_accum" # accumulated leaf pool 

        final_args=()
        i=0
        while (( i < ${#COMMON_PARAMS[@]} )); do
            key="${COMMON_PARAMS[i]}"
            final_args+=("$key")
            if (( i + 1 < ${#COMMON_PARAMS[@]} )) && [[ "${COMMON_PARAMS[i+1]}" != --* ]]; then
                final_args+=("${COMMON_PARAMS[i+1]}")
                ((i+=2))
                continue
            fi
            ((i++))
        done

        final_args+=("--subset" "$subset")
        final_args+=("--tree_version" "$tree_version")
        final_args+=("--node_emb_path" "$NODE_EMB_PATH")
        final_args+=("--query_source" "$query_source")
        final_args+=("--suffix" "$suffix")

        cmd=( python run_round5.py "${final_args[@]}" )
        printf -v cmd_str '%q ' "${cmd[@]}"
        log "Executing: $cmd_str"

        "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            log "Error in subset=${subset} query_source=${query_source}"
            exit 1
        fi

        log "Completed subset=${subset} query_source=${query_source}"
        log "---"
    done
done

log "All runs completed successfully"
