#!/bin/bash

conda activate lattice2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Create log file with timestamp
mkdir -p "${REPO_ROOT}/logs"
LOG_FILE="${REPO_ROOT}/logs/run_baseline3_leaf_only_loop_emr_$(date '+%Y_%m_%d').log"

LEAF_EMR_MEMORY_MODE="${LEAF_EMR_MEMORY_MODE:-accumulated}"
LEAF_EMR_HISTORY_RANK_TOPK="${LEAF_EMR_HISTORY_RANK_TOPK:-10}"
LEAF_EMR_DOC_TOPK="${LEAF_EMR_DOC_TOPK:-10}"
LEAF_EMR_SENT_TOPK="${LEAF_EMR_SENT_TOPK:-10}"
LEAF_EMR_COMPRESSION="${LEAF_EMR_COMPRESSION:-on}"
LEAF_EMR_MEMORY_MAX_TOKENS="${LEAF_EMR_MEMORY_MAX_TOKENS:-0}"
REWRITE_PROMPT_NAME="${REWRITE_PROMPT_NAME:-agent_executor_v1_icl2}"
if [[ "${LEAF_EMR_MEMORY_MODE}" != "off" && "${REWRITE_PROMPT_NAME}" == "agent_executor_v1_icl2" ]]; then
    # Intent: keep baseline prompt stable while auto-switching to the memory-aware variant only for EMR runs.
    REWRITE_PROMPT_NAME="agent_executor_v1_icl2_emr_memory"
fi

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting run_baseline3_leaf_only_loop_emr.sh script"
log "run_leaf_rank.py will emit leaf_iter_records.jsonl with retrieved/rewrite-context doc ids and paths per iteration."
log "leaf_emr_memory_mode=${LEAF_EMR_MEMORY_MODE} leaf_emr_compression=${LEAF_EMR_COMPRESSION} rewrite_prompt_name=${REWRITE_PROMPT_NAME} leaf_emr_memory_max_tokens=${LEAF_EMR_MEMORY_MAX_TOKENS}"

# Edit these paths for your setup
# RETRIEVER_MODEL_PATH="/data4/jaeyoung/models/Diver-Retriever-4B"
RETRIEVER_MODEL_PATH="/data2/pretrained_models/reason-embed-qwen3-8b-0928"
NODE_EMB_BASE="${REPO_ROOT}/trees/BRIGHT"

# Common params (key value pairs or flags). Run-specific params override these.
COMMON_PARAMS=(
    --suffix baseline3_leaf_only_loop_emr
    --reasoning_in_traversal_prompt -1
    --load_existing
    --num_iters 10
    --llm_api_backend vllm
    --llm /data2/da02/models/Qwen3-4B-Instruct-2507
    --llm_api_staggering_delay 0.02
    --llm_api_timeout 60
    --llm_api_max_retries 3

    --flat_then_tree
    --leaf_only_retrieval
    --retriever_model_path "$RETRIEVER_MODEL_PATH"
    --flat_topk 100
    --rewrite_prompt_name "$REWRITE_PROMPT_NAME"
    # --rewrite_prompt_name thinkqe
    --rewrite_context_topk 10
    --leaf_emr_memory_mode "$LEAF_EMR_MEMORY_MODE"
    --leaf_emr_history_rank_topk "$LEAF_EMR_HISTORY_RANK_TOPK"
    --leaf_emr_doc_topk "$LEAF_EMR_DOC_TOPK"
    --leaf_emr_sent_topk "$LEAF_EMR_SENT_TOPK"
    --leaf_emr_compression "$LEAF_EMR_COMPRESSION"
    --leaf_emr_memory_max_tokens "$LEAF_EMR_MEMORY_MAX_TOKENS"
)

# Define RUNS directly as strings (space-separated args)
RUNS=(
    "--subset biology --tree_version bottom-up"
    "--subset economics --tree_version bottom-up"
    "--subset earth_science --tree_version bottom-up"
    "--subset psychology --tree_version bottom-up"
    "--subset robotics --tree_version bottom-up"
    "--subset stackoverflow --tree_version bottom-up"
    "--subset sustainable_living --tree_version bottom-up"
    "--subset theoremqa_theorems --tree_version top-down"
    "--subset theoremqa_questions --tree_version top-down"
    "--subset pony --tree_version bottom-up"
    "--subset leetcode --tree_version top-down"
    "--subset aops --tree_version top-down"
)

for idx in "${!RUNS[@]}"; do
    iter_def="${RUNS[idx]}"
    read -r -a ITER_ARR <<< "$iter_def"

    # Extract subset for node embedding path
    subset=""
    for ((i=0; i<${#ITER_ARR[@]}; i++)); do
        if [[ "${ITER_ARR[i]}" == "--subset" ]]; then
            subset="${ITER_ARR[i+1]}"
            break
        fi
    done
    if [[ -z "$subset" ]]; then
        log "Missing --subset in RUNS entry: $iter_def"
        exit 1
    fi
    # Intent: keep the non-memory retrieval/tree inputs aligned with the baseline launcher for cleaner ablation.
    NODE_EMB_PATH="${NODE_EMB_BASE}/${subset}/node_embs.reasonembed8b.npy"

    # Build final args: first common params, then iteration-specific params
    final_args=()
    i=0
    while (( i < ${#COMMON_PARAMS[@]} )); do
        key="${COMMON_PARAMS[i]}"
        final_args+=("$key")
        if (( i+1 < ${#COMMON_PARAMS[@]} )) && [[ "${COMMON_PARAMS[i+1]}" != --* ]]; then
            final_args+=("${COMMON_PARAMS[i+1]}")
            ((i+=2))
            continue
        fi
        ((i++))
    done

    # Add node embeddings path for this subset
    final_args+=("--node_emb_path" "$NODE_EMB_PATH")

    # Append iteration-specific params
    final_args+=("${ITER_ARR[@]}")

    cmd=( python "${REPO_ROOT}/src/run_leaf_rank.py" "${final_args[@]}" )
    printf -v cmd_str '%q ' "${cmd[@]}"
    log "Executing: $cmd_str"

    "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log "Error in iteration: $((idx+1))"
        exit 1
    fi

    log "Completed iteration: $((idx+1))"
    log "---"
done

log "All RUNS completed successfully!"
