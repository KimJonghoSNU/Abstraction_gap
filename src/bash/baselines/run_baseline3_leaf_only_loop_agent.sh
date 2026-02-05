#!/bin/bash

set -euo pipefail

# Create log file with timestamp
mkdir -p ../logs
LOG_FILE="../logs/run_baseline3_leaf_only_loop_$(date '+%Y_%m_%d').log"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting run_baseline3_leaf_only_loop.sh script"

# Edit these paths for your setup
RETRIEVER_MODEL_PATH="/data4/jaeyoung/models/Diver-Retriever-4B"
NODE_EMB_BASE="../trees/BRIGHT"

# Common params (key value pairs or flags). Run-specific params override these.
COMMON_PARAMS=(
    --suffix baseline3_leaf_only_loop
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
    --rewrite_prompt_name agent_executor_v1
    # --rewrite_prompt_name baseline_round3_action_v1
)

LEAF_NO_INITIAL_REWRITE=true

# Define RUNS directly as strings (space-separated args)
RUNS=(
    "--subset biology --tree_version bottom-up"
    "--subset economics --tree_version bottom-up"
    "--subset earth_science --tree_version bottom-up"
    "--subset psychology --tree_version bottom-up"
    "--subset robotics --tree_version bottom-up"
    # "--subset stackoverflow --tree_version bottom-up"
    # "--subset sustainable_living --tree_version bottom-up"
    # "--subset theoremqa_theorems --tree_version top-down"
    # "--subset pony --tree_version bottom-up"
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
    NODE_EMB_PATH="${NODE_EMB_BASE}/${subset}/node_embs.diver.npy"

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
    if [[ "${LEAF_NO_INITIAL_REWRITE}" == "true" ]]; then
        final_args+=("--leaf_no_initial_rewrite")
    fi

    cmd=( python run_leaf_rank.py "${final_args[@]}" )
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
