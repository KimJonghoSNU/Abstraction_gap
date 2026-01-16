#!/bin/bash

# Create log file with timestamp
mkdir -p ../logs
LOG_FILE="../logs/run_flat_qe_iter_$(date '+%Y_%m_%d').log"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting run_flat_qe_iter.sh script"

# Edit these paths for your setup
RETRIEVER_MODEL_PATH="/data4/jaeyoung/models/Diver-Retriever-4B"
NODE_EMB_BASE="../trees/BRIGHT"

# Cache base dirs (optional, will be created if missing)
QE_CACHE_BASE="/data4/jongho/Search-o1/data/QA_Datasets/bright/stepback_json/pre_flat/qwen3-4b-instruct-2507"
REWRITE_CACHE_BASE="/data4/jongho/Search-o1/data/QA_Datasets/bright/stepback_json/qwen3-4b-instruct-2507"

# Common params (key value pairs or flags). Run-specific params override these.
COMMON_PARAMS=(
    --suffix flat_gate_qe_iter
    --reasoning_in_traversal_prompt -1
    --load_existing
    --num_iters 5
    --llm_api_backend vllm
    --llm /data2/da02/models/Qwen3-4B-Instruct-2507
    --llm_api_staggering_delay 0.02
    --llm_api_timeout 60
    --llm_api_max_retries 3

    # Flat -> Gate -> Traversal
    --flat_then_tree
    --retriever_model_path "$RETRIEVER_MODEL_PATH"
    --flat_topk 100
    --gate_branches_topb 10

    # Pre-flat rewrite (query -> rewrite -> flat retrieval)
    # --qe_prompt_name pre_flat_rewrite_v1
    --qe_prompt_name stepback_json_pre

    # Rewrite after each traversal iteration
    # --rewrite_prompt_name gate_rewrite_v1
    --rewrite_prompt_name stepback_json
    --rewrite_at_start
    --rewrite_every 1
    --rewrite_context_topk 5
    --rewrite_context_source fused
    --rewrite_mode concat
    --qe_force_refresh
    --rewrite_force_refresh
)


# Define RUNS directly as strings (space-separated args)
RUNS=(
    "--subset biology --tree_version bottom-up"
    # "--subset economics --tree_version bottom-up"
    # "--subset earth_science --tree_version bottom-up"
    "--subset psychology --tree_version bottom-up"
    # "--subset robotics --tree_version bottom-up"
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
    QE_CACHE_PATH="${QE_CACHE_BASE}/${subset}_pre_flat_rewrite.jsonl"
    REWRITE_CACHE_PATH="${REWRITE_CACHE_BASE}/${subset}_rewrite.jsonl"

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
    final_args+=("--qe_cache_path" "$QE_CACHE_PATH")
    final_args+=("--rewrite_cache_path" "$REWRITE_CACHE_PATH")

    # Append iteration-specific params
    final_args+=("${ITER_ARR[@]}")

    cmd=( python run.py "${final_args[@]}" )
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
