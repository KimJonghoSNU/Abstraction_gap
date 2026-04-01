#!/bin/bash

set -euo pipefail

mkdir -p ../logs
LOG_FILE="../logs/run_baseline3_leaf_only_loop_thinkqe_style_$(date '+%Y_%m_%d').log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting run_baseline3_leaf_only_loop_thinkqe_style.sh script"
log "run_leaf_rank.py will use the bundled ThinkQE-style loop: previous-context exclusion + cumulative blocklist + accumulated concat query updates."

RETRIEVER_MODEL_PATH="/data2/pretrained_models/reason-embed-qwen3-8b-0928"
NODE_EMB_BASE="../trees/BRIGHT"

COMMON_PARAMS=(
    --suffix baseline3_leaf_only_loop_thinkqe_style
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
    --leaf_thinkqe_style
    --retriever_model_path "$RETRIEVER_MODEL_PATH"
    --flat_topk 100
    # Intent: keep the prompt fixed and isolate the ablation to the ThinkQE-style interaction loop.
    # --rewrite_prompt_name thinkqe
    --rewrite_prompt_name agent_executor_v1_icl2
    --rewrite_context_topk 10
)

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
    NODE_EMB_PATH="${NODE_EMB_BASE}/${subset}/node_embs.reasonembed8b.npy"

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

    final_args+=("--node_emb_path" "$NODE_EMB_PATH")
    final_args+=("${ITER_ARR[@]}")

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
