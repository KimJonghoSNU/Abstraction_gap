#!/bin/bash

mkdir -p ../logs
LOG_FILE="../logs/run_round5_$(date '+%Y_%m_%d').log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting run_round5.sh script"

RETRIEVER_MODEL_PATH="/data4/jaeyoung/models/Diver-Retriever-4B"
NODE_EMB_BASE="../trees/BRIGHT"

COMMON_PARAMS=(
    --reasoning_in_traversal_prompt -1
    --load_existing
    --num_iters 10
    --num_eval_samples 1000
    --max_beam_size 10

    --llm_api_backend vllm
    --llm /data2/Qwen3-30B-A3B-Instruct-2507
    --llm_api_staggering_delay 0.02
    --llm_api_timeout 60
    --llm_api_max_retries 3

    --retriever_model_path "$RETRIEVER_MODEL_PATH"
    --flat_topk 1000
    --rewrite_context_topk 5
    --round5_mrr_pool_k 100
)

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
