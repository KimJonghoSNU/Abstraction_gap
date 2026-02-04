#!/bin/bash

# Create log file with timestamp
mkdir -p ../logs
LOG_FILE="../logs/run_round3_thinkqe_$(date '+%Y_%m_%d').log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting run_round3_thinkqe.sh script"

RETRIEVER_MODEL_PATH="/data4/jaeyoung/models/Diver-Retriever-4B"
NODE_EMB_BASE="../trees/BRIGHT"

COMMON_PARAMS=(
    --reasoning_in_traversal_prompt -1
    --load_existing
    --num_iters 10
    --llm_api_backend vllm
    --llm /data2/da02/models/Qwen3-4B-Instruct-2507
    --llm_api_staggering_delay 0.02
    --llm_api_timeout 60
    --llm_api_max_retries 3

    --retriever_model_path "$RETRIEVER_MODEL_PATH"
    --flat_topk 1000
    --rewrite_every 1
    --rewrite_context_topk 5
    --round3_rewrite_context leaf
)

RUN_SUBSETS=(
    "psychology"
    "economics"
    "biology"
    "earth_science"
    "robotics"
    # "sustainable_living"
    # "stackoverflow"
    # "theoremqa_questions"
    # "theoremqa_theorems"
    # "pony"
)

# subset -> tree_version (from trees/BRIGHT/*)
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

PROMPT_NAME="thinkqe_round3"

ROUND3_EXPLORE_MODE="concat"
ANCHOR_LOCAL_RANK_MODES=(
    # "v2"
    # "v3"
    "none"
    # "v4"
)

for subset in "${RUN_SUBSETS[@]}"; do
    for anchor_mode in "${ANCHOR_LOCAL_RANK_MODES[@]}"; do
        suffix="round3_anchor_local_rank_${anchor_mode}_${PROMPT_NAME}"
        NODE_EMB_PATH="${NODE_EMB_BASE}/${subset}/node_embs.diver.npy"

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
        final_args+=("--rewrite_prompt_name" "$PROMPT_NAME")
        final_args+=("--round3_explore_mode" "$ROUND3_EXPLORE_MODE")
        final_args+=("--round3_anchor_local_rank" "$anchor_mode")
        final_args+=("--subset" "$subset")
        tree_version="${TREE_VERSION_MAP[$subset]}"
        if [[ -z "$tree_version" ]]; then
            log "Missing tree version for subset=${subset}"
            exit 1
        fi
        final_args+=("--tree_version" "$tree_version")
        final_args+=("--suffix" "$suffix")

        cmd=( python run_round3_1.py "${final_args[@]}" )
        printf -v cmd_str '%q ' "${cmd[@]}"
        log "Executing: $cmd_str"

        "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            log "Error in subset=${subset} prompt=${PROMPT_NAME} anchor_mode=${anchor_mode}"
            exit 1
        fi

        log "Completed subset=${subset} prompt=${PROMPT_NAME} anchor_mode=${anchor_mode}"
        log "---"
    done
done
