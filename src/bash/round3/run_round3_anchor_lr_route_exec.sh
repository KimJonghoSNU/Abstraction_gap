#!/bin/bash

# Create log file with timestamp
mkdir -p ../logs
LOG_FILE="../logs/run_round3_anchor_lr_route_exec_$(date '+%Y_%m_%d').log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting run_round3_anchor_lr_route_exec.sh script"

RETRIEVER_MODEL_PATH="/data4/jaeyoung/models/Diver-Retriever-4B"
NODE_EMB_BASE="../trees/BRIGHT"
CACHE_BASE_ROOT="/data4/jongho/Search-o1/data/QA_Datasets/bright/cache"

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
    --round3_anchor_local_rank v2
    --round3_router_prompt_name round3_action_v1_router
)

RUN_SUBSETS=(
    "psychology"
    "economics"
    "biology"
    "earth_science"
    "robotics"
    "sustainable_living"
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

PROMPTS=(
    "round3_action_v1"
    # "round3_action_v2"
)

ROUND3_EXPLORE_MODE="concat"
ROUTER_CONTEXTS=(
    "leaf_branch"
    "leaf_branch_depth1"
)

for prompt in "${PROMPTS[@]}"; do
    for router_context in "${ROUTER_CONTEXTS[@]}"; do
        for subset in "${RUN_SUBSETS[@]}"; do
            suffix="round3_anchor_local_rank_${prompt}"
            NODE_EMB_PATH="${NODE_EMB_BASE}/${subset}/node_embs.diver.npy"
            REWRITE_CACHE_BASE="${CACHE_BASE_ROOT}/rewrite_${prompt}"
            ROUTER_CACHE_BASE="${CACHE_BASE_ROOT}/router_${prompt}"
            REWRITE_CACHE_TAG="round3_${prompt}_ALR=v2_REM=${ROUND3_EXPLORE_MODE}"
            ROUTER_CACHE_TAG="round3_${prompt}_ALR=v2_REM=${ROUND3_EXPLORE_MODE}_router_${router_context}"
            REWRITE_CACHE_PATH="${REWRITE_CACHE_BASE}/${subset}_${prompt}_${REWRITE_CACHE_TAG}.jsonl"
            ROUTER_CACHE_PATH="${ROUTER_CACHE_BASE}/${subset}_${prompt}_${ROUTER_CACHE_TAG}.jsonl"

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
        final_args+=("--rewrite_cache_path" "$REWRITE_CACHE_PATH")
        final_args+=("--round3_router_cache_path" "$ROUTER_CACHE_PATH")
            final_args+=("--rewrite_prompt_name" "$prompt")
            final_args+=("--round3_router_context" "$router_context")
            final_args+=("--round3_explore_mode" "$ROUND3_EXPLORE_MODE")
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
                log "Error in subset=${subset} prompt=${prompt} router_context=${router_context}"
                exit 1
            fi

            log "Completed subset=${subset} prompt=${prompt} router_context=${router_context}"
            log "---"
        done
    done
done
