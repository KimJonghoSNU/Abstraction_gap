#!/bin/bash

# Create log file with timestamp
mkdir -p ../logs
LOG_FILE="../logs/run_round3_category_support_$(date '+%Y_%m_%d').log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting run_round3_category_support.sh script"

RETRIEVER_MODEL_PATH="/data4/jaeyoung/models/Diver-Retriever-4B"
NODE_EMB_BASE="../trees/BRIGHT"
V6_KNN_TOPK=100

ROUND3_CATEGORY_POLICY="soft"
ROUND3_CATEGORY_SOFT_KEEP=2
ROUND3_CATEGORY_SUPPORT_TOPK=10
ROUND3_CATEGORY_EXPLORE_BETA=0.1
ROUND3_REWRITE_HISTORY_TOPK=10

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
    --round3_category_policy "$ROUND3_CATEGORY_POLICY"
    # Intent: keep=1 gives hard category focus while reusing one soft-policy code path.
    # --round3_category_soft_keep "$ROUND3_CATEGORY_SOFT_KEEP"
    # --round3_category_support_topk "$ROUND3_CATEGORY_SUPPORT_TOPK"
    # --round3_category_explore_beta "$ROUND3_CATEGORY_EXPLORE_BETA"
)

RUN_SUBSETS=(
    "biology"
    "psychology"
    "economics"
    "earth_science"
    "robotics"
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
)

ROUND3_EXPLORE_MODE="concat"
ANCHOR_LOCAL_RANK_MODES=(
    "none"
    # "v2"
    # "v3"
    # "v4"
    # "v5"
    # "v6"
)

HISTORY_PREFIX_MODES=(
    "off"
    "on"
)

for history_mode in "${HISTORY_PREFIX_MODES[@]}"; do
    for prompt in "${PROMPTS[@]}"; do
        for subset in "${RUN_SUBSETS[@]}"; do
            for anchor_mode in "${ANCHOR_LOCAL_RANK_MODES[@]}"; do
                suffix="round3_category_support_${anchor_mode}_${prompt}_history_${history_mode}"
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
                final_args+=("--rewrite_prompt_name" "$prompt")
                final_args+=("--round3_explore_mode" "$ROUND3_EXPLORE_MODE")
                final_args+=("--round3_anchor_local_rank" "$anchor_mode")
                # Intent: toggle retrieval-history prefix in a single script for clean A/B comparison.
                if [[ "$history_mode" == "on" ]]; then
                    final_args+=("--round3_rewrite_use_history")
                    final_args+=("--round3_rewrite_history_topk" "$ROUND3_REWRITE_HISTORY_TOPK")
                fi
                if [[ "$anchor_mode" == "v6" ]]; then
                    V6_KNN_PATH="${NODE_EMB_BASE}/${subset}/leaf_knn_top${V6_KNN_TOPK}.npz"
                    if [[ ! -f "$V6_KNN_PATH" ]]; then
                        log "Missing v6 kNN file for subset=${subset}: ${V6_KNN_PATH}"
                        exit 1
                    fi
                    # Intent: force v6 runs to use precomputed leaf-kNN for consistent and faster expansion.
                    final_args+=("--round3_v6_leaf_knn_path" "$V6_KNN_PATH")
                fi
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
                log "Executing (history=${history_mode}): $cmd_str"

                "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"
                if [ ${PIPESTATUS[0]} -ne 0 ]; then
                    log "Error in subset=${subset} prompt=${prompt} anchor_mode=${anchor_mode} history=${history_mode}"
                    exit 1
                fi

                log "Completed subset=${subset} prompt=${prompt} anchor_mode=${anchor_mode} history=${history_mode}"
                log "---"
            done
        done
    done
done
