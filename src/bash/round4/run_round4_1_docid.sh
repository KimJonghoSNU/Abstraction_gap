#!/bin/bash

# Create log file with timestamp
mkdir -p ../logs
LOG_FILE="../logs/run_round4_1_docid_$(date '+%Y_%m_%d').log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting run_round4_1_docid.sh script"

RETRIEVER_MODEL_PATH="/data4/jaeyoung/models/Diver-Retriever-4B"
NODE_EMB_BASE="../trees/BRIGHT"
V6_KNN_TOPK=100

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
    # Intent: isolate doc_id-driven pre-planner runs from website_title baseline.
    --round4_preplanner_reference_mode doc_id
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
    "round3_agent_executor_v5"
)
ROUND4_ITER0_PROMPT_NAME="round3_agent_executor_v1"

ROUND3_EXPLORE_MODE="concat"
ANCHOR_LOCAL_RANK_MODES=(
    # "v2"
    # "v3"
    "none"
    # "v4"
    # "v6"
    # "v5"
)

for prompt in "${PROMPTS[@]}"; do
    for subset in "${RUN_SUBSETS[@]}"; do
        for anchor_mode in "${ANCHOR_LOCAL_RANK_MODES[@]}"; do
            suffix="round4_1_docid_v5_iter0v1_anchor_local_rank_${anchor_mode}_${prompt}"
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
            # Intent: first rewrite iteration uses broad v1 prompt, then iter>=1 switches to v5 taxonomy prompt.
            final_args+=("--round4_iter0_prompt_name" "$ROUND4_ITER0_PROMPT_NAME")
            final_args+=("--round3_explore_mode" "$ROUND3_EXPLORE_MODE")
            final_args+=("--round3_anchor_local_rank" "$anchor_mode")
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

            # Intent: use round4_1 runner that adds pre-planner guidance before rewrite.
            cmd=( python run_round4_1.py "${final_args[@]}" )
            printf -v cmd_str '%q ' "${cmd[@]}"
            log "Executing: $cmd_str"

            "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"
            if [ ${PIPESTATUS[0]} -ne 0 ]; then
                log "Error in subset=${subset} prompt=${prompt} anchor_mode=${anchor_mode}"
                exit 1
            fi

            log "Completed subset=${subset} prompt=${prompt} anchor_mode=${anchor_mode}"
            log "---"
        done
    done
done
