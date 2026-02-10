#!/bin/bash

# Create log file with timestamp
mkdir -p ../logs
LOG_FILE="../logs/run_round4_anchor_local_rank_$(date '+%Y_%m_%d').log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting run_round4_anchor_local_rank.sh script"

RETRIEVER_MODEL_PATH="/data4/jaeyoung/models/Diver-Retriever-4B"
NODE_EMB_BASE="../trees/BRIGHT"

ROUND3_CATEGORY_POLICY="soft"
ROUND4_SUPPORT_TOPM=10
ROUND4_RULE_A_MARGIN_TAU=0.035
ROUND4_RULE_B_DROP_TAU=0.01

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
    --round4_support_topm "$ROUND4_SUPPORT_TOPM"
    --round4_rule_a_margin_tau "$ROUND4_RULE_A_MARGIN_TAU"
    --round4_rule_b_drop_tau "$ROUND4_RULE_B_DROP_TAU"
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
    # "round3_action_v1"
    "round3_agent_executor_v1"
    # "round3_free_rewrite_v2"
)

ROUND3_EXPLORE_MODE="concat"
ROUND4_RULES=(
    "rule_a"
    "rule_b"
)
ANCHOR_LOCAL_RANK_MODES=(
    # Intent: run both baseline anchor settings in round4 for apples-to-apples comparison.
    "none"
    # "v2"
)

for round4_rule in "${ROUND4_RULES[@]}"; do
    for prompt in "${PROMPTS[@]}"; do
        for subset in "${RUN_SUBSETS[@]}"; do
            for anchor_mode in "${ANCHOR_LOCAL_RANK_MODES[@]}"; do
                suffix="round4_${round4_rule}_${anchor_mode}_${prompt}"
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
                final_args+=("--round4_rule_name" "$round4_rule")
                final_args+=("--subset" "$subset")
                tree_version="${TREE_VERSION_MAP[$subset]}"
                if [[ -z "$tree_version" ]]; then
                    log "Missing tree version for subset=${subset}"
                    exit 1
                fi
                final_args+=("--tree_version" "$tree_version")
                final_args+=("--suffix" "$suffix")

                cmd=( python run_round4.py "${final_args[@]}" )
                printf -v cmd_str '%q ' "${cmd[@]}"
                log "Executing (rule=${round4_rule}): $cmd_str"

                "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"
                if [ ${PIPESTATUS[0]} -ne 0 ]; then
                    log "Error in subset=${subset} prompt=${prompt} anchor_mode=${anchor_mode} rule=${round4_rule}"
                    exit 1
                fi

                log "Completed subset=${subset} prompt=${prompt} anchor_mode=${anchor_mode} rule=${round4_rule}"
                log "---"
            done
        done
    done
done
