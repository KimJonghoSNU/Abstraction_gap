#!/bin/bash

mkdir -p ../logs
LOG_FILE="../logs/run_mcts_reseat_$(date '+%Y_%m_%d').log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting run_mcts.sh script"

RETRIEVER_MODEL_PATH="${RETRIEVER_MODEL_PATH:-/data2/pretrained_models/reason-embed-qwen3-8b-0928}"
NODE_EMB_BASE="${NODE_EMB_BASE:-../trees/BRIGHT}"
ROUND5_DISABLE_CALIBRATION="${ROUND5_DISABLE_CALIBRATION:-1}"
ROUND5_SELECTOR_MODES="${ROUND5_SELECTOR_MODES:-meanscore_global}"
ROUND6_EXPANDABLE_MODE="${ROUND6_EXPANDABLE_MODE:-ended_reseat}"
ROUND6_EXPANDABLE_CANDIDATE_MODE="${ROUND6_EXPANDABLE_CANDIDATE_MODE:-direct_children}"
ROUND6_EXPANDABLE_ENDED_SCOPE="${ROUND6_EXPANDABLE_ENDED_SCOPE:-leftover_expandable}"
ROUND5_REWRITE_PROMPT_NAME="${ROUND5_REWRITE_PROMPT_NAME:-agent_executor_v1_icl2}"

ROUND6_EMR_MEMORY="${ROUND6_EMR_MEMORY:-0}"
ROUND6_EMR_TOPK="${ROUND6_EMR_TOPK:-10}"
ROUND6_EMR_SENT_TOPK="${ROUND6_EMR_SENT_TOPK:-10}"
ROUND6_EMR_COMPRESSION="${ROUND6_EMR_COMPRESSION:-on}"
ROUND6_EMR_MEMORY_MAX_TOKENS="${ROUND6_EMR_MEMORY_MAX_TOKENS:-0}"

NUM_ITERS="${NUM_ITERS:-10}"
NUM_EVAL_SAMPLES="${NUM_EVAL_SAMPLES:-1000}"
MAX_BEAM_SIZE="${MAX_BEAM_SIZE:-10}"
FLAT_TOPK="${FLAT_TOPK:-1000}"
REWRITE_CONTEXT_TOPK="${REWRITE_CONTEXT_TOPK:-10}"
LOCAL_POOL_TOPK="${LOCAL_POOL_TOPK:-100}"

LLM_API_BACKEND="${LLM_API_BACKEND:-vllm}"
LLM_PATH="${LLM_PATH:-/data2/da02/models/Qwen3-4B-Instruct-2507}"
LLM_API_STAGGERING_DELAY="${LLM_API_STAGGERING_DELAY:-0.02}"
LLM_API_TIMEOUT="${LLM_API_TIMEOUT:-60}"
LLM_API_MAX_RETRIES="${LLM_API_MAX_RETRIES:-3}"

MCTS_NUM_SIMULATIONS="${MCTS_NUM_SIMULATIONS:-32}"
MCTS_EXPLORATION_C="${MCTS_EXPLORATION_C:-1.4}"

COMMON_PARAMS=(
    --reasoning_in_traversal_prompt -1
    --load_existing
    --num_iters "$NUM_ITERS"
    --num_eval_samples "$NUM_EVAL_SAMPLES"
    --max_beam_size "$MAX_BEAM_SIZE"

    --llm_api_backend "$LLM_API_BACKEND"
    --llm "$LLM_PATH"
    --llm_api_staggering_delay "$LLM_API_STAGGERING_DELAY"
    --llm_api_timeout "$LLM_API_TIMEOUT"
    --llm_api_max_retries "$LLM_API_MAX_RETRIES"

    --retriever_model_path "$RETRIEVER_MODEL_PATH"
    --flat_topk "$FLAT_TOPK"
    --rewrite_context_topk "$REWRITE_CONTEXT_TOPK"
    --round5_mrr_pool_k "$LOCAL_POOL_TOPK"

    --round6_expandable_mode "$ROUND6_EXPANDABLE_MODE"
    --round6_expandable_candidate_mode "$ROUND6_EXPANDABLE_CANDIDATE_MODE"
    --round6_expandable_ended_scope "$ROUND6_EXPANDABLE_ENDED_SCOPE"
    --round6_expandable_reseat_policy mcts

    --mcts_num_simulations "$MCTS_NUM_SIMULATIONS"
    --mcts_exploration_c "$MCTS_EXPLORATION_C"
)

if [[ "$ROUND5_DISABLE_CALIBRATION" == "1" ]]; then
    # Intent: keep the parity MCTS ablation on the same retriever-score branch logic as the round6 ended-reseat runs.
    COMMON_PARAMS+=(--disable_calibration)
fi

if [[ "$ROUND6_EMR_MEMORY" == "1" ]]; then
    # Intent: parity MCTS must share the same EMR query-memory path when the comparison target enables EMR.
    COMMON_PARAMS+=(
        --round6_emr_memory
        --round6_emr_topk "$ROUND6_EMR_TOPK"
        --round6_emr_sent_topk "$ROUND6_EMR_SENT_TOPK"
        --round6_emr_compression "$ROUND6_EMR_COMPRESSION"
        --round6_emr_memory_max_tokens "$ROUND6_EMR_MEMORY_MAX_TOKENS"
    )
fi

RUN_SUBSETS=(
    "biology"
    "psychology"
    "economics"
    "earth_science"
    "robotics"
    "sustainable_living"
    "stackoverflow"
    "theoremqa_questions"
    "theoremqa_theorems"
    "pony"
    "aops"
    "leetcode"
)

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
)

read -r -a SELECTOR_MODES <<< "$ROUND5_SELECTOR_MODES"
for selector_mode in "${SELECTOR_MODES[@]}"; do
    for query_source in "${QUERY_SOURCES[@]}"; do
        for subset in "${RUN_SUBSETS[@]}"; do
            tree_version="${TREE_VERSION_MAP[$subset]}"
            if [[ -z "$tree_version" ]]; then
                log "Missing tree version for subset=${subset}"
                exit 1
            fi

            NODE_EMB_PATH="${NODE_EMB_BASE}/${subset}/node_embs.reasonembed8b.npy"
            if [[ ! -f "$NODE_EMB_PATH" ]]; then
                log "Missing node embedding file for subset=${subset}: ${NODE_EMB_PATH}"
                exit 1
            fi

            suffix="mcts_reseat_uct_${ROUND6_EXPANDABLE_ENDED_SCOPE}"

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

            final_args+=("--round5_selector_mode" "$selector_mode")
            final_args+=("--subset" "$subset")
            final_args+=("--tree_version" "$tree_version")
            final_args+=("--node_emb_path" "$NODE_EMB_PATH")
            final_args+=("--query_source" "$query_source")
            final_args+=("--suffix" "$suffix")
            if [[ -n "$ROUND5_REWRITE_PROMPT_NAME" ]]; then
                # Intent: keep rewrite prompting matched to round6 so only the ended-slot replacement controller changes.
                final_args+=("--rewrite_prompt_name" "$ROUND5_REWRITE_PROMPT_NAME")
            fi

            cmd=( python run_mcts.py "${final_args[@]}" )
            printf -v cmd_str '%q ' "${cmd[@]}"
            log "Executing: $cmd_str"

            "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"
            if [ ${PIPESTATUS[0]} -ne 0 ]; then
                log "Error in selector_mode=${selector_mode} subset=${subset} query_source=${query_source}"
                exit 1
            fi

            log "Completed selector_mode=${selector_mode} subset=${subset} query_source=${query_source}"
            log "---"
        done
    done
done

log "All runs completed successfully"
