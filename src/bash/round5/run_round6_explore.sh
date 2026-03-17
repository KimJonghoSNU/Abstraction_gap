#!/bin/bash

mkdir -p ../logs
LOG_FILE="../logs/run_round6_explore_$(date '+%Y_%m_%d').log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting run_round6_explore.sh script"

RETRIEVER_MODEL_PATH="/data4/jaeyoung/models/Diver-Retriever-4B"
NODE_EMB_BASE="../trees/BRIGHT"
ROUND5_DISABLE_CALIBRATION="${ROUND5_DISABLE_CALIBRATION:-1}"
ROUND5_SELECTOR_MODES="${ROUND5_SELECTOR_MODES:-meanscore_global}" # retriever_slate maxscore_global meanscore_global  max_hit_global
ROUND6_GLOBAL_ESCAPE="${ROUND6_GLOBAL_ESCAPE:-0}"
ROUND6_GLOBAL_ESCAPE_SLOTS="${ROUND6_GLOBAL_ESCAPE_SLOTS:-2}"
ROUND6_METHOD2="${ROUND6_METHOD2:-1}"
ROUND6_METHOD2_MODE="${ROUND6_METHOD2_MODE:-expandable_pool}"
ROUND6_EXPANDABLE_POOL_FREEZE_TERMINAL_BEAM="${ROUND6_EXPANDABLE_POOL_FREEZE_TERMINAL_BEAM:-0}"
ROUND6_FUSION_MODES="${ROUND6_FUSION_MODES:-rrf max_score sum_score}"
ROUND6_EXPLORE_PROMPT_NAME="${ROUND6_EXPLORE_PROMPT_NAME:-agent_executor_v1_icl2_explore}"
ROUND5_REWRITE_PROMPT_NAME="${ROUND5_REWRITE_PROMPT_NAME:-agent_executor_v1_icl2}"
# ROUND5_REWRITE_PROMPT_NAME="${ROUND5_REWRITE_PROMPT_NAME:-agent_executor_v1_icl2_rubric}"
# ROUND5_REWRITE_PROMPT_NAME="${ROUND5_REWRITE_PROMPT_NAME:-thinkqe_round3}"

COMMON_PARAMS=(
    --reasoning_in_traversal_prompt -1
    --load_existing
    --num_iters 10
    --num_eval_samples 1000
    --max_beam_size 10

    --llm_api_backend vllm
    --llm /data2/da02/models/Qwen3-4B-Instruct-2507
    --llm_api_staggering_delay 0.02
    --llm_api_timeout 60
    --llm_api_max_retries 3

    --retriever_model_path "$RETRIEVER_MODEL_PATH"
    --flat_topk 1000
    --rewrite_context_topk 10
    --round5_mrr_pool_k 100
)

if [[ "$ROUND5_DISABLE_CALIBRATION" == "1" ]]; then
    # Intent: default to retriever-score-only branch logic in round5 legacy runs.
    COMMON_PARAMS+=(--disable_calibration)
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

read -r -a SELECTOR_MODES <<< "$ROUND5_SELECTOR_MODES"
read -r -a FUSION_MODES <<< "$ROUND6_FUSION_MODES"
if [[ "$ROUND6_METHOD2" == "1" && "$ROUND6_METHOD2_MODE" == "expandable_pool" ]]; then
    # Intent: expandable_pool uses fusion as diagnostics only, so avoid running duplicate jobs per fusion mode.
    FUSION_MODES=("rrf")
fi
for selector_mode in "${SELECTOR_MODES[@]}"; do
    for fusion_mode in "${FUSION_MODES[@]}"; do
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

                # Intent: append selector mode to suffix so each selector run remains clearly separable in logs/results.
                suffix="round6_mrr_selector_accum_${selector_mode}"
                if [[ "$ROUND6_GLOBAL_ESCAPE" == "1" ]]; then
                    # Intent: tag method1 runs explicitly so global-escape results are separable from local-only runs.
                    suffix="${suffix}_gescape${ROUND6_GLOBAL_ESCAPE_SLOTS}"
                fi
                if [[ "$ROUND6_METHOD2" == "1" ]]; then
                    # Intent: include method2 mode explicitly so archive_replace and expandable_pool runs stay separable.
                    suffix="${suffix}_method2_${ROUND6_METHOD2_MODE}"
                    if [[ "$ROUND6_METHOD2_MODE" == "archive_replace" ]]; then
                        suffix="${suffix}_${fusion_mode}"
                    elif [[ "$ROUND6_EXPANDABLE_POOL_FREEZE_TERMINAL_BEAM" == "1" ]]; then
                        suffix="${suffix}_freeze_terminal"
                    fi
                fi

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
                if [[ "$ROUND6_GLOBAL_ESCAPE" == "1" ]]; then
                    final_args+=("--round6_global_escape")
                    final_args+=("--round6_global_escape_slots" "$ROUND6_GLOBAL_ESCAPE_SLOTS")
                fi
                if [[ "$ROUND6_METHOD2" == "1" ]]; then
                    final_args+=("--round6_method2")
                    final_args+=("--round6_method2_mode" "$ROUND6_METHOD2_MODE")
                    if [[ "$ROUND6_METHOD2_MODE" == "archive_replace" ]]; then
                        final_args+=("--round6_fusion_mode" "$fusion_mode")
                        final_args+=("--round6_explore_prompt_name" "$ROUND6_EXPLORE_PROMPT_NAME")
                    elif [[ "$ROUND6_EXPANDABLE_POOL_FREEZE_TERMINAL_BEAM" == "1" ]]; then
                        final_args+=("--round6_expandable_pool_freeze_terminal_beam")
                    fi
                fi
                if [[ -n "$ROUND5_REWRITE_PROMPT_NAME" ]]; then
                    # Intent: expose rewrite prompt override without changing default legacy behavior.
                    final_args+=("--rewrite_prompt_name" "$ROUND5_REWRITE_PROMPT_NAME")
                fi

                cmd=( python run_round6.py "${final_args[@]}" )
                printf -v cmd_str '%q ' "${cmd[@]}"
                log "Executing: $cmd_str"

                "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"
                if [ ${PIPESTATUS[0]} -ne 0 ]; then
                    log "Error in selector_mode=${selector_mode} fusion_mode=${fusion_mode} subset=${subset} query_source=${query_source}"
                    exit 1
                fi

                log "Completed selector_mode=${selector_mode} fusion_mode=${fusion_mode} subset=${subset} query_source=${query_source}"
                log "---"
            done
        done
    done
done

log "All runs completed successfully"
