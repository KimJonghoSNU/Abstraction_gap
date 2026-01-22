#!/bin/bash

# Create log file with timestamp
mkdir -p ../logs
LOG_FILE="../logs/run_round3_explore_mode_ablation_$(date '+%Y_%m_%d').log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting run_round3_explore_mode_ablation.sh script"

RETRIEVER_MODEL_PATH="/data4/jaeyoung/models/Diver-Retriever-4B"
NODE_EMB_BASE="../trees/BRIGHT"

CACHE_BASE_ROOT="/data4/jongho/Search-o1/data/QA_Datasets/bright/cache"
REWRITE_PROMPT_NAME="round3_action_v1"
REWRITE_CACHE_BASE="${CACHE_BASE_ROOT}/rewrite_${REWRITE_PROMPT_NAME}"

COMMON_PARAMS=(
    --reasoning_in_traversal_prompt -1
    --load_existing
    --num_iters 5
    --llm_api_backend vllm
    --llm /data2/da02/models/Qwen3-4B-Instruct-2507
    --llm_api_staggering_delay 0.02
    --llm_api_timeout 60
    --llm_api_max_retries 3

    --retriever_model_path "$RETRIEVER_MODEL_PATH"
    --flat_topk 100
    --rewrite_prompt_name "$REWRITE_PROMPT_NAME"
    --rewrite_every 1
    --rewrite_context_topk 5
    --round3_rewrite_context leaf
)

RUN_SUBSETS=(
    "biology"
    "psychology"
)

EXPLORE_MODES=(
    "replace"
    "original"
)

for subset in "${RUN_SUBSETS[@]}"; do
    for explore_mode in "${EXPLORE_MODES[@]}"; do
        suffix="round3_explore_${explore_mode}"
        NODE_EMB_PATH="${NODE_EMB_BASE}/${subset}/node_embs.diver.npy"
        REWRITE_CACHE_TAG="round3_explore_${explore_mode}"
        REWRITE_CACHE_PATH="${REWRITE_CACHE_BASE}/${subset}_${REWRITE_PROMPT_NAME}_${REWRITE_CACHE_TAG}.jsonl"

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
        final_args+=("--subset" "$subset")
        final_args+=("--tree_version" "bottom-up")
        final_args+=("--suffix" "$suffix")
        final_args+=("--round3_explore_mode" "$explore_mode")

        cmd=( python run_round3.py "${final_args[@]}" )
        printf -v cmd_str '%q ' "${cmd[@]}"
        log "Executing: $cmd_str"

        "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            log "Error in subset=${subset} explore_mode=${explore_mode}"
            exit 1
        fi

        log "Completed subset=${subset} explore_mode=${explore_mode}"
        log "---"
    done
done
