#!/bin/bash

mkdir -p ../logs
LOG_FILE="../logs/run_round4_biology_drop_ablation_$(date '+%Y_%m_%d').log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting run_round4_biology_drop_ablation.sh script"

RETRIEVER_MODEL_PATH="/data4/jaeyoung/models/Diver-Retriever-4B"
NODE_EMB_BASE="../trees/BRIGHT"

SUBSETS=(
    "biology"
    "psychology"
    "earth_science"
)
TREE_VERSION="bottom-up"
PROMPT="round3_agent_executor_v1"
ANCHOR_MODE="none"

ROUND3_CATEGORY_POLICY="soft"
ROUND4_RULE_NAME="rule_a"
ROUND4_SUPPORT_TOPM=10
ROUND4_RULE_A_MARGIN_TAU=0.035

ANALYSIS_MODES=(
    # Keep exploit-only analysis mode to measure drop-case gain frequency directly.
    "force_drop_one"
)

COMMON_PARAMS=(
    --reasoning_in_traversal_prompt -1
    --load_existing
    --num_iters 5
    --num_eval_samples 1000
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
    --round4_rule_name "$ROUND4_RULE_NAME"
    --round4_support_topm "$ROUND4_SUPPORT_TOPM"
    --round4_rule_a_margin_tau "$ROUND4_RULE_A_MARGIN_TAU"
)

for subset in "${SUBSETS[@]}"; do
    for mode in "${ANALYSIS_MODES[@]}"; do
        suffix="round4_${subset}_drop_ablation_${mode}_${PROMPT}"
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
        final_args+=("--rewrite_prompt_name" "$PROMPT")
        final_args+=("--round3_explore_mode" "concat")
        final_args+=("--round3_anchor_local_rank" "$ANCHOR_MODE")
        final_args+=("--round4_analysis_category_mode" "$mode")
        final_args+=("--subset" "$subset")
        final_args+=("--tree_version" "$TREE_VERSION")
        final_args+=("--suffix" "$suffix")

        cmd=( python run_round4.py "${final_args[@]}" )
        printf -v cmd_str '%q ' "${cmd[@]}"
        log "Executing (subset=${subset}, analysis_mode=${mode}): $cmd_str"

        "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            log "Error in subset=${subset}, analysis_mode=${mode}"
            exit 1
        fi

        log "Completed subset=${subset}, analysis_mode=${mode}"
        log "---"
    done
done

log "Finished run_round4_biology_drop_ablation.sh script"
