#!/bin/bash

set -euo pipefail

# Resolve repository root so path references are stable regardless of current working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Create log file with timestamp
mkdir -p "${REPO_ROOT}/src/logs"
LOG_FILE="${REPO_ROOT}/src/logs/run_baseline1_tree_only_$(date '+%Y_%m_%d').log"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting run_baseline1_tree_only.sh script"

VLLM_BASE_URL_FILE="${REPO_ROOT}/scripts/logs/vllm_base_url.txt"
if [[ -f "${VLLM_BASE_URL_FILE}" ]]; then
    # Intent: baseline also follows active vLLM topology to avoid dead endpoint retries.
    export VLLM_BASE_URL="$(cat "${VLLM_BASE_URL_FILE}")"
    log "Using VLLM_BASE_URL from ${VLLM_BASE_URL_FILE}: ${VLLM_BASE_URL}"
else
    log "VLLM base URL file not found: ${VLLM_BASE_URL_FILE}"
    log "Falling back to existing VLLM_BASE_URL env or code default endpoints."
fi

# Keep existing relative paths stable by executing from the src directory.
cd "${REPO_ROOT}/src" || exit 1

TARGET_SUBSET="${TARGET_SUBSET:-biology}"
TARGET_TREE_VERSION="${TARGET_TREE_VERSION:-category-topdown-algo4-v3}"

# Common params (key value pairs or flags). Run-specific params override these.
COMMON_PARAMS=(
    --suffix baseline1_tree_only
    --reasoning_in_traversal_prompt -1
    --load_existing
    --num_iters 10
    --llm_api_backend vllm
    --llm /data2/Qwen3-30B-A3B-Instruct-2507
    --llm_api_staggering_delay 0.02
    --llm_api_timeout 60
    --llm_api_max_retries 3
)

# Define RUNS directly as strings (space-separated args)
RUNS=(
    "--subset ${TARGET_SUBSET} --tree_version ${TARGET_TREE_VERSION}"
    # "--subset economics --tree_version bottom-up"
    # "--subset earth_science --tree_version bottom-up"
    # "--subset psychology --tree_version bottom-up"
    # "--subset robotics --tree_version bottom-up"
    # "--subset stackoverflow --tree_version bottom-up"
    # "--subset sustainable_living --tree_version bottom-up"
    # "--subset theoremqa_theorems --tree_version top-down"
    # "--subset pony --tree_version bottom-up"
)

for idx in "${!RUNS[@]}"; do
    iter_def="${RUNS[idx]}"
    read -r -a ITER_ARR <<< "$iter_def"

    # Build final args: first common params, then iteration-specific params
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

    # Append iteration-specific params
    final_args+=("${ITER_ARR[@]}")

    subset=""
    tree_version=""
    for ((j=0; j<${#ITER_ARR[@]}; j++)); do
        if [[ "${ITER_ARR[j]}" == "--subset" ]] && (( j + 1 < ${#ITER_ARR[@]} )); then
            subset="${ITER_ARR[j+1]}"
        fi
        if [[ "${ITER_ARR[j]}" == "--tree_version" ]] && (( j + 1 < ${#ITER_ARR[@]} )); then
            tree_version="${ITER_ARR[j+1]}"
        fi
    done

    # Intent: DAG runtime is enabled by category-topdown-* tree_version, so validate DAG json first in that mode.
    if [[ "${tree_version}" == category-topdown-* ]]; then
        dag_version="${tree_version#category-topdown-}"
        dag_version_u="${dag_version//-/_}"
        dag_path="${REPO_ROOT}/trees/BRIGHT/${subset}/category_dag_topdown_${dag_version_u}.json"
        if [[ ! -f "${dag_path}" ]]; then
            log "Missing DAG file: ${dag_path}"
            exit 1
        fi
    else
        tree_path="${REPO_ROOT}/trees/BRIGHT/${subset}/tree-${tree_version}.pkl"
        if [[ ! -f "${tree_path}" ]]; then
            log "Missing tree file: ${tree_path}"
            exit 1
        fi
    fi

    cmd=( python run.py "${final_args[@]}" )
    printf -v cmd_str '%q ' "${cmd[@]}"
    log "Executing: $cmd_str"

    "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log "Error in iteration: $((idx+1))"
        exit 1
    fi

    log "Completed iteration: $((idx+1))"
    log "---"
done

log "All RUNS completed successfully!"
