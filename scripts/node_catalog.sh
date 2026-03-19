#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_GPU="4"
MODEL_PATH="/data2/pretrained_models/reason-embed-qwen3-8b-0928"
OUT_FILE_NAME="node_embs.reasonembed8b.npy"
DEFAULT_CATEGORIES=(
    "biology"
    "earth_science"
    "psychology"
    "leetcode"
    "economics"
    "robotics"
    "stackoverflow"
    "sustainable_living"
    "pony"
    "aops"
    "theoremqa_questions"
    "theoremqa_theorems"
)

usage() {
    cat <<'EOF'
Usage:
    bash scripts/node_catalog.sh
    bash scripts/node_catalog.sh <gpu_id> <category> [<category> ...]
    bash scripts/node_catalog.sh --print-default-categories
EOF
}

if [[ "${1:-}" == "--print-default-categories" ]]; then
    printf '%s\n' "${DEFAULT_CATEGORIES[@]}"
    exit 0
fi

gpu_id="${DEFAULT_GPU}"
categories=()

if (( $# == 0 )); then
    # Intent: preserve the previous single-GPU/full-category behavior for direct invocations.
    categories=("${DEFAULT_CATEGORIES[@]}")
else
    if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
        usage
        exit 0
    fi

    gpu_id="$1"
    shift
    if (( $# == 0 )); then
        usage >&2
        exit 1
    fi
    categories=("$@")
fi

export CUDA_VISIBLE_DEVICES="${gpu_id}"

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] MODEL_PATH=${MODEL_PATH}"
echo "[INFO] categories=${categories[*]}"

for category in "${categories[@]}"; do
    cmd=(
        python "${REPO_ROOT}/scripts/embed_node_catalog.py"
        --node_catalog_jsonl "${REPO_ROOT}/trees/BRIGHT/${category}/node_catalog.jsonl"
        --model_path "${MODEL_PATH}"
        --out_npy "${REPO_ROOT}/trees/BRIGHT/${category}/${OUT_FILE_NAME}"
    )

    printf '[INFO] '
    printf '%q ' "${cmd[@]}"
    printf '\n'

    # Intent: keep skip semantics centralized in embed_node_catalog.py so worker orchestration stays thin.
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        continue
    fi

    "${cmd[@]}"
done
