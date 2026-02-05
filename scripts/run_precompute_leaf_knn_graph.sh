#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET="${DATASET:-BRIGHT}"
TOPK="${TOPK:-100}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SUBSETS=(
    # "aops"
    "biology"
    "earth_science"
    "economics"
    "leetcode"
    "pony"
    "psychology"
    "robotics"
    "stackoverflow"
    "sustainable_living"
    "theoremqa_questions"
    "theoremqa_theorems"
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

echo "[INFO] dataset=${DATASET} topk=${TOPK} block_size=${BLOCK_SIZE}"
for subset in "${SUBSETS[@]}"; do
    tree_version="${TREE_VERSION_MAP[$subset]:-}"
    if [[ -z "${tree_version}" ]]; then
        echo "[ERROR] Missing tree version for subset=${subset}"
        exit 1
    fi

    echo "[INFO] subset=${subset} tree_version=${tree_version}"
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/precompute_leaf_knn_graph.py" \
        --dataset "${DATASET}" \
        --subset "${subset}" \
        --tree_version "${tree_version}" \
        --topk "${TOPK}" \
        --block_size "${BLOCK_SIZE}"
done

echo "[OK] Finished precomputing leaf-kNN for all subsets."
