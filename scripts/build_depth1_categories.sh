#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3

DATASET="BRIGHT"
PROMPT_NAME="cat_abstract"
LLM_PATH="${5:-/data2/da02/models/Qwen3-4B-Instruct-2507}"
TP_SIZE="4"

SUBSETS=(
    "biology"
    "economics"
    # "earth_science"
    # "psychology"
    # "robotics"
    # "stackoverflow"
    # "sustainable_living"
    # "theoremqa_theorems"
    # "pony"
    # "theoremqa_questions"
    # "leetcode"
    # "aops"
)

for subset in "${SUBSETS[@]}"; do
    python scripts/build_depth1_categories.py \
        --dataset "${DATASET}" \
        --subset "${subset}" \
        --prompt_name "${PROMPT_NAME}" \
        --llm "${LLM_PATH}" \
        --tensor_parallel_size "${TP_SIZE}" \
        --gpu_memory_utilization 0.8
done
