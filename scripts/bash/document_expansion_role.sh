#!/usr/bin/env bash
set -euo pipefail

# export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES="0,1"
MODEL_PATH="${MODEL_PATH:-/data2/Qwen3-30B-A3B-Instruct-2507}"
PROMPT_NAME="${PROMPT_NAME:-category_assign_v2}"

subsets=(
    biology
    earth_science
    economics
    psychology
    robotics
    sustainable_living
    pony
    stackoverflow
)

# extra_args=()
# if [[ "${OVERWRITE:-0}" == "1" ]]; then
#     extra_args+=(--overwrite)
# fi

for subset in "${subsets[@]}"; do
    python scripts/document_expansion_role.py \
        --dataset BRIGHT \
        --subset "${subset}" \
        --prompt_name "${PROMPT_NAME}" \
        --llm "${MODEL_PATH}" \
        --batch_size 24 \
        --max_tokens 256 \
        --prompt_token_margin 32 \
        --max_desc_words 4096 \
        --overwrite
done
