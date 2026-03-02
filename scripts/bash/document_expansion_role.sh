#!/usr/bin/env bash
set -euo pipefail

# export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES="0,1"
MODEL_PATH="${MODEL_PATH:-/data2/Qwen3-30B-A3B-Instruct-2507}"
MAX_TOTAL_CATEGORIES_V2="${MAX_TOTAL_CATEGORIES_V2:-10}"
MAX_TOTAL_CATEGORIES_V3="${MAX_TOTAL_CATEGORIES_V3:-10}"
PARSE_RETRY_MAX="${PARSE_RETRY_MAX:-2}"

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

versions=(
    v2
    v3
)

for version in "${versions[@]}"; do
    prompt_name="category_assign_${version}"
    max_total_categories="${MAX_TOTAL_CATEGORIES_V3}"
    if [[ "${version}" == "v2" ]]; then
        max_total_categories="${MAX_TOTAL_CATEGORIES_V2}"
    fi

    # Intent: run v2 then v3 in one sweep while keeping outputs isolated by prompt_name.
    for subset in "${subsets[@]}"; do
        echo "[Run] version=${version} subset=${subset} prompt=${prompt_name}"
        python scripts/document_expansion_role.py \
            --dataset BRIGHT \
            --subset "${subset}" \
            --category_version "${version}" \
            --prompt_name "${prompt_name}" \
            --llm "${MODEL_PATH}" \
            --batch_size 24 \
            --max_tokens 256 \
            --prompt_token_margin 32 \
            --max_desc_words 4096 \
            --max_total_categories "${max_total_categories}" \
            --parse_retry_max "${PARSE_RETRY_MAX}" \
            --overwrite
    done
done
