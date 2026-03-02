#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-claude-sonnet-4-5}"
# SUBSET="${SUBSET:-all}"
SUBSET="${SUBSET:-biology}"
VERSION="${VERSION:-v3}"
ENV_FILE="${ENV_FILE:-.env}"
SNOWFLAKE_REQUEST_TIMEOUT="${SNOWFLAKE_REQUEST_TIMEOUT:-600}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_TOKENS="${MAX_TOKENS:-256}"
SUMMARY_MAX_TOKENS="${SUMMARY_MAX_TOKENS:-256}"
CLUSTER_MAX_TOKENS="${CLUSTER_MAX_TOKENS:-3072}"
MAX_INPUT_PROMPT_CHARS="${MAX_INPUT_PROMPT_CHARS:-120000}"
MAX_DESC_WORDS="${MAX_DESC_WORDS:-4096}"
PARSE_RETRY_MAX="${PARSE_RETRY_MAX:-3}"
MAX_BRANCHING="${MAX_BRANCHING:-20}"
MAX_PARTITION_DEPTH="${MAX_PARTITION_DEPTH:-5}"
CLUSTER_PROMPT_WORD_BUDGET="${CLUSTER_PROMPT_WORD_BUDGET:-12000}"
CATEGORY_DESC_KEY_TERMS_TOPK="${CATEGORY_DESC_KEY_TERMS_TOPK:-0}"
CATEGORY_DESC_MAX_CHARS="${CATEGORY_DESC_MAX_CHARS:-320}"
CATEGORY_DESC_CHILD_ANCHOR_TOPK="${CATEGORY_DESC_CHILD_ANCHOR_TOPK:-0}"
CATEGORY_DESC_CHILD_ANCHOR_MIN_RATIO="${CATEGORY_DESC_CHILD_ANCHOR_MIN_RATIO:-0.12}"
CATEGORY_DESC_TAIL_ANCHOR_TOPK="${CATEGORY_DESC_TAIL_ANCHOR_TOPK:-0}"
SUMMARY_CACHE_PATH="${SUMMARY_CACHE_PATH:-}"
DISABLE_SUMMARY_CACHE="${DISABLE_SUMMARY_CACHE:-0}"
OVERWRITE="${OVERWRITE:-0}"

cmd=(
    python scripts/tree_builder/build_category_dag_topdown_algo4_v3.py
    --dataset BRIGHT
    --subset "${SUBSET}"
    --version "${VERSION}"
    --llm "${MODEL_NAME}"
    --env_file "${ENV_FILE}"
    --snowflake_request_timeout "${SNOWFLAKE_REQUEST_TIMEOUT}"
    --batch_size "${BATCH_SIZE}"
    --max_tokens "${MAX_TOKENS}"
    --summary_max_tokens "${SUMMARY_MAX_TOKENS}"
    --cluster_max_tokens "${CLUSTER_MAX_TOKENS}"
    --max_input_prompt_chars "${MAX_INPUT_PROMPT_CHARS}"
    --max_desc_words "${MAX_DESC_WORDS}"
    --parse_retry_max "${PARSE_RETRY_MAX}"
    --max_branching "${MAX_BRANCHING}"
    --max_partition_depth "${MAX_PARTITION_DEPTH}"
    --cluster_prompt_word_budget "${CLUSTER_PROMPT_WORD_BUDGET}"
    --category_desc_key_terms_topk "${CATEGORY_DESC_KEY_TERMS_TOPK}"
    --category_desc_max_chars "${CATEGORY_DESC_MAX_CHARS}"
    --category_desc_child_anchor_topk "${CATEGORY_DESC_CHILD_ANCHOR_TOPK}"
    --category_desc_child_anchor_min_ratio "${CATEGORY_DESC_CHILD_ANCHOR_MIN_RATIO}"
    --category_desc_tail_anchor_topk "${CATEGORY_DESC_TAIL_ANCHOR_TOPK}"
)

if [[ -n "${SUMMARY_CACHE_PATH}" ]]; then
    # Intent: allow explicit cache file control while keeping default cache path behavior in Python.
    cmd+=(--summary_cache_path "${SUMMARY_CACHE_PATH}")
fi

if [[ "${DISABLE_SUMMARY_CACHE}" == "1" ]]; then
    cmd+=(--disable_summary_cache)
fi

if [[ "${OVERWRITE}" == "1" ]]; then
    # Intent: explicit overwrite toggle prevents accidental regeneration of expensive DAG artifacts.
    cmd+=(--overwrite)
fi

echo "[Run] subset=${SUBSET} version=${VERSION} model=${MODEL_NAME} algo=topdown_algo4_v3 summary_max_tokens=${SUMMARY_MAX_TOKENS} cluster_max_tokens=${CLUSTER_MAX_TOKENS}"
"${cmd[@]}"
