#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-claude-sonnet-4-5}"
SUBSET="${SUBSET:-all}"
VERSION="${VERSION:-v1}"
ENV_FILE="${ENV_FILE:-.env}"
SNOWFLAKE_REQUEST_TIMEOUT="${SNOWFLAKE_REQUEST_TIMEOUT:-600}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_TOKENS="${MAX_TOKENS:-256}"
MAX_DESC_WORDS="${MAX_DESC_WORDS:-4096}"
PARSE_RETRY_MAX="${PARSE_RETRY_MAX:-1}"
MAX_ALT_PATHS="${MAX_ALT_PATHS:-1}"
LEAF_PARENT_CAP="${LEAF_PARENT_CAP:-2}"
MAX_BRANCHING="${MAX_BRANCHING:-20}"
OVERWRITE="${OVERWRITE:-0}"

cmd=(
    python scripts/tree_builder/build_category_dag_topdown.py
    --dataset BRIGHT
    --subset "${SUBSET}"
    --version "${VERSION}"
    --llm "${MODEL_NAME}"
    --env_file "${ENV_FILE}"
    --snowflake_request_timeout "${SNOWFLAKE_REQUEST_TIMEOUT}"
    --batch_size "${BATCH_SIZE}"
    --max_tokens "${MAX_TOKENS}"
    --max_desc_words "${MAX_DESC_WORDS}"
    --parse_retry_max "${PARSE_RETRY_MAX}"
    --max_alt_paths "${MAX_ALT_PATHS}"
    --leaf_parent_cap "${LEAF_PARENT_CAP}"
    --max_branching "${MAX_BRANCHING}"
)

if [[ "${OVERWRITE}" == "1" ]]; then
    # Intent: explicit overwrite toggle prevents accidental regeneration of expensive DAG artifacts.
    cmd+=(--overwrite)
fi

echo "[Run] subset=${SUBSET} version=${VERSION} model=${MODEL_NAME}"
"${cmd[@]}"
