#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKER_SCRIPT="${SCRIPT_DIR}/node_catalog.sh"
LOG_DIR="${SCRIPT_DIR}/logs"
GPUS=(4 5 6 7)
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"

mkdir -p "${LOG_DIR}"

mapfile -t categories < <(bash "${WORKER_SCRIPT}" --print-default-categories)

if (( ${#categories[@]} == 0 )); then
    echo "[ERROR] No categories discovered from ${WORKER_SCRIPT}" >&2
    exit 1
fi

declare -A category_sizes=()
declare -A gpu_loads=()
declare -A gpu_assignments=()
declare -A gpu_pids=()
declare -A gpu_logs=()

for gpu in "${GPUS[@]}"; do
    gpu_loads["${gpu}"]=0
    gpu_assignments["${gpu}"]=""
done

size_file="$(mktemp)"
cleanup() {
    rm -f "${size_file}"
}
trap cleanup EXIT

for category in "${categories[@]}"; do
    catalog_path="${REPO_ROOT}/trees/BRIGHT/${category}/node_catalog.jsonl"
    if [[ ! -f "${catalog_path}" ]]; then
        echo "[ERROR] Missing node catalog: ${catalog_path}" >&2
        exit 1
    fi

    line_count="$(wc -l < "${catalog_path}")"
    category_sizes["${category}"]="${line_count}"
    printf '%s\t%s\n' "${category}" "${line_count}" >> "${size_file}"
done

while IFS=$'\t' read -r category line_count; do
    target_gpu="${GPUS[0]}"
    for gpu in "${GPUS[@]}"; do
        if (( ${gpu_loads["${gpu}"]} < ${gpu_loads["${target_gpu}"]} )); then
            target_gpu="${gpu}"
        fi
    done

    # Intent: greedily balance line-count-heavy subsets because embedding cost scales with catalog size.
    gpu_loads["${target_gpu}"]=$(( ${gpu_loads["${target_gpu}"]} + line_count ))
    if [[ -n "${gpu_assignments["${target_gpu}"]}" ]]; then
        gpu_assignments["${target_gpu}"]+=" "
    fi
    gpu_assignments["${target_gpu}"]+="${category}"
done < <(sort -t $'\t' -k2,2nr -k1,1 "${size_file}")

echo "[INFO] Size-aware GPU assignment:"
for gpu in "${GPUS[@]}"; do
    echo "[INFO] GPU ${gpu} | estimated_lines=${gpu_loads["${gpu}"]} | categories=${gpu_assignments["${gpu}"]}"
done

for gpu in "${GPUS[@]}"; do
    assignment="${gpu_assignments["${gpu}"]}"
    if [[ -z "${assignment}" ]]; then
        continue
    fi

    read -r -a gpu_categories <<< "${assignment}"
    log_file="${LOG_DIR}/node_catalog_gpu${gpu}_${TIMESTAMP}.log"
    gpu_logs["${gpu}"]="${log_file}"

    cmd=(bash "${WORKER_SCRIPT}" "${gpu}" "${gpu_categories[@]}")
    printf '[INFO] log=%s | command=' "${log_file}"
    printf '%q ' "${cmd[@]}"
    printf '\n'

    # Intent: keep one log per GPU worker so partial failures are diagnosable without interleaved output.
    (
        export DRY_RUN="${DRY_RUN:-0}"
        "${cmd[@]}"
    ) > "${log_file}" 2>&1 &
    gpu_pids["${gpu}"]=$!
done

failed=0
for gpu in "${GPUS[@]}"; do
    pid="${gpu_pids["${gpu}"]:-}"
    if [[ -z "${pid}" ]]; then
        continue
    fi

    if wait "${pid}"; then
        echo "[INFO] GPU ${gpu} finished successfully. log=${gpu_logs["${gpu}"]}"
    else
        status="$?"
        echo "[ERROR] GPU ${gpu} failed with status ${status}. log=${gpu_logs["${gpu}"]}" >&2
        failed=1
    fi
done

exit "${failed}"
