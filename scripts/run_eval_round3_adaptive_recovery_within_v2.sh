#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3

set -euo pipefail

SUBSETS=(
    "biology"
    "earth_science"
    "economics"
    "psychology"
    "robotics"
)

RETRIEVER_MODEL_PATH="/data4/jaeyoung/models/Diver-Retriever-4B"
OUT_DIR="results/analysis/adaptive_recovery_within_v2"
mkdir -p "${OUT_DIR}"

pick_single_or_fail() {
    local label="$1"
    local values="$2"
    local count
    count=$(printf "%s\n" "${values}" | sed '/^$/d' | wc -l)
    if [[ "${count}" -ne 1 ]]; then
        echo "[ERROR] ${label}: expected exactly 1 match, found ${count}" >&2
        printf "%s\n" "${values}" >&2
        exit 1
    fi
    printf "%s\n" "${values}" | sed '/^$/d'
}

for subset in "${SUBSETS[@]}"; do
    echo "[INFO] subset=${subset}"

    v2_candidates=$(
        find "results/BRIGHT/${subset}" -type f -name "all_eval_sample_dicts.pkl" \
        | grep "S=round3_anchor_local_rank_v2_round3_action_v1-" \
        | grep "RPN=round3_action_v1" \
        | grep "RALR=v2" || true
    )
    v2_pkl=$(pick_single_or_fail "v2(${subset})" "${v2_candidates}")

    out_json="${OUT_DIR}/${subset}_action_v1_within_v2_off_vs_on_iter01.json"
    echo "[INFO] v2=${v2_pkl}"
    echo "[INFO] out=${out_json}"

    python scripts/eval_round3_adaptive_recovery_within_v2.py \
        --eval_pkl "${v2_pkl}" \
        --dataset BRIGHT \
        --subset "${subset}" \
        --retriever_model_path "${RETRIEVER_MODEL_PATH}" \
        --eval_k 10 \
        --iter_indices "0,1" \
        --output_json "${out_json}"
done

echo "[DONE] Wrote reports to ${OUT_DIR}"
