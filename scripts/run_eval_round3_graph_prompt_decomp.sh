#!/bin/bash

set -euo pipefail

SUBSETS=(
    "biology"
    "earth_science"
    "economics"
    "psychology"
    "robotics"
)

OUT_DIR="results/analysis/graph_prompt_decomp"
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

    baseline_candidates=$(
        find "results/BRIGHT/${subset}" -type f -name "all_eval_sample_dicts.pkl" \
        | grep "RPN=thinkqe_round3" \
        | grep -v "RALR=" || true
    )


    ours_candidates=$(
        find "results/BRIGHT/${subset}" -type f -name "all_eval_sample_dicts.pkl" \
        | grep "RPN=round3_action_v1" \
        | grep -v "RALR=" || true
    )

    baseline_pkl=$(pick_single_or_fail "baseline(${subset})" "${baseline_candidates}")
    ours_pkl=$(pick_single_or_fail "ours(${subset})" "${ours_candidates}")

    out_json="${OUT_DIR}/${subset}_thinkqe_round3_vs_action_v1_none_iter0_to_1.json"
    echo "[INFO] baseline=${baseline_pkl}"
    echo "[INFO] ours=${ours_pkl}"
    echo "[INFO] out=${out_json}"
    

    python scripts/eval_round3_graph_prompt_decomp.py \
        --baseline_pkl "${baseline_pkl}" \
        --ours_pkl "${ours_pkl}" \
        --iter_pre 0 \
        --iter_post 1 \
        --output_json "${out_json}"
done

echo "[DONE] Wrote reports to ${OUT_DIR}"
