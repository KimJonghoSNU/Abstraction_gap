#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="results/BRIGHT/analysis/round5_vs_baseline"
mkdir -p "$OUT_DIR"

SUBSETS=(
    "biology"
    "robotics"
    "economics"
    "psychology"
    "earth_science"
)

for subset in "${SUBSETS[@]}"; do
    round5_dir="results/BRIGHT/${subset}/round5/S=${subset}-TV=bottom-up-TPV=5-RInTP=/1-NumLC=10-PlTau=5.0-RCF=0.5-LlmApiB=vllm/Llm=Qwen3-4B-Instruct-2507-NumI=10/NumES=1000-MaxBS=2-S=round5_mrr_selector-FT=1000-GBT=10/PreFRS=branch-RPN=agent_executor_v1-RM=concat-RE=1-RCT=5/RCS=mixed-RGT=10-RRrfK=60-RRC=leaf-REM=replace"
    baseline_dir="results/BRIGHT/${subset}/S=${subset}-TV=bottom-up-TPV=5-RInTP=/1-NumLC=10-PlTau=5.0-RCF=0.5-LlmApiB=vllm/Llm=Qwen3-4B-Instruct-2507-NumI=10/NumES=1000-MaxBS=2-S=baseline1_tree_only-TPTF=evidence_support_v1.txt-FT=200/GBT=10-PreFRS=branch-RM=concat-RE=1-RCT=5/RCS=mixed-RGT=10-RRrfK=60-RRC=leaf-REM=replace/RSC=on"

    round5_eval_samples_pkl="${round5_dir}/all_eval_sample_dicts.pkl"
    baseline_eval_samples_pkl="${baseline_dir}/all_eval_sample_dicts.pkl"
    baseline_all_metrics_pkl="${baseline_dir}/all_eval_metrics.pkl"
    out_csv="${OUT_DIR}/${subset}_round5_branchhit.csv"
    compare_out_csv="${OUT_DIR}/${subset}_round5_vs_baseline.csv"

    if [[ ! -f "$round5_eval_samples_pkl" ]]; then
        echo "[WARN] Missing round5 eval samples for ${subset}: ${round5_eval_samples_pkl}"
        continue
    fi
    if [[ ! -f "$baseline_eval_samples_pkl" ]]; then
        echo "[WARN] Missing baseline eval samples for ${subset}: ${baseline_eval_samples_pkl}"
        continue
    fi

    cmd=(
        python scripts/analyze_round5_branch_selection.py
        --round5_eval_samples_pkl "$round5_eval_samples_pkl"
        --baseline_eval_samples_pkl "$baseline_eval_samples_pkl"
        --out_csv "$out_csv"
        --compare_out_csv "$compare_out_csv"
    )
    if [[ -f "$baseline_all_metrics_pkl" ]]; then
        cmd+=(--baseline_all_eval_metrics_pkl "$baseline_all_metrics_pkl")
    fi

    echo "[INFO] Comparing round5 vs baseline for ${subset}"
    "${cmd[@]}"
done

echo "[INFO] Done. CSV outputs: ${OUT_DIR}"
