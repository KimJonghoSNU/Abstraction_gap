#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="results/BRIGHT/analysis/baseline_branchhit"
mkdir -p "$OUT_DIR"

SUBSETS=(
    "biology"
    "robotics"
    "economics"
    "psychology"
    "earth_science"
)

for subset in "${SUBSETS[@]}"; do
    base_dir="results/BRIGHT/${subset}/S=${subset}-TV=bottom-up-TPV=5-RInTP=/1-NumLC=10-PlTau=5.0-RCF=0.5-LlmApiB=vllm/Llm=Qwen3-4B-Instruct-2507-NumI=10/NumES=1000-MaxBS=2-S=baseline1_tree_only-TPTF=evidence_support_v1.txt-FT=200/GBT=10-PreFRS=branch-RM=concat-RE=1-RCT=5/RCS=mixed-RGT=10-RRrfK=60-RRC=leaf-REM=replace/RSC=on"
    eval_samples_pkl="${base_dir}/all_eval_sample_dicts.pkl"
    all_metrics_pkl="${base_dir}/all_eval_metrics.pkl"
    out_csv="${OUT_DIR}/${subset}_baseline_branchhit.csv"

    if [[ ! -f "$eval_samples_pkl" ]]; then
        echo "[WARN] Missing baseline eval samples for ${subset}: ${eval_samples_pkl}"
        continue
    fi

    cmd=(python scripts/analyze_baseline_branch_selection.py --baseline_eval_samples_pkl "$eval_samples_pkl" --out_csv "$out_csv")
    if [[ -f "$all_metrics_pkl" ]]; then
        # Intent: include coverage side-summary for the exact same run when available.
        cmd+=(--baseline_all_eval_metrics_pkl "$all_metrics_pkl")
    fi

    echo "[INFO] Running baseline branch analysis for ${subset}"
    "${cmd[@]}"
done

echo "[INFO] Done. CSV outputs: ${OUT_DIR}"
