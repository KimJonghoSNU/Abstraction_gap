import argparse
import os
import pickle as pkl
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def _is_prefix(prefix: Sequence[int], full: Sequence[int]) -> bool:
    p = tuple(prefix)
    f = tuple(full)
    return (len(p) <= len(f)) and (f[: len(p)] == p)


def _branch_is_gold_ancestor(branch_path: Sequence[int], gold_paths: Sequence[Sequence[int]]) -> bool:
    return any(_is_prefix(branch_path, gold_path) for gold_path in gold_paths)


def _collect_predicted_parents_by_step(prediction_tree: Dict[str, Any]) -> Dict[int, List[Tuple[int, ...]]]:
    by_step: Dict[int, List[Tuple[int, ...]]] = defaultdict(list)
    stack: List[Dict[str, Any]] = [prediction_tree]
    while stack:
        node = stack.pop()
        children = node.get("child") or []
        if children:
            step = children[0].get("creation_step")
            if step is not None:
                by_step[int(step)].append(tuple(node.get("path", ())))
            for child in children:
                stack.append(child)
    return by_step


def analyze_baseline_branch_selection(eval_sample_dicts: List[Dict[str, Any]]) -> pd.DataFrame:
    per_iter_rows: Dict[int, List[Dict[str, float]]] = defaultdict(list)

    for sample in eval_sample_dicts:
        prediction_tree = sample.get("prediction_tree")
        gold_paths = [tuple(x) for x in sample.get("gold_paths", []) if x]
        if not prediction_tree or not gold_paths:
            continue

        by_step = _collect_predicted_parents_by_step(prediction_tree)
        if not by_step:
            continue

        # Intent: normalize creation_step offsets to iteration index so samples are aligned temporally.
        ordered_steps = sorted(by_step.keys())
        sample_iter_idx = 0
        for step in ordered_steps:
            selected_branches = [tuple(x) for x in by_step.get(step, []) if len(tuple(x)) > 0]
            if not selected_branches:
                continue
            good_flags = [_branch_is_gold_ancestor(branch, gold_paths) for branch in selected_branches]

            per_iter_rows[sample_iter_idx].append(
                {
                    "BranchHit@B": 100.0 * float(any(good_flags)),
                    "BranchAllHit@B": 100.0 * float(all(good_flags)),
                    "BranchPrecision@B": 100.0 * float(np.mean(good_flags)),
                    "NumSelectedBranches": float(len(selected_branches)),
                    "SelectedDepth": float(np.mean([len(x) for x in selected_branches])),
                }
            )
            sample_iter_idx += 1

    rows: List[Dict[str, float]] = []
    for iter_idx in sorted(per_iter_rows.keys()):
        iter_df = pd.DataFrame(per_iter_rows[iter_idx])
        rows.append(
            {
                "iter": int(iter_idx),
                "BranchHit@B_mean": float(iter_df["BranchHit@B"].mean()),
                "BranchAllHit@B_mean": float(iter_df["BranchAllHit@B"].mean()),
                "BranchPrecision@B_mean": float(iter_df["BranchPrecision@B"].mean()),
                "NumSelectedBranches_mean": float(iter_df["NumSelectedBranches"].mean()),
                "SelectedDepth_mean": float(iter_df["SelectedDepth"].mean()),
                "num_samples_used": int(iter_df.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def summarize_coverage(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    iter_ids = sorted(
        {
            int(str(col[0]).replace("Iter ", ""))
            for col in metrics_df.columns
            if isinstance(col, tuple) and str(col[0]).startswith("Iter ")
        }
    )
    for iter_idx in iter_ids:
        cov_col = (f"Iter {iter_idx}", "Coverage")
        if cov_col not in metrics_df.columns:
            continue
        rows.append(
            {
                "iter": int(iter_idx),
                "Coverage_mean": float(metrics_df[cov_col].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze baseline branch-hit metrics from prediction_tree.")
    parser.add_argument(
        "--baseline_eval_samples_pkl",
        type=str,
        required=True,
        help="Path to baseline all_eval_sample_dicts.pkl",
    )
    parser.add_argument(
        "--baseline_all_eval_metrics_pkl",
        type=str,
        default=None,
        help="Optional path to baseline all_eval_metrics.pkl (for Coverage summary)",
    )
    parser.add_argument("--out_csv", type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.baseline_eval_samples_pkl):
        raise FileNotFoundError(f"baseline eval samples not found: {args.baseline_eval_samples_pkl}")

    samples = pkl.load(open(args.baseline_eval_samples_pkl, "rb"))
    branch_df = analyze_baseline_branch_selection(samples)
    print("[Baseline] Branch selection metrics by iteration")
    print(branch_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        branch_df.to_csv(args.out_csv, index=False)
        print(f"\nSaved branch metrics CSV: {args.out_csv}")

    if args.baseline_all_eval_metrics_pkl:
        if not os.path.exists(args.baseline_all_eval_metrics_pkl):
            raise FileNotFoundError(f"baseline metrics not found: {args.baseline_all_eval_metrics_pkl}")
        metrics_df = pkl.load(open(args.baseline_all_eval_metrics_pkl, "rb"))
        coverage_df = summarize_coverage(metrics_df)
        print("\n[Baseline] Coverage mean by iteration")
        print(coverage_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(
            "\nNote: Coverage is retrieval breadth (number of predicted/retrieved leaves), "
            "while BranchHit measures branch-direction correctness."
        )


if __name__ == "__main__":
    main()
