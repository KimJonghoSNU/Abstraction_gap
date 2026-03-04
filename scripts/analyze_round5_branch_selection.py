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
    for gold_path in gold_paths:
        if _is_prefix(branch_path, gold_path):
            return True
    return False


def _analyze_round5_samples(eval_sample_dicts: List[Dict]) -> pd.DataFrame:
    max_iters = 0
    for sample in eval_sample_dicts:
        max_iters = max(max_iters, len(sample.get("iter_records", [])))

    rows: List[Dict] = []
    for iter_idx in range(max_iters):
        hit_any_list: List[float] = []
        hit_all_list: List[float] = []
        precision_list: List[float] = []
        selected_depths: List[int] = []
        selected_counts: List[int] = []

        for sample in eval_sample_dicts:
            iter_records = sample.get("iter_records", [])
            if iter_idx >= len(iter_records):
                continue

            iter_record = iter_records[iter_idx]
            selected = [tuple(x) for x in iter_record.get("selected_branches_after", []) if x]
            gold_paths = [tuple(x) for x in sample.get("gold_paths", []) if x]
            if not selected:
                continue

            # Intent: compute branch-quality directly from selected branch paths, independent of retrieval size.
            is_good = [_branch_is_gold_ancestor(branch, gold_paths) for branch in selected]

            hit_any_list.append(1.0 if any(is_good) else 0.0)
            hit_all_list.append(1.0 if all(is_good) else 0.0)
            precision_list.append(float(np.mean(is_good)))
            selected_counts.append(len(selected))
            selected_depths.extend([len(x) for x in selected])

        rows.append(
            {
                "iter": int(iter_idx),
                "BranchHit@B_mean": 100.0 * float(np.mean(hit_any_list)) if hit_any_list else np.nan,
                "BranchAllHit@B_mean": 100.0 * float(np.mean(hit_all_list)) if hit_all_list else np.nan,
                "BranchPrecision@B_mean": 100.0 * float(np.mean(precision_list)) if precision_list else np.nan,
                "NumSelectedBranches_mean": float(np.mean(selected_counts)) if selected_counts else np.nan,
                "SelectedDepth_mean": float(np.mean(selected_depths)) if selected_depths else np.nan,
                "num_samples_used": int(len(hit_any_list)),
            }
        )

    return pd.DataFrame(rows)


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


def _analyze_baseline_samples(eval_sample_dicts: List[Dict]) -> pd.DataFrame:
    per_iter_rows: Dict[int, List[Dict[str, float]]] = defaultdict(list)

    for sample in eval_sample_dicts:
        prediction_tree = sample.get("prediction_tree")
        gold_paths = [tuple(x) for x in sample.get("gold_paths", []) if x]
        if (not prediction_tree) or (not gold_paths):
            continue

        by_step = _collect_predicted_parents_by_step(prediction_tree)
        if not by_step:
            continue

        # Intent: align variable creation_step offsets into iteration order so samples are comparable.
        ordered_steps = sorted(by_step.keys())
        sample_iter_idx = 0
        for step in ordered_steps:
            selected = [tuple(x) for x in by_step.get(step, []) if len(tuple(x)) > 0]
            if not selected:
                continue
            is_good = [_branch_is_gold_ancestor(branch, gold_paths) for branch in selected]
            per_iter_rows[sample_iter_idx].append(
                {
                    "BranchHit@B_mean": 100.0 * float(any(is_good)),
                    "BranchAllHit@B_mean": 100.0 * float(all(is_good)),
                    "BranchPrecision@B_mean": 100.0 * float(np.mean(is_good)),
                    "NumSelectedBranches_mean": float(len(selected)),
                    "SelectedDepth_mean": float(np.mean([len(x) for x in selected])),
                }
            )
            sample_iter_idx += 1

    rows: List[Dict[str, float]] = []
    for iter_idx in sorted(per_iter_rows.keys()):
        iter_df = pd.DataFrame(per_iter_rows[iter_idx])
        rows.append(
            {
                "iter": int(iter_idx),
                "BranchHit@B_mean": float(iter_df["BranchHit@B_mean"].mean()),
                "BranchAllHit@B_mean": float(iter_df["BranchAllHit@B_mean"].mean()),
                "BranchPrecision@B_mean": float(iter_df["BranchPrecision@B_mean"].mean()),
                "NumSelectedBranches_mean": float(iter_df["NumSelectedBranches_mean"].mean()),
                "SelectedDepth_mean": float(iter_df["SelectedDepth_mean"].mean()),
                "num_samples_used": int(iter_df.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def _summarize_coverage(all_eval_metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    iter_ids = sorted(
        {
            int(str(col[0]).replace("Iter ", ""))
            for col in all_eval_metrics_df.columns
            if isinstance(col, tuple) and str(col[0]).startswith("Iter ")
        }
    )
    for iter_idx in iter_ids:
        cov_col = (f"Iter {iter_idx}", "Coverage")
        if cov_col not in all_eval_metrics_df.columns:
            continue
        rows.append(
            {
                "iter": int(iter_idx),
                "Coverage_mean": float(all_eval_metrics_df[cov_col].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze round5 per-iteration branch selection quality and compare with coverage metrics."
    )
    parser.add_argument(
        "--round5_eval_samples_pkl",
        type=str,
        required=True,
        help="Path to round5 all_eval_sample_dicts.pkl",
    )
    parser.add_argument(
        "--baseline_all_eval_metrics_pkl",
        type=str,
        default=None,
        help="Optional path to baseline all_eval_metrics.pkl for coverage summary",
    )
    parser.add_argument(
        "--baseline_eval_samples_pkl",
        type=str,
        default=None,
        help="Optional path to baseline all_eval_sample_dicts.pkl for branch-hit comparison",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Optional CSV output path for round5 branch metrics",
    )
    parser.add_argument(
        "--compare_out_csv",
        type=str,
        default=None,
        help="Optional CSV output path for round5-vs-baseline merged branch metrics",
    )
    args = parser.parse_args()

    if not os.path.exists(args.round5_eval_samples_pkl):
        raise FileNotFoundError(f"round5 eval samples not found: {args.round5_eval_samples_pkl}")

    round5_samples = pkl.load(open(args.round5_eval_samples_pkl, "rb"))
    branch_df = _analyze_round5_samples(round5_samples)
    print("[Round5] Branch selection metrics by iteration")
    if branch_df.empty:
        print("No iteration records found.")
    else:
        print(branch_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        branch_df.to_csv(args.out_csv, index=False)
        print(f"\nSaved branch metrics CSV: {args.out_csv}")

    baseline_branch_df = None
    if args.baseline_eval_samples_pkl:
        if not os.path.exists(args.baseline_eval_samples_pkl):
            raise FileNotFoundError(f"baseline eval samples not found: {args.baseline_eval_samples_pkl}")
        baseline_samples = pkl.load(open(args.baseline_eval_samples_pkl, "rb"))
        baseline_branch_df = _analyze_baseline_samples(baseline_samples)
        print("\n[Baseline] Branch selection metrics by iteration")
        if baseline_branch_df.empty:
            print("No baseline branch records found.")
        else:
            print(baseline_branch_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        if not baseline_branch_df.empty and not branch_df.empty:
            r5 = branch_df.rename(columns={c: f"round5_{c}" for c in branch_df.columns if c != "iter"})
            base = baseline_branch_df.rename(
                columns={c: f"baseline_{c}" for c in baseline_branch_df.columns if c != "iter"}
            )
            merged = pd.merge(r5, base, on="iter", how="outer").sort_values("iter")
            for metric in ("BranchHit@B_mean", "BranchPrecision@B_mean", "BranchAllHit@B_mean"):
                r5_col = f"round5_{metric}"
                base_col = f"baseline_{metric}"
                if (r5_col in merged.columns) and (base_col in merged.columns):
                    merged[f"delta_{metric}"] = merged[r5_col] - merged[base_col]

            print("\n[Round5 vs Baseline] Branch metric comparison")
            show_cols = [
                "iter",
                "round5_BranchHit@B_mean",
                "baseline_BranchHit@B_mean",
                "delta_BranchHit@B_mean",
                "round5_BranchPrecision@B_mean",
                "baseline_BranchPrecision@B_mean",
                "delta_BranchPrecision@B_mean",
            ]
            show_cols = [c for c in show_cols if c in merged.columns]
            print(merged[show_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

            if args.compare_out_csv:
                os.makedirs(os.path.dirname(args.compare_out_csv), exist_ok=True)
                merged.to_csv(args.compare_out_csv, index=False)
                print(f"\nSaved comparison CSV: {args.compare_out_csv}")

    if args.baseline_all_eval_metrics_pkl:
        if not os.path.exists(args.baseline_all_eval_metrics_pkl):
            raise FileNotFoundError(f"baseline metrics not found: {args.baseline_all_eval_metrics_pkl}")
        baseline_df = pkl.load(open(args.baseline_all_eval_metrics_pkl, "rb"))
        coverage_df = _summarize_coverage(baseline_df)
        print("\n[Baseline] Coverage mean by iteration")
        if coverage_df.empty:
            print("Coverage columns not found.")
        else:
            print(coverage_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        print(
            "\nNote: Coverage is the number of predicted/retrieved leaf items, "
            "not a branch-correctness metric."
        )


if __name__ == "__main__":
    main()
