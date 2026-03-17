#!/usr/bin/env python3
"""
Analyze how often round5 selector reaches the "no_candidate_children" state.

By default this reproduces the devlog analysis for:
    - selector_mode=meanscore_global
    - rewrite prompt=agent_executor_v1_icl2
    - REM=replace
    - non-fused-memory runs only

Examples
--------
python scripts/analyze_round5_no_candidate_children.py

python scripts/analyze_round5_no_candidate_children.py \
    --glob_pattern 'results/BRIGHT/*/round5/**/all_eval_sample_dicts.pkl' \
    --require_substrings round5_mrr_selector_accum_meanscore_global-FT=1000 \
    --require_substrings RPN=agent_executor_v1_icl2-RM=concat-RE=1/RCT=10-RCS=mixed-RGT=10-RSM=meanscore_global-RRrfK=60/RRC=leaf-REM=replace
"""

import argparse
import glob
import os
import pickle as pkl
import statistics as st
from collections import Counter
from typing import Any, Dict, List, Sequence

import pandas as pd


DEFAULT_GLOB_PATTERN = "results/BRIGHT/*/round5/**/all_eval_sample_dicts.pkl"
DEFAULT_REQUIRE_SUBSTRINGS = [
    "round5_mrr_selector_accum_meanscore_global-FT=1000",
    "RPN=agent_executor_v1_icl2-RM=concat-RE=1/RCT=10-RCS=mixed-RGT=10-RSM=meanscore_global-RRrfK=60/RRC=leaf-REM=replace",
]
DEFAULT_EXCLUDE_SUBSTRINGS = [
    "fused_memory",
]


def _resolve_eval_paths(
    inputs: Sequence[str],
    glob_pattern: str,
    require_substrings: Sequence[str],
    exclude_substrings: Sequence[str],
) -> List[str]:
    resolved: List[str] = []

    for path in inputs:
        if not path:
            continue
        abs_path = os.path.abspath(path)
        if os.path.isdir(abs_path):
            candidate = os.path.join(abs_path, "all_eval_sample_dicts.pkl")
            if not os.path.exists(candidate):
                raise FileNotFoundError(f"Directory does not contain all_eval_sample_dicts.pkl: {abs_path}")
            resolved.append(candidate)
            continue
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Input path not found: {abs_path}")
        resolved.append(abs_path)

    if glob_pattern:
        resolved.extend(glob.glob(glob_pattern, recursive=True))

    deduped = sorted({os.path.abspath(path) for path in resolved})
    filtered: List[str] = []
    for path in deduped:
        if require_substrings and any(token not in path for token in require_substrings):
            continue
        if exclude_substrings and any(token in path for token in exclude_substrings):
            continue
        filtered.append(path)

    if not filtered:
        raise FileNotFoundError("No all_eval_sample_dicts.pkl files matched the given filters.")
    return filtered


def _infer_subset_from_path(path: str) -> str:
    parts = os.path.abspath(path).split(os.sep)
    for idx, part in enumerate(parts):
        if part == "results" and (idx + 2) < len(parts):
            return str(parts[idx + 2])
    return "unknown"


def _safe_iter_records(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    records = sample.get("iter_records", []) or []
    return [record for record in records if isinstance(record, dict)]


def _format_pct(numer: int, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return 100.0 * float(numer) / float(denom)


def _pctl_index(values: Sequence[int], frac: float) -> int:
    if not values:
        return 0
    ordered = sorted(int(v) for v in values)
    idx = min(len(ordered) - 1, max(0, int(len(ordered) * float(frac))))
    return int(ordered[idx])


def _analyze_paths(paths: Sequence[str]) -> Dict[str, Any]:
    overall_total = 0
    overall_ncc = 0
    total_queries = 0
    queries_with_ncc = 0
    ncc_after_nonempty = 0
    ncc_after_empty = 0

    per_iter_total = Counter()
    per_iter_ncc = Counter()
    per_subset_total = Counter()
    per_subset_ncc = Counter()
    per_subset_query_with_ncc = Counter()
    first_iter_dist = Counter()
    reason_counts = Counter()

    ncc_candidate_branch_count: List[int] = []
    ncc_before_branch_count: List[int] = []
    ncc_before_depths: List[int] = []

    matched_paths_meta: List[Dict[str, str]] = []

    for path in paths:
        subset = _infer_subset_from_path(path)
        matched_paths_meta.append({"subset": subset, "path": path})
        with open(path, "rb") as f:
            samples = pkl.load(f)

        for sample in samples:
            total_queries += 1
            had_ncc = False
            first_ncc_iter = None

            for rec in _safe_iter_records(sample):
                iter_idx = int(rec.get("iter", -1))
                reason = str(rec.get("selector_pick_reason", "") or "")
                reason_counts[reason] += 1
                overall_total += 1
                per_iter_total[iter_idx] += 1
                per_subset_total[subset] += 1

                if reason != "no_candidate_children":
                    continue

                had_ncc = True
                if first_ncc_iter is None:
                    first_ncc_iter = iter_idx

                overall_ncc += 1
                per_iter_ncc[iter_idx] += 1
                per_subset_ncc[subset] += 1

                after = rec.get("selected_branches_after", []) or []
                before = rec.get("selected_branches_before", []) or []
                if after:
                    ncc_after_nonempty += 1
                else:
                    ncc_after_empty += 1

                ncc_candidate_branch_count.append(int(rec.get("candidate_branch_count", 0) or 0))
                ncc_before_branch_count.append(len(before))
                ncc_before_depths.extend(len(path_row) for path_row in before if path_row)

            if had_ncc:
                queries_with_ncc += 1
                per_subset_query_with_ncc[subset] += 1
                if first_ncc_iter is not None:
                    first_iter_dist[first_ncc_iter] += 1

    overall_row = {
        "matched_paths": int(len(paths)),
        "total_queries": int(total_queries),
        "queries_with_ncc": int(queries_with_ncc),
        "queries_with_ncc_pct": _format_pct(queries_with_ncc, total_queries),
        "total_sample_iters": int(overall_total),
        "no_candidate_children": int(overall_ncc),
        "no_candidate_children_pct": _format_pct(overall_ncc, overall_total),
        "ncc_after_nonempty": int(ncc_after_nonempty),
        "ncc_after_empty": int(ncc_after_empty),
        # Intent: summarize whether selector termination happens near deep/leaf-adjacent beams.
        "ncc_candidate_branch_count_mean": round(st.mean(ncc_candidate_branch_count), 4) if ncc_candidate_branch_count else 0.0,
        "ncc_candidate_branch_count_min": min(ncc_candidate_branch_count) if ncc_candidate_branch_count else 0,
        "ncc_candidate_branch_count_median": int(st.median(ncc_candidate_branch_count)) if ncc_candidate_branch_count else 0,
        "ncc_candidate_branch_count_p90": _pctl_index(ncc_candidate_branch_count, 0.9) if ncc_candidate_branch_count else 0,
        "ncc_candidate_branch_count_max": max(ncc_candidate_branch_count) if ncc_candidate_branch_count else 0,
        "ncc_before_branch_count_mean": round(st.mean(ncc_before_branch_count), 4) if ncc_before_branch_count else 0.0,
        "ncc_before_depth_mean": round(st.mean(ncc_before_depths), 4) if ncc_before_depths else 0.0,
    }

    per_iter_rows: List[Dict[str, Any]] = []
    for iter_idx in sorted(per_iter_total.keys()):
        total = int(per_iter_total[iter_idx])
        ncc = int(per_iter_ncc[iter_idx])
        per_iter_rows.append(
            {
                "iter": int(iter_idx),
                "no_candidate_children": ncc,
                "total": total,
                "pct": _format_pct(ncc, total),
            }
        )

    first_iter_rows: List[Dict[str, Any]] = []
    for iter_idx in sorted(first_iter_dist.keys()):
        first_iter_rows.append(
            {
                "first_iter": int(iter_idx),
                "queries": int(first_iter_dist[iter_idx]),
            }
        )

    per_subset_rows: List[Dict[str, Any]] = []
    for subset in sorted(per_subset_total.keys()):
        total = int(per_subset_total[subset])
        ncc = int(per_subset_ncc[subset])
        per_subset_rows.append(
            {
                "subset": subset,
                "no_candidate_children": ncc,
                "total": total,
                "pct": _format_pct(ncc, total),
                "queries_with_ncc": int(per_subset_query_with_ncc[subset]),
            }
        )

    reason_rows: List[Dict[str, Any]] = []
    for reason, count in reason_counts.most_common():
        reason_rows.append(
            {
                "selector_pick_reason": str(reason),
                "count": int(count),
            }
        )

    return {
        "matched_paths_df": pd.DataFrame(matched_paths_meta),
        "overall_df": pd.DataFrame([overall_row]),
        "per_iter_df": pd.DataFrame(per_iter_rows),
        "first_iter_df": pd.DataFrame(first_iter_rows),
        "per_subset_df": pd.DataFrame(per_subset_rows),
        "reason_df": pd.DataFrame(reason_rows),
    }


def _print_df(title: str, df: pd.DataFrame) -> None:
    print(f"\n[{title}]")
    if df.empty:
        print("No rows.")
        return
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def _save_df(df: pd.DataFrame, path: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze how often round5 selector reaches the no_candidate_children state."
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=[],
        help="Optional list of run directories or all_eval_sample_dicts.pkl paths.",
    )
    parser.add_argument(
        "--glob_pattern",
        type=str,
        default=DEFAULT_GLOB_PATTERN,
        help="Glob used to discover all_eval_sample_dicts.pkl files.",
    )
    parser.add_argument(
        "--require_substrings",
        nargs="*",
        default=DEFAULT_REQUIRE_SUBSTRINGS,
        help="Only keep paths that contain all of these substrings.",
    )
    parser.add_argument(
        "--exclude_substrings",
        nargs="*",
        default=DEFAULT_EXCLUDE_SUBSTRINGS,
        help="Drop any path that contains one of these substrings.",
    )
    parser.add_argument(
        "--print_paths",
        action="store_true",
        help="Print all matched evaluation paths before summaries.",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default=None,
        help="Optional prefix for CSV outputs. Example: results/BRIGHT/round5_no_candidate_children",
    )
    args = parser.parse_args()

    resolved_paths = _resolve_eval_paths(
        inputs=args.inputs,
        glob_pattern=args.glob_pattern,
        require_substrings=args.require_substrings,
        exclude_substrings=args.exclude_substrings,
    )
    analysis = _analyze_paths(resolved_paths)

    if args.print_paths:
        _print_df("Matched Paths", analysis["matched_paths_df"])

    _print_df("Overall", analysis["overall_df"])
    _print_df("Per Iter", analysis["per_iter_df"])
    _print_df("First NCC Iter", analysis["first_iter_df"])
    _print_df("Per Subset", analysis["per_subset_df"])
    _print_df("Selector Reasons", analysis["reason_df"])

    if args.out_prefix:
        _save_df(analysis["matched_paths_df"], f"{args.out_prefix}_matched_paths.csv")
        _save_df(analysis["overall_df"], f"{args.out_prefix}_overall.csv")
        _save_df(analysis["per_iter_df"], f"{args.out_prefix}_per_iter.csv")
        _save_df(analysis["first_iter_df"], f"{args.out_prefix}_first_iter.csv")
        _save_df(analysis["per_subset_df"], f"{args.out_prefix}_per_subset.csv")
        _save_df(analysis["reason_df"], f"{args.out_prefix}_selector_reasons.csv")
        print(f"\nSaved CSVs with prefix: {args.out_prefix}")


if __name__ == "__main__":
    main()
