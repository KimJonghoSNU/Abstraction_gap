#!/usr/bin/env python3
"""Analyze dynamics of the round6 direct-child ended-reseat run.

Purpose
-------
Explain three things for the current "ours" run:

1. Why the overall nDCG recovers after the iter-4 dip.
2. How deep the reseated branches are, especially after iter 3.
3. How much each dataset drops or recovers across iterations.

This script stays on branch-state/controller signals:
- gold-region branch hit
- re-entry into gold region
- ended-beam reseat rate
- selected / reseated branch depth

Examples
--------
python scripts/analysis/analyze_round6_ours_dynamics.py

python scripts/analysis/analyze_round6_ours_dynamics.py \
    --out_prefix results/BRIGHT/analysis/round6_ours_dynamics
"""

import argparse
import glob
import os
import pickle as pkl
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_GLOB = "results/BRIGHT/*/round6*/**/all_eval_sample_dicts.pkl"
DEFAULT_REQUIRE = [
    "MaxBS=10-S=round6_mrr_selector_accum_meanscore_global_expandable_ended_reseat",
    "embed-qwen3-8b-0928",
    "agent_executor_v1_icl2",
    "RSM=meanscore_global",
    "REM=ended_reseat",
]
DEFAULT_EXCLUDE = [
    "descendant_flat",
    "MaxBS=1-",
    "_emr",
]


def _infer_subset_from_path(path: str) -> str:
    parts = os.path.abspath(path).split(os.sep)
    for idx, part in enumerate(parts):
        if part == "results" and (idx + 2) < len(parts):
            return str(parts[idx + 2])
    return "unknown"


def _resolve_paths(
    glob_pattern: str,
    require_substrings: Sequence[str],
    exclude_substrings: Sequence[str],
) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    for path in sorted(glob.glob(glob_pattern, recursive=True)):
        abs_path = os.path.abspath(path)
        if require_substrings and any(token not in abs_path for token in require_substrings):
            continue
        if exclude_substrings and any(token in abs_path for token in exclude_substrings):
            continue
        subset = _infer_subset_from_path(abs_path)
        if subset in resolved:
            raise ValueError(f"Multiple matching files for subset={subset}: {resolved[subset]} and {abs_path}")
        resolved[subset] = abs_path
    if not resolved:
        raise FileNotFoundError("No files matched the requested round6 ours filters.")
    return resolved


def _load_samples_by_subset(path_map: Dict[str, str]) -> Dict[str, List[Dict[str, Any]]]:
    loaded: Dict[str, List[Dict[str, Any]]] = {}
    for subset, path in path_map.items():
        with open(path, "rb") as f:
            loaded[subset] = pkl.load(f)
    return loaded


def _is_prefix(prefix: Sequence[int], full: Sequence[int]) -> bool:
    prefix_t = tuple(prefix)
    full_t = tuple(full)
    return len(prefix_t) <= len(full_t) and full_t[: len(prefix_t)] == prefix_t


def _branch_is_gold_ancestor(branch_path: Sequence[int], gold_paths: Sequence[Sequence[int]]) -> bool:
    return any(_is_prefix(branch_path, gold_path) for gold_path in gold_paths)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _normalize_path_rows(items: Sequence[Any]) -> List[Tuple[int, ...]]:
    out: List[Tuple[int, ...]] = []
    for item in list(items or []):
        if isinstance(item, dict):
            path = item.get("path", [])
        else:
            path = item
        if not path:
            continue
        out.append(tuple(int(x) for x in list(path)))
    return out


def _iter_rows_for_sample(subset: str, query_idx: int, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    gold_paths = [tuple(path) for path in (sample.get("gold_paths", []) or []) if path]
    iter_records = sample.get("iter_records", []) or []
    prev_on_region = None
    rows: List[Dict[str, Any]] = []

    for iter_idx, rec in enumerate(iter_records):
        selected = [tuple(path) for path in (rec.get("selected_branches_after", []) or []) if path]
        reseated = _normalize_path_rows(rec.get("ended_beam_reseat_selected_paths", []) or [])
        gold_flags = [_branch_is_gold_ancestor(path, gold_paths) for path in selected]
        on_region = bool(any(gold_flags))

        selected_depths = [len(path) for path in selected]
        gold_selected_depths = [len(path) for path, is_gold in zip(selected, gold_flags) if is_gold]
        reseat_depths = [len(path) for path in reseated]

        rows.append(
            {
                "subset": subset,
                "query_idx": int(query_idx),
                "iter": int(iter_idx),
                "ndcg10": _safe_float(rec.get("metrics", {}).get("nDCG@10")),
                "branch_hit": int(on_region),
                "reenter": int(prev_on_region is False and on_region is True),
                "leave_region": int(prev_on_region is True and on_region is False),
                "ended_beam_count": int(rec.get("ended_beam_count", 0) or 0),
                "reseat_rate": int(bool(reseated)),
                "selected_depth_mean": float(np.mean(selected_depths)) if selected_depths else float("nan"),
                "selected_depth_max": float(max(selected_depths)) if selected_depths else float("nan"),
                "gold_selected_depth_mean": float(np.mean(gold_selected_depths)) if gold_selected_depths else float("nan"),
                "gold_selected_depth_max": float(max(gold_selected_depths)) if gold_selected_depths else float("nan"),
                "reseat_depth_mean": float(np.mean(reseat_depths)) if reseat_depths else float("nan"),
                "reseat_depth_max": float(max(reseat_depths)) if reseat_depths else float("nan"),
            }
        )
        prev_on_region = on_region
    return rows


def _build_query_iter_rows(samples_by_subset: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for subset, samples in samples_by_subset.items():
        for query_idx, sample in enumerate(samples):
            rows.extend(_iter_rows_for_sample(subset, query_idx, sample))
    return pd.DataFrame(rows)


def _aggregate_subset_iter(query_iter_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "ndcg10",
        "branch_hit",
        "reenter",
        "leave_region",
        "ended_beam_count",
        "reseat_rate",
        "selected_depth_mean",
        "selected_depth_max",
        "gold_selected_depth_mean",
        "gold_selected_depth_max",
        "reseat_depth_mean",
        "reseat_depth_max",
    ]
    subset_iter = (
        query_iter_df.groupby(["subset", "iter"], as_index=False)[metric_cols]
        .mean(numeric_only=True)
        .sort_values(["subset", "iter"])
    )
    counts = (
        query_iter_df.groupby(["subset", "iter"], as_index=False)["query_idx"]
        .nunique()
        .rename(columns={"query_idx": "num_queries"})
    )
    return subset_iter.merge(counts, on=["subset", "iter"], how="left")


def _aggregate_overall(subset_iter_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [col for col in subset_iter_df.columns if col not in {"subset", "iter", "num_queries"}]
    overall = subset_iter_df.groupby("iter", as_index=False)[metric_cols].mean(numeric_only=True)
    subset_counts = subset_iter_df.groupby("iter", as_index=False)["subset"].nunique().rename(columns={"subset": "num_subsets"})
    return overall.merge(subset_counts, on="iter", how="left").sort_values("iter").reset_index(drop=True)


def _build_recovery_summary(overall_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for idx in range(1, len(overall_df)):
        prev_row = overall_df.iloc[idx - 1]
        cur_row = overall_df.iloc[idx]
        rows.append(
            {
                "iter_from": int(prev_row["iter"]),
                "iter_to": int(cur_row["iter"]),
                "delta_ndcg10": float(cur_row["ndcg10"] - prev_row["ndcg10"]),
                "delta_branch_hit": float(cur_row["branch_hit"] - prev_row["branch_hit"]),
                "delta_reenter": float(cur_row["reenter"] - prev_row["reenter"]),
                "delta_leave_region": float(cur_row["leave_region"] - prev_row["leave_region"]),
                "delta_ended_beam_count": float(cur_row["ended_beam_count"] - prev_row["ended_beam_count"]),
                "delta_reseat_rate": float(cur_row["reseat_rate"] - prev_row["reseat_rate"]),
                "delta_reseat_depth_mean": float(cur_row["reseat_depth_mean"] - prev_row["reseat_depth_mean"]),
                "delta_selected_depth_mean": float(cur_row["selected_depth_mean"] - prev_row["selected_depth_mean"]),
            }
        )
    return pd.DataFrame(rows)


def _build_subset_curve_wide(subset_iter_df: pd.DataFrame) -> pd.DataFrame:
    pivot = subset_iter_df.pivot(index="subset", columns="iter", values="ndcg10").sort_index()
    pivot.columns = [f"iter_{int(col)}" for col in pivot.columns]
    return pivot.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze dynamics of the round6 ended-reseat ours run.")
    parser.add_argument("--glob_pattern", default=DEFAULT_GLOB)
    parser.add_argument("--out_prefix", default="results/BRIGHT/analysis/round6_ours_dynamics")
    parser.add_argument("--require_substrings", nargs="*", default=DEFAULT_REQUIRE)
    parser.add_argument("--exclude_substrings", nargs="*", default=DEFAULT_EXCLUDE)
    args = parser.parse_args()

    path_map = _resolve_paths(
        glob_pattern=str(args.glob_pattern),
        require_substrings=list(args.require_substrings),
        exclude_substrings=list(args.exclude_substrings),
    )
    samples_by_subset = _load_samples_by_subset(path_map)
    query_iter_df = _build_query_iter_rows(samples_by_subset)
    subset_iter_df = _aggregate_subset_iter(query_iter_df)
    overall_df = _aggregate_overall(subset_iter_df)
    recovery_df = _build_recovery_summary(overall_df)
    subset_curve_df = _build_subset_curve_wide(subset_iter_df)

    out_dir = os.path.dirname(os.path.abspath(args.out_prefix))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    query_iter_df.to_csv(f"{args.out_prefix}_query_iter.csv", index=False)
    subset_iter_df.to_csv(f"{args.out_prefix}_subset_iter.csv", index=False)
    overall_df.to_csv(f"{args.out_prefix}_overall_iter.csv", index=False)
    recovery_df.to_csv(f"{args.out_prefix}_recovery.csv", index=False)
    subset_curve_df.to_csv(f"{args.out_prefix}_subset_ndcg_curve.csv", index=False)

    print(f"Saved query-iter rows to {args.out_prefix}_query_iter.csv")
    print(f"Saved subset-iter summary to {args.out_prefix}_subset_iter.csv")
    print(f"Saved overall iter summary to {args.out_prefix}_overall_iter.csv")
    print(f"Saved recovery summary to {args.out_prefix}_recovery.csv")
    print(f"Saved subset ndcg curves to {args.out_prefix}_subset_ndcg_curve.csv")


if __name__ == "__main__":
    main()
