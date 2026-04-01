#!/usr/bin/env python3
"""Analyze whether ended-beam reseat adds exact gold docs into the next pool.

Purpose
-------
For the frontiercum_qstate round6 run, measure at each iteration:

1. performance (`nDCG@10`)
2. reseat depth
3. whether a reseat transition adds exact gold docs into the retrieval region

The key transition metric is computed against the actual frontiercum_qstate
pool semantics:

    pool_t = descendants(selected_branches_before_t) U cumulative_reached_leaves_t

and the post-transition region that will be used on the next iteration:

    pool_after_t = descendants(selected_branches_after_t) U cumulative_reached_leaves_after_t

where `cumulative_reached_leaves_after_t` includes `new_leaf_paths` from iter t.
"""

import argparse
import glob
import os
import pickle as pkl
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd


DEFAULT_GLOB = "results/BRIGHT/*/round6*/**/all_eval_sample_dicts.pkl"
DEFAULT_REQUIRE = [
    "MaxBS=10-S=round6_mrr_selector_accum_meanscore_global_expandable_ended_reseat_frontiercum_qstate",
    "embed-qwen3-8b-0928",
    "agent_executor_v1_icl2",
    "RSM=meanscore_global",
    "REM=ended_reseat",
    "RB=frontiercum_qstate_v1",
]
DEFAULT_EXCLUDE = [
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
        raise FileNotFoundError("No files matched the requested round6 frontiercum_qstate filters.")
    return resolved


def _load_samples_by_subset(path_map: Dict[str, str]) -> Dict[str, List[Dict[str, Any]]]:
    loaded: Dict[str, List[Dict[str, Any]]] = {}
    for subset, path in path_map.items():
        with open(path, "rb") as f:
            loaded[subset] = pkl.load(f)
    return loaded


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _is_prefix(prefix: Sequence[int], full: Sequence[int]) -> bool:
    prefix_t = tuple(prefix)
    full_t = tuple(full)
    return len(prefix_t) <= len(full_t) and full_t[: len(prefix_t)] == prefix_t


def _normalize_paths(items: Sequence[Any]) -> List[Tuple[int, ...]]:
    out: List[Tuple[int, ...]] = []
    for item in list(items or []):
        if not item:
            continue
        if isinstance(item, dict):
            path = item.get("path", [])
        else:
            path = item
        if not path:
            continue
        out.append(tuple(int(x) for x in list(path)))
    return out


def _count_pool_gold_hits(
    selected_branches: Sequence[Tuple[int, ...]],
    cumulative_reached_leaves: Set[Tuple[int, ...]],
    gold_paths: Sequence[Tuple[int, ...]],
) -> int:
    hits = 0
    for gold_path in gold_paths:
        if gold_path in cumulative_reached_leaves:
            hits += 1
            continue
        if any(_is_prefix(branch_path, gold_path) for branch_path in selected_branches):
            hits += 1
    return hits


def _iter_rows_for_sample(subset: str, query_idx: int, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    gold_paths = [tuple(path) for path in (sample.get("gold_paths", []) or []) if path]
    gold_path_set = set(gold_paths)
    iter_records = sample.get("iter_records", []) or []
    cumulative_reached_before: Set[Tuple[int, ...]] = set()
    rows: List[Dict[str, Any]] = []

    for iter_idx, rec in enumerate(iter_records):
        selected_before = _normalize_paths(rec.get("selected_branches_before", []) or [])
        selected_after = _normalize_paths(rec.get("selected_branches_after", []) or [])
        new_leaf_paths = _normalize_paths(rec.get("new_leaf_paths", []) or [])
        reseat_selected = _normalize_paths(rec.get("ended_beam_reseat_selected_paths", []) or [])

        pool_gold_hits_before = _count_pool_gold_hits(
            selected_branches=selected_before,
            cumulative_reached_leaves=cumulative_reached_before,
            gold_paths=gold_paths,
        )
        cumulative_reached_after = set(cumulative_reached_before)
        cumulative_reached_after.update(new_leaf_paths)
        # Intent: measure the region that becomes reachable after the reseat transition and is used on the next iter.
        pool_gold_hits_after_transition = _count_pool_gold_hits(
            selected_branches=selected_after,
            cumulative_reached_leaves=cumulative_reached_after,
            gold_paths=gold_paths,
        )
        new_gold_leaf_hits = int(len(gold_path_set.intersection(new_leaf_paths)))
        reseat_depths = [len(path) for path in reseat_selected]
        reseat_active = int(bool(reseat_selected))
        gain = int(pool_gold_hits_after_transition - pool_gold_hits_before)

        rows.append(
            {
                "subset": subset,
                "query_idx": int(query_idx),
                "iter": int(iter_idx),
                "ndcg10": _safe_float(rec.get("metrics", {}).get("nDCG@10")),
                "reseat_rate": int(reseat_active),
                "reseat_depth_mean": float(np.mean(reseat_depths)) if reseat_depths else float("nan"),
                "reseat_depth_max": float(max(reseat_depths)) if reseat_depths else float("nan"),
                "num_gold_docs": int(len(gold_paths)),
                "pool_gold_hits_before": int(pool_gold_hits_before),
                "pool_gold_hits_after_transition": int(pool_gold_hits_after_transition),
                "pool_gold_recall_before": float(pool_gold_hits_before / len(gold_paths)) if gold_paths else float("nan"),
                "pool_gold_recall_after_transition": float(pool_gold_hits_after_transition / len(gold_paths)) if gold_paths else float("nan"),
                "pool_gold_gain": float(gain) if reseat_active else float("nan"),
                "pool_gold_gain_any": float(gain > 0) if reseat_active else float("nan"),
                "new_gold_leaf_hits": float(new_gold_leaf_hits) if reseat_active else float("nan"),
                "new_gold_leaf_any": float(new_gold_leaf_hits > 0) if reseat_active else float("nan"),
            }
        )
        cumulative_reached_before = cumulative_reached_after
    return rows


def _build_query_iter_df(samples_by_subset: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for subset, samples in samples_by_subset.items():
        for query_idx, sample in enumerate(samples):
            rows.extend(_iter_rows_for_sample(subset, query_idx, sample))
    return pd.DataFrame(rows)


def _aggregate_subset_iter(query_iter_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "ndcg10",
        "reseat_rate",
        "reseat_depth_mean",
        "reseat_depth_max",
        "pool_gold_hits_before",
        "pool_gold_hits_after_transition",
        "pool_gold_recall_before",
        "pool_gold_recall_after_transition",
        "pool_gold_gain",
        "pool_gold_gain_any",
        "new_gold_leaf_hits",
        "new_gold_leaf_any",
    ]
    subset_iter = (
        query_iter_df.groupby(["subset", "iter"], as_index=False)[metric_cols]
        .mean(numeric_only=True)
        .sort_values(["subset", "iter"])
    )
    counts = (
        query_iter_df.groupby(["subset", "iter"], as_index=False)
        .agg(num_queries=("query_idx", "nunique"), num_reseat_queries=("reseat_rate", "sum"))
    )
    return subset_iter.merge(counts, on=["subset", "iter"], how="left")


def _aggregate_overall(subset_iter_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [col for col in subset_iter_df.columns if col not in {"subset", "iter", "num_queries", "num_reseat_queries"}]
    overall = subset_iter_df.groupby("iter", as_index=False)[metric_cols].mean(numeric_only=True)
    subset_counts = subset_iter_df.groupby("iter", as_index=False)["subset"].nunique().rename(columns={"subset": "num_subsets"})
    return overall.merge(subset_counts, on="iter", how="left").sort_values("iter").reset_index(drop=True)


def _build_reseat_rows(query_iter_df: pd.DataFrame) -> pd.DataFrame:
    reseat_df = query_iter_df[query_iter_df["reseat_rate"] > 0].copy()
    return reseat_df.sort_values(["subset", "iter", "query_idx"]).reset_index(drop=True)


def _build_subset_curve_wide(subset_iter_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    pivot = subset_iter_df.pivot(index="subset", columns="iter", values=value_col).sort_index()
    pivot.columns = [f"iter_{int(col)}" for col in pivot.columns]
    return pivot.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze reseat depth and gold-doc gain for frontiercum_qstate round6.")
    parser.add_argument("--glob_pattern", default=DEFAULT_GLOB)
    parser.add_argument("--out_prefix", default="results/BRIGHT/analysis/round6_frontiercum_qstate_reseat_gold_gain")
    parser.add_argument("--require_substrings", nargs="*", default=DEFAULT_REQUIRE)
    parser.add_argument("--exclude_substrings", nargs="*", default=DEFAULT_EXCLUDE)
    args = parser.parse_args()

    path_map = _resolve_paths(
        glob_pattern=str(args.glob_pattern),
        require_substrings=list(args.require_substrings),
        exclude_substrings=list(args.exclude_substrings),
    )
    samples_by_subset = _load_samples_by_subset(path_map)
    query_iter_df = _build_query_iter_df(samples_by_subset)
    subset_iter_df = _aggregate_subset_iter(query_iter_df)
    overall_df = _aggregate_overall(subset_iter_df)
    reseat_rows_df = _build_reseat_rows(query_iter_df)
    subset_ndcg_curve_df = _build_subset_curve_wide(subset_iter_df, "ndcg10")
    subset_gain_curve_df = _build_subset_curve_wide(subset_iter_df, "pool_gold_gain_any")

    out_dir = os.path.dirname(os.path.abspath(args.out_prefix))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    query_iter_df.to_csv(f"{args.out_prefix}_query_iter.csv", index=False)
    subset_iter_df.to_csv(f"{args.out_prefix}_subset_iter.csv", index=False)
    overall_df.to_csv(f"{args.out_prefix}_overall_iter.csv", index=False)
    reseat_rows_df.to_csv(f"{args.out_prefix}_reseat_rows.csv", index=False)
    subset_ndcg_curve_df.to_csv(f"{args.out_prefix}_subset_ndcg_curve.csv", index=False)
    subset_gain_curve_df.to_csv(f"{args.out_prefix}_subset_gold_gain_curve.csv", index=False)

    print(f"Saved query-iter rows to {args.out_prefix}_query_iter.csv")
    print(f"Saved subset-iter summary to {args.out_prefix}_subset_iter.csv")
    print(f"Saved overall iter summary to {args.out_prefix}_overall_iter.csv")
    print(f"Saved reseat-only rows to {args.out_prefix}_reseat_rows.csv")
    print(f"Saved subset nDCG curve to {args.out_prefix}_subset_ndcg_curve.csv")
    print(f"Saved subset gold-gain curve to {args.out_prefix}_subset_gold_gain_curve.csv")


if __name__ == "__main__":
    main()
