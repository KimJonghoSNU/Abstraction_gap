#!/usr/bin/env python3
"""Analyze when the round6 ended-reseat run leaves the gold branch region.

Purpose
-------
Track the branch-state trajectory of the direct-child ended-reseat run:

- when does a query first enter a gold-ancestor branch region?
- after entering, when does it first leave that region?
- at what branch depth does that exit happen?

This is primarily branch-state analysis, but we also reconstruct one strict
retrieval-side signal from saved iter records: whether the current retrieval
pool still contains any exact gold doc leaf.
The target question is whether the controller leaves the correct region, when
that happens in iteration/depth terms, and whether retrieval recoverability
persists after branch-state exit.

Examples
--------
python scripts/analysis/analyze_round6_gold_region_exit.py

python scripts/analysis/analyze_round6_gold_region_exit.py \
    --out_prefix results/BRIGHT/analysis/round6_gold_region_exit
"""

import argparse
import glob
import json
import math
import os
import pickle as pkl
from collections import Counter
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
        raise FileNotFoundError("No files matched the requested round6 run filters.")
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


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _normalize_paths(items: Sequence[Any]) -> List[Tuple[int, ...]]:
    out: List[Tuple[int, ...]] = []
    for item in list(items or []):
        if not item:
            continue
        out.append(tuple(int(x) for x in list(item)))
    return out


def _count_pool_gold_hits(
    selected_branches: Sequence[Tuple[int, ...]],
    cumulative_reached_leaves: Sequence[Tuple[int, ...]],
    gold_paths: Sequence[Tuple[int, ...]],
) -> int:
    cumulative_set = {tuple(path) for path in list(cumulative_reached_leaves or [])}
    selected_list = [tuple(path) for path in list(selected_branches or [])]
    if (not cumulative_set) and (not selected_list):
        return int(len(gold_paths))

    hits = 0
    for gold_path in gold_paths:
        gp_t = tuple(gold_path)
        if gp_t in cumulative_set:
            hits += 1
            continue
        if any(_is_prefix(branch_path, gp_t) for branch_path in selected_list):
            hits += 1
    return hits


def _analyze_sample(subset: str, query_idx: int, sample: Dict[str, Any]) -> Dict[str, Any]:
    gold_paths = [tuple(path) for path in (sample.get("gold_paths", []) or []) if path]
    gold_path_set = set(gold_paths)
    iter_records = sample.get("iter_records", []) or []
    per_iter_rows: List[Dict[str, Any]] = []
    on_region_flags: List[bool] = []
    per_iter_gold_depths: List[List[int]] = []
    cumulative_reached_leaf_paths: set[Tuple[int, ...]] = set()

    for iter_idx, rec in enumerate(iter_records):
        selected = [tuple(path) for path in (rec.get("selected_branches_after", []) or []) if path]
        selected_before = [tuple(path) for path in (rec.get("selected_branches_before", []) or []) if path]
        gold_selected = [path for path in selected if _branch_is_gold_ancestor(path, gold_paths)]
        gold_depths = [len(path) for path in gold_selected]
        on_region = bool(gold_selected)
        # Intent: frontiercum_qstate retrieval pool is frontier descendants union cumulative reached leaves before the current iter.
        pool_gold_hits = _count_pool_gold_hits(
            selected_branches=selected_before,
            cumulative_reached_leaves=sorted(cumulative_reached_leaf_paths),
            gold_paths=gold_paths,
        )
        pool_has_any_gold_doc = int(pool_gold_hits > 0)
        pool_gold_doc_recall = (
            float(pool_gold_hits / len(gold_path_set))
            if gold_path_set
            else float("nan")
        )
        on_region_flags.append(on_region)
        per_iter_gold_depths.append(gold_depths)
        per_iter_rows.append(
            {
                "subset": subset,
                "query_idx": int(query_idx),
                "iter": int(iter_idx),
                "on_region": int(on_region),
                "num_selected": int(len(selected)),
                "num_gold_selected": int(len(gold_selected)),
                "gold_selected_depth_mean": float(np.mean(gold_depths)) if gold_depths else float("nan"),
                "gold_selected_depth_max": float(max(gold_depths)) if gold_depths else float("nan"),
                "pool_has_any_gold_doc": int(pool_has_any_gold_doc),
                "pool_gold_doc_hits": int(pool_gold_hits),
                "pool_gold_doc_recall": float(pool_gold_doc_recall),
                "ndcg10": _safe_float(rec.get("metrics", {}).get("nDCG@10")),
            }
        )
        cumulative_reached_leaf_paths.update(_normalize_paths(rec.get("new_leaf_paths", []) or []))

    first_enter_iter = next((idx for idx, flag in enumerate(on_region_flags) if flag), None)
    first_exit_iter = None
    contiguous_span = 0
    last_on_region_iter = None
    exit_prev_depth_mean = float("nan")
    exit_prev_depth_max = float("nan")
    enter_depth_mean = float("nan")
    enter_depth_max = float("nan")

    if first_enter_iter is not None:
        enter_depths = per_iter_gold_depths[first_enter_iter]
        enter_depth_mean = float(np.mean(enter_depths)) if enter_depths else float("nan")
        enter_depth_max = float(max(enter_depths)) if enter_depths else float("nan")

        idx = first_enter_iter
        while idx < len(on_region_flags) and on_region_flags[idx]:
            contiguous_span += 1
            last_on_region_iter = idx
            idx += 1
        if idx < len(on_region_flags):
            first_exit_iter = idx
            prev_depths = per_iter_gold_depths[idx - 1]
            exit_prev_depth_mean = float(np.mean(prev_depths)) if prev_depths else float("nan")
            exit_prev_depth_max = float(max(prev_depths)) if prev_depths else float("nan")
        else:
            prev_depths = per_iter_gold_depths[last_on_region_iter] if last_on_region_iter is not None else []
            exit_prev_depth_mean = float(np.mean(prev_depths)) if prev_depths else float("nan")
            exit_prev_depth_max = float(max(prev_depths)) if prev_depths else float("nan")

    survive_to_end = int(first_enter_iter is not None and first_exit_iter is None)

    return {
        "subset": subset,
        "query_idx": int(query_idx),
        "query": str(sample.get("original_query", "") or ""),
        "gold_paths": _json_dumps([list(path) for path in gold_paths]),
        "num_iters": int(len(iter_records)),
        "first_enter_iter": float(first_enter_iter) if first_enter_iter is not None else float("nan"),
        "first_exit_iter": float(first_exit_iter) if first_exit_iter is not None else float("nan"),
        "survive_to_end": int(survive_to_end),
        "contiguous_on_region_span": float(contiguous_span) if first_enter_iter is not None else float("nan"),
        "last_on_region_iter": float(last_on_region_iter) if last_on_region_iter is not None else float("nan"),
        "enter_depth_mean": enter_depth_mean,
        "enter_depth_max": enter_depth_max,
        "exit_prev_depth_mean": exit_prev_depth_mean,
        "exit_prev_depth_max": exit_prev_depth_max,
        "on_region_flags": _json_dumps([int(flag) for flag in on_region_flags]),
        "iter_rows": per_iter_rows,
    }


def _build_rows(samples_by_subset: Dict[str, List[Dict[str, Any]]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: List[Dict[str, Any]] = []
    iter_rows: List[Dict[str, Any]] = []
    for subset, samples in samples_by_subset.items():
        for query_idx, sample in enumerate(samples):
            row = _analyze_sample(subset, query_idx, sample)
            iter_rows.extend(row.pop("iter_rows"))
            summary_rows.append(row)
    return pd.DataFrame(summary_rows), pd.DataFrame(iter_rows)


def _aggregate_pool_subset_iter(iter_rows_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "pool_has_any_gold_doc",
        "pool_gold_doc_hits",
        "pool_gold_doc_recall",
        "ndcg10",
    ]
    subset_iter = (
        iter_rows_df.groupby(["subset", "iter"], as_index=False)[metric_cols]
        .mean(numeric_only=True)
        .sort_values(["subset", "iter"])
    )
    counts = (
        iter_rows_df.groupby(["subset", "iter"], as_index=False)["query_idx"]
        .nunique()
        .rename(columns={"query_idx": "num_queries"})
    )
    return subset_iter.merge(counts, on=["subset", "iter"], how="left")


def _build_pool_curve_wide(pool_subset_iter_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    pivot = pool_subset_iter_df.pivot(index="subset", columns="iter", values=value_col).sort_index()
    pivot.columns = [f"iter_{int(col)}" for col in pivot.columns]
    return pivot.reset_index()


def _aggregate_subset(rows_df: pd.DataFrame) -> pd.DataFrame:
    subset_rows: List[Dict[str, Any]] = []
    for subset, df_subset in rows_df.groupby("subset"):
        entered = df_subset[df_subset["first_enter_iter"].notna()]
        exited = df_subset[df_subset["first_exit_iter"].notna()]
        subset_rows.append(
            {
                "subset": subset,
                "num_queries": int(len(df_subset)),
                "enter_rate": float(len(entered) / len(df_subset)) if len(df_subset) else float("nan"),
                "exit_after_enter_rate": float(len(exited) / len(entered)) if len(entered) else float("nan"),
                "survive_to_end_rate": float(entered["survive_to_end"].mean()) if len(entered) else float("nan"),
                "first_enter_iter_mean": float(entered["first_enter_iter"].mean()) if len(entered) else float("nan"),
                "first_exit_iter_mean": float(exited["first_exit_iter"].mean()) if len(exited) else float("nan"),
                "contiguous_on_region_span_mean": float(entered["contiguous_on_region_span"].mean()) if len(entered) else float("nan"),
                "enter_depth_mean": float(entered["enter_depth_mean"].mean()) if len(entered) else float("nan"),
                "enter_depth_max_mean": float(entered["enter_depth_max"].mean()) if len(entered) else float("nan"),
                "exit_prev_depth_mean": float(exited["exit_prev_depth_mean"].mean()) if len(exited) else float("nan"),
                "exit_prev_depth_max_mean": float(exited["exit_prev_depth_max"].mean()) if len(exited) else float("nan"),
                "last_on_region_iter_mean": float(entered["last_on_region_iter"].mean()) if len(entered) else float("nan"),
            }
        )
    return pd.DataFrame(subset_rows).sort_values("subset").reset_index(drop=True)


def _aggregate_overall(subset_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [col for col in subset_df.columns if col not in {"subset", "num_queries"}]
    overall = {"num_subsets": int(len(subset_df))}
    for col in metric_cols:
        overall[col] = float(subset_df[col].mean()) if len(subset_df) else float("nan")
    return pd.DataFrame([overall])


def _build_histograms(rows_df: pd.DataFrame) -> pd.DataFrame:
    counter: Counter[Tuple[str, str]] = Counter()
    entered = rows_df[rows_df["first_enter_iter"].notna()]
    exited = rows_df[rows_df["first_exit_iter"].notna()]

    for value in entered["first_enter_iter"].tolist():
        counter[("first_enter_iter", str(int(value)))] += 1
    for value in exited["first_exit_iter"].tolist():
        counter[("first_exit_iter", str(int(value)))] += 1
    for value in exited["exit_prev_depth_max"].tolist():
        if not math.isnan(float(value)):
            counter[("exit_prev_depth_max", str(int(value)))] += 1

    rows = [
        {"metric": metric, "bucket": bucket, "count": int(count)}
        for (metric, bucket), count in sorted(counter.items())
    ]
    return pd.DataFrame(rows)


def _build_examples(rows_df: pd.DataFrame) -> pd.DataFrame:
    exited = rows_df[rows_df["first_exit_iter"].notna()].copy()
    exited = exited.sort_values(["first_exit_iter", "exit_prev_depth_max", "query_idx"], ascending=[True, False, True])
    return exited.head(100).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze when round6 ours leaves the gold branch region.")
    parser.add_argument("--glob_pattern", default=DEFAULT_GLOB)
    parser.add_argument("--out_prefix", default="results/BRIGHT/analysis/round6_gold_region_exit")
    parser.add_argument("--require_substrings", nargs="*", default=DEFAULT_REQUIRE)
    parser.add_argument("--exclude_substrings", nargs="*", default=DEFAULT_EXCLUDE)
    args = parser.parse_args()

    path_map = _resolve_paths(
        glob_pattern=str(args.glob_pattern),
        require_substrings=list(args.require_substrings),
        exclude_substrings=list(args.exclude_substrings),
    )
    samples_by_subset = _load_samples_by_subset(path_map)

    rows_df, iter_rows_df = _build_rows(samples_by_subset)
    subset_df = _aggregate_subset(rows_df)
    overall_df = _aggregate_overall(subset_df)
    hist_df = _build_histograms(rows_df)
    examples_df = _build_examples(rows_df)
    pool_subset_iter_df = _aggregate_pool_subset_iter(iter_rows_df)
    pool_curve_has_any_df = _build_pool_curve_wide(pool_subset_iter_df, "pool_has_any_gold_doc")
    pool_curve_recall_df = _build_pool_curve_wide(pool_subset_iter_df, "pool_gold_doc_recall")

    out_dir = os.path.dirname(os.path.abspath(args.out_prefix))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    rows_df.to_csv(f"{args.out_prefix}_rows.csv", index=False)
    iter_rows_df.to_csv(f"{args.out_prefix}_iter_rows.csv", index=False)
    subset_df.to_csv(f"{args.out_prefix}_subset_summary.csv", index=False)
    overall_df.to_csv(f"{args.out_prefix}_overall_summary.csv", index=False)
    hist_df.to_csv(f"{args.out_prefix}_hist.csv", index=False)
    examples_df.to_csv(f"{args.out_prefix}_examples.csv", index=False)
    pool_subset_iter_df.to_csv(f"{args.out_prefix}_pool_subset_iter.csv", index=False)
    pool_curve_has_any_df.to_csv(f"{args.out_prefix}_pool_curve_has_any.csv", index=False)
    pool_curve_recall_df.to_csv(f"{args.out_prefix}_pool_curve_recall.csv", index=False)

    print(f"Saved rows to {args.out_prefix}_rows.csv")
    print(f"Saved iter rows to {args.out_prefix}_iter_rows.csv")
    print(f"Saved subset summary to {args.out_prefix}_subset_summary.csv")
    print(f"Saved overall summary to {args.out_prefix}_overall_summary.csv")
    print(f"Saved histograms to {args.out_prefix}_hist.csv")
    print(f"Saved examples to {args.out_prefix}_examples.csv")
    print(f"Saved pool subset-iter summary to {args.out_prefix}_pool_subset_iter.csv")
    print(f"Saved pool has-any curve to {args.out_prefix}_pool_curve_has_any.csv")
    print(f"Saved pool recall curve to {args.out_prefix}_pool_curve_recall.csv")


if __name__ == "__main__":
    main()
