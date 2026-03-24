#!/usr/bin/env python3
"""Analyze why descendant-flat whole-tree underperforms direct-child ended-reseat.

Purpose
-------
Compare two round6 runs at the branch-selection level:

- ours: direct-child active selection + leftover-expandable ended reseat
- flat: descendant-flat active selection + whole-tree-flat ended reseat

The main questions are:

    1. Does flat hit gold-ancestor branches less often, especially early?
    2. Does flat select deeper but noisier branches, which would mean it loses
       the coarse-to-fine filtering benefit of child-only expansion?

This script aligns runs by (subset, query_idx), aggregates subset means first,
and then reports overall equal-weight subset means to stay compatible with the
summary CSV interpretation.

Examples
--------
python scripts/analysis/analyze_round6_flat_vs_ended_reseat.py

python scripts/analysis/analyze_round6_flat_vs_ended_reseat.py \
    --out_prefix results/BRIGHT/analysis/round6_flat_vs_ended_reseat
"""

import argparse
import glob
import json
import math
import os
import pickle as pkl
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_BASE_DIR = "results/BRIGHT"
DEFAULT_GLOB = "results/BRIGHT/*/round6/**/all_eval_sample_dicts.pkl"
DEFAULT_OURS_REQUIRE = [
    "MaxBS=10-S=round6_mrr_selector_accum_meanscore_global_expandable_ended_reseat",
    "embed-qwen3-8b-0928",
    "agent_executor_v1_icl2",
    "RSM=meanscore_global",
    "REM=ended_reseat",
]
DEFAULT_OURS_EXCLUDE = [
    "descendant_flat",
    "MaxBS=1-",
]
DEFAULT_FLAT_REQUIRE = [
    "MaxBS=10-S=round6_mrr_selector_accum_meanscore_global_expandable_ended_reseat_descendant_flat_whole_tree_flat",
    "embed-qwen3-8b-0928",
    "agent_executor_v1_icl2",
    "RECM=descendant_flat",
    "REES=whole_tree_flat",
]
DEFAULT_FLAT_EXCLUDE = [
    "goexplore_direct_child",
    "MaxBS=1-",
]


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _infer_subset_from_path(path: str) -> str:
    parts = os.path.abspath(path).split(os.sep)
    for idx, part in enumerate(parts):
        if part == "results" and (idx + 2) < len(parts):
            return str(parts[idx + 2])
    return "unknown"


def _resolve_candidate_paths(
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
        raise FileNotFoundError("No files matched the given filters.")
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


def _first_gold_hit_iter(iter_records: Sequence[Dict[str, Any]], gold_paths: Sequence[Sequence[int]]) -> float:
    for iter_idx, rec in enumerate(iter_records):
        selected = [tuple(path) for path in (rec.get("selected_branches_after", []) or []) if path]
        if any(_branch_is_gold_ancestor(path, gold_paths) for path in selected):
            return float(iter_idx)
    return float("nan")


def _iter_metrics_for_sample(
    subset: str,
    query_idx: int,
    model: str,
    sample: Dict[str, Any],
) -> List[Dict[str, Any]]:
    gold_paths = [tuple(path) for path in (sample.get("gold_paths", []) or []) if path]
    iter_records = sample.get("iter_records", []) or []
    first_gold_hit_iter = _first_gold_hit_iter(iter_records, gold_paths)
    ever_gold_hit = int(not math.isnan(first_gold_hit_iter))

    rows: List[Dict[str, Any]] = []
    for iter_idx, rec in enumerate(iter_records):
        selected = [tuple(path) for path in (rec.get("selected_branches_after", []) or []) if path]
        if not selected:
            continue

        gold_flags = [_branch_is_gold_ancestor(path, gold_paths) for path in selected]
        branch_hit = int(any(gold_flags))
        branch_precision = float(np.mean(gold_flags)) if gold_flags else float("nan")

        selected_depths = [len(path) for path in selected]
        gold_selected_depths = [len(path) for path, is_gold in zip(selected, gold_flags) if is_gold]
        nongold_selected_depths = [len(path) for path, is_gold in zip(selected, gold_flags) if not is_gold]

        rows.append(
            {
                "model": model,
                "subset": subset,
                "query_idx": int(query_idx),
                "iter": int(iter_idx),
                "ndcg_mean": _safe_float(rec.get("metrics", {}).get("nDCG@10")),
                "branch_hit_rate": float(branch_hit),
                "branch_precision_mean": branch_precision,
                "selected_depth_mean": float(np.mean(selected_depths)) if selected_depths else float("nan"),
                "selected_depth_max_mean": float(max(selected_depths)) if selected_depths else float("nan"),
                "gold_hit_selected_depth_mean": float(np.mean(gold_selected_depths)) if gold_selected_depths else float("nan"),
                "nongold_selected_depth_mean": float(np.mean(nongold_selected_depths)) if nongold_selected_depths else float("nan"),
                "nongold_selected_ratio": float(np.mean([not flag for flag in gold_flags])) if gold_flags else float("nan"),
                "num_selected_branches_mean": float(len(selected)),
                "ever_gold_hit_rate": float(ever_gold_hit),
                "first_gold_hit_iter_mean": float(first_gold_hit_iter),
                "selected_branches_after": _json_dumps([list(path) for path in selected]),
                "selected_gold_flags": _json_dumps(gold_flags),
                "gold_paths": _json_dumps([list(path) for path in gold_paths]),
            }
        )
    return rows


def _build_query_rows(
    model: str,
    samples_by_subset: Dict[str, List[Dict[str, Any]]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for subset, samples in samples_by_subset.items():
        for query_idx, sample in enumerate(samples):
            rows.extend(_iter_metrics_for_sample(subset, query_idx, model, sample))
    return pd.DataFrame(rows)


def _aggregate_iter_rows(query_rows: pd.DataFrame) -> pd.DataFrame:
    if query_rows.empty:
        return query_rows
    metric_cols = [
        "ndcg_mean",
        "branch_hit_rate",
        "branch_precision_mean",
        "selected_depth_mean",
        "selected_depth_max_mean",
        "gold_hit_selected_depth_mean",
        "nongold_selected_depth_mean",
        "nongold_selected_ratio",
        "num_selected_branches_mean",
        "ever_gold_hit_rate",
        "first_gold_hit_iter_mean",
    ]
    grouped = (
        query_rows.groupby(["model", "subset", "iter"], as_index=False)[metric_cols]
        .mean(numeric_only=True)
        .sort_values(["model", "subset", "iter"])
    )
    counts = (
        query_rows.groupby(["model", "subset", "iter"], as_index=False)["query_idx"]
        .nunique()
        .rename(columns={"query_idx": "num_queries"})
    )
    return grouped.merge(counts, on=["model", "subset", "iter"], how="left")


def _build_iter_compare(iter_rows: pd.DataFrame) -> pd.DataFrame:
    ours = iter_rows[iter_rows["model"] == "ours"].copy()
    flat = iter_rows[iter_rows["model"] == "flat"].copy()
    ours = ours.rename(columns={col: f"ours_{col}" for col in ours.columns if col not in {"subset", "iter"}})
    flat = flat.rename(columns={col: f"flat_{col}" for col in flat.columns if col not in {"subset", "iter"}})
    merged = ours.merge(flat, on=["subset", "iter"], how="inner")

    # Intent: delta signs follow the hypothesis we want to test directly in the CSV.
    merged["delta_ndcg_mean"] = merged["ours_ndcg_mean"] - merged["flat_ndcg_mean"]
    merged["delta_branch_hit_rate"] = merged["ours_branch_hit_rate"] - merged["flat_branch_hit_rate"]
    merged["delta_branch_precision_mean"] = merged["ours_branch_precision_mean"] - merged["flat_branch_precision_mean"]
    merged["delta_selected_depth_mean"] = merged["flat_selected_depth_mean"] - merged["ours_selected_depth_mean"]
    merged["delta_selected_depth_max_mean"] = merged["flat_selected_depth_max_mean"] - merged["ours_selected_depth_max_mean"]
    merged["delta_nongold_selected_ratio"] = merged["flat_nongold_selected_ratio"] - merged["ours_nongold_selected_ratio"]
    merged["delta_gold_hit_selected_depth_mean"] = (
        merged["flat_gold_hit_selected_depth_mean"] - merged["ours_gold_hit_selected_depth_mean"]
    )
    return merged.sort_values(["subset", "iter"]).reset_index(drop=True)


def _build_overall_compare(iter_compare: pd.DataFrame) -> pd.DataFrame:
    if iter_compare.empty:
        return iter_compare
    metric_cols = [col for col in iter_compare.columns if col not in {"subset", "iter"}]
    # Intent: overall should match summary-style interpretation, so subset means get equal weight.
    overall = iter_compare.groupby("iter", as_index=False)[metric_cols].mean(numeric_only=True)
    subset_counts = iter_compare.groupby("iter", as_index=False)["subset"].nunique().rename(columns={"subset": "num_subsets"})
    overall = overall.merge(subset_counts, on="iter", how="left")
    return overall.sort_values("iter").reset_index(drop=True)


def _build_examples(query_rows: pd.DataFrame) -> pd.DataFrame:
    ours = query_rows[query_rows["model"] == "ours"].copy()
    flat = query_rows[query_rows["model"] == "flat"].copy()
    merge_cols = ["subset", "query_idx", "iter"]
    ours = ours.rename(columns={col: f"ours_{col}" for col in ours.columns if col not in merge_cols})
    flat = flat.rename(columns={col: f"flat_{col}" for col in flat.columns if col not in merge_cols})
    merged = ours.merge(flat, on=merge_cols, how="inner")

    merged["ours_minus_flat_ndcg"] = merged["ours_ndcg_mean"] - merged["flat_ndcg_mean"]
    merged["flat_minus_ours_selected_depth_mean"] = merged["flat_selected_depth_mean"] - merged["ours_selected_depth_mean"]
    merged["flat_minus_ours_nongold_selected_ratio"] = (
        merged["flat_nongold_selected_ratio"] - merged["ours_nongold_selected_ratio"]
    )

    ours_hits_flat_miss = merged[
        (merged["ours_branch_hit_rate"] > 0.5) & (merged["flat_branch_hit_rate"] < 0.5)
    ].sort_values(
        ["ours_minus_flat_ndcg", "iter"], ascending=[False, True]
    ).head(40)
    flat_deeper_nongold = merged[
        (merged["flat_minus_ours_selected_depth_mean"] > 0.0) &
        (merged["flat_minus_ours_nongold_selected_ratio"] > 0.0)
    ].sort_values(
        ["flat_minus_ours_nongold_selected_ratio", "flat_minus_ours_selected_depth_mean"],
        ascending=[False, False],
    ).head(40)

    examples = pd.concat([ours_hits_flat_miss, flat_deeper_nongold], ignore_index=True).drop_duplicates(
        subset=["subset", "query_idx", "iter"]
    )
    return examples.sort_values(["subset", "query_idx", "iter"]).reset_index(drop=True)


def _write_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze flat vs ended-reseat branch-selection behavior.")
    parser.add_argument("--base_dir", type=str, default=DEFAULT_BASE_DIR, help="Base results directory.")
    parser.add_argument("--glob_pattern", type=str, default=DEFAULT_GLOB, help="Glob pattern for eval sample pickles.")
    parser.add_argument(
        "--ours_require",
        action="append",
        default=list(DEFAULT_OURS_REQUIRE),
        help="Substring required for the ours run. Can be repeated.",
    )
    parser.add_argument(
        "--ours_exclude",
        action="append",
        default=list(DEFAULT_OURS_EXCLUDE),
        help="Substring excluded from the ours run. Can be repeated.",
    )
    parser.add_argument(
        "--flat_require",
        action="append",
        default=list(DEFAULT_FLAT_REQUIRE),
        help="Substring required for the flat run. Can be repeated.",
    )
    parser.add_argument(
        "--flat_exclude",
        action="append",
        default=list(DEFAULT_FLAT_EXCLUDE),
        help="Substring excluded from the flat run. Can be repeated.",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="results/BRIGHT/analysis/round6_flat_vs_ended_reseat",
        help="Output prefix for CSV files.",
    )
    args = parser.parse_args()

    if not os.path.isabs(args.glob_pattern):
        glob_pattern = os.path.join(os.getcwd(), args.glob_pattern)
    else:
        glob_pattern = args.glob_pattern

    ours_paths = _resolve_candidate_paths(glob_pattern, args.ours_require, args.ours_exclude)
    flat_paths = _resolve_candidate_paths(glob_pattern, args.flat_require, args.flat_exclude)
    common_subsets = sorted(set(ours_paths) & set(flat_paths))
    if not common_subsets:
        raise RuntimeError("No common subsets found between ours and flat runs.")

    ours_samples = _load_samples_by_subset({subset: ours_paths[subset] for subset in common_subsets})
    flat_samples = _load_samples_by_subset({subset: flat_paths[subset] for subset in common_subsets})

    for subset in common_subsets:
        if len(ours_samples[subset]) != len(flat_samples[subset]):
            raise ValueError(
                f"Sample count mismatch for subset={subset}: ours={len(ours_samples[subset])} flat={len(flat_samples[subset])}"
            )

    ours_query_rows = _build_query_rows("ours", ours_samples)
    flat_query_rows = _build_query_rows("flat", flat_samples)
    query_rows = pd.concat([ours_query_rows, flat_query_rows], ignore_index=True)

    iter_rows = _aggregate_iter_rows(query_rows)
    iter_compare = _build_iter_compare(iter_rows)
    overall_compare = _build_overall_compare(iter_compare)
    examples = _build_examples(query_rows)

    _write_csv(iter_rows, f"{args.out_prefix}_iter_rows.csv")
    _write_csv(iter_compare, f"{args.out_prefix}_iter_compare.csv")
    _write_csv(overall_compare, f"{args.out_prefix}_overall_compare.csv")
    _write_csv(examples, f"{args.out_prefix}_examples.csv")

    print(f"Wrote {len(iter_rows)} rows to {args.out_prefix}_iter_rows.csv")
    print(f"Wrote {len(iter_compare)} rows to {args.out_prefix}_iter_compare.csv")
    print(f"Wrote {len(overall_compare)} rows to {args.out_prefix}_overall_compare.csv")
    print(f"Wrote {len(examples)} rows to {args.out_prefix}_examples.csv")
    print(f"Compared {len(common_subsets)} common subsets: {', '.join(common_subsets)}")


if __name__ == "__main__":
    main()
