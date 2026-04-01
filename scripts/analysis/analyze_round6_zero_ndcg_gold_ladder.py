#!/usr/bin/env python3
"""Analyze exact-gold survival for zero-nDCG rows inside the gold family.

Purpose
-------
The prefix-depth analysis showed that many `nDCG@10 = 0` rows still reach the
gold branch family. This script asks the narrower follow-up:

- once the run is already near the gold leaf family,
- where does the exact gold document disappear?

We stay artifact-only and do not use an LLM judge here.
"""

import argparse
import glob
import json
import math
import os
import pickle as pkl
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_GLOB = "results/BRIGHT/*/round6*/**/all_eval_sample_dicts.pkl"
DEFAULT_REQUIRE = [
    "MaxBS=10-S=round6_mrr_selector_accum_meanscore_global_expandable_ended_reseat_frontiercum_qstate",
    "reason/embed-qwen3-8b-0928",
    "agent_executor_v1_icl2",
    "RSM=meanscore_global-REM=ended_reseat",
    "RRrfK=60-RRC=leaf-REM=replace-RB=frontiercum_qstate_v1",
]
DEFAULT_EXCLUDE = [
    "RERP=random",
    "_emr",
    "descendant_flat",
    "MaxBS=1-",
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
        raise FileNotFoundError("No files matched the requested run filters.")
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


def _normalize_paths(items: Sequence[Any]) -> List[Tuple[int, ...]]:
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


def _normalize_doc_ids(items: Sequence[Any]) -> List[str]:
    return [str(x) for x in list(items or []) if str(x)]


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _lcp_len(path_a: Sequence[int], path_b: Sequence[int]) -> int:
    limit = min(len(path_a), len(path_b))
    depth = 0
    for idx in range(limit):
        if int(path_a[idx]) != int(path_b[idx]):
            break
        depth += 1
    return int(depth)


def _best_gold_prefix_depth(candidate_paths: Sequence[Sequence[int]], gold_paths: Sequence[Sequence[int]]) -> int:
    best = 0
    for candidate in list(candidate_paths or []):
        for gold in list(gold_paths or []):
            best = max(best, _lcp_len(candidate, gold))
    return int(best)


def _best_rank(doc_ids: Sequence[str], gold_doc_ids: Sequence[str], limit: int) -> float:
    gold_set = set(str(x) for x in list(gold_doc_ids or []))
    for rank_idx, doc_id in enumerate(list(doc_ids or [])[:limit], start=1):
        if str(doc_id) in gold_set:
            return float(rank_idx)
    return float("nan")


def _has_any_gold(doc_ids: Sequence[str], gold_doc_ids: Sequence[str], limit: int) -> int:
    gold_set = set(str(x) for x in list(gold_doc_ids or []))
    return int(any(str(doc_id) in gold_set for doc_id in list(doc_ids or [])[:limit]))


def _on_family_non_gold_ratio(
    ranked_paths: Sequence[Tuple[int, ...]],
    ranked_doc_ids: Sequence[str],
    gold_paths: Sequence[Tuple[int, ...]],
    gold_doc_ids: Sequence[str],
    family_gate_depth: int,
    topk: int,
) -> float:
    gold_set = set(str(x) for x in list(gold_doc_ids or []))
    limit = min(topk, len(ranked_paths), len(ranked_doc_ids))
    if limit <= 0:
        return float("nan")
    wrong = 0
    for path, doc_id in zip(list(ranked_paths)[:limit], list(ranked_doc_ids)[:limit]):
        on_family = any(_lcp_len(path, gold_path) >= family_gate_depth for gold_path in gold_paths)
        if on_family and str(doc_id) not in gold_set:
            wrong += 1
    return float(wrong / limit)


def _on_family_ratio(
    ranked_paths: Sequence[Tuple[int, ...]],
    gold_paths: Sequence[Tuple[int, ...]],
    family_gate_depth: int,
    topk: int,
) -> float:
    limit = min(topk, len(ranked_paths))
    if limit <= 0:
        return float("nan")
    hits = 0
    for path in list(ranked_paths)[:limit]:
        if any(_lcp_len(path, gold_path) >= family_gate_depth for gold_path in gold_paths):
            hits += 1
    return float(hits / limit)


def _bucket_label(
    family_hit: int,
    gold_in_prehit100: int,
    gold_in_active100: int,
    gold_in_active10: int,
) -> str:
    # Intent: keep the ladder exclusive, but preserve the prehit-miss/active-recover branch as its own failure mode.
    if not family_hit:
        return "family_miss"
    if gold_in_active10:
        return "gold_present_in_active10"
    if gold_in_active100 and gold_in_prehit100:
        return "gold_present_in_prehit100_and_active100_but_missing_in_active10"
    if gold_in_active100 and not gold_in_prehit100:
        return "gold_recovered_in_active100_after_prehit_miss_but_missing_in_active10"
    if gold_in_prehit100 and not gold_in_active100:
        return "gold_present_in_prehit100_but_missing_in_active100"
    return "gold_absent_from_prehit100_and_active100"


def _row_from_iter(subset: str, query_idx: int, sample: Dict[str, Any], rec: Dict[str, Any]) -> Dict[str, Any]:
    gold_paths = _normalize_paths(sample.get("gold_paths", []) or [])
    gold_doc_ids = _normalize_doc_ids(sample.get("gold_doc_ids", []) or rec.get("gold_doc_ids", []) or [])
    selected_after = _normalize_paths(rec.get("selected_branches_after", []) or [])
    prehit_paths = _normalize_paths(rec.get("pre_hit_paths", []) or [])
    active_paths = _normalize_paths(rec.get("active_eval_paths", []) or [])
    prehit_doc_ids = _normalize_doc_ids(rec.get("pre_hit_doc_ids", []) or [])
    active_doc_ids = _normalize_doc_ids(rec.get("active_eval_doc_ids", []) or [])

    min_gold_leaf_depth = int(min(len(path) for path in gold_paths)) if gold_paths else 0
    family_gate_depth = max(int(min_gold_leaf_depth) - 1, 0)

    selected_best_depth = _best_gold_prefix_depth(selected_after, gold_paths)
    prehit_best_depth = _best_gold_prefix_depth(prehit_paths, gold_paths)
    active_best_depth = _best_gold_prefix_depth(active_paths, gold_paths)

    family_hit = int(active_best_depth >= family_gate_depth) if gold_paths else 0
    gold_in_prehit100 = _has_any_gold(prehit_doc_ids, gold_doc_ids, 100)
    gold_in_active100 = _has_any_gold(active_doc_ids, gold_doc_ids, 100)
    gold_in_active10 = _has_any_gold(active_doc_ids, gold_doc_ids, 10)

    return {
        "subset": subset,
        "query_idx": int(query_idx),
        "iter": int(rec.get("iter", 0) or 0),
        "ndcg10": _safe_float(rec.get("metrics", {}).get("nDCG@10")),
        "query_pre": str(rec.get("query_pre", "") or ""),
        "query_post": str(rec.get("query_post", "") or ""),
        "possible_answer_docs": _json_dumps(rec.get("possible_answer_docs", {}) or {}),
        "gold_doc_ids": _json_dumps(gold_doc_ids),
        "gold_paths": _json_dumps([list(path) for path in gold_paths]),
        "selected_branches_after": _json_dumps([list(path) for path in selected_after]),
        "pre_hit_doc_ids_top20": _json_dumps(prehit_doc_ids[:20]),
        "active_eval_doc_ids_top20": _json_dumps(active_doc_ids[:20]),
        "selected_best_gold_prefix_depth": int(selected_best_depth),
        "prehit_best_gold_prefix_depth": int(prehit_best_depth),
        "active_best_gold_prefix_depth": int(active_best_depth),
        "min_gold_leaf_depth_for_query": int(min_gold_leaf_depth),
        "family_gate_depth": int(family_gate_depth),
        "family_hit_active_near_leaf": int(family_hit),
        "gold_in_prehit100": int(gold_in_prehit100),
        "gold_in_active100": int(gold_in_active100),
        "gold_in_active10": int(gold_in_active10),
        "best_gold_rank_prehit100": _best_rank(prehit_doc_ids, gold_doc_ids, 100),
        "best_gold_rank_active100": _best_rank(active_doc_ids, gold_doc_ids, 100),
        "best_gold_rank_active1000": _best_rank(active_doc_ids, gold_doc_ids, 1000),
        "on_family_top10_ratio": _on_family_ratio(
            ranked_paths=active_paths,
            gold_paths=gold_paths,
            family_gate_depth=family_gate_depth,
            topk=10,
        ),
        "on_family_top100_ratio": _on_family_ratio(
            ranked_paths=active_paths,
            gold_paths=gold_paths,
            family_gate_depth=family_gate_depth,
            topk=100,
        ),
        "on_family_top10_non_gold_ratio": _on_family_non_gold_ratio(
            ranked_paths=active_paths,
            ranked_doc_ids=active_doc_ids,
            gold_paths=gold_paths,
            gold_doc_ids=gold_doc_ids,
            family_gate_depth=family_gate_depth,
            topk=10,
        ),
        "on_family_top100_non_gold_ratio": _on_family_non_gold_ratio(
            ranked_paths=active_paths,
            ranked_doc_ids=active_doc_ids,
            gold_paths=gold_paths,
            gold_doc_ids=gold_doc_ids,
            family_gate_depth=family_gate_depth,
            topk=100,
        ),
        "ladder_bucket": _bucket_label(
            family_hit=family_hit,
            gold_in_prehit100=gold_in_prehit100,
            gold_in_active100=gold_in_active100,
            gold_in_active10=gold_in_active10,
        ),
    }


def _build_rows(samples_by_subset: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for subset, samples in samples_by_subset.items():
        for query_idx, sample in enumerate(samples):
            for rec in list(sample.get("iter_records", []) or []):
                ndcg10 = _safe_float(rec.get("metrics", {}).get("nDCG@10"))
                if abs(ndcg10) > 1e-9:
                    continue
                rows.append(_row_from_iter(subset, query_idx, sample, rec))
    if not rows:
        raise ValueError("No nDCG=0 iteration rows found for the requested run.")
    return pd.DataFrame(rows)


def _overall_summary(rows_df: pd.DataFrame) -> pd.DataFrame:
    total_rows = float(len(rows_df))
    family_df = rows_df[rows_df["family_hit_active_near_leaf"] == 1].copy()
    bucket_counts = rows_df["ladder_bucket"].value_counts().sort_index()
    out_rows: List[Dict[str, Any]] = []
    out_rows.append({
        "metric": "total_zero_ndcg_rows",
        "count": int(len(rows_df)),
        "pct_of_all_zero_ndcg": 100.0,
    })
    out_rows.append({
        "metric": "family_hit_active_near_leaf",
        "count": int(len(family_df)),
        "pct_of_all_zero_ndcg": float(len(family_df) / total_rows * 100.0) if total_rows else float("nan"),
    })
    for bucket, count in bucket_counts.items():
        out_rows.append({
            "metric": str(bucket),
            "count": int(count),
            "pct_of_all_zero_ndcg": float(count / total_rows * 100.0) if total_rows else float("nan"),
            "pct_within_family_hit": float(count / len(family_df) * 100.0) if len(family_df) and bucket != "family_miss" else (
                0.0 if len(family_df) and bucket == "family_miss" else float("nan")
            ),
        })
    if len(family_df):
        out_rows.extend([
            {
                "metric": "family_hit_best_gold_rank_active100_mean",
                "count": int(family_df["best_gold_rank_active100"].notna().sum()),
                "value": float(family_df["best_gold_rank_active100"].dropna().mean()),
            },
            {
                "metric": "family_hit_on_family_top10_ratio_mean",
                "count": int(family_df["on_family_top10_ratio"].notna().sum()),
                "value": float(family_df["on_family_top10_ratio"].dropna().mean()),
            },
            {
                "metric": "family_hit_on_family_top100_ratio_mean",
                "count": int(family_df["on_family_top100_ratio"].notna().sum()),
                "value": float(family_df["on_family_top100_ratio"].dropna().mean()),
            },
            {
                "metric": "family_hit_on_family_top10_non_gold_ratio_mean",
                "count": int(family_df["on_family_top10_non_gold_ratio"].notna().sum()),
                "value": float(family_df["on_family_top10_non_gold_ratio"].dropna().mean()),
            },
            {
                "metric": "family_hit_on_family_top100_non_gold_ratio_mean",
                "count": int(family_df["on_family_top100_non_gold_ratio"].notna().sum()),
                "value": float(family_df["on_family_top100_non_gold_ratio"].dropna().mean()),
            },
        ])
    return pd.DataFrame(out_rows)


def _subset_summary(rows_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for subset, subset_df in rows_df.groupby("subset"):
        total = float(len(subset_df))
        family_df = subset_df[subset_df["family_hit_active_near_leaf"] == 1].copy()
        bucket_counts = subset_df["ladder_bucket"].value_counts().sort_index()
        rows.append({
            "subset": subset,
            "metric": "family_hit_active_near_leaf",
            "count": int(len(family_df)),
            "pct_of_subset_zero_ndcg": float(len(family_df) / total * 100.0) if total else float("nan"),
        })
        for bucket, count in bucket_counts.items():
            rows.append({
                "subset": subset,
                "metric": str(bucket),
                "count": int(count),
                "pct_of_subset_zero_ndcg": float(count / total * 100.0) if total else float("nan"),
            })
    return pd.DataFrame(rows)


def _rank_hist(rows_df: pd.DataFrame) -> pd.DataFrame:
    family_df = rows_df[rows_df["family_hit_active_near_leaf"] == 1].copy()
    family_df = family_df[family_df["best_gold_rank_active100"].notna()].copy()
    if family_df.empty:
        return pd.DataFrame(columns=["rank_bucket", "count", "pct"])
    bins = [
        ("1-10", family_df["best_gold_rank_active100"].between(1, 10, inclusive="both")),
        ("11-20", family_df["best_gold_rank_active100"].between(11, 20, inclusive="both")),
        ("21-50", family_df["best_gold_rank_active100"].between(21, 50, inclusive="both")),
        ("51-100", family_df["best_gold_rank_active100"].between(51, 100, inclusive="both")),
    ]
    total = float(len(family_df))
    out_rows: List[Dict[str, Any]] = []
    for label, mask in bins:
        count = int(mask.sum())
        out_rows.append({
            "rank_bucket": label,
            "count": count,
            "pct": float(count / total * 100.0) if total else float("nan"),
        })
    return pd.DataFrame(out_rows)


def _examples_df(rows_df: pd.DataFrame) -> pd.DataFrame:
    example_parts: List[pd.DataFrame] = []
    for bucket in [
        "family_miss",
        "gold_absent_from_prehit100_and_active100",
        "gold_present_in_prehit100_but_missing_in_active100",
        "gold_recovered_in_active100_after_prehit_miss_but_missing_in_active10",
        "gold_present_in_prehit100_and_active100_but_missing_in_active10",
        "gold_present_in_active10",
    ]:
        bucket_df = rows_df[rows_df["ladder_bucket"] == bucket].copy()
        if bucket_df.empty:
            continue
        # Intent: sort by the strongest family-hit / exact-gold contradiction first for readable examples.
        bucket_df = bucket_df.sort_values(
            ["subset", "iter", "active_best_gold_prefix_depth", "best_gold_rank_active100"],
            ascending=[True, True, False, True],
            na_position="last",
        ).head(3)
        bucket_df.insert(0, "example_bucket", bucket)
        example_parts.append(bucket_df)
    if not example_parts:
        return pd.DataFrame(columns=["example_bucket"] + list(rows_df.columns))
    return pd.concat(example_parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze exact-gold ladder for zero-nDCG rows.")
    parser.add_argument("--glob_pattern", default=DEFAULT_GLOB)
    parser.add_argument("--out_prefix", default="results/BRIGHT/analysis/round6_zero_ndcg_gold_ladder")
    parser.add_argument("--require_substrings", nargs="*", default=DEFAULT_REQUIRE)
    parser.add_argument("--exclude_substrings", nargs="*", default=DEFAULT_EXCLUDE)
    args = parser.parse_args()

    path_map = _resolve_paths(
        glob_pattern=str(args.glob_pattern),
        require_substrings=list(args.require_substrings),
        exclude_substrings=list(args.exclude_substrings),
    )
    samples_by_subset = _load_samples_by_subset(path_map)
    rows_df = _build_rows(samples_by_subset)

    overall_df = _overall_summary(rows_df)
    subset_df = _subset_summary(rows_df)
    rank_hist_df = _rank_hist(rows_df)
    examples_df = _examples_df(rows_df)

    out_dir = os.path.dirname(os.path.abspath(args.out_prefix))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    rows_df.to_csv(f"{args.out_prefix}_rows.csv", index=False)
    overall_df.to_csv(f"{args.out_prefix}_overall_summary.csv", index=False)
    subset_df.to_csv(f"{args.out_prefix}_subset_summary.csv", index=False)
    rank_hist_df.to_csv(f"{args.out_prefix}_rank_hist.csv", index=False)
    examples_df.to_csv(f"{args.out_prefix}_examples.csv", index=False)

    print(f"matched_subsets={len(path_map)}")
    print(f"zero_ndcg_rows={len(rows_df)}")
    print(overall_df.to_string(index=False))


if __name__ == "__main__":
    main()
