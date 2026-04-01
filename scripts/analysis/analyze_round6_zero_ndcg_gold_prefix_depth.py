#!/usr/bin/env python3
"""Analyze zero-nDCG iterations by gold-prefix depth distribution.

Purpose
-------
For the frontiercum_qstate ended-reseat run, answer a narrower question:
when `nDCG@10 = 0`, how deep into the gold branch does the controller/retrieval
still go?

We avoid an early binary taxonomy and instead measure depth distributions for
three stages:
- selected branches (`selected_branches_after`)
- retrieval evidence (`pre_hit_paths`)
- final active ranking (`active_eval_paths`)

Examples
--------
python scripts/analysis/analyze_round6_zero_ndcg_gold_prefix_depth.py

python scripts/analysis/analyze_round6_zero_ndcg_gold_prefix_depth.py \
    --out_prefix results/BRIGHT/analysis/round6_zero_ndcg_gold_prefix_depth
"""

import argparse
import glob
import json
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


def _depth_drop_label(selected_depth: int, prehit_depth: int, active_depth: int) -> str:
    # Intent: separate branch-selection failure from retrieval-follow failure before using any LLM judge.
    if selected_depth == 0:
        return "branch_miss"
    if selected_depth > prehit_depth:
        return "branch_to_prehit_drop"
    if prehit_depth > active_depth:
        return "prehit_to_active_drop"
    return "depth_preserved"


def _row_from_iter(subset: str, query_idx: int, sample: Dict[str, Any], rec: Dict[str, Any]) -> Dict[str, Any]:
    gold_paths = _normalize_paths(sample.get("gold_paths", []) or [])
    selected_after = _normalize_paths(rec.get("selected_branches_after", []) or [])
    selected_before = _normalize_paths(rec.get("selected_branches_before", []) or [])
    prehit_paths = _normalize_paths(rec.get("pre_hit_paths", []) or [])
    active_paths = _normalize_paths(rec.get("active_eval_paths", []) or [])

    selected_best_depth = _best_gold_prefix_depth(selected_after, gold_paths)
    prehit_best_depth = _best_gold_prefix_depth(prehit_paths, gold_paths)
    active_best_depth = _best_gold_prefix_depth(active_paths, gold_paths)

    return {
        "subset": subset,
        "query_idx": int(query_idx),
        "iter": int(rec.get("iter", 0) or 0),
        "ndcg10": _safe_float(rec.get("metrics", {}).get("nDCG@10")),
        "query_pre": str(rec.get("query_pre", "") or ""),
        "query_post": str(rec.get("query_post", "") or ""),
        "possible_answer_docs": _json_dumps(rec.get("possible_answer_docs", {}) or {}),
        "selected_branches_before": _json_dumps([list(path) for path in selected_before]),
        "selected_branches_after": _json_dumps([list(path) for path in selected_after]),
        "pre_hit_paths": _json_dumps([list(path) for path in prehit_paths[:20]]),
        "active_eval_paths": _json_dumps([list(path) for path in active_paths[:20]]),
        "gold_paths": _json_dumps([list(path) for path in gold_paths]),
        "selected_branch_count": int(len(selected_after)),
        "pre_hit_count": int(len(prehit_paths)),
        "active_eval_count": int(len(active_paths)),
        "selected_best_gold_prefix_depth": int(selected_best_depth),
        "prehit_best_gold_prefix_depth": int(prehit_best_depth),
        "active_best_gold_prefix_depth": int(active_best_depth),
        "depth_drop_label": _depth_drop_label(selected_best_depth, prehit_best_depth, active_best_depth),
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


def _hist_df(rows_df: pd.DataFrame, depth_col: str, value_name: str) -> pd.DataFrame:
    counts = rows_df[depth_col].value_counts(dropna=False).sort_index()
    total = int(counts.sum())
    out = pd.DataFrame({
        "depth": [int(x) for x in counts.index.tolist()],
        value_name: counts.values.astype(int),
    })
    out[f"{value_name}_pct"] = out[value_name] / float(total) * 100.0 if total else np.nan
    out["total_rows"] = total
    return out


def _subset_hist_df(rows_df: pd.DataFrame, depth_col: str) -> pd.DataFrame:
    grouped = (
        rows_df.groupby(["subset", depth_col], as_index=False)
        .size()
        .rename(columns={depth_col: "depth", "size": "count"})
    )
    totals = rows_df.groupby("subset", as_index=False).size().rename(columns={"size": "total_rows"})
    out = grouped.merge(totals, on="subset", how="left")
    out["pct"] = out["count"] / out["total_rows"] * 100.0
    return out.sort_values(["subset", "depth"]).reset_index(drop=True)


def _transition_summary(rows_df: pd.DataFrame) -> pd.DataFrame:
    total = float(len(rows_df))
    labels = rows_df["depth_drop_label"].value_counts().sort_index()
    return pd.DataFrame({
        "label": labels.index.tolist(),
        "count": labels.values.astype(int),
        "pct": (labels.values / total) * 100.0 if total else np.nan,
    })


def _examples_df(rows_df: pd.DataFrame) -> pd.DataFrame:
    example_parts: List[pd.DataFrame] = []
    for depth in sorted(rows_df["selected_best_gold_prefix_depth"].unique().tolist()):
        bucket = rows_df[rows_df["selected_best_gold_prefix_depth"] == depth].copy()
        bucket = bucket.sort_values(["subset", "iter", "query_idx"]).head(3)
        bucket.insert(0, "example_bucket", f"selected_depth_{int(depth)}")
        example_parts.append(bucket)
    for label in sorted(rows_df["depth_drop_label"].unique().tolist()):
        bucket = rows_df[rows_df["depth_drop_label"] == label].copy()
        bucket = bucket.sort_values(["subset", "iter", "query_idx"]).head(3)
        bucket.insert(0, "example_bucket", f"drop_{label}")
        example_parts.append(bucket)
    if not example_parts:
        return pd.DataFrame(columns=["example_bucket"] + list(rows_df.columns))
    return pd.concat(example_parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze nDCG=0 rows by gold-prefix depth.")
    parser.add_argument("--glob_pattern", default=DEFAULT_GLOB)
    parser.add_argument("--out_prefix", default="results/BRIGHT/analysis/round6_zero_ndcg_gold_prefix_depth")
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

    selected_hist_df = _hist_df(rows_df, "selected_best_gold_prefix_depth", "selected_count")
    prehit_hist_df = _hist_df(rows_df, "prehit_best_gold_prefix_depth", "prehit_count")
    active_hist_df = _hist_df(rows_df, "active_best_gold_prefix_depth", "active_count")
    subset_hist_df = _subset_hist_df(rows_df, "selected_best_gold_prefix_depth")
    transition_df = _transition_summary(rows_df)
    examples_df = _examples_df(rows_df)

    out_dir = os.path.dirname(os.path.abspath(args.out_prefix))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    rows_df.to_csv(f"{args.out_prefix}_rows.csv", index=False)
    selected_hist_df.to_csv(f"{args.out_prefix}_selected_depth_hist.csv", index=False)
    prehit_hist_df.to_csv(f"{args.out_prefix}_prehit_depth_hist.csv", index=False)
    active_hist_df.to_csv(f"{args.out_prefix}_active_depth_hist.csv", index=False)
    subset_hist_df.to_csv(f"{args.out_prefix}_subset_depth_hist.csv", index=False)
    transition_df.to_csv(f"{args.out_prefix}_transition_summary.csv", index=False)
    examples_df.to_csv(f"{args.out_prefix}_examples.csv", index=False)

    print(f"matched_subsets={len(path_map)}")
    print(f"zero_ndcg_rows={len(rows_df)}")
    print("selected_depth_hist")
    print(selected_hist_df.to_string(index=False))
    print("transition_summary")
    print(transition_df.to_string(index=False))


if __name__ == "__main__":
    main()
