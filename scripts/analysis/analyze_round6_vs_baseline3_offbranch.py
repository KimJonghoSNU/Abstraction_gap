#!/usr/bin/env python3
"""
Compare off-branch noisy feedback between round6 ended-reseat and baseline3.

Purpose
-------
This script compares two runs under one aligned question:

    When top-k retrieval results that will become next-step corpus feedback
    include many off-branch documents, how much does next-iteration nDCG drop?

The comparison is not perfectly symmetric:
    - round6 uses real selected branch state (`selected_branches_before`)
    - baseline3 uses a proxy branch state built from previous
      `rewrite_context_paths`

That caveat is intentional and should be reported in the write-up.

Main metric
-----------
Off-branch (%):
    fraction of retrieval top-k documents that are not descendants of the
    current branch reference.

nDCG drop:
    next-step delta `nDCG@10(t+1) - nDCG@10(t)`

Examples
--------
python scripts/analysis/analyze_round6_vs_baseline3_offbranch.py

python scripts/analysis/analyze_round6_vs_baseline3_offbranch.py \
    --out_prefix results/BRIGHT/analysis/round6_vs_baseline3_offbranch
"""

import argparse
import glob
import json
import math
import os
import pickle as pkl
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd


DEFAULT_ROUND6_GLOB = "results/BRIGHT/*/round6/**/all_eval_sample_dicts.pkl"
DEFAULT_BASELINE_RECORDS_GLOB = "results/BRIGHT/**/leaf_iter_records.jsonl"

DEFAULT_ROUND6_REQUIRE = [
    "round6_mrr_selector_accum_meanscore_global_expandable_ended_reseat",
    "agent_executor_v1_icl2",
]
DEFAULT_BASELINE_REQUIRE = [
    "baseline3_leaf_only_loop",
    "agent_executor_v1_icl2",
]


def _infer_subset_from_path(path: str) -> str:
    parts = os.path.abspath(path).split(os.sep)
    for idx, part in enumerate(parts):
        if part == "results" and (idx + 2) < len(parts):
            return str(parts[idx + 2])
    return "unknown"


def _resolve_paths(glob_pattern: str, require_substrings: Sequence[str]) -> List[str]:
    resolved = glob.glob(glob_pattern, recursive=True)
    filtered: List[str] = []
    for path in sorted({os.path.abspath(x) for x in resolved}):
        if require_substrings and any(token not in path for token in require_substrings):
            continue
        filtered.append(path)
    if not filtered:
        raise FileNotFoundError(f"No files matched glob={glob_pattern} require={list(require_substrings)}")
    return filtered


def _is_prefix(prefix: Sequence[int], full: Sequence[int]) -> bool:
    p = tuple(prefix)
    f = tuple(full)
    return len(p) <= len(f) and f[: len(p)] == p


def _is_descendant_of_selected(path: Sequence[int], selected_branches: Sequence[Sequence[int]]) -> bool:
    return any(_is_prefix(branch, path) for branch in selected_branches)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _safe_mean(series: pd.Series) -> float:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return float("nan")
    return float(valid.mean())


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    joined = pd.concat([a, b], axis=1).dropna()
    if len(joined) < 2:
        return float("nan")
    return float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))


def _context_prefix_branches(context_paths: Sequence[Tuple[int, ...]], depth: int) -> List[Tuple[int, ...]]:
    branches: List[Tuple[int, ...]] = []
    seen = set()
    for path in context_paths:
        if len(path) < 2:
            continue
        d = min(int(depth), len(path) - 1)
        if d <= 0:
            continue
        branch = tuple(path[:d])
        if branch in seen:
            continue
        seen.add(branch)
        branches.append(branch)
    return branches


def _off_branch_pct(
    ranked_paths: Sequence[Sequence[int]],
    selected_branches: Sequence[Sequence[int]],
    top_k: int,
) -> float:
    top_paths = [tuple(path) for path in (ranked_paths or [])[: int(top_k)] if path]
    if (not selected_branches) or (not top_paths):
        return float("nan")
    off_count = 0
    for path in top_paths:
        if not _is_descendant_of_selected(path, selected_branches):
            off_count += 1
    return 100.0 * float(off_count) / float(len(top_paths))


def _load_round6_rows(paths: Sequence[str], top_k: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        subset = _infer_subset_from_path(path)
        with open(path, "rb") as f:
            samples = pkl.load(f)
        for sample_idx, sample in enumerate(samples):
            iter_records = sample.get("iter_records", []) or []
            for iter_idx in range(len(iter_records) - 1):
                rec = iter_records[iter_idx]
                next_rec = iter_records[iter_idx + 1]
                selected_before = [tuple(x) for x in (rec.get("selected_branches_before", []) or []) if x]
                off_feedback_pct = _off_branch_pct(
                    ranked_paths=rec.get("pre_hit_paths", []) or [],
                    selected_branches=selected_before,
                    top_k=top_k,
                )
                off_eval_pct = _off_branch_pct(
                    ranked_paths=rec.get("active_eval_paths", []) or [],
                    selected_branches=selected_before,
                    top_k=top_k,
                )
                rows.append(
                    {
                        "method": "round6_ended_reseat",
                        "subset": subset,
                        "run_path": path,
                        "query_idx": int(sample_idx),
                        "iter": int(iter_idx),
                        # Intent: use pre-hit retrieval as the main feedback metric because these docs become next-step rewrite feedback.
                        "off_branch_pct_feedback": float(off_feedback_pct),
                        "off_branch_pct_eval": float(off_eval_pct),
                        "off_event_feedback": bool((not math.isnan(off_feedback_pct)) and (off_feedback_pct > 0.0)),
                        "ndcg10_t": _safe_float(rec.get("metrics", {}).get("nDCG@10")),
                        "ndcg10_t1": _safe_float(next_rec.get("metrics", {}).get("nDCG@10")),
                        "ndcg10_delta_t1_minus_t": _safe_float(next_rec.get("metrics", {}).get("nDCG@10"))
                        - _safe_float(rec.get("metrics", {}).get("nDCG@10")),
                        "selected_branch_count": int(len(selected_before)),
                    }
                )
    return pd.DataFrame(rows)


def _load_baseline_rows(paths: Sequence[str], top_k: int, context_branch_depth: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for records_path in paths:
        metrics_path = records_path.replace("leaf_iter_records.jsonl", "leaf_iter_metrics.jsonl")
        if not os.path.exists(metrics_path):
            continue
        subset = _infer_subset_from_path(records_path)

        by_q: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
        with open(records_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                qidx = int(rec.get("query_idx", -1))
                if qidx < 0:
                    continue
                phase = str(rec.get("phase", ""))
                if phase == "initial_rewrite":
                    by_q[qidx][-1] = rec
                elif phase == "iter_retrieval":
                    iter_idx = int(rec.get("iter", -1))
                    if iter_idx >= 0:
                        by_q[qidx][iter_idx] = rec

        ndcg_by_q: Dict[int, Dict[int, float]] = defaultdict(dict)
        with open(metrics_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                qidx = int(rec.get("query_idx", -1))
                iter_idx = int(rec.get("iter", -1))
                ndcg = float(rec.get("nDCG@10", math.nan))
                if qidx >= 0 and iter_idx >= 0:
                    ndcg_by_q[qidx][iter_idx] = ndcg

        for qidx, by_iter in by_q.items():
            iter_keys = sorted([k for k in by_iter.keys() if k >= 0])
            for iter_idx in iter_keys:
                prev = by_iter.get(iter_idx - 1)
                cur = by_iter.get(iter_idx)
                if prev is None or cur is None:
                    continue

                prev_context_paths = [tuple(x) for x in (prev.get("rewrite_context_paths", []) or []) if x]
                selected_proxy = _context_prefix_branches(prev_context_paths, depth=int(context_branch_depth))
                off_feedback_pct = _off_branch_pct(
                    ranked_paths=cur.get("retrieved_paths", []) or [],
                    selected_branches=selected_proxy,
                    top_k=top_k,
                )
                ndcg_t = _safe_float(ndcg_by_q.get(qidx, {}).get(iter_idx))
                ndcg_t1 = _safe_float(ndcg_by_q.get(qidx, {}).get(iter_idx + 1))
                rows.append(
                    {
                        "method": "baseline3_leaf_only_proxy",
                        "subset": subset,
                        "run_path": records_path,
                        "query_idx": int(qidx),
                        "iter": int(iter_idx),
                        "off_branch_pct_feedback": float(off_feedback_pct),
                        "off_branch_pct_eval": math.nan,
                        "off_event_feedback": bool((not math.isnan(off_feedback_pct)) and (off_feedback_pct > 0.0)),
                        "ndcg10_t": ndcg_t,
                        "ndcg10_t1": ndcg_t1,
                        "ndcg10_delta_t1_minus_t": ndcg_t1 - ndcg_t,
                        "selected_branch_count": int(len(selected_proxy)),
                    }
                )
    return pd.DataFrame(rows)


def _summarize(rows_df: pd.DataFrame) -> pd.DataFrame:
    out_rows: List[Dict[str, Any]] = []
    group_cols = ["method", "subset"]
    for (method, subset), grp in rows_df.groupby(group_cols, dropna=False):
        off = grp[grp["off_event_feedback"]]
        on = grp[~grp["off_event_feedback"]]
        d_off = _safe_mean(off["ndcg10_delta_t1_minus_t"])
        d_on = _safe_mean(on["ndcg10_delta_t1_minus_t"])
        out_rows.append(
            {
                "method": method,
                "subset": subset,
                "num_rows": int(len(grp)),
                "OffBranchPct@K_mean_feedback": _safe_mean(grp["off_branch_pct_feedback"]),
                "OffBranchEventRate(>0%)_feedback": 100.0 * _safe_mean(grp["off_event_feedback"].astype(float)),
                "nDCG10_mean": _safe_mean(grp["ndcg10_t"]),
                "nDCG10_delta_mean_all": _safe_mean(grp["ndcg10_delta_t1_minus_t"]),
                "nDCG10_delta_mean_when_off_feedback": d_off,
                "nDCG10_delta_mean_when_onbranch_only_feedback": d_on,
                # Intent: negative means off-branch feedback is associated with larger next-iter nDCG decrease.
                "nDCG10_drop_delta_feedback(off-on)": d_off - d_on,
                "corr(off_feedback_pct, ndcg_delta)": _safe_corr(
                    grp["off_branch_pct_feedback"],
                    grp["ndcg10_delta_t1_minus_t"],
                ),
            }
        )
    return pd.DataFrame(out_rows).sort_values(["subset", "method"]).reset_index(drop=True)


def _overall(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for method, grp in summary_df.groupby("method", dropna=False):
        rows.append(
            {
                "method": method,
                "num_subsets": int(len(grp)),
                "OffBranchPct@K_mean_feedback_macro": _safe_mean(grp["OffBranchPct@K_mean_feedback"]),
                "OffBranchEventRate_macro": _safe_mean(grp["OffBranchEventRate(>0%)_feedback"]),
                "nDCG10_delta_mean_all_macro": _safe_mean(grp["nDCG10_delta_mean_all"]),
                "nDCG10_delta_mean_when_off_feedback_macro": _safe_mean(grp["nDCG10_delta_mean_when_off_feedback"]),
                "nDCG10_delta_mean_when_onbranch_only_feedback_macro": _safe_mean(grp["nDCG10_delta_mean_when_onbranch_only_feedback"]),
                "nDCG10_drop_delta_feedback(off-on)_macro": _safe_mean(grp["nDCG10_drop_delta_feedback(off-on)"]),
                "corr(off_feedback_pct, ndcg_delta)_macro": _safe_mean(grp["corr(off_feedback_pct, ndcg_delta)"]),
            }
        )
    return pd.DataFrame(rows).sort_values("method").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--round6_glob_pattern",
        type=str,
        default=DEFAULT_ROUND6_GLOB,
        help="Glob for round6 all_eval_sample_dicts.pkl files.",
    )
    parser.add_argument(
        "--baseline_records_glob",
        type=str,
        default=DEFAULT_BASELINE_RECORDS_GLOB,
        help="Glob for baseline leaf_iter_records.jsonl files.",
    )
    parser.add_argument(
        "--round6_require_substrings",
        nargs="*",
        default=DEFAULT_ROUND6_REQUIRE,
        help="All substrings that must appear in round6 paths.",
    )
    parser.add_argument(
        "--baseline_require_substrings",
        nargs="*",
        default=DEFAULT_BASELINE_REQUIRE,
        help="All substrings that must appear in baseline paths.",
    )
    parser.add_argument("--top_k", type=int, default=10, help="Top-K cutoff for off-branch ratio.")
    parser.add_argument(
        "--context_branch_depth",
        type=int,
        default=3,
        help="Baseline proxy branch prefix depth derived from previous rewrite context.",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="results/BRIGHT/analysis/round6_vs_baseline3_offbranch",
        help="Output prefix for CSVs.",
    )
    args = parser.parse_args()

    round6_paths = _resolve_paths(args.round6_glob_pattern, args.round6_require_substrings)
    baseline_paths = _resolve_paths(args.baseline_records_glob, args.baseline_require_substrings)

    round6_rows = _load_round6_rows(round6_paths, top_k=int(args.top_k))
    baseline_rows = _load_baseline_rows(
        baseline_paths,
        top_k=int(args.top_k),
        context_branch_depth=int(args.context_branch_depth),
    )
    rows_df = pd.concat([round6_rows, baseline_rows], ignore_index=True)

    summary_df = _summarize(rows_df)
    overall_df = _overall(summary_df)

    out_dir = os.path.dirname(os.path.abspath(args.out_prefix))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    rows_path = f"{args.out_prefix}_rows.csv"
    summary_path = f"{args.out_prefix}_summary.csv"
    overall_path = f"{args.out_prefix}_overall.csv"

    rows_df.to_csv(rows_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    overall_df.to_csv(overall_path, index=False)

    print(f"[saved] {rows_path}")
    print(f"[saved] {summary_path}")
    print(f"[saved] {overall_path}")
    print("\n[overall]")
    print(overall_df.to_string(index=False))


if __name__ == "__main__":
    main()
