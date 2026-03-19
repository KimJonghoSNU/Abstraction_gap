#!/usr/bin/env python3
"""
Analyze whether round6 partial reseat causes true off-region derailment or
same-region ranking collapse.

Purpose
-------
This script answers one concrete question:

    When round6 transitions after a leaf-trigger / ended-beam reseat, does the
    next iteration move retrieval outside the previously selected gold-aligned
    branch region, or does it stay in that region and still lose ranking
    quality?

The canonical derailment signal is:
    - iter t selected at least one gold-ancestor branch
    - at iter t+1, fewer than half of top-k active-eval paths remain inside the
      subtree of those gold-aligned branches

The script also measures:
    - any-survival and pre-hit variants of derailment
    - on-region vs off-region gold counts and DCG contributions
    - baseline same-step metric deltas for a no-tree reference

Default targets
---------------
Round6:
    round6_mrr_selector_accum_meanscore_global_expandable_ended_reseat
    with agent_executor_v1_icl2

Baseline:
    baseline3_leaf_only_loop
    with agent_executor_v1_icl2

Examples
--------
python scripts/analysis/analyze_round6_derailment.py

python scripts/analysis/analyze_round6_derailment.py \
    --out_prefix results/BRIGHT/analysis/round6_derailment
"""

import argparse
import glob
import json
import math
import os
import pickle as pkl
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


DEFAULT_ROUND6_GLOB = "results/BRIGHT/*/round6/**/all_eval_sample_dicts.pkl"
DEFAULT_BASELINE_GLOB = "results/BRIGHT/**/leaf_iter_metrics.jsonl"

DEFAULT_ROUND6_REQUIRE = [
    "round6_mrr_selector_accum_meanscore_global_expandable_ended_reseat",
    "agent_executor_v1_icl2",
]
DEFAULT_BASELINE_REQUIRE = [
    "baseline3_leaf_only_loop",
    "agent_executor_v1_icl2",
]

BUCKETS = ["on_region", "off_region"]


def _infer_subset_from_path(path: str) -> str:
    parts = os.path.abspath(path).split(os.sep)
    for idx, part in enumerate(parts):
        if part == "results" and (idx + 2) < len(parts):
            return str(parts[idx + 2])
    return "unknown"


def _resolve_paths(
    inputs: Sequence[str],
    glob_pattern: str,
    require_substrings: Sequence[str],
) -> List[str]:
    resolved: List[str] = []

    for path in inputs:
        if not path:
            continue
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Input path not found: {abs_path}")
        resolved.append(abs_path)

    if glob_pattern:
        resolved.extend(glob.glob(glob_pattern, recursive=True))

    filtered: List[str] = []
    for path in sorted({os.path.abspath(x) for x in resolved}):
        if require_substrings and any(token not in path for token in require_substrings):
            continue
        filtered.append(path)

    if not filtered:
        raise FileNotFoundError("No files matched the given filters.")
    return filtered


def _load_baseline_metrics(paths: Sequence[str]) -> Dict[str, Dict[int, Dict[int, Dict[str, Any]]]]:
    baseline_by_subset: Dict[str, Dict[int, Dict[int, Dict[str, Any]]]] = {}
    for path in paths:
        subset = _infer_subset_from_path(path)
        by_query: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                by_query[int(row["query_idx"])][int(row["iter"])] = row
        baseline_by_subset[subset] = dict(by_query)
    return baseline_by_subset


def _load_round6_samples(paths: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    round6_by_subset: Dict[str, List[Dict[str, Any]]] = {}
    for path in paths:
        subset = _infer_subset_from_path(path)
        with open(path, "rb") as f:
            round6_by_subset[subset] = pkl.load(f)
    return round6_by_subset


def _safe_float(value: Any) -> float:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value_f


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _has_prefix(path: Sequence[int], prefix: Sequence[int]) -> bool:
    if len(prefix) > len(path):
        return False
    return tuple(path[: len(prefix)]) == tuple(prefix)


def _branch_is_gold_ancestor(branch_path: Sequence[int], gold_paths: Sequence[Sequence[int]]) -> bool:
    return any(_has_prefix(gold_path, branch_path) for gold_path in gold_paths)


def _gold_anchor_branches(
    selected_paths: Sequence[Sequence[int]],
    gold_paths: Sequence[Sequence[int]],
) -> List[Tuple[int, ...]]:
    # Intent: anchor derailment on the exact gold-aligned branches the system selected at iter t.
    anchors = [tuple(path) for path in selected_paths if _branch_is_gold_ancestor(path, gold_paths)]
    return sorted({tuple(path) for path in anchors})


def _dcg_weight(rank_idx: int) -> float:
    return 1.0 / math.log2(rank_idx + 2.0)


def _summarize_region(
    ranked_paths: Sequence[Sequence[int]],
    ranked_doc_ids: Sequence[str],
    anchor_branches: Sequence[Tuple[int, ...]],
    gold_doc_ids: Iterable[str],
    topk: int,
) -> Dict[str, Any]:
    gold_set = set(str(x) for x in (gold_doc_ids or []))
    counts = {bucket: 0 for bucket in BUCKETS}
    gold_counts = {bucket: 0 for bucket in BUCKETS}
    dcg_scores = {bucket: 0.0 for bucket in BUCKETS}

    top_paths = list(ranked_paths[:topk])
    top_docs = list(ranked_doc_ids[:topk])
    for rank_idx, (path, doc_id) in enumerate(zip(top_paths, top_docs)):
        bucket = "on_region" if any(_has_prefix(path, anchor) for anchor in anchor_branches) else "off_region"
        counts[bucket] += 1
        if str(doc_id) in gold_set:
            gold_counts[bucket] += 1
            dcg_scores[bucket] += _dcg_weight(rank_idx)

    total = max(1, len(top_paths))
    out: Dict[str, Any] = {
        "on_region_share": float(counts["on_region"] / total),
        "off_region_share": float(counts["off_region"] / total),
        "derailed_majority": int(counts["on_region"] < math.ceil(topk / 2.0)),
        "derailed_anymiss": int(counts["on_region"] == 0),
    }
    for bucket in BUCKETS:
        out[f"{bucket}_count"] = int(counts[bucket])
        out[f"{bucket}_gold_count"] = int(gold_counts[bucket])
        out[f"{bucket}_dcg"] = float(dcg_scores[bucket])
    return out


def _best_gold_rank(doc_ids: Sequence[str], gold_doc_ids: Sequence[str]) -> Optional[int]:
    gold_set = set(str(x) for x in (gold_doc_ids or []))
    for rank_idx, doc_id in enumerate(doc_ids):
        if str(doc_id) in gold_set:
            return int(rank_idx + 1)
    return None


def _mean(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(series.mean())


def _is_reseat_event(rec: Dict[str, Any]) -> bool:
    if rec.get("ended_beam_reseat_selected_paths"):
        return True
    return "ended_reseat" in str(rec.get("selector_pick_reason", ""))


def _reseat_group(rec: Dict[str, Any]) -> str:
    if not _is_reseat_event(rec):
        return "non_reseat"
    ended_count = int(rec.get("ended_beam_count", 0))
    if ended_count <= 2:
        return "partial_reseat"
    if ended_count <= 5:
        return "medium_reseat"
    return "large_reseat"


def build_rows(
    round6_by_subset: Dict[str, List[Dict[str, Any]]],
    baseline_by_subset: Dict[str, Dict[int, Dict[int, Dict[str, Any]]]],
    topk: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for subset, samples in round6_by_subset.items():
        baseline_subset = baseline_by_subset.get(subset, {})
        for query_idx, sample in enumerate(samples):
            iter_records = sample.get("iter_records", []) or []
            gold_paths = [tuple(path) for path in (sample.get("gold_paths", []) or []) if path]
            gold_doc_ids = [str(x) for x in (sample.get("gold_doc_ids", []) or [])]
            if not gold_paths:
                continue

            for t in range(len(iter_records) - 1):
                curr = iter_records[t]
                nxt = iter_records[t + 1]
                selected_after = [tuple(path) for path in (curr.get("selected_branches_after", []) or []) if path]
                anchor_branches = _gold_anchor_branches(selected_after, gold_paths)
                if not anchor_branches:
                    continue

                curr_top10 = _summarize_region(
                    ranked_paths=curr.get("active_eval_paths", []) or [],
                    ranked_doc_ids=curr.get("active_eval_doc_ids", []) or [],
                    anchor_branches=anchor_branches,
                    gold_doc_ids=gold_doc_ids,
                    topk=topk,
                )
                next_top10 = _summarize_region(
                    ranked_paths=nxt.get("active_eval_paths", []) or [],
                    ranked_doc_ids=nxt.get("active_eval_doc_ids", []) or [],
                    anchor_branches=anchor_branches,
                    gold_doc_ids=gold_doc_ids,
                    topk=topk,
                )
                next_prehit = _summarize_region(
                    ranked_paths=nxt.get("pre_hit_paths", []) or [],
                    ranked_doc_ids=nxt.get("pre_hit_doc_ids", []) or [],
                    anchor_branches=anchor_branches,
                    gold_doc_ids=gold_doc_ids,
                    topk=topk,
                )

                curr_doc_ids = list(curr.get("active_eval_doc_ids", []) or [])
                next_doc_ids = list(nxt.get("active_eval_doc_ids", []) or [])
                curr_path_tuples = [tuple(path) for path in (curr.get("active_eval_paths", []) or [])[:topk]]
                next_path_tuples = [tuple(path) for path in (nxt.get("active_eval_paths", []) or [])[:topk]]

                baseline_curr = baseline_subset.get(query_idx, {}).get(t, {})
                baseline_next = baseline_subset.get(query_idx, {}).get(t + 1, {})

                row: Dict[str, Any] = {
                    "subset": subset,
                    "query_idx": int(query_idx),
                    "iter": int(t),
                    "query": str(sample.get("original_query") or sample.get("query") or ""),
                    "anchor_gold_branch_count_t": int(len(anchor_branches)),
                    "anchor_gold_branches_json": _json_dumps([list(path) for path in anchor_branches]),
                    "selected_branches_after_json": _json_dumps(curr.get("selected_branches_after", []) or []),
                    "ended_beam_count": int(curr.get("ended_beam_count", 0)),
                    "reseat_group": _reseat_group(curr),
                    "is_reseat": int(_is_reseat_event(curr)),
                    "round6_ndcg_t": _safe_float(curr.get("metrics", {}).get("nDCG@10")),
                    "round6_ndcg_t1": _safe_float(nxt.get("metrics", {}).get("nDCG@10")),
                    "round6_ndcg_delta": _safe_float(nxt.get("metrics", {}).get("nDCG@10")) - _safe_float(curr.get("metrics", {}).get("nDCG@10")),
                    "round6_recall10_t": _safe_float(curr.get("metrics", {}).get("Recall@10")),
                    "round6_recall10_t1": _safe_float(nxt.get("metrics", {}).get("Recall@10")),
                    "round6_recall100_t": _safe_float(curr.get("metrics", {}).get("Recall@100")),
                    "round6_recall100_t1": _safe_float(nxt.get("metrics", {}).get("Recall@100")),
                    "round6_coverage_t": _safe_float(curr.get("metrics", {}).get("Coverage")),
                    "round6_coverage_t1": _safe_float(nxt.get("metrics", {}).get("Coverage")),
                    "round6_branchhit_t": _safe_float(curr.get("metrics", {}).get("BranchHit@B")),
                    "round6_branchhit_t1": _safe_float(nxt.get("metrics", {}).get("BranchHit@B")),
                    "baseline_ndcg_t": _safe_float(baseline_curr.get("nDCG@10")),
                    "baseline_ndcg_t1": _safe_float(baseline_next.get("nDCG@10")),
                    "baseline_ndcg_delta": _safe_float(baseline_next.get("nDCG@10")) - _safe_float(baseline_curr.get("nDCG@10")),
                    "top10_doc_overlap_t_to_t1": int(len(set(curr_doc_ids[:topk]) & set(next_doc_ids[:topk]))),
                    "top10_path_overlap_t_to_t1": int(len(set(curr_path_tuples) & set(next_path_tuples))),
                    "best_gold_rank_t": _best_gold_rank(curr_doc_ids, gold_doc_ids),
                    "best_gold_rank_t1": _best_gold_rank(next_doc_ids, gold_doc_ids),
                }

                for prefix, summary in [
                    ("curr_top10", curr_top10),
                    ("next_top10", next_top10),
                    ("next_prehit", next_prehit),
                ]:
                    row[f"{prefix}_on_region_share"] = float(summary["on_region_share"])
                    row[f"{prefix}_off_region_share"] = float(summary["off_region_share"])
                    row[f"{prefix}_derailed_majority"] = int(summary["derailed_majority"])
                    row[f"{prefix}_derailed_anymiss"] = int(summary["derailed_anymiss"])
                    for bucket in BUCKETS:
                        row[f"{prefix}_{bucket}_count"] = int(summary[f"{bucket}_count"])
                        row[f"{prefix}_{bucket}_gold_count"] = int(summary[f"{bucket}_gold_count"])
                        row[f"{prefix}_{bucket}_dcg"] = float(summary[f"{bucket}_dcg"])

                rows.append(row)

    return pd.DataFrame(rows)


def _summarize_scope(rows_df: pd.DataFrame, scope_name: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "scope": scope_name,
        "n": int(len(rows_df)),
    }
    if rows_df.empty:
        return out

    metric_cols = [
        "round6_ndcg_t",
        "round6_ndcg_t1",
        "round6_ndcg_delta",
        "baseline_ndcg_t",
        "baseline_ndcg_t1",
        "baseline_ndcg_delta",
        "round6_recall10_t",
        "round6_recall10_t1",
        "round6_recall100_t",
        "round6_recall100_t1",
        "round6_coverage_t",
        "round6_coverage_t1",
        "round6_branchhit_t",
        "round6_branchhit_t1",
        "top10_doc_overlap_t_to_t1",
        "top10_path_overlap_t_to_t1",
        "anchor_gold_branch_count_t",
        "ended_beam_count",
    ]
    for col in metric_cols:
        out[col] = _mean(rows_df[col])

    for prefix in ["curr_top10", "next_top10", "next_prehit"]:
        for suffix in [
            "on_region_share",
            "off_region_share",
            "derailed_majority",
            "derailed_anymiss",
            "on_region_count",
            "off_region_count",
            "on_region_gold_count",
            "off_region_gold_count",
            "on_region_dcg",
            "off_region_dcg",
        ]:
            out[f"{prefix}_{suffix}"] = _mean(rows_df[f"{prefix}_{suffix}"])

    out["best_gold_rank_t_mean"] = _mean(rows_df["best_gold_rank_t"].dropna())
    out["best_gold_rank_t1_mean"] = _mean(rows_df["best_gold_rank_t1"].dropna())
    return out


def build_summary(rows_df: pd.DataFrame) -> pd.DataFrame:
    scopes = {
        "all_anchor_transitions": rows_df,
        "all_reseat": rows_df[rows_df["is_reseat"] == 1],
        "non_reseat": rows_df[rows_df["is_reseat"] == 0],
        "partial_reseat": rows_df[rows_df["reseat_group"] == "partial_reseat"],
        "medium_reseat": rows_df[rows_df["reseat_group"] == "medium_reseat"],
        "large_reseat": rows_df[rows_df["reseat_group"] == "large_reseat"],
    }
    return pd.DataFrame([_summarize_scope(df, scope) for scope, df in scopes.items()])


def build_by_subset(rows_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if rows_df.empty:
        return pd.DataFrame()
    for subset, subset_df in rows_df.groupby("subset", dropna=False):
        rows.append(_summarize_scope(subset_df, f"{subset}:all_anchor_transitions"))
        partial_df = subset_df[subset_df["reseat_group"] == "partial_reseat"]
        rows.append(_summarize_scope(partial_df, f"{subset}:partial_reseat"))
    return pd.DataFrame(rows)


def build_examples(rows_df: pd.DataFrame, limit: int) -> pd.DataFrame:
    if rows_df.empty:
        return rows_df
    partial_df = rows_df[rows_df["reseat_group"] == "partial_reseat"].copy()
    if partial_df.empty:
        return partial_df
    partial_df = partial_df.sort_values(
        by=["round6_ndcg_delta", "round6_ndcg_t1", "subset", "query_idx", "iter"],
        ascending=[True, True, True, True, True],
    )
    keep_cols = [
        "subset",
        "query_idx",
        "iter",
        "query",
        "ended_beam_count",
        "anchor_gold_branches_json",
        "round6_ndcg_t",
        "round6_ndcg_t1",
        "round6_ndcg_delta",
        "baseline_ndcg_t",
        "baseline_ndcg_t1",
        "baseline_ndcg_delta",
        "curr_top10_on_region_count",
        "curr_top10_on_region_gold_count",
        "curr_top10_on_region_dcg",
        "next_top10_on_region_count",
        "next_top10_on_region_gold_count",
        "next_top10_on_region_dcg",
        "next_top10_off_region_count",
        "next_top10_off_region_gold_count",
        "next_top10_off_region_dcg",
        "next_prehit_on_region_count",
        "next_prehit_on_region_gold_count",
        "next_prehit_on_region_dcg",
        "next_prehit_off_region_count",
        "next_prehit_off_region_gold_count",
        "next_prehit_off_region_dcg",
        "top10_doc_overlap_t_to_t1",
        "top10_path_overlap_t_to_t1",
        "best_gold_rank_t",
        "best_gold_rank_t1",
    ]
    return partial_df[keep_cols].head(limit).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--round6_glob_pattern",
        default=DEFAULT_ROUND6_GLOB,
        help="Glob for round6 all_eval_sample_dicts.pkl files.",
    )
    parser.add_argument(
        "--baseline_glob_pattern",
        default=DEFAULT_BASELINE_GLOB,
        help="Glob for baseline leaf_iter_metrics.jsonl files.",
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
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Top-k rank window to analyze.",
    )
    parser.add_argument(
        "--examples_limit",
        type=int,
        default=50,
        help="Number of worst partial-reseat examples to save.",
    )
    parser.add_argument(
        "--out_prefix",
        default="results/BRIGHT/analysis/round6_derailment",
        help="Output prefix for CSV files.",
    )
    args = parser.parse_args()

    round6_paths = _resolve_paths(
        inputs=[],
        glob_pattern=args.round6_glob_pattern,
        require_substrings=args.round6_require_substrings,
    )
    baseline_paths = _resolve_paths(
        inputs=[],
        glob_pattern=args.baseline_glob_pattern,
        require_substrings=args.baseline_require_substrings,
    )

    round6_by_subset = _load_round6_samples(round6_paths)
    baseline_by_subset = _load_baseline_metrics(baseline_paths)

    rows_df = build_rows(
        round6_by_subset=round6_by_subset,
        baseline_by_subset=baseline_by_subset,
        topk=int(args.topk),
    )
    summary_df = build_summary(rows_df)
    partial_summary_df = summary_df[summary_df["scope"] == "partial_reseat"].reset_index(drop=True)
    by_subset_df = build_by_subset(rows_df)
    examples_df = build_examples(rows_df, limit=int(args.examples_limit))

    out_dir = os.path.dirname(os.path.abspath(args.out_prefix))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    rows_path = f"{args.out_prefix}_rows.csv"
    summary_path = f"{args.out_prefix}_summary.csv"
    partial_summary_path = f"{args.out_prefix}_partial_reseat_summary.csv"
    by_subset_path = f"{args.out_prefix}_by_subset.csv"
    examples_path = f"{args.out_prefix}_examples.csv"

    rows_df.to_csv(rows_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    partial_summary_df.to_csv(partial_summary_path, index=False)
    by_subset_df.to_csv(by_subset_path, index=False)
    examples_df.to_csv(examples_path, index=False)

    print(f"[saved] {rows_path}")
    print(f"[saved] {summary_path}")
    print(f"[saved] {partial_summary_path}")
    print(f"[saved] {by_subset_path}")
    print(f"[saved] {examples_path}")

    if not summary_df.empty:
        print("\n[summary]")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
