#!/usr/bin/env python3
"""
Analyze the first ended-beam reseat transition in round6 and compare it
against the no-tree baseline.

Purpose
-------
This script is designed to answer one concrete error-analysis question:

    After the first partial reseat (ended_beam_count in {1,2}),
    does performance drop because newly reseated-branch documents enter the
    top-10, or does it drop even while the ranking is still dominated by the
    old subtree?

The script focuses on the first reseat per query, splits cases into:
    - partial: ended_beam_count in {1,2}
    - medium: ended_beam_count in {3,4,5}
    - large: ended_beam_count >= 6

It then measures:
    - round6 next-step metric changes
    - same-step baseline changes
    - where top-10 documents at t+1 live under the tree
    - DCG contribution by tree-location bucket
    - whether rewrite evidence at t+1 moved with the reseated branch

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
python scripts/analysis/analyze_round6_partial_reseat.py

python scripts/analysis/analyze_round6_partial_reseat.py \
    --out_prefix results/BRIGHT/analysis/round6_partial_reseat
"""

import argparse
import glob
import json
import math
import os
import pickle as pkl
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


DEFAULT_ROUND6_GLOB = "results/BRIGHT/*/round6/**/all_eval_sample_dicts.pkl"
DEFAULT_BASELINE_GLOB = "results/BRIGHT/*/**/leaf_iter_metrics.jsonl"

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


def _has_prefix(path: Sequence[int], prefix: Sequence[int]) -> bool:
    if len(prefix) > len(path):
        return False
    return tuple(path[: len(prefix)]) == tuple(prefix)


def _dcg_weight(rank_idx: int) -> float:
    return 1.0 / math.log2(rank_idx + 2.0)


def _bucket_name(
    path: Sequence[int],
    ended_paths: Sequence[Tuple[int, ...]],
    active_paths: Sequence[Tuple[int, ...]],
    reseated_paths: Sequence[Tuple[int, ...]],
) -> str:
    # Intent: resolve bucket precedence so we can tell whether ranking stays in the old ended subtree or truly moves into reseated branches.
    if any(_has_prefix(path, prefix) for prefix in ended_paths):
        return "old_ended"
    if any(_has_prefix(path, prefix) for prefix in active_paths):
        return "old_active"
    if any(_has_prefix(path, prefix) for prefix in reseated_paths):
        return "new_reseated"
    return "other"


def _summarize_ranked_items(
    ranked_paths: Sequence[Sequence[int]],
    ranked_doc_ids: Sequence[str],
    ended_paths: Sequence[Tuple[int, ...]],
    active_paths: Sequence[Tuple[int, ...]],
    reseated_paths: Sequence[Tuple[int, ...]],
    gold_doc_ids: Iterable[str],
    topk: int,
) -> Dict[str, Any]:
    gold_set = set(gold_doc_ids or [])
    counts = Counter()
    gold_counts = Counter()
    dcg_scores = Counter()
    top_paths = list(ranked_paths[:topk])
    top_docs = list(ranked_doc_ids[:topk])

    for rank_idx, (path, doc_id) in enumerate(zip(top_paths, top_docs)):
        bucket = _bucket_name(path, ended_paths, active_paths, reseated_paths)
        counts[bucket] += 1
        if doc_id in gold_set:
            gold_counts[bucket] += 1
            dcg_scores[bucket] += _dcg_weight(rank_idx)

    dominant_bucket = ""
    if counts:
        dominant_bucket = max(
            ["old_ended", "old_active", "new_reseated", "other"],
            key=lambda key: (counts[key], key),
        )

    out: Dict[str, Any] = {
        "dominant_bucket": dominant_bucket,
        "has_new_reseated": int(counts["new_reseated"] > 0),
    }
    for bucket in ["old_ended", "old_active", "new_reseated", "other"]:
        out[f"{bucket}_count"] = int(counts[bucket])
        out[f"{bucket}_gold_count"] = int(gold_counts[bucket])
        out[f"{bucket}_dcg"] = float(dcg_scores[bucket])
    return out


def _group_name(ended_beam_count: int) -> str:
    if ended_beam_count <= 2:
        return "partial"
    if ended_beam_count <= 5:
        return "medium"
    return "large"


def _safe_float(value: Any) -> float:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value_f


def _first_reseat_record(iter_records: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    # Intent: isolate the first reseat only, because the main performance cliff appears at that transition.
    for rec in iter_records:
        if rec.get("ended_beam_reseat_selected_paths"):
            return rec
        if "ended_reseat" in str(rec.get("selector_pick_reason", "")):
            return rec
    return None


def _mean(values: Sequence[Any]) -> float:
    floats = []
    for value in values:
        value_f = _safe_float(value)
        if math.isnan(value_f):
            continue
        floats.append(value_f)
    if not floats:
        return float("nan")
    return float(sum(floats) / len(floats))


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


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
            first = _first_reseat_record(iter_records)
            if first is None:
                continue

            t = int(first["iter"])
            if (t + 1) >= len(iter_records):
                continue

            curr = iter_records[t]
            nxt = iter_records[t + 1]
            ended_count = int(curr.get("ended_beam_count", 0))
            group = _group_name(ended_count)

            ended_paths = [tuple(path) for path in (curr.get("ended_beam_paths", []) or []) if path]
            selected_before = [tuple(path) for path in (curr.get("selected_branches_before", []) or []) if path]
            active_before = [path for path in selected_before if path not in set(ended_paths)]
            reseated_paths = [
                tuple(row.get("path", []))
                for row in (curr.get("ended_beam_reseat_selected_paths", []) or [])
                if row.get("path")
            ]

            gold_doc_ids = sample.get("gold_doc_ids", []) or curr.get("gold_doc_ids", []) or []

            next_top10 = _summarize_ranked_items(
                ranked_paths=nxt.get("active_eval_paths", []) or [],
                ranked_doc_ids=nxt.get("active_eval_doc_ids", []) or [],
                ended_paths=ended_paths,
                active_paths=active_before,
                reseated_paths=reseated_paths,
                gold_doc_ids=gold_doc_ids,
                topk=topk,
            )
            curr_top10 = _summarize_ranked_items(
                ranked_paths=curr.get("active_eval_paths", []) or [],
                ranked_doc_ids=curr.get("active_eval_doc_ids", []) or [],
                ended_paths=ended_paths,
                active_paths=active_before,
                reseated_paths=reseated_paths,
                gold_doc_ids=gold_doc_ids,
                topk=topk,
            )
            next_prehit = _summarize_ranked_items(
                ranked_paths=nxt.get("pre_hit_paths", []) or [],
                ranked_doc_ids=nxt.get("pre_hit_doc_ids", []) or [],
                ended_paths=ended_paths,
                active_paths=active_before,
                reseated_paths=reseated_paths,
                gold_doc_ids=gold_doc_ids,
                topk=topk,
            )

            curr_doc_ids = list(curr.get("active_eval_doc_ids", [])[:topk])
            next_doc_ids = list(nxt.get("active_eval_doc_ids", [])[:topk])
            curr_path_tuples = [tuple(path) for path in (curr.get("active_eval_paths", []) or [])[:topk]]
            next_path_tuples = [tuple(path) for path in (nxt.get("active_eval_paths", []) or [])[:topk]]

            baseline_curr = baseline_subset.get(query_idx, {}).get(t, {})
            baseline_next = baseline_subset.get(query_idx, {}).get(t + 1, {})

            row: Dict[str, Any] = {
                "subset": subset,
                "query_idx": int(query_idx),
                "group": group,
                "first_reseat_iter": int(t),
                "ended_beam_count": ended_count,
                "selector_pick_reason": str(curr.get("selector_pick_reason", "")),
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
                "top10_doc_overlap_t_to_t1": int(len(set(curr_doc_ids) & set(next_doc_ids))),
                "top10_path_overlap_t_to_t1": int(len(set(curr_path_tuples) & set(next_path_tuples))),
                "query": str(sample.get("original_query") or sample.get("query") or ""),
                "ended_paths_json": _json_dumps(curr.get("ended_beam_paths", []) or []),
                "active_before_paths_json": _json_dumps([list(path) for path in active_before]),
                "reseated_paths_json": _json_dumps(curr.get("ended_beam_reseat_selected_paths", []) or []),
            }

            for prefix, summary in [
                ("curr_top10", curr_top10),
                ("next_top10", next_top10),
                ("next_prehit", next_prehit),
            ]:
                row[f"{prefix}_dominant_bucket"] = str(summary["dominant_bucket"])
                row[f"{prefix}_has_new_reseated"] = int(summary["has_new_reseated"])
                for bucket in ["old_ended", "old_active", "new_reseated", "other"]:
                    row[f"{prefix}_{bucket}_count"] = int(summary[f"{bucket}_count"])
                    row[f"{prefix}_{bucket}_gold_count"] = int(summary[f"{bucket}_gold_count"])
                    row[f"{prefix}_{bucket}_dcg"] = float(summary[f"{bucket}_dcg"])

            rows.append(row)

    return pd.DataFrame(rows)


def summarize_rows(rows_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    ]
    for prefix in ["curr_top10", "next_top10", "next_prehit"]:
        for bucket in ["old_ended", "old_active", "new_reseated", "other"]:
            metric_cols.append(f"{prefix}_{bucket}_count")
            metric_cols.append(f"{prefix}_{bucket}_gold_count")
            metric_cols.append(f"{prefix}_{bucket}_dcg")

    def _agg(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=group_cols + ["n"])
        summary = df.groupby(group_cols, dropna=False).agg(
            n=("query_idx", "count"),
            **{col: (col, "mean") for col in metric_cols},
        )
        return summary.reset_index()

    group_df = _agg(rows_df, ["group"])
    subset_df = _agg(rows_df, ["subset", "group"])
    return group_df, subset_df


def build_examples(rows_df: pd.DataFrame, limit_per_group: int) -> pd.DataFrame:
    if rows_df.empty:
        return rows_df
    partial = rows_df[rows_df["group"] == "partial"].copy()
    partial = partial.sort_values(
        by=["round6_ndcg_delta", "round6_ndcg_t1", "subset", "query_idx"],
        ascending=[True, True, True, True],
    )
    return partial.head(limit_per_group).reset_index(drop=True)


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
        help="All substrings that must appear in the round6 path.",
    )
    parser.add_argument(
        "--baseline_require_substrings",
        nargs="*",
        default=DEFAULT_BASELINE_REQUIRE,
        help="All substrings that must appear in the baseline path.",
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
        default="results/BRIGHT/analysis/round6_partial_reseat",
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
    group_df, subset_df = summarize_rows(rows_df)
    examples_df = build_examples(rows_df, limit_per_group=int(args.examples_limit))

    out_dir = os.path.dirname(os.path.abspath(args.out_prefix))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    rows_path = f"{args.out_prefix}_rows.csv"
    group_path = f"{args.out_prefix}_group_summary.csv"
    subset_path = f"{args.out_prefix}_subset_summary.csv"
    examples_path = f"{args.out_prefix}_examples.csv"

    rows_df.to_csv(rows_path, index=False)
    group_df.to_csv(group_path, index=False)
    subset_df.to_csv(subset_path, index=False)
    examples_df.to_csv(examples_path, index=False)

    print(f"[saved] {rows_path}")
    print(f"[saved] {group_path}")
    print(f"[saved] {subset_path}")
    print(f"[saved] {examples_path}")

    if not group_df.empty:
        print("\n[group summary]")
        print(group_df.to_string(index=False))


if __name__ == "__main__":
    main()
