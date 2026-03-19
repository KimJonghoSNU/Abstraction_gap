#!/usr/bin/env python3
"""
Analyze whether round6 reseat chooses gold-return branches and whether those
branches survive in retrieval state until the end of the trajectory.

Purpose
-------
This script answers two questions about reseat events:

    1. Does reseat actually choose a branch whose descendants contain gold?
    2. If yes, does that gold-return branch remain alive in later retrieval
       state, or does it disappear before the end iteration?

The analysis is event-based. For each reseat event at iter t, we inspect:
    - reseated branch set at t
    - whether any reseated branch is a gold ancestor
    - future active-eval and pre-hit region survival from t+1 until the final iter
    - final and future performance relative to the reseat point

Examples
--------
python scripts/analysis/analyze_round6_reseat_return.py

python scripts/analysis/analyze_round6_reseat_return.py \
    --out_prefix results/BRIGHT/analysis/round6_reseat_return
"""

import argparse
import glob
import json
import math
import os
import pickle as pkl
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


DEFAULT_ROUND6_GLOB = "results/BRIGHT/*/round6/**/all_eval_sample_dicts.pkl"
DEFAULT_ROUND6_REQUIRE = [
    "round6_mrr_selector_accum_meanscore_global_expandable_ended_reseat",
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


def _load_round6_samples(paths: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    round6_by_subset: Dict[str, List[Dict[str, Any]]] = {}
    for path in paths:
        subset = _infer_subset_from_path(path)
        with open(path, "rb") as f:
            round6_by_subset[subset] = pkl.load(f)
    return round6_by_subset


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _has_prefix(path: Sequence[int], prefix: Sequence[int]) -> bool:
    if len(prefix) > len(path):
        return False
    return tuple(path[: len(prefix)]) == tuple(prefix)


def _branch_is_gold_ancestor(branch_path: Sequence[int], gold_paths: Sequence[Sequence[int]]) -> bool:
    return any(_has_prefix(gold_path, branch_path) for gold_path in gold_paths)


def _is_reseat_event(rec: Dict[str, Any]) -> bool:
    if rec.get("ended_beam_reseat_selected_paths"):
        return True
    return "ended_reseat" in str(rec.get("selector_pick_reason", ""))


def _is_first_reseat(iter_records: Sequence[Dict[str, Any]], iter_idx: int) -> bool:
    for idx in range(iter_idx):
        if _is_reseat_event(iter_records[idx]):
            return False
    return True


def _path_rows_to_paths(rows: Sequence[Dict[str, Any]]) -> List[Tuple[int, ...]]:
    paths: List[Tuple[int, ...]] = []
    for row in rows or []:
        path = row.get("path", [])
        if path:
            paths.append(tuple(path))
    return sorted({tuple(path) for path in paths})


def _region_count(paths: Sequence[Sequence[int]], anchors: Sequence[Tuple[int, ...]]) -> int:
    count = 0
    for path in paths or []:
        if any(_has_prefix(path, anchor) for anchor in anchors):
            count += 1
    return int(count)


def _region_share(paths: Sequence[Sequence[int]], anchors: Sequence[Tuple[int, ...]]) -> float:
    total = len(paths or [])
    if total == 0:
        return 0.0
    return float(_region_count(paths, anchors) / total)


def _survival_stats(
    iter_records: Sequence[Dict[str, Any]],
    start_iter: int,
    anchor_paths: Sequence[Tuple[int, ...]],
    key: str,
    start_offset: int,
) -> Dict[str, Any]:
    shares: List[float] = []
    counts: List[int] = []
    consecutive_survival = 0
    first_zero_rel_iter: Optional[int] = None

    for future_iter in range(start_iter + start_offset, len(iter_records)):
        ranked_paths = iter_records[future_iter].get(key, []) or []
        share = _region_share(ranked_paths, anchor_paths)
        count = _region_count(ranked_paths, anchor_paths)
        shares.append(share)
        counts.append(count)
        if count > 0 and first_zero_rel_iter is None:
            consecutive_survival += 1
        elif count == 0 and first_zero_rel_iter is None:
            first_zero_rel_iter = future_iter - start_iter

    if first_zero_rel_iter is None and len(iter_records) > (start_iter + start_offset):
        survive_to_end = int(all(count > 0 for count in counts))
    else:
        survive_to_end = 0

    if not shares:
        return {
            "future_steps": 0,
            "survive_to_end": int(False),
            "consecutive_survival_steps": 0,
            "first_zero_rel_iter": math.nan,
            "mean_share": math.nan,
            "max_share": math.nan,
            "last_positive_rel_iter": math.nan,
        }

    last_positive_rel_iter = math.nan
    for rel_idx, count in enumerate(counts, start=1):
        if count > 0:
            last_positive_rel_iter = float(rel_idx)

    return {
        "future_steps": int(len(shares)),
        "survive_to_end": int(all(count > 0 for count in counts)),
        "consecutive_survival_steps": int(consecutive_survival),
        "first_zero_rel_iter": float(first_zero_rel_iter) if first_zero_rel_iter is not None else math.nan,
        "mean_share": float(sum(shares) / len(shares)),
        "max_share": float(max(shares)),
        "last_positive_rel_iter": float(last_positive_rel_iter),
    }


def _max_future_ndcg(iter_records: Sequence[Dict[str, Any]], start_iter: int) -> float:
    scores: List[float] = []
    for future_iter in range(start_iter + 1, len(iter_records)):
        scores.append(_safe_float(iter_records[future_iter].get("metrics", {}).get("nDCG@10")))
    if not scores:
        return float("nan")
    return float(max(scores))


def _selected_after_gold_stats(
    iter_records: Sequence[Dict[str, Any]],
    start_iter: int,
    gold_paths: Sequence[Tuple[int, ...]],
) -> Dict[str, Any]:
    current_after = [tuple(path) for path in (iter_records[start_iter].get("selected_branches_after", []) or []) if path]
    current_has_gold = int(any(_branch_is_gold_ancestor(path, gold_paths) for path in current_after))

    future_hit = 0
    first_future_hit_rel_iter = math.nan
    for future_iter in range(start_iter + 1, len(iter_records)):
        future_after = [tuple(path) for path in (iter_records[future_iter].get("selected_branches_after", []) or []) if path]
        if any(_branch_is_gold_ancestor(path, gold_paths) for path in future_after):
            future_hit = 1
            first_future_hit_rel_iter = float(future_iter - start_iter)
            break

    # Intent: strict reacquire means gold was absent at the reseat step itself and only appeared again later.
    strict_reacquire = int((current_has_gold == 0) and (future_hit == 1))
    return {
        "selected_after_has_gold_t": current_has_gold,
        "future_gold_branch_hit": int(future_hit),
        "future_first_gold_hit_rel_iter": float(first_future_hit_rel_iter),
        "strict_gold_reacquire": strict_reacquire,
    }


def build_rows(round6_by_subset: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for subset, samples in round6_by_subset.items():
        for query_idx, sample in enumerate(samples):
            iter_records = sample.get("iter_records", []) or []
            gold_paths = [tuple(path) for path in (sample.get("gold_paths", []) or []) if path]
            if not gold_paths:
                continue

            final_ndcg = float("nan")
            if iter_records:
                final_ndcg = _safe_float(iter_records[-1].get("metrics", {}).get("nDCG@10"))

            for iter_idx, rec in enumerate(iter_records):
                if not _is_reseat_event(rec):
                    continue

                reseated_paths = _path_rows_to_paths(rec.get("ended_beam_reseat_selected_paths", []) or [])
                if not reseated_paths:
                    continue

                gold_return_paths = [path for path in reseated_paths if _branch_is_gold_ancestor(path, gold_paths)]
                gold_return_paths = sorted({tuple(path) for path in gold_return_paths})
                has_gold_return = int(len(gold_return_paths) > 0)

                row: Dict[str, Any] = {
                    "subset": subset,
                    "query_idx": int(query_idx),
                    "iter": int(iter_idx),
                    "query": str(sample.get("original_query") or sample.get("query") or ""),
                    "is_first_reseat": int(_is_first_reseat(iter_records, iter_idx)),
                    "ended_beam_count": int(rec.get("ended_beam_count", 0)),
                    "return_branch_count": int(len(reseated_paths)),
                    "gold_return_branch_count": int(len(gold_return_paths)),
                    "return_has_gold": has_gold_return,
                    "selector_pick_reason": str(rec.get("selector_pick_reason", "")),
                    "reseated_paths_json": _json_dumps([list(path) for path in reseated_paths]),
                    "gold_return_paths_json": _json_dumps([list(path) for path in gold_return_paths]),
                    "ndcg_t": _safe_float(rec.get("metrics", {}).get("nDCG@10")),
                    "final_ndcg": final_ndcg,
                    "max_future_ndcg": _max_future_ndcg(iter_records, iter_idx),
                    "delta_end_from_t": final_ndcg - _safe_float(rec.get("metrics", {}).get("nDCG@10")),
                    "delta_max_future_from_t": _max_future_ndcg(iter_records, iter_idx) - _safe_float(rec.get("metrics", {}).get("nDCG@10")),
                }
                row.update(_selected_after_gold_stats(iter_records, iter_idx, gold_paths))

                if has_gold_return:
                    # Intent: measure whether a gold-return branch remains alive in later retrieval, not just whether it was picked once.
                    active_stats = _survival_stats(iter_records, iter_idx, gold_return_paths, "active_eval_paths", start_offset=1)
                    prehit_stats = _survival_stats(iter_records, iter_idx, gold_return_paths, "pre_hit_paths", start_offset=1)
                    # Intent: branch-state retention should ignore the immediate t+1 handoff and check whether the returned subtree survives from t+2 onward.
                    branch_stats = _survival_stats(iter_records, iter_idx, gold_return_paths, "selected_branches_after", start_offset=2)
                else:
                    active_stats = {
                        "future_steps": 0,
                        "survive_to_end": math.nan,
                        "consecutive_survival_steps": math.nan,
                        "first_zero_rel_iter": math.nan,
                        "mean_share": math.nan,
                        "max_share": math.nan,
                        "last_positive_rel_iter": math.nan,
                    }
                    prehit_stats = dict(active_stats)
                    branch_stats = dict(active_stats)

                for prefix, stats in [("active", active_stats), ("prehit", prehit_stats), ("branch", branch_stats)]:
                    row[f"{prefix}_future_steps"] = stats["future_steps"]
                    row[f"{prefix}_survive_to_end"] = stats["survive_to_end"]
                    row[f"{prefix}_consecutive_survival_steps"] = stats["consecutive_survival_steps"]
                    row[f"{prefix}_first_zero_rel_iter"] = stats["first_zero_rel_iter"]
                    row[f"{prefix}_mean_share"] = stats["mean_share"]
                    row[f"{prefix}_max_share"] = stats["max_share"]
                    row[f"{prefix}_last_positive_rel_iter"] = stats["last_positive_rel_iter"]

                rows.append(row)

    return pd.DataFrame(rows)


def _mean(series: pd.Series) -> float:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return float("nan")
    return float(valid.mean())


def _summarize_scope(rows_df: pd.DataFrame, scope_name: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "scope": scope_name,
        "n": int(len(rows_df)),
    }
    if rows_df.empty:
        return out

    metrics = [
        "ended_beam_count",
        "return_branch_count",
        "gold_return_branch_count",
        "return_has_gold",
        "ndcg_t",
        "final_ndcg",
        "max_future_ndcg",
        "delta_end_from_t",
        "delta_max_future_from_t",
        "selected_after_has_gold_t",
        "future_gold_branch_hit",
        "future_first_gold_hit_rel_iter",
        "strict_gold_reacquire",
        "active_survive_to_end",
        "active_consecutive_survival_steps",
        "active_first_zero_rel_iter",
        "active_mean_share",
        "active_max_share",
        "active_last_positive_rel_iter",
        "prehit_survive_to_end",
        "prehit_consecutive_survival_steps",
        "prehit_first_zero_rel_iter",
        "prehit_mean_share",
        "prehit_max_share",
        "prehit_last_positive_rel_iter",
        "branch_survive_to_end",
        "branch_consecutive_survival_steps",
        "branch_first_zero_rel_iter",
        "branch_mean_share",
        "branch_max_share",
        "branch_last_positive_rel_iter",
    ]
    for metric in metrics:
        out[metric] = _mean(rows_df[metric])
    return out


def build_summary(rows_df: pd.DataFrame) -> pd.DataFrame:
    scopes = {
        "all_reseat_events": rows_df,
        "first_reseat_only": rows_df[rows_df["is_first_reseat"] == 1],
        "gold_return_events": rows_df[rows_df["return_has_gold"] == 1],
        "non_gold_return_events": rows_df[rows_df["return_has_gold"] == 0],
        "gold_return_first_reseat": rows_df[(rows_df["return_has_gold"] == 1) & (rows_df["is_first_reseat"] == 1)],
        "non_gold_return_first_reseat": rows_df[(rows_df["return_has_gold"] == 0) & (rows_df["is_first_reseat"] == 1)],
        "strict_reacquire_events": rows_df[rows_df["strict_gold_reacquire"] == 1],
    }
    return pd.DataFrame([_summarize_scope(df, scope) for scope, df in scopes.items()])


def build_by_subset(rows_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if rows_df.empty:
        return pd.DataFrame()
    for subset, subset_df in rows_df.groupby("subset", dropna=False):
        rows.append(_summarize_scope(subset_df, f"{subset}:all_reseat_events"))
        rows.append(_summarize_scope(subset_df[subset_df["return_has_gold"] == 1], f"{subset}:gold_return_events"))
    return pd.DataFrame(rows)


def build_examples(rows_df: pd.DataFrame, limit: int) -> pd.DataFrame:
    if rows_df.empty:
        return rows_df
    target_df = rows_df[(rows_df["return_has_gold"] == 0) & (rows_df["is_first_reseat"] == 1)].copy()
    if target_df.empty:
        return target_df
    target_df = target_df.sort_values(
        by=["strict_gold_reacquire", "future_gold_branch_hit", "delta_max_future_from_t", "delta_end_from_t"],
        ascending=[True, True, True, True],
    )
    keep_cols = [
        "subset",
        "query_idx",
        "iter",
        "query",
        "is_first_reseat",
        "ended_beam_count",
        "return_branch_count",
        "gold_return_branch_count",
        "selected_after_has_gold_t",
        "future_gold_branch_hit",
        "future_first_gold_hit_rel_iter",
        "strict_gold_reacquire",
        "ndcg_t",
        "final_ndcg",
        "max_future_ndcg",
        "delta_end_from_t",
        "delta_max_future_from_t",
        "branch_survive_to_end",
        "branch_consecutive_survival_steps",
        "branch_first_zero_rel_iter",
        "branch_mean_share",
        "branch_last_positive_rel_iter",
        "active_survive_to_end",
        "active_consecutive_survival_steps",
        "active_first_zero_rel_iter",
        "active_mean_share",
        "active_last_positive_rel_iter",
        "prehit_survive_to_end",
        "prehit_consecutive_survival_steps",
        "prehit_first_zero_rel_iter",
        "prehit_mean_share",
        "prehit_last_positive_rel_iter",
        "gold_return_paths_json",
    ]
    return target_df[keep_cols].head(limit).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--round6_glob_pattern",
        default=DEFAULT_ROUND6_GLOB,
        help="Glob for round6 all_eval_sample_dicts.pkl files.",
    )
    parser.add_argument(
        "--round6_require_substrings",
        nargs="*",
        default=DEFAULT_ROUND6_REQUIRE,
        help="All substrings that must appear in round6 paths.",
    )
    parser.add_argument(
        "--examples_limit",
        type=int,
        default=50,
        help="Number of weak gold-return examples to save.",
    )
    parser.add_argument(
        "--out_prefix",
        default="results/BRIGHT/analysis/round6_reseat_return",
        help="Output prefix for CSV files.",
    )
    args = parser.parse_args()

    round6_paths = _resolve_paths(
        inputs=[],
        glob_pattern=args.round6_glob_pattern,
        require_substrings=args.round6_require_substrings,
    )
    round6_by_subset = _load_round6_samples(round6_paths)

    rows_df = build_rows(round6_by_subset)
    summary_df = build_summary(rows_df)
    by_subset_df = build_by_subset(rows_df)
    examples_df = build_examples(rows_df, int(args.examples_limit))

    out_dir = os.path.dirname(os.path.abspath(args.out_prefix))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    rows_path = f"{args.out_prefix}_rows.csv"
    summary_path = f"{args.out_prefix}_summary.csv"
    by_subset_path = f"{args.out_prefix}_by_subset.csv"
    examples_path = f"{args.out_prefix}_examples.csv"

    rows_df.to_csv(rows_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    by_subset_df.to_csv(by_subset_path, index=False)
    examples_df.to_csv(examples_path, index=False)

    print(f"[saved] {rows_path}")
    print(f"[saved] {summary_path}")
    print(f"[saved] {by_subset_path}")
    print(f"[saved] {examples_path}")

    if not summary_df.empty:
        print("\n[summary]")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
