#!/usr/bin/env python3
"""
Compare frontiercum_qstate ended-reseat runs with scored vs random reseat.

This analysis answers one narrow question:
- why does random ended-slot reseat outperform score-based reseat in the
  frontiercum_qstate round6 variant?

It focuses on per-iteration branch-state and gold-signal metrics:
- nDCG@10
- selected branch depth
- gold-ancestor branch hit / precision
- active-eval gold-doc-any
- pre-hit gold-doc-any
- reseat depth

Outputs:
- <out_prefix>_query_iter.csv
- <out_prefix>_subset_iter.csv
- <out_prefix>_overall_iter.csv
- <out_prefix>_overall_compare.csv
- <out_prefix>_subset_compare.csv
"""

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd


DEFAULT_SUBSETS: List[str] = [
    "biology",
    "psychology",
    "economics",
    "earth_science",
    "robotics",
    "sustainable_living",
    "stackoverflow",
    "theoremqa_theorems",
    "pony",
]


def _is_prefix(prefix: Sequence[int], path: Sequence[int]) -> bool:
    prefix_t = tuple(prefix)
    path_t = tuple(path)
    return len(prefix_t) <= len(path_t) and prefix_t == path_t[: len(prefix_t)]


def _mean(values: Iterable[float]) -> float:
    filtered = [float(v) for v in values if not math.isnan(float(v))]
    if not filtered:
        return float("nan")
    return float(sum(filtered) / len(filtered))


def _best_gold_rank(doc_ids: Sequence[str], gold_doc_ids: Sequence[str] | set[str]) -> float:
    gold_set = set(gold_doc_ids or [])
    for rank_idx, doc_id in enumerate(doc_ids or [], start=1):
        if doc_id in gold_set:
            return float(rank_idx)
    return float("nan")


def _normalize_paths(paths: Sequence[Any]) -> List[Tuple[int, ...]]:
    normalized: List[Tuple[int, ...]] = []
    for path in paths or []:
        if path:
            normalized.append(tuple(int(x) for x in list(path)))
    return normalized


def _normalize_path_rows(path_rows: Sequence[Any]) -> List[Tuple[int, ...]]:
    normalized: List[Tuple[int, ...]] = []
    for row in path_rows or []:
        if isinstance(row, dict):
            path = row.get("path", []) or []
        else:
            path = row or []
        if path:
            normalized.append(tuple(path))
    return normalized


def _count_pool_gold_hits(
    *,
    selected_branches_before: Sequence[Tuple[int, ...]],
    cumulative_reached_leaves: Sequence[Tuple[int, ...]],
    gold_paths: Sequence[Tuple[int, ...]],
) -> int:
    cumulative_set = {tuple(path) for path in cumulative_reached_leaves or []}
    selected_list = [tuple(path) for path in selected_branches_before or []]
    if (not cumulative_set) and (not selected_list):
        return int(len(gold_paths))
    hits = 0
    for gold_path in gold_paths:
        gold_t = tuple(gold_path)
        if gold_t in cumulative_set:
            hits += 1
            continue
        if any(_is_prefix(branch_path, gold_t) for branch_path in selected_list):
            hits += 1
    return hits


def _count_frontier_component_gold_hits(
    *,
    selected_branches_before: Sequence[Tuple[int, ...]],
    cumulative_reached_leaves: Sequence[Tuple[int, ...]],
    gold_paths: Sequence[Tuple[int, ...]],
) -> int:
    cumulative_set = {tuple(path) for path in cumulative_reached_leaves or []}
    selected_list = [tuple(path) for path in selected_branches_before or []]
    if not selected_list:
        return 0
    hits = 0
    for gold_path in gold_paths:
        gold_t = tuple(gold_path)
        if gold_t in cumulative_set:
            continue
        if any(_is_prefix(branch_path, gold_t) for branch_path in selected_list):
            hits += 1
    return hits


def _has_gold_branch(branches: Sequence[Sequence[int]], gold_paths: Sequence[Sequence[int]]) -> bool:
    branch_list = [tuple(path) for path in branches or [] if path]
    gold_list = [tuple(path) for path in gold_paths or [] if path]
    return any(any(_is_prefix(branch, gold) for gold in gold_list) for branch in branch_list)


def _branch_precision(branches: Sequence[Sequence[int]], gold_paths: Sequence[Sequence[int]]) -> float:
    branch_list = [tuple(path) for path in branches or [] if path]
    gold_list = [tuple(path) for path in gold_paths or [] if path]
    if not branch_list:
        return float("nan")
    gold_hits = sum(1 for branch in branch_list if any(_is_prefix(branch, gold) for gold in gold_list))
    return float(gold_hits / len(branch_list))


def _find_run_path(base_dir: Path, subset: str, mode: str) -> Path:
    candidates = sorted((base_dir / subset / "round6").glob("**/all_eval_sample_dicts.pkl"))
    matches: List[Path] = []
    for path in candidates:
        path_str = str(path)
        if "reason/embed-qwen3-8b-0928" not in path_str:
            continue
        if "agent_executor_v1_icl2" not in path_str:
            continue
        if "emr" in path_str:
            continue
        if "expandable_ended_reseat_frontiercum_qstate" not in path_str:
            continue
        is_random = (
            "RERP=random" in path_str
            or "_random-" in path_str
            or "frontiercum_qstate_random" in path_str
        )
        if mode == "score" and not is_random:
            matches.append(path)
        elif mode == "random" and is_random:
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {mode} run for subset={subset}, got {len(matches)}: "
            f"{[str(path) for path in matches]}"
        )
    return matches[0]


def _sample_iter_rows(*, subset: str, mode: str, samples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for query_idx, sample in enumerate(samples):
        gold_doc_ids = set(sample.get("gold_doc_ids", []) or [])
        gold_paths = [tuple(path) for path in (sample.get("gold_paths", []) or []) if path]
        cumulative_reached_leaf_paths: set[Tuple[int, ...]] = set()
        for rec in sample.get("iter_records", []) or []:
            metrics = rec.get("metrics", {}) or {}
            selected_branches = rec.get("selected_branches_after", []) or []
            selected_branches_before = _normalize_paths(rec.get("selected_branches_before", []) or [])
            ended_reseat_paths = _normalize_path_rows(rec.get("ended_beam_reseat_selected_paths", []) or [])
            active_eval_doc_ids = rec.get("active_eval_doc_ids", []) or []
            pre_hit_doc_ids = rec.get("pre_hit_doc_ids", []) or []
            unique_root_prefixes = {tuple(path[:1]) for path in selected_branches if len(path) >= 1}
            # Intent: track whether random reseat preserves a wider frontier instead of collapsing onto a few scored subtrees.
            unique_depth2_prefixes = {tuple(path[:2]) for path in selected_branches if len(path) >= 2}
            # Intent: keep pool-gold metrics aligned with the earlier frontiercum_qstate analysis: frontier descendants before the iter plus cumulative reached leaves before the iter.
            pool_gold_hits = _count_pool_gold_hits(
                selected_branches_before=selected_branches_before,
                cumulative_reached_leaves=sorted(cumulative_reached_leaf_paths),
                gold_paths=gold_paths,
            )
            pool_has_any_gold_doc = 1.0 if pool_gold_hits > 0 else 0.0
            pool_gold_doc_recall = (
                float(pool_gold_hits / len(gold_paths))
                if gold_paths
                else float("nan")
            )
            # Intent: isolate the newly added frontier component by excluding already reached leaves from the current pool-gold count.
            frontier_added_gold_hits = _count_frontier_component_gold_hits(
                selected_branches_before=selected_branches_before,
                cumulative_reached_leaves=sorted(cumulative_reached_leaf_paths),
                gold_paths=gold_paths,
            )
            frontier_added_has_any_gold_doc = 1.0 if frontier_added_gold_hits > 0 else 0.0
            frontier_added_gold_doc_recall = (
                float(frontier_added_gold_hits / len(gold_paths))
                if gold_paths
                else float("nan")
            )
            prehit_best_gold_rank = _best_gold_rank(pre_hit_doc_ids, gold_doc_ids)
            active_best_gold_rank = _best_gold_rank(active_eval_doc_ids, gold_doc_ids)
            row = {
                "mode": mode,
                "subset": subset,
                "query_idx": int(query_idx),
                "iter": int(rec.get("iter", 0)),
                "ndcg10": float(metrics.get("nDCG@10", float("nan"))),
                "selected_depth": float(metrics.get("SelectedDepth", float("nan"))),
                "ended_beam_count": float(rec.get("ended_beam_count", 0) or 0),
                "reseat_rate": 1.0 if ended_reseat_paths else 0.0,
                # Intent: use actual selected branch prefixes rather than proxy paths when judging gold-region alignment.
                "branch_hit": 1.0 if _has_gold_branch(selected_branches, gold_paths) else 0.0,
                "branch_precision": _branch_precision(selected_branches, gold_paths),
                "active_has_gold_doc_any": 1.0 if any(doc_id in gold_doc_ids for doc_id in active_eval_doc_ids) else 0.0,
                "prehit_has_gold_doc_any": 1.0 if any(doc_id in gold_doc_ids for doc_id in pre_hit_doc_ids) else 0.0,
                "active_best_gold_rank": float(active_best_gold_rank),
                "prehit_best_gold_rank": float(prehit_best_gold_rank),
                "active_gold_in_top10": 1.0 if active_best_gold_rank == active_best_gold_rank and active_best_gold_rank <= 10.0 else 0.0,
                "prehit_gold_in_top10": 1.0 if prehit_best_gold_rank == prehit_best_gold_rank and prehit_best_gold_rank <= 10.0 else 0.0,
                "pool_has_any_gold_doc": float(pool_has_any_gold_doc),
                "pool_gold_doc_recall": float(pool_gold_doc_recall),
                "frontier_added_has_any_gold_doc": float(frontier_added_has_any_gold_doc),
                "frontier_added_gold_doc_hits": float(frontier_added_gold_hits),
                "frontier_added_gold_doc_recall": float(frontier_added_gold_doc_recall),
                "unique_root_prefix_count": float(len(unique_root_prefixes)),
                "unique_depth2_prefix_count": float(len(unique_depth2_prefixes)),
                # Intent: isolate how aggressively the reseat policy dives deeper when it replaces ended slots.
                "reseat_depth_mean": _mean(len(path) for path in ended_reseat_paths),
                "reseat_depth_max": max((len(path) for path in ended_reseat_paths), default=float("nan")),
                "reseat_has_gold_branch_any": (
                    1.0 if _has_gold_branch(ended_reseat_paths, gold_paths) else 0.0
                ) if ended_reseat_paths else float("nan"),
            }
            rows.append(row)
            cumulative_reached_leaf_paths.update(_normalize_paths(rec.get("new_leaf_paths", []) or []))
    return rows


def _aggregate_subset_iter(query_iter_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "ndcg10",
        "selected_depth",
        "ended_beam_count",
        "reseat_rate",
        "branch_hit",
        "branch_precision",
        "active_has_gold_doc_any",
        "prehit_has_gold_doc_any",
        "active_best_gold_rank",
        "prehit_best_gold_rank",
        "active_gold_in_top10",
        "prehit_gold_in_top10",
        "pool_has_any_gold_doc",
        "pool_gold_doc_recall",
        "frontier_added_has_any_gold_doc",
        "frontier_added_gold_doc_hits",
        "frontier_added_gold_doc_recall",
        "unique_root_prefix_count",
        "unique_depth2_prefix_count",
        "reseat_depth_mean",
        "reseat_depth_max",
        "reseat_has_gold_branch_any",
    ]
    grouped = (
        query_iter_df
        .groupby(["mode", "subset", "iter"], as_index=False)[metric_cols]
        .mean()
        .sort_values(["mode", "subset", "iter"])
        .reset_index(drop=True)
    )
    return grouped


def _aggregate_overall_iter(subset_iter_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "ndcg10",
        "selected_depth",
        "ended_beam_count",
        "reseat_rate",
        "branch_hit",
        "branch_precision",
        "active_has_gold_doc_any",
        "prehit_has_gold_doc_any",
        "active_best_gold_rank",
        "prehit_best_gold_rank",
        "active_gold_in_top10",
        "prehit_gold_in_top10",
        "pool_has_any_gold_doc",
        "pool_gold_doc_recall",
        "frontier_added_has_any_gold_doc",
        "frontier_added_gold_doc_hits",
        "frontier_added_gold_doc_recall",
        "unique_root_prefix_count",
        "unique_depth2_prefix_count",
        "reseat_depth_mean",
        "reseat_depth_max",
        "reseat_has_gold_branch_any",
    ]
    overall = (
        subset_iter_df
        .groupby(["mode", "iter"], as_index=False)[metric_cols]
        .mean()
        .sort_values(["mode", "iter"])
        .reset_index(drop=True)
    )
    return overall


def _build_compare_df(df: pd.DataFrame, index_cols: List[str]) -> pd.DataFrame:
    score_df = df[df["mode"] == "score"].copy()
    random_df = df[df["mode"] == "random"].copy()
    score_df = score_df.rename(columns={col: f"{col}_score" for col in df.columns if col not in index_cols + ["mode"]})
    random_df = random_df.rename(columns={col: f"{col}_random" for col in df.columns if col not in index_cols + ["mode"]})
    merged = score_df[index_cols + [col for col in score_df.columns if col.endswith("_score")]].merge(
        random_df[index_cols + [col for col in random_df.columns if col.endswith("_random")]],
        on=index_cols,
        how="inner",
    )
    base_metrics = [
        "ndcg10",
        "selected_depth",
        "ended_beam_count",
        "reseat_rate",
        "branch_hit",
        "branch_precision",
        "active_has_gold_doc_any",
        "prehit_has_gold_doc_any",
        "active_best_gold_rank",
        "prehit_best_gold_rank",
        "active_gold_in_top10",
        "prehit_gold_in_top10",
        "pool_has_any_gold_doc",
        "pool_gold_doc_recall",
        "frontier_added_has_any_gold_doc",
        "frontier_added_gold_doc_hits",
        "frontier_added_gold_doc_recall",
        "unique_root_prefix_count",
        "unique_depth2_prefix_count",
        "reseat_depth_mean",
        "reseat_depth_max",
        "reseat_has_gold_branch_any",
    ]
    for metric in base_metrics:
        merged[f"{metric}_delta_random_minus_score"] = (
            merged[f"{metric}_random"] - merged[f"{metric}_score"]
        )
    return merged.sort_values(index_cols).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=Path, default=Path("results/BRIGHT"))
    parser.add_argument(
        "--out_prefix",
        type=Path,
        default=Path("results/BRIGHT/analysis/round6_random_vs_score_frontiercum"),
    )
    parser.add_argument(
        "--subsets",
        nargs="*",
        default=None,
        help="Optional explicit subset list. Default uses the 9 completed frontiercum_qstate subsets.",
    )
    return parser.parse_args()


def _aggregate_conditioned_iter(df: pd.DataFrame, *, condition_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    conditioned = df[df[condition_col] > 0.0].copy()
    subset_iter_df = _aggregate_subset_iter(conditioned)
    overall_iter_df = _aggregate_overall_iter(subset_iter_df)
    return subset_iter_df, overall_iter_df


def main() -> None:
    args = parse_args()
    subsets = args.subsets or DEFAULT_SUBSETS
    query_rows: List[Dict[str, Any]] = []
    for mode in ("score", "random"):
        for subset in subsets:
            run_path = _find_run_path(args.base_dir, subset, mode)
            with run_path.open("rb") as file_obj:
                samples = pickle.load(file_obj)
            query_rows.extend(_sample_iter_rows(subset=subset, mode=mode, samples=samples))

    query_iter_df = pd.DataFrame(query_rows)
    subset_iter_df = _aggregate_subset_iter(query_iter_df)
    overall_iter_df = _aggregate_overall_iter(subset_iter_df)
    overall_compare_df = _build_compare_df(overall_iter_df, ["iter"])
    subset_compare_df = _build_compare_df(subset_iter_df, ["subset", "iter"])
    pool_subset_iter_df, pool_overall_iter_df = _aggregate_conditioned_iter(
        query_iter_df,
        condition_col="pool_has_any_gold_doc",
    )
    pool_overall_compare_df = _build_compare_df(pool_overall_iter_df, ["iter"])

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    query_iter_df.to_csv(f"{args.out_prefix}_query_iter.csv", index=False)
    subset_iter_df.to_csv(f"{args.out_prefix}_subset_iter.csv", index=False)
    overall_iter_df.to_csv(f"{args.out_prefix}_overall_iter.csv", index=False)
    overall_compare_df.to_csv(f"{args.out_prefix}_overall_compare.csv", index=False)
    subset_compare_df.to_csv(f"{args.out_prefix}_subset_compare.csv", index=False)
    pool_subset_iter_df.to_csv(f"{args.out_prefix}_pool_conditioned_subset_iter.csv", index=False)
    pool_overall_iter_df.to_csv(f"{args.out_prefix}_pool_conditioned_overall_iter.csv", index=False)
    pool_overall_compare_df.to_csv(f"{args.out_prefix}_pool_conditioned_overall_compare.csv", index=False)

    print(f"Wrote {args.out_prefix}_query_iter.csv")
    print(f"Wrote {args.out_prefix}_subset_iter.csv")
    print(f"Wrote {args.out_prefix}_overall_iter.csv")
    print(f"Wrote {args.out_prefix}_overall_compare.csv")
    print(f"Wrote {args.out_prefix}_subset_compare.csv")
    print(f"Wrote {args.out_prefix}_pool_conditioned_subset_iter.csv")
    print(f"Wrote {args.out_prefix}_pool_conditioned_overall_iter.csv")
    print(f"Wrote {args.out_prefix}_pool_conditioned_overall_compare.csv")


if __name__ == "__main__":
    main()
