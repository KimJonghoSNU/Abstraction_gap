#!/usr/bin/env python3
"""
Analyze whether baseline3 hits a gold branch in rewrite context and then drifts
away without a tree-constrained retrieval pool.

Examples
--------
python scripts/analysis/analyze_baseline3_derailment.py

python scripts/analysis/analyze_baseline3_derailment.py \
    --subsets biology psychology robotics \
    --out_dir results/BRIGHT/analysis/baseline3_derailment
"""

import argparse
import json
import math
import os
import pickle as pkl
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tree_objects import SemanticNode  # noqa: F401


DEFAULT_SUBSETS = [
    "aops",
    "biology",
    "earth_science",
    "economics",
    "leetcode",
    "pony",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
    "theoremqa_questions",
    "theoremqa_theorems",
]

DEFAULT_RUN_TOKENS = [
    "baseline3_leaf_only_loop",
    "agent_executor_v1_icl2",
    "PlTau=5.0",
    "RCT=10",
    "RSC=on",
]

DEFAULT_DEPTHS = [2, 3, 4]


@dataclass
class QueryGold:
    gold_doc_ids: List[str]
    gold_paths: List[Tuple[int, ...]]


@dataclass
class SubsetResources:
    subset: str
    tree_version: str
    num_queries: int
    gold_by_query: Dict[int, QueryGold]


def _infer_subset_from_path(path: Path) -> str:
    parts = list(path.resolve().parts)
    if "BRIGHT" in parts:
        idx = parts.index("BRIGHT")
        if idx + 1 < len(parts):
            return str(parts[idx + 1])
    return "unknown"


def _extract_tree_version_from_path(path: Path) -> str:
    match = re.search(r"-TV=(.+?)-TPV=", str(path))
    if not match:
        raise ValueError(f"Could not parse tree version from path: {path}")
    return str(match.group(1))


def _select_matching_path(
    paths: Sequence[Path],
    required_tokens: Sequence[str],
    label: str,
    subset: str,
) -> Path:
    matches = [path for path in paths if all(token in str(path) for token in required_tokens)]
    if not matches:
        raise FileNotFoundError(
            f"No {label} matched for subset={subset}. required_tokens={list(required_tokens)}"
        )
    if len(matches) == 1:
        return matches[0]

    # Intent: repeated reruns can leave multiple compatible artifacts, so prefer the latest one deterministically.
    latest = max(matches, key=lambda path: path.stat().st_mtime)
    print(f"[warn] Multiple {label} matches for subset={subset}; using latest mtime: {latest}")
    return latest


def _resolve_subset_run_paths(
    subset: str,
    run_tokens: Sequence[str],
) -> Tuple[Path, Path]:
    subset_root = REPO_ROOT / "results" / "BRIGHT" / subset
    record_paths = sorted(subset_root.glob("**/leaf_iter_records.jsonl"))
    metric_paths = sorted(subset_root.glob("**/leaf_iter_metrics.jsonl"))
    records_path = _select_matching_path(record_paths, run_tokens, "records", subset)
    metrics_path = _select_matching_path(metric_paths, run_tokens, "metrics", subset)
    return records_path, metrics_path


def _load_iter_records(path: Path) -> Dict[int, Dict[int, Dict[str, Any]]]:
    by_query: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            phase = str(rec.get("phase", ""))
            if phase not in {"initial_rewrite", "iter_retrieval"}:
                continue
            query_idx = int(rec.get("query_idx", -1))
            if query_idx < 0:
                continue
            iter_idx = -1 if phase == "initial_rewrite" else int(rec.get("iter", -1))
            if iter_idx < -1:
                continue
            # Intent: old result files may be appended multiple times, so keep the latest row per query/iter.
            by_query[query_idx][iter_idx] = rec
    return {query_idx: dict(rows) for query_idx, rows in by_query.items()}


def _load_iter_metrics(path: Path) -> Dict[int, Dict[int, Dict[str, Any]]]:
    by_query: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            query_idx = int(rec.get("query_idx", -1))
            iter_idx = int(rec.get("iter", -1))
            if query_idx < 0 or iter_idx < 0:
                continue
            by_query[query_idx][iter_idx] = rec
    return {query_idx: dict(rows) for query_idx, rows in by_query.items()}


def _load_docs_examples(subset: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    docs_path = REPO_ROOT / "data" / "BRIGHT" / subset / "documents.jsonl"
    examples_path = REPO_ROOT / "data" / "BRIGHT" / subset / "examples.jsonl"
    if docs_path.exists() and examples_path.exists():
        docs_df = pd.read_json(docs_path, lines=True, dtype={"id": str})
        examples_df = pd.read_json(examples_path, lines=True)
    else:
        docs_df = pd.DataFrame(load_dataset("xlangai/BRIGHT", "documents", split=subset))
        examples_df = pd.DataFrame(load_dataset("xlangai/BRIGHT", "examples", split=subset))
    docs_df["id"] = docs_df["id"].astype(str)
    examples_df["gold_ids"] = examples_df["gold_ids"].apply(lambda xs: [str(x) for x in xs])
    return docs_df, examples_df


def _resolve_node_id(node_id: Any, docs_df: pd.DataFrame) -> Optional[str]:
    if isinstance(node_id, str):
        if node_id.startswith("["):
            parts = node_id.split(" ", 1)
            return str(parts[1]) if len(parts) == 2 else str(node_id)
        return str(node_id)
    if isinstance(node_id, (int, np.integer)):
        return str(docs_df.id.iloc[int(node_id)])
    return None


def _collect_leaf_nodes_with_path(node: Any, path: Tuple[int, ...] = ()) -> List[Tuple[Any, Tuple[int, ...]]]:
    children = getattr(node, "child", []) or []
    if len(children) == 0:
        return [(node, path)]
    out: List[Tuple[Any, Tuple[int, ...]]] = []
    for child_idx, child in enumerate(children):
        out.extend(_collect_leaf_nodes_with_path(child, path + (child_idx,)))
    return out


def _load_doc_id_to_path(subset: str, tree_version: str, docs_df: pd.DataFrame) -> Dict[str, Tuple[int, ...]]:
    tree_path = REPO_ROOT / "trees" / "BRIGHT" / subset / f"tree-{tree_version}.pkl"
    if not tree_path.exists():
        raise FileNotFoundError(f"Tree file not found: {tree_path}")
    tree_payload = pkl.load(open(tree_path, "rb"))
    root = SemanticNode().load_dict(tree_payload) if isinstance(tree_payload, dict) else tree_payload

    doc_id_to_path: Dict[str, Tuple[int, ...]] = {}
    for leaf_node, path in _collect_leaf_nodes_with_path(root):
        doc_id = _resolve_node_id(getattr(leaf_node, "id", None), docs_df)
        if not doc_id:
            continue
        doc_id_to_path[str(doc_id)] = tuple(path)
    return doc_id_to_path


def _load_subset_resources(
    subset: str,
    tree_version: str,
    cache: Dict[Tuple[str, str], SubsetResources],
) -> SubsetResources:
    cache_key = (subset, tree_version)
    if cache_key in cache:
        return cache[cache_key]

    docs_df, examples_df = _load_docs_examples(subset)
    doc_id_to_path = _load_doc_id_to_path(subset, tree_version, docs_df)

    gold_by_query: Dict[int, QueryGold] = {}
    for query_idx in range(len(examples_df)):
        gold_doc_ids = [str(x) for x in examples_df.iloc[query_idx]["gold_ids"]]
        gold_paths = [tuple(doc_id_to_path[doc_id]) for doc_id in gold_doc_ids if doc_id in doc_id_to_path]
        gold_by_query[int(query_idx)] = QueryGold(
            gold_doc_ids=gold_doc_ids,
            gold_paths=sorted({tuple(path) for path in gold_paths}),
        )

    resources = SubsetResources(
        subset=subset,
        tree_version=tree_version,
        num_queries=int(len(examples_df)),
        gold_by_query=gold_by_query,
    )
    cache[cache_key] = resources
    return resources


def _dcg_weight(rank_idx: int) -> float:
    return 1.0 / math.log2(rank_idx + 2.0)


def _compute_ndcg(sorted_preds: Sequence[str], gold: Sequence[str], k: int = 10) -> float:
    if not sorted_preds or not gold:
        return 0.0
    gold_set = set(str(x) for x in gold)
    dcg = 0.0
    for rank_idx, item in enumerate(list(sorted_preds)[:k]):
        if str(item) in gold_set:
            dcg += _dcg_weight(rank_idx)
    ideal_k = min(k, len(gold_set))
    idcg = sum(_dcg_weight(idx) for idx in range(ideal_k))
    if idcg == 0.0:
        return 0.0
    return float(dcg / idcg)


def _best_gold_rank(doc_ids: Sequence[str], gold_doc_ids: Sequence[str]) -> Optional[int]:
    gold_set = set(str(x) for x in (gold_doc_ids or []))
    for rank_idx, doc_id in enumerate(doc_ids):
        if str(doc_id) in gold_set:
            return int(rank_idx + 1)
    return None


def _has_prefix(path: Sequence[int], prefix: Sequence[int]) -> bool:
    if len(prefix) > len(path):
        return False
    return tuple(path[: len(prefix)]) == tuple(prefix)


def _branch_prefix(path: Sequence[int], depth: int) -> Optional[Tuple[int, ...]]:
    if len(path) < 2:
        return None
    use_depth = min(int(depth), len(path) - 1)
    if use_depth <= 0:
        return None
    return tuple(path[:use_depth])


def _gold_branches_at_depth(gold_paths: Sequence[Sequence[int]], depth: int) -> List[Tuple[int, ...]]:
    branches = []
    seen = set()
    for gold_path in gold_paths:
        branch = _branch_prefix(gold_path, depth)
        if branch is None or branch in seen:
            continue
        seen.add(branch)
        branches.append(branch)
    return branches


def _context_branches_at_depth(paths: Sequence[Sequence[int]], depth: int) -> List[Tuple[int, ...]]:
    branches = []
    seen = set()
    for path in paths:
        branch = _branch_prefix(path, depth)
        if branch is None or branch in seen:
            continue
        seen.add(branch)
        branches.append(branch)
    return branches


def _anchor_branches_from_context(
    context_paths: Sequence[Sequence[int]],
    gold_paths: Sequence[Sequence[int]],
    depth: int,
) -> List[Tuple[int, ...]]:
    gold_branch_set = set(_gold_branches_at_depth(gold_paths, depth))
    context_branches = _context_branches_at_depth(context_paths, depth)
    # Intent: anchor baseline derailment on gold-grounded context branches, not on generic branch proxies.
    return sorted({branch for branch in context_branches if branch in gold_branch_set})


def _count_on_anchor(paths: Sequence[Sequence[int]], anchor_branches: Sequence[Tuple[int, ...]], topk: int) -> int:
    count = 0
    for path in list(paths)[:topk]:
        if any(_has_prefix(path, anchor) for anchor in anchor_branches):
            count += 1
    return int(count)


def _off_branch_pct(paths: Sequence[Sequence[int]], branches: Sequence[Tuple[int, ...]], topk: int) -> float:
    ranked_paths = [tuple(path) for path in list(paths)[:topk] if path]
    if not ranked_paths:
        return float("nan")
    off_count = 0
    for path in ranked_paths:
        if not any(_has_prefix(path, branch) for branch in branches):
            off_count += 1
    return 100.0 * float(off_count) / float(len(ranked_paths))


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _mean(series: pd.Series) -> float:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return float("nan")
    return float(valid.mean())


def _build_anchor_event_rows(
    subset: str,
    records_by_iter: Dict[int, Dict[str, Any]],
    metrics_by_iter: Dict[int, Dict[str, Any]],
    query_idx: int,
    depth: int,
    resources: SubsetResources,
) -> List[Dict[str, Any]]:
    gold = resources.gold_by_query.get(query_idx)
    if gold is None or not gold.gold_paths:
        return []

    iter_keys = sorted(records_by_iter.keys())
    final_iter_candidates = [iter_idx for iter_idx in iter_keys if iter_idx >= 0]
    if not final_iter_candidates:
        return []

    final_iter = max(final_iter_candidates)
    final_rec = records_by_iter[final_iter]
    final_doc_ids = [str(x) for x in (final_rec.get("retrieved_doc_ids", []) or [])]
    final_metric = metrics_by_iter.get(final_iter, {})
    final_ndcg10 = _safe_float(final_metric.get("nDCG@10"))
    if math.isnan(final_ndcg10):
        final_ndcg10 = _compute_ndcg(final_doc_ids, gold.gold_doc_ids, k=10) * 100.0
    gold_doc_id_set = set(gold.gold_doc_ids)

    rows: List[Dict[str, Any]] = []
    for iter_anchor in iter_keys:
        anchor_rec = records_by_iter[iter_anchor]
        anchor_branches = _anchor_branches_from_context(
            context_paths=anchor_rec.get("rewrite_context_paths", []) or [],
            gold_paths=gold.gold_paths,
            depth=depth,
        )
        if not anchor_branches:
            continue

        context_branches = _context_branches_at_depth(anchor_rec.get("rewrite_context_paths", []) or [], depth)
        next_rec = records_by_iter.get(iter_anchor + 1)
        anchor_doc_ids = [str(x) for x in (anchor_rec.get("retrieved_doc_ids", []) or [])]
        next_doc_ids = [str(x) for x in (next_rec.get("retrieved_doc_ids", []) or [])] if next_rec else []

        anchor_ndcg10 = _compute_ndcg(anchor_doc_ids, gold.gold_doc_ids, k=10) * 100.0
        next_ndcg10 = float("nan")
        if next_rec is not None:
            next_metric = metrics_by_iter.get(iter_anchor + 1, {})
            next_ndcg10 = _safe_float(next_metric.get("nDCG@10"))
            if math.isnan(next_ndcg10):
                next_ndcg10 = _compute_ndcg(next_doc_ids, gold.gold_doc_ids, k=10) * 100.0

        future_scores: List[float] = []
        for future_iter in sorted(idx for idx in iter_keys if idx > iter_anchor and idx >= 0):
            metric = metrics_by_iter.get(future_iter, {})
            ndcg10 = _safe_float(metric.get("nDCG@10"))
            if math.isnan(ndcg10):
                future_rec = records_by_iter.get(future_iter, {})
                ndcg10 = _compute_ndcg(future_rec.get("retrieved_doc_ids", []) or [], gold.gold_doc_ids, k=10) * 100.0
            future_scores.append(ndcg10)
        best_future_ndcg10 = float(max(future_scores)) if future_scores else float("nan")

        next_top10_on_anchor_count = int(_count_on_anchor(next_rec.get("retrieved_paths", []) or [], anchor_branches, 10)) if next_rec else math.nan
        next_top100_on_anchor_count = int(_count_on_anchor(next_rec.get("retrieved_paths", []) or [], anchor_branches, 100)) if next_rec else math.nan
        next_context_paths = (next_rec.get("rewrite_context_paths", []) or [])[:10] if next_rec else []
        next_ctx_on_anchor_count = int(_count_on_anchor(next_context_paths, anchor_branches, 10)) if next_rec else math.nan

        proxy_next_top10_off_pct = float("nan")
        proxy_next_top100_off_pct = float("nan")
        if next_rec is not None:
            proxy_next_top10_off_pct = _off_branch_pct(
                next_rec.get("retrieved_paths", []) or [],
                context_branches,
                topk=10,
            )
            proxy_next_top100_off_pct = _off_branch_pct(
                next_rec.get("retrieved_paths", []) or [],
                context_branches,
                topk=100,
            )

        next_top10_on_anchor_share = float(next_top10_on_anchor_count / 10.0) if next_rec is not None else float("nan")
        next_top100_on_anchor_share = float(next_top100_on_anchor_count / 100.0) if next_rec is not None else float("nan")
        next_ctx_on_anchor_share = float(next_ctx_on_anchor_count / max(1, len(next_context_paths))) if next_rec is not None else float("nan")
        final_top10_hit = int(any(str(doc_id) in gold_doc_id_set for doc_id in final_doc_ids[:10]))

        rows.append(
            {
                "subset": subset,
                "depth": int(depth),
                "query_idx": int(query_idx),
                "query": str(records_by_iter.get(-1, {}).get("query", "") or ""),
                "anchor_found": 1,
                "iter_anchor": int(iter_anchor),
                "anchor_phase": str(anchor_rec.get("phase", "")),
                "has_next_iter": int(next_rec is not None),
                "final_iter": int(final_iter),
                "anchor_query_for_retrieval": str(anchor_rec.get("query_for_retrieval", "") or ""),
                "next_query_for_retrieval": str(next_rec.get("query_for_retrieval", "") or "") if next_rec else "",
                "final_query_for_retrieval": str(final_rec.get("query_for_retrieval", "") or ""),
                "anchor_gold_branch_count": int(len(anchor_branches)),
                "anchor_branches_json": _json_dumps([list(path) for path in anchor_branches]),
                "context_branches_json": _json_dumps([list(path) for path in context_branches]),
                "gold_doc_ids_json": _json_dumps(gold.gold_doc_ids),
                "gold_paths_json": _json_dumps([list(path) for path in gold.gold_paths]),
                "anchor_rewrite_context_paths_json": _json_dumps(anchor_rec.get("rewrite_context_paths", []) or []),
                "next_rewrite_context_paths_json": _json_dumps(next_rec.get("rewrite_context_paths", []) or []) if next_rec else "[]",
                "anchor_ndcg10": float(anchor_ndcg10),
                "next_ndcg10": float(next_ndcg10),
                "final_ndcg10": float(final_ndcg10),
                "best_future_ndcg10": float(best_future_ndcg10),
                "next_minus_anchor_ndcg10": float(next_ndcg10 - anchor_ndcg10) if next_rec is not None and not math.isnan(next_ndcg10) else float("nan"),
                "end_minus_anchor_ndcg10": float(final_ndcg10 - anchor_ndcg10),
                "best_future_minus_anchor_ndcg10": float(best_future_ndcg10 - anchor_ndcg10) if not math.isnan(best_future_ndcg10) else float("nan"),
                "next_top10_on_anchor_count": next_top10_on_anchor_count,
                "next_top10_on_anchor_share": float(next_top10_on_anchor_share),
                "next_top10_anymiss": int(next_top10_on_anchor_count == 0) if next_rec is not None else math.nan,
                "next_top100_on_anchor_count": next_top100_on_anchor_count,
                "next_top100_on_anchor_share": float(next_top100_on_anchor_share),
                "next_top100_anymiss": int(next_top100_on_anchor_count == 0) if next_rec is not None else math.nan,
                "next_ctx_on_anchor_count": next_ctx_on_anchor_count,
                "next_ctx_on_anchor_share": float(next_ctx_on_anchor_share),
                "next_ctx_anymiss": int(next_ctx_on_anchor_count == 0) if next_rec is not None else math.nan,
                "final_top10_hit": int(final_top10_hit),
                "final_top10_miss": int(1 - final_top10_hit),
                "next_best_gold_rank": _best_gold_rank(next_doc_ids, gold.gold_doc_ids),
                "end_best_gold_rank": _best_gold_rank(final_doc_ids, gold.gold_doc_ids),
                "proxy_next_top10_off_pct": float(proxy_next_top10_off_pct),
                "proxy_next_top10_off_event": int(proxy_next_top10_off_pct > 0.0) if not math.isnan(proxy_next_top10_off_pct) else math.nan,
                "proxy_next_top100_off_pct": float(proxy_next_top100_off_pct),
                "proxy_next_top100_off_event": int(proxy_next_top100_off_pct > 0.0) if not math.isnan(proxy_next_top100_off_pct) else math.nan,
                "anchor_top10_doc_ids_json": _json_dumps(anchor_doc_ids[:10]),
                "next_top10_doc_ids_json": _json_dumps(next_doc_ids[:10]),
                "final_top10_doc_ids_json": _json_dumps(final_doc_ids[:10]),
            }
        )
    return rows


def build_first_anchor_rows(
    event_rows_df: pd.DataFrame,
    subsets: Sequence[str],
    depths: Sequence[int],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if not event_rows_df.empty:
        sort_df = event_rows_df.sort_values(["subset", "depth", "query_idx", "iter_anchor"]).copy()
        subset_query_totals = {
            str(subset): int(group["num_queries_total_subset"].dropna().iloc[0])
            for subset, group in sort_df.groupby("subset", dropna=False)
        }
    else:
        sort_df = event_rows_df
        subset_query_totals = {}

    for subset in subsets:
        total_queries = int(subset_query_totals.get(str(subset), 0))
        for depth in depths:
            subset_depth_df = sort_df[
                (sort_df["subset"] == subset)
                & (sort_df["depth"] == int(depth))
            ].copy() if not sort_df.empty else pd.DataFrame()
            first_by_query: Dict[int, Dict[str, Any]] = {}
            if not subset_depth_df.empty:
                for row in subset_depth_df.to_dict("records"):
                    query_idx = int(row["query_idx"])
                    if query_idx in first_by_query:
                        continue
                    first_by_query[query_idx] = row
                    if len(first_by_query) == subset_depth_df["query_idx"].nunique():
                        break

            inferred_total = max(first_by_query.keys(), default=-1) + 1
            query_total_for_depth = max(total_queries, inferred_total)
            for query_idx in range(query_total_for_depth):
                if query_idx in first_by_query:
                    rows.append(dict(first_by_query[query_idx]))
                else:
                    rows.append(
                        {
                            "subset": subset,
                            "depth": int(depth),
                            "query_idx": int(query_idx),
                            "anchor_found": 0,
                        }
                    )
    return pd.DataFrame(rows)


def build_event_rows(
    subsets: Sequence[str],
    run_tokens: Sequence[str],
    depths: Sequence[int],
) -> pd.DataFrame:
    resource_cache: Dict[Tuple[str, str], SubsetResources] = {}
    rows: List[Dict[str, Any]] = []

    for subset in subsets:
        records_path, metrics_path = _resolve_subset_run_paths(subset, run_tokens)
        tree_version = _extract_tree_version_from_path(records_path)
        resources = _load_subset_resources(subset, tree_version, resource_cache)
        records_by_query = _load_iter_records(records_path)
        metrics_by_query = _load_iter_metrics(metrics_path)

        for depth in depths:
            for query_idx in range(resources.num_queries):
                query_rows = _build_anchor_event_rows(
                    subset=subset,
                    records_by_iter=records_by_query.get(query_idx, {}),
                    metrics_by_iter=metrics_by_query.get(query_idx, {}),
                    query_idx=query_idx,
                    depth=int(depth),
                    resources=resources,
                )
                for row in query_rows:
                    row["records_path"] = str(records_path)
                    row["metrics_path"] = str(metrics_path)
                    row["tree_version"] = tree_version
                    row["num_queries_total_subset"] = int(resources.num_queries)
                    rows.append(row)

    return pd.DataFrame(rows)


def build_summary(first_anchor_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if first_anchor_df.empty:
        return pd.DataFrame(rows)

    for depth in sorted(first_anchor_df["depth"].dropna().astype(int).unique().tolist()):
        depth_df = first_anchor_df[first_anchor_df["depth"] == depth].copy()
        for subset in ["overall"] + sorted(depth_df["subset"].dropna().unique().tolist()):
            subset_df = depth_df if subset == "overall" else depth_df[depth_df["subset"] == subset]
            if subset_df.empty:
                continue
            anchor_df = subset_df[subset_df["anchor_found"] == 1].copy()
            with_next_df = anchor_df[anchor_df["has_next_iter"] == 1].copy() if "has_next_iter" in anchor_df.columns else anchor_df.copy()
            drift_df = with_next_df[with_next_df["next_top100_anymiss"] == 1].copy()
            rows.append(
                {
                    "subset": subset,
                    "depth": int(depth),
                    "num_queries_total": int(len(subset_df)),
                    "anchor_queries": int(anchor_df.shape[0]),
                    "AnchorQueryRate": 100.0 * float(anchor_df.shape[0] / max(1, subset_df.shape[0])),
                    "AnchorWithNextCount": int(with_next_df.shape[0]),
                    "EndMiss@10|FirstAnchor": 100.0 * _mean(anchor_df["final_top10_miss"]) if not anchor_df.empty else float("nan"),
                    "HitToSuccess@10|FirstAnchor": 100.0 * _mean(anchor_df["final_top10_hit"]) if not anchor_df.empty else float("nan"),
                    "NextTop10DriftRate|FirstAnchor": 100.0 * _mean(with_next_df["next_top10_anymiss"]) if not with_next_df.empty else float("nan"),
                    "NextTop100DriftRate|FirstAnchor": 100.0 * _mean(with_next_df["next_top100_anymiss"]) if not with_next_df.empty else float("nan"),
                    "NextCtxDriftRate|FirstAnchor": 100.0 * _mean(with_next_df["next_ctx_anymiss"]) if not with_next_df.empty else float("nan"),
                    "EndMiss@10|FirstAnchor&NextTop100Drift": 100.0 * _mean(drift_df["final_top10_miss"]) if not drift_df.empty else float("nan"),
                    "MeanAnchorNDCG@10|FirstAnchor": _mean(anchor_df["anchor_ndcg10"]) if not anchor_df.empty else float("nan"),
                    "MeanNextMinusAnchorNDCG@10|FirstAnchor": _mean(anchor_df["next_minus_anchor_ndcg10"]) if not anchor_df.empty else float("nan"),
                    "MeanEndMinusAnchorNDCG@10|FirstAnchor": _mean(anchor_df["end_minus_anchor_ndcg10"]) if not anchor_df.empty else float("nan"),
                    "MeanBestFutureMinusAnchorNDCG@10|FirstAnchor": _mean(anchor_df["best_future_minus_anchor_ndcg10"]) if not anchor_df.empty else float("nan"),
                    "MeanNextTop10OnAnchorShare|FirstAnchor": _mean(with_next_df["next_top10_on_anchor_share"]) if not with_next_df.empty else float("nan"),
                    "MeanNextTop100OnAnchorShare|FirstAnchor": _mean(with_next_df["next_top100_on_anchor_share"]) if not with_next_df.empty else float("nan"),
                    "MeanNextCtxOnAnchorShare|FirstAnchor": _mean(with_next_df["next_ctx_on_anchor_share"]) if not with_next_df.empty else float("nan"),
                    "MeanProxyNextTop10OffPct|FirstAnchor": _mean(with_next_df["proxy_next_top10_off_pct"]) if not with_next_df.empty else float("nan"),
                    "MeanProxyNextTop100OffPct|FirstAnchor": _mean(with_next_df["proxy_next_top100_off_pct"]) if not with_next_df.empty else float("nan"),
                    "MeanEndBestGoldRank|FirstAnchor": _mean(anchor_df["end_best_gold_rank"]) if not anchor_df.empty else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def build_examples(first_anchor_df: pd.DataFrame, limit: int, headline_depth: int) -> pd.DataFrame:
    if first_anchor_df.empty:
        return first_anchor_df
    df = first_anchor_df[
        (first_anchor_df["depth"] == int(headline_depth))
        & (first_anchor_df["anchor_found"] == 1)
        & (first_anchor_df["next_top100_anymiss"] == 1)
        & (first_anchor_df["final_top10_miss"] == 1)
    ].copy()
    if df.empty:
        return df
    df = df.sort_values(
        by=["anchor_ndcg10", "end_minus_anchor_ndcg10", "subset", "query_idx"],
        ascending=[False, True, True, True],
    )
    keep_cols = [
        "subset",
        "depth",
        "query_idx",
        "iter_anchor",
        "query",
        "anchor_query_for_retrieval",
        "next_query_for_retrieval",
        "final_query_for_retrieval",
        "anchor_branches_json",
        "context_branches_json",
        "anchor_ndcg10",
        "next_ndcg10",
        "final_ndcg10",
        "best_future_ndcg10",
        "next_top10_on_anchor_share",
        "next_top100_on_anchor_share",
        "next_ctx_on_anchor_share",
        "proxy_next_top10_off_pct",
        "proxy_next_top100_off_pct",
        "anchor_top10_doc_ids_json",
        "next_top10_doc_ids_json",
        "final_top10_doc_ids_json",
        "gold_doc_ids_json",
    ]
    return df[keep_cols].head(int(limit)).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze baseline3 gold-branch derailment.")
    parser.add_argument(
        "--subsets",
        nargs="*",
        default=DEFAULT_SUBSETS,
        help="BRIGHT subsets to analyze.",
    )
    parser.add_argument(
        "--run_tokens",
        nargs="*",
        default=DEFAULT_RUN_TOKENS,
        help="All tokens that must appear in selected baseline3 paths.",
    )
    parser.add_argument(
        "--depths",
        nargs="*",
        type=int,
        default=DEFAULT_DEPTHS,
        help="Branch depths to analyze.",
    )
    parser.add_argument(
        "--headline_depth",
        type=int,
        default=3,
        help="Depth used for qualitative examples and headline printing.",
    )
    parser.add_argument(
        "--examples_limit",
        type=int,
        default=50,
        help="Number of example rows to save.",
    )
    parser.add_argument(
        "--out_dir",
        default="results/BRIGHT/analysis/baseline3_derailment",
        help="Output directory for CSV files.",
    )
    args = parser.parse_args()

    event_rows_df = build_event_rows(
        subsets=[str(x) for x in args.subsets],
        run_tokens=[str(x) for x in args.run_tokens],
        depths=[int(x) for x in args.depths],
    )
    first_anchor_df = build_first_anchor_rows(
        event_rows_df=event_rows_df,
        subsets=[str(x) for x in args.subsets],
        depths=[int(x) for x in args.depths],
    )
    summary_df = build_summary(first_anchor_df)
    examples_df = build_examples(first_anchor_df, int(args.examples_limit), int(args.headline_depth))

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_path = out_dir / "baseline3_derailment_rows.csv"
    firsthit_path = out_dir / "baseline3_derailment_firsthit.csv"
    summary_path = out_dir / "baseline3_derailment_summary.csv"
    examples_path = out_dir / "baseline3_derailment_examples.csv"

    event_rows_df.to_csv(rows_path, index=False)
    first_anchor_df.to_csv(firsthit_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    examples_df.to_csv(examples_path, index=False)

    print(f"[saved] {rows_path}")
    print(f"[saved] {firsthit_path}")
    print(f"[saved] {summary_path}")
    print(f"[saved] {examples_path}")

    headline_df = summary_df[summary_df["depth"] == int(args.headline_depth)].copy()
    if not headline_df.empty:
        print("\n[headline summary]")
        keep_cols = [
            "subset",
            "depth",
            "AnchorQueryRate",
            "EndMiss@10|FirstAnchor",
            "HitToSuccess@10|FirstAnchor",
            "NextTop100DriftRate|FirstAnchor",
            "EndMiss@10|FirstAnchor&NextTop100Drift",
            "MeanNextTop100OnAnchorShare|FirstAnchor",
            "MeanProxyNextTop100OffPct|FirstAnchor",
        ]
        print(headline_df[keep_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
