#!/usr/bin/env python3
"""
Compare gold-grounded derailment behavior between baseline3 and round6.

This script deliberately keeps the retrieval-level evaluation shared while
allowing the anchor definition to follow each method's native state:

- baseline3 anchor: first gold-aligned `rewrite_context_paths` hit
- round6 anchor: first gold-aligned `selected_branches_after` hit

Examples
--------
python scripts/analysis/analyze_round6_vs_baseline3_derailment.py

python scripts/analysis/analyze_round6_vs_baseline3_derailment.py \
    --out_dir results/BRIGHT/analysis/round6_vs_baseline3_derailment
"""

import argparse
import json
import math
import pickle as pkl
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

DEFAULT_BASELINE_TOKENS = [
    "baseline3_leaf_only_loop",
    "agent_executor_v1_icl2",
    "PlTau=5.0",
    "RCT=10",
    "RSC=on",
]

DEFAULT_ROUND6_TOKENS = [
    "round6_mrr_selector_accum_meanscore_global_expandable_ended_reseat",
    "agent_executor_v1_icl2",
    "PlTau=5.0",
    "RCT=10",
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

    # Intent: reruns leave multiple compatible artifacts, so choose the latest one deterministically.
    latest = max(matches, key=lambda path: path.stat().st_mtime)
    print(f"[warn] Multiple {label} matches for subset={subset}; using latest mtime: {latest}")
    return latest


def _resolve_baseline_paths(subset: str, run_tokens: Sequence[str]) -> Tuple[Path, Path]:
    subset_root = REPO_ROOT / "results" / "BRIGHT" / subset
    record_paths = sorted(subset_root.glob("**/leaf_iter_records.jsonl"))
    metric_paths = sorted(subset_root.glob("**/leaf_iter_metrics.jsonl"))
    records_path = _select_matching_path(record_paths, run_tokens, "baseline records", subset)
    metrics_path = _select_matching_path(metric_paths, run_tokens, "baseline metrics", subset)
    return records_path, metrics_path


def _resolve_round6_path(subset: str, run_tokens: Sequence[str]) -> Path:
    subset_root = REPO_ROOT / "results" / "BRIGHT" / subset
    sample_paths = sorted(subset_root.glob("round6/**/all_eval_sample_dicts.pkl"))
    return _select_matching_path(sample_paths, run_tokens, "round6 samples", subset)


def _load_baseline_records(path: Path) -> Dict[int, Dict[int, Dict[str, Any]]]:
    by_query: Dict[int, Dict[int, Dict[str, Any]]] = {}
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
            by_query.setdefault(query_idx, {})[iter_idx] = rec
    return by_query


def _load_baseline_metrics(path: Path) -> Dict[int, Dict[int, Dict[str, Any]]]:
    by_query: Dict[int, Dict[int, Dict[str, Any]]] = {}
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
            by_query.setdefault(query_idx, {})[iter_idx] = rec
    return by_query


def _load_round6_samples(path: Path) -> List[Dict[str, Any]]:
    with open(path, "rb") as f:
        return list(pkl.load(f))


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


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


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


def _mean(series: pd.Series) -> float:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return float("nan")
    return float(valid.mean())


def _best_gold_rank(doc_ids: Sequence[str], gold_doc_ids: Sequence[str]) -> Optional[int]:
    gold_set = set(str(x) for x in (gold_doc_ids or []))
    for rank_idx, doc_id in enumerate(doc_ids):
        if str(doc_id) in gold_set:
            return int(rank_idx + 1)
    return None


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


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
    # Intent: baseline anchor should only count rewrite evidence that already points into a gold branch region.
    return sorted({branch for branch in context_branches if branch in gold_branch_set})


def _path_consistent_with_branch(path: Sequence[int], branch: Sequence[int]) -> bool:
    if not path or not branch:
        return False
    limit = min(len(path), len(branch))
    return tuple(path[:limit]) == tuple(branch[:limit])


def _anchor_branches_from_selected(
    selected_paths: Sequence[Sequence[int]],
    gold_paths: Sequence[Sequence[int]],
    depth: int,
) -> List[Tuple[int, ...]]:
    gold_branches = _gold_branches_at_depth(gold_paths, depth)
    anchors = []
    for gold_branch in gold_branches:
        if any(_path_consistent_with_branch(path, gold_branch) for path in selected_paths):
            anchors.append(gold_branch)
    # Intent: round6 anchor should reflect gold branch regions already covered by the explicit branch state.
    return sorted({tuple(path) for path in anchors})


def _count_on_anchor(paths: Sequence[Sequence[int]], anchor_branches: Sequence[Tuple[int, ...]], topk: int) -> int:
    count = 0
    for path in list(paths)[:topk]:
        if any(_has_prefix(path, anchor) for anchor in anchor_branches):
            count += 1
    return int(count)


def _build_baseline_first_anchor_rows(
    subsets: Sequence[str],
    baseline_tokens: Sequence[str],
    depths: Sequence[int],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    resource_cache: Dict[Tuple[str, str], SubsetResources] = {}

    for subset in subsets:
        records_path, metrics_path = _resolve_baseline_paths(subset, baseline_tokens)
        tree_version = _extract_tree_version_from_path(records_path)
        resources = _load_subset_resources(subset, tree_version, resource_cache)
        records_by_query = _load_baseline_records(records_path)
        metrics_by_query = _load_baseline_metrics(metrics_path)

        for depth in depths:
            for query_idx in range(resources.num_queries):
                gold = resources.gold_by_query.get(query_idx)
                row_base = {
                    "model": "baseline3",
                    "subset": subset,
                    "depth": int(depth),
                    "query_idx": int(query_idx),
                    "anchor_type": "rewrite_context",
                    "tree_version": tree_version,
                    "anchor_found": 0,
                    "records_path": str(records_path),
                    "num_queries_total_subset": int(resources.num_queries),
                }
                if gold is None or not gold.gold_paths:
                    rows.append(dict(row_base))
                    continue

                query_records = records_by_query.get(query_idx, {})
                query_metrics = metrics_by_query.get(query_idx, {})
                iter_keys = sorted(query_records.keys())
                if not iter_keys:
                    rows.append(dict(row_base))
                    continue

                anchor_iter = None
                anchor_rec = None
                anchor_branches: List[Tuple[int, ...]] = []
                for iter_idx in iter_keys:
                    rec = query_records[iter_idx]
                    anchor_branches = _anchor_branches_from_context(
                        context_paths=rec.get("rewrite_context_paths", []) or [],
                        gold_paths=gold.gold_paths,
                        depth=int(depth),
                    )
                    if anchor_branches:
                        anchor_iter = int(iter_idx)
                        anchor_rec = rec
                        break

                if anchor_iter is None or anchor_rec is None:
                    rows.append(dict(row_base))
                    continue

                final_iter_candidates = [iter_idx for iter_idx in iter_keys if iter_idx >= 0]
                final_iter = max(final_iter_candidates) if final_iter_candidates else None
                final_rec = query_records.get(final_iter, {}) if final_iter is not None else {}
                next_rec = query_records.get(anchor_iter + 1)

                anchor_doc_ids = [str(x) for x in (anchor_rec.get("retrieved_doc_ids", []) or [])]
                next_doc_ids = [str(x) for x in (next_rec.get("retrieved_doc_ids", []) or [])] if next_rec else []
                final_doc_ids = [str(x) for x in (final_rec.get("retrieved_doc_ids", []) or [])]

                anchor_ndcg10 = _compute_ndcg(anchor_doc_ids, gold.gold_doc_ids, k=10) * 100.0
                next_ndcg10 = float("nan")
                if next_rec is not None:
                    next_ndcg10 = _safe_float(query_metrics.get(anchor_iter + 1, {}).get("nDCG@10"))
                    if math.isnan(next_ndcg10):
                        next_ndcg10 = _compute_ndcg(next_doc_ids, gold.gold_doc_ids, k=10) * 100.0

                final_ndcg10 = _safe_float(query_metrics.get(final_iter, {}).get("nDCG@10")) if final_iter is not None else float("nan")
                if final_iter is not None and math.isnan(final_ndcg10):
                    final_ndcg10 = _compute_ndcg(final_doc_ids, gold.gold_doc_ids, k=10) * 100.0

                next_top10_on_anchor_count = int(_count_on_anchor(next_rec.get("retrieved_paths", []) or [], anchor_branches, 10)) if next_rec else math.nan
                next_top100_on_anchor_count = int(_count_on_anchor(next_rec.get("retrieved_paths", []) or [], anchor_branches, 100)) if next_rec else math.nan
                next_ctx_paths = (next_rec.get("rewrite_context_paths", []) or [])[:10] if next_rec else []
                next_ctx_on_anchor_count = int(_count_on_anchor(next_ctx_paths, anchor_branches, 10)) if next_rec else math.nan
                final_top10_hit = int(any(str(doc_id) in set(gold.gold_doc_ids) for doc_id in final_doc_ids[:10])) if final_doc_ids else 0

                row = dict(row_base)
                row.update(
                    {
                        "anchor_found": 1,
                        "iter_anchor": int(anchor_iter),
                        "has_next_iter": int(next_rec is not None),
                        "query": str(anchor_rec.get("query", "") or ""),
                        "anchor_branches_json": _json_dumps([list(path) for path in anchor_branches]),
                        "anchor_query_for_retrieval": str(anchor_rec.get("query_for_retrieval", "") or ""),
                        "next_query_for_retrieval": str(next_rec.get("query_for_retrieval", "") or "") if next_rec else "",
                        "final_query_for_retrieval": str(final_rec.get("query_for_retrieval", "") or ""),
                        "anchor_ndcg10": float(anchor_ndcg10),
                        "next_ndcg10": float(next_ndcg10),
                        "final_ndcg10": float(final_ndcg10),
                        "next_minus_anchor_ndcg10": float(next_ndcg10 - anchor_ndcg10) if next_rec is not None and not math.isnan(next_ndcg10) else float("nan"),
                        "end_minus_anchor_ndcg10": float(final_ndcg10 - anchor_ndcg10) if not math.isnan(final_ndcg10) else float("nan"),
                        "next_top10_on_anchor_share": float(next_top10_on_anchor_count / 10.0) if next_rec else float("nan"),
                        "next_top100_on_anchor_share": float(next_top100_on_anchor_count / 100.0) if next_rec else float("nan"),
                        "next_ctx_on_anchor_share": float(next_ctx_on_anchor_count / max(1, len(next_ctx_paths))) if next_rec else float("nan"),
                        "next_top10_anymiss": int(next_top10_on_anchor_count == 0) if next_rec else math.nan,
                        "next_top100_anymiss": int(next_top100_on_anchor_count == 0) if next_rec else math.nan,
                        "next_ctx_anymiss": int(next_ctx_on_anchor_count == 0) if next_rec else math.nan,
                        "next_prehit_top100_on_anchor_share": float("nan"),
                        "next_prehit_top100_anymiss": float("nan"),
                        "final_top10_hit": int(final_top10_hit),
                        "final_top10_miss": int(1 - final_top10_hit),
                        "end_best_gold_rank": _best_gold_rank(final_doc_ids, gold.gold_doc_ids),
                    }
                )
                rows.append(row)

    return pd.DataFrame(rows)


def _build_round6_first_anchor_rows(
    subsets: Sequence[str],
    round6_tokens: Sequence[str],
    depths: Sequence[int],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    resource_cache: Dict[Tuple[str, str], SubsetResources] = {}

    for subset in subsets:
        samples_path = _resolve_round6_path(subset, round6_tokens)
        tree_version = _extract_tree_version_from_path(samples_path)
        resources = _load_subset_resources(subset, tree_version, resource_cache)
        samples = _load_round6_samples(samples_path)

        for depth in depths:
            for query_idx in range(resources.num_queries):
                gold = resources.gold_by_query.get(query_idx)
                row_base = {
                    "model": "round6",
                    "subset": subset,
                    "depth": int(depth),
                    "query_idx": int(query_idx),
                    "anchor_type": "selected_branches_after",
                    "tree_version": tree_version,
                    "anchor_found": 0,
                    "records_path": str(samples_path),
                    "num_queries_total_subset": int(resources.num_queries),
                }
                if gold is None or not gold.gold_paths or query_idx >= len(samples):
                    rows.append(dict(row_base))
                    continue

                sample = samples[query_idx] or {}
                iter_records = sample.get("iter_records", []) or []
                if not iter_records:
                    rows.append(dict(row_base))
                    continue

                sample_gold_doc_ids = [str(x) for x in (sample.get("gold_doc_ids", []) or [])] or list(gold.gold_doc_ids)
                sample_gold_paths = [tuple(path) for path in (sample.get("gold_paths", []) or []) if path] or list(gold.gold_paths)

                anchor_iter = None
                anchor_rec = None
                anchor_branches: List[Tuple[int, ...]] = []
                for iter_idx, rec in enumerate(iter_records):
                    anchor_branches = _anchor_branches_from_selected(
                        selected_paths=rec.get("selected_branches_after", []) or [],
                        gold_paths=sample_gold_paths,
                        depth=int(depth),
                    )
                    if anchor_branches:
                        anchor_iter = int(iter_idx)
                        anchor_rec = rec
                        break

                if anchor_iter is None or anchor_rec is None:
                    rows.append(dict(row_base))
                    continue

                next_rec = iter_records[anchor_iter + 1] if (anchor_iter + 1) < len(iter_records) else None
                final_rec = iter_records[-1]
                final_doc_ids = [str(x) for x in (final_rec.get("active_eval_doc_ids", []) or [])]

                anchor_doc_ids = [str(x) for x in (anchor_rec.get("active_eval_doc_ids", []) or [])]
                next_doc_ids = [str(x) for x in (next_rec.get("active_eval_doc_ids", []) or [])] if next_rec else []

                anchor_ndcg10 = _safe_float(anchor_rec.get("metrics", {}).get("nDCG@10"))
                if math.isnan(anchor_ndcg10):
                    anchor_ndcg10 = _compute_ndcg(anchor_doc_ids, sample_gold_doc_ids, k=10) * 100.0
                next_ndcg10 = _safe_float(next_rec.get("metrics", {}).get("nDCG@10")) if next_rec else float("nan")
                if next_rec is not None and math.isnan(next_ndcg10):
                    next_ndcg10 = _compute_ndcg(next_doc_ids, sample_gold_doc_ids, k=10) * 100.0
                final_ndcg10 = _safe_float(final_rec.get("metrics", {}).get("nDCG@10"))
                if math.isnan(final_ndcg10):
                    final_ndcg10 = _compute_ndcg(final_doc_ids, sample_gold_doc_ids, k=10) * 100.0

                next_top10_on_anchor_count = int(_count_on_anchor(next_rec.get("active_eval_paths", []) or [], anchor_branches, 10)) if next_rec else math.nan
                next_top100_on_anchor_count = int(_count_on_anchor(next_rec.get("active_eval_paths", []) or [], anchor_branches, 100)) if next_rec else math.nan
                next_prehit_top100_on_anchor_count = int(_count_on_anchor(next_rec.get("pre_hit_paths", []) or [], anchor_branches, 100)) if next_rec else math.nan
                final_top10_hit = int(any(str(doc_id) in set(sample_gold_doc_ids) for doc_id in final_doc_ids[:10])) if final_doc_ids else 0

                row = dict(row_base)
                row.update(
                    {
                        "anchor_found": 1,
                        "iter_anchor": int(anchor_iter),
                        "has_next_iter": int(next_rec is not None),
                        "query": str(sample.get("original_query") or sample.get("query") or ""),
                        "anchor_branches_json": _json_dumps([list(path) for path in anchor_branches]),
                        "anchor_query_for_retrieval": str(anchor_rec.get("query_post") or anchor_rec.get("query_pre") or ""),
                        "next_query_for_retrieval": str(next_rec.get("query_post") or next_rec.get("query_pre") or "") if next_rec else "",
                        "final_query_for_retrieval": str(final_rec.get("query_post") or final_rec.get("query_pre") or ""),
                        "anchor_ndcg10": float(anchor_ndcg10),
                        "next_ndcg10": float(next_ndcg10),
                        "final_ndcg10": float(final_ndcg10),
                        "next_minus_anchor_ndcg10": float(next_ndcg10 - anchor_ndcg10) if next_rec is not None and not math.isnan(next_ndcg10) else float("nan"),
                        "end_minus_anchor_ndcg10": float(final_ndcg10 - anchor_ndcg10) if not math.isnan(final_ndcg10) else float("nan"),
                        "next_top10_on_anchor_share": float(next_top10_on_anchor_count / 10.0) if next_rec else float("nan"),
                        "next_top100_on_anchor_share": float(next_top100_on_anchor_count / 100.0) if next_rec else float("nan"),
                        "next_ctx_on_anchor_share": float("nan"),
                        "next_top10_anymiss": int(next_top10_on_anchor_count == 0) if next_rec else math.nan,
                        "next_top100_anymiss": int(next_top100_on_anchor_count == 0) if next_rec else math.nan,
                        "next_ctx_anymiss": float("nan"),
                        "next_prehit_top100_on_anchor_share": float(next_prehit_top100_on_anchor_count / 100.0) if next_rec else float("nan"),
                        "next_prehit_top100_anymiss": int(next_prehit_top100_on_anchor_count == 0) if next_rec else math.nan,
                        "final_top10_hit": int(final_top10_hit),
                        "final_top10_miss": int(1 - final_top10_hit),
                        "end_best_gold_rank": _best_gold_rank(final_doc_ids, sample_gold_doc_ids),
                    }
                )
                rows.append(row)

    return pd.DataFrame(rows)


def _build_summary(rows_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if rows_df.empty:
        return pd.DataFrame(rows)

    for model in sorted(rows_df["model"].dropna().unique().tolist()):
        model_df = rows_df[rows_df["model"] == model].copy()
        for depth in sorted(model_df["depth"].dropna().astype(int).unique().tolist()):
            depth_df = model_df[model_df["depth"] == depth].copy()
            for subset in ["overall"] + sorted(depth_df["subset"].dropna().unique().tolist()):
                subset_df = depth_df if subset == "overall" else depth_df[depth_df["subset"] == subset]
                if subset_df.empty:
                    continue
                anchor_df = subset_df[subset_df["anchor_found"] == 1].copy()
                with_next_df = anchor_df[anchor_df["has_next_iter"] == 1].copy()
                drift_df = with_next_df[with_next_df["next_top100_anymiss"] == 1].copy()
                row = {
                    "model": model,
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
                    "EndMiss@10|FirstAnchor&NextTop100Drift": 100.0 * _mean(drift_df["final_top10_miss"]) if not drift_df.empty else float("nan"),
                    "MeanAnchorNDCG@10|FirstAnchor": _mean(anchor_df["anchor_ndcg10"]) if not anchor_df.empty else float("nan"),
                    "MeanNextMinusAnchorNDCG@10|FirstAnchor": _mean(anchor_df["next_minus_anchor_ndcg10"]) if not anchor_df.empty else float("nan"),
                    "MeanEndMinusAnchorNDCG@10|FirstAnchor": _mean(anchor_df["end_minus_anchor_ndcg10"]) if not anchor_df.empty else float("nan"),
                    "MeanNextTop10OnAnchorShare|FirstAnchor": _mean(with_next_df["next_top10_on_anchor_share"]) if not with_next_df.empty else float("nan"),
                    "MeanNextTop100OnAnchorShare|FirstAnchor": _mean(with_next_df["next_top100_on_anchor_share"]) if not with_next_df.empty else float("nan"),
                    "MeanEndBestGoldRank|FirstAnchor": _mean(anchor_df["end_best_gold_rank"]) if not anchor_df.empty else float("nan"),
                }
                if model == "baseline3":
                    row["NextCtxDriftRate|FirstAnchor"] = 100.0 * _mean(with_next_df["next_ctx_anymiss"]) if not with_next_df.empty else float("nan")
                    row["MeanNextCtxOnAnchorShare|FirstAnchor"] = _mean(with_next_df["next_ctx_on_anchor_share"]) if not with_next_df.empty else float("nan")
                    row["NextPreHitTop100DriftRate|FirstAnchor"] = float("nan")
                    row["MeanNextPreHitTop100OnAnchorShare|FirstAnchor"] = float("nan")
                else:
                    row["NextCtxDriftRate|FirstAnchor"] = float("nan")
                    row["MeanNextCtxOnAnchorShare|FirstAnchor"] = float("nan")
                    row["NextPreHitTop100DriftRate|FirstAnchor"] = 100.0 * _mean(with_next_df["next_prehit_top100_anymiss"]) if not with_next_df.empty else float("nan")
                    row["MeanNextPreHitTop100OnAnchorShare|FirstAnchor"] = _mean(with_next_df["next_prehit_top100_on_anchor_share"]) if not with_next_df.empty else float("nan")
                rows.append(row)
    return pd.DataFrame(rows)


def _build_compare_delta(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    keep_cols = [
        "AnchorQueryRate",
        "EndMiss@10|FirstAnchor",
        "HitToSuccess@10|FirstAnchor",
        "NextTop10DriftRate|FirstAnchor",
        "NextTop100DriftRate|FirstAnchor",
        "MeanNextTop10OnAnchorShare|FirstAnchor",
        "MeanNextTop100OnAnchorShare|FirstAnchor",
        "MeanEndMinusAnchorNDCG@10|FirstAnchor",
    ]
    rows: List[Dict[str, Any]] = []
    group_cols = ["subset", "depth"]
    for (subset, depth), group in summary_df.groupby(group_cols, dropna=False):
        model_map = {str(row["model"]): row for _, row in group.iterrows()}
        baseline = model_map.get("baseline3")
        round6 = model_map.get("round6")
        if baseline is None or round6 is None:
            continue
        row = {
            "subset": subset,
            "depth": int(depth),
            "baseline_anchor_type": "rewrite_context",
            "round6_anchor_type": "selected_branches_after",
        }
        for col in keep_cols:
            row[f"baseline_{col}"] = baseline.get(col)
            row[f"round6_{col}"] = round6.get(col)
            row[f"delta_round6_minus_baseline_{col}"] = _safe_float(round6.get(col)) - _safe_float(baseline.get(col))
        row["round6_NextPreHitTop100DriftRate|FirstAnchor"] = round6.get("NextPreHitTop100DriftRate|FirstAnchor")
        row["round6_MeanNextPreHitTop100OnAnchorShare|FirstAnchor"] = round6.get("MeanNextPreHitTop100OnAnchorShare|FirstAnchor")
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare gold-grounded derailment between baseline3 and round6.")
    parser.add_argument(
        "--subsets",
        nargs="*",
        default=DEFAULT_SUBSETS,
        help="BRIGHT subsets to analyze.",
    )
    parser.add_argument(
        "--depths",
        nargs="*",
        type=int,
        default=DEFAULT_DEPTHS,
        help="Branch depths to analyze.",
    )
    parser.add_argument(
        "--baseline_tokens",
        nargs="*",
        default=DEFAULT_BASELINE_TOKENS,
        help="All tokens that must appear in baseline paths.",
    )
    parser.add_argument(
        "--round6_tokens",
        nargs="*",
        default=DEFAULT_ROUND6_TOKENS,
        help="All tokens that must appear in round6 paths.",
    )
    parser.add_argument(
        "--out_dir",
        default="results/BRIGHT/analysis/round6_vs_baseline3_derailment",
        help="Output directory for CSV files.",
    )
    args = parser.parse_args()

    baseline_rows_df = _build_baseline_first_anchor_rows(
        subsets=[str(x) for x in args.subsets],
        baseline_tokens=[str(x) for x in args.baseline_tokens],
        depths=[int(x) for x in args.depths],
    )
    round6_rows_df = _build_round6_first_anchor_rows(
        subsets=[str(x) for x in args.subsets],
        round6_tokens=[str(x) for x in args.round6_tokens],
        depths=[int(x) for x in args.depths],
    )

    all_rows_df = pd.concat([baseline_rows_df, round6_rows_df], ignore_index=True)
    summary_df = _build_summary(all_rows_df)
    compare_df = _build_compare_delta(summary_df)

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = out_dir / "baseline3_first_anchor.csv"
    round6_path = out_dir / "round6_first_anchor.csv"
    summary_path = out_dir / "derailment_summary_by_model.csv"
    compare_path = out_dir / "derailment_compare_delta.csv"

    baseline_rows_df.to_csv(baseline_path, index=False)
    round6_rows_df.to_csv(round6_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    compare_df.to_csv(compare_path, index=False)

    print(f"[saved] {baseline_path}")
    print(f"[saved] {round6_path}")
    print(f"[saved] {summary_path}")
    print(f"[saved] {compare_path}")

    headline = compare_df[(compare_df["subset"] == "overall") & (compare_df["depth"] == 3)].copy()
    if not headline.empty:
        print("\n[headline delta depth=3]")
        print(headline.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
