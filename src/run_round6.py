import asyncio
import json
import logging
import os
import pickle as pkl
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from json_repair import repair_json
from tqdm.autonotebook import tqdm

from cache_utils import _prompt_cache_key, append_jsonl
from flat_then_tree import FlatHit
from hyperparams import HyperParams
from llm_apis import GenAIAPI, VllmAPI
from retrievers import build_retriever
from retrievers.diver import DiverEmbeddingModel
from rewrite_prompts import REWRITE_PROMPT_TEMPLATES
from tree_objects import InferSample, SemanticNode
from utils import (
    compute_ndcg,
    compute_node_registry,
    compute_recall,
    get_node_id,
    normalize_embeddings,
    pad_node_embeddings_to_registry,
    save_exp,
    setup_logger,
)


CATEGORY_ORDER = ["Theory", "Entity", "Example", "Other"]


@dataclass
class Round5Sample:
    original_query: str
    gold_paths: List[Tuple[int, ...]]
    gold_doc_ids: List[str]
    excluded_ids: List[str]
    last_rewrite: str = ""
    last_query: str = ""
    selected_branches: List[Tuple[int, ...]] = None
    rewrite_history: List[Dict[str, Any]] = None
    iter_records: List[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.selected_branches is None:
            self.selected_branches = []
        if self.rewrite_history is None:
            self.rewrite_history = []
        if self.iter_records is None:
            self.iter_records = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "gold_paths": [list(p) for p in self.gold_paths],
            "gold_doc_ids": list(self.gold_doc_ids),
            "excluded_ids": list(self.excluded_ids),
            "last_rewrite": self.last_rewrite,
            "last_query": self.last_query,
            "selected_branches": [list(p) for p in self.selected_branches],
            "rewrite_history": self.rewrite_history,
            "iter_records": self.iter_records,
        }


def _ordered_doc_keys(docs: Dict[str, str]) -> List[str]:
    known = [key for key in CATEGORY_ORDER if str((docs or {}).get(key, "")).strip()]
    extra: List[str] = []
    for key, val in (docs or {}).items():
        key_s = str(key or "").strip()
        if not key_s or key_s in CATEGORY_ORDER:
            continue
        if str(val or "").strip():
            extra.append(key_s)
    return known + extra


def _clean_docs_map(docs: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, val in (docs or {}).items():
        key_s = str(key or "").strip()
        if not key_s:
            continue
        if isinstance(val, (dict, list)):
            val = json.dumps(val, ensure_ascii=False)
        val_s = str(val or "").strip()
        if val_s:
            out[key_s] = val_s
    return out


def _flatten_docs(docs: Dict[str, str]) -> str:
    pieces: List[str] = []
    for key in _ordered_doc_keys(docs):
        text = str(docs.get(key, "")).strip()
        if text:
            pieces.append(text)
    return "\n".join(pieces).strip()


def _extract_json_candidate(raw: str) -> str:
    text = str(raw or "").strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    if "```" in text:
        parts = text.split("```")
        fenced = [parts[i] for i in range(1, len(parts), 2)]
        if fenced:
            text = fenced[-1].strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
    return text


def _recursive_get(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for val in obj.values():
            got = _recursive_get(val, key)
            if got is not None:
                return got
    if isinstance(obj, list):
        for item in obj:
            got = _recursive_get(item, key)
            if got is not None:
                return got
    return None


def _parse_possible_answer_docs(text: str) -> Tuple[Dict[str, str], str]:
    candidate = _extract_json_candidate(text)
    obj: Any = None
    try:
        obj = json.loads(candidate)
    except Exception:
        try:
            obj = repair_json(candidate, return_objects=True)
        except Exception:
            obj = None

    if isinstance(obj, dict):
        docs_obj = _recursive_get(obj, "Possible_Answer_Docs")
        if docs_obj is None:
            docs_obj = _recursive_get(obj, "possible_answer_docs")
        if isinstance(docs_obj, dict):
            docs = _clean_docs_map(docs_obj)
            return docs, _flatten_docs(docs)
        if isinstance(docs_obj, list):
            docs = {f"Doc_{i + 1}": str(v).strip() for i, v in enumerate(docs_obj) if str(v).strip()}
            docs = _clean_docs_map(docs)
            return docs, _flatten_docs(docs)

        raw_rewrite = obj.get("rewrite")
        if isinstance(raw_rewrite, (dict, list)):
            raw_rewrite = json.dumps(raw_rewrite, ensure_ascii=False)
        rewrite = str(raw_rewrite or "").strip()
        if rewrite:
            return {}, rewrite

    return {}, str(candidate or "").strip()


def _get_relevance_definition(subset_name: str) -> str:
    name = str(subset_name or "").strip().lower()
    relevance_definitions = {
        "leetcode": "The relevance between queries and positive documents is defined by whether the coding problem "
        "(i.e., query) involves the same algorithm and/or data structure. The queries and documents are "
        "problems and solutions from LeetCode. The problem descriptions are used as queries Q, and the "
        "positive documents D+Q are solved problems (with solutions) that were annotated as similar "
        "problems by LeetCode.",
        "theoremqa_questions": "A query is relevant to a document if the document references the same/similar theorem used in the query.",
        "pony": "The relevance between queries and positive documents is defined by whether the coding problem "
        "(i.e., query) requires the corresponding syntax documentation.",
        "stackexchange": "A document is considered relevant to a query if it can be cited in an accepted or highly voted "
        "answer that helps reason through the query with critical concepts or theories.",
        "scifact": "A document is relevant if it contains sufficient, non-redundant, and minimal evidence (rationale sentences) "
        "that can be used to determine the veracity (either support OR refutation) of a specific scientific claim. "
        "The query is a scientific claim, and the documents are abstracts of scientific papers.",
        "nq": "A document is relevant to a query if it contains the answer to the question posed in the query. "
        "The query is a natural language web search question, and the documents are passages from wikipedia "
        "that may contain the answer.",
        "fiqa": "A document is considered relevant if it contains the specific information needed to answer the associated question.",
        "scidocs": "A query is a source scientific paper (represented by its title and abstract) and a document is a candidate paper from the corpus. "
        "A gold document is defined as relevant because it is either directly cited by the query paper or co-viewed "
        "(accessed in the same user session) with the query paper in user activity logs.",
    }
    relevance_definitions["aops"] = relevance_definitions["theoremqa_questions"]
    relevance_definitions["theoremqa_theorems"] = relevance_definitions["theoremqa_questions"]
    relevance_definitions["theoq"] = relevance_definitions["theoremqa_questions"]
    relevance_definitions["theot"] = relevance_definitions["theoremqa_questions"]
    return relevance_definitions.get(name, relevance_definitions["stackexchange"])


def _build_domain_route_hint(subset_name: str) -> str:
    relevance_definition = _get_relevance_definition(subset_name)

    return (
        "Use this relevance definition as your routing prior for retrieval rewrite:\n"
        f"{relevance_definition}"
    )


def _format_rewrite_prompt(
    template: str,
    *,
    original_query: str,
    previous_rewrite: str,
    leaf_descs: Sequence[str],
    domain_route_hint: str = "",
    relevance_definition: str = "",
    seen_direction_evidence: str = "",
) -> str:
    leaf_blob = "\n".join([x for x in leaf_descs if x])
    gate_blob = leaf_blob

    template = template.replace("Branch Context:\n{branch_descs}\n", "")
    template = template.replace("Retrieved Topic Cluster Summaries:\n{branch_descs}\n\n", "")
    template = template.replace("Retrieved Topic Cluster Summaries:\n{branch_descs}\n", "")
    template = template.replace("Topic Cluster Summaries:\n{branch_descs}\n", "")

    try:
        return template.format(
            original_query=(original_query or ""),
            previous_rewrite=(previous_rewrite or ""),
            previous_docs="",
            leaf_descs=leaf_blob,
            branch_descs="",
            gate_descs=gate_blob,
            domain_route_hint=domain_route_hint or "",
            relevance_definition=relevance_definition or "",
            seen_direction_evidence=seen_direction_evidence or "",
            corpus_categories="",
        )
    except KeyError:
        return (
            template
            .replace("{original_query}", original_query or "")
            .replace("{previous_rewrite}", previous_rewrite or "")
            .replace("{previous_docs}", "")
            .replace("{leaf_descs}", leaf_blob)
            .replace("{branch_descs}", "")
            .replace("{gate_descs}", gate_blob)
            .replace("{domain_route_hint}", domain_route_hint or "")
            .replace("{relevance_definition}", relevance_definition or "")
            .replace("{seen_direction_evidence}", seen_direction_evidence or "")
            .replace("{corpus_categories}", "")
        )


def _compose_next_query(original_query: str, rewrite_blob: str, query_pre: str) -> str:
    rewrite = str(rewrite_blob or "").strip()
    if rewrite:
        # Intent: keep retrieval target anchored to the original query while appending abstract evidence hints.
        return (str(original_query or "").strip() + " " + rewrite).strip()
    return str(query_pre or original_query or "").strip()


def _load_rewrite_cache(
    path: str,
    force_refresh: bool,
) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
    rewrite_map: Dict[str, str] = {}
    docs_map: Dict[str, Dict[str, str]] = {}
    if not path or force_refresh or (not os.path.exists(path)):
        return rewrite_map, docs_map

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            key = str(rec.get("key", "")).strip()
            if not key:
                continue
            if "rewritten_query" in rec:
                rewrite_map[key] = str(rec.get("rewritten_query", "") or "")
            raw_docs = rec.get("possible_answer_docs")
            if isinstance(raw_docs, dict):
                docs_map[key] = _clean_docs_map(raw_docs)
    return rewrite_map, docs_map


def _resolve_vllm_base_url(base_dir: str) -> Tuple[str, str]:
    env_url = str(os.getenv("VLLM_BASE_URL", "") or "").strip()
    if env_url:
        return env_url, "env:VLLM_BASE_URL"

    candidate_files = [
        os.path.join(base_dir, "scripts", "logs", "vllm_base_url.txt"),
        os.path.join(base_dir, "logs", "vllm_base_url.txt"),
    ]
    for path in candidate_files:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                value = str(f.read().strip())
        except Exception:
            value = ""
        if value:
            return value, f"file:{path}"

    # Intent: single-server fallback keeps compatibility when cluster metadata is unavailable.
    return "http://localhost:8000/v1", "fallback:localhost:8000"


def _topk_from_scores(scores: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    if topk <= 0 or scores.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    k = min(topk, scores.shape[0])
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx.astype(np.int64, copy=False), scores[idx].astype(np.float32, copy=False)


def _hits_from_scores(
    *,
    scores: np.ndarray,
    subset_indices: Sequence[int] | None,
    node_registry: Sequence[object],
    topk: int,
) -> List[FlatHit]:
    idx, vals = _topk_from_scores(scores, topk)
    hits: List[FlatHit] = []
    for ridx, score in zip(idx.tolist(), vals.tolist()):
        registry_idx = int(ridx if subset_indices is None else subset_indices[int(ridx)])
        node = node_registry[registry_idx]
        hits.append(
            FlatHit(
                registry_idx=registry_idx,
                path=tuple(node.path),
                score=float(score),
                is_leaf=node.is_leaf,
            )
        )
    return hits


def _hits_to_context_descs(
    hits: Sequence[FlatHit],
    node_registry: Sequence[object],
    topk: int,
    max_desc_len: int | None,
) -> List[str]:
    descs: List[str] = []
    seen: Set[int] = set()
    for h in hits:
        ridx = int(h.registry_idx)
        if ridx in seen:
            continue
        seen.add(ridx)
        desc = str(getattr(node_registry[ridx], "desc", "") or "").strip()
        if max_desc_len:
            desc = desc[:max_desc_len]
        if not desc:
            continue
        descs.append(desc)
        if len(descs) >= topk:
            break
    return descs


def _paths_to_ranked_doc_ids(
    paths: Sequence[Tuple[int, ...]],
    path_to_doc_id: Dict[Tuple[int, ...], str],
) -> List[str]:
    ranked: List[str] = []
    seen: Set[str] = set()
    for path in paths:
        doc_id = path_to_doc_id.get(tuple(path))
        if not doc_id:
            continue
        if doc_id in seen:
            continue
        seen.add(doc_id)
        ranked.append(doc_id)
    return ranked


def _collect_blocked_leaf_indices(
    archive_prefixes: Sequence[Tuple[int, ...]],
    leaf_indices_by_prefix: Dict[Tuple[int, ...], List[int]],
) -> Set[int]:
    blocked: Set[int] = set()
    for prefix in archive_prefixes:
        blocked.update(int(idx) for idx in leaf_indices_by_prefix.get(tuple(prefix), []))
    return blocked


def _filter_leaf_pool_by_blocked(
    leaf_pool_indices: Sequence[int],
    blocked_leaf_indices: Set[int],
) -> List[int]:
    if not blocked_leaf_indices:
        return [int(idx) for idx in leaf_pool_indices]
    return [int(idx) for idx in leaf_pool_indices if int(idx) not in blocked_leaf_indices]


def _top_context_descs_from_indices(
    *,
    query: str,
    leaf_pool_indices: Sequence[int],
    retriever: DiverEmbeddingModel,
    node_embs: np.ndarray,
    node_registry: Sequence[object],
    topk: int,
    max_desc_len: int | None,
) -> List[str]:
    hits = _retrieve_leaf_hits(
        query=query,
        leaf_pool_indices=leaf_pool_indices,
        retriever=retriever,
        node_embs=node_embs,
        node_registry=node_registry,
        topk=max(1, int(topk)),
    )
    return _hits_to_context_descs(
        hits,
        node_registry,
        topk=max(1, int(topk)),
        max_desc_len=max_desc_len,
    )


def _leaf_path_is_under_prefix(path: Tuple[int, ...], prefix: Tuple[int, ...]) -> bool:
    return (len(prefix) <= len(path)) and (tuple(path[: len(prefix)]) == tuple(prefix))


def _deepest_selected_ancestor_for_leaf(
    leaf_path: Tuple[int, ...],
    selected_paths: Sequence[Tuple[int, ...]],
) -> Optional[Tuple[int, ...]]:
    candidates = [
        tuple(path)
        for path in selected_paths
        if path and _leaf_path_is_under_prefix(tuple(leaf_path), tuple(path))
    ]
    if not candidates:
        return None
    return max(candidates, key=len)


def _selected_paths_covering_new_leafs(
    new_leaf_paths: Sequence[Tuple[int, ...]],
    selected_paths: Sequence[Tuple[int, ...]],
) -> List[Tuple[int, ...]]:
    covered: Set[Tuple[int, ...]] = set()
    for leaf_path in new_leaf_paths:
        for selected_path in selected_paths:
            selected_t = tuple(selected_path)
            if selected_t and _leaf_path_is_under_prefix(tuple(leaf_path), selected_t):
                covered.add(selected_t)
    return sorted(covered, key=lambda x: (len(x), x))


def _update_archive_records(
    archive_records: Dict[Tuple[int, ...], Dict[str, Any]],
    chosen_child_parent_map: Dict[Tuple[int, ...], List[Tuple[int, ...]]],
    iter_idx: int,
) -> Dict[Tuple[int, ...], Dict[str, Any]]:
    merged: Dict[Tuple[int, ...], Dict[str, Any]] = {}
    for parent_path, row in dict(archive_records or {}).items():
        parent_t = tuple(parent_path)
        merged[parent_t] = {
            "parent_path": parent_t,
            "selected_children": {tuple(x) for x in list(row.get("selected_children", []))},
            "first_iter": int(row.get("first_iter", iter_idx)),
            "last_iter": int(row.get("last_iter", iter_idx)),
        }

    for child_path, parent_paths in dict(chosen_child_parent_map or {}).items():
        child_t = tuple(child_path)
        for parent_path in list(parent_paths or []):
            parent_t = tuple(parent_path)
            record = merged.setdefault(
                parent_t,
                {
                    "parent_path": parent_t,
                    "selected_children": set(),
                    "first_iter": int(iter_idx),
                    "last_iter": int(iter_idx),
                },
            )
            record["selected_children"].add(child_t)
            record["last_iter"] = int(iter_idx)

    return merged


def _collect_unexplored_sibling_candidates(
    archive_records: Dict[Tuple[int, ...], Dict[str, Any]],
    child_branch_paths_by_path: Dict[Tuple[int, ...], List[Tuple[int, ...]]],
) -> Tuple[List[Tuple[int, ...]], Dict[Tuple[int, ...], List[Tuple[int, ...]]]]:
    candidate_paths: List[Tuple[int, ...]] = []
    candidate_parent_map: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = defaultdict(list)
    seen: Set[Tuple[int, ...]] = set()

    for parent_path in sorted(archive_records.keys(), key=lambda x: (len(x), x)):
        record = archive_records.get(tuple(parent_path), {})
        selected_children = {
            tuple(x)
            for x in list(record.get("selected_children", []))
        }
        for child_path in child_branch_paths_by_path.get(tuple(parent_path), []):
            child_t = tuple(child_path)
            if child_t in selected_children:
                continue
            candidate_parent_map[child_t].append(tuple(parent_path))
            if child_t in seen:
                continue
            seen.add(child_t)
            candidate_paths.append(child_t)

    return candidate_paths, {
        tuple(child): sorted({tuple(parent) for parent in parents}, key=lambda x: (len(x), x))
        for child, parents in candidate_parent_map.items()
    }


def _serialize_archive_records(
    archive_records: Dict[Tuple[int, ...], Dict[str, Any]],
    child_branch_paths_by_path: Dict[Tuple[int, ...], List[Tuple[int, ...]]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for parent_path in sorted(archive_records.keys(), key=lambda x: (len(x), x)):
        record = archive_records.get(tuple(parent_path), {})
        selected_children = sorted(
            {tuple(x) for x in list(record.get("selected_children", []))},
            key=lambda x: (len(x), x),
        )
        unexplored_children = sorted(
            [
                tuple(child_path)
                for child_path in child_branch_paths_by_path.get(tuple(parent_path), [])
                if tuple(child_path) not in set(selected_children)
            ],
            key=lambda x: (len(x), x),
        )
        rows.append(
            {
                "parent_path": [int(x) for x in parent_path],
                "selected_children": [[int(x) for x in path] for path in selected_children],
                "unexplored_children": [[int(x) for x in path] for path in unexplored_children],
                "first_iter": int(record.get("first_iter", -1)),
                "last_iter": int(record.get("last_iter", -1)),
            }
        )
    return rows


def _bank_entry_from_hits(
    *,
    hits: Sequence[FlatHit],
    gold_doc_ids: Sequence[str],
    path_to_doc_id: Dict[Tuple[int, ...], str],
    iter_idx: int,
    explore_step: bool,
    query_post: str,
) -> Dict[str, Any]:
    registry_indices = np.asarray([int(hit.registry_idx) for hit in hits], dtype=np.int32)
    scores = np.asarray([float(hit.score) for hit in hits], dtype=np.float32)
    doc_ids = _paths_to_ranked_doc_ids([tuple(hit.path) for hit in hits], path_to_doc_id)
    run_ndcg10 = compute_ndcg(doc_ids[:10], list(gold_doc_ids), k=10) * 100.0
    return {
        "iter": int(iter_idx),
        "explore_step": bool(explore_step),
        "query_post": str(query_post or ""),
        "registry_indices": registry_indices,
        "scores": scores,
        "ndcg10": float(run_ndcg10),
        "top_doc_ids": list(doc_ids[:10]),
    }


def _compute_retrieval_metrics(
    doc_ids: Sequence[str],
    gold_doc_ids: Sequence[str],
) -> Dict[str, float]:
    ranked_doc_ids = list(doc_ids)
    gold_ranked_doc_ids = list(gold_doc_ids)
    return {
        "nDCG@10": compute_ndcg(ranked_doc_ids[:10], gold_ranked_doc_ids, k=10) * 100,
        "Recall@10": compute_recall(ranked_doc_ids[:10], gold_ranked_doc_ids, k=10) * 100,
        "Recall@100": compute_recall(ranked_doc_ids[:100], gold_ranked_doc_ids, k=100) * 100,
        "Recall@all": compute_recall(ranked_doc_ids, gold_ranked_doc_ids, k=len(ranked_doc_ids)) * 100,
        "Coverage": float(len(ranked_doc_ids)),
    }


def _nan_retrieval_metrics() -> Dict[str, float]:
    # Intent: keep diagnostic metric columns stable even before the leaf-trigger bank becomes non-empty.
    return {
        "nDCG@10": float("nan"),
        "Recall@10": float("nan"),
        "Recall@100": float("nan"),
        "Recall@all": float("nan"),
        "Coverage": float("nan"),
    }


def _fuse_bank_entries(
    *,
    bank_entries: Sequence[Dict[str, Any]],
    mode: str,
    node_registry: Sequence[object],
    topk: int,
    rrf_k: int,
    blocked_leaf_indices: Optional[Set[int]] = None,
    allowed_leaf_indices: Optional[Set[int]] = None,
) -> List[FlatHit]:
    fused_scores: Dict[int, float] = {}
    best_rank: Dict[int, int] = {}
    blocked = {int(x) for x in list(blocked_leaf_indices or [])}
    allowed = None if allowed_leaf_indices is None else {int(x) for x in list(allowed_leaf_indices)}
    mode_s = str(mode or "rrf").strip().lower()
    if mode_s not in {"rrf", "max_score", "sum_score"}:
        raise ValueError(f"Unsupported round6 fusion mode: {mode}")

    for entry in bank_entries:
        registry_indices = np.asarray(entry.get("registry_indices", []), dtype=np.int64)
        scores = np.asarray(entry.get("scores", []), dtype=np.float32)
        for rank_idx, (registry_idx, score) in enumerate(zip(registry_indices.tolist(), scores.tolist()), start=1):
            ridx = int(registry_idx)
            if ridx in blocked:
                continue
            if (allowed is not None) and (ridx not in allowed):
                continue
            best_rank[ridx] = min(best_rank.get(ridx, rank_idx), rank_idx)
            if mode_s == "rrf":
                fused_scores[ridx] = fused_scores.get(ridx, 0.0) + (1.0 / float(rrf_k + rank_idx))
            elif mode_s == "max_score":
                fused_scores[ridx] = max(fused_scores.get(ridx, float("-inf")), float(score))
            else:
                fused_scores[ridx] = fused_scores.get(ridx, 0.0) + float(score)

    if not fused_scores:
        return []

    ordered = sorted(
        fused_scores.items(),
        key=lambda kv: (-float(kv[1]), int(best_rank.get(int(kv[0]), 10**9)), int(kv[0])),
    )[: max(1, int(topk))]
    hits: List[FlatHit] = []
    for registry_idx, score in ordered:
        node = node_registry[int(registry_idx)]
        hits.append(
            FlatHit(
                registry_idx=int(registry_idx),
                path=tuple(node.path),
                score=float(score),
                is_leaf=node.is_leaf,
            )
        )
    return hits


def _summarize_bank_entries(bank_entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "iter": int(entry.get("iter", -1)),
            "explore_step": bool(entry.get("explore_step", False)),
            "ndcg10": float(entry.get("ndcg10", 0.0)),
            "top_doc_ids": list(entry.get("top_doc_ids", [])),
        }
        for entry in bank_entries
    ]


def _build_tree_leaf_support_maps(
    node_registry: Sequence[object],
) -> Tuple[List[int], List[Tuple[int, ...]], Dict[Tuple[int, ...], List[int]], Dict[Tuple[int, ...], List[Tuple[int, ...]]]]:
    leaf_indices = [idx for idx, node in enumerate(node_registry) if node.is_leaf]
    leaf_paths = [tuple(node_registry[idx].path) for idx in leaf_indices]
    leaf_indices_by_prefix: Dict[Tuple[int, ...], List[int]] = {}
    leaf_ancestor_paths: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = {}

    for idx, path in zip(leaf_indices, leaf_paths):
        ancestors: List[Tuple[int, ...]] = []
        for d in range(1, len(path)):
            prefix = path[:d]
            leaf_indices_by_prefix.setdefault(prefix, []).append(idx)
            ancestors.append(prefix)
        leaf_ancestor_paths[path] = ancestors

    return leaf_indices, leaf_paths, leaf_indices_by_prefix, leaf_ancestor_paths


def _tree_version_to_dag_version(tree_version: str) -> Optional[str]:
    prefix = "category-topdown-"
    tv = str(tree_version or "").strip()
    if tv.startswith(prefix):
        return tv[len(prefix):]
    return None


def _build_dag_runtime(
    *,
    base_dir: str,
    dataset: str,
    subset: str,
    tree_version: str,
) -> Optional[Dict[str, Any]]:
    dag_version = _tree_version_to_dag_version(tree_version)
    if not dag_version:
        return None

    dag_version_u = dag_version.replace("-", "_")
    dag_path = os.path.join(
        base_dir,
        "trees",
        dataset,
        subset,
        f"category_dag_topdown_{dag_version_u}.json",
    )
    if not os.path.exists(dag_path):
        raise FileNotFoundError(
            f"DAG artifact not found for tree_version={tree_version}: {dag_path}"
        )

    with open(dag_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    nodes_raw = payload.get("nodes", [])
    edges_raw = payload.get("edges", [])
    if not isinstance(nodes_raw, list) or not isinstance(edges_raw, list):
        raise ValueError(f"Invalid DAG payload format: {dag_path}")

    row_by_id: Dict[str, Dict[str, Any]] = {}
    for row in nodes_raw:
        if not isinstance(row, dict):
            continue
        node_id = str(row.get("id", "")).strip()
        if not node_id:
            continue
        row_by_id[node_id] = row

    if not row_by_id:
        raise ValueError(f"No nodes found in DAG payload: {dag_path}")

    outgoing_ids: Dict[str, Set[str]] = defaultdict(set)
    incoming_ids: Dict[str, Set[str]] = defaultdict(set)
    for edge in edges_raw:
        if not isinstance(edge, dict):
            continue
        parent_id = str(edge.get("parent_id", "")).strip()
        child_id = str(edge.get("child_id", "")).strip()
        if (not parent_id) or (not child_id):
            continue
        if parent_id not in row_by_id or child_id not in row_by_id:
            continue
        outgoing_ids[parent_id].add(child_id)
        incoming_ids[child_id].add(parent_id)

    root_id = ""
    for node_id, row in row_by_id.items():
        if str(row.get("kind", "")).strip().lower() == "root":
            root_id = node_id
            break
    if not root_id and "L0|root" in row_by_id:
        root_id = "L0|root"
    if not root_id:
        root_id = sorted(row_by_id.keys())[0]

    reachable: Set[str] = set()
    queue: List[str] = [root_id]
    while queue:
        cur = queue.pop(0)
        if cur in reachable:
            continue
        reachable.add(cur)
        for nxt in sorted(outgoing_ids.get(cur, set())):
            if nxt not in reachable:
                queue.append(nxt)

    node_obj_by_id: Dict[str, SemanticNode] = {}
    for node_id in sorted(reachable):
        row = row_by_id[node_id]
        display_id = row.get("display_id")
        node_desc = str(row.get("desc", "") or "")
        semantic_id = display_id
        if semantic_id is None and node_id != root_id:
            semantic_id = node_id
        node_obj_by_id[node_id] = SemanticNode(id=semantic_id, desc=node_desc, child=[])

    def _child_sort_key(child_id: str) -> Tuple[int, int, str]:
        row = row_by_id[child_id]
        level = int(row.get("level", 999))
        kind = str(row.get("kind", "")).strip().lower()
        is_leaf = 1 if kind == "leaf" else 0
        return (level, is_leaf, child_id)

    for parent_id in sorted(reachable):
        parent_obj = node_obj_by_id[parent_id]
        child_ids = [x for x in outgoing_ids.get(parent_id, set()) if x in reachable]
        child_ids = sorted(child_ids, key=_child_sort_key)
        parent_obj.child = [node_obj_by_id[cid] for cid in child_ids]

    ordered_ids = sorted(
        list(reachable),
        key=lambda nid: (int(row_by_id[nid].get("level", 999)), nid),
    )
    node_registry: List[SemanticNode] = [node_obj_by_id[nid] for nid in ordered_ids]
    id_to_registry_idx: Dict[str, int] = {}
    for idx, nid in enumerate(ordered_ids):
        node = node_obj_by_id[nid]
        node.path = (idx,)
        node.registry_idx = idx
        id_to_registry_idx[nid] = idx

    children_by_idx: Dict[int, List[int]] = {}
    parents_by_idx: Dict[int, List[int]] = defaultdict(list)
    for parent_id in ordered_ids:
        pidx = id_to_registry_idx[parent_id]
        child_indices: List[int] = []
        for child_id in sorted(outgoing_ids.get(parent_id, set()), key=lambda x: id_to_registry_idx.get(x, 10**9)):
            if child_id not in id_to_registry_idx:
                continue
            cidx = id_to_registry_idx[child_id]
            child_indices.append(cidx)
            parents_by_idx[cidx].append(pidx)
        children_by_idx[pidx] = child_indices

    root_idx = id_to_registry_idx[root_id]
    leaf_indices = [idx for idx, node in enumerate(node_registry) if node.is_leaf]
    leaf_idx_set = set(leaf_indices)

    memo_desc_leafs: Dict[int, Set[int]] = {}

    def _collect_desc_leaf_indices(node_idx: int, stack: Set[int]) -> Set[int]:
        if node_idx in memo_desc_leafs:
            return memo_desc_leafs[node_idx]
        if node_idx in stack:
            return set()
        if node_idx in leaf_idx_set:
            memo_desc_leafs[node_idx] = {node_idx}
            return memo_desc_leafs[node_idx]
        out: Set[int] = set()
        next_stack = set(stack)
        next_stack.add(node_idx)
        for child_idx in children_by_idx.get(node_idx, []):
            out.update(_collect_desc_leaf_indices(child_idx, next_stack))
        memo_desc_leafs[node_idx] = out
        return out

    leaf_indices_by_prefix: Dict[Tuple[int, ...], List[int]] = {}
    for idx, node in enumerate(node_registry):
        if idx == root_idx:
            continue
        if node.is_leaf:
            continue
        descendants = sorted(_collect_desc_leaf_indices(idx, set()))
        if descendants:
            leaf_indices_by_prefix[(idx,)] = descendants

    leaf_ancestor_paths: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = {}
    for leaf_idx in leaf_indices:
        ancestors: Set[int] = set()
        stack = list(parents_by_idx.get(leaf_idx, []))
        seen: Set[int] = set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            if cur != root_idx:
                ancestors.add(cur)
            stack.extend(parents_by_idx.get(cur, []))
        leaf_ancestor_paths[(leaf_idx,)] = [tuple([x]) for x in sorted(ancestors)]

    semantic_root_node = node_obj_by_id[root_id]
    semantic_root_node.id = None
    node_by_path = {tuple(node.path): node for node in node_registry}
    path_to_registry_idx = {tuple(node.path): int(idx) for idx, node in enumerate(node_registry)}
    leaf_paths = [tuple(node_registry[idx].path) for idx in leaf_indices]

    return {
        "dag_path": dag_path,
        "semantic_root_node": semantic_root_node,
        "node_registry": node_registry,
        "node_by_path": node_by_path,
        "path_to_registry_idx": path_to_registry_idx,
        "leaf_indices": leaf_indices,
        "leaf_paths": leaf_paths,
        "leaf_indices_by_prefix": leaf_indices_by_prefix,
        "leaf_ancestor_paths": leaf_ancestor_paths,
    }


def _collect_leaf_pool(
    selected_branches: Sequence[Tuple[int, ...]],
    leaf_indices_by_prefix: Dict[Tuple[int, ...], List[int]],
    all_leaf_indices: Sequence[int],
    fallback_to_all: bool = True,
) -> List[int]:
    if not selected_branches:
        return list(all_leaf_indices) if fallback_to_all else []

    out: List[int] = []
    seen: Set[int] = set()
    for branch_path in selected_branches:
        for idx in leaf_indices_by_prefix.get(tuple(branch_path), []):
            if idx in seen:
                continue
            seen.add(idx)
            out.append(int(idx))

    if not out:
        # Intent: explore-time evidence should be able to stay empty instead of silently jumping to full-corpus evidence.
        return list(all_leaf_indices) if fallback_to_all else []
    return out


def _retrieve_leaf_hits(
    *,
    query: str,
    leaf_pool_indices: Sequence[int],
    retriever: DiverEmbeddingModel,
    node_embs: np.ndarray,
    node_registry: Sequence[object],
    topk: int,
) -> List[FlatHit]:
    if not leaf_pool_indices:
        return []
    q_emb = retriever.encode_query(str(query or ""))
    scores = (node_embs[list(leaf_pool_indices)] @ q_emb).astype(np.float32, copy=False)
    return _hits_from_scores(
        scores=scores,
        subset_indices=list(leaf_pool_indices),
        node_registry=node_registry,
        topk=topk,
    )


def _score_candidate_branches_score(
    *,
    local_hits: Sequence[FlatHit],
    candidate_child_paths: Sequence[Tuple[int, ...]],
    leaf_ancestor_paths: Dict[Tuple[int, ...], List[Tuple[int, ...]]],
    score_mode: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    mode = str(score_mode or "max").strip().lower()
    if mode not in {"max", "mean", "hit"}:
        raise ValueError(f"Unsupported branch score_mode: {score_mode}")
    for candidate in candidate_child_paths:
        cand_t = tuple(candidate)
        best_rank: Optional[int] = None
        best_leaf_score = float("-inf")
        best_leaf_path: List[int] = []
        matched_scores: List[float] = []

        for rank_idx, hit in enumerate(local_hits):
            leaf_path = tuple(hit.path)
            ancestor_paths = leaf_ancestor_paths.get(leaf_path, [])
            if (cand_t == leaf_path) or (cand_t in ancestor_paths):
                hit_score = float(hit.score)
                matched_scores.append(hit_score)
                if hit_score > best_leaf_score:
                    best_rank = rank_idx + 1
                    best_leaf_score = hit_score
                    best_leaf_path = list(leaf_path)

        if not matched_scores:
            continue
        if mode == "max":
            selector_score = float(max(matched_scores))
        elif mode == "mean":
            selector_score = float(np.mean(matched_scores))
        else:
            # Intent: max_hit_global ranks child branches by how many top-K local hits they cover.
            selector_score = float(len(matched_scores))
        rows.append(
            {
                "path": list(cand_t),
                "score": selector_score,
                "score_mode": mode,
                "max_score": float(best_leaf_score),
                "mean_score": float(np.mean(matched_scores)),
                "matched_count": int(len(matched_scores)),
                "best_rank": int(best_rank) if best_rank is not None else None,
                "best_leaf_path": best_leaf_path,
            }
        )

    rows.sort(
        key=lambda r: (
            -float(r.get("score", float("-inf")) or float("-inf")),
            int(r.get("best_rank", 10**9) or 10**9),
            -float(r.get("max_score", float("-inf")) or float("-inf")),
            len(r.get("path", [])),
            tuple(r.get("path", [])),
        )
    )
    return rows


def _row_path_tuple(row: Dict[str, Any]) -> Tuple[int, ...]:
    return tuple(int(x) for x in row.get("path", []) if x is not None)


def _filter_scored_rows_to_expandable(
    scored_rows: Sequence[Dict[str, Any]],
    expandable_state_path_map: Dict[Tuple[int, ...], List[object]],
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    seen: Set[Tuple[int, ...]] = set()
    for row in scored_rows:
        path_t = _row_path_tuple(row)
        if (not path_t) or (path_t not in expandable_state_path_map) or (path_t in seen):
            continue
        seen.add(path_t)
        filtered.append(row)
    return filtered


def _merge_local_with_global_escape(
    *,
    local_rows: Sequence[Dict[str, Any]],
    global_rows: Sequence[Dict[str, Any]],
    beam_size: int,
    escape_slots: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], str]:
    local_selected = list(local_rows[: max(1, int(beam_size))])
    if not local_selected:
        return [], [], [], "no_local_selected"

    effective_escape_slots = min(max(0, int(escape_slots)), max(0, len(local_selected) - 1))
    if effective_escape_slots <= 0:
        return local_selected, [], [], "no_escape_slots"

    local_core_count = max(1, len(local_selected) - effective_escape_slots)
    local_core = list(local_selected[:local_core_count])
    local_tail = list(local_selected[local_core_count:])
    if not local_tail:
        return local_selected, [], [], "no_local_tail"
    if not global_rows:
        return local_selected, [], [], "no_global_candidates"

    accepted_escape_rows: List[Dict[str, Any]] = []
    replaced_local_rows: List[Dict[str, Any]] = []
    chosen_paths: Set[Tuple[int, ...]] = {_row_path_tuple(row) for row in local_core}
    surviving_tail = list(local_tail)

    for escape_row in global_rows:
        if len(accepted_escape_rows) >= effective_escape_slots or (not surviving_tail):
            break
        escape_path = _row_path_tuple(escape_row)
        if (not escape_path) or (escape_path in chosen_paths):
            continue

        worst_local_row = min(
            surviving_tail,
            key=lambda row: (
                float(row.get("max_score", float("-inf"))),
                float(row.get("score", float("-inf"))),
                int(row.get("best_rank", 10**9) or 10**9),
                _row_path_tuple(row),
            ),
        )
        # Intent: compare local tail vs global escape on max leaf score so replacement uses a shared score axis.
        if float(escape_row.get("max_score", float("-inf"))) <= float(
            worst_local_row.get("max_score", float("-inf"))
        ):
            continue

        accepted_escape_rows.append(escape_row)
        replaced_local_rows.append(worst_local_row)
        chosen_paths.add(escape_path)
        surviving_tail.remove(worst_local_row)

    if not accepted_escape_rows:
        return local_selected, [], [], "no_escape_replacement"

    replaced_paths = {_row_path_tuple(row) for row in replaced_local_rows}
    final_rows = list(local_core)
    final_rows.extend(accepted_escape_rows)
    final_rows.extend([row for row in local_tail if _row_path_tuple(row) not in replaced_paths])
    final_rows = final_rows[: max(1, int(beam_size))]

    pick_reason = (
        "global_escape_replaced_full"
        if len(accepted_escape_rows) >= effective_escape_slots
        else "global_escape_replaced_partial"
    )
    return final_rows, accepted_escape_rows, replaced_local_rows, pick_reason


def _serialize_scored_rows(
    rows: Sequence[Dict[str, Any]],
    limit: int = 3,
) -> List[Dict[str, Any]]:
    return [
        {
            "path": [int(x) for x in row.get("path", [])],
            "score_mode": str(row.get("score_mode", "")),
            "score": float(row.get("score", 0.0)),
            "max_score": float(row.get("max_score", 0.0)),
            "mean_score": float(row.get("mean_score", 0.0)),
            "matched_count": int(row.get("matched_count", 0)),
            "best_rank": int(row.get("best_rank")) if row.get("best_rank") is not None else None,
            "best_leaf_path": [int(x) for x in row.get("best_leaf_path", [])],
        }
        for row in list(rows)[: max(0, int(limit))]
    ]


def _collect_candidate_child_branches(
    *,
    selected_branches: Sequence[Tuple[int, ...]],
    child_branch_paths_by_path: Dict[Tuple[int, ...], List[Tuple[int, ...]]],
    root_branch_children: Sequence[Tuple[int, ...]],
) -> List[Tuple[int, ...]]:
    if selected_branches:
        candidate_paths: List[Tuple[int, ...]] = []
        seen: Set[Tuple[int, ...]] = set()
        for parent_path in selected_branches:
            for child_path in child_branch_paths_by_path.get(tuple(parent_path), []):
                child_t = tuple(child_path)
                if child_t in seen:
                    continue
                seen.add(child_t)
                candidate_paths.append(child_t)
        return candidate_paths
    return [tuple(path) for path in root_branch_children]


def _collect_candidate_child_branches_with_parents(
    *,
    selected_branches: Sequence[Tuple[int, ...]],
    child_branch_paths_by_path: Dict[Tuple[int, ...], List[Tuple[int, ...]]],
    root_branch_children: Sequence[Tuple[int, ...]],
    root_path: Tuple[int, ...],
) -> Tuple[List[Tuple[int, ...]], Dict[Tuple[int, ...], List[Tuple[int, ...]]]]:
    candidate_paths: List[Tuple[int, ...]] = []
    parent_map: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = defaultdict(list)
    seen: Set[Tuple[int, ...]] = set()

    if selected_branches:
        for parent_path in selected_branches:
            for child_path in child_branch_paths_by_path.get(tuple(parent_path), []):
                child_t = tuple(child_path)
                parent_map[child_t].append(tuple(parent_path))
                if child_t in seen:
                    continue
                seen.add(child_t)
                candidate_paths.append(child_t)
        return candidate_paths, {
            tuple(child): sorted({tuple(parent) for parent in parents}, key=lambda x: (len(x), x))
            for child, parents in parent_map.items()
        }

    for child_path in root_branch_children:
        child_t = tuple(child_path)
        parent_map[child_t].append(tuple(root_path))
        if child_t in seen:
            continue
        seen.add(child_t)
        candidate_paths.append(child_t)
    return candidate_paths, {
        tuple(child): sorted({tuple(parent) for parent in parents}, key=lambda x: (len(x), x))
        for child, parents in parent_map.items()
    }


def _partition_selected_branches_by_child_candidates(
    *,
    selected_branches: Sequence[Tuple[int, ...]],
    child_branch_paths_by_path: Dict[Tuple[int, ...], List[Tuple[int, ...]]],
) -> Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]]]:
    active_paths: List[Tuple[int, ...]] = []
    ended_paths: List[Tuple[int, ...]] = []
    for branch_path in selected_branches:
        if child_branch_paths_by_path.get(tuple(branch_path), []):
            active_paths.append(tuple(branch_path))
        else:
            ended_paths.append(tuple(branch_path))
    return active_paths, ended_paths


def _gate_hit_dag(
    gate_paths: Sequence[Tuple[int, ...]],
    gold_paths: Sequence[Tuple[int, ...]],
    leaf_ancestor_paths: Dict[Tuple[int, ...], List[Tuple[int, ...]]],
) -> bool:
    gate_set = {tuple(p) for p in gate_paths}
    if not gate_set:
        return False
    for gp in gold_paths:
        gp_t = tuple(gp)
        if gp_t in gate_set:
            return True
        for anc in leaf_ancestor_paths.get(gp_t, []):
            if anc in gate_set:
                return True
    return False


def _branch_is_gold_ancestor(
    branch_path: Tuple[int, ...],
    gold_paths: Sequence[Tuple[int, ...]],
    leaf_ancestor_paths: Dict[Tuple[int, ...], List[Tuple[int, ...]]],
) -> bool:
    for gp in gold_paths:
        gp_t = tuple(gp)
        if gp_t == branch_path:
            return True
        for anc in leaf_ancestor_paths.get(gp_t, []):
            if tuple(anc) == tuple(branch_path):
                return True
    return False


def _branch_quality_metrics(
    selected_paths: Sequence[Tuple[int, ...]],
    gold_paths: Sequence[Tuple[int, ...]],
    leaf_ancestor_paths: Dict[Tuple[int, ...], List[Tuple[int, ...]]],
) -> Tuple[float, float]:
    if not selected_paths:
        return 0.0, 0.0
    good_flags = [
        _branch_is_gold_ancestor(tuple(path), gold_paths, leaf_ancestor_paths)
        for path in selected_paths
    ]
    # Intent: measure both "at least one good branch" and "how many selected branches are useful".
    precision = float(np.mean(good_flags)) if good_flags else 0.0
    all_hit = float(all(good_flags)) if good_flags else 0.0
    return precision, all_hit


def _selected_branch_paths_from_sample(sample: InferSample) -> List[Tuple[int, ...]]:
    selected: List[Tuple[int, ...]] = []
    for state_path in (getattr(sample, "beam_state_paths", None) or []):
        if not state_path:
            continue
        cur = state_path[-1]
        path_t = tuple(getattr(cur, "path", ()) or ())
        if path_t:
            selected.append(path_t)
    return selected


def _build_expandable_state_path_map(sample: InferSample) -> Dict[Tuple[int, ...], List[object]]:
    endpoint_to_state_path: Dict[Tuple[int, ...], List[object]] = {}
    for state_path in sample.get_all_expandable_paths(sample.prediction_tree):
        if not state_path:
            continue
        endpoint = tuple(getattr(state_path[-1], "path", ()) or ())
        if not endpoint:
            continue
        endpoint_to_state_path.setdefault(endpoint, state_path)
    return endpoint_to_state_path


def _compute_branch_metrics_from_samples(samples: Sequence[InferSample]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for sample in samples:
        selected_paths = _selected_branch_paths_from_sample(sample)
        gold_paths = [tuple(x) for x in getattr(sample, "gold_paths", []) if x]
        if not selected_paths:
            rows.append(
                {
                    "BranchHit@B": 0.0,
                    "BranchAllHit@B": 0.0,
                    "BranchPrecision@B": 0.0,
                    "NumSelectedBranches": 0.0,
                    "SelectedDepth": 0.0,
                }
            )
            continue
        # Intent: keep round5 branch metrics aligned with baseline beam-state semantics.
        good_flags = [any(tuple(branch) == tuple(gold[: len(branch)]) for gold in gold_paths) for branch in selected_paths]
        rows.append(
            {
                "BranchHit@B": 100.0 * float(any(good_flags)),
                "BranchAllHit@B": 100.0 * float(all(good_flags)),
                "BranchPrecision@B": 100.0 * float(np.mean(good_flags)),
                "NumSelectedBranches": float(len(selected_paths)),
                "SelectedDepth": float(np.mean([len(x) for x in selected_paths])),
            }
        )
    return pd.DataFrame(rows)


hp = HyperParams.from_args()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

startup_warnings: List[str] = []
requested_round5_mode = str(getattr(hp, "ROUND5_MODE", "legacy") or "legacy").strip().lower()
if requested_round5_mode not in {"legacy", "category"}:
    raise ValueError(f'Unsupported --round5_mode "{requested_round5_mode}". Allowed: legacy|category')
if requested_round5_mode != "legacy":
    # Intent: keep round6 execution legacy-only even when legacy category flags are provided.
    startup_warnings.append(f'Ignoring --round5_mode "{requested_round5_mode}" in run_round6.py (forced to legacy).')
if any(arg.startswith("--round5_category_") for arg in sys.argv[1:]):
    # Intent: make ignored category controls explicit to avoid silent config drift.
    startup_warnings.append("Ignoring --round5_category_* arguments in run_round6.py (legacy-only runtime).")
round5_mode = "legacy"

if hp.REWRITE_PROMPT_PATH:
    print("Ignoring --rewrite_prompt_path in run_round6.py (template path override is disabled).")

round5_rewrite_prompt_name = "agent_executor_v1"
requested_prompt_name = str(hp.REWRITE_PROMPT_NAME or "").strip()
if requested_prompt_name:
    round5_rewrite_prompt_name = requested_prompt_name
# Intent: legacy mode accepts any registered rewrite template so prompt ablations can reuse --rewrite_prompt_name.
if round5_rewrite_prompt_name not in REWRITE_PROMPT_TEMPLATES:
    raise ValueError(
        f'Unknown --rewrite_prompt_name "{round5_rewrite_prompt_name}" in run_round6.py legacy mode. '
        f"Available: {sorted(REWRITE_PROMPT_TEMPLATES.keys())}"
    )

hp.add_param("rewrite_prompt_name", round5_rewrite_prompt_name)

if str(hp.ROUND3_SUMMARIZED_CONTEXT or "off").lower() != "off":
    print("Ignoring --round3_summarized_context in run_round6.py (fixed to off).")
hp.add_param("round3_summarized_context", "off")

round5_mrr_pool_k = max(1, int(getattr(hp, "ROUND5_MRR_POOL_K", 100) or 100))
hp.add_param("round5_mrr_pool_k", round5_mrr_pool_k)
round5_selector_mode = str(getattr(hp, "ROUND5_SELECTOR_MODE", "retriever_slate") or "retriever_slate").strip().lower()
if round5_selector_mode not in {"retriever_slate", "maxscore_global", "meanscore_global", "max_hit_global"}:
    raise ValueError(
        f'Unsupported --round5_selector_mode "{round5_selector_mode}". '
        "Allowed: retriever_slate|maxscore_global|meanscore_global|max_hit_global"
    )
hp.add_param("round5_selector_mode", round5_selector_mode)
round6_global_escape = bool(getattr(hp, "ROUND6_GLOBAL_ESCAPE", False))
round6_global_escape_slots = max(1, int(getattr(hp, "ROUND6_GLOBAL_ESCAPE_SLOTS", 2) or 2))
round6_expandable_mode = str(getattr(hp, "ROUND6_EXPANDABLE_MODE", "off") or "off").strip().lower()
if round6_expandable_mode not in {"off", "ended_reseat"}:
    raise ValueError(
        f'Unsupported --round6_expandable_mode "{round6_expandable_mode}". '
        "Allowed: off|ended_reseat"
    )
round6_method2 = bool(getattr(hp, "ROUND6_METHOD2", False))
round6_method2_mode = str(
    getattr(hp, "ROUND6_METHOD2_MODE", "archive_replace") or "archive_replace"
).strip().lower()
if round6_method2_mode not in {"archive_replace", "expandable_pool"}:
    raise ValueError(
        f'Unsupported --round6_method2_mode "{round6_method2_mode}". '
        "Allowed: archive_replace|expandable_pool"
    )
round6_expandable_pool_freeze_terminal_beam = bool(
    getattr(hp, "ROUND6_EXPANDABLE_POOL_FREEZE_TERMINAL_BEAM", False)
)
round6_fusion_mode = str(getattr(hp, "ROUND6_FUSION_MODE", "rrf") or "rrf").strip().lower()
if round6_fusion_mode not in {"rrf", "max_score", "sum_score"}:
    raise ValueError(
        f'Unsupported --round6_fusion_mode "{round6_fusion_mode}". Allowed: rrf|max_score|sum_score'
    )
round6_explore_prompt_name = str(
    getattr(hp, "ROUND6_EXPLORE_PROMPT_NAME", "agent_executor_v1_icl2_explore") or "agent_executor_v1_icl2_explore"
).strip()
if round6_explore_prompt_name not in REWRITE_PROMPT_TEMPLATES:
    raise ValueError(
        f'Unknown --round6_explore_prompt_name "{round6_explore_prompt_name}". '
        f"Available: {sorted(REWRITE_PROMPT_TEMPLATES.keys())}"
    )
if round6_global_escape and round5_selector_mode == "retriever_slate":
    raise ValueError(
        "--round6_global_escape requires a score-based --round5_selector_mode "
        "(maxscore_global|meanscore_global|max_hit_global)."
    )
if round6_method2 and round5_selector_mode == "retriever_slate":
    raise ValueError(
        "--round6_method2 requires a score-based --round5_selector_mode "
        "(maxscore_global|meanscore_global|max_hit_global)."
    )
if round6_expandable_mode != "off" and round5_selector_mode == "retriever_slate":
    raise ValueError(
        "--round6_expandable_mode requires a score-based --round5_selector_mode "
        "(maxscore_global|meanscore_global|max_hit_global)."
    )
if round6_method2 and round6_global_escape:
    raise ValueError("--round6_method2 and --round6_global_escape are mutually exclusive in round6.")
if round6_expandable_mode != "off" and round6_global_escape:
    raise ValueError("--round6_expandable_mode and --round6_global_escape are mutually exclusive in round6.")
if round6_expandable_mode != "off" and round6_method2:
    raise ValueError("--round6_expandable_mode and --round6_method2 are mutually exclusive in round6.")
hp.add_param("round6_global_escape", round6_global_escape)
hp.add_param("round6_global_escape_slots", round6_global_escape_slots)
hp.add_param("round6_expandable_mode", round6_expandable_mode)
hp.add_param("round6_method2", round6_method2)
hp.add_param("round6_method2_mode", round6_method2_mode)
hp.add_param("round6_expandable_pool_freeze_terminal_beam", round6_expandable_pool_freeze_terminal_beam)
hp.add_param("round6_fusion_mode", round6_fusion_mode)
hp.add_param("round6_explore_prompt_name", round6_explore_prompt_name)

exp_dir_name = str(hp)
RESULTS_DIR = f"{BASE_DIR}/results/{hp.DATASET}/{hp.SUBSET}/round6/{exp_dir_name}/"
if os.path.exists(RESULTS_DIR) and os.listdir(RESULTS_DIR):
    print(f"Results already exist at {RESULTS_DIR}. Skipping run.")
    raise SystemExit(0)
os.makedirs(RESULTS_DIR, exist_ok=True)

logger = setup_logger("lattice_runner_round6", f"{RESULTS_DIR}/run.log", logging.INFO)
with open(f"{RESULTS_DIR}/hparams.json", "w", encoding="utf-8") as f:
    json.dump(vars(hp), f, indent=2, ensure_ascii=True, sort_keys=True)
for warning in startup_warnings:
    logger.warning(warning)
logger.info(
    "Round6 config | mode=%s | rewrite_prompt=%s | explore_prompt=%s | selector_mode=%s | selector_topk=%d | global_escape=%s | escape_slots=%d | expandable_mode=%s | method2=%s | method2_mode=%s | freeze_terminal_beam=%s | fusion_mode=%s",
    "legacy",
    round5_rewrite_prompt_name,
    round6_explore_prompt_name,
    round5_selector_mode,
    int(round5_mrr_pool_k),
    str(round6_global_escape).lower(),
    int(round6_global_escape_slots),
    round6_expandable_mode,
    str(round6_method2).lower(),
    round6_method2_mode,
    str(round6_expandable_pool_freeze_terminal_beam).lower(),
    round6_fusion_mode,
)

if not hp.REWRITE_CACHE_PATH:
    cache_root = os.path.join(BASE_DIR, "cache", "rewrite")
    os.makedirs(cache_root, exist_ok=True)
    cache_mode_tag = "round5_legacy"
    cache_name = f"{hp.SUBSET}_{round5_rewrite_prompt_name}_{cache_mode_tag}_{hp.exp_hash(8)}.jsonl"
    hp.add_param("rewrite_cache_path", os.path.join(cache_root, cache_name))

query_source = str(hp.QUERY_SOURCE or "original").lower()
if query_source not in {"original", "gpt4"}:
    raise ValueError(f"Unknown --query_source '{hp.QUERY_SOURCE}'. Expected: original|gpt4")

if os.path.exists(f"{BASE_DIR}/data/{hp.DATASET}/{hp.SUBSET}/documents.jsonl"):
    logger.info("Loading dataset %s split=%s from local JSONL files", hp.DATASET, hp.SUBSET)
    docs_df = pd.read_json(
        f"{BASE_DIR}/data/{hp.DATASET}/{hp.SUBSET}/documents.jsonl",
        lines=True,
        dtype={"id": str},
    )
    local_examples_name = "gpt4_reason" if query_source == "gpt4" else "examples"
    local_examples_path = f"{BASE_DIR}/data/{hp.DATASET}/{hp.SUBSET}/{local_examples_name}.jsonl"
    if not os.path.exists(local_examples_path):
        raise FileNotFoundError(
            f"Requested --query_source={query_source}, but local file not found: {local_examples_path}"
        )
    examples_df = pd.read_json(local_examples_path, lines=True, dtype={"gold_ids": List[str]})
    examples_df["gold_ids"] = examples_df["gold_ids"].apply(lambda x: [str(i) for i in x])
else:
    logger.info("Loading dataset xlangai/BRIGHT split=%s from HuggingFace Datasets", hp.SUBSET)
    docs_df = pd.DataFrame(load_dataset("xlangai/BRIGHT", "documents", split=hp.SUBSET))
    examples_config = "gpt4_reason" if query_source == "gpt4" else "examples"
    examples_df = pd.DataFrame(load_dataset("xlangai/BRIGHT", examples_config, split=hp.SUBSET))

dag_runtime = _build_dag_runtime(
    base_dir=BASE_DIR,
    dataset=hp.DATASET,
    subset=hp.SUBSET,
    tree_version=hp.TREE_VERSION,
)
using_dag_runtime = dag_runtime is not None
if using_dag_runtime:
    logger.info(
        "Round5 DAG mode enabled | tree_version=%s | dag_path=%s",
        hp.TREE_VERSION,
        dag_runtime["dag_path"],
    )
    semantic_root_node = dag_runtime["semantic_root_node"]
    node_registry = dag_runtime["node_registry"]
    leaf_indices = dag_runtime["leaf_indices"]
    leaf_paths = dag_runtime["leaf_paths"]
    leaf_indices_by_prefix = dag_runtime["leaf_indices_by_prefix"]
    leaf_ancestor_paths = dag_runtime["leaf_ancestor_paths"]
else:
    tree_dict = pkl.load(open(f"{BASE_DIR}/trees/{hp.DATASET}/{hp.SUBSET}/tree-{hp.TREE_VERSION}.pkl", "rb"))
    semantic_root_node = SemanticNode().load_dict(tree_dict) if isinstance(tree_dict, dict) else tree_dict
    node_registry = compute_node_registry(semantic_root_node)
    leaf_indices, leaf_paths, leaf_indices_by_prefix, leaf_ancestor_paths = _build_tree_leaf_support_maps(node_registry)

root_path = tuple(getattr(semantic_root_node, "path", ()) or ())

path_to_node: Dict[Tuple[int, ...], object] = {tuple(node.path): node for node in node_registry}
path_to_registry_idx: Dict[Tuple[int, ...], int] = {tuple(node.path): int(idx) for idx, node in enumerate(node_registry)}

child_branch_paths_by_path: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = {}
for node in node_registry:
    path_t = tuple(node.path)
    child_paths = [tuple(child.path) for child in node.child if not child.is_leaf]
    if child_paths:
        dedup = sorted({tuple(p) for p in child_paths})
        child_branch_paths_by_path[path_t] = dedup
root_branch_children = child_branch_paths_by_path.get(root_path, [])
if not root_branch_children:
    logger.warning("Root has no non-leaf children. Branch selector may fallback frequently.")

doc_id_to_paths: Dict[str, List[Tuple[int, ...]]] = defaultdict(list)
for leaf_idx in leaf_indices:
    node = node_registry[int(leaf_idx)]
    doc_id = get_node_id(node.id, docs_df)
    if not doc_id:
        continue
    doc_id_to_paths[str(doc_id)].append(tuple(node.path))
for doc_id in list(doc_id_to_paths.keys()):
    deduped = sorted({tuple(p) for p in doc_id_to_paths[doc_id]})
    doc_id_to_paths[doc_id] = deduped

path_to_doc_id: Dict[Tuple[int, ...], str] = {}
for doc_id, paths in doc_id_to_paths.items():
    for path in paths:
        path_to_doc_id[tuple(path)] = str(doc_id)
registry_idx_to_doc_id: Dict[int, str] = {}
for path_t, doc_id in path_to_doc_id.items():
    if path_t in path_to_registry_idx:
        registry_idx_to_doc_id[int(path_to_registry_idx[path_t])] = str(doc_id)

if not hp.RETRIEVER_MODEL_PATH:
    raise ValueError("--retriever_model_path is required")
retriever = build_retriever(hp.RETRIEVER_MODEL_PATH, subset=hp.SUBSET, local_files_only=True)

node_embs: Optional[np.ndarray] = None
if hp.NODE_EMB_PATH:
    if not os.path.exists(hp.NODE_EMB_PATH):
        logger.warning("node_emb_path not found: %s; fallback to on-the-fly encoding", hp.NODE_EMB_PATH)
    else:
        loaded = np.load(hp.NODE_EMB_PATH, allow_pickle=False)
        try:
            aligned = pad_node_embeddings_to_registry(loaded, node_registry)
            node_embs = normalize_embeddings(aligned)
            if loaded.shape[0] != len(node_registry):
                logger.warning(
                    "Resolved node_emb row mismatch via legacy blank-desc padding (got=%d expected=%d)",
                    loaded.shape[0],
                    len(node_registry),
                )
        except ValueError:
            logger.warning(
                "node_emb rows mismatch (got=%d expected=%d); fallback to on-the-fly encoding",
                loaded.shape[0],
                len(node_registry),
            )
if node_embs is None:
    logger.info("Encoding %d node descriptions on-the-fly for round5 runtime", len(node_registry))
    node_descs = [str(node.desc or "").strip() or "No Description." for node in node_registry]
    node_embs = retriever.encode(node_descs, max_length=4096, batch_size=4)
    node_embs = normalize_embeddings(node_embs)

rewrite_template = REWRITE_PROMPT_TEMPLATES[round5_rewrite_prompt_name]
explore_rewrite_template = REWRITE_PROMPT_TEMPLATES[round6_explore_prompt_name]
subset_domain_route_hint = _build_domain_route_hint(hp.SUBSET)
# Intent: keep rewrite rubric text aligned with the canonical subset relevance definition.
subset_relevance_definition = _get_relevance_definition(hp.SUBSET)
rewrite_map, docs_map = _load_rewrite_cache(hp.REWRITE_CACHE_PATH, hp.REWRITE_FORCE_REFRESH)

if hp.LLM_API_BACKEND == "genai":
    llm_api = GenAIAPI(hp.LLM, logger=logger, timeout=hp.LLM_API_TIMEOUT, max_retries=hp.LLM_API_MAX_RETRIES)
elif hp.LLM_API_BACKEND == "vllm":
    vllm_base_url, vllm_base_url_src = _resolve_vllm_base_url(BASE_DIR)
    logger.info("Round6 vLLM endpoints source: %s", vllm_base_url_src)
    logger.info("Round6 vLLM endpoints: %s", vllm_base_url)
    llm_api = VllmAPI(
        hp.LLM,
        logger=logger,
        timeout=hp.LLM_API_TIMEOUT,
        max_retries=hp.LLM_API_MAX_RETRIES,
        base_url=vllm_base_url,
    )
else:
    raise ValueError(f"Unknown LM API backend: {hp.LLM_API_BACKEND}")

llm_api_kwargs = {
    "max_concurrent_calls": hp.LLM_MAX_CONCURRENT_CALLS,
    "staggering_delay": hp.LLM_API_STAGGERING_DELAY,
    # Intent: deterministic rewrite generation for stable selector comparison.
    "temperature": 0.0,
}

num_samples = min(examples_df.shape[0], hp.NUM_EVAL_SAMPLES)
all_eval_samples: List[InferSample] = []
for i in range(num_samples):
    raw_gold_ids = [str(x) for x in examples_df.iloc[i]["gold_ids"]]
    gold_doc_ids = [doc_id for doc_id in raw_gold_ids if doc_id in doc_id_to_paths]
    gold_paths: List[Tuple[int, ...]] = []
    for doc_id in gold_doc_ids:
        gold_paths.extend(doc_id_to_paths.get(doc_id, []))
    gold_paths = sorted({tuple(p) for p in gold_paths})

    if len(gold_doc_ids) < len(raw_gold_ids):
        logger.warning("Some gold IDs for example %d not found in document paths.", i)

    original_query = examples_df.iloc[i]["query"][: hp.MAX_QUERY_CHAR_LEN]
    sample = InferSample(
        semantic_root_node,
        node_registry,
        hp=hp,
        logger=logger,
        query=original_query,
        gold_paths=[list(x) for x in gold_paths],
        excluded_ids_set=set(examples_df.iloc[i].get("excluded_ids", [])),
    )
    # Intent: keep per-iter diagnostics persisted while branch traversal uses baseline InferSample mechanics.
    sample.original_query = original_query
    sample.gold_doc_ids = list(gold_doc_ids)
    sample.last_rewrite_raw = ""
    sample.rewrite_history = []
    sample.iter_records = []
    sample.round6_pending_explore = False
    # Intent: Go-Explore state is the archived parent decision and which child branches have already been taken there.
    sample.round6_archive_records = {}
    sample.round6_last_trigger_selected_prefixes = []
    # Intent: separate official all-iter memory from diagnostic leaf-trigger-only memory.
    sample.round6_fusion_bank_all_iters = []
    sample.round6_fusion_bank_leaf_trigger_only = []
    # Intent: terminal-freeze ablation keeps the last beam-local retrieval scope fixed after leaf-only frontier completion.
    sample.round6_terminal_freeze_active = False
    sample.round6_terminal_freeze_iter = -1
    sample.round6_terminal_freeze_reason = ""
    sample.round6_terminal_frozen_gate_pool_indices = []
    sample.round6_terminal_frozen_selector_pool_indices = []
    if "original_query" not in sample.SAVE_LIST:
        sample.SAVE_LIST.append("original_query")
    if "gold_doc_ids" not in sample.SAVE_LIST:
        sample.SAVE_LIST.append("gold_doc_ids")
    if "last_rewrite_raw" not in sample.SAVE_LIST:
        sample.SAVE_LIST.append("last_rewrite_raw")
    if "iter_records" not in sample.SAVE_LIST:
        sample.SAVE_LIST.append("iter_records")
    all_eval_samples.append(sample)

all_eval_metric_dfs: List[pd.DataFrame] = []
cumulative_leaf_indices_by_sample: List[Set[int]] = [set() for _ in all_eval_samples]
for iter_idx in range(hp.NUM_ITERS):
    logger.info("Round6 iteration %d", iter_idx)

    # Intent: keep cumulative leaf pools from all previously reached leaves per sample.
    for sample_idx, sample in enumerate(all_eval_samples):
        reached = sample.get_top_predictions(k=None, rel_fn=sample.get_rel_fn(leaf=True))
        for node, _ in reached:
            ridx = int(getattr(node, "registry_idx", -1))
            if ridx >= 0:
                cumulative_leaf_indices_by_sample[sample_idx].add(ridx)

    rewrite_prompts: List[str] = []
    rewrite_meta: List[Dict[str, Any]] = []
    rewrite_state_by_sample_idx: Dict[int, Dict[str, Any]] = {}
    retrieval_rows: List[Dict[str, float]] = []
    fusion_diag_ndcg_by_mode_all_iters: Dict[str, List[float]] = defaultdict(list)
    fusion_diag_ndcg_leaf_trigger_only: List[float] = []
    bank_run_ndcg_values_all_iters: List[float] = []

    for sample_idx, sample in enumerate(tqdm(all_eval_samples, desc=f"Iter {iter_idx} rewrite prep", leave=False)):
        explore_step = bool(round6_method2 and getattr(sample, "round6_pending_explore", False))
        terminal_freeze_active = bool(
            round6_method2
            and round6_method2_mode == "expandable_pool"
            and round6_expandable_pool_freeze_terminal_beam
            and getattr(sample, "round6_terminal_freeze_active", False)
        )
        seen_query = str(sample.query or sample.original_query).strip()
        cumulative_pool = sorted(cumulative_leaf_indices_by_sample[sample_idx])
        if not cumulative_pool:
            cumulative_pool = list(leaf_indices)

        query_pre = seen_query
        prompt_name_for_iter = round5_rewrite_prompt_name
        prompt_template_for_iter = rewrite_template
        explore_effective = False
        explore_mask_fallback = False
        explore_fallback_reason = ""
        explore_candidate_paths: List[Tuple[int, ...]] = []
        explore_candidate_parent_map: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = {}
        explore_evidence_pool = list(cumulative_pool)
        explore_pool_source = "cumulative_reached_leaves"
        explore_candidate_scope = "direct_children"
        last_trigger_selected_prefixes = [
            tuple(path)
            for path in list(getattr(sample, "round6_last_trigger_selected_prefixes", []) or [])
            if path
        ]
        last_trigger_blocked_leaf_indices = (
            _collect_blocked_leaf_indices(
                last_trigger_selected_prefixes,
                leaf_indices_by_prefix,
            )
            if (round6_method2_mode == "archive_replace" and last_trigger_selected_prefixes)
            else set()
        )
        archive_records = dict(getattr(sample, "round6_archive_records", {}) or {})
        archive_records_summary = _serialize_archive_records(archive_records, child_branch_paths_by_path)

        if terminal_freeze_active:
            # Intent: once terminal freeze is active, keep rewrite evidence on the last beam-local pool instead of reopening any new branch scope.
            frozen_gate_pool = list(getattr(sample, "round6_terminal_frozen_gate_pool_indices", []) or [])
            explore_evidence_pool = [int(x) for x in frozen_gate_pool] if frozen_gate_pool else list(cumulative_pool)
            query_pre = seen_query
            prompt_name_for_iter = round5_rewrite_prompt_name
            prompt_template_for_iter = rewrite_template
            explore_pool_source = "terminal_frozen_gate_pool"
            explore_candidate_scope = "terminal_frozen_beam"
            explore_step = False
        elif explore_step:
            if round6_method2_mode == "archive_replace":
                raw_candidate_paths, raw_candidate_parent_map = _collect_unexplored_sibling_candidates(
                    archive_records,
                    child_branch_paths_by_path,
                )
                current_expandable_state_path_map = _build_expandable_state_path_map(sample)
                explore_candidate_paths = [
                    tuple(path_t)
                    for path_t in raw_candidate_paths
                    if tuple(path_t) in current_expandable_state_path_map
                ]
                explore_candidate_parent_map = {
                    tuple(path_t): [
                        tuple(parent_path)
                        for parent_path in raw_candidate_parent_map.get(tuple(path_t), [])
                    ]
                    for path_t in explore_candidate_paths
                }
                explore_evidence_pool = _collect_leaf_pool(
                    selected_branches=explore_candidate_paths,
                    leaf_indices_by_prefix=leaf_indices_by_prefix,
                    all_leaf_indices=leaf_indices,
                    fallback_to_all=False,
                )
                if last_trigger_blocked_leaf_indices:
                    # Intent: archive_replace keeps the first explore step from immediately reusing the last exploited subtree.
                    explore_evidence_pool = _filter_leaf_pool_by_blocked(
                        explore_evidence_pool,
                        last_trigger_blocked_leaf_indices,
                    )
                if explore_candidate_paths and explore_evidence_pool:
                    query_pre = str(sample.original_query).strip()
                    prompt_name_for_iter = round6_explore_prompt_name
                    prompt_template_for_iter = explore_rewrite_template
                    explore_effective = True
                    explore_pool_source = "archive_sibling_descendants"
                    explore_candidate_scope = "archived_sibling_candidates"
            else:
                current_expandable_state_path_map = _build_expandable_state_path_map(sample)
                # Intent: expandable_pool explore widens both rewrite evidence and selector scope to the whole live expandable frontier.
                explore_candidate_paths = [tuple(path_t) for path_t in current_expandable_state_path_map.keys()]
                explore_candidate_parent_map = {}
                explore_evidence_pool = _collect_leaf_pool(
                    selected_branches=explore_candidate_paths,
                    leaf_indices_by_prefix=leaf_indices_by_prefix,
                    all_leaf_indices=leaf_indices,
                    fallback_to_all=False,
                )
                if explore_candidate_paths and explore_evidence_pool:
                    query_pre = seen_query
                    prompt_name_for_iter = round5_rewrite_prompt_name
                    prompt_template_for_iter = rewrite_template
                    explore_effective = True
                    explore_pool_source = "all_expandable_descendants"
                    explore_candidate_scope = "all_expandable_paths"
            if not explore_effective:
                explore_fallback_reason = (
                    "no_recoverable_candidates"
                    if not explore_candidate_paths
                    else "empty_explore_evidence_pool"
                )
                explore_mask_fallback = True
                explore_candidate_paths = []
                explore_candidate_parent_map = {}
                explore_evidence_pool = list(cumulative_pool)

        pre_hits = _retrieve_leaf_hits(
            query=query_pre,
            leaf_pool_indices=explore_evidence_pool,
            retriever=retriever,
            node_embs=node_embs,
            node_registry=node_registry,
            topk=round5_mrr_pool_k,
        )
        pre_hit_paths = [tuple(hit.path) for hit in pre_hits]
        pre_hit_doc_ids = _paths_to_ranked_doc_ids(pre_hit_paths, path_to_doc_id)
        leaf_descs = _hits_to_context_descs(
            pre_hits,
            node_registry,
            topk=hp.REWRITE_CONTEXT_TOPK,
            max_desc_len=hp.MAX_DOC_DESC_CHAR_LEN,
        )
        rewrite_state_by_sample_idx[sample_idx] = {
            "cache_key": "",
            "cache_hit": False,
            "cached_rewrite": "",
            "cached_docs": {},
            "prompt_name": prompt_name_for_iter,
            "explore_step": bool(explore_step),
            "explore_effective": bool(explore_effective),
            "terminal_freeze_active": bool(terminal_freeze_active),
            "terminal_freeze_triggered": False,
            "explore_fallback_reason": str(explore_fallback_reason),
            "explore_pool_source": str(explore_pool_source),
            "explore_candidate_scope": str(explore_candidate_scope),
            "explore_archive_records": archive_records_summary,
            "explore_candidate_paths": [[int(x) for x in path] for path in explore_candidate_paths],
            "explore_candidate_paths_raw": [tuple(path) for path in explore_candidate_paths],
            "explore_candidate_parent_map": {
                "|".join(str(int(x)) for x in child_path): [[int(x) for x in parent_path] for parent_path in parent_paths]
                for child_path, parent_paths in explore_candidate_parent_map.items()
            },
            "explore_candidate_parent_map_raw": {
                tuple(child_path): [tuple(parent_path) for parent_path in parent_paths]
                for child_path, parent_paths in explore_candidate_parent_map.items()
            },
            "explore_last_trigger_selected_prefixes": [[int(x) for x in path] for path in last_trigger_selected_prefixes],
            "explore_masked_leaf_count": int(len(last_trigger_blocked_leaf_indices)),
            "explore_mask_fallback": bool(explore_mask_fallback),
            "query_pre_reset_to_original": bool(
                explore_effective and round6_method2_mode == "archive_replace"
            ),
            "terminal_freeze_reason": str(getattr(sample, "round6_terminal_freeze_reason", "") or ""),
            "terminal_freeze_iter": int(getattr(sample, "round6_terminal_freeze_iter", -1) or -1),
            "terminal_frozen_gate_pool_size": int(
                len(list(getattr(sample, "round6_terminal_frozen_gate_pool_indices", []) or []))
            ),
            "terminal_frozen_selector_pool_size": int(
                len(list(getattr(sample, "round6_terminal_frozen_selector_pool_indices", []) or []))
            ),
            "query_pre": query_pre,
            "query_post": query_pre,
            "leaf_descs": leaf_descs,
            "rewrite": "",
            "rewrite_docs": {},
            "raw_output": "",
            "cumulative_pool_pre_size": int(len(cumulative_pool)),
            "masked_pool_pre_size": int(len(explore_evidence_pool)),
            "explore_evidence_pool_size": int(len(explore_evidence_pool)),
            "explore_evidence_pool_indices": [int(x) for x in explore_evidence_pool],
            # Intent: persist rewrite-time retrieval evidence (query_pre) for off-branch/noise attribution analysis.
            "pre_hit_paths": [list(p) for p in pre_hit_paths],
            "pre_hit_doc_ids": list(pre_hit_doc_ids),
            "eval_paths": [],
            "eval_doc_ids": [],
            "active_eval_paths": [],
            "active_eval_doc_ids": [],
            "active_eval_metrics": {},
            "current_run_metrics": {},
            "current_run_bank_entry": {},
            "fusion_bank_all_iters_runs": [],
            "fusion_bank_leaf_trigger_only_runs": [],
            "fusion_metrics_all_iters_by_mode": {},
            "fusion_metrics_leaf_trigger_only_by_mode": {},
        }

        prompt = _format_rewrite_prompt(
            prompt_template_for_iter,
            original_query=str(sample.original_query),
            previous_rewrite=str(getattr(sample, "last_rewrite_raw", "") or ""),
            leaf_descs=leaf_descs,
            domain_route_hint=subset_domain_route_hint,
            relevance_definition=subset_relevance_definition,
        )
        cache_key = _prompt_cache_key("round5", prompt)
        cache_hit = (not hp.REWRITE_FORCE_REFRESH) and (cache_key in rewrite_map)
        rewrite_state_by_sample_idx[sample_idx]["cache_key"] = cache_key
        rewrite_state_by_sample_idx[sample_idx]["cache_hit"] = bool(cache_hit)
        rewrite_state_by_sample_idx[sample_idx]["cached_rewrite"] = str(rewrite_map.get(cache_key, "") or "")
        rewrite_state_by_sample_idx[sample_idx]["cached_docs"] = (
            _clean_docs_map(docs_map.get(cache_key, {})) if cache_hit else {}
        )
        if not cache_hit:
            rewrite_prompts.append(prompt)
            rewrite_meta.append(
                {
                    "sample_idx": sample_idx,
                    "cache_key": cache_key,
                    "prompt": prompt,
                    "leaf_descs": leaf_descs,
                    "prompt_name": prompt_name_for_iter,
                    "query_pre": query_pre,
                    "mode": "explore" if explore_effective else "legacy",
                }
            )

    generated_by_key: Dict[str, Dict[str, Any]] = {}
    new_cache_records: List[Dict[str, Any]] = []

    if rewrite_prompts:
        logger.info("Iter %d: starting rewrite batch (%d prompts)", iter_idx, len(rewrite_prompts))
        rewrite_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(rewrite_loop)
        try:
            rewrite_outputs = rewrite_loop.run_until_complete(llm_api.run_batch(rewrite_prompts, **llm_api_kwargs))
        finally:
            rewrite_loop.close()
            asyncio.set_event_loop(None)

        for meta, out in zip(rewrite_meta, rewrite_outputs):
            sample_idx = int(meta.get("sample_idx", -1))
            docs_out, rewrite_out = _parse_possible_answer_docs(out)
            docs_out = _clean_docs_map(docs_out)

            if not rewrite_out:
                rewrite_out = _flatten_docs(docs_out)

            cache_key = str(meta["cache_key"])
            generated_by_key[cache_key] = {
                "rewrite": rewrite_out,
                "docs": docs_out,
                "raw_output": str(out or ""),
            }

            rewrite_map[cache_key] = rewrite_out
            if docs_out:
                docs_map[cache_key] = docs_out

            new_cache_records.append(
                {
                    "key": cache_key,
                    "rewritten_query": rewrite_out,
                    "possible_answer_docs": docs_out if docs_out else None,
                    "prompt_name": str(meta.get("prompt_name", round5_rewrite_prompt_name)),
                    "llm": hp.LLM,
                    "leaf_descs": meta.get("leaf_descs", []),
                    "query_pre": meta.get("query_pre", ""),
                    "mode": meta.get("mode", "legacy"),
                }
            )

    if hp.REWRITE_CACHE_PATH and new_cache_records:
        append_jsonl(hp.REWRITE_CACHE_PATH, new_cache_records)

    for sample_idx, sample in enumerate(all_eval_samples):
        info = rewrite_state_by_sample_idx[sample_idx]
        cache_key = str(info.get("cache_key", ""))
        if info.get("cache_hit", False):
            rewrite_blob = str(info.get("cached_rewrite", "") or "")
            rewrite_docs = _clean_docs_map(info.get("cached_docs", {}))
            raw_output = ""
        else:
            generated = generated_by_key.get(cache_key, {})
            rewrite_blob = str(generated.get("rewrite", "") or "")
            rewrite_docs = _clean_docs_map(generated.get("docs", {}))
            raw_output = str(generated.get("raw_output", "") or "")

        query_pre = str(info.get("query_pre", "") or sample.original_query)
        query_post = _compose_next_query(str(sample.original_query), rewrite_blob, query_pre)
        sample.last_rewrite_raw = rewrite_blob
        sample.query = query_post

        info["rewrite"] = rewrite_blob
        info["rewrite_docs"] = rewrite_docs
        info["raw_output"] = raw_output
        info["query_post"] = query_post

        cumulative_pool_eval = sorted(cumulative_leaf_indices_by_sample[sample_idx])
        if not cumulative_pool_eval:
            cumulative_pool_eval = list(leaf_indices)
        eval_hits = _retrieve_leaf_hits(
            query=query_post,
            leaf_pool_indices=cumulative_pool_eval,
            retriever=retriever,
            node_embs=node_embs,
            node_registry=node_registry,
            topk=max(1, int(hp.FLAT_TOPK)),
        )
        eval_paths = [tuple(hit.path) for hit in eval_hits]
        eval_doc_ids = _paths_to_ranked_doc_ids(eval_paths, path_to_doc_id)
        gold_doc_ids = [str(x) for x in sample.gold_doc_ids]
        bank_entry = _bank_entry_from_hits(
            hits=eval_hits,
            gold_doc_ids=gold_doc_ids,
            path_to_doc_id=path_to_doc_id,
            iter_idx=iter_idx,
            explore_step=bool(info.get("explore_effective", False)),
            query_post=query_post,
        )
        current_run_metrics = _compute_retrieval_metrics(eval_doc_ids, gold_doc_ids)
        info["cumulative_pool_eval_size"] = int(len(cumulative_pool_eval))
        info["eval_paths"] = [list(p) for p in eval_paths]
        info["eval_doc_ids"] = list(eval_doc_ids)
        info["gold_doc_ids"] = list(gold_doc_ids)
        info["current_run_bank_entry"] = bank_entry
        info["current_run_metrics"] = {str(k): float(v) for k, v in current_run_metrics.items()}
        sample.rewrite_history.append(
            {
                "iter": iter_idx,
                "cache_hit": bool(info.get("cache_hit", False)),
                "prompt_name": str(info.get("prompt_name", round5_rewrite_prompt_name)),
                "query_pre": query_pre,
                "query_post": query_post,
                "rewrite": rewrite_blob,
                "explore_step": bool(info.get("explore_step", False)),
                "explore_effective": bool(info.get("explore_effective", False)),
                "possible_answer_docs": rewrite_docs,
                "leaf_descs": info.get("leaf_descs", []),
                "raw_output": raw_output,
            }
        )

    inputs = []
    for sample in all_eval_samples:
        if bool(getattr(sample, "round6_terminal_freeze_active", False)):
            # Intent: a frozen sample should stop traversal expansion entirely; only rewrite/eval retrieval continue.
            inputs.append([])
        else:
            inputs.append(sample.get_step_prompts())
    indptr = np.cumsum([0, *[len(x) for x in inputs]])
    flat_inputs = [y for x in inputs for y in x]
    flat_slates = [x[1] for x in flat_inputs]
    slates = [flat_slates[indptr[j] : indptr[j + 1]] for j in range(len(inputs))]

    response_jsons: List[List[Dict[str, Any]]] = []
    for sample, sample_slates in tqdm(
        zip(all_eval_samples, slates),
        total=len(all_eval_samples),
        desc=f"Iter {iter_idx} retriever scoring",
        leave=False,
    ):
        q_emb = retriever.encode_query(str(sample.query or ""))
        per_sample: List[Dict[str, Any]] = []
        for slate in sample_slates:
            slate_indices = list(slate)
            if not slate_indices:
                per_sample.append({"reasoning": "retriever_slate", "ranking": [], "relevance_scores": []})
                continue
            scores = (node_embs[slate_indices] @ q_emb).astype(np.float32, copy=False)
            scores_01 = np.clip((scores + 1.0) / 2.0, 0.0, 1.0)
            order = np.argsort(-scores_01)
            per_sample.append(
                {
                    "reasoning": "retriever_slate",
                    "ranking": [int(i) for i in order.tolist()],
                    "relevance_scores": [[int(i), float(scores_01[i] * 100.0)] for i in order.tolist()],
                }
            )
        response_jsons.append(per_sample)

    selected_before_by_idx: Dict[int, List[Tuple[int, ...]]] = {}
    selector_pick_reason_by_idx: Dict[int, str] = {}
    selector_source_by_idx: Dict[int, str] = {}
    global_escape_pick_reason_by_idx: Dict[int, str] = {}
    selector_candidate_paths_by_idx: Dict[int, List[Tuple[int, ...]]] = {}
    selector_scored_rows_by_idx: Dict[int, List[Dict[str, Any]]] = {}
    ended_reseat_scored_rows_by_idx: Dict[int, List[Dict[str, Any]]] = {}
    ended_reseat_selected_rows_by_idx: Dict[int, List[Dict[str, Any]]] = {}
    selector_global_escape_rows_by_idx: Dict[int, List[Dict[str, Any]]] = {}
    selector_global_escape_selected_by_idx: Dict[int, List[Dict[str, Any]]] = {}
    selector_global_escape_replaced_by_idx: Dict[int, List[Dict[str, Any]]] = {}
    selector_chosen_child_parent_map_by_idx: Dict[int, Dict[Tuple[int, ...], List[Tuple[int, ...]]]] = {}
    for sample_idx, sample in enumerate(all_eval_samples):
        info = rewrite_state_by_sample_idx[sample_idx]
        selected_before = _selected_branch_paths_from_sample(sample)
        selected_before_by_idx[sample_idx] = selected_before
        active_selected_before, ended_selected_before = _partition_selected_branches_by_child_candidates(
            selected_branches=selected_before,
            child_branch_paths_by_path=child_branch_paths_by_path,
        )
        explore_effective = bool(info.get("explore_effective", False))
        ended_reseat_enabled = bool(round6_expandable_mode == "ended_reseat")
        ended_reseat_count = int(len(ended_selected_before)) if ended_reseat_enabled else 0
        freeze_ablation_enabled = bool(
            round6_method2
            and round6_method2_mode == "expandable_pool"
            and round6_expandable_pool_freeze_terminal_beam
        )
        terminal_freeze_active = bool(
            freeze_ablation_enabled and getattr(sample, "round6_terminal_freeze_active", False)
        )
        use_archive_replace = bool(explore_effective and round6_method2_mode == "archive_replace")
        direct_child_candidate_paths, _ = _collect_candidate_child_branches_with_parents(
            selected_branches=selected_before,
            child_branch_paths_by_path=child_branch_paths_by_path,
            root_branch_children=root_branch_children,
            root_path=root_path,
        )
        terminal_freeze_triggered = False
        terminal_freeze_reason = ""
        terminal_frozen_gate_pool: List[int] = []
        terminal_frozen_selector_pool: List[int] = []
        if terminal_freeze_active:
            terminal_frozen_gate_pool = [
                int(x) for x in list(getattr(sample, "round6_terminal_frozen_gate_pool_indices", []) or [])
            ]
            terminal_frozen_selector_pool = [
                int(x) for x in list(getattr(sample, "round6_terminal_frozen_selector_pool_indices", []) or [])
            ]
        elif use_archive_replace:
            pass
        elif freeze_ablation_enabled and not direct_child_candidate_paths:
            terminal_frozen_gate_pool = _collect_leaf_pool(
                selected_branches=selected_before,
                leaf_indices_by_prefix=leaf_indices_by_prefix,
                all_leaf_indices=leaf_indices,
            )
            terminal_frozen_selector_pool = list(terminal_frozen_gate_pool)
            # Intent: instantiate the current leaf slates once but prevent the default jump to unrelated leftover expandable paths.
            sample.update_keep_beam(slates[sample_idx], response_jsons[sample_idx])
            terminal_freeze_triggered = True
            terminal_freeze_reason = "no_direct_child_branch_candidates"
        else:
            sample.update(slates[sample_idx], response_jsons[sample_idx], rel_fn=sample.get_rel_fn())
        selector_pick_reason = "retriever_slate_default"
        selector_source = "retriever_slate"
        global_escape_pick_reason = "disabled"
        selector_candidate_paths: List[Tuple[int, ...]] = []
        selector_scored_rows: List[Dict[str, Any]] = []
        ended_reseat_scored_rows: List[Dict[str, Any]] = []
        ended_reseat_selected_rows: List[Dict[str, Any]] = []
        selector_global_escape_rows: List[Dict[str, Any]] = []
        selector_global_escape_selected_rows: List[Dict[str, Any]] = []
        selector_global_escape_replaced_rows: List[Dict[str, Any]] = []
        selector_chosen_child_parent_map: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = {}
        if terminal_freeze_active or terminal_freeze_triggered:
            selector_pick_reason = (
                "terminal_freeze_active"
                if terminal_freeze_active
                else "terminal_freeze_triggered_no_child_branch"
            )
            selector_source = "terminal_frozen_noop"
        if round5_selector_mode in {"maxscore_global", "meanscore_global", "max_hit_global"}:
            if round5_selector_mode == "maxscore_global":
                score_mode = "max"
            elif round5_selector_mode == "meanscore_global":
                score_mode = "mean"
            else:
                score_mode = "hit"
            candidate_parent_map: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = {}
            if terminal_freeze_active or terminal_freeze_triggered:
                selector_candidate_paths = []
                candidate_parent_map = {}
            elif use_archive_replace:
                selector_candidate_paths = [
                    tuple(path_t)
                    for path_t in list(info.get("explore_candidate_paths_raw", []) or [])
                ]
                candidate_parent_map = {
                    tuple(child_path): [tuple(parent_path) for parent_path in parent_paths]
                    for child_path, parent_paths in dict(info.get("explore_candidate_parent_map_raw", {}) or {}).items()
                }
            elif explore_effective:
                # Intent: expandable_pool explore scores the full live expandable frontier instead of only direct children of the previous beam.
                selector_candidate_paths = list(_build_expandable_state_path_map(sample).keys())
                candidate_parent_map = {}
            else:
                selector_candidate_paths, candidate_parent_map = _collect_candidate_child_branches_with_parents(
                    selected_branches=(active_selected_before if ended_reseat_enabled else selected_before),
                    child_branch_paths_by_path=child_branch_paths_by_path,
                    root_branch_children=root_branch_children,
                    root_path=root_path,
                )
            local_row_limit = max(0, int(hp.MAX_BEAM_SIZE) - ended_reseat_count) if ended_reseat_enabled else max(1, int(hp.MAX_BEAM_SIZE))
            if not selector_candidate_paths and not (ended_reseat_enabled and ended_reseat_count > 0):
                if not (terminal_freeze_active or terminal_freeze_triggered):
                    # Intent: if there is no expandable child-branch candidate, keep retriever-slate beam unchanged.
                    selector_pick_reason = "no_candidate_children" if not explore_effective else "explore_no_candidate_children"
            else:
                if terminal_freeze_active or terminal_freeze_triggered:
                    selector_local_pool = list(terminal_frozen_selector_pool)
                elif explore_effective:
                    # Intent: the first explore controller only scores recoverable sibling directions, not the just-expanded live frontier.
                    selector_local_pool = [int(x) for x in list(info.get("explore_evidence_pool_indices", []) or [])]
                else:
                    selector_local_pool = _collect_leaf_pool(
                        selected_branches=selected_before,
                        leaf_indices_by_prefix=leaf_indices_by_prefix,
                        all_leaf_indices=leaf_indices,
                    )
                selector_query = str(
                    info.get("query_post", "")
                    or sample.query
                    or getattr(sample, "original_query", "")
                )
                if round6_method2 and round6_method2_mode == "archive_replace":
                    # Intent: method2 controller only uses completed all-iter memory; current-run retrieval is fallback, not banked controller input.
                    bank_entries_all_iters = list(getattr(sample, "round6_fusion_bank_all_iters", []) or [])
                    selector_local_hits = []
                    if bank_entries_all_iters:
                        selector_source = f"fusion_bank:{round6_fusion_mode}:all_iters"
                        selector_local_hits = _fuse_bank_entries(
                            bank_entries=bank_entries_all_iters,
                            mode=round6_fusion_mode,
                            node_registry=node_registry,
                            topk=round5_mrr_pool_k,
                            rrf_k=max(1, int(getattr(hp, "ROUND3_RRF_K", 60) or 60)),
                            blocked_leaf_indices=None,
                            allowed_leaf_indices=set(selector_local_pool),
                        )
                    if not selector_local_hits:
                        selector_source = (
                            "current_explore_retrieval_fallback"
                            if explore_effective
                            else "current_local_retrieval_fallback"
                        )
                        selector_local_hits = _retrieve_leaf_hits(
                            query=selector_query,
                            leaf_pool_indices=selector_local_pool,
                            retriever=retriever,
                            node_embs=node_embs,
                            node_registry=node_registry,
                            topk=round5_mrr_pool_k,
                        )
                else:
                    selector_source = "current_explore_retrieval" if explore_effective else "current_local_retrieval"
                    selector_local_hits = _retrieve_leaf_hits(
                        query=selector_query,
                        leaf_pool_indices=selector_local_pool,
                        retriever=retriever,
                        node_embs=node_embs,
                        node_registry=node_registry,
                        topk=round5_mrr_pool_k,
                    )
                expandable_state_path_map = _build_expandable_state_path_map(sample)
                if selector_local_hits and selector_candidate_paths:
                    selector_scored_rows = _score_candidate_branches_score(
                        local_hits=selector_local_hits,
                        candidate_child_paths=selector_candidate_paths,
                        leaf_ancestor_paths=leaf_ancestor_paths,
                        score_mode=score_mode,
                    )
                    selector_scored_rows = _filter_scored_rows_to_expandable(
                        selector_scored_rows,
                        expandable_state_path_map,
                    )
                elif not ended_reseat_enabled:
                    selector_pick_reason = "no_local_hits"
                final_selector_rows = list(selector_scored_rows[:local_row_limit]) if local_row_limit > 0 else []
                if selector_scored_rows or (ended_reseat_enabled and ended_reseat_count > 0):
                    if selector_scored_rows and not final_selector_rows and not ended_reseat_enabled:
                        selector_pick_reason = "no_candidate_match"
                    else:
                        if round6_global_escape:
                            global_candidate_paths = [
                                tuple(path_t)
                                for path_t in expandable_state_path_map.keys()
                                if tuple(path_t) not in {_row_path_tuple(row) for row in final_selector_rows}
                            ]
                            if not global_candidate_paths:
                                global_escape_pick_reason = "no_global_candidates"
                            else:
                                # Intent: method1 keeps rewrite local but adds a selector-stage full-corpus escape probe.
                                selector_global_hits = _retrieve_leaf_hits(
                                    query=selector_query,
                                    leaf_pool_indices=leaf_indices,
                                    retriever=retriever,
                                    node_embs=node_embs,
                                    node_registry=node_registry,
                                    topk=round5_mrr_pool_k,
                                )
                                if not selector_global_hits:
                                    global_escape_pick_reason = "no_global_hits"
                                else:
                                    selector_global_escape_rows = _score_candidate_branches_score(
                                        local_hits=selector_global_hits,
                                        candidate_child_paths=global_candidate_paths,
                                        leaf_ancestor_paths=leaf_ancestor_paths,
                                        score_mode="max",
                                    )
                                    selector_global_escape_rows = _filter_scored_rows_to_expandable(
                                        selector_global_escape_rows,
                                        expandable_state_path_map,
                                    )
                                    (
                                        final_selector_rows,
                                        selector_global_escape_selected_rows,
                                        selector_global_escape_replaced_rows,
                                        global_escape_pick_reason,
                                    ) = _merge_local_with_global_escape(
                                        local_rows=final_selector_rows,
                                        global_rows=selector_global_escape_rows,
                                        beam_size=max(1, int(hp.MAX_BEAM_SIZE)),
                                        escape_slots=round6_global_escape_slots,
                                    )

                        selected_state_paths: List[List[object]] = []
                        selected_endpoints: Set[Tuple[int, ...]] = set()
                        for row in final_selector_rows:
                            path_t = _row_path_tuple(row)
                            matched_state_path = expandable_state_path_map.get(path_t)
                            if not matched_state_path:
                                continue
                            endpoint = tuple(getattr(matched_state_path[-1], "path", ()) or ())
                            if not endpoint or endpoint in selected_endpoints:
                                continue
                            selected_endpoints.add(endpoint)
                            selected_state_paths.append(matched_state_path)
                        if ended_reseat_enabled and ended_reseat_count > 0:
                            leftover_candidate_paths = [
                                tuple(path_t)
                                for path_t in expandable_state_path_map.keys()
                                if tuple(path_t) not in selected_endpoints
                            ]
                            if leftover_candidate_paths:
                                replacement_leaf_pool = _collect_leaf_pool(
                                    selected_branches=leftover_candidate_paths,
                                    leaf_indices_by_prefix=leaf_indices_by_prefix,
                                    all_leaf_indices=leaf_indices,
                                    fallback_to_all=False,
                                )
                                if replacement_leaf_pool:
                                    # Intent: score only the leftover expandable frontier when reseating ended beam slots.
                                    replacement_hits = _retrieve_leaf_hits(
                                        query=selector_query,
                                        leaf_pool_indices=replacement_leaf_pool,
                                        retriever=retriever,
                                        node_embs=node_embs,
                                        node_registry=node_registry,
                                        topk=round5_mrr_pool_k,
                                    )
                                    if replacement_hits:
                                        ended_reseat_scored_rows = _score_candidate_branches_score(
                                            local_hits=replacement_hits,
                                            candidate_child_paths=leftover_candidate_paths,
                                            leaf_ancestor_paths=leaf_ancestor_paths,
                                            score_mode=score_mode,
                                        )
                                        ended_reseat_scored_rows = _filter_scored_rows_to_expandable(
                                            ended_reseat_scored_rows,
                                            expandable_state_path_map,
                                        )
                                        ended_reseat_selected_rows = list(
                                            ended_reseat_scored_rows[: max(0, ended_reseat_count)]
                                        )
                        if not selected_state_paths and not ended_reseat_selected_rows:
                            selector_pick_reason = "no_expandable_match"
                        else:
                            selector_chosen_child_parent_map = {
                                tuple(_row_path_tuple(row)): [
                                    tuple(parent_path)
                                    for parent_path in candidate_parent_map.get(tuple(_row_path_tuple(row)), [])
                                ]
                                for row in final_selector_rows
                                if tuple(_row_path_tuple(row)) in candidate_parent_map
                            }
                            if use_archive_replace:
                                # Intent: Go-Explore reopens archived sibling states by replacing the live beam instead of mixing exploit and explore beams.
                                sample.beam_state_paths = selected_state_paths[: max(1, int(hp.MAX_BEAM_SIZE))]
                                selector_pick_reason = f"explore_{round5_selector_mode}_replace"
                            else:
                                fallback_state_paths = list(getattr(sample, "beam_state_paths", None) or [])
                                merged_state_paths = list(selected_state_paths)
                                merged_endpoints = {
                                    tuple(getattr(path[-1], "path", ()) or ())
                                    for path in merged_state_paths
                                    if path
                                }
                                if ended_reseat_enabled and ended_reseat_selected_rows:
                                    # Intent: reseat only the locally ended slots, leaving non-ended local continuation intact.
                                    for row in ended_reseat_selected_rows:
                                        path_t = _row_path_tuple(row)
                                        matched_state_path = expandable_state_path_map.get(path_t)
                                        if not matched_state_path:
                                            continue
                                        replacement_endpoint = tuple(getattr(matched_state_path[-1], "path", ()) or ())
                                        if not replacement_endpoint or replacement_endpoint in merged_endpoints:
                                            continue
                                        merged_endpoints.add(replacement_endpoint)
                                        merged_state_paths.append(matched_state_path)
                                        if len(merged_state_paths) >= max(1, int(hp.MAX_BEAM_SIZE)):
                                            break
                                # Intent: preserve baseline traversal continuity by filling missing beams with retriever-slate picks.
                                for fallback_path in fallback_state_paths:
                                    if not fallback_path:
                                        continue
                                    fallback_endpoint = tuple(getattr(fallback_path[-1], "path", ()) or ())
                                    if not fallback_endpoint or fallback_endpoint in merged_endpoints:
                                        continue
                                    merged_endpoints.add(fallback_endpoint)
                                    merged_state_paths.append(fallback_path)
                                    if len(merged_state_paths) >= max(1, int(hp.MAX_BEAM_SIZE)):
                                        break
                                sample.beam_state_paths = merged_state_paths[: max(1, int(hp.MAX_BEAM_SIZE))]
                                if explore_effective:
                                    local_pick_reason = (
                                        f"explore_{round5_selector_mode}_all_expandable_full"
                                        if len(selected_state_paths) >= max(1, int(hp.MAX_BEAM_SIZE))
                                        else f"explore_{round5_selector_mode}_all_expandable_partial_fill"
                                    )
                                else:
                                    local_pick_reason = (
                                        f"{round5_selector_mode}_applied_full"
                                        if len(selected_state_paths) >= max(1, int(hp.MAX_BEAM_SIZE))
                                        else f"{round5_selector_mode}_applied_partial_fill"
                                    )
                                selector_pick_reason = (
                                    f"{local_pick_reason}|{global_escape_pick_reason}"
                                    if round6_global_escape
                                    else local_pick_reason
                                )
                                if ended_reseat_enabled and ended_reseat_count > 0:
                                    ended_reseat_status = (
                                        "full"
                                        if len(ended_reseat_selected_rows) >= ended_reseat_count
                                        else ("partial" if ended_reseat_selected_rows else "none")
                                    )
                                    selector_pick_reason = f"{round5_selector_mode}_ended_reseat_{ended_reseat_status}"
                elif not selector_scored_rows and not (ended_reseat_enabled and ended_reseat_count > 0):
                    selector_pick_reason = "no_candidate_match"
        selector_pick_reason_by_idx[sample_idx] = selector_pick_reason
        selector_source_by_idx[sample_idx] = selector_source
        global_escape_pick_reason_by_idx[sample_idx] = global_escape_pick_reason
        selector_candidate_paths_by_idx[sample_idx] = selector_candidate_paths
        selector_scored_rows_by_idx[sample_idx] = _serialize_scored_rows(selector_scored_rows, limit=3)
        ended_reseat_scored_rows_by_idx[sample_idx] = _serialize_scored_rows(ended_reseat_scored_rows, limit=3)
        ended_reseat_selected_rows_by_idx[sample_idx] = _serialize_scored_rows(
            ended_reseat_selected_rows,
            limit=max(0, ended_reseat_count),
        )
        selector_global_escape_rows_by_idx[sample_idx] = _serialize_scored_rows(selector_global_escape_rows, limit=3)
        selector_global_escape_selected_by_idx[sample_idx] = _serialize_scored_rows(
            selector_global_escape_selected_rows,
            limit=round6_global_escape_slots,
        )
        selector_global_escape_replaced_by_idx[sample_idx] = _serialize_scored_rows(
            selector_global_escape_replaced_rows,
            limit=round6_global_escape_slots,
        )
        selector_chosen_child_parent_map_by_idx[sample_idx] = selector_chosen_child_parent_map
        info["ended_beam_count"] = int(ended_reseat_count)
        info["ended_beam_paths"] = [[int(x) for x in path] for path in ended_selected_before]
        info["ended_beam_reseat_enabled"] = bool(ended_reseat_enabled)
        info["ended_beam_reseat_candidate_count"] = int(len(ended_reseat_scored_rows))
        reached_after = sample.get_top_predictions(k=None, rel_fn=sample.get_rel_fn(leaf=True))
        existing_leaf_indices = set(cumulative_leaf_indices_by_sample[sample_idx])
        new_leaf_paths: List[Tuple[int, ...]] = []
        new_trigger_selected_prefixes: List[Tuple[int, ...]] = []
        if round6_method2 and selector_chosen_child_parent_map:
            sample.round6_archive_records = _update_archive_records(
                dict(getattr(sample, "round6_archive_records", {}) or {}),
                selector_chosen_child_parent_map,
                iter_idx,
            )
        for node, _ in reached_after:
            ridx = int(getattr(node, "registry_idx", -1))
            if ridx >= 0:
                if ridx not in existing_leaf_indices:
                    leaf_path = tuple(getattr(node, "path", ()) or ())
                    if leaf_path:
                        new_leaf_paths.append(leaf_path)
                cumulative_leaf_indices_by_sample[sample_idx].add(ridx)
        if new_leaf_paths:
            new_trigger_selected_prefixes = _selected_paths_covering_new_leafs(new_leaf_paths, selected_before)
        sample.round6_pending_explore = bool(round6_method2 and new_leaf_paths)
        if freeze_ablation_enabled and terminal_freeze_triggered:
            if not terminal_frozen_gate_pool:
                terminal_frozen_gate_pool = sorted(cumulative_leaf_indices_by_sample[sample_idx]) or list(leaf_indices)
            if not terminal_frozen_selector_pool:
                terminal_frozen_selector_pool = list(terminal_frozen_gate_pool)
            sample.round6_terminal_freeze_active = True
            sample.round6_terminal_freeze_iter = int(iter_idx)
            sample.round6_terminal_freeze_reason = str(terminal_freeze_reason or "no_direct_child_branch_candidates")
            sample.round6_terminal_frozen_gate_pool_indices = [int(x) for x in terminal_frozen_gate_pool]
            sample.round6_terminal_frozen_selector_pool_indices = [int(x) for x in terminal_frozen_selector_pool]
        if freeze_ablation_enabled and (terminal_freeze_triggered or terminal_freeze_active):
            sample.round6_pending_explore = False
            sample.round6_last_trigger_selected_prefixes = []
        if round6_method2 and new_trigger_selected_prefixes and not (freeze_ablation_enabled and (terminal_freeze_triggered or terminal_freeze_active)):
            sample.round6_last_trigger_selected_prefixes = list(new_trigger_selected_prefixes)
        elif not round6_method2:
            sample.round6_pending_explore = False
            sample.round6_last_trigger_selected_prefixes = []

        info = rewrite_state_by_sample_idx[sample_idx]
        info["terminal_freeze_active"] = bool(getattr(sample, "round6_terminal_freeze_active", False))
        info["terminal_freeze_triggered"] = bool(terminal_freeze_triggered)
        info["terminal_freeze_reason"] = str(getattr(sample, "round6_terminal_freeze_reason", "") or terminal_freeze_reason)
        info["terminal_freeze_iter"] = int(getattr(sample, "round6_terminal_freeze_iter", -1) or -1)
        info["terminal_frozen_gate_pool_size"] = int(
            len(list(getattr(sample, "round6_terminal_frozen_gate_pool_indices", []) or []))
        )
        info["terminal_frozen_selector_pool_size"] = int(
            len(list(getattr(sample, "round6_terminal_frozen_selector_pool_indices", []) or []))
        )
        current_run_bank_entry = dict(info.get("current_run_bank_entry", {}) or {})
        current_run_metrics = {
            str(k): float(v)
            for k, v in dict(info.get("current_run_metrics", {}) or {}).items()
        }
        active_eval_hits: List[FlatHit] = []
        active_eval_doc_ids = list(info.get("eval_doc_ids", []))
        active_metrics = dict(current_run_metrics)
        leaf_trigger_active_hits: List[FlatHit] = []
        leaf_trigger_active_doc_ids: List[str] = []
        leaf_trigger_active_metrics = _nan_retrieval_metrics()
        fusion_metrics_all_iters_by_mode: Dict[str, Dict[str, float]] = {}
        fusion_metrics_leaf_trigger_only_by_mode: Dict[str, Dict[str, float]] = {}
        bank_append_reason_all_iters = "disabled"
        bank_append_reason_leaf_trigger_only = "disabled"

        if round6_method2 and current_run_bank_entry:
            # Intent: append completed runs only after leaf-trigger status is known, so controller never consumes same-iter memory.
            sample.round6_fusion_bank_all_iters.append(current_run_bank_entry)
            bank_run_ndcg_values_all_iters.append(float(current_run_bank_entry.get("ndcg10", 0.0)))
            bank_append_reason_all_iters = "completed_iter"
            if new_leaf_paths:
                sample.round6_fusion_bank_leaf_trigger_only.append(current_run_bank_entry)
                bank_append_reason_leaf_trigger_only = "new_leaf_reached"
            else:
                bank_append_reason_leaf_trigger_only = "not_leaf_trigger"

            bank_entries_all_iters = list(getattr(sample, "round6_fusion_bank_all_iters", []) or [])
            bank_entries_leaf_trigger_only = list(getattr(sample, "round6_fusion_bank_leaf_trigger_only", []) or [])
            for fusion_mode_name in ("rrf", "max_score", "sum_score"):
                fused_hits_all_iters = _fuse_bank_entries(
                    bank_entries=bank_entries_all_iters,
                    mode=fusion_mode_name,
                    node_registry=node_registry,
                    topk=max(1, int(hp.FLAT_TOPK)),
                    rrf_k=max(1, int(getattr(hp, "ROUND3_RRF_K", 60) or 60)),
                    blocked_leaf_indices=None,
                )
                fused_doc_ids_all_iters = _paths_to_ranked_doc_ids(
                    [tuple(hit.path) for hit in fused_hits_all_iters],
                    path_to_doc_id,
                )
                metrics_all_iters = _compute_retrieval_metrics(fused_doc_ids_all_iters, info.get("gold_doc_ids", []))
                fusion_metrics_all_iters_by_mode[fusion_mode_name] = {
                    str(k): float(v) for k, v in metrics_all_iters.items()
                }
                fusion_diag_ndcg_by_mode_all_iters[fusion_mode_name].append(float(metrics_all_iters["nDCG@10"]))
                if round6_method2_mode == "archive_replace" and fusion_mode_name == round6_fusion_mode:
                    active_eval_hits = list(fused_hits_all_iters)
                    active_eval_doc_ids = list(fused_doc_ids_all_iters)
                    active_metrics = dict(metrics_all_iters)

                if bank_entries_leaf_trigger_only:
                    fused_hits_leaf_trigger_only = _fuse_bank_entries(
                        bank_entries=bank_entries_leaf_trigger_only,
                        mode=fusion_mode_name,
                        node_registry=node_registry,
                        topk=max(1, int(hp.FLAT_TOPK)),
                        rrf_k=max(1, int(getattr(hp, "ROUND3_RRF_K", 60) or 60)),
                        blocked_leaf_indices=None,
                    )
                    fused_doc_ids_leaf_trigger_only = _paths_to_ranked_doc_ids(
                        [tuple(hit.path) for hit in fused_hits_leaf_trigger_only],
                        path_to_doc_id,
                    )
                    metrics_leaf_trigger_only = _compute_retrieval_metrics(
                        fused_doc_ids_leaf_trigger_only,
                        info.get("gold_doc_ids", []),
                    )
                else:
                    fused_hits_leaf_trigger_only = []
                    fused_doc_ids_leaf_trigger_only = []
                    metrics_leaf_trigger_only = _nan_retrieval_metrics()
                fusion_metrics_leaf_trigger_only_by_mode[fusion_mode_name] = {
                    str(k): float(v) for k, v in metrics_leaf_trigger_only.items()
                }
                if fusion_mode_name == round6_fusion_mode:
                    leaf_trigger_active_hits = list(fused_hits_leaf_trigger_only)
                    leaf_trigger_active_doc_ids = list(fused_doc_ids_leaf_trigger_only)
                    leaf_trigger_active_metrics = dict(metrics_leaf_trigger_only)
                    fusion_diag_ndcg_leaf_trigger_only.append(float(metrics_leaf_trigger_only["nDCG@10"]))
            if round6_method2_mode != "archive_replace":
                # Intent: expandable_pool uses fusion outputs as diagnostics only; official metrics stay on the current retrieval run.
                active_eval_hits = []
                active_eval_doc_ids = list(info.get("eval_doc_ids", []))
                active_metrics = dict(current_run_metrics)
        else:
            active_eval_doc_ids = list(info.get("eval_doc_ids", []))
            active_metrics = dict(current_run_metrics)

        active_eval_paths = [list(tuple(hit.path)) for hit in active_eval_hits]
        if not active_eval_paths:
            active_eval_paths = list(info.get("eval_paths", []))
        info["active_eval_paths"] = active_eval_paths
        info["active_eval_doc_ids"] = list(active_eval_doc_ids)
        info["active_eval_metrics"] = {str(k): float(v) for k, v in active_metrics.items()}
        info["nDCG_leaf_trigger_only"] = float(leaf_trigger_active_metrics.get("nDCG@10", float("nan")))
        info["active_eval_leaf_trigger_only_paths"] = [list(tuple(hit.path)) for hit in leaf_trigger_active_hits]
        info["active_eval_leaf_trigger_only_doc_ids"] = list(leaf_trigger_active_doc_ids)
        info["active_eval_leaf_trigger_only_metrics"] = {
            str(k): float(v) for k, v in leaf_trigger_active_metrics.items()
        }
        info["fusion_bank_all_iters_runs"] = _summarize_bank_entries(
            list(getattr(sample, "round6_fusion_bank_all_iters", []) or [])
        )
        info["fusion_bank_leaf_trigger_only_runs"] = _summarize_bank_entries(
            list(getattr(sample, "round6_fusion_bank_leaf_trigger_only", []) or [])
        )
        info["fusion_bank_runs"] = list(info.get("fusion_bank_all_iters_runs", []))
        info["fusion_metrics_all_iters_by_mode"] = {
            mode_name: {str(k): float(v) for k, v in metrics_by_mode.items()}
            for mode_name, metrics_by_mode in fusion_metrics_all_iters_by_mode.items()
        }
        info["fusion_metrics_leaf_trigger_only_by_mode"] = {
            mode_name: {str(k): float(v) for k, v in metrics_by_mode.items()}
            for mode_name, metrics_by_mode in fusion_metrics_leaf_trigger_only_by_mode.items()
        }
        info["fusion_metrics_by_mode"] = dict(info.get("fusion_metrics_all_iters_by_mode", {}))
        info["bank_append_reason_all_iters"] = bank_append_reason_all_iters
        info["bank_append_reason_leaf_trigger_only"] = bank_append_reason_leaf_trigger_only
        info["fusion_leaf_trigger_bank_empty"] = bool(
            round6_method2 and not list(getattr(sample, "round6_fusion_bank_leaf_trigger_only", []) or [])
        )

        retrieval_row = {str(k): float(v) for k, v in active_metrics.items()}
        retrieval_row["nDCG_leaf_trigger_only"] = float(leaf_trigger_active_metrics.get("nDCG@10", float("nan")))
        retrieval_rows.append(retrieval_row)
        rewrite_state_by_sample_idx[sample_idx]["new_leaf_paths"] = [list(path) for path in new_leaf_paths]
        rewrite_state_by_sample_idx[sample_idx]["new_trigger_selected_prefixes"] = [
            [int(x) for x in path] for path in new_trigger_selected_prefixes
        ]
        rewrite_state_by_sample_idx[sample_idx]["archive_records_post"] = _serialize_archive_records(
            dict(getattr(sample, "round6_archive_records", {}) or {}),
            child_branch_paths_by_path,
        )

    iter_df = pd.DataFrame(retrieval_rows)
    branch_metric_df = _compute_branch_metrics_from_samples(all_eval_samples)
    if (not branch_metric_df.empty) and (len(branch_metric_df) == len(iter_df)):
        iter_df = pd.concat(
            [iter_df.reset_index(drop=True), branch_metric_df.reset_index(drop=True)],
            axis=1,
        )
    else:
        logger.warning(
            "Branch metric merge skipped at iter %d (branch_rows=%d eval_rows=%d)",
            iter_idx,
            int(len(branch_metric_df)),
            int(len(iter_df)),
        )

    for sample_idx, sample in enumerate(all_eval_samples):
        info = rewrite_state_by_sample_idx[sample_idx]
        selected_after = _selected_branch_paths_from_sample(sample)
        row_metrics = iter_df.iloc[sample_idx].to_dict() if sample_idx < len(iter_df) else {}
        metrics = {str(k): float(v) for k, v in row_metrics.items() if np.isscalar(v)}
        sample.iter_records.append(
            {
                "iter": iter_idx,
                "round5_mode": round5_mode,
                "query_pre": str(info.get("query_pre", "")),
                "query_post": str(info.get("query_post", "")),
                "rewrite": str(info.get("rewrite", "")),
                "rewrite_context_topk": int(hp.REWRITE_CONTEXT_TOPK),
                "rewrite_leaf_descs_count": int(len(info.get("leaf_descs", []))),
                "possible_answer_docs": info.get("rewrite_docs", {}),
                "selected_branches_before": [list(p) for p in selected_before_by_idx.get(sample_idx, [])],
                "selected_branches_after": [list(p) for p in selected_after],
                "selector_mode": round5_selector_mode,
                "selector_source": str(selector_source_by_idx.get(sample_idx, "retriever_slate")),
                "selector_pick_reason": str(selector_pick_reason_by_idx.get(sample_idx, "retriever_slate_default")),
                "global_escape_enabled": bool(round6_global_escape),
                "global_escape_slots": int(round6_global_escape_slots),
                "round6_expandable_mode": round6_expandable_mode if round6_expandable_mode != "off" else "",
                "round6_method2_mode": round6_method2_mode if round6_method2 else "",
                "global_escape_pick_reason": str(global_escape_pick_reason_by_idx.get(sample_idx, "disabled")),
                "ended_beam_count": int(info.get("ended_beam_count", 0)),
                "ended_beam_paths": info.get("ended_beam_paths", []),
                "ended_beam_reseat_enabled": bool(info.get("ended_beam_reseat_enabled", False)),
                "ended_beam_reseat_candidate_count": int(info.get("ended_beam_reseat_candidate_count", 0)),
                "explore_step": bool(info.get("explore_step", False)),
                "explore_effective": bool(info.get("explore_effective", False)),
                "explore_trigger_reason": "new_leaf_reached" if bool(info.get("explore_step", False)) else "",
                "explore_fallback_reason": str(info.get("explore_fallback_reason", "")),
                "explore_pool_source": str(info.get("explore_pool_source", "")),
                "explore_candidate_scope": str(info.get("explore_candidate_scope", "")),
                "terminal_freeze_active": bool(info.get("terminal_freeze_active", False)),
                "terminal_freeze_triggered": bool(info.get("terminal_freeze_triggered", False)),
                "terminal_freeze_reason": str(info.get("terminal_freeze_reason", "")),
                "terminal_freeze_iter": int(info.get("terminal_freeze_iter", -1)),
                "terminal_frozen_gate_pool_size": int(info.get("terminal_frozen_gate_pool_size", 0)),
                "terminal_frozen_selector_pool_size": int(info.get("terminal_frozen_selector_pool_size", 0)),
                "explore_archive_records": info.get("explore_archive_records", []),
                "explore_archive_records_post": info.get("archive_records_post", []),
                "explore_candidate_paths": info.get("explore_candidate_paths", []),
                "explore_last_trigger_selected_prefixes": info.get("explore_last_trigger_selected_prefixes", []),
                "explore_masked_leaf_count": int(info.get("explore_masked_leaf_count", 0)),
                "explore_mask_fallback": bool(info.get("explore_mask_fallback", False)),
                "new_leaf_paths": info.get("new_leaf_paths", []),
                "new_trigger_selected_prefixes": info.get("new_trigger_selected_prefixes", []),
                "query_pre_reset_to_original": bool(info.get("query_pre_reset_to_original", False)),
                "fusion_mode_active": (
                    round6_fusion_mode
                    if (round6_method2 and round6_method2_mode == "archive_replace")
                    else ""
                ),
                "fusion_bank_size": int(len(info.get("fusion_bank_all_iters_runs", []))),
                "fusion_bank_source_iters": [int(x.get("iter", -1)) for x in info.get("fusion_bank_all_iters_runs", [])],
                "fusion_bank_runs": info.get("fusion_bank_all_iters_runs", []),
                "fusion_bank_all_iters_size": int(len(info.get("fusion_bank_all_iters_runs", []))),
                "fusion_bank_all_iters_source_iters": [
                    int(x.get("iter", -1)) for x in info.get("fusion_bank_all_iters_runs", [])
                ],
                "fusion_bank_all_iters_runs": info.get("fusion_bank_all_iters_runs", []),
                "fusion_bank_leaf_trigger_only_size": int(len(info.get("fusion_bank_leaf_trigger_only_runs", []))),
                "fusion_bank_leaf_trigger_only_source_iters": [
                    int(x.get("iter", -1)) for x in info.get("fusion_bank_leaf_trigger_only_runs", [])
                ],
                "fusion_bank_leaf_trigger_only_runs": info.get("fusion_bank_leaf_trigger_only_runs", []),
                "fusion_metrics_by_mode": info.get("fusion_metrics_all_iters_by_mode", {}),
                "fusion_metrics_all_iters_by_mode": info.get("fusion_metrics_all_iters_by_mode", {}),
                "fusion_metrics_leaf_trigger_only_by_mode": info.get("fusion_metrics_leaf_trigger_only_by_mode", {}),
                "bank_append_reason_all_iters": str(info.get("bank_append_reason_all_iters", "")),
                "bank_append_reason_leaf_trigger_only": str(info.get("bank_append_reason_leaf_trigger_only", "")),
                "fusion_leaf_trigger_bank_empty": bool(info.get("fusion_leaf_trigger_bank_empty", False)),
                "selector_candidate_branch_count": int(len(selector_candidate_paths_by_idx.get(sample_idx, []))),
                "selector_scored_top": selector_scored_rows_by_idx.get(sample_idx, []),
                "local_selector_scored_top": selector_scored_rows_by_idx.get(sample_idx, []),
                "ended_beam_reseat_scored_top": ended_reseat_scored_rows_by_idx.get(sample_idx, []),
                "ended_beam_reseat_selected_paths": ended_reseat_selected_rows_by_idx.get(sample_idx, []),
                "global_escape_scored_top": selector_global_escape_rows_by_idx.get(sample_idx, []),
                "global_escape_selected_paths": selector_global_escape_selected_by_idx.get(sample_idx, []),
                "global_escape_replaced_local_paths": selector_global_escape_replaced_by_idx.get(sample_idx, []),
                "candidate_branch_count": int(sum(len(x) for x in slates[sample_idx])) if sample_idx < len(slates) else 0,
                "cumulative_pool_pre_size": int(info.get("cumulative_pool_pre_size", 0)),
                "masked_pool_pre_size": int(info.get("masked_pool_pre_size", 0)),
                "cumulative_pool_eval_size": int(info.get("cumulative_pool_eval_size", 0)),
                "pre_hit_paths": info.get("pre_hit_paths", []),
                "pre_hit_doc_ids": info.get("pre_hit_doc_ids", []),
                "local_paths": info.get("eval_paths", []),
                "local_doc_ids": info.get("eval_doc_ids", []),
                "active_eval_paths": info.get("active_eval_paths", []),
                "active_eval_doc_ids": info.get("active_eval_doc_ids", []),
                "active_eval_leaf_trigger_only_paths": info.get("active_eval_leaf_trigger_only_paths", []),
                "active_eval_leaf_trigger_only_doc_ids": info.get("active_eval_leaf_trigger_only_doc_ids", []),
                "gold_doc_ids": info.get("gold_doc_ids", []),
                "metrics": metrics,
            }
        )

    all_eval_metric_dfs.append(iter_df)

    if not iter_df.empty:
        logger.info(
            "Iter %d | nDCG@10=%.2f | Recall@100=%.2f | Coverage=%.2f | BranchHit@B=%.2f | BranchPrecision@B=%.2f | NumSelectedBranches=%.2f",
            iter_idx,
            float(iter_df["nDCG@10"].mean()),
            float(iter_df["Recall@100"].mean()),
            float(iter_df["Coverage"].mean()),
            float(iter_df["BranchHit@B"].mean()),
            float(iter_df["BranchPrecision@B"].mean()),
            float(iter_df["NumSelectedBranches"].mean()),
        )
        if round6_method2:
            leaf_trigger_diag_values = [
                float(x) for x in fusion_diag_ndcg_leaf_trigger_only if np.isfinite(float(x))
            ]
            logger.info(
                "Iter %d method2 | bank_all_iters_nDCG@10_mean=%.2f | fused_rrf_all_iters=%.2f | fused_max_all_iters=%.2f | fused_sum_all_iters=%.2f | nDCG_leaf_trigger_only=%.2f",
                iter_idx,
                float(np.mean(bank_run_ndcg_values_all_iters)) if bank_run_ndcg_values_all_iters else float("nan"),
                float(np.mean(fusion_diag_ndcg_by_mode_all_iters.get("rrf", [])))
                if fusion_diag_ndcg_by_mode_all_iters.get("rrf")
                else float("nan"),
                float(np.mean(fusion_diag_ndcg_by_mode_all_iters.get("max_score", [])))
                if fusion_diag_ndcg_by_mode_all_iters.get("max_score")
                else float("nan"),
                float(np.mean(fusion_diag_ndcg_by_mode_all_iters.get("sum_score", [])))
                if fusion_diag_ndcg_by_mode_all_iters.get("sum_score")
                else float("nan"),
                float(np.mean(leaf_trigger_diag_values))
                if leaf_trigger_diag_values
                else float("nan"),
            )
    else:
        logger.info("Iter %d | no rows", iter_idx)

save_exp(
    RESULTS_DIR,
    hp,
    llm_api,
    all_eval_samples,
    all_eval_metric_dfs,
    allow_overwrite=True,
    save_llm_api_history=True,
)
logger.info("Saved Round6 results to %s", RESULTS_DIR)
