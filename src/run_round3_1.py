import asyncio
import json
import logging
import os
import pickle as pkl
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from json_repair import repair_json
from tqdm.autonotebook import tqdm

from cache_utils import _prompt_cache_key, append_jsonl
from flat_then_tree import FlatHit, gate_hit, is_prefix
from history_prompts import build_retrieval_history_block, prepend_history_to_prompt
from hyperparams import HyperParams
from llm_apis import GenAIAPI, VllmAPI
from retrievers.diver import DiverEmbeddingModel
from rewrite_prompts import REWRITE_PROMPT_TEMPLATES
from tree_objects import SemanticNode
from utils import (
    compute_ndcg,
    compute_node_registry,
    compute_recall,
    get_all_leaf_nodes_with_path,
    get_node_id,
    normalize_embeddings,
    save_exp,
    setup_logger,
)


@dataclass
class Round3Sample:
    original_query: str
    gold_paths: List[Tuple[int, ...]]
    excluded_ids: List[str]
    last_rewrite: str = ""
    last_action: str = "exploit"
    last_actions: Dict[str, str] = None
    last_possible_docs: Dict[str, str] = None
    rewrite_history: List[Dict] = None
    iter_records: List[Dict] = None

    def __post_init__(self) -> None:
        if self.rewrite_history is None:
            self.rewrite_history = []
        if self.iter_records is None:
            self.iter_records = []
        if self.last_actions is None:
            self.last_actions = {}
        if self.last_possible_docs is None:
            self.last_possible_docs = {}

    def to_dict(self) -> Dict:
        return {
            "original_query": self.original_query,
            "gold_paths": self.gold_paths,
            "excluded_ids": self.excluded_ids,
            "last_rewrite": self.last_rewrite,
            "last_action": self.last_action,
            "last_actions": self.last_actions,
            "last_possible_docs": self.last_possible_docs,
            "rewrite_history": self.rewrite_history,
            "iter_records": self.iter_records,
        }


CATEGORY_ORDER = ["Theory", "Entity", "Example", "Other"]


def _format_action_prompt(
    template: str,
    original_query: str,
    previous_rewrite: str,
    previous_docs: Dict[str, str],
    leaf_descs: List[str],
    branch_descs: List[str],
    retrieval_history: str = "",
) -> str:
    leaf_blob = "\n".join([x for x in leaf_descs if x])
    branch_blob = "\n".join([x for x in branch_descs if x])
    gate_blob = "\n".join([x for x in (leaf_descs + branch_descs) if x])
    prev_lines = []
    for key in CATEGORY_ORDER:
        val = (previous_docs or {}).get(key, "")
        if val:
            prev_lines.append(f"- {key}: {val}")
    prev_blob = "\n".join(prev_lines) if prev_lines else "None"
    if not branch_blob:
        template = template.replace("Branch Context:\n{branch_descs}\n", "")
    try:
        prompt = template.format(
            original_query=(original_query or ""),
            previous_rewrite=(previous_rewrite or ""),
            previous_docs=prev_blob,
            leaf_descs=leaf_blob,
            branch_descs=branch_blob,
            gate_descs=gate_blob,
        )
        return prepend_history_to_prompt(prompt, retrieval_history)
    except KeyError:
        prompt = (
            template
            .replace("{original_query}", original_query or "")
            .replace("{previous_rewrite}", previous_rewrite or "")
            .replace("{previous_docs}", prev_blob)
            .replace("{leaf_descs}", leaf_blob)
            .replace("{branch_descs}", branch_blob)
            .replace("{gate_descs}", gate_blob)
        )
        return prepend_history_to_prompt(prompt, retrieval_history)


def _normalize_action(value: object) -> str:
    action = str(value or "").strip().upper()
    if action not in {"EXPLORE", "EXPLOIT", "PRUNE", "HOLD"}:
        return "EXPLOIT"
    return action


def _parse_action_output(text: str) -> Tuple[Dict[str, str] | None, Dict[str, str] | None, str, str]:
    cleaned = text.split("</think>\n")[-1].strip()
    if "```" in cleaned:
        try:
            parts = cleaned.split("```")
            fenced = [parts[i] for i in range(1, len(parts), 2)]
            if fenced:
                cleaned = fenced[-1].strip()
        except Exception:
            pass
    raw = cleaned
    if "```" in raw:
        try:
            parts = raw.split("```")
            fenced = [parts[i] for i in range(1, len(parts), 2)]
            if fenced:
                raw = fenced[-1].strip()
        except Exception:
            pass
    obj = None
    try:
        obj = json.loads(raw)
    except Exception:
        try:
            obj = repair_json(raw, return_objects=True)
        except Exception:
            obj = None
    if isinstance(obj, dict):
        raw_actions = obj.get("Actions") or obj.get("actions") or {}
        raw_docs = obj.get("Possible_Answer_Docs") or obj.get("possible_answer_docs") or {}
        actions: Dict[str, str] = {}
        docs: Dict[str, str] = {}
        if isinstance(raw_actions, dict) and raw_actions:
            for key in CATEGORY_ORDER:
                if key in raw_actions:
                    actions[key] = _normalize_action(raw_actions.get(key))
        if isinstance(raw_docs, dict) and raw_docs:
            for key in CATEGORY_ORDER:
                val = raw_docs.get(key, "")
                if isinstance(val, (list, dict)):
                    val = json.dumps(val, ensure_ascii=False)
                if isinstance(val, str) and val.strip():
                    docs[key] = val.strip()
        if actions:
            return actions, docs, "", ""
        action = _normalize_action(obj.get("action") or obj.get("Action") or "EXPLOIT").lower()
        rewrite = ""
        if "Possible_Answer_Docs" in obj:
            docs_map = obj.get("Possible_Answer_Docs")
            if isinstance(docs_map, dict):
                rewrite = "\n".join([str(v) for v in docs_map.values() if v]).strip()
            elif isinstance(docs_map, list):
                rewrite = "\n".join([str(v) for v in docs_map if v]).strip()
        if not rewrite:
            raw_rewrite = obj.get("rewrite") or ""
            if isinstance(raw_rewrite, (list, dict)):
                raw_rewrite = json.dumps(raw_rewrite, ensure_ascii=False)
            rewrite = str(raw_rewrite or "").strip()
        return None, None, action, rewrite
    return None, None, "exploit", cleaned.strip()


def _format_v5_rerank_prompt(
    query: str,
    candidates: Sequence[Dict[str, str]],
) -> str:
    lines = [
        "You are given a query and retrieved documents.\n",
        "The documents we want are the ones that would be used as core evidence or justification for that answer to the query. Infer which academic terms, theories, models, examples, or canonical methods would be cited as a core evidence.\n",
        "Reorder the retrieved documents (do remove non-relevant ones).",
        "The results should be sorted in descending order of relevance.",
        "Output format:",
        "{",
        "\"ranks\": [\"[doc_idx]\", \"[doc_idx]\", ...],",
        "\"reason\": \"<reason for this action>\"",
        "}",
        "",
        f"Query:\n{query}",
        "",
        "Retrieved documents:",
    ]
    for i, cand in enumerate(candidates, start=1):
        doc_id = str(cand.get("doc_id", "")).strip()
        desc = str(cand.get("desc", "")).strip().replace("\n", " ")
        lines.append(f"[{i}]. summary={doc_id} {desc}")
    return "\n".join(lines)


def _parse_v5_rerank_output(text: str) -> Tuple[List[int], str]:
    cleaned = text.split("</think>\n")[-1].strip()
    if "```" in cleaned:
        try:
            parts = cleaned.split("```")
            fenced = [parts[i] for i in range(1, len(parts), 2)]
            if fenced:
                cleaned = fenced[-1].strip()
        except Exception:
            pass
    obj = None
    try:
        obj = json.loads(cleaned)
    except Exception:
        try:
            obj = repair_json(cleaned, return_objects=True)
        except Exception:
            obj = None
    if not isinstance(obj, dict):
        return [], ""
    raw_ranks = obj.get("ranks", [])
    ranks: List[int] = []
    if isinstance(raw_ranks, list):
        for x in raw_ranks:
            s = str(x or "").strip()
            if not s:
                continue
            if s.startswith("[") and s.endswith("]"):
                s = s[1:-1].strip()
            try:
                ranks.append(int(s))
            except Exception:
                continue
    reason = str(obj.get("reason", obj.get("reason ", "")) or "").strip()
    return ranks, reason


def _apply_v5_rerank_to_candidates(
    ranked_indices: Sequence[int],
    candidate_paths: Sequence[Tuple[int, ...]],
    path_to_doc_id: Dict[Tuple[int, ...], str],
    topk: int,
) -> Tuple[List[Tuple[int, ...]], List[str]]:
    topk = max(1, int(topk))
    selected_paths: List[Tuple[int, ...]] = []
    selected_doc_ids: List[str] = []
    seen_paths: set[Tuple[int, ...]] = set()
    for rank_idx in ranked_indices:
        cand_pos = int(rank_idx) - 1
        if cand_pos < 0 or cand_pos >= len(candidate_paths):
            continue
        path = tuple(candidate_paths[cand_pos])
        if path in seen_paths:
            continue
        selected_paths.append(path)
        selected_doc_ids.append(str(path_to_doc_id.get(path, "")).strip() or f"path:{path}")
        seen_paths.add(path)
        if len(selected_paths) >= topk:
            break
    # Intent: if LLM returns fewer than top-k indices, backfill by the original candidate order.
    for path in candidate_paths:
        path_t = tuple(path)
        if path_t in seen_paths:
            continue
        doc_id = str(path_to_doc_id.get(path_t, "")).strip()
        selected_paths.append(path_t)
        selected_doc_ids.append(doc_id or f"path:{path_t}")
        seen_paths.add(path_t)
        if len(selected_paths) >= topk:
            break
    return selected_paths[:topk], selected_doc_ids[:topk]


def _truncate_text_words(text: str, max_words: int) -> str:
    if max_words <= 0:
        return ""
    words = str(text or "").split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def _flatten_docs_by_action(docs: Dict[str, str], actions: Dict[str, str]) -> str:
    pieces: List[str] = []
    for key in CATEGORY_ORDER:
        action = actions.get(key, "EXPLOIT")
        if action == "PRUNE":
            continue
        text = docs.get(key, "")
        if text:
            pieces.append(text)
    return "\n".join(pieces).strip()


def _compose_query_from_docs(original_query: str, docs: Dict[str, str], actions: Dict[str, str]) -> str:
    blob = _flatten_docs_by_action(docs, actions)
    if not blob:
        return original_query
    return (original_query + " " + blob).strip()


def _apply_action_rewrite(original_query: str, action: str, rewrite: str, explore_mode: str) -> str:
    rewrite = (rewrite or "").strip()
    if not rewrite:
        return original_query
    if action == "explore":
        if explore_mode == "concat":
            return (original_query + " " + rewrite).strip()
        return rewrite
    return (original_query + " " + rewrite).strip()


def _strip_original_query_prefix(original_query: str, query_t: str) -> str:
    original = (original_query or "").strip()
    query = (query_t or "").strip()
    if not original:
        return query
    # Intent: history should focus on iterative rewrite deltas, not repeated original-query prefix.
    if query == original:
        return ""
    if query.startswith(original + " "):
        return query[len(original):].strip()
    if query.startswith(original):
        return query[len(original):].strip()
    return query


def _hits_to_context_descs(
    hits: Sequence[FlatHit],
    node_registry: Sequence[object],
    topk: int,
    max_desc_len: int | None,
) -> List[str]:
    descs: List[str] = []
    seen: set[int] = set()
    for h in hits:
        ridx = int(h.registry_idx)
        if ridx in seen:
            continue
        seen.add(ridx)
        desc = node_registry[ridx].desc
        if max_desc_len:
            desc = desc[:max_desc_len]
        descs.append(desc)
        if len(descs) >= topk:
            return descs
    return descs


def _paths_to_context_descs(
    paths: Sequence[Tuple[int, ...]],
    node_by_path: Dict[Tuple[int, ...], object],
    topk: int,
    max_desc_len: int | None,
) -> List[str]:
    descs: List[str] = []
    seen: set[Tuple[int, ...]] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        node = node_by_path.get(tuple(path))
        if not node:
            continue
        desc = node.desc
        if max_desc_len:
            desc = desc[:max_desc_len]
        descs.append(desc)
        if len(descs) >= topk:
            return descs
    return descs


def _build_active_branches(
    leaf_hits: Sequence[FlatHit],
    branch_hits: Sequence[FlatHit],
) -> Tuple[List[Tuple[int, ...]], Dict[Tuple[int, ...], float], Dict[Tuple[int, ...], float]]:
    leaf_prefix_counts: Dict[Tuple[int, ...], int] = {}
    for h in leaf_hits:
        path = h.path
        for d in range(1, len(path)):
            prefix = path[:d]
            leaf_prefix_counts[prefix] = leaf_prefix_counts.get(prefix, 0) + 1
    branch_scores: Dict[Tuple[int, ...], float] = {}
    for h in branch_hits:
        branch_scores[h.path] = max(branch_scores.get(h.path, float("-inf")), h.score)

    active = set(branch_scores.keys())
    leaf_total = max(1, len(leaf_hits))
    densities = {p: leaf_prefix_counts.get(p, 0) / float(leaf_total) for p in active}
    return list(active), densities, branch_scores


def _rank_branch_paths(
    paths: Sequence[Tuple[int, ...]],
    densities: Dict[Tuple[int, ...], float],
    branch_scores: Dict[Tuple[int, ...], float],
) -> List[Tuple[int, ...]]:
    ranked = sorted(
        paths,
        key=lambda p: (
            densities.get(p, 0.0),
            branch_scores.get(p, 0.0),
            -len(p),
        ),
        reverse=True,
    )
    return ranked


def _filter_leaf_indices_by_prefixes(
    leaf_indices: Sequence[int],
    leaf_paths: Sequence[Tuple[int, ...]],
    prefixes: Sequence[Tuple[int, ...]],
) -> List[int]:
    prefix_set = set(prefixes)
    if not prefix_set:
        return []
    local_indices: List[int] = []
    for idx, path in zip(leaf_indices, leaf_paths):
        for prefix in prefix_set:
            if is_prefix(prefix, path):
                local_indices.append(idx)
                break
    return local_indices


def _greedy_descendant_leaf_path(
    branch_path: Tuple[int, ...],
    scores_all: np.ndarray,
    node_by_path: Dict[Tuple[int, ...], object],
) -> Optional[Tuple[int, ...]]:
    node = node_by_path.get(tuple(branch_path))
    if node is None:
        return None
    # Intent: v3 follows the highest-score child at each depth until reaching a leaf.
    while node.child:
        best_child = None
        best_score = float("-inf")
        for child in node.child:
            score = float(scores_all[child.registry_idx])
            if score > best_score:
                best_score = score
                best_child = child
        if best_child is None:
            return None
        node = best_child
    return tuple(node.path)


def _path_avg_score(
    branch_path: Tuple[int, ...],
    leaf_path: Tuple[int, ...],
    scores_all: np.ndarray,
    node_by_path: Dict[Tuple[int, ...], object],
) -> float:
    total = 0.0
    count = 0
    for depth in range(len(branch_path), len(leaf_path) + 1):
        prefix = leaf_path[:depth]
        node = node_by_path.get(prefix)
        if node is None:
            continue
        total += float(scores_all[node.registry_idx])
        count += 1
    if count == 0:
        return float("-inf")
    return total / float(count)


def _topk_from_scores(scores: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    if topk <= 0 or scores.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    k = min(topk, scores.shape[0])
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx.astype(np.int64, copy=False), scores[idx].astype(np.float32, copy=False)


def _anchor_leaf_only_paths(anchor_hits: Sequence[FlatHit], topk: int) -> List[Tuple[int, ...]]:
    paths: List[Tuple[int, ...]] = []
    seen_paths: set[Tuple[int, ...]] = set()
    for h in anchor_hits:
        if not h.is_leaf:
            continue
        path_t = tuple(h.path)
        if path_t in seen_paths:
            continue
        paths.append(path_t)
        seen_paths.add(path_t)
        if len(paths) >= topk:
            break
    return paths


def _doc_id_prefix(doc_id: str) -> str:
    text = str(doc_id or "").strip()
    if not text:
        return ""
    if text.endswith(".txt"):
        text = text[:-4]
    return re.sub(r"_\d+$", "", text)


def _build_v6_candidate_paths(
    anchor_hits: Sequence[FlatHit],
    node_embs: np.ndarray,
    node_registry: Sequence[object],
    path_to_doc_id: Dict[Tuple[int, ...], str],
    leaf_indices: Sequence[int],
    base_topk: int,
    candidate_topk: int,
    leaf_knn_indices: Optional[np.ndarray],
    leaf_knn_scores: Optional[np.ndarray],
    leaf_knn_row_by_registry: Optional[Dict[int, int]],
) -> List[Tuple[int, ...]]:
    base_paths = _anchor_leaf_only_paths(anchor_hits, topk=base_topk)
    if len(base_paths) >= candidate_topk:
        return list(base_paths[:candidate_topk])

    seed_leaf_indices: List[int] = []
    seen_seed_paths: set[Tuple[int, ...]] = set()
    for h in anchor_hits:
        if not h.is_leaf:
            continue
        path_t = tuple(h.path)
        if path_t in seen_seed_paths:
            continue
        seed_leaf_indices.append(int(h.registry_idx))
        seen_seed_paths.add(path_t)
        if len(seed_leaf_indices) >= base_topk:
            break

    if not seed_leaf_indices:
        return list(base_paths)

    seed_set = set(seed_leaf_indices)
    seen_paths: set[Tuple[int, ...]] = {tuple(p) for p in base_paths}
    base_prefixes = {
        _doc_id_prefix(path_to_doc_id.get(tuple(path), ""))
        for path in base_paths
    }
    base_prefixes.discard("")
    extra_needed = max(0, int(candidate_topk) - len(base_paths))
    if extra_needed <= 0:
        return list(base_paths[:candidate_topk])

    if leaf_knn_indices is not None and leaf_knn_scores is not None and leaf_knn_row_by_registry:
        score_by_registry_idx: Dict[int, float] = {}
        for seed_idx in seed_leaf_indices:
            row = leaf_knn_row_by_registry.get(int(seed_idx))
            if row is None:
                continue
            neigh_idx_row = leaf_knn_indices[int(row)]
            neigh_score_row = leaf_knn_scores[int(row)]
            for neigh_idx, neigh_score in zip(neigh_idx_row.tolist(), neigh_score_row.tolist()):
                neigh_idx = int(neigh_idx)
                if neigh_idx < 0 or neigh_idx >= len(node_registry):
                    continue
                if neigh_idx in seed_set:
                    continue
                path_t = tuple(node_registry[neigh_idx].path)
                if path_t in seen_paths:
                    continue
                doc_prefix = _doc_id_prefix(path_to_doc_id.get(path_t, ""))
                if doc_prefix in base_prefixes:
                    continue
                prev = score_by_registry_idx.get(neigh_idx, float("-inf"))
                if float(neigh_score) > prev:
                    score_by_registry_idx[neigh_idx] = float(neigh_score)
        if score_by_registry_idx:
            ranked = sorted(score_by_registry_idx.items(), key=lambda x: x[1], reverse=True)
            extra_paths: List[Tuple[int, ...]] = []
            for reg_idx, _ in ranked:
                path_t = tuple(node_registry[int(reg_idx)].path)
                if path_t in seen_paths:
                    continue
                extra_paths.append(path_t)
                seen_paths.add(path_t)
                if len(extra_paths) >= extra_needed:
                    break
            if len(extra_paths) >= extra_needed:
                # Intent: precomputed leaf-kNN graph accelerates v6 without changing seed-prefix filtering logic.
                return list(base_paths) + extra_paths[:extra_needed]

    candidate_indices: List[int] = []
    candidate_paths: List[Tuple[int, ...]] = []

    for leaf_idx in leaf_indices:
        leaf_idx = int(leaf_idx)
        if leaf_idx in seed_set:
            continue
        leaf_node = node_registry[leaf_idx]
        path_t = tuple(leaf_node.path)
        doc_prefix = _doc_id_prefix(path_to_doc_id.get(path_t, ""))
        if doc_prefix in base_prefixes:
            continue
        candidate_indices.append(leaf_idx)
        candidate_paths.append(path_t)

    if not candidate_indices:
        return list(base_paths)

    # Intent: v6 extra docs are chosen by max cosine similarity to any seed leaf, not by query scores.
    seed_embs = node_embs[np.array(seed_leaf_indices, dtype=np.int64)]
    cand_embs = node_embs[np.array(candidate_indices, dtype=np.int64)]
    sim_matrix = cand_embs @ seed_embs.T
    cand_scores = sim_matrix.max(axis=1)
    order = np.argsort(-cand_scores)

    extra_paths: List[Tuple[int, ...]] = []
    for ridx in order.tolist():
        path_t = candidate_paths[int(ridx)]
        if path_t in seen_paths:
            continue
        extra_paths.append(path_t)
        seen_paths.add(path_t)
        if len(extra_paths) >= extra_needed:
            break

    return list(base_paths) + extra_paths


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
        hits.append(FlatHit(registry_idx=registry_idx, path=tuple(node.path), score=float(score), is_leaf=node.is_leaf))
    return hits


def _anchor_ordered_local_hits(
    anchor_hits: Sequence[FlatHit],
    anchor_topk: int,
    scores_all: np.ndarray,
    node_registry: Sequence[object],
    leaf_indices_by_prefix: Dict[Tuple[int, ...], List[int]],
    local_rank_mode: str,
    node_by_path: Dict[Tuple[int, ...], object],
    fallback_hits: Sequence[FlatHit],
    local_topk: int,
) -> List[FlatHit]:
    ordered: List[FlatHit] = []
    seen: set[int] = set()
    for h in anchor_hits[:anchor_topk]:
        if h.is_leaf:
            if h.registry_idx in seen:
                continue
            ordered.append(h)
            seen.add(h.registry_idx)
            continue
        if local_rank_mode == "v3":
            leaf_path = _greedy_descendant_leaf_path(h.path, scores_all, node_by_path)
            if leaf_path is None:
                continue
            leaf_node = node_by_path.get(tuple(leaf_path))
            if leaf_node is None:
                continue
            if leaf_node.registry_idx in seen:
                continue
            ordered.append(
                FlatHit(
                    registry_idx=leaf_node.registry_idx,
                    path=tuple(leaf_node.path),
                    score=float(scores_all[leaf_node.registry_idx]),
                    is_leaf=True,
                )
            )
            seen.add(leaf_node.registry_idx)
            continue
        if local_rank_mode == "v4":
            # Intent: v4 selects the leaf with the best average score along the branch→leaf path.
            best_leaf_idx = None
            best_score = float("-inf")
            candidate_indices = leaf_indices_by_prefix.get(h.path, [])
            if not candidate_indices:
                continue
            for leaf_idx in candidate_indices:
                leaf_node = node_registry[int(leaf_idx)]
                avg_score = _path_avg_score(h.path, tuple(leaf_node.path), scores_all, node_by_path)
                if avg_score > best_score:
                    best_score = avg_score
                    best_leaf_idx = int(leaf_idx)
            if best_leaf_idx is None:
                continue
            if best_leaf_idx in seen:
                continue
            best_node = node_registry[best_leaf_idx]
            ordered.append(
                FlatHit(
                    registry_idx=best_leaf_idx,
                    path=tuple(best_node.path),
                    score=float(best_score),
                    is_leaf=True,
                )
            )
            seen.add(best_leaf_idx)
            continue
        candidate_indices = leaf_indices_by_prefix.get(h.path, [])
        if not candidate_indices:
            continue
        candidate_scores = scores_all[candidate_indices]
        if candidate_scores.size == 0:
            continue
        order = np.argsort(-candidate_scores)
        for ridx in order.tolist():
            leaf_idx = int(candidate_indices[int(ridx)])
            if leaf_idx in seen:
                continue
            node = node_registry[leaf_idx]
            ordered.append(
                FlatHit(
                    registry_idx=leaf_idx,
                    path=tuple(node.path),
                    score=float(scores_all[leaf_idx]),
                    is_leaf=True,
                )
            )
            seen.add(leaf_idx)
            break
    if len(ordered) < local_topk:
        for h in sorted(fallback_hits, key=lambda hit: hit.score, reverse=True):
            if h.registry_idx in seen:
                continue
            ordered.append(h)
            seen.add(h.registry_idx)
            if len(ordered) >= local_topk:
                break
    return ordered[:local_topk]


def _anchor_local_context_paths(
    anchor_hits: Sequence[FlatHit],
    anchor_topk: int,
    scores_all: np.ndarray,
    node_registry: Sequence[object],
    leaf_indices_by_prefix: Dict[Tuple[int, ...], List[int]],
    topk: int,
    local_rank_mode: str,
    node_by_path: Dict[Tuple[int, ...], object],
) -> List[Tuple[int, ...]]:
    ordered: List[Tuple[int, ...]] = []
    seen: set[int] = set()
    for h in anchor_hits[:anchor_topk]:
        if h.is_leaf:
            if h.registry_idx in seen:
                continue
            ordered.append(tuple(h.path))
            seen.add(h.registry_idx)
            if len(ordered) >= topk:
                return ordered
            continue
        if local_rank_mode == "v3":
            leaf_path = _greedy_descendant_leaf_path(h.path, scores_all, node_by_path)
            if leaf_path is None:
                continue
            leaf_node = node_by_path.get(tuple(leaf_path))
            if leaf_node is None:
                continue
            if leaf_node.registry_idx in seen:
                continue
            ordered.append(tuple(leaf_node.path))
            seen.add(leaf_node.registry_idx)
            if len(ordered) >= topk:
                return ordered
            continue
        if local_rank_mode == "v4":
            # Intent: v4 selects the leaf with the best average score along the branch→leaf path.
            best_leaf_path = None
            best_score = float("-inf")
            candidate_indices = leaf_indices_by_prefix.get(h.path, [])
            if not candidate_indices:
                continue
            for leaf_idx in candidate_indices:
                leaf_node = node_registry[int(leaf_idx)]
                avg_score = _path_avg_score(h.path, tuple(leaf_node.path), scores_all, node_by_path)
                if avg_score > best_score:
                    best_score = avg_score
                    best_leaf_path = tuple(leaf_node.path)
            if best_leaf_path is None:
                continue
            best_node = node_by_path.get(tuple(best_leaf_path))
            if best_node is None:
                continue
            if best_node.registry_idx in seen:
                continue
            ordered.append(tuple(best_node.path))
            seen.add(best_node.registry_idx)
            if len(ordered) >= topk:
                return ordered
            continue
        candidate_indices = leaf_indices_by_prefix.get(h.path, [])
        if not candidate_indices:
            continue
        candidate_scores = scores_all[candidate_indices]
        if candidate_scores.size == 0:
            continue
        order = np.argsort(-candidate_scores)
        for ridx in order.tolist():
            leaf_idx = int(candidate_indices[int(ridx)])
            if leaf_idx in seen:
                continue
            node = node_registry[leaf_idx]
            ordered.append(tuple(node.path))
            seen.add(leaf_idx)
            if len(ordered) >= topk:
                return ordered
            break
    return ordered


def _load_rewrite_action_cache(path: str, force_refresh: bool) -> Tuple[Dict[str, str], Dict[str, object], Dict[str, Dict[str, str]]]:
    rewrite_map: Dict[str, str] = {}
    action_map: Dict[str, object] = {}
    docs_map: Dict[str, Dict[str, str]] = {}
    if not path or force_refresh or (not os.path.exists(path)):
        return rewrite_map, action_map, docs_map
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = rec.get("key")
            if not key:
                continue
            if "rewritten_query" in rec:
                rewrite_map[str(key)] = str(rec.get("rewritten_query", ""))
            if "actions" in rec and isinstance(rec.get("actions"), dict):
                action_map[str(key)] = rec.get("actions")
            elif "action" in rec:
                action_map[str(key)] = str(rec.get("action", "exploit")).strip().lower()
            if "possible_answer_docs" in rec and isinstance(rec.get("possible_answer_docs"), dict):
                docs_map[str(key)] = rec.get("possible_answer_docs")
    return rewrite_map, action_map, docs_map


def _load_router_action_cache(path: str, force_refresh: bool) -> Dict[str, str]:
    action_map: Dict[str, str] = {}
    if not path or force_refresh or (not os.path.exists(path)):
        return action_map
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = rec.get("key")
            if not key:
                continue
            if "action" in rec:
                action_map[str(key)] = str(rec.get("action", "exploit")).strip().lower()
    return action_map


hp = HyperParams.from_args()
if not hp.REWRITE_PROMPT_NAME and not hp.REWRITE_PROMPT_PATH and not hp.REWRITE_CACHE_PATH:
    hp.add_param("rewrite_prompt_name", "round3_action_v1")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
exp_dir_name = str(hp)
RESULTS_DIR = f"{BASE_DIR}/results/{hp.DATASET}/{hp.SUBSET}/{exp_dir_name}/"
if os.path.exists(RESULTS_DIR) and os.listdir(RESULTS_DIR):
    print(f"Results already exist at {RESULTS_DIR}. Skipping run.")
    raise SystemExit(0)
os.makedirs(RESULTS_DIR, exist_ok=True)
log_path = f"{RESULTS_DIR}/run.log"
os.makedirs(os.path.dirname(log_path), exist_ok=True)
logger = setup_logger("lattice_runner_round3", log_path, logging.INFO)
with open(f"{RESULTS_DIR}/hparams.json", "w", encoding="utf-8") as f:
    json.dump(vars(hp), f, indent=2, ensure_ascii=True, sort_keys=True)

if not hp.REWRITE_CACHE_PATH:
    cache_root = os.path.join(BASE_DIR, "cache", "rewrite")
    os.makedirs(cache_root, exist_ok=True)
    prompt_name = hp.REWRITE_PROMPT_NAME or "rewrite"
    tag_parts = [
        f"REM={hp.ROUND3_EXPLORE_MODE}",
        f"ALR={hp.ROUND3_ANCHOR_LOCAL_RANK}",
    ]
    if hp.SUFFIX:
        tag_parts.append(f"S={hp.SUFFIX}")
    tag = "-".join(tag_parts)
    cache_name = f"{hp.SUBSET}_{prompt_name}_{tag}_{hp.exp_hash(8)}.jsonl"
    hp.add_param("rewrite_cache_path", os.path.join(cache_root, cache_name))

if os.path.exists(f"{BASE_DIR}/data/{hp.DATASET}/{hp.SUBSET}/documents.jsonl"):
    docs_df = pd.read_json(f"{BASE_DIR}/data/{hp.DATASET}/{hp.SUBSET}/documents.jsonl", lines=True, dtype={"id": str})
    examples_df = pd.read_json(f"{BASE_DIR}/data/{hp.DATASET}/{hp.SUBSET}/examples.jsonl", lines=True, dtype={"gold_ids": List[str]})
    examples_df["gold_ids"] = examples_df["gold_ids"].apply(lambda x: [str(i) for i in x])
else:
    docs_df = pd.DataFrame(load_dataset("xlangai/BRIGHT", "documents", split=hp.SUBSET))
    examples_df = pd.DataFrame(load_dataset("xlangai/BRIGHT", "examples", split=hp.SUBSET))

doc_id_to_content = {docs_df.iloc[i].id: docs_df.iloc[i].content for i in range(len(docs_df))}

tree_dict = pkl.load(open(f"{BASE_DIR}/trees/{hp.DATASET}/{hp.SUBSET}/tree-{hp.TREE_VERSION}.pkl", "rb"))
semantic_root_node = SemanticNode().load_dict(tree_dict) if isinstance(tree_dict, dict) else tree_dict
node_registry = compute_node_registry(semantic_root_node)
all_leaf_nodes = get_all_leaf_nodes_with_path(semantic_root_node)
doc_id_to_path = {get_node_id(leaf.id, docs_df): path for leaf, path in all_leaf_nodes}
path_to_doc_id = {tuple(path): str(doc_id) for doc_id, path in doc_id_to_path.items()}
node_by_path = {tuple(node.path): node for node in node_registry}

if not hp.RETRIEVER_MODEL_PATH:
    raise ValueError("--retriever_model_path is required")
if not hp.NODE_EMB_PATH:
    raise ValueError("--node_emb_path is required")

node_embs = np.load(hp.NODE_EMB_PATH, allow_pickle=False)
if node_embs.shape[0] != len(node_registry):
    raise ValueError(f"node_embs rows ({node_embs.shape[0]}) must match node_registry size ({len(node_registry)})")
node_embs = normalize_embeddings(node_embs)

retriever = DiverEmbeddingModel(hp.RETRIEVER_MODEL_PATH, local_files_only=True)

leaf_indices = [idx for idx, node in enumerate(node_registry) if node.is_leaf]
leaf_paths = [tuple(node_registry[idx].path) for idx in leaf_indices]
leaf_indices_by_prefix: Dict[Tuple[int, ...], List[int]] = {}
for idx, path in zip(leaf_indices, leaf_paths):
    for d in range(1, len(path)):
        prefix = path[:d]
        leaf_indices_by_prefix.setdefault(prefix, []).append(idx)

v6_leaf_knn_indices: Optional[np.ndarray] = None
v6_leaf_knn_scores: Optional[np.ndarray] = None
v6_leaf_knn_row_by_registry: Optional[Dict[int, int]] = None
if hp.ROUND3_ANCHOR_LOCAL_RANK == "v6" and hp.ROUND3_V6_LEAF_KNN_PATH:
    if not os.path.exists(hp.ROUND3_V6_LEAF_KNN_PATH):
        raise FileNotFoundError(f"--round3_v6_leaf_knn_path not found: {hp.ROUND3_V6_LEAF_KNN_PATH}")
    knn_pack = np.load(hp.ROUND3_V6_LEAF_KNN_PATH, allow_pickle=False)
    leaf_registry_indices = knn_pack["leaf_registry_indices"].astype(np.int64, copy=False)
    v6_leaf_knn_indices = knn_pack["neighbor_registry_indices"].astype(np.int64, copy=False)
    v6_leaf_knn_scores = knn_pack["neighbor_scores"].astype(np.float32, copy=False)
    if v6_leaf_knn_indices.shape != v6_leaf_knn_scores.shape:
        raise ValueError("neighbor_registry_indices and neighbor_scores must have the same shape")
    if v6_leaf_knn_indices.shape[0] != leaf_registry_indices.shape[0]:
        raise ValueError("leaf_registry_indices and neighbor arrays row counts must match")
    v6_leaf_knn_row_by_registry = {int(reg_idx): row for row, reg_idx in enumerate(leaf_registry_indices.tolist())}
    logger.info(
        "Loaded v6 leaf-kNN graph: rows=%d, topk=%d, path=%s",
        v6_leaf_knn_indices.shape[0],
        v6_leaf_knn_indices.shape[1] if v6_leaf_knn_indices.ndim == 2 else -1,
        hp.ROUND3_V6_LEAF_KNN_PATH,
    )

rewrite_enabled = True
rewrite_template = None
rewrite_map: Dict[str, str] = {}
action_map: Dict[str, str] = {}
router_template = None
router_action_map: Dict[str, str] = {}
router_enabled = bool(hp.ROUND3_ROUTER_PROMPT_NAME)
if hp.REWRITE_PROMPT_NAME:
    if hp.REWRITE_PROMPT_NAME not in REWRITE_PROMPT_TEMPLATES:
        raise ValueError(
            f"Unknown --rewrite_prompt_name \"{hp.REWRITE_PROMPT_NAME}\". "
            f"Available: {sorted(REWRITE_PROMPT_TEMPLATES.keys())}"
        )
    if hp.ROUND3_EXPLORE_MODE not in {"replace", "concat"}:
        raise ValueError(
            f"Unknown --round3_explore_mode \"{hp.ROUND3_EXPLORE_MODE}\". "
            "Expected: replace|concat"
        )
    rewrite_template = REWRITE_PROMPT_TEMPLATES[hp.REWRITE_PROMPT_NAME]
if hp.REWRITE_PROMPT_PATH:
    if not os.path.exists(hp.REWRITE_PROMPT_PATH):
        raise ValueError(f"--rewrite_prompt_path not found: {hp.REWRITE_PROMPT_PATH}")
    with open(hp.REWRITE_PROMPT_PATH, "r", encoding="utf-8") as f:
        rewrite_template = f.read()
rewrite_map, action_map, docs_map = _load_rewrite_action_cache(hp.REWRITE_CACHE_PATH, hp.REWRITE_FORCE_REFRESH)
if router_enabled:
    if hp.ROUND3_ROUTER_PROMPT_NAME not in REWRITE_PROMPT_TEMPLATES:
        raise ValueError(
            f"Unknown --round3_router_prompt_name \"{hp.ROUND3_ROUTER_PROMPT_NAME}\". "
            f"Available: {sorted(REWRITE_PROMPT_TEMPLATES.keys())}"
        )
    router_template = REWRITE_PROMPT_TEMPLATES[hp.ROUND3_ROUTER_PROMPT_NAME]
    router_action_map = _load_router_action_cache(hp.ROUND3_ROUTER_CACHE_PATH, hp.ROUND3_ROUTER_FORCE_REFRESH)

if rewrite_template is None:
    rewrite_template = REWRITE_PROMPT_TEMPLATES["round3_action_v1"]

if hp.LLM_API_BACKEND == "genai":
    llm_api = GenAIAPI(hp.LLM, logger=logger, timeout=hp.LLM_API_TIMEOUT, max_retries=hp.LLM_API_MAX_RETRIES)
elif hp.LLM_API_BACKEND == "vllm":
    llm_api = VllmAPI(
        hp.LLM,
        logger=logger,
        timeout=hp.LLM_API_TIMEOUT,
        max_retries=hp.LLM_API_MAX_RETRIES,
        base_url=",".join([f"http://localhost:{8000 + i}/v1" for i in range(4)]),
    )
else:
    raise ValueError(f"Unknown LM API backend: {hp.LLM_API_BACKEND}")

llm_api_kwargs = {
    "max_concurrent_calls": hp.LLM_MAX_CONCURRENT_CALLS,
    "staggering_delay": hp.LLM_API_STAGGERING_DELAY,
    "temperature": 0.7,
}

num_samples = min(examples_df.shape[0], hp.NUM_EVAL_SAMPLES)
all_eval_samples: List[Round3Sample] = []
for i in range(num_samples):
    gold_paths = [doc_id_to_path[doc_id] for doc_id in examples_df.iloc[i]["gold_ids"] if doc_id in doc_id_to_path]
    if len(gold_paths) < len(examples_df.iloc[i]["gold_ids"]):
        logger.warning(f"Some gold IDs for example {i} not found in document paths.")
    sample = Round3Sample(
        original_query=examples_df.iloc[i]["query"][: hp.MAX_QUERY_CHAR_LEN],
        gold_paths=[tuple(p) for p in gold_paths],
        excluded_ids=list(examples_df.iloc[i]["excluded_ids"]),
    )
    all_eval_samples.append(sample)

anchor_topk = hp.ROUND3_ANCHOR_TOPK or hp.FLAT_TOPK
local_topk = hp.ROUND3_LOCAL_TOPK or hp.FLAT_TOPK
global_topk = hp.ROUND3_GLOBAL_TOPK

all_eval_metric_dfs: List[pd.DataFrame] = []
for iter_idx in range(hp.NUM_ITERS):
    logger.info("Round3 iteration %d", iter_idx)
    iter_rows: List[Dict[str, float]] = []

    rewrite_candidates: List[Dict] = []

    anchor_topk_logged = max(1, min(10, int(anchor_topk)))
    anchor_hits_by_sample: List[List[FlatHit]] = []
    anchor_eval_paths_by_sample: List[List[Tuple[int, ...]]] = []
    leaf_hits_by_sample: List[List[FlatHit]] = []
    branch_hits_by_sample: List[List[FlatHit]] = []
    local_hits_by_sample: List[List[FlatHit]] = []
    global_hits_by_sample: List[List[FlatHit]] = []
    branch_paths_by_sample: List[List[Tuple[int, ...]]] = []
    densities_by_sample: List[Dict[Tuple[int, ...], float]] = []
    retrieval_queries_by_sample: List[str] = []
    retrieval_query_actions_by_sample: List[str] = []
    retrieval_query_actions_map_by_sample: List[Dict[str, str]] = []
    retrieval_query_docs_by_sample: List[Dict[str, str]] = []
    rewrite_candidate_idx_by_sample: Dict[int, int] = {}

    v5_enabled = hp.ROUND3_ANCHOR_LOCAL_RANK in {"v5", "v6"}
    v5_mode = hp.ROUND3_ANCHOR_LOCAL_RANK
    v5_candidate_topk = 20
    v5_output_topk = 10
    v5_rerank_prompts: List[str] = []
    v5_rerank_meta: List[Dict] = []
    v5_rerank_doc_ids_by_sample: List[List[str]] = [[] for _ in all_eval_samples]
    v5_rerank_reason_by_sample: List[str] = ["" for _ in all_eval_samples]
    v6_seed_top10_recall_by_sample: List[float] = []
    v6_seed_top20_recall_by_sample: List[float] = []
    v6_expanded_top20_recall_by_sample: List[float] = []

    # tqdm shows per-iteration retrieval progress in terminal runs.
    for sample_idx, sample in enumerate(tqdm(
        all_eval_samples,
        desc=f"Iter {iter_idx} anchor retrieval",
        total=len(all_eval_samples),
        leave=False,
    )):
        if iter_idx == 0 and sample is all_eval_samples[0]:
            logger.info("Iter %d: starting anchor retrieval", iter_idx)
        #   - Single‑action mode → _apply_action_rewrite is used every iter.
        #   - Per‑level actions mode → _compose_query_from_docs is used instead, once last_possible_docs is filled.
        if sample.last_possible_docs:
            # Intent: snapshot the exact query-control state used for this retrieval before rewrite updates it.
            query_action_t = "multi"
            query_actions_t = dict(sample.last_actions or {})
            query_docs_t = dict(sample.last_possible_docs or {})
            anchor_query = _compose_query_from_docs(sample.original_query, sample.last_possible_docs, sample.last_actions)
        else:
            query_action_t = str(sample.last_action or "exploit").strip().lower()
            query_actions_t = {}
            query_docs_t = {}
            anchor_query = _apply_action_rewrite(
                sample.original_query,
                sample.last_action,
                sample.last_rewrite,
                hp.ROUND3_EXPLORE_MODE,
            )
        q_emb = retriever.encode_query(anchor_query)
        scores_all = (node_embs @ q_emb).astype(np.float32, copy=False)
        hits = _hits_from_scores(
            scores=scores_all,
            subset_indices=None,
            node_registry=node_registry,
            topk=anchor_topk,
        )
        anchor_hits_by_sample.append(hits)
        if v5_enabled:
            # Intent: v5/v6 first expands to top-20 candidates, then reranks to top-10 with LLM.
            if v5_mode == "v6":
                v6_seed_top10_paths = _anchor_leaf_only_paths(hits, topk=10)
                v6_seed_top20_paths = _anchor_leaf_only_paths(hits, topk=20)
                v5_candidate_paths = _build_v6_candidate_paths(
                    anchor_hits=hits,
                    node_embs=node_embs,
                    node_registry=node_registry,
                    path_to_doc_id=path_to_doc_id,
                    leaf_indices=leaf_indices,
                    base_topk=10,
                    candidate_topk=v5_candidate_topk,
                    leaf_knn_indices=v6_leaf_knn_indices,
                    leaf_knn_scores=v6_leaf_knn_scores,
                    leaf_knn_row_by_registry=v6_leaf_knn_row_by_registry,
                )
                v6_seed_top10_recall = compute_recall(
                    [list(p) for p in v6_seed_top10_paths[:10]],
                    [list(p) for p in sample.gold_paths],
                    k=10,
                ) * 100.0
                v6_seed_top20_recall = compute_recall(
                    [list(p) for p in v6_seed_top20_paths[:20]],
                    [list(p) for p in sample.gold_paths],
                    k=20,
                ) * 100.0
                v6_expanded_top20_recall = compute_recall(
                    [list(p) for p in v5_candidate_paths[:20]],
                    [list(p) for p in sample.gold_paths],
                    k=20,
                ) * 100.0
                # Intent: compare v6 expansion against leaf-only baselines before LLM rerank.
                v6_seed_top10_recall_by_sample.append(float(v6_seed_top10_recall))
                v6_seed_top20_recall_by_sample.append(float(v6_seed_top20_recall))
                v6_expanded_top20_recall_by_sample.append(float(v6_expanded_top20_recall))
            else:
                # Intent: v5 seed candidates should follow anchor_local_rank=none behavior (leaf-only from anchor order).
                v5_candidate_paths = _anchor_leaf_only_paths(hits, topk=v5_candidate_topk)
            candidate_docs: List[Dict[str, str]] = []
            for path in v5_candidate_paths:
                node = node_by_path.get(tuple(path))
                if not node:
                    continue
                doc_id = str(path_to_doc_id.get(tuple(path), "")).strip()
                if not doc_id:
                    doc_id = f"path:{tuple(path)}"
                desc = node.desc
                if hp.MAX_DOC_DESC_CHAR_LEN:
                    desc = desc[: hp.MAX_DOC_DESC_CHAR_LEN]
                # Intent: cap rerank evidence length per document to control prompt latency and timeout risk.
                desc = _truncate_text_words(desc, 1024)
                candidate_docs.append({"doc_id": doc_id, "desc": desc})
            rerank_prompt = _format_v5_rerank_prompt(anchor_query, candidate_docs)
            v5_rerank_prompts.append(rerank_prompt)
            v5_rerank_meta.append({
                "sample_idx": sample_idx,
                "candidate_paths": v5_candidate_paths,
            })
            anchor_eval_paths = list(v5_candidate_paths)
        elif hp.ROUND3_ANCHOR_LOCAL_RANK != "none":
            anchor_eval_paths = _anchor_local_context_paths(
                anchor_hits=hits,
                anchor_topk=anchor_topk,
                scores_all=scores_all,
                node_registry=node_registry,
                leaf_indices_by_prefix=leaf_indices_by_prefix,
                topk=anchor_topk,
                local_rank_mode=hp.ROUND3_ANCHOR_LOCAL_RANK,
                node_by_path=node_by_path,
            )
        else:
            anchor_eval_paths = [h.path for h in hits if h.is_leaf][:anchor_topk]
        anchor_eval_paths_by_sample.append(anchor_eval_paths)
        leaf_hits = [h for h in hits if h.is_leaf]
        branch_hits = [h for h in hits if not h.is_leaf]
        leaf_hits_by_sample.append(leaf_hits)
        branch_hits_by_sample.append(branch_hits)

        active_branches, densities, branch_scores = _build_active_branches(leaf_hits, branch_hits)
        ranked_branches = _rank_branch_paths(active_branches, densities, branch_scores)
        branch_paths_by_sample.append(ranked_branches)
        densities_by_sample.append(densities)
        retrieval_queries_by_sample.append(anchor_query)
        retrieval_query_actions_by_sample.append(query_action_t)
        retrieval_query_actions_map_by_sample.append(query_actions_t)
        retrieval_query_docs_by_sample.append(query_docs_t)

        local_leaf_indices = _filter_leaf_indices_by_prefixes(leaf_indices, leaf_paths, ranked_branches)
        if local_leaf_indices:
            local_scores = scores_all[local_leaf_indices]
            local_hits = _hits_from_scores(
                scores=local_scores,
                subset_indices=local_leaf_indices,
                node_registry=node_registry,
                topk=local_topk,
            )
        else:
            local_hits = []
        if hp.ROUND3_ANCHOR_LOCAL_RANK != "none":
            local_hits = _anchor_ordered_local_hits(
                anchor_hits=hits,
                anchor_topk=anchor_topk_logged,
                scores_all=scores_all,
                node_registry=node_registry,
                leaf_indices_by_prefix=leaf_indices_by_prefix,
                local_rank_mode=hp.ROUND3_ANCHOR_LOCAL_RANK,
                node_by_path=node_by_path,
                fallback_hits=local_hits,
                local_topk=local_topk,
            )
        elif anchor_topk_logged > 0:
            anchor_branch_paths = [h.path for h in hits[:anchor_topk_logged] if not h.is_leaf]
            if anchor_branch_paths:
                seen_ids = {h.registry_idx for h in local_hits}
                extra_hits: List[FlatHit] = []
                for branch_path in anchor_branch_paths:
                    candidate_indices = leaf_indices_by_prefix.get(branch_path, [])
                    if not candidate_indices:
                        continue
                    candidate_scores = scores_all[candidate_indices]
                    if candidate_scores.size == 0:
                        continue
                    order = np.argsort(-candidate_scores)
                    for ridx in order.tolist():
                        leaf_idx = int(candidate_indices[int(ridx)])
                        if leaf_idx in seen_ids:
                            continue
                        node = node_registry[leaf_idx]
                        extra_hits.append(
                            FlatHit(
                                registry_idx=leaf_idx,
                                path=tuple(node.path),
                                score=float(scores_all[leaf_idx]),
                                is_leaf=True,
                            )
                        )
                        seen_ids.add(leaf_idx)
                        break
                if extra_hits:
                    local_hits.extend(extra_hits)
                    local_hits = sorted(local_hits, key=lambda h: h.score, reverse=True)[:local_topk]
        global_scores = scores_all[leaf_indices]
        global_hits = _hits_from_scores(
            scores=global_scores,
            subset_indices=leaf_indices,
            node_registry=node_registry,
            topk=global_topk,
        )
        local_hits_by_sample.append(local_hits)
        global_hits_by_sample.append(global_hits)

        # NOTE: In this dataset, leaf depth is always > 1. If depth-1 leaves appear later,
        # they will not contribute prefixes to B_active.
        if hp.REWRITE_EVERY <= 0:
            do_rewrite = False
        elif hp.ROUND3_REWRITE_ONCE:
            do_rewrite = (iter_idx == 0) and (not sample.last_rewrite)
        else:
            do_rewrite = (iter_idx % hp.REWRITE_EVERY == 0) or (not sample.last_rewrite)
        if rewrite_enabled and do_rewrite:
            if hp.ROUND3_ANCHOR_LOCAL_RANK in {"v2", "v3", "v4"}:
                # Intent: build rewrite context from flat retrieval order, replacing branch hits with best descendant leaf.
                context_paths = _anchor_local_context_paths(
                    anchor_hits=hits,
                    anchor_topk=anchor_topk,
                    scores_all=scores_all,
                    node_registry=node_registry,
                    leaf_indices_by_prefix=leaf_indices_by_prefix,
                    topk=hp.REWRITE_CONTEXT_TOPK,
                    local_rank_mode=hp.ROUND3_ANCHOR_LOCAL_RANK,
                    node_by_path=node_by_path,
                )
                leaf_descs = _paths_to_context_descs(
                    context_paths,
                    node_by_path,
                    hp.REWRITE_CONTEXT_TOPK,
                    hp.MAX_DOC_DESC_CHAR_LEN,
                )
            else:
                leaf_descs = _hits_to_context_descs(
                    leaf_hits,
                    node_registry,
                    hp.REWRITE_CONTEXT_TOPK,
                    hp.MAX_DOC_DESC_CHAR_LEN,
                )
            needs_branch_descs = hp.ROUND3_REWRITE_CONTEXT == "leaf_branch"
            if needs_branch_descs:
                # Intent: use flat retrieval branch hits (top-K) for rewrite context.
                branch_descs = _hits_to_context_descs(
                    branch_hits,
                    node_registry,
                    hp.REWRITE_CONTEXT_TOPK,
                    hp.MAX_DOC_DESC_CHAR_LEN,
                )
            else:
                branch_descs = []

            router_leaf_descs: List[str] = []
            router_branch_descs: List[str] = []
            if router_enabled:
                # Intent: router sees current evidence (anchor leaf top-K) and alternative directions (non-overlap branches).
                router_leaf_paths: List[Tuple[int, ...]] = []
                for h in hits:
                    if not h.is_leaf:
                        continue
                    node = node_by_path.get(tuple(h.path))
                    if not node:
                        continue
                    desc = node.desc
                    if hp.MAX_DOC_DESC_CHAR_LEN:
                        desc = desc[: hp.MAX_DOC_DESC_CHAR_LEN]
                    router_leaf_descs.append(desc)
                    router_leaf_paths.append(tuple(h.path))
                    if len(router_leaf_descs) >= hp.REWRITE_CONTEXT_TOPK:
                        break
                if hp.ROUND3_ROUTER_CONTEXT in {"leaf_branch", "leaf_branch_depth1"}:
                    for h in hits:
                        if h.is_leaf:
                            continue
                        branch_path = tuple(h.path)
                        if any(is_prefix(branch_path, leaf_path) for leaf_path in router_leaf_paths):
                            continue
                        node = node_by_path.get(branch_path)
                        if not node:
                            continue
                        desc = node.desc
                        if hp.MAX_DOC_DESC_CHAR_LEN:
                            desc = desc[: hp.MAX_DOC_DESC_CHAR_LEN]
                        router_branch_descs.append(desc)
                        if len(router_branch_descs) >= hp.REWRITE_CONTEXT_TOPK:
                            break
                if hp.ROUND3_ROUTER_CONTEXT == "leaf_branch_depth1":
                    # Intent: expose all depth-1 branches to reveal global alternatives.
                    for node in node_registry:
                        if (not node.is_leaf) and len(node.path) == 1:
                            desc = node.desc
                            if hp.MAX_DOC_DESC_CHAR_LEN:
                                desc = desc[: hp.MAX_DOC_DESC_CHAR_LEN]
                            router_branch_descs.append(desc)

            prev_blob = "\n".join([f"{k}: {v}" for k, v in sample.last_possible_docs.items() if v])
            if hp.ROUND3_REWRITE_USE_HISTORY:
                retrieval_history = build_retrieval_history_block(
                    sample.iter_records,
                    path_to_doc_id,
                    topk=hp.ROUND3_REWRITE_HISTORY_TOPK,
                )
            else:
                retrieval_history = ""
            rewrite_candidates.append({
                "sample": sample,
                "leaf_descs": leaf_descs,
                "branch_descs": branch_descs,
                "router_leaf_descs": router_leaf_descs,
                "router_branch_descs": router_branch_descs,
                "prev_blob": prev_blob,
                "retrieval_history": retrieval_history,
            })
            rewrite_candidate_idx_by_sample[sample_idx] = len(rewrite_candidates) - 1

    if v5_enabled and v5_rerank_prompts:
        v5_llm_kwargs = dict(llm_api_kwargs)
        v5_llm_kwargs["temperature"] = 0.0
        logger.info("Iter %d: starting %s rerank batch (%d prompts)", iter_idx, v5_mode, len(v5_rerank_prompts))
        v5_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(v5_loop)
        try:
            v5_outputs = v5_loop.run_until_complete(
                llm_api.run_batch(v5_rerank_prompts, **v5_llm_kwargs)
            )
        finally:
            v5_loop.close()
            asyncio.set_event_loop(None)
        for meta, out in zip(v5_rerank_meta, v5_outputs):
            sample_idx = int(meta["sample_idx"])
            candidate_paths = [tuple(p) for p in meta.get("candidate_paths", [])]
            ranked_indices, reason = _parse_v5_rerank_output(out)
            top_paths, selected_doc_ids = _apply_v5_rerank_to_candidates(
                ranked_indices=ranked_indices,
                candidate_paths=candidate_paths,
                path_to_doc_id=path_to_doc_id,
                topk=v5_output_topk,
            )
            seen_paths: set[Tuple[int, ...]] = set(top_paths)
            tail_paths: List[Tuple[int, ...]] = []
            for path in candidate_paths:
                path_t = tuple(path)
                if path_t in seen_paths:
                    continue
                tail_paths.append(path_t)
            anchor_eval_paths_by_sample[sample_idx] = list(top_paths) + tail_paths
            v5_rerank_doc_ids_by_sample[sample_idx] = selected_doc_ids
            v5_rerank_reason_by_sample[sample_idx] = reason
            if sample_idx in rewrite_candidate_idx_by_sample:
                # Intent: next rewrite context should follow v5 reranked top documents.
                idx = rewrite_candidate_idx_by_sample[sample_idx]
                rewrite_candidates[idx]["leaf_descs"] = _paths_to_context_descs(
                    top_paths,
                    node_by_path,
                    hp.REWRITE_CONTEXT_TOPK,
                    hp.MAX_DOC_DESC_CHAR_LEN,
                )

    if rewrite_candidates:
        # Intent: split router (action) from executor (rewrite) to keep round3_action_v1 content while decoupling decisions.
        router_prompts: List[str] = []
        router_meta: List[Dict] = []
        if router_enabled:
            for meta in rewrite_candidates:
                sample = meta["sample"]
                router_leaf_descs = meta["router_leaf_descs"]
                router_branch_descs = meta["router_branch_descs"] if hp.ROUND3_ROUTER_CONTEXT == "leaf_branch" else []
                retrieval_history = meta.get("retrieval_history", "")
                prompt = _format_action_prompt(
                    router_template,
                    sample.original_query,
                    sample.last_rewrite,
                    sample.last_possible_docs,
                    router_leaf_descs,
                    router_branch_descs,
                    retrieval_history=retrieval_history,
                )
                router_cache_key = _prompt_cache_key("round3_router", prompt)
                if (not hp.ROUND3_ROUTER_FORCE_REFRESH) and (router_cache_key in router_action_map):
                    router_action = router_action_map[router_cache_key]
                    sample.last_action = router_action
                    sample.rewrite_history.append({
                        "iter": iter_idx,
                        "cache_hit": True,
                        "phase": "router",
                        "action": sample.last_action,
                    })
                else:
                    router_prompts.append(prompt)
                    router_meta.append({
                        "sample": sample,
                        "cache_key": router_cache_key,
                        "leaf_descs": router_leaf_descs,
                        "branch_descs": router_branch_descs,
                        "retrieval_history": retrieval_history,
                    })
            if router_prompts:
                logger.info("Iter %d: starting router batch (%d prompts)", iter_idx, len(router_prompts))
                router_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(router_loop)
                try:
                    router_outputs = router_loop.run_until_complete(
                        llm_api.run_batch(router_prompts, **llm_api_kwargs)
                    )
                finally:
                    router_loop.close()
                    asyncio.set_event_loop(None)
                router_records = []
                for meta, out in zip(router_meta, router_outputs):
                    _, _, action, _ = _parse_action_output(out)
                    sample = meta["sample"]
                    sample.last_action = str(action or "exploit").strip().lower()
                    router_action_map[meta["cache_key"]] = sample.last_action
                    sample.rewrite_history.append({
                        "iter": iter_idx,
                        "cache_hit": False,
                        "phase": "router",
                        "action": sample.last_action,
                    })
                    router_records.append({
                        "key": meta["cache_key"],
                        "action": sample.last_action,
                        "prompt_name": hp.ROUND3_ROUTER_PROMPT_NAME,
                        "llm": hp.LLM,
                        "leaf_descs": meta.get("leaf_descs", []),
                        "branch_descs": meta.get("branch_descs", []),
                        "retrieval_history": meta.get("retrieval_history", ""),
                    })
                if hp.ROUND3_ROUTER_CACHE_PATH and router_records:
                    append_jsonl(hp.ROUND3_ROUTER_CACHE_PATH, router_records)

        rewrite_prompts: List[str] = []
        rewrite_meta: List[Dict] = []
        for meta in rewrite_candidates:
            sample = meta["sample"]
            leaf_descs = meta["leaf_descs"]
            branch_descs = meta["branch_descs"] if hp.ROUND3_REWRITE_CONTEXT == "leaf_branch" else []
            retrieval_history = meta.get("retrieval_history", "")
            if router_enabled:
                action = str(sample.last_action or "exploit").strip().lower()
                if action not in {"explore", "exploit"}:
                    action = "exploit"
                executor_prompt_name = (
                    "round3_action_v1_explore" if action == "explore" else "round3_action_v1_exploit"
                )
                executor_template = REWRITE_PROMPT_TEMPLATES[executor_prompt_name]
            else:
                executor_template = rewrite_template
            prompt = _format_action_prompt(
                executor_template,
                sample.original_query,
                sample.last_rewrite,
                sample.last_possible_docs,
                leaf_descs,
                branch_descs,
                retrieval_history=retrieval_history,
            )
            cache_key = _prompt_cache_key("round3", prompt)
            if (not hp.REWRITE_FORCE_REFRESH) and (cache_key in rewrite_map):
                rewrite = rewrite_map[cache_key]
                cached_actions = action_map.get(cache_key)
                cached_docs = docs_map.get(cache_key, {})
                if isinstance(cached_actions, dict):
                    sample.last_actions = cached_actions
                    sample.last_possible_docs = cached_docs
                    sample.last_rewrite = rewrite
                    sample.last_action = "exploit"
                else:
                    action = str(cached_actions or "exploit").strip().lower()
                    if action == "hold" and sample.last_rewrite:
                        rewrite = sample.last_rewrite
                    else:
                        sample.last_rewrite = rewrite
                    sample.last_action = action
                    sample.last_actions = {}
                    sample.last_possible_docs = {}
                sample.rewrite_history.append({
                    "iter": iter_idx,
                    "cache_hit": True,
                    "action": sample.last_action,
                    "actions": sample.last_actions,
                    "possible_docs": sample.last_possible_docs,
                    "rewrite": rewrite,
                })
            else:
                rewrite_prompts.append(prompt)
                rewrite_meta.append({
                    "sample": sample,
                    "cache_key": cache_key,
                    "leaf_descs": leaf_descs,
                    "branch_descs": branch_descs,
                    "retrieval_history": retrieval_history,
                })

        if rewrite_prompts:
            logger.info("Iter %d: starting rewrite batch (%d prompts)", iter_idx, len(rewrite_prompts))
            rewrite_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(rewrite_loop)
            try:
                rewrite_outputs = rewrite_loop.run_until_complete(
                    llm_api.run_batch(rewrite_prompts, **llm_api_kwargs)
                )
            finally:
                rewrite_loop.close()
                asyncio.set_event_loop(None)

            new_records = []
            for meta, out in zip(rewrite_meta, rewrite_outputs):
                actions, docs, action, rewrite = _parse_action_output(out)
                sample = meta["sample"]
                if actions:
                    sample.last_actions = actions
                    sample.last_possible_docs = docs or {}
                    sample.last_rewrite = _flatten_docs_by_action(sample.last_possible_docs, sample.last_actions)
                    sample.last_action = "exploit"
                    rewrite = sample.last_rewrite
                else:
                    if action == "hold" and sample.last_rewrite:
                        rewrite = sample.last_rewrite
                    else:
                        sample.last_rewrite = rewrite
                    sample.last_action = action
                    sample.last_actions = {}
                    sample.last_possible_docs = {}
                sample.rewrite_history.append({
                    "iter": iter_idx,
                    "cache_hit": False,
                    "action": sample.last_action,
                    "actions": sample.last_actions,
                    "possible_docs": sample.last_possible_docs,
                    "rewrite": rewrite,
                })
                rewrite_map[meta["cache_key"]] = rewrite
                action_map[meta["cache_key"]] = sample.last_actions if sample.last_actions else sample.last_action
                if sample.last_possible_docs:
                    docs_map[meta["cache_key"]] = sample.last_possible_docs
                new_records.append({
                    "key": meta["cache_key"],
                    "rewritten_query": rewrite,
                    "action": sample.last_action if not sample.last_actions else None,
                    "actions": sample.last_actions if sample.last_actions else None,
                    "possible_answer_docs": sample.last_possible_docs if sample.last_possible_docs else None,
                    "prompt_name": hp.REWRITE_PROMPT_NAME,
                    "llm": hp.LLM,
                    "leaf_descs": meta.get("leaf_descs", []),
                    "branch_descs": meta.get("branch_descs", []),
                    "retrieval_history": meta.get("retrieval_history", ""),
                })
            if hp.REWRITE_CACHE_PATH and new_records:
                append_jsonl(hp.REWRITE_CACHE_PATH, new_records)

    logger.info("Iter %d: starting local/global scoring", iter_idx)
    anchor_leaf_counts = [sum(1 for h in hits[:anchor_topk_logged] if h.is_leaf) for hits in anchor_hits_by_sample]
    anchor_branch_counts = [sum(1 for h in hits[:anchor_topk_logged] if not h.is_leaf) for hits in anchor_hits_by_sample]
    anchor_branch_hits = []
    for sample, hits in zip(all_eval_samples, anchor_hits_by_sample):
        branches = [h.path for h in hits if not h.is_leaf][:anchor_topk_logged]
        anchor_branch_hits.append(1.0 if gate_hit(branches, sample.gold_paths) else 0.0)
    if anchor_leaf_counts:
        logger.info(
            "Iter %d | Anchor top-%d: leaf=%.2f, branch=%.2f (avg counts)",
            iter_idx,
            anchor_topk_logged,
            float(np.mean(anchor_leaf_counts)),
            float(np.mean(anchor_branch_counts)),
        )
        logger.info(
            "Iter %d | Anchor AncestorHit@%d=%.2f",
            iter_idx,
            anchor_topk_logged,
            float(np.mean(anchor_branch_hits)) * 100.0,
        )
    # tqdm shows per-iteration local/global scoring progress in terminal runs.
    for sample, anchor_eval_paths, leaf_hits, branch_hits, local_hits, global_hits, active_paths, densities, query_t, query_action_t, query_actions_t, query_docs_t, v5_doc_ids_t, v5_reason_t in tqdm(
        zip(
            all_eval_samples,
            anchor_eval_paths_by_sample,
            leaf_hits_by_sample,
            branch_hits_by_sample,
            local_hits_by_sample,
            global_hits_by_sample,
            branch_paths_by_sample,
            densities_by_sample,
            retrieval_queries_by_sample,
            retrieval_query_actions_by_sample,
            retrieval_query_actions_map_by_sample,
            retrieval_query_docs_by_sample,
            v5_rerank_doc_ids_by_sample,
            v5_rerank_reason_by_sample,
        ),
        desc=f"Iter {iter_idx} local/global scoring",
        total=len(all_eval_samples),
        leave=False,
    ):
        local_paths = [list(h.path) for h in local_hits]
        global_paths = [list(h.path) for h in global_hits]
        anchor_paths = [list(p) for p in anchor_eval_paths]
        gold_paths = [list(p) for p in sample.gold_paths]
        local_metrics = {
            "nDCG@10": compute_ndcg(local_paths[:10], gold_paths, k=10) * 100,
            "Recall@10": compute_recall(local_paths[:10], gold_paths, k=10) * 100,
            "Recall@100": compute_recall(local_paths[:100], gold_paths, k=100) * 100,
            "Recall@all": compute_recall(local_paths, gold_paths, k=len(local_paths)) * 100,
            "Coverage": len(local_paths),
        }
        global_metrics = {
            "nDCG@10": compute_ndcg(global_paths[:10], gold_paths, k=10) * 100,
            "Recall@10": compute_recall(global_paths[:10], gold_paths, k=10) * 100,
            "Recall@100": compute_recall(global_paths[:100], gold_paths, k=100) * 100,
            "Recall@all": compute_recall(global_paths, gold_paths, k=len(global_paths)) * 100,
            "Coverage": len(global_paths),
        }
        anchor_metrics = {
            "nDCG@10": compute_ndcg(anchor_paths[:10], gold_paths, k=10) * 100,
            "Recall@10": compute_recall(anchor_paths[:10], gold_paths, k=10) * 100,
            "Recall@100": compute_recall(anchor_paths[:100], gold_paths, k=100) * 100,
            "Recall@all": compute_recall(anchor_paths, gold_paths, k=len(anchor_paths)) * 100,
            "Coverage": len(anchor_paths),
        }
        # Intent: use anchor(flat retrieval) order as the main evaluation list.
        metrics = {
            "nDCG@10": anchor_metrics["nDCG@10"],
            "Recall@10": anchor_metrics["Recall@10"],
            "Recall@100": anchor_metrics["Recall@100"],
            "Recall@all": anchor_metrics["Recall@all"],
            "Coverage": anchor_metrics["Coverage"],
            "Local_nDCG@10": local_metrics["nDCG@10"],
            "Local_Recall@10": local_metrics["Recall@10"],
            "Local_Recall@100": local_metrics["Recall@100"],
            "Local_Recall@all": local_metrics["Recall@all"],
            "Local_Coverage": local_metrics["Coverage"],
            "Global_nDCG@10": global_metrics["nDCG@10"],
            "Global_Recall@10": global_metrics["Recall@10"],
            "Global_Recall@100": global_metrics["Recall@100"],
            "Global_Recall@all": global_metrics["Recall@all"],
            "Global_Coverage": global_metrics["Coverage"],
        }
        iter_rows.append(metrics)

        sample.iter_records.append({
            "iter": iter_idx,
            "action": sample.last_action,
            "actions": sample.last_actions,
            "possible_docs": sample.last_possible_docs,
            "rewrite": sample.last_rewrite,
            "query_t": query_t,
            "query_t_history": _strip_original_query_prefix(sample.original_query, query_t),
            "query_action": query_action_t,
            "query_actions": query_actions_t,
            "query_possible_docs": query_docs_t,
            "v5_rerank_doc_ids": v5_doc_ids_t,
            "v5_rerank_reason": v5_reason_t,
            "anchor_eval_paths": anchor_paths,
            "anchor_leaf_paths": [h.path for h in leaf_hits],
            "anchor_branch_paths": [h.path for h in branch_hits],
            "active_branch_paths": active_paths,
            "density": {str(k): v for k, v in densities.items()},
            "local_paths": local_paths,
            "global_paths": global_paths,
            "local_metrics": local_metrics,
            "global_metrics": global_metrics,
        })

    iter_df = pd.DataFrame(iter_rows)
    all_eval_metric_dfs.append(iter_df)
    if not iter_df.empty:
        logger.info(
            "Iter %d | Anchor nDCG@10=%.2f | Local nDCG@10=%.2f | "
            "Anchor Recall@100=%.2f | Local Recall@100=%.2f",
            iter_idx,
            iter_df["nDCG@10"].mean(),
            iter_df["Local_nDCG@10"].mean(),
            iter_df["Recall@100"].mean(),
            iter_df["Local_Recall@100"].mean(),
        )
        if (
            v5_mode == "v6"
            and v6_seed_top10_recall_by_sample
            and v6_seed_top20_recall_by_sample
            and v6_expanded_top20_recall_by_sample
        ):
            seed10_mean = float(np.mean(v6_seed_top10_recall_by_sample))
            seed20_mean = float(np.mean(v6_seed_top20_recall_by_sample))
            expanded20_mean = float(np.mean(v6_expanded_top20_recall_by_sample))
            delta_vs_seed20 = expanded20_mean - seed20_mean
            improved_ratio = float(
                np.mean(
                    [
                        1.0 if exp > seed else 0.0
                        for seed, exp in zip(v6_seed_top20_recall_by_sample, v6_expanded_top20_recall_by_sample)
                    ]
                )
            ) * 100.0
            logger.info(
                "Iter %d | V6 pre-rerank recall: SeedTop10@10=%.2f | SeedTop20@20=%.2f | ExpandedTop20@20=%.2f (DeltaVsSeed20=%.2f, ImprovedVsSeed20=%.2f%%)",
                iter_idx,
                seed10_mean,
                seed20_mean,
                expanded20_mean,
                delta_vs_seed20,
                improved_ratio,
            )
    else:
        logger.info(
            "Iter %d | Anchor nDCG@10=0.00 | Local nDCG@10=0.00 | "
            "Anchor Recall@100=0.00 | Local Recall@100=0.00",
            iter_idx,
        )

save_exp(RESULTS_DIR, hp, llm_api, all_eval_samples, all_eval_metric_dfs, allow_overwrite=True, save_llm_api_history=True)
logger.info("Saved Round3 results to %s", RESULTS_DIR)
