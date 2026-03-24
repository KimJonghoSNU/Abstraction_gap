import asyncio
import json
import logging
import os
import pickle as pkl
from collections import defaultdict
from dataclasses import dataclass, field
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
from tree_objects import InferSample, PredictionNode, SemanticNode
from utils import (
    chain_path_rel_fn,
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
class MCTSNodeStats:
    path: Tuple[int, ...]
    visits: int = 0
    value_sum: float = 0.0
    immediate_reward: float = 0.0
    expanded: bool = False
    child_paths: List[Tuple[int, ...]] = field(default_factory=list)
    scored_rows: List[Dict[str, Any]] = field(default_factory=list)
    local_pool_size: int = 0

    @property
    def q_value(self) -> float:
        if self.visits > 0:
            return float(self.value_sum) / float(self.visits)
        return float(self.immediate_reward)


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
            seen_direction_evidence="",
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
            .replace("{seen_direction_evidence}", "")
            .replace("{corpus_categories}", "")
        )


def _compose_next_query(original_query: str, rewrite_blob: str, query_pre: str) -> str:
    rewrite = str(rewrite_blob or "").strip()
    if rewrite:
        # Intent: keep retrieval anchored to the original problem statement while appending rewrite hints.
        return (str(original_query or "").strip() + " " + rewrite).strip()
    return str(query_pre or original_query or "").strip()


def _load_rewrite_cache(path: str, force_refresh: bool) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
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
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        ranked.append(doc_id)
    return ranked


def _compute_retrieval_metrics(doc_ids: Sequence[str], gold_doc_ids: Sequence[str]) -> Dict[str, float]:
    ranked_doc_ids = list(doc_ids)
    gold_ranked_doc_ids = list(gold_doc_ids)
    return {
        "nDCG@10": compute_ndcg(ranked_doc_ids[:10], gold_ranked_doc_ids, k=10) * 100,
        "Recall@10": compute_recall(ranked_doc_ids[:10], gold_ranked_doc_ids, k=10) * 100,
        "Recall@100": compute_recall(ranked_doc_ids[:100], gold_ranked_doc_ids, k=100) * 100,
        "Recall@all": compute_recall(ranked_doc_ids, gold_ranked_doc_ids, k=len(ranked_doc_ids)) * 100,
        "Coverage": float(len(ranked_doc_ids)),
    }


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


def _serialize_scored_rows(rows: Sequence[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
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
        good_flags = [any(tuple(branch) == tuple(gold[: len(branch)]) for gold in gold_paths) for branch in selected_paths]
        rows.append(
            {
                "BranchHit@B": float(any(good_flags)),
                "BranchAllHit@B": float(all(good_flags)),
                "BranchPrecision@B": float(np.mean(good_flags)) if good_flags else 0.0,
                "NumSelectedBranches": float(len(selected_paths)),
                "SelectedDepth": float(np.mean([len(path) for path in selected_paths])) if selected_paths else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _reward_mode_to_branch_score_mode(reward_mode: str) -> str:
    reward_mode_s = str(reward_mode or "mean_score").strip().lower()
    if reward_mode_s == "max_score":
        return "max"
    if reward_mode_s == "hit_count":
        return "hit"
    return "mean"


def _score_row_to_relevance(row: Dict[str, Any], reward_mode: str, rollout_topk: int) -> float:
    reward_mode_s = str(reward_mode or "mean_score").strip().lower()
    if reward_mode_s == "hit_count":
        denom = max(1, int(rollout_topk))
        return float(np.clip(float(row.get("matched_count", 0)) / float(denom), 0.0, 1.0))
    score = float(row.get("score", 0.0))
    return float(np.clip((score + 1.0) / 2.0, 0.0, 1.0))


def _ensure_stats(stats_by_path: Dict[Tuple[int, ...], MCTSNodeStats], path: Tuple[int, ...]) -> MCTSNodeStats:
    if path not in stats_by_path:
        stats_by_path[path] = MCTSNodeStats(path=tuple(path))
    return stats_by_path[path]


def _get_prediction_node_by_path(sample: InferSample, path: Tuple[int, ...]) -> Optional[PredictionNode]:
    node = sample.prediction_tree
    if not path:
        return node
    for child_idx in path:
        if not node.child or child_idx < 0 or child_idx >= len(node.child):
            return None
        node = node.child[child_idx]
    return node


def _get_prediction_path_nodes(sample: InferSample, path: Tuple[int, ...]) -> List[PredictionNode]:
    node = sample.prediction_tree
    nodes: List[PredictionNode] = [node]
    for child_idx in path:
        if not node.child or child_idx < 0 or child_idx >= len(node.child):
            break
        node = node.child[child_idx]
        nodes.append(node)
    return nodes


def _path_leaf_pool(
    path: Tuple[int, ...],
    leaf_indices_by_prefix: Dict[Tuple[int, ...], List[int]],
    all_leaf_indices: Sequence[int],
) -> List[int]:
    if not path:
        return list(all_leaf_indices)
    return list(leaf_indices_by_prefix.get(tuple(path), []))


def _expand_prediction_node_for_query(
    *,
    sample: InferSample,
    path: Tuple[int, ...],
    query: str,
    retriever: DiverEmbeddingModel,
    node_embs: np.ndarray,
    node_registry: Sequence[object],
    leaf_indices_by_prefix: Dict[Tuple[int, ...], List[int]],
    leaf_indices: Sequence[int],
    leaf_ancestor_paths: Dict[Tuple[int, ...], List[Tuple[int, ...]]],
    reward_mode: str,
    rollout_topk: int,
    creation_step: int,
) -> Dict[str, Any]:
    pred_node = _get_prediction_node_by_path(sample, path)
    if pred_node is None:
        return {"candidate_rows": [], "all_rows": [], "local_pool_size": 0}

    child_paths_all = [tuple((*path, idx)) for idx, _ in enumerate(pred_node.semantic_node.child)]
    candidate_paths = [tuple((*path, idx)) for idx, child in enumerate(pred_node.semantic_node.child) if not child.is_leaf]
    local_pool = _path_leaf_pool(path, leaf_indices_by_prefix, leaf_indices)
    local_hits = _retrieve_leaf_hits(
        query=query,
        leaf_pool_indices=local_pool,
        retriever=retriever,
        node_embs=node_embs,
        node_registry=node_registry,
        topk=max(1, int(rollout_topk)),
    )
    score_mode = _reward_mode_to_branch_score_mode(reward_mode)
    all_rows_raw = _score_candidate_branches_score(
        local_hits=local_hits,
        candidate_child_paths=child_paths_all,
        leaf_ancestor_paths=leaf_ancestor_paths,
        score_mode=score_mode,
    )
    row_by_path = {tuple(_row_path_tuple(row)): dict(row) for row in all_rows_raw}
    all_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    child_relevances: List[float] = []
    calib_scores: Dict[int, float] = {}

    for idx, child in enumerate(pred_node.semantic_node.child):
        child_path = tuple((*path, idx))
        row = dict(row_by_path.get(child_path, {}))
        if not row:
            row = {
                "path": list(child_path),
                "score": 0.0,
                "score_mode": score_mode,
                "max_score": float("-inf"),
                "mean_score": 0.0,
                "matched_count": 0,
                "best_rank": None,
                "best_leaf_path": [],
            }
        all_rows.append(row)
        local_rel = _score_row_to_relevance(row, reward_mode, rollout_topk)
        child_relevances.append(local_rel)
        calib_scores[int(child.registry_idx)] = float(local_rel)
        if not child.is_leaf:
            candidate_rows.append(row)

    if pred_node.child is None:
        # Intent: MCTS only materializes prediction-tree children when the controller actually visits that branch state.
        pred_node.instantiate_children(child_relevances, reasoning="mcts_retriever", creation_step=creation_step)
    else:
        for child_node, local_rel in zip(pred_node.child, child_relevances):
            child_node.local_relevance = float(local_rel)
            child_node.calibrated_relevance = float(local_rel)
            parent_rel = child_node.parent.path_relevance if child_node.parent else 1.0
            child_node.path_relevance = chain_path_rel_fn(float(local_rel), parent_rel, sample.relevance_chain_factor)

    if not sample.disable_calibration and calib_scores:
        sample.calib_model.add(calib_scores)
        sample.calib_model.fit()
        sample.update_relevances(sample.prediction_tree)

    candidate_rows.sort(
        key=lambda row: (
            -float(row.get("score", 0.0)),
            int(row.get("best_rank", 10**9) or 10**9),
            tuple(row.get("path", [])),
        )
    )
    return {
        "candidate_rows": candidate_rows,
        "all_rows": all_rows,
        "local_pool_size": int(len(local_pool)),
    }


def _ensure_mcts_expanded(
    *,
    sample: InferSample,
    stats_by_path: Dict[Tuple[int, ...], MCTSNodeStats],
    expansion_cache: Dict[Tuple[int, ...], Dict[str, Any]],
    path: Tuple[int, ...],
    query: str,
    retriever: DiverEmbeddingModel,
    node_embs: np.ndarray,
    node_registry: Sequence[object],
    leaf_indices_by_prefix: Dict[Tuple[int, ...], List[int]],
    leaf_indices: Sequence[int],
    leaf_ancestor_paths: Dict[Tuple[int, ...], List[Tuple[int, ...]]],
    reward_mode: str,
    rollout_topk: int,
    creation_step: int,
) -> Dict[str, Any]:
    if path not in expansion_cache:
        expansion_cache[path] = _expand_prediction_node_for_query(
            sample=sample,
            path=path,
            query=query,
            retriever=retriever,
            node_embs=node_embs,
            node_registry=node_registry,
            leaf_indices_by_prefix=leaf_indices_by_prefix,
            leaf_indices=leaf_indices,
            leaf_ancestor_paths=leaf_ancestor_paths,
            reward_mode=reward_mode,
            rollout_topk=rollout_topk,
            creation_step=creation_step,
        )

    cached = expansion_cache[path]
    node_stats = _ensure_stats(stats_by_path, path)
    node_stats.expanded = True
    node_stats.child_paths = [tuple(_row_path_tuple(row)) for row in cached.get("candidate_rows", [])]
    node_stats.scored_rows = _serialize_scored_rows(cached.get("candidate_rows", []), limit=5)
    node_stats.local_pool_size = int(cached.get("local_pool_size", 0))

    for row in cached.get("candidate_rows", []):
        child_path = tuple(_row_path_tuple(row))
        child_stats = _ensure_stats(stats_by_path, child_path)
        child_stats.immediate_reward = float(row.get("score", 0.0))
    return cached


def _uct_value(child: MCTSNodeStats, parent_visits: int, exploration_c: float) -> float:
    exploit = child.q_value
    explore = float(exploration_c) * np.sqrt(np.log(float(parent_visits) + 1.0) / (float(child.visits) + 1.0))
    return float(exploit + explore)


def _run_local_mcts(
    *,
    sample: InferSample,
    search_root_path: Tuple[int, ...],
    query: str,
    stats_by_path: Dict[Tuple[int, ...], MCTSNodeStats],
    retriever: DiverEmbeddingModel,
    node_embs: np.ndarray,
    node_registry: Sequence[object],
    leaf_indices_by_prefix: Dict[Tuple[int, ...], List[int]],
    leaf_indices: Sequence[int],
    leaf_ancestor_paths: Dict[Tuple[int, ...], List[Tuple[int, ...]]],
    reward_mode: str,
    rollout_topk: int,
    num_simulations: int,
    exploration_c: float,
    creation_step: int,
) -> Dict[str, Any]:
    expansion_cache: Dict[Tuple[int, ...], Dict[str, Any]] = {}
    _ensure_stats(stats_by_path, search_root_path)
    for _ in range(max(1, int(num_simulations))):
        visited_paths: List[Tuple[int, ...]] = [search_root_path]
        current_path = search_root_path
        reward = float(_ensure_stats(stats_by_path, current_path).immediate_reward)

        while True:
            cached = _ensure_mcts_expanded(
                sample=sample,
                stats_by_path=stats_by_path,
                expansion_cache=expansion_cache,
                path=current_path,
                query=query,
                retriever=retriever,
                node_embs=node_embs,
                node_registry=node_registry,
                leaf_indices_by_prefix=leaf_indices_by_prefix,
                leaf_indices=leaf_indices,
                leaf_ancestor_paths=leaf_ancestor_paths,
                reward_mode=reward_mode,
                rollout_topk=rollout_topk,
                creation_step=creation_step,
            )
            child_rows = cached.get("candidate_rows", [])
            if not child_rows:
                reward = float(_ensure_stats(stats_by_path, current_path).q_value)
                break

            child_paths = [tuple(_row_path_tuple(row)) for row in child_rows]
            unvisited = [child_path for child_path in child_paths if _ensure_stats(stats_by_path, child_path).visits == 0]
            if unvisited:
                chosen_path = max(
                    unvisited,
                    key=lambda path_t: (
                        float(_ensure_stats(stats_by_path, path_t).immediate_reward),
                        tuple(path_t),
                    ),
                )
                child_stats = _ensure_stats(stats_by_path, chosen_path)
                visited_paths.append(chosen_path)
                reward = float(child_stats.immediate_reward)
                break

            parent_stats = _ensure_stats(stats_by_path, current_path)
            chosen_path = max(
                child_paths,
                key=lambda path_t: (
                    _uct_value(_ensure_stats(stats_by_path, path_t), parent_stats.visits, exploration_c),
                    float(_ensure_stats(stats_by_path, path_t).immediate_reward),
                    tuple(path_t),
                ),
            )
            visited_paths.append(chosen_path)
            current_path = chosen_path

        for visited_path in visited_paths:
            node_stats = _ensure_stats(stats_by_path, visited_path)
            node_stats.visits += 1
            node_stats.value_sum += float(reward)

    root_stats = _ensure_stats(stats_by_path, search_root_path)
    root_child_paths = list(root_stats.child_paths)
    chosen_path: Tuple[int, ...] = search_root_path
    chosen_stats: Optional[MCTSNodeStats] = None
    if root_child_paths:
        chosen_path = max(
            root_child_paths,
            key=lambda path_t: (
                _ensure_stats(stats_by_path, path_t).visits,
                _ensure_stats(stats_by_path, path_t).q_value,
                tuple(path_t),
            ),
        )
        chosen_stats = _ensure_stats(stats_by_path, chosen_path)

    top_children = []
    for child_path in sorted(
        root_child_paths,
        key=lambda path_t: (
            -_ensure_stats(stats_by_path, path_t).visits,
            -_ensure_stats(stats_by_path, path_t).q_value,
            tuple(path_t),
        ),
    )[:5]:
        child_stats = _ensure_stats(stats_by_path, child_path)
        top_children.append(
            {
                "path": [int(x) for x in child_path],
                "visits": int(child_stats.visits),
                "q": float(child_stats.q_value),
                "immediate_reward": float(child_stats.immediate_reward),
            }
        )

    return {
        "search_root_path": search_root_path,
        "chosen_path": chosen_path,
        "chosen_stats": chosen_stats,
        "root_scored_rows": list(root_stats.scored_rows),
        "root_candidate_count": int(len(root_child_paths)),
        "top_children": top_children,
    }


def _backtrack_to_expandable_ancestor(
    active_path: Tuple[int, ...],
    child_branch_paths_by_path: Dict[Tuple[int, ...], List[Tuple[int, ...]]],
) -> Tuple[Tuple[int, ...], bool]:
    current = tuple(active_path)
    backtracked = False
    while current and (not child_branch_paths_by_path.get(tuple(current), [])):
        current = tuple(current[:-1])
        backtracked = True
    return current, backtracked


def _initialize_best_root_child(
    *,
    sample: InferSample,
    stats_by_path: Dict[Tuple[int, ...], MCTSNodeStats],
    retriever: DiverEmbeddingModel,
    node_embs: np.ndarray,
    node_registry: Sequence[object],
    leaf_indices_by_prefix: Dict[Tuple[int, ...], List[int]],
    leaf_indices: Sequence[int],
    leaf_ancestor_paths: Dict[Tuple[int, ...], List[Tuple[int, ...]]],
    reward_mode: str,
    rollout_topk: int,
) -> Tuple[int, ...]:
    expansion_cache: Dict[Tuple[int, ...], Dict[str, Any]] = {}
    cached = _ensure_mcts_expanded(
        sample=sample,
        stats_by_path=stats_by_path,
        expansion_cache=expansion_cache,
        path=(),
        query=str(sample.original_query or sample.query or ""),
        retriever=retriever,
        node_embs=node_embs,
        node_registry=node_registry,
        leaf_indices_by_prefix=leaf_indices_by_prefix,
        leaf_indices=leaf_indices,
        leaf_ancestor_paths=leaf_ancestor_paths,
        reward_mode=reward_mode,
        rollout_topk=rollout_topk,
        creation_step=0,
    )
    child_rows = list(cached.get("candidate_rows", []))
    if not child_rows:
        return ()
    best_path = tuple(_row_path_tuple(child_rows[0]))
    sample.beam_state_paths = [_get_prediction_path_nodes(sample, best_path)]
    sample.mcts_active_path = list(best_path)
    return best_path


hp = HyperParams.from_args()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for forbidden_flag in (
    "ROUND6_GLOBAL_ESCAPE",
    "ROUND6_METHOD2",
):
    if bool(getattr(hp, forbidden_flag, False)):
        raise ValueError(f"run_mcts.py does not support {forbidden_flag.lower()}")
if str(getattr(hp, "ROUND6_EXPANDABLE_MODE", "off") or "off").strip().lower() != "off":
    raise ValueError("run_mcts.py does not support --round6_expandable_mode")

rewrite_prompt_name = str(getattr(hp, "REWRITE_PROMPT_NAME", "agent_executor_v1_icl2") or "agent_executor_v1_icl2")
if rewrite_prompt_name not in REWRITE_PROMPT_TEMPLATES:
    raise ValueError(f'Unknown --rewrite_prompt_name "{rewrite_prompt_name}"')
hp.add_param("rewrite_prompt_name", rewrite_prompt_name)
hp.add_param("mcts_controller", "uct")

if not hp.REWRITE_CACHE_PATH:
    cache_root = os.path.join(BASE_DIR, "cache", "rewrite")
    os.makedirs(cache_root, exist_ok=True)
    cache_name = f"{hp.SUBSET}_{rewrite_prompt_name}_mcts_{hp.exp_hash(8)}.jsonl"
    hp.add_param("rewrite_cache_path", os.path.join(cache_root, cache_name))

exp_dir_name = str(hp)
RESULTS_DIR = f"{BASE_DIR}/results/{hp.DATASET}/{hp.SUBSET}/{exp_dir_name}/"
os.makedirs(RESULTS_DIR, exist_ok=True)
log_path = f"{RESULTS_DIR}/run.log"
logger = setup_logger("mcts_runner", log_path, logging.INFO)
with open(f"{RESULTS_DIR}/hparams.json", "w", encoding="utf-8") as f:
    json.dump(vars(hp), f, indent=2, ensure_ascii=True, sort_keys=True)

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

tree_dict = pkl.load(open(f"{BASE_DIR}/trees/{hp.DATASET}/{hp.SUBSET}/tree-{hp.TREE_VERSION}.pkl", "rb"))
semantic_root_node = SemanticNode().load_dict(tree_dict) if isinstance(tree_dict, dict) else tree_dict
node_registry = compute_node_registry(semantic_root_node)
leaf_indices, _, leaf_indices_by_prefix, leaf_ancestor_paths = _build_tree_leaf_support_maps(node_registry)

path_to_node: Dict[Tuple[int, ...], object] = {tuple(node.path): node for node in node_registry}
child_branch_paths_by_path: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = {}
for node in node_registry:
    path_t = tuple(node.path)
    child_paths = [tuple(child.path) for child in node.child if not child.is_leaf]
    if child_paths:
        child_branch_paths_by_path[path_t] = sorted({tuple(p) for p in child_paths})

leaf_doc_ids: Dict[str, List[Tuple[int, ...]]] = defaultdict(list)
for leaf_idx in leaf_indices:
    node = node_registry[int(leaf_idx)]
    doc_id = get_node_id(node.id, docs_df)
    if not doc_id:
        continue
    leaf_doc_ids[doc_id].append(tuple(node.path))
for doc_id in list(leaf_doc_ids.keys()):
    leaf_doc_ids[doc_id] = sorted({tuple(path) for path in leaf_doc_ids[doc_id]})
path_to_doc_id: Dict[Tuple[int, ...], str] = {}
for doc_id, paths in leaf_doc_ids.items():
    for path in paths:
        path_to_doc_id[tuple(path)] = str(doc_id)

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
    logger.info("Encoding %d node descriptions on-the-fly for MCTS runtime", len(node_registry))
    node_descs = [str(node.desc or "").strip() or "No Description." for node in node_registry]
    node_embs = retriever.encode(node_descs, max_length=4096, batch_size=4)
    node_embs = normalize_embeddings(node_embs)

rewrite_template = REWRITE_PROMPT_TEMPLATES[rewrite_prompt_name]
subset_domain_route_hint = _build_domain_route_hint(hp.SUBSET)
subset_relevance_definition = _get_relevance_definition(hp.SUBSET)
rewrite_map, docs_map = _load_rewrite_cache(hp.REWRITE_CACHE_PATH, hp.REWRITE_FORCE_REFRESH)

if hp.LLM_API_BACKEND == "genai":
    llm_api = GenAIAPI(hp.LLM, logger=logger, timeout=hp.LLM_API_TIMEOUT, max_retries=hp.LLM_API_MAX_RETRIES)
elif hp.LLM_API_BACKEND == "vllm":
    vllm_base_url, vllm_base_url_src = _resolve_vllm_base_url(BASE_DIR)
    logger.info("MCTS vLLM endpoints source: %s", vllm_base_url_src)
    logger.info("MCTS vLLM endpoints: %s", vllm_base_url)
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
    "temperature": 0.0,
}

num_samples = min(examples_df.shape[0], hp.NUM_EVAL_SAMPLES)
all_eval_samples: List[InferSample] = []
all_mcts_stats: List[Dict[Tuple[int, ...], MCTSNodeStats]] = []
for i in range(num_samples):
    raw_gold_ids = [str(x) for x in examples_df.iloc[i]["gold_ids"]]
    gold_doc_ids = [doc_id for doc_id in raw_gold_ids if doc_id in leaf_doc_ids]
    gold_paths: List[Tuple[int, ...]] = []
    for doc_id in gold_doc_ids:
        gold_paths.extend(leaf_doc_ids.get(doc_id, []))
    gold_paths = sorted({tuple(path) for path in gold_paths})

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
    sample.original_query = original_query
    sample.gold_doc_ids = list(gold_doc_ids)
    sample.last_rewrite_raw = ""
    sample.rewrite_history = []
    sample.iter_records = []
    sample.mcts_active_path = []
    if "original_query" not in sample.SAVE_LIST:
        sample.SAVE_LIST.append("original_query")
    if "gold_doc_ids" not in sample.SAVE_LIST:
        sample.SAVE_LIST.append("gold_doc_ids")
    if "last_rewrite_raw" not in sample.SAVE_LIST:
        sample.SAVE_LIST.append("last_rewrite_raw")
    if "iter_records" not in sample.SAVE_LIST:
        sample.SAVE_LIST.append("iter_records")
    if "mcts_active_path" not in sample.SAVE_LIST:
        sample.SAVE_LIST.append("mcts_active_path")
    all_eval_samples.append(sample)
    all_mcts_stats.append({(): MCTSNodeStats(path=())})

if str(hp.MCTS_STATE_INIT or "root").lower() == "best_root_child":
    for sample_idx, sample in enumerate(all_eval_samples):
        best_path = _initialize_best_root_child(
            sample=sample,
            stats_by_path=all_mcts_stats[sample_idx],
            retriever=retriever,
            node_embs=node_embs,
            node_registry=node_registry,
            leaf_indices_by_prefix=leaf_indices_by_prefix,
            leaf_indices=leaf_indices,
            leaf_ancestor_paths=leaf_ancestor_paths,
            reward_mode=str(hp.MCTS_REWARD_MODE or "mean_score"),
            rollout_topk=int(hp.MCTS_ROLLOUT_TOPK),
        )
        logger.info("Initialized sample %d active path to %s", sample_idx, best_path)

all_eval_metric_dfs: List[pd.DataFrame] = []
cumulative_leaf_indices_by_sample: List[Set[int]] = [set() for _ in all_eval_samples]

for iter_idx in range(hp.NUM_ITERS):
    logger.info("MCTS iteration %d", iter_idx)

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

    for sample_idx, sample in enumerate(tqdm(all_eval_samples, desc=f"Iter {iter_idx} rewrite prep", leave=False)):
        active_path_before = tuple(getattr(sample, "mcts_active_path", []) or [])
        search_root_path, backtracked = _backtrack_to_expandable_ancestor(active_path_before, child_branch_paths_by_path)
        cumulative_pool = sorted(cumulative_leaf_indices_by_sample[sample_idx])
        if not cumulative_pool:
            cumulative_pool = list(leaf_indices)

        query_pre = str(sample.query or sample.original_query).strip()
        pre_hits = _retrieve_leaf_hits(
            query=query_pre,
            leaf_pool_indices=cumulative_pool,
            retriever=retriever,
            node_embs=node_embs,
            node_registry=node_registry,
            topk=max(1, int(hp.ROUND5_MRR_POOL_K)),
        )
        pre_hit_paths = [tuple(hit.path) for hit in pre_hits]
        pre_hit_doc_ids = _paths_to_ranked_doc_ids(pre_hit_paths, path_to_doc_id)
        leaf_descs = _hits_to_context_descs(
            pre_hits,
            node_registry,
            topk=hp.REWRITE_CONTEXT_TOPK,
            max_desc_len=hp.MAX_DOC_DESC_CHAR_LEN,
        )
        prompt = _format_rewrite_prompt(
            rewrite_template,
            original_query=str(sample.original_query),
            previous_rewrite=str(getattr(sample, "last_rewrite_raw", "") or ""),
            leaf_descs=leaf_descs,
            domain_route_hint=subset_domain_route_hint,
            relevance_definition=subset_relevance_definition,
        )
        cache_key = _prompt_cache_key("mcts", prompt)
        cache_hit = (not hp.REWRITE_FORCE_REFRESH) and (cache_key in rewrite_map)
        rewrite_state_by_sample_idx[sample_idx] = {
            "query_pre": query_pre,
            "cache_key": cache_key,
            "cache_hit": bool(cache_hit),
            "cached_rewrite": str(rewrite_map.get(cache_key, "") or ""),
            "cached_docs": _clean_docs_map(docs_map.get(cache_key, {})) if cache_hit else {},
            "leaf_descs": leaf_descs,
            "pre_hit_paths": [list(path) for path in pre_hit_paths],
            "pre_hit_doc_ids": list(pre_hit_doc_ids),
            "active_path_before": [int(x) for x in active_path_before],
            "search_root_path": [int(x) for x in search_root_path],
            "backtracked": bool(backtracked),
            "backtracked_from": [int(x) for x in active_path_before] if backtracked else [],
        }
        if not cache_hit:
            rewrite_prompts.append(prompt)
            rewrite_meta.append(
                {
                    "sample_idx": sample_idx,
                    "cache_key": cache_key,
                    "prompt": prompt,
                }
            )

    generated_by_key: Dict[str, Dict[str, Any]] = {}
    new_cache_records: List[Dict[str, Any]] = []
    if rewrite_prompts:
        rewrite_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(rewrite_loop)
        try:
            rewrite_outputs = rewrite_loop.run_until_complete(llm_api.run_batch(rewrite_prompts, **llm_api_kwargs))
        finally:
            rewrite_loop.close()
            asyncio.set_event_loop(None)
        for meta, out in zip(rewrite_meta, rewrite_outputs):
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
                    "prompt_name": rewrite_prompt_name,
                    "llm": hp.LLM,
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
        info["query_post"] = query_post
        info["rewrite"] = rewrite_blob
        info["rewrite_docs"] = rewrite_docs
        info["raw_output"] = raw_output

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
        current_run_metrics = _compute_retrieval_metrics(eval_doc_ids, [str(x) for x in sample.gold_doc_ids])
        info["eval_paths"] = [list(path) for path in eval_paths]
        info["eval_doc_ids"] = list(eval_doc_ids)
        info["metrics"] = {str(k): float(v) for k, v in current_run_metrics.items()}
        info["cumulative_pool_eval_size"] = int(len(cumulative_pool_eval))
        sample.rewrite_history.append(
            {
                "iter": iter_idx,
                "cache_hit": bool(info.get("cache_hit", False)),
                "prompt_name": rewrite_prompt_name,
                "query_pre": query_pre,
                "query_post": query_post,
                "rewrite": rewrite_blob,
                "possible_answer_docs": rewrite_docs,
                "leaf_descs": info.get("leaf_descs", []),
                "raw_output": raw_output,
            }
        )

    for sample_idx, sample in enumerate(tqdm(all_eval_samples, desc=f"Iter {iter_idx} mcts", leave=False)):
        info = rewrite_state_by_sample_idx[sample_idx]
        active_path_before = tuple(info.get("active_path_before", []))
        search_root_path = tuple(info.get("search_root_path", []))
        mcts_result = _run_local_mcts(
            sample=sample,
            search_root_path=search_root_path,
            query=str(info.get("query_post", sample.query)),
            stats_by_path=all_mcts_stats[sample_idx],
            retriever=retriever,
            node_embs=node_embs,
            node_registry=node_registry,
            leaf_indices_by_prefix=leaf_indices_by_prefix,
            leaf_indices=leaf_indices,
            leaf_ancestor_paths=leaf_ancestor_paths,
            reward_mode=str(hp.MCTS_REWARD_MODE or "mean_score"),
            rollout_topk=int(hp.MCTS_ROLLOUT_TOPK),
            num_simulations=int(hp.MCTS_NUM_SIMULATIONS),
            exploration_c=float(hp.MCTS_EXPLORATION_C),
            creation_step=iter_idx + 1,
        )
        active_path_after = tuple(mcts_result.get("chosen_path", search_root_path))
        sample.mcts_active_path = list(active_path_after)
        sample.beam_state_paths = [_get_prediction_path_nodes(sample, active_path_after)] if active_path_after else []
        sample.num_iters = iter_idx + 1
        info["mcts_search_root_path"] = [int(x) for x in search_root_path]
        info["mcts_selected_path"] = [int(x) for x in active_path_after]
        info["mcts_root_scored_top"] = list(mcts_result.get("root_scored_rows", []))
        info["mcts_top_children"] = list(mcts_result.get("top_children", []))
        info["mcts_root_candidate_count"] = int(mcts_result.get("root_candidate_count", 0))
        chosen_stats = mcts_result.get("chosen_stats")
        info["mcts_selected_visits"] = int(chosen_stats.visits) if chosen_stats is not None else 0
        info["mcts_selected_q"] = float(chosen_stats.q_value) if chosen_stats is not None else float("nan")

    branch_metric_df = _compute_branch_metrics_from_samples(all_eval_samples)
    iter_df = pd.DataFrame([rewrite_state_by_sample_idx[idx]["metrics"] for idx in range(len(all_eval_samples))])
    if (not branch_metric_df.empty) and (len(branch_metric_df) == len(iter_df)):
        iter_df = pd.concat([iter_df.reset_index(drop=True), branch_metric_df.reset_index(drop=True)], axis=1)

    for sample_idx, sample in enumerate(all_eval_samples):
        info = rewrite_state_by_sample_idx[sample_idx]
        row_metrics = iter_df.iloc[sample_idx].to_dict() if sample_idx < len(iter_df) else {}
        metrics = {str(k): float(v) for k, v in row_metrics.items() if np.isscalar(v)}
        active_before = [list(info.get("active_path_before", []))] if info.get("active_path_before", []) else []
        active_after = [list(info.get("mcts_selected_path", []))] if info.get("mcts_selected_path", []) else []
        sample.iter_records.append(
            {
                "iter": iter_idx,
                "query_pre": str(info.get("query_pre", "")),
                "query_post": str(info.get("query_post", "")),
                "rewrite": str(info.get("rewrite", "")),
                "rewrite_context_topk": int(hp.REWRITE_CONTEXT_TOPK),
                "rewrite_leaf_descs_count": int(len(info.get("leaf_descs", []))),
                "possible_answer_docs": info.get("rewrite_docs", {}),
                "selected_branches_before": active_before,
                "selected_branches_after": active_after,
                "mcts_controller": "uct",
                "mcts_num_simulations": int(hp.MCTS_NUM_SIMULATIONS),
                "mcts_exploration_c": float(hp.MCTS_EXPLORATION_C),
                "mcts_reward_mode": str(hp.MCTS_REWARD_MODE or "mean_score"),
                "mcts_rollout_topk": int(hp.MCTS_ROLLOUT_TOPK),
                "mcts_state_init": str(hp.MCTS_STATE_INIT or "root"),
                "mcts_terminal_backtrack": bool(info.get("backtracked", False)),
                "mcts_terminal_backtrack_from": [list(info.get("backtracked_from", []))] if info.get("backtracked_from", []) else [],
                "mcts_search_root_path": [list(info.get("mcts_search_root_path", []))] if info.get("mcts_search_root_path", []) else [],
                "mcts_selected_path": list(info.get("mcts_selected_path", [])),
                "mcts_selected_visits": int(info.get("mcts_selected_visits", 0)),
                "mcts_selected_q": float(info.get("mcts_selected_q", float("nan"))),
                "mcts_root_candidate_count": int(info.get("mcts_root_candidate_count", 0)),
                "mcts_root_scored_top": info.get("mcts_root_scored_top", []),
                "mcts_top_children": info.get("mcts_top_children", []),
                "pre_hit_paths": info.get("pre_hit_paths", []),
                "pre_hit_doc_ids": info.get("pre_hit_doc_ids", []),
                "local_paths": info.get("eval_paths", []),
                "local_doc_ids": info.get("eval_doc_ids", []),
                "active_eval_paths": info.get("eval_paths", []),
                "active_eval_doc_ids": info.get("eval_doc_ids", []),
                "gold_doc_ids": list(sample.gold_doc_ids),
                "cumulative_pool_eval_size": int(info.get("cumulative_pool_eval_size", 0)),
                "metrics": metrics,
            }
        )

    all_eval_metric_dfs.append(iter_df)
    if not iter_df.empty:
        logger.info(
            "Iter %d | nDCG@10=%.2f | Recall@100=%.2f | Coverage=%.2f | BranchHit@B=%.2f | BranchPrecision@B=%.2f",
            iter_idx,
            float(iter_df["nDCG@10"].mean()),
            float(iter_df["Recall@100"].mean()),
            float(iter_df["Coverage"].mean()),
            float(iter_df["BranchHit@B"].mean()),
            float(iter_df["BranchPrecision@B"].mean()),
        )

save_exp(
    RESULTS_DIR,
    hp,
    llm_api,
    all_eval_samples,
    all_eval_metric_dfs,
    allow_overwrite=True,
    save_llm_api_history=True,
)
logger.info("Saved MCTS results to %s", RESULTS_DIR)
