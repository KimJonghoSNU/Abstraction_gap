import asyncio
import hashlib
import json
import logging
import os
import pickle as pkl
import random
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
from emr_memory_utils import (
    EMR_CROSS_ENCODER_MODEL_NAME,
    EmrMemoryCompressor,
    RewritePromptTokenBudgeter,
    append_emr_history_entry as append_shared_emr_history_entry,
    build_current_doc_memory_items as build_shared_current_doc_memory_items,
    build_doc_text_map as build_shared_doc_text_map,
    build_emr_prompt_memory as build_shared_emr_prompt_memory,
    format_emr_history as format_shared_emr_history,
    update_accumulated_doc_bank as update_shared_accumulated_doc_bank,
)
from flat_then_tree import FlatHit
from hyperparams import (
    EMR_DEFAULT_COMPRESSION,
    EMR_DEFAULT_DOC_TOPK,
    EMR_DEFAULT_HISTORY_RANK_TOPK,
    EMR_DEFAULT_MEMORY_MAX_TOKENS,
    EMR_DEFAULT_SENT_TOPK,
    HyperParams,
)
from llm_apis import GenAIAPI, VllmAPI
from retrievers import build_retriever
from retrievers.diver import DiverEmbeddingModel
from rewrite_prompts import REWRITE_PROMPT_TEMPLATES
from tree_objects import InferSample, SemanticNode
from utils import (
    compute_ndcg,
    compute_node_registry,
    compute_recall,
    filter_excluded_predictions,
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
    carryover_descs: Sequence[str] = (),
    new_branch_descs: Sequence[str] = (),
    domain_route_hint: str = "",
    relevance_definition: str = "",
    seen_direction_evidence: str = "",
    history_actions: str = "",
    memory_docs: str = "",
) -> str:
    leaf_blob = "\n".join([x for x in leaf_descs if x])
    carryover_blob = "\n".join([x for x in carryover_descs if x]) or leaf_blob
    new_branch_blob = "\n".join([x for x in new_branch_descs if x])
    gate_blob = leaf_blob
    missing_seen_direction_placeholder = "{seen_direction_evidence}" not in template
    missing_carryover_placeholder = "{carryover_evidence}" not in template
    missing_new_branch_placeholder = "{new_branch_evidence}" not in template

    template = template.replace("Branch Context:\n{branch_descs}\n", "")
    template = template.replace("Retrieved Topic Cluster Summaries:\n{branch_descs}\n\n", "")
    template = template.replace("Retrieved Topic Cluster Summaries:\n{branch_descs}\n", "")
    template = template.replace("Topic Cluster Summaries:\n{branch_descs}\n", "")

    try:
        rendered = template.format(
            original_query=(original_query or ""),
            previous_rewrite=(previous_rewrite or ""),
            previous_docs="",
            leaf_descs=leaf_blob,
            branch_descs="",
            gate_descs=gate_blob,
            domain_route_hint=domain_route_hint or "",
            relevance_definition=relevance_definition or "",
            seen_direction_evidence=seen_direction_evidence or "",
            history_actions=history_actions or "",
            memory_docs=memory_docs or "",
            carryover_evidence=carryover_blob,
            new_branch_evidence=new_branch_blob,
            corpus_categories="",
        )
    except KeyError:
        rendered = (
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
            .replace("{history_actions}", history_actions or "")
            .replace("{memory_docs}", memory_docs or "")
            .replace("{carryover_evidence}", carryover_blob)
            .replace("{new_branch_evidence}", new_branch_blob)
            .replace("{corpus_categories}", "")
        )
    if seen_direction_evidence and missing_seen_direction_placeholder:
        # Intent: legacy prompt templates still need explicit local-failure guidance during reseat even if they lack a dedicated placeholder.
        rendered = rendered.rstrip() + f"\n\nAvoid Repeating These Weak Directions:\n{seen_direction_evidence}\n"
    if (carryover_descs or new_branch_descs) and (missing_carryover_placeholder or missing_new_branch_placeholder):
        # Intent: custom prompt variants without dedicated placeholders still need explicit old/new evidence separation for context-pack ablations.
        rendered = (
            rendered.rstrip()
            + f"\n\nCarryover Evidence:\n{carryover_blob or '(none)'}\n\n"
            + f"New Branch Evidence:\n{new_branch_blob or '(none)'}\n"
        )
    return rendered


def _compose_next_query(
    original_query: str,
    query_state_before: str,
    rewrite_blob: str,
    append_mode: bool,
) -> Tuple[str, str]:
    query_state_before_str = str(query_state_before or "").strip()
    rewrite = str(rewrite_blob or "").strip()
    if rewrite:
        if append_mode and query_state_before_str:
            # Intent: ended-beam transition steps should extend the previously useful retrieval direction instead of discarding it.
            query_state_after = f"{query_state_before_str} {rewrite}".strip()
        else:
            # Intent: ordinary steps replace the suffix state so stale retrieval directions do not accumulate indefinitely.
            query_state_after = rewrite
    else:
        query_state_after = query_state_before_str

    original_query_str = str(original_query or "").strip()
    if query_state_after:
        query_post = f"{original_query_str} {query_state_after}".strip()
    else:
        query_post = original_query_str
    return query_state_after, query_post


def _compose_query_from_state(original_query: str, query_state: str) -> str:
    query_state_str = str(query_state or "").strip()
    original_query_str = str(original_query or "").strip()
    if query_state_str:
        return f"{original_query_str} {query_state_str}".strip()
    return original_query_str


def _resolve_round6_rewrite_prompt_name(
    base_prompt_name: str,
    *,
    use_context_packing: bool,
) -> str:
    if use_context_packing and str(base_prompt_name or "").strip() == "agent_executor_v1_icl2":
        # Intent: the context-pack ablation should expose old/new evidence as separate sections instead of hiding them in one mixed block.
        return "agent_executor_v1_icl2_context_pack"
    return str(base_prompt_name or "").strip()


def _normalize_hypothesis_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _slot_sort_key(slot_name: str) -> Tuple[int, str]:
    slot = str(slot_name or "").strip()
    try:
        idx = CATEGORY_ORDER.index(slot)
    except ValueError:
        idx = len(CATEGORY_ORDER)
    return idx, slot.lower()


def _make_hypothesis_id(slot_name: str, text: str) -> str:
    payload = f"{str(slot_name or '').strip().lower()}||{_normalize_hypothesis_text(text).lower()}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:16]


def _extract_hypothesis_rows(
    rewrite_docs: Dict[str, str],
    rewrite_blob: str,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    docs = _clean_docs_map(rewrite_docs)
    if docs:
        for slot_name, text in sorted(docs.items(), key=lambda kv: _slot_sort_key(kv[0])):
            text_norm = _normalize_hypothesis_text(text)
            if not text_norm:
                continue
            rows.append({"slot": str(slot_name), "text": text_norm})
        return rows

    fallback_text = _normalize_hypothesis_text(rewrite_blob)
    if fallback_text:
        rows.append({"slot": "Other", "text": fallback_text})
    return rows


def _upsert_hypotheses(
    *,
    hypothesis_bank: Dict[str, Dict[str, Any]],
    next_order: int,
    rewrite_docs: Dict[str, str],
    rewrite_blob: str,
    iter_idx: int,
) -> Tuple[List[str], int]:
    hypothesis_ids: List[str] = []
    for row in _extract_hypothesis_rows(rewrite_docs, rewrite_blob):
        slot_name = str(row["slot"])
        text = str(row["text"])
        hypothesis_id = _make_hypothesis_id(slot_name, text)
        hypothesis_ids.append(hypothesis_id)
        hypothesis_entry = hypothesis_bank.get(hypothesis_id)
        if hypothesis_entry is None:
            # Intent: hypothesis ids must be stable across revisits so branch-local memory can reuse past evidence routes.
            hypothesis_bank[hypothesis_id] = {
                "id": hypothesis_id,
                "slot": slot_name,
                "text": text,
                "source_iter": int(iter_idx),
                "order": int(next_order),
                "last_support_score": float("nan"),
                "last_used_iter": -1,
                "status": "new",
            }
            next_order += 1
        else:
            hypothesis_entry["slot"] = slot_name
            hypothesis_entry["text"] = text
    return hypothesis_ids, next_order


def _collect_state_hypothesis_ids(
    state_ledger: Dict[Tuple[int, ...], Dict[str, Any]],
    state_paths: Sequence[Tuple[int, ...]],
    field_name: str,
) -> List[str]:
    collected: List[str] = []
    seen: Set[str] = set()
    for state_path in state_paths:
        entry = dict(state_ledger.get(tuple(state_path), {}) or {})
        for hypothesis_id in list(entry.get(field_name, []) or []):
            hid = str(hypothesis_id or "").strip()
            if (not hid) or (hid in seen):
                continue
            seen.add(hid)
            collected.append(hid)
    return collected


def _score_hypothesis_candidates(
    *,
    candidate_ids: Sequence[str],
    hypothesis_bank: Dict[str, Dict[str, Any]],
    retrieval_pool_indices: Sequence[int],
    retriever: DiverEmbeddingModel,
    node_embs: np.ndarray,
    node_registry: Sequence[object],
    path_to_doc_id: Dict[Tuple[int, ...], str],
    support_hits_topk: int,
) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    scores_by_id: Dict[str, float] = {}
    support_doc_ids_by_id: Dict[str, List[str]] = {}
    for hypothesis_id in candidate_ids:
        text = str(dict(hypothesis_bank.get(str(hypothesis_id), {}) or {}).get("text", "") or "").strip()
        if not text:
            scores_by_id[str(hypothesis_id)] = float("-inf")
            support_doc_ids_by_id[str(hypothesis_id)] = []
            continue
        hits = _retrieve_leaf_hits(
            query=text,
            leaf_pool_indices=retrieval_pool_indices,
            retriever=retriever,
            node_embs=node_embs,
            node_registry=node_registry,
            topk=max(1, int(support_hits_topk)),
        )
        if hits:
            scores_by_id[str(hypothesis_id)] = float(max(float(hit.score) for hit in hits))
            support_doc_ids_by_id[str(hypothesis_id)] = _paths_to_ranked_doc_ids(
                [tuple(hit.path) for hit in hits],
                path_to_doc_id,
            )
        else:
            scores_by_id[str(hypothesis_id)] = float("-inf")
            support_doc_ids_by_id[str(hypothesis_id)] = []
    return scores_by_id, support_doc_ids_by_id


def _select_supported_hypothesis_ids(
    *,
    candidate_ids: Sequence[str],
    hypothesis_bank: Dict[str, Dict[str, Any]],
    support_scores: Dict[str, float],
    topk: int,
    local_supported_ids: Optional[Set[str]] = None,
) -> List[str]:
    local_supported = {str(x) for x in list(local_supported_ids or set())}
    ranked = sorted(
        [str(hid) for hid in candidate_ids if str(hid) in hypothesis_bank],
        key=lambda hid: (
            float(support_scores.get(hid, float("-inf"))),
            1 if hid in local_supported else 0,
            -int(dict(hypothesis_bank.get(hid, {}) or {}).get("order", 10**9)),
        ),
        reverse=True,
    )
    selected: List[str] = []
    for hypothesis_id in ranked:
        score = float(support_scores.get(hypothesis_id, float("-inf")))
        if not np.isfinite(score):
            continue
        selected.append(hypothesis_id)
        if len(selected) >= max(1, int(topk)):
            break
    return selected


def _compose_hypothesis_query_state(
    active_hypothesis_ids: Sequence[str],
    hypothesis_bank: Dict[str, Dict[str, Any]],
) -> str:
    parts: List[str] = []
    seen_texts: Set[str] = set()
    for hypothesis_id in active_hypothesis_ids:
        text = _normalize_hypothesis_text(dict(hypothesis_bank.get(str(hypothesis_id), {}) or {}).get("text", ""))
        if (not text) or (text in seen_texts):
            continue
        seen_texts.add(text)
        parts.append(text)
    return " ".join(parts).strip()


def _format_failed_hypothesis_hint(
    hypothesis_ids: Sequence[str],
    hypothesis_bank: Dict[str, Dict[str, Any]],
    limit: int,
) -> str:
    rows: List[str] = []
    for hypothesis_id in list(hypothesis_ids or [])[: max(0, int(limit))]:
        entry = dict(hypothesis_bank.get(str(hypothesis_id), {}) or {})
        text = _normalize_hypothesis_text(entry.get("text", ""))
        if not text:
            continue
        slot_name = str(entry.get("slot", "Other") or "Other")
        rows.append(f"- {slot_name}: {text}")
    return "\n".join(rows).strip()


def _update_state_hypothesis_ledger(
    *,
    state_ledger: Dict[Tuple[int, ...], Dict[str, Any]],
    state_paths: Sequence[Tuple[int, ...]],
    used_ids: Sequence[str],
    supported_ids: Sequence[str],
    failed_ids: Sequence[str],
    iter_idx: int,
) -> None:
    for state_path in state_paths:
        state_key = tuple(state_path)
        entry = state_ledger.setdefault(
            state_key,
            {
                "used_ids": [],
                "supported_ids": [],
                "failed_ids": [],
                "last_iter": -1,
            },
        )
        for field_name, new_ids in (
            ("used_ids", used_ids),
            ("supported_ids", supported_ids),
            ("failed_ids", failed_ids),
        ):
            seen = {str(x) for x in list(entry.get(field_name, []) or [])}
            merged = list(entry.get(field_name, []) or [])
            for hypothesis_id in new_ids:
                hid = str(hypothesis_id or "").strip()
                if (not hid) or (hid in seen):
                    continue
                seen.add(hid)
                merged.append(hid)
            entry[field_name] = merged
        entry["last_iter"] = int(iter_idx)


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


def _dedupe_hits_in_order(hits: Sequence[FlatHit]) -> List[FlatHit]:
    deduped: List[FlatHit] = []
    seen_registry: Set[int] = set()
    for hit in hits:
        registry_idx = int(hit.registry_idx)
        if registry_idx in seen_registry:
            continue
        seen_registry.add(registry_idx)
        deduped.append(hit)
    return deduped


def _take_hits_with_extras(
    hits: Sequence[FlatHit],
    topk: int,
) -> Tuple[List[FlatHit], List[FlatHit]]:
    ordered = _dedupe_hits_in_order(hits)
    if topk <= 0:
        return [], ordered
    return ordered[:topk], ordered[topk:]


def _path_to_debug_key(path: Sequence[int]) -> str:
    return "/".join(str(int(x)) for x in list(path))


def _hits_to_paths(hits: Sequence[FlatHit]) -> List[List[int]]:
    return [list(tuple(hit.path)) for hit in hits]


def _hits_to_ranked_doc_ids(
    hits: Sequence[FlatHit],
    path_to_doc_id: Dict[Tuple[int, ...], str],
) -> List[str]:
    return _paths_to_ranked_doc_ids([tuple(hit.path) for hit in hits], path_to_doc_id)


def _resolve_new_branch_frontier_child(
    leaf_path: Tuple[int, ...],
    new_branch_paths: Sequence[Tuple[int, ...]],
) -> Tuple[int, ...]:
    matched_branch_paths = [
        tuple(branch_path)
        for branch_path in list(new_branch_paths or [])
        if branch_path and _leaf_path_is_under_prefix(tuple(leaf_path), tuple(branch_path))
    ]
    if not matched_branch_paths:
        return tuple(leaf_path)
    anchor_path = max(matched_branch_paths, key=len)
    child_depth = min(len(tuple(leaf_path)), len(anchor_path) + 1)
    return tuple(leaf_path[:child_depth])


def _count_hits_by_frontier_child(
    hits: Sequence[FlatHit],
    new_branch_paths: Sequence[Tuple[int, ...]],
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for hit in _dedupe_hits_in_order(hits):
        group_key = _path_to_debug_key(_resolve_new_branch_frontier_child(tuple(hit.path), new_branch_paths))
        counts[group_key] = int(counts.get(group_key, 0)) + 1
    return counts


def _select_diverse_new_branch_hits(
    hits: Sequence[FlatHit],
    new_branch_paths: Sequence[Tuple[int, ...]],
    topk: int,
) -> Tuple[List[FlatHit], List[FlatHit], Dict[str, int], Dict[str, int]]:
    ordered_hits = _dedupe_hits_in_order(hits)
    if topk <= 0 or not ordered_hits:
        return [], ordered_hits, _count_hits_by_frontier_child(ordered_hits, new_branch_paths), {}

    grouped_hits: Dict[Tuple[int, ...], List[FlatHit]] = {}
    for hit in ordered_hits:
        group_path = _resolve_new_branch_frontier_child(tuple(hit.path), new_branch_paths)
        grouped_hits.setdefault(group_path, []).append(hit)

    ordered_groups = sorted(
        grouped_hits.keys(),
        key=lambda group_path: (
            -float(grouped_hits[group_path][0].score),
            len(group_path),
            tuple(group_path),
        ),
    )
    selected: List[FlatHit] = []
    selected_registry: Set[int] = set()

    # Intent: expose one high-scoring hit per frontier child first so a deep single branch family cannot monopolize the new-branch context.
    for group_path in ordered_groups:
        hit = grouped_hits[group_path][0]
        registry_idx = int(hit.registry_idx)
        if registry_idx in selected_registry:
            continue
        selected_registry.add(registry_idx)
        selected.append(hit)
        if len(selected) >= topk:
            break

    if len(selected) < topk:
        leftover_hits: List[FlatHit] = []
        for group_path in ordered_groups:
            leftover_hits.extend(grouped_hits[group_path][1:])
        leftover_hits = sorted(
            leftover_hits,
            key=lambda hit: (
                -float(hit.score),
                len(tuple(hit.path)),
                tuple(hit.path),
            ),
        )
        for hit in leftover_hits:
            registry_idx = int(hit.registry_idx)
            if registry_idx in selected_registry:
                continue
            selected_registry.add(registry_idx)
            selected.append(hit)
            if len(selected) >= topk:
                break

    unused_hits = [
        hit
        for hit in ordered_hits
        if int(hit.registry_idx) not in selected_registry
    ]
    unused_hits = sorted(
        unused_hits,
        key=lambda hit: (
            -float(hit.score),
            len(tuple(hit.path)),
            tuple(hit.path),
        ),
    )
    return (
        selected,
        unused_hits,
        _count_hits_by_frontier_child(ordered_hits, new_branch_paths),
        _count_hits_by_frontier_child(selected, new_branch_paths),
    )


def _interleave_source_hits(
    new_branch_hits: Sequence[FlatHit],
    carryover_hits: Sequence[FlatHit],
) -> Tuple[List[FlatHit], List[str]]:
    ordered_hits: List[FlatHit] = []
    ordered_sources: List[str] = []
    new_branch_list = list(new_branch_hits)
    carryover_list = list(carryover_hits)
    while new_branch_list or carryover_list:
        if new_branch_list:
            ordered_hits.append(new_branch_list.pop(0))
            ordered_sources.append("new_branch")
        if carryover_list:
            ordered_hits.append(carryover_list.pop(0))
            ordered_sources.append("carryover")
    return ordered_hits, ordered_sources


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
    excluded_doc_ids: Optional[Sequence[str]],
    path_to_doc_id: Dict[Tuple[int, ...], str],
    iter_idx: int,
    explore_step: bool,
    query_post: str,
) -> Dict[str, Any]:
    registry_indices = np.asarray([int(hit.registry_idx) for hit in hits], dtype=np.int32)
    scores = np.asarray([float(hit.score) for hit in hits], dtype=np.float32)
    doc_ids = _paths_to_ranked_doc_ids([tuple(hit.path) for hit in hits], path_to_doc_id)
    eval_doc_ids = filter_excluded_predictions(doc_ids, excluded_doc_ids)
    run_ndcg10 = compute_ndcg(eval_doc_ids, list(gold_doc_ids), k=10) * 100.0
    return {
        "iter": int(iter_idx),
        "explore_step": bool(explore_step),
        "query_post": str(query_post or ""),
        "registry_indices": registry_indices,
        "scores": scores,
        "ndcg10": float(run_ndcg10),
        "top_doc_ids": list(eval_doc_ids[:10]),
    }


def _compute_retrieval_metrics(
    doc_ids: Sequence[str],
    gold_doc_ids: Sequence[str],
    excluded_doc_ids: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    # Intent: BRIGHT excluded_ids should be removed before top-k truncation so official scores match benchmark evaluation.
    ranked_doc_ids = filter_excluded_predictions(list(doc_ids), excluded_doc_ids)
    gold_ranked_doc_ids = list(gold_doc_ids)
    return {
        "nDCG@10": compute_ndcg(ranked_doc_ids, gold_ranked_doc_ids, k=10) * 100,
        "Recall@10": compute_recall(ranked_doc_ids, gold_ranked_doc_ids, k=10) * 100,
        "Recall@100": compute_recall(ranked_doc_ids, gold_ranked_doc_ids, k=100) * 100,
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


def _build_union_leaf_pool(
    *,
    base_branch_paths: Sequence[Tuple[int, ...]],
    cumulative_leaf_indices: Sequence[int],
    leaf_indices_by_prefix: Dict[Tuple[int, ...], List[int]],
    all_leaf_indices: Sequence[int],
    fallback_to_all: bool = True,
) -> Tuple[List[int], int, int]:
    branch_component = _collect_leaf_pool(
        selected_branches=base_branch_paths,
        leaf_indices_by_prefix=leaf_indices_by_prefix,
        all_leaf_indices=all_leaf_indices,
        fallback_to_all=False,
    )
    cumulative_component = [int(x) for x in list(cumulative_leaf_indices or [])]
    pool: List[int] = []
    seen: Set[int] = set()
    for idx in branch_component:
        if idx in seen:
            continue
        seen.add(idx)
        pool.append(int(idx))
    for idx in cumulative_component:
        if idx in seen:
            continue
        seen.add(idx)
        pool.append(int(idx))
    if not pool and fallback_to_all:
        # Intent: if both live-frontier descendants and cumulative reached leaves are empty, keep round6 recoverable with a full-leaf fallback.
        pool = [int(x) for x in list(all_leaf_indices)]
    return pool, int(len(branch_component)), int(len(cumulative_component))


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


def _path_sort_key(path: Tuple[int, ...]) -> Tuple[int, Tuple[int, ...]]:
    return (len(tuple(path)), tuple(path))


def _filter_scored_rows_to_allowed_paths(
    scored_rows: Sequence[Dict[str, Any]],
    allowed_paths: Sequence[Tuple[int, ...]],
) -> List[Dict[str, Any]]:
    allowed_set = {tuple(path) for path in allowed_paths if path}
    filtered: List[Dict[str, Any]] = []
    seen: Set[Tuple[int, ...]] = set()
    for row in scored_rows:
        path_t = _row_path_tuple(row)
        if (not path_t) or (path_t not in allowed_set) or (path_t in seen):
            continue
        seen.add(path_t)
        filtered.append(row)
    return filtered


def _collect_strict_descendant_branch_paths(
    *,
    selected_branches: Sequence[Tuple[int, ...]],
    child_branch_paths_by_path: Dict[Tuple[int, ...], List[Tuple[int, ...]]],
    all_nonroot_nonleaf_paths: Sequence[Tuple[int, ...]],
    root_path: Tuple[int, ...],
) -> Tuple[List[Tuple[int, ...]], Dict[Tuple[int, ...], List[Tuple[int, ...]]]]:
    selected_set = {tuple(path) for path in selected_branches if path}
    if not selected_set:
        return list(all_nonroot_nonleaf_paths), {
            tuple(path): [tuple(root_path)] for path in all_nonroot_nonleaf_paths
        }

    parent_map: Dict[Tuple[int, ...], Set[Tuple[int, ...]]] = defaultdict(set)
    seen: Set[Tuple[int, ...]] = set()
    for source_path in sorted(selected_set, key=_path_sort_key):
        stack: List[Tuple[int, ...]] = [tuple(path) for path in child_branch_paths_by_path.get(source_path, [])]
        local_seen: Set[Tuple[int, ...]] = set()
        while stack:
            cur_path = tuple(stack.pop())
            if cur_path in local_seen:
                continue
            local_seen.add(cur_path)
            if cur_path not in selected_set:
                seen.add(cur_path)
                parent_map[cur_path].add(source_path)
            for child_path in child_branch_paths_by_path.get(cur_path, []):
                stack.append(tuple(child_path))

    ordered_paths = sorted(seen, key=_path_sort_key)
    return ordered_paths, {
        tuple(path): sorted({tuple(parent) for parent in parents}, key=_path_sort_key)
        for path, parents in parent_map.items()
    }


def _collect_goexplore_direct_child_candidates(
    *,
    selected_branches: Sequence[Tuple[int, ...]],
    child_branch_paths_by_path: Dict[Tuple[int, ...], List[Tuple[int, ...]]],
    root_branch_children: Sequence[Tuple[int, ...]],
    root_path: Tuple[int, ...],
) -> Tuple[List[Tuple[int, ...]], Dict[Tuple[int, ...], List[Tuple[int, ...]]]]:
    selected_set = {tuple(path) for path in selected_branches if path}
    if not selected_set:
        root_children = [tuple(path) for path in root_branch_children]
        return root_children, {
            tuple(path): [tuple(root_path)] for path in root_children
        }

    parent_map: Dict[Tuple[int, ...], Set[Tuple[int, ...]]] = defaultdict(set)
    decision_points: Set[Tuple[int, ...]] = {tuple(root_path)}
    exploited_child_by_parent: Dict[Tuple[int, ...], Set[Tuple[int, ...]]] = defaultdict(set)

    for branch_path in sorted(selected_set, key=_path_sort_key):
        for depth in range(len(branch_path)):
            parent_path = tuple(branch_path[:depth])
            child_path = tuple(branch_path[: depth + 1])
            exploited_child_by_parent[parent_path].add(child_path)
            if depth > 0:
                decision_points.add(parent_path)
        decision_points.add(tuple(branch_path))

    candidate_paths: Set[Tuple[int, ...]] = set()
    for parent_path in sorted(decision_points, key=_path_sort_key):
        child_paths = (
            [tuple(path) for path in root_branch_children]
            if parent_path == tuple(root_path)
            else [tuple(path) for path in child_branch_paths_by_path.get(parent_path, [])]
        )
        for child_path in child_paths:
            if child_path in selected_set:
                continue
            # Intent: go-explore replacement reopens missed direct children instead of teleporting arbitrarily.
            if child_path in exploited_child_by_parent.get(parent_path, set()):
                continue
            candidate_paths.add(child_path)
            parent_map[child_path].add(parent_path)

    ordered_paths = sorted(candidate_paths, key=_path_sort_key)
    return ordered_paths, {
        tuple(path): sorted({tuple(parent) for parent in parents}, key=_path_sort_key)
        for path, parents in parent_map.items()
    }


def _state_paths_to_endpoint_paths(state_paths: Sequence[Sequence[object]]) -> List[Tuple[int, ...]]:
    endpoints: List[Tuple[int, ...]] = []
    seen: Set[Tuple[int, ...]] = set()
    for state_path in state_paths:
        if not state_path:
            continue
        endpoint = tuple(getattr(state_path[-1], "path", ()) or ())
        if (not endpoint) or (endpoint in seen):
            continue
        seen.add(endpoint)
        endpoints.append(endpoint)
    return endpoints


def _materialize_beam_from_paths(
    *,
    sample: InferSample,
    beam_paths: Sequence[Tuple[int, ...]],
    path_scores: Dict[Tuple[int, ...], float],
) -> None:
    ordered_paths: List[Tuple[int, ...]] = []
    seen: Set[Tuple[int, ...]] = set()
    gate_scores: Dict[Tuple[int, ...], float] = {}
    for path in beam_paths:
        path_t = tuple(path)
        if (not path_t) or (path_t in seen):
            continue
        seen.add(path_t)
        ordered_paths.append(path_t)
        gate_scores[path_t] = float(path_scores.get(path_t, 1.0))
    if ordered_paths:
        # Intent: convert retrieval-selected deeper branches back into actual beam states for the next traversal step.
        sample.seed_beam_from_gate_paths(ordered_paths, gate_scores=gate_scores, reset_history=False)


def _deterministic_random_rows(
    *,
    candidate_paths: Sequence[Tuple[int, ...]],
    sample_count: int,
    seed_key: str,
) -> List[Dict[str, Any]]:
    unique_paths = sorted(
        {
            tuple(path_t)
            for path_t in candidate_paths
            if tuple(path_t)
        },
        key=_path_sort_key,
    )
    if not unique_paths or sample_count <= 0:
        return []
    seed_int = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed_int)
    chosen_paths = (
        unique_paths
        if sample_count >= len(unique_paths)
        else rng.sample(unique_paths, sample_count)
    )
    # Intent: keep the ended-reseat candidate pool fixed and vary only the replacement policy.
    return [
        {
            "path": tuple(path_t),
            "score_mode": "random",
            "score": 0.0,
            "max_score": 0.0,
            "mean_score": 0.0,
            "matched_count": 0,
            "best_rank": None,
            "best_leaf_path": [],
        }
        for path_t in chosen_paths
    ]


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
    startup_warnings.append(f'Ignoring --round5_mode "{requested_round5_mode}" in run_round6_hypbank.py (forced to legacy).')
if any(arg.startswith("--round5_category_") for arg in sys.argv[1:]):
    # Intent: make ignored category controls explicit to avoid silent config drift.
    startup_warnings.append("Ignoring --round5_category_* arguments in run_round6_hypbank.py (legacy-only runtime).")
round5_mode = "legacy"

if hp.REWRITE_PROMPT_PATH:
    print("Ignoring --rewrite_prompt_path in run_round6_hypbank.py (template path override is disabled).")

round6_emr_memory = bool(getattr(hp, "ROUND6_EMR_MEMORY", False))
round6_hypbank = bool(getattr(hp, "ROUND6_HYPBANK", False))
round6_context_packing = bool(getattr(hp, "ROUND6_CONTEXT_PACKING", False))
round6_hypbank_context_packing = bool(getattr(hp, "ROUND6_HYPBANK_CONTEXT_PACKING", False))
effective_round6_context_packing = bool(round6_context_packing or round6_hypbank_context_packing)
round5_rewrite_prompt_name = "agent_executor_v1"
requested_prompt_name = str(hp.REWRITE_PROMPT_NAME or "").strip()
if requested_prompt_name:
    round5_rewrite_prompt_name = requested_prompt_name
# Intent: keep the default round6 expandable prompt stable while auto-switching to the memory-aware variant only for EMR runs.
if round6_emr_memory and round5_rewrite_prompt_name == "agent_executor_v1_icl2":
    round5_rewrite_prompt_name = "agent_executor_v1_icl2_emr_memory"
# Intent: legacy mode accepts any registered rewrite template so prompt ablations can reuse --rewrite_prompt_name.
if round5_rewrite_prompt_name not in REWRITE_PROMPT_TEMPLATES:
    raise ValueError(
        f'Unknown --rewrite_prompt_name "{round5_rewrite_prompt_name}" in run_round6_hypbank.py legacy mode. '
        f"Available: {sorted(REWRITE_PROMPT_TEMPLATES.keys())}"
    )

hp.add_param("rewrite_prompt_name", round5_rewrite_prompt_name)

if str(hp.ROUND3_SUMMARIZED_CONTEXT or "off").lower() != "off":
    print("Ignoring --round3_summarized_context in run_round6_hypbank.py (fixed to off).")
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
round6_expandable_candidate_mode = str(
    getattr(hp, "ROUND6_EXPANDABLE_CANDIDATE_MODE", "direct_children") or "direct_children"
).strip().lower()
if round6_expandable_candidate_mode not in {"direct_children", "descendant_flat"}:
    raise ValueError(
        f'Unsupported --round6_expandable_candidate_mode "{round6_expandable_candidate_mode}". '
        "Allowed: direct_children|descendant_flat"
    )
round6_expandable_ended_scope = str(
    getattr(hp, "ROUND6_EXPANDABLE_ENDED_SCOPE", "leftover_expandable") or "leftover_expandable"
).strip().lower()
if round6_expandable_ended_scope not in {"leftover_expandable", "whole_tree_flat", "goexplore_direct_child"}:
    raise ValueError(
        f'Unsupported --round6_expandable_ended_scope "{round6_expandable_ended_scope}". '
        "Allowed: leftover_expandable|whole_tree_flat|goexplore_direct_child"
    )
round6_expandable_reseat_policy = str(
    getattr(hp, "ROUND6_EXPANDABLE_RESEAT_POLICY", "score") or "score"
).strip().lower()
if round6_expandable_reseat_policy not in {"score", "random"}:
    raise ValueError(
        f'Unsupported --round6_expandable_reseat_policy "{round6_expandable_reseat_policy}". '
        "Allowed: score|random"
    )
if round6_global_escape and round5_selector_mode == "retriever_slate":
    raise ValueError(
        "--round6_global_escape requires a score-based --round5_selector_mode "
        "(maxscore_global|meanscore_global|max_hit_global)."
    )
if round6_expandable_mode != "off" and round5_selector_mode == "retriever_slate":
    raise ValueError(
        "--round6_expandable_mode requires a score-based --round5_selector_mode "
        "(maxscore_global|meanscore_global|max_hit_global)."
    )
if round6_expandable_mode != "off" and round6_global_escape:
    raise ValueError("--round6_expandable_mode and --round6_global_escape are mutually exclusive in run_round6_hypbank.py.")
if round6_hypbank and round6_expandable_mode != "ended_reseat":
    raise ValueError("--round6_hypbank requires --round6_expandable_mode=ended_reseat in run_round6_hypbank.py.")
hp.add_param("round6_global_escape", round6_global_escape)
hp.add_param("round6_global_escape_slots", round6_global_escape_slots)
hp.add_param("round6_expandable_mode", round6_expandable_mode)
hp.add_param("round6_expandable_candidate_mode", round6_expandable_candidate_mode)
hp.add_param("round6_expandable_ended_scope", round6_expandable_ended_scope)
hp.add_param("round6_expandable_reseat_policy", round6_expandable_reseat_policy)
# Intent: behavior-level result naming must change when round6 retrieval/query-state semantics change so old runs are never overwritten.
hp.add_param("round6_hypbank", round6_hypbank)
hp.add_param("round6_context_packing", effective_round6_context_packing)
hp.add_param(
    "round6_behavior",
    (
        "hypbank_reseat_ctxpack_v1"
        if round6_hypbank and effective_round6_context_packing
        else (
            "hypbank_reseat_v1"
            if round6_hypbank
            else ("frontiercum_ctxpack_v1" if effective_round6_context_packing else "frontiercum_qstate_v1")
        )
    ),
)
round6_emr_topk = max(1, int(getattr(hp, "ROUND6_EMR_TOPK", EMR_DEFAULT_DOC_TOPK) or EMR_DEFAULT_DOC_TOPK))
round6_emr_sent_topk = max(1, int(getattr(hp, "ROUND6_EMR_SENT_TOPK", EMR_DEFAULT_SENT_TOPK) or EMR_DEFAULT_SENT_TOPK))
round6_emr_compression = str(getattr(hp, "ROUND6_EMR_COMPRESSION", EMR_DEFAULT_COMPRESSION) or EMR_DEFAULT_COMPRESSION).strip().lower()
if round6_emr_compression not in {"on", "off"}:
    raise ValueError(f'Unsupported --round6_emr_compression "{round6_emr_compression}". Allowed: on|off')
round6_emr_memory_max_tokens = int(getattr(hp, "ROUND6_EMR_MEMORY_MAX_TOKENS", EMR_DEFAULT_MEMORY_MAX_TOKENS) or EMR_DEFAULT_MEMORY_MAX_TOKENS)
hp.add_param("round6_emr_memory", round6_emr_memory)
hp.add_param("round6_emr_topk", round6_emr_topk)
hp.add_param("round6_emr_sent_topk", round6_emr_sent_topk)
hp.add_param("round6_emr_compression", round6_emr_compression)
hp.add_param("round6_emr_memory_max_tokens", round6_emr_memory_max_tokens)

ROUND6_HYPBANK_ACTIVE_TOPK = 3
ROUND6_HYPBANK_SUPPORT_HITS_TOPK = 5
ROUND6_HYPBANK_FAILED_HINT_TOPK = 3

exp_dir_name = str(hp)
RESULTS_DIR = f"{BASE_DIR}/results/{hp.DATASET}/{hp.SUBSET}/round6/{exp_dir_name}/"
if os.path.exists(RESULTS_DIR):
    # Intent: allow reruns after partial writes like logs or hparams; skip only when result pickles already exist.
    has_result_pkl = any(
        entry.endswith(".pkl") and os.path.isfile(os.path.join(RESULTS_DIR, entry))
        for entry in os.listdir(RESULTS_DIR)
    )
    if has_result_pkl:
        print(f"Results already exist at {RESULTS_DIR}. Skipping run.")
        raise SystemExit(0)
os.makedirs(RESULTS_DIR, exist_ok=True)

logger = setup_logger("lattice_runner_round6_hypbank", f"{RESULTS_DIR}/run.log", logging.INFO)
with open(f"{RESULTS_DIR}/hparams.json", "w", encoding="utf-8") as f:
    json.dump(vars(hp), f, indent=2, ensure_ascii=True, sort_keys=True)
for warning in startup_warnings:
    logger.warning(warning)
logger.info(
    "Round6 hypbank config | mode=%s | rewrite_prompt=%s | selector_mode=%s | selector_topk=%d | global_escape=%s | escape_slots=%d | expandable_mode=%s | expandable_candidate_mode=%s | expandable_ended_scope=%s | expandable_reseat_policy=%s | hypbank=%s | ctxpack=%s",
    "legacy",
    round5_rewrite_prompt_name,
    round5_selector_mode,
    int(round5_mrr_pool_k),
    str(round6_global_escape).lower(),
    int(round6_global_escape_slots),
    round6_expandable_mode,
    round6_expandable_candidate_mode,
    round6_expandable_ended_scope,
    round6_expandable_reseat_policy,
    str(round6_hypbank).lower(),
    str(effective_round6_context_packing).lower(),
)
logger.info(
    "Round6 EMR memory | enabled=%s | topk=%d | sent_topk=%d | compression=%s | max_memory_tokens=%d",
    str(round6_emr_memory).lower(),
    int(round6_emr_topk),
    int(round6_emr_sent_topk),
    round6_emr_compression,
    int(round6_emr_memory_max_tokens),
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

if using_dag_runtime and (
    round6_expandable_candidate_mode == "descendant_flat"
    or round6_expandable_ended_scope == "goexplore_direct_child"
):
    raise ValueError(
        "DAG runtime does not support descendant-flat/goexplore ended-reseat path semantics in run_round6.py."
    )

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
all_nonroot_nonleaf_paths = sorted(
    {
        tuple(node.path)
        for node in node_registry
        if (not node.is_leaf) and tuple(node.path) != root_path
    },
    key=_path_sort_key,
)

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
doc_text_by_id = build_shared_doc_text_map(docs_df)

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

round6_emr_compressor: Optional[EmrMemoryCompressor] = None
round6_rewrite_prompt_budgeter: Optional[RewritePromptTokenBudgeter] = None
if round6_emr_memory:
    round6_emr_compressor = EmrMemoryCompressor(
        model_name=EMR_CROSS_ENCODER_MODEL_NAME,
        sent_topk=round6_emr_sent_topk,
        logger=logger,
    )
    round6_rewrite_prompt_budgeter = RewritePromptTokenBudgeter(
        model_name=str(hp.LLM),
        backend=str(hp.LLM_API_BACKEND),
        logger=logger,
    )

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
    sample.round6_query_state = ""
    sample.rewrite_history = []
    sample.iter_records = []
    sample.round6_emr_history = []
    sample.round6_emr_doc_bank = {}
    sample.round6_emr_doc_bank_next_order = 0
    sample.round6_hypothesis_bank = {}
    sample.round6_hypothesis_next_order = 0
    sample.round6_state_hypothesis_ledger = {}
    sample.round6_newly_added_branch_paths_after_iter = []
    sample.round6_newly_added_branch_descendant_leaf_pool = []
    if "original_query" not in sample.SAVE_LIST:
        sample.SAVE_LIST.append("original_query")
    if "gold_doc_ids" not in sample.SAVE_LIST:
        sample.SAVE_LIST.append("gold_doc_ids")
    if "last_rewrite_raw" not in sample.SAVE_LIST:
        sample.SAVE_LIST.append("last_rewrite_raw")
    if "round6_query_state" not in sample.SAVE_LIST:
        sample.SAVE_LIST.append("round6_query_state")
    if "iter_records" not in sample.SAVE_LIST:
        sample.SAVE_LIST.append("iter_records")
    if "round6_hypothesis_bank" not in sample.SAVE_LIST:
        sample.SAVE_LIST.append("round6_hypothesis_bank")
    if "round6_hypothesis_next_order" not in sample.SAVE_LIST:
        sample.SAVE_LIST.append("round6_hypothesis_next_order")
    if "round6_state_hypothesis_ledger" not in sample.SAVE_LIST:
        sample.SAVE_LIST.append("round6_state_hypothesis_ledger")
    if "round6_newly_added_branch_paths_after_iter" not in sample.SAVE_LIST:
        sample.SAVE_LIST.append("round6_newly_added_branch_paths_after_iter")
    if "round6_newly_added_branch_descendant_leaf_pool" not in sample.SAVE_LIST:
        sample.SAVE_LIST.append("round6_newly_added_branch_descendant_leaf_pool")
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

    for sample_idx, sample in enumerate(tqdm(all_eval_samples, desc=f"Iter {iter_idx} rewrite prep", leave=False)):
        selected_before_current = _selected_branch_paths_from_sample(sample)
        _, ended_selected_before_current = _partition_selected_branches_by_child_candidates(
            selected_branches=selected_before_current,
            child_branch_paths_by_path=child_branch_paths_by_path,
        )
        ended_beam_transition_step = bool(
            round6_expandable_mode == "ended_reseat" and len(ended_selected_before_current) > 0
        )
        query_state_before_raw = str(getattr(sample, "round6_query_state", "") or "").strip()
        query_state_before_effective = query_state_before_raw
        cumulative_pool = sorted(cumulative_leaf_indices_by_sample[sample_idx])
        retrieval_pool_indices, retrieval_pool_branch_component_size, retrieval_pool_cumulative_component_size = _build_union_leaf_pool(
            base_branch_paths=selected_before_current,
            cumulative_leaf_indices=cumulative_pool,
            leaf_indices_by_prefix=leaf_indices_by_prefix,
            all_leaf_indices=leaf_indices,
        )
        retrieval_pool_source = "frontier_descendants_plus_cumulative"
        candidate_hypothesis_ids: List[str] = []
        active_hypothesis_ids_before: List[str] = []
        dropped_hypothesis_ids_before: List[str] = []
        local_supported_hypothesis_ids = _collect_state_hypothesis_ids(
            getattr(sample, "round6_state_hypothesis_ledger", {}),
            selected_before_current,
            "supported_ids",
        )
        local_failed_hypothesis_ids = _collect_state_hypothesis_ids(
            getattr(sample, "round6_state_hypothesis_ledger", {}),
            selected_before_current,
            "failed_ids",
        )
        local_failed_hypothesis_hint = ""
        hypothesis_support_scores_before: Dict[str, float] = {}
        hypothesis_support_doc_ids_before: Dict[str, List[str]] = {}

        if round6_hypbank and ended_beam_transition_step:
            hypothesis_bank = dict(getattr(sample, "round6_hypothesis_bank", {}) or {})
            global_candidate_ids = [str(hypothesis_id) for hypothesis_id in hypothesis_bank.keys()]
            nonfailed_candidate_ids = [
                hypothesis_id
                for hypothesis_id in global_candidate_ids
                if hypothesis_id not in set(local_failed_hypothesis_ids)
            ]
            candidate_hypothesis_ids = list(nonfailed_candidate_ids or global_candidate_ids)
            hypothesis_support_scores_before, hypothesis_support_doc_ids_before = _score_hypothesis_candidates(
                candidate_ids=candidate_hypothesis_ids,
                hypothesis_bank=hypothesis_bank,
                retrieval_pool_indices=retrieval_pool_indices,
                retriever=retriever,
                node_embs=node_embs,
                node_registry=node_registry,
                path_to_doc_id=path_to_doc_id,
                support_hits_topk=ROUND6_HYPBANK_SUPPORT_HITS_TOPK,
            )
            active_hypothesis_ids_before = _select_supported_hypothesis_ids(
                candidate_ids=candidate_hypothesis_ids,
                hypothesis_bank=hypothesis_bank,
                support_scores=hypothesis_support_scores_before,
                topk=ROUND6_HYPBANK_ACTIVE_TOPK,
                local_supported_ids=set(local_supported_hypothesis_ids),
            )
            if active_hypothesis_ids_before:
                # Intent: reseat should preserve only retrieval-supported abstractions instead of blindly carrying the full stale suffix.
                query_state_before_effective = _compose_hypothesis_query_state(
                    active_hypothesis_ids_before,
                    hypothesis_bank,
                )
            dropped_hypothesis_ids_before = [
                hypothesis_id
                for hypothesis_id in candidate_hypothesis_ids
                if hypothesis_id not in set(active_hypothesis_ids_before)
            ]
            local_failed_hypothesis_hint = _format_failed_hypothesis_hint(
                local_failed_hypothesis_ids,
                hypothesis_bank,
                ROUND6_HYPBANK_FAILED_HINT_TOPK,
            )

        query_pre = _compose_query_from_state(str(sample.original_query), query_state_before_effective)
        prompt_name_for_iter = round5_rewrite_prompt_name
        previous_newly_added_branch_paths = [
            tuple(path)
            for path in list(getattr(sample, "round6_newly_added_branch_paths_after_iter", []) or [])
            if path
        ]
        previous_newly_added_branch_leaf_pool = [
            int(idx)
            for idx in list(getattr(sample, "round6_newly_added_branch_descendant_leaf_pool", []) or [])
        ]
        source_balanced_rewrite_context = False
        rewrite_context_total_topk = max(1, int(hp.REWRITE_CONTEXT_TOPK))
        carryover_context_target = int(rewrite_context_total_topk // 2)
        # Intent: when the rewrite budget is odd, give the extra slot to the new-branch half to counter old-memory bias after reseat.
        new_branch_context_target = int(rewrite_context_total_topk - carryover_context_target)
        carryover_source_candidate_hits: List[FlatHit] = []
        new_branch_source_candidate_hits: List[FlatHit] = []
        carryover_source_selected_hits: List[FlatHit] = []
        new_branch_source_selected_hits: List[FlatHit] = []
        selected_context_hits: List[FlatHit] = []
        selected_context_source_sequence: List[str] = []
        new_branch_group_candidate_counts: Dict[str, int] = {}
        new_branch_group_selected_counts: Dict[str, int] = {}
        carryover_backfill_count = 0
        new_branch_backfill_count = 0

        if effective_round6_context_packing and previous_newly_added_branch_paths and rewrite_context_total_topk > 1:
            retrieval_pool_set = {int(idx) for idx in retrieval_pool_indices}
            previous_newly_added_branch_leaf_pool = [
                int(idx)
                for idx in previous_newly_added_branch_leaf_pool
                if int(idx) in retrieval_pool_set
            ]
            if previous_newly_added_branch_leaf_pool:
                source_balanced_rewrite_context = True
                new_branch_pool_set = {int(idx) for idx in previous_newly_added_branch_leaf_pool}
                carryover_pool_indices = [
                    int(idx)
                    for idx in retrieval_pool_indices
                    if int(idx) not in new_branch_pool_set
                ]
                new_branch_pool_indices = [
                    int(idx)
                    for idx in retrieval_pool_indices
                    if int(idx) in new_branch_pool_set
                ]
                if carryover_pool_indices:
                    carryover_source_candidate_hits = _retrieve_leaf_hits(
                        query=query_pre,
                        leaf_pool_indices=carryover_pool_indices,
                        retriever=retriever,
                        node_embs=node_embs,
                        node_registry=node_registry,
                        topk=round5_mrr_pool_k,
                    )
                if new_branch_pool_indices:
                    new_branch_source_candidate_hits = _retrieve_leaf_hits(
                        query=query_pre,
                        leaf_pool_indices=new_branch_pool_indices,
                        retriever=retriever,
                        node_embs=node_embs,
                        node_registry=node_registry,
                        topk=round5_mrr_pool_k,
                    )
                carryover_source_selected_hits, carryover_source_unused_hits = _take_hits_with_extras(
                    carryover_source_candidate_hits,
                    carryover_context_target,
                )
                (
                    new_branch_source_selected_hits,
                    new_branch_source_unused_hits,
                    new_branch_group_candidate_counts,
                    new_branch_group_selected_counts,
                ) = _select_diverse_new_branch_hits(
                    new_branch_source_candidate_hits,
                    previous_newly_added_branch_paths,
                    new_branch_context_target,
                )
                if len(new_branch_source_selected_hits) < new_branch_context_target:
                    carryover_backfill_count = int(
                        min(
                            len(carryover_source_unused_hits),
                            max(0, new_branch_context_target - len(new_branch_source_selected_hits)),
                        )
                    )
                    if carryover_backfill_count > 0:
                        carryover_source_selected_hits.extend(carryover_source_unused_hits[:carryover_backfill_count])
                        carryover_source_unused_hits = carryover_source_unused_hits[carryover_backfill_count:]
                if len(carryover_source_selected_hits) < carryover_context_target:
                    new_branch_backfill_count = int(
                        min(
                            len(new_branch_source_unused_hits),
                            max(0, carryover_context_target - len(carryover_source_selected_hits)),
                        )
                    )
                    if new_branch_backfill_count > 0:
                        new_branch_source_selected_hits.extend(new_branch_source_unused_hits[:new_branch_backfill_count])
                        new_branch_source_unused_hits = new_branch_source_unused_hits[new_branch_backfill_count:]
                        new_branch_group_selected_counts = _count_hits_by_frontier_child(
                            new_branch_source_selected_hits,
                            previous_newly_added_branch_paths,
                        )
                selected_context_hits, selected_context_source_sequence = _interleave_source_hits(
                    new_branch_source_selected_hits,
                    carryover_source_selected_hits,
                )

        if not source_balanced_rewrite_context:
            pre_hits = _retrieve_leaf_hits(
                query=query_pre,
                leaf_pool_indices=retrieval_pool_indices,
                retriever=retriever,
                node_embs=node_embs,
                node_registry=node_registry,
                topk=round5_mrr_pool_k,
            )
            selected_context_hits = list(pre_hits)
            selected_context_source_sequence = ["default"] * int(min(len(selected_context_hits), rewrite_context_total_topk))
        else:
            pre_hits = list(selected_context_hits)

        pre_hit_paths = [tuple(hit.path) for hit in pre_hits]
        pre_hit_doc_ids = _paths_to_ranked_doc_ids(pre_hit_paths, path_to_doc_id)
        leaf_descs = _hits_to_context_descs(
            selected_context_hits,
            node_registry,
            topk=rewrite_context_total_topk,
            max_desc_len=hp.MAX_DOC_DESC_CHAR_LEN,
        )
        carryover_leaf_descs = _hits_to_context_descs(
            carryover_source_selected_hits,
            node_registry,
            topk=max(1, len(carryover_source_selected_hits)) if carryover_source_selected_hits else 1,
            max_desc_len=hp.MAX_DOC_DESC_CHAR_LEN,
        ) if source_balanced_rewrite_context else []
        new_branch_leaf_descs = _hits_to_context_descs(
            new_branch_source_selected_hits,
            node_registry,
            topk=max(1, len(new_branch_source_selected_hits)) if new_branch_source_selected_hits else 1,
            max_desc_len=hp.MAX_DOC_DESC_CHAR_LEN,
        ) if source_balanced_rewrite_context else []
        prompt_name_for_iter = _resolve_round6_rewrite_prompt_name(
            round5_rewrite_prompt_name,
            use_context_packing=bool(source_balanced_rewrite_context),
        )
        rewrite_template_for_iter = REWRITE_PROMPT_TEMPLATES[prompt_name_for_iter]
        emr_state = {
            "history_text": format_shared_emr_history(getattr(sample, "round6_emr_history", []) or []),
            "memory_text": "",
            "memory_doc_ids": [],
            "memory_items": [],
            "overflow_dropped": [],
            "memory_source": "off",
            "memory_pool_doc_ids": [],
            "selected_sentences": [],
            "compression_mode": round6_emr_compression,
            "prompt_total_tokens": None,
            "memory_prompt_tokens": None,
            "prompt_budget_tokens": None,
            "model_context_tokens": (
                round6_rewrite_prompt_budgeter.max_context_tokens
                if round6_rewrite_prompt_budgeter is not None
                else None
            ),
        }
        if round6_emr_memory:
            emr_state = build_shared_emr_prompt_memory(
                mode="accumulated",
                history_entries=list(getattr(sample, "round6_emr_history", []) or []),
                doc_bank=dict(getattr(sample, "round6_emr_doc_bank", {}) or {}),
                query_for_memory=query_pre,
                compressor=round6_emr_compressor,
                doc_text_by_id=doc_text_by_id,
                render_prompt=lambda history_text, memory_text: _format_rewrite_prompt(
                    rewrite_template_for_iter,
                    original_query=str(sample.original_query),
                    previous_rewrite=query_state_before_effective,
                    leaf_descs=leaf_descs,
                    carryover_descs=carryover_leaf_descs,
                    new_branch_descs=new_branch_leaf_descs,
                    domain_route_hint=subset_domain_route_hint,
                    relevance_definition=subset_relevance_definition,
                    seen_direction_evidence=local_failed_hypothesis_hint,
                    history_actions=history_text,
                    memory_docs=memory_text,
                ),
                prompt_budgeter=round6_rewrite_prompt_budgeter,
                compression_mode=round6_emr_compression,
                max_memory_tokens=round6_emr_memory_max_tokens,
                memory_source_label="eval_hits_accumulated",
            )
        rewrite_state_by_sample_idx[sample_idx] = {
            "cache_key": "",
            "cache_hit": False,
            "cached_rewrite": "",
            "cached_docs": {},
            "prompt_name": prompt_name_for_iter,
            "ended_beam_transition_step": bool(ended_beam_transition_step),
            "retrieval_pool_source": str(retrieval_pool_source),
            "query_state_before_raw": query_state_before_raw,
            "query_state_before_effective": query_state_before_effective,
            "query_state_before": query_state_before_effective,
            "query_state_after_raw": query_state_before_raw,
            "query_state_after_effective": query_state_before_effective,
            "query_state_after": query_state_before_effective,
            "query_pre": query_pre,
            "query_post": query_pre,
            "leaf_descs": leaf_descs,
            "rewrite": "",
            "rewrite_docs": {},
            "raw_output": "",
            "cumulative_pool_pre_size": int(len(cumulative_pool)),
            "masked_pool_pre_size": int(len(retrieval_pool_indices)),
            "explore_evidence_pool_size": int(len(retrieval_pool_indices)),
            "explore_evidence_pool_indices": [int(x) for x in retrieval_pool_indices],
            "retrieval_pool_indices": [int(x) for x in retrieval_pool_indices],
            "retrieval_pool_branch_component_size": int(retrieval_pool_branch_component_size),
            "retrieval_pool_cumulative_component_size": int(retrieval_pool_cumulative_component_size),
            "retrieval_pool_total_size": int(len(retrieval_pool_indices)),
            "source_balanced_rewrite_context": bool(source_balanced_rewrite_context),
            "rewrite_context_total_topk": int(rewrite_context_total_topk),
            "rewrite_context_carryover_target": int(carryover_context_target),
            "rewrite_context_new_branch_target": int(new_branch_context_target),
            "rewrite_context_carryover_candidate_count": int(len(carryover_source_candidate_hits)),
            "rewrite_context_new_branch_candidate_count": int(len(new_branch_source_candidate_hits)),
            "rewrite_context_carryover_selected_count": int(len(carryover_source_selected_hits)),
            "rewrite_context_new_branch_selected_count": int(len(new_branch_source_selected_hits)),
            "rewrite_context_carryover_backfill_count": int(carryover_backfill_count),
            "rewrite_context_new_branch_backfill_count": int(new_branch_backfill_count),
            "previous_newly_added_branch_paths": [list(path) for path in previous_newly_added_branch_paths],
            "previous_newly_added_branch_descendant_leaf_pool": [int(idx) for idx in previous_newly_added_branch_leaf_pool],
            "rewrite_context_carryover_paths": _hits_to_paths(carryover_source_selected_hits),
            "rewrite_context_new_branch_paths": _hits_to_paths(new_branch_source_selected_hits),
            "rewrite_context_carryover_doc_ids": _hits_to_ranked_doc_ids(
                carryover_source_selected_hits,
                path_to_doc_id,
            ),
            "rewrite_context_new_branch_doc_ids": _hits_to_ranked_doc_ids(
                new_branch_source_selected_hits,
                path_to_doc_id,
            ),
            "rewrite_context_selected_source_sequence": list(selected_context_source_sequence[: len(leaf_descs)]),
            "rewrite_context_new_branch_candidate_group_counts": dict(new_branch_group_candidate_counts),
            "rewrite_context_new_branch_selected_group_counts": dict(new_branch_group_selected_counts),
            "candidate_hypothesis_ids": list(candidate_hypothesis_ids),
            "active_hypothesis_ids_before": list(active_hypothesis_ids_before),
            "active_hypothesis_ids_after": [],
            "dropped_hypothesis_ids": list(dropped_hypothesis_ids_before),
            "state_local_supported_hypothesis_ids": list(local_supported_hypothesis_ids),
            "state_local_failed_hypothesis_ids": list(local_failed_hypothesis_ids),
            "hypothesis_support_scores_before": {
                str(k): float(v) for k, v in hypothesis_support_scores_before.items()
            },
            "hypothesis_support_doc_ids_before": {
                str(k): list(v) for k, v in hypothesis_support_doc_ids_before.items()
            },
            "local_failed_hypothesis_hint": str(local_failed_hypothesis_hint),
            # Intent: persist rewrite-time retrieval evidence (query_pre) for off-branch/noise attribution analysis.
            "pre_hit_paths": [list(p) for p in pre_hit_paths],
            "pre_hit_doc_ids": list(pre_hit_doc_ids),
            "emr_history": list(getattr(sample, "round6_emr_history", []) or []),
            "emr_memory_mode": "accumulated" if round6_emr_memory else "off",
            "emr_memory_compression": round6_emr_compression,
            "emr_history_prompt_text": emr_state["history_text"],
            "emr_memory_source": emr_state["memory_source"],
            "emr_memory_doc_ids": list(emr_state["memory_doc_ids"]),
            "emr_memory_docs_count": int(len(emr_state["memory_doc_ids"])),
            "emr_memory_prompt_text": emr_state["memory_text"],
            "emr_memory_overflow_dropped": list(emr_state["overflow_dropped"]),
            "emr_memory_pool_doc_ids": list(emr_state["memory_pool_doc_ids"]),
            "emr_memory_selected_sentences": list(emr_state["selected_sentences"]),
            "emr_memory_prompt_tokens": emr_state["memory_prompt_tokens"],
            "emr_prompt_total_tokens": emr_state["prompt_total_tokens"],
            "emr_prompt_budget_tokens": emr_state["prompt_budget_tokens"],
            "emr_model_context_tokens": emr_state["model_context_tokens"],
            "eval_paths": [],
            "eval_doc_ids": [],
            "active_eval_paths": [],
            "active_eval_doc_ids": [],
            "active_eval_metrics": {},
            "current_run_metrics": {},
            "current_run_bank_entry": {},
        }

        prompt = _format_rewrite_prompt(
            rewrite_template_for_iter,
            original_query=str(sample.original_query),
            previous_rewrite=query_state_before_effective,
            leaf_descs=leaf_descs,
            carryover_descs=carryover_leaf_descs,
            new_branch_descs=new_branch_leaf_descs,
            domain_route_hint=subset_domain_route_hint,
            relevance_definition=subset_relevance_definition,
            seen_direction_evidence=local_failed_hypothesis_hint,
            history_actions=emr_state["history_text"],
            memory_docs=emr_state["memory_text"],
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
                    "mode": "legacy",
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
        query_state_before_raw = str(info.get("query_state_before_raw", "") or "")
        query_state_before_effective = str(info.get("query_state_before_effective", "") or "")
        generated_hypothesis_ids: List[str] = []
        generated_hypothesis_scores: Dict[str, float] = {}
        generated_hypothesis_doc_ids: Dict[str, List[str]] = {}
        hypothesis_bank = getattr(sample, "round6_hypothesis_bank", {})
        if round6_hypbank:
            generated_hypothesis_ids, sample.round6_hypothesis_next_order = _upsert_hypotheses(
                hypothesis_bank=hypothesis_bank,
                next_order=int(getattr(sample, "round6_hypothesis_next_order", 0) or 0),
                rewrite_docs=rewrite_docs,
                rewrite_blob=rewrite_blob,
                iter_idx=iter_idx,
            )
            generated_hypothesis_scores, generated_hypothesis_doc_ids = _score_hypothesis_candidates(
                candidate_ids=generated_hypothesis_ids,
                hypothesis_bank=hypothesis_bank,
                retrieval_pool_indices=list(info.get("retrieval_pool_indices", [])),
                retriever=retriever,
                node_embs=node_embs,
                node_registry=node_registry,
                path_to_doc_id=path_to_doc_id,
                support_hits_topk=ROUND6_HYPBANK_SUPPORT_HITS_TOPK,
            )

        query_state_after_raw, query_post_raw = _compose_next_query(
            str(sample.original_query),
            query_state_before_raw,
            rewrite_blob,
            bool(info.get("ended_beam_transition_step", False)),
        )
        query_state_after_effective = query_state_after_raw
        active_hypothesis_ids_after: List[str] = []
        if round6_hypbank and bool(info.get("ended_beam_transition_step", False)):
            local_supported_ids = set(info.get("state_local_supported_hypothesis_ids", []))
            post_candidate_ids: List[str] = []
            seen_candidate_ids: Set[str] = set()
            for hypothesis_id in list(info.get("active_hypothesis_ids_before", [])) + list(generated_hypothesis_ids):
                hid = str(hypothesis_id or "").strip()
                if (not hid) or (hid in seen_candidate_ids):
                    continue
                seen_candidate_ids.add(hid)
                post_candidate_ids.append(hid)
            post_support_scores = {
                str(k): float(v) for k, v in dict(info.get("hypothesis_support_scores_before", {}) or {}).items()
            }
            post_support_scores.update({str(k): float(v) for k, v in generated_hypothesis_scores.items()})
            active_hypothesis_ids_after = _select_supported_hypothesis_ids(
                candidate_ids=post_candidate_ids,
                hypothesis_bank=hypothesis_bank,
                support_scores=post_support_scores,
                topk=ROUND6_HYPBANK_ACTIVE_TOPK,
                local_supported_ids=local_supported_ids,
            )
            composed_query_state = _compose_hypothesis_query_state(active_hypothesis_ids_after, hypothesis_bank)
            if composed_query_state:
                # Intent: reseat should carry forward only hypotheses that still earn support in the current reachable pool.
                query_state_after_effective = composed_query_state
                query_post_raw = _compose_query_from_state(str(sample.original_query), query_state_after_effective)
        elif round6_hypbank:
            active_hypothesis_ids_after = _select_supported_hypothesis_ids(
                candidate_ids=generated_hypothesis_ids,
                hypothesis_bank=hypothesis_bank,
                support_scores=generated_hypothesis_scores,
                topk=ROUND6_HYPBANK_ACTIVE_TOPK,
            )

        sample.last_rewrite_raw = rewrite_blob
        sample.round6_query_state = query_state_after_effective
        sample.query = query_post_raw

        info["rewrite"] = rewrite_blob
        info["rewrite_docs"] = rewrite_docs
        info["raw_output"] = raw_output
        info["query_state_after_raw"] = query_state_after_raw
        info["query_state_after_effective"] = query_state_after_effective
        info["query_state_after"] = query_state_after_effective
        info["query_post"] = query_post_raw
        info["generated_hypothesis_ids"] = list(generated_hypothesis_ids)
        info["generated_hypothesis_scores"] = {
            str(k): float(v) for k, v in generated_hypothesis_scores.items()
        }
        info["generated_hypothesis_doc_ids"] = {
            str(k): list(v) for k, v in generated_hypothesis_doc_ids.items()
        }
        info["active_hypothesis_ids_after"] = list(active_hypothesis_ids_after)
        if round6_hypbank:
            candidate_hypothesis_ids = list(info.get("candidate_hypothesis_ids", [])) + list(generated_hypothesis_ids)
            seen_candidate_ids: Set[str] = set()
            deduped_candidate_ids: List[str] = []
            for hypothesis_id in candidate_hypothesis_ids:
                hid = str(hypothesis_id or "").strip()
                if (not hid) or (hid in seen_candidate_ids):
                    continue
                seen_candidate_ids.add(hid)
                deduped_candidate_ids.append(hid)
            info["candidate_hypothesis_ids"] = deduped_candidate_ids
            info["dropped_hypothesis_ids"] = [
                hypothesis_id
                for hypothesis_id in deduped_candidate_ids
                if hypothesis_id not in set(active_hypothesis_ids_after)
            ]

        eval_hits = _retrieve_leaf_hits(
            query=query_post_raw,
            leaf_pool_indices=list(info.get("retrieval_pool_indices", [])),
            retriever=retriever,
            node_embs=node_embs,
            node_registry=node_registry,
            topk=max(1, int(hp.FLAT_TOPK)),
        )
        eval_paths = [tuple(hit.path) for hit in eval_hits]
        eval_doc_ids = _paths_to_ranked_doc_ids(eval_paths, path_to_doc_id)
        eval_doc_ids_eval = filter_excluded_predictions(eval_doc_ids, getattr(sample, "excluded_ids_set", None))
        gold_doc_ids = [str(x) for x in sample.gold_doc_ids]
        bank_entry = _bank_entry_from_hits(
            hits=eval_hits,
            gold_doc_ids=gold_doc_ids,
            excluded_doc_ids=list(getattr(sample, "excluded_ids_set", set()) or []),
            path_to_doc_id=path_to_doc_id,
            iter_idx=iter_idx,
            explore_step=False,
            query_post=query_post_raw,
        )
        current_run_metrics = _compute_retrieval_metrics(
            eval_doc_ids,
            gold_doc_ids,
            excluded_doc_ids=getattr(sample, "excluded_ids_set", None),
        )
        info["cumulative_pool_eval_size"] = int(info.get("retrieval_pool_total_size", 0))
        info["eval_paths"] = [list(p) for p in eval_paths]
        info["eval_doc_ids"] = list(eval_doc_ids)
        info["eval_doc_ids_eval"] = list(eval_doc_ids_eval)
        info["gold_doc_ids"] = list(gold_doc_ids)
        info["current_run_bank_entry"] = bank_entry
        info["current_run_metrics"] = {str(k): float(v) for k, v in current_run_metrics.items()}
        sample.rewrite_history.append(
            {
                "iter": iter_idx,
                "cache_hit": bool(info.get("cache_hit", False)),
                "prompt_name": str(info.get("prompt_name", round5_rewrite_prompt_name)),
                "query_state_before": query_state_before_effective,
                "query_state_after": query_state_after_effective,
                "query_pre": query_pre,
                "query_post": query_post_raw,
                "rewrite": rewrite_blob,
                "possible_answer_docs": rewrite_docs,
                "leaf_descs": info.get("leaf_descs", []),
                "raw_output": raw_output,
                "candidate_hypothesis_ids": list(info.get("candidate_hypothesis_ids", [])),
                "active_hypothesis_ids_after": list(active_hypothesis_ids_after),
            }
        )

    inputs = []
    for sample in all_eval_samples:
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
        ended_reseat_enabled = bool(round6_expandable_mode == "ended_reseat")
        ended_reseat_count = int(len(ended_selected_before)) if ended_reseat_enabled else 0
        sample.update(slates[sample_idx], response_jsons[sample_idx], rel_fn=sample.get_rel_fn())
        selector_pick_reason = "retriever_slate_default"
        selector_source = "retriever_slate"
        global_escape_pick_reason = "disabled"
        selector_candidate_paths: List[Tuple[int, ...]] = []
        selector_scored_rows: List[Dict[str, Any]] = []
        ended_reseat_scored_rows: List[Dict[str, Any]] = []
        ended_reseat_selected_rows: List[Dict[str, Any]] = []
        ended_reseat_candidate_count = 0
        selector_global_escape_rows: List[Dict[str, Any]] = []
        selector_global_escape_selected_rows: List[Dict[str, Any]] = []
        selector_global_escape_replaced_rows: List[Dict[str, Any]] = []
        selector_chosen_child_parent_map: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = {}
        if round5_selector_mode in {"maxscore_global", "meanscore_global", "max_hit_global"}:
            if round5_selector_mode == "maxscore_global":
                score_mode = "max"
            elif round5_selector_mode == "meanscore_global":
                score_mode = "mean"
            else:
                score_mode = "hit"
            candidate_parent_map: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = {}
            ended_reseat_parent_map: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = {}
            use_descendant_flat_candidates = bool(
                ended_reseat_enabled and round6_expandable_candidate_mode == "descendant_flat"
            )
            use_path_materialization = bool(
                ended_reseat_enabled
                and (
                    round6_expandable_candidate_mode == "descendant_flat"
                    or round6_expandable_ended_scope != "leftover_expandable"
                )
            )
            if use_descendant_flat_candidates:
                # Intent: flat-descendant mode lets active beams jump to any deeper branch within their current subtree.
                selector_candidate_paths, candidate_parent_map = _collect_strict_descendant_branch_paths(
                    selected_branches=selected_before,
                    child_branch_paths_by_path=child_branch_paths_by_path,
                    all_nonroot_nonleaf_paths=all_nonroot_nonleaf_paths,
                    root_path=root_path,
                )
            else:
                selector_candidate_paths, candidate_parent_map = _collect_candidate_child_branches_with_parents(
                    selected_branches=(active_selected_before if ended_reseat_enabled else selected_before),
                    child_branch_paths_by_path=child_branch_paths_by_path,
                    root_branch_children=root_branch_children,
                    root_path=root_path,
                )
            local_row_limit = max(0, int(hp.MAX_BEAM_SIZE) - ended_reseat_count) if ended_reseat_enabled else max(1, int(hp.MAX_BEAM_SIZE))
            if not selector_candidate_paths and not (ended_reseat_enabled and ended_reseat_count > 0):
                # Intent: if there is no expandable child-branch candidate, keep retriever-slate beam unchanged.
                selector_pick_reason = "no_candidate_children"
            else:
                # Intent: selector scoring should see the same frontier-plus-cumulative retrieval boundary as rewrite/eval retrieval.
                selector_local_pool = [int(x) for x in list(info.get("retrieval_pool_indices", []) or [])]
                selector_query = str(
                    info.get("query_post", "")
                    or sample.query
                    or getattr(sample, "original_query", "")
                )
                selector_source = "current_local_retrieval"
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
                    if use_descendant_flat_candidates:
                        selector_scored_rows = _filter_scored_rows_to_allowed_paths(
                            selector_scored_rows,
                            selector_candidate_paths,
                        )
                    else:
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
                        selected_path_order: List[Tuple[int, ...]] = []
                        selected_path_scores: Dict[Tuple[int, ...], float] = {}
                        if use_path_materialization:
                            for row in final_selector_rows:
                                path_t = _row_path_tuple(row)
                                if (not path_t) or (path_t in selected_endpoints):
                                    continue
                                selected_endpoints.add(path_t)
                                selected_path_order.append(path_t)
                                selected_path_scores[path_t] = float(row.get("score", 0.0))
                        else:
                            for row in final_selector_rows:
                                path_t = _row_path_tuple(row)
                                matched_state_path = expandable_state_path_map.get(path_t)
                                if not matched_state_path:
                                    continue
                                endpoint = tuple(getattr(matched_state_path[-1], "path", ()) or ())
                                if not endpoint or endpoint in selected_endpoints:
                                    continue
                                selected_endpoints.add(endpoint)
                                selected_path_scores[endpoint] = float(row.get("score", 0.0))
                                selected_state_paths.append(matched_state_path)
                        if ended_reseat_enabled and ended_reseat_count > 0:
                            if round6_expandable_ended_scope == "whole_tree_flat":
                                # Intent: flat ended-slot reseat scores the whole semantic-tree branch registry, not just instantiated leftovers.
                                leftover_candidate_paths = [
                                    tuple(path_t)
                                    for path_t in all_nonroot_nonleaf_paths
                                    if tuple(path_t) not in selected_endpoints and tuple(path_t) not in set(selected_before)
                                ]
                                replacement_leaf_pool = list(leaf_indices)
                            elif round6_expandable_ended_scope == "goexplore_direct_child":
                                leftover_candidate_paths, ended_reseat_parent_map = _collect_goexplore_direct_child_candidates(
                                    selected_branches=selected_before,
                                    child_branch_paths_by_path=child_branch_paths_by_path,
                                    root_branch_children=root_branch_children,
                                    root_path=root_path,
                                )
                                leftover_candidate_paths = [
                                    tuple(path_t)
                                    for path_t in leftover_candidate_paths
                                    if tuple(path_t) not in selected_endpoints
                                ]
                                replacement_leaf_pool, _, _ = _build_union_leaf_pool(
                                    base_branch_paths=leftover_candidate_paths,
                                    cumulative_leaf_indices=sorted(cumulative_leaf_indices_by_sample[sample_idx]),
                                    leaf_indices_by_prefix=leaf_indices_by_prefix,
                                    all_leaf_indices=leaf_indices,
                                    fallback_to_all=False,
                                )
                            else:
                                leftover_candidate_paths = [
                                    tuple(path_t)
                                    for path_t in expandable_state_path_map.keys()
                                    if tuple(path_t) not in selected_endpoints
                                ]
                                replacement_leaf_pool, _, _ = _build_union_leaf_pool(
                                    base_branch_paths=leftover_candidate_paths,
                                    cumulative_leaf_indices=sorted(cumulative_leaf_indices_by_sample[sample_idx]),
                                    leaf_indices_by_prefix=leaf_indices_by_prefix,
                                    all_leaf_indices=leaf_indices,
                                    fallback_to_all=False,
                                )
                            ended_reseat_candidate_count = int(len(leftover_candidate_paths))
                            if leftover_candidate_paths:
                                if round6_expandable_reseat_policy == "random":
                                    random_seed_key = (
                                        f"{hp.SUBSET}|{sample_idx}|{iter_idx}|"
                                        f"{ended_reseat_count}|{round6_expandable_ended_scope}|ended_reseat_random"
                                    )
                                    ended_reseat_selected_rows = _deterministic_random_rows(
                                        candidate_paths=leftover_candidate_paths,
                                        sample_count=max(0, ended_reseat_count),
                                        seed_key=random_seed_key,
                                    )
                                elif replacement_leaf_pool:
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
                                        if round6_expandable_ended_scope == "leftover_expandable":
                                            ended_reseat_scored_rows = _filter_scored_rows_to_expandable(
                                                ended_reseat_scored_rows,
                                                expandable_state_path_map,
                                            )
                                        else:
                                            ended_reseat_scored_rows = _filter_scored_rows_to_allowed_paths(
                                                ended_reseat_scored_rows,
                                                leftover_candidate_paths,
                                            )
                                        ended_reseat_selected_rows = list(
                                            ended_reseat_scored_rows[: max(0, ended_reseat_count)]
                                        )
                        local_selected_count = len(selected_path_order) if use_path_materialization else len(selected_state_paths)
                        if local_selected_count == 0 and not ended_reseat_selected_rows:
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
                            fallback_state_paths = list(getattr(sample, "beam_state_paths", None) or [])
                            if use_path_materialization:
                                final_beam_paths = list(selected_path_order)
                                if ended_reseat_enabled and ended_reseat_selected_rows:
                                    # Intent: keep non-ended continuation but let ended slots jump to retrieval-selected branch paths.
                                    for row in ended_reseat_selected_rows:
                                        path_t = _row_path_tuple(row)
                                        if (not path_t) or (path_t in selected_endpoints):
                                            continue
                                        selected_endpoints.add(path_t)
                                        final_beam_paths.append(path_t)
                                        selected_path_scores[path_t] = float(row.get("score", 0.0))
                                        if len(final_beam_paths) >= max(1, int(hp.MAX_BEAM_SIZE)):
                                            break
                                # Intent: preserve baseline traversal continuity by filling missing beams with current beam endpoints.
                                for fallback_endpoint in _state_paths_to_endpoint_paths(fallback_state_paths):
                                    if (not fallback_endpoint) or (fallback_endpoint in selected_endpoints):
                                        continue
                                    selected_endpoints.add(fallback_endpoint)
                                    final_beam_paths.append(fallback_endpoint)
                                    if len(final_beam_paths) >= max(1, int(hp.MAX_BEAM_SIZE)):
                                        break
                                _materialize_beam_from_paths(
                                    sample=sample,
                                    beam_paths=final_beam_paths[: max(1, int(hp.MAX_BEAM_SIZE))],
                                    path_scores=selected_path_scores,
                                )
                            else:
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
                            local_pick_reason = (
                                f"{round5_selector_mode}_applied_full"
                                if local_selected_count >= max(1, int(hp.MAX_BEAM_SIZE))
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
                                if round6_expandable_reseat_policy == "random":
                                    if (
                                        round6_expandable_candidate_mode == "direct_children"
                                        and round6_expandable_ended_scope == "leftover_expandable"
                                    ):
                                        selector_pick_reason = (
                                            f"{round5_selector_mode}_ended_reseat_random_{ended_reseat_status}"
                                        )
                                    else:
                                        selector_pick_reason = (
                                            f"{round5_selector_mode}_ended_reseat_random_"
                                            f"{round6_expandable_candidate_mode}_{round6_expandable_ended_scope}_{ended_reseat_status}"
                                        )
                                elif (
                                    round6_expandable_candidate_mode == "direct_children"
                                    and round6_expandable_ended_scope == "leftover_expandable"
                                ):
                                    selector_pick_reason = f"{round5_selector_mode}_ended_reseat_{ended_reseat_status}"
                                else:
                                    selector_pick_reason = (
                                        f"{round5_selector_mode}_ended_reseat_"
                                        f"{round6_expandable_candidate_mode}_{round6_expandable_ended_scope}_{ended_reseat_status}"
                                    )
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
        info["ended_beam_reseat_candidate_mode"] = str(round6_expandable_candidate_mode if ended_reseat_enabled else "")
        info["ended_beam_reseat_scope"] = str(round6_expandable_ended_scope if ended_reseat_enabled else "")
        info["ended_beam_reseat_policy"] = str(round6_expandable_reseat_policy if ended_reseat_enabled else "")
        info["ended_beam_reseat_candidate_count"] = int(ended_reseat_candidate_count)
        reached_after = sample.get_top_predictions(k=None, rel_fn=sample.get_rel_fn(leaf=True))
        existing_leaf_indices = set(cumulative_leaf_indices_by_sample[sample_idx])
        new_leaf_paths: List[Tuple[int, ...]] = []
        for node, _ in reached_after:
            ridx = int(getattr(node, "registry_idx", -1))
            if ridx >= 0:
                if ridx not in existing_leaf_indices:
                    leaf_path = tuple(getattr(node, "path", ()) or ())
                    if leaf_path:
                        new_leaf_paths.append(leaf_path)
                cumulative_leaf_indices_by_sample[sample_idx].add(ridx)
        selected_after_current = _selected_branch_paths_from_sample(sample)
        info = rewrite_state_by_sample_idx[sample_idx]
        selected_before_current = list(selected_before_by_idx.get(sample_idx, []))
        selected_before_set = {tuple(path) for path in selected_before_current}
        reseat_added_branch_paths: List[Tuple[int, ...]] = []
        if bool(info.get("ended_beam_transition_step", False)):
            seen_reseat_added_paths: Set[Tuple[int, ...]] = set()
            for row in ended_reseat_selected_rows:
                path_t = tuple(_row_path_tuple(row))
                if (not path_t) or (path_t in selected_before_set) or (path_t in seen_reseat_added_paths):
                    continue
                if path_t not in set(selected_after_current):
                    continue
                seen_reseat_added_paths.add(path_t)
                reseat_added_branch_paths.append(path_t)
        reseat_added_branch_leaf_pool = sorted(
            {
                int(idx)
                for branch_path in reseat_added_branch_paths
                for idx in leaf_indices_by_prefix.get(tuple(branch_path), [])
            }
        )
        sample.round6_newly_added_branch_paths_after_iter = [list(path) for path in reseat_added_branch_paths]
        sample.round6_newly_added_branch_descendant_leaf_pool = [int(idx) for idx in reseat_added_branch_leaf_pool]
        info["newly_added_branch_paths_after_iter"] = [list(path) for path in reseat_added_branch_paths]
        info["newly_added_branch_descendant_leaf_pool"] = [int(idx) for idx in reseat_added_branch_leaf_pool]
        if round6_hypbank:
            hypothesis_bank = getattr(sample, "round6_hypothesis_bank", {})
            used_hypothesis_ids = list(info.get("generated_hypothesis_ids", []))
            supported_hypothesis_ids = list(info.get("active_hypothesis_ids_after", []))
            if bool(info.get("ended_beam_transition_step", False)):
                used_hypothesis_ids = list(info.get("candidate_hypothesis_ids", []))
                supported_hypothesis_ids = list(info.get("active_hypothesis_ids_after", []))
            failed_hypothesis_ids = [
                hypothesis_id
                for hypothesis_id in used_hypothesis_ids
                if hypothesis_id not in set(supported_hypothesis_ids)
            ]
            _update_state_hypothesis_ledger(
                state_ledger=getattr(sample, "round6_state_hypothesis_ledger", {}),
                state_paths=selected_after_current,
                used_ids=used_hypothesis_ids,
                supported_ids=supported_hypothesis_ids,
                failed_ids=failed_hypothesis_ids,
                iter_idx=iter_idx,
            )
            for hypothesis_id in used_hypothesis_ids:
                entry = dict(hypothesis_bank.get(str(hypothesis_id), {}) or {})
                if not entry:
                    continue
                entry["last_used_iter"] = int(iter_idx)
                if hypothesis_id in set(supported_hypothesis_ids):
                    entry["status"] = "supported"
                elif hypothesis_id in set(failed_hypothesis_ids):
                    entry["status"] = "failed"
                score = dict(info.get("generated_hypothesis_scores", {}) or {}).get(str(hypothesis_id))
                if score is None:
                    score = dict(info.get("hypothesis_support_scores_before", {}) or {}).get(str(hypothesis_id))
                if score is not None:
                    entry["last_support_score"] = float(score)
                # Intent: state-local reseat memory only works if support/failure status is persisted back into the shared hypothesis bank.
                hypothesis_bank[str(hypothesis_id)] = entry

        current_run_metrics = {
            str(k): float(v)
            for k, v in dict(info.get("current_run_metrics", {}) or {}).items()
        }
        active_eval_doc_ids = list(info.get("eval_doc_ids_eval", []))
        active_metrics = dict(current_run_metrics)
        active_eval_paths = list(info.get("eval_paths", []))
        info["active_eval_paths"] = active_eval_paths
        info["active_eval_doc_ids"] = list(active_eval_doc_ids)
        info["active_eval_metrics"] = {str(k): float(v) for k, v in active_metrics.items()}
        retrieval_row = {str(k): float(v) for k, v in active_metrics.items()}
        retrieval_rows.append(retrieval_row)
        rewrite_state_by_sample_idx[sample_idx]["new_leaf_paths"] = [list(path) for path in new_leaf_paths]
        if round6_emr_memory:
            appended_doc_ids = build_shared_current_doc_memory_items(
                retrieved_doc_ids=info.get("eval_doc_ids", []),
                doc_text_by_id=doc_text_by_id,
                doc_topk=round6_emr_topk,
            )
            sample.round6_emr_doc_bank_next_order = update_shared_accumulated_doc_bank(
                getattr(sample, "round6_emr_doc_bank", {}),
                next_order=int(getattr(sample, "round6_emr_doc_bank_next_order", 0) or 0),
                current_doc_ids=appended_doc_ids,
                iter_idx=iter_idx,
                query_for_memory=str(info.get("query_post", "") or ""),
            )
            append_shared_emr_history_entry(
                getattr(sample, "round6_emr_history", []),
                applied_query=str(info.get("query_post", "") or ""),
                retrieved_doc_ids=info.get("eval_doc_ids", []),
                history_rank_topk=round6_emr_topk,
            )
            info["emr_history_after_write"] = list(getattr(sample, "round6_emr_history", []) or [])
            info["emr_memory_appended_doc_ids"] = list(appended_doc_ids)
        else:
            info["emr_history_after_write"] = list(getattr(sample, "round6_emr_history", []) or [])
            info["emr_memory_appended_doc_ids"] = []

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
                "query_state_before": str(info.get("query_state_before", "")),
                "query_state_after": str(info.get("query_state_after", "")),
                "query_pre": str(info.get("query_pre", "")),
                "query_post": str(info.get("query_post", "")),
                "rewrite": str(info.get("rewrite", "")),
                "rewrite_context_topk": int(hp.REWRITE_CONTEXT_TOPK),
                "rewrite_leaf_descs_count": int(len(info.get("leaf_descs", []))),
                "possible_answer_docs": info.get("rewrite_docs", {}),
                "selected_branches_before": [list(p) for p in selected_before_by_idx.get(sample_idx, [])],
                "selected_branches_after": [list(p) for p in selected_after],
                "ended_beam_transition_step": bool(info.get("ended_beam_transition_step", False)),
                "selector_mode": round5_selector_mode,
                "selector_source": str(selector_source_by_idx.get(sample_idx, "retriever_slate")),
                "selector_pick_reason": str(selector_pick_reason_by_idx.get(sample_idx, "retriever_slate_default")),
                "global_escape_enabled": bool(round6_global_escape),
                "global_escape_slots": int(round6_global_escape_slots),
                "round6_expandable_mode": round6_expandable_mode if round6_expandable_mode != "off" else "",
                "round6_expandable_candidate_mode": (
                    round6_expandable_candidate_mode if round6_expandable_mode != "off" else ""
                ),
                "round6_expandable_ended_scope": (
                    round6_expandable_ended_scope if round6_expandable_mode != "off" else ""
                ),
                "round6_expandable_reseat_policy": (
                    round6_expandable_reseat_policy if round6_expandable_mode != "off" else ""
                ),
                "global_escape_pick_reason": str(global_escape_pick_reason_by_idx.get(sample_idx, "disabled")),
                "ended_beam_count": int(info.get("ended_beam_count", 0)),
                "ended_beam_paths": info.get("ended_beam_paths", []),
                "ended_beam_reseat_enabled": bool(info.get("ended_beam_reseat_enabled", False)),
                "ended_beam_reseat_candidate_mode": str(info.get("ended_beam_reseat_candidate_mode", "")),
                "ended_beam_reseat_scope": str(info.get("ended_beam_reseat_scope", "")),
                "ended_beam_reseat_policy": str(info.get("ended_beam_reseat_policy", "")),
                "ended_beam_reseat_candidate_count": int(info.get("ended_beam_reseat_candidate_count", 0)),
                "new_leaf_paths": info.get("new_leaf_paths", []),
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
                "retrieval_pool_source": str(info.get("retrieval_pool_source", "")),
                "retrieval_pool_branch_component_size": int(info.get("retrieval_pool_branch_component_size", 0)),
                "retrieval_pool_cumulative_component_size": int(info.get("retrieval_pool_cumulative_component_size", 0)),
                "retrieval_pool_total_size": int(info.get("retrieval_pool_total_size", 0)),
                "source_balanced_rewrite_context": bool(info.get("source_balanced_rewrite_context", False)),
                "rewrite_context_total_topk": int(info.get("rewrite_context_total_topk", 0)),
                "rewrite_context_carryover_target": int(info.get("rewrite_context_carryover_target", 0)),
                "rewrite_context_new_branch_target": int(info.get("rewrite_context_new_branch_target", 0)),
                "rewrite_context_carryover_candidate_count": int(info.get("rewrite_context_carryover_candidate_count", 0)),
                "rewrite_context_new_branch_candidate_count": int(info.get("rewrite_context_new_branch_candidate_count", 0)),
                "rewrite_context_carryover_selected_count": int(info.get("rewrite_context_carryover_selected_count", 0)),
                "rewrite_context_new_branch_selected_count": int(info.get("rewrite_context_new_branch_selected_count", 0)),
                "rewrite_context_carryover_backfill_count": int(info.get("rewrite_context_carryover_backfill_count", 0)),
                "rewrite_context_new_branch_backfill_count": int(info.get("rewrite_context_new_branch_backfill_count", 0)),
                "previous_newly_added_branch_paths": info.get("previous_newly_added_branch_paths", []),
                "previous_newly_added_branch_descendant_leaf_pool": info.get("previous_newly_added_branch_descendant_leaf_pool", []),
                "newly_added_branch_paths_after_iter": info.get("newly_added_branch_paths_after_iter", []),
                "newly_added_branch_descendant_leaf_pool": info.get("newly_added_branch_descendant_leaf_pool", []),
                "rewrite_context_carryover_paths": info.get("rewrite_context_carryover_paths", []),
                "rewrite_context_new_branch_paths": info.get("rewrite_context_new_branch_paths", []),
                "rewrite_context_carryover_doc_ids": info.get("rewrite_context_carryover_doc_ids", []),
                "rewrite_context_new_branch_doc_ids": info.get("rewrite_context_new_branch_doc_ids", []),
                "rewrite_context_selected_source_sequence": info.get("rewrite_context_selected_source_sequence", []),
                "rewrite_context_new_branch_candidate_group_counts": info.get("rewrite_context_new_branch_candidate_group_counts", {}),
                "rewrite_context_new_branch_selected_group_counts": info.get("rewrite_context_new_branch_selected_group_counts", {}),
                "query_state_before_raw": str(info.get("query_state_before_raw", "")),
                "query_state_before_effective": str(info.get("query_state_before_effective", "")),
                "query_state_after_raw": str(info.get("query_state_after_raw", "")),
                "query_state_after_effective": str(info.get("query_state_after_effective", "")),
                "candidate_hypothesis_ids": info.get("candidate_hypothesis_ids", []),
                "active_hypothesis_ids_before": info.get("active_hypothesis_ids_before", []),
                "active_hypothesis_ids_after": info.get("active_hypothesis_ids_after", []),
                "generated_hypothesis_ids": info.get("generated_hypothesis_ids", []),
                "dropped_hypothesis_ids": info.get("dropped_hypothesis_ids", []),
                "state_local_supported_hypothesis_ids": info.get("state_local_supported_hypothesis_ids", []),
                "state_local_failed_hypothesis_ids": info.get("state_local_failed_hypothesis_ids", []),
                "local_failed_hypothesis_hint": str(info.get("local_failed_hypothesis_hint", "")),
                "hypothesis_support_scores_before": info.get("hypothesis_support_scores_before", {}),
                "hypothesis_support_doc_ids_before": info.get("hypothesis_support_doc_ids_before", {}),
                "generated_hypothesis_scores": info.get("generated_hypothesis_scores", {}),
                "generated_hypothesis_doc_ids": info.get("generated_hypothesis_doc_ids", {}),
                "emr_history": info.get("emr_history", []),
                "emr_memory_mode": str(info.get("emr_memory_mode", "off")),
                "emr_memory_compression": str(info.get("emr_memory_compression", round6_emr_compression)),
                "emr_history_prompt_text": str(info.get("emr_history_prompt_text", "")),
                "emr_history_after_write": info.get("emr_history_after_write", []),
                "emr_memory_source": str(info.get("emr_memory_source", "off")),
                "emr_memory_doc_ids": info.get("emr_memory_doc_ids", []),
                "emr_memory_docs_count": int(info.get("emr_memory_docs_count", 0)),
                "emr_memory_pool_doc_ids": info.get("emr_memory_pool_doc_ids", []),
                "emr_memory_selected_sentences": info.get("emr_memory_selected_sentences", []),
                "emr_memory_prompt_text": str(info.get("emr_memory_prompt_text", "")),
                "emr_memory_overflow_dropped": info.get("emr_memory_overflow_dropped", []),
                "emr_memory_prompt_tokens": info.get("emr_memory_prompt_tokens", None),
                "emr_prompt_total_tokens": info.get("emr_prompt_total_tokens", None),
                "emr_prompt_budget_tokens": info.get("emr_prompt_budget_tokens", None),
                "emr_model_context_tokens": info.get("emr_model_context_tokens", None),
                "emr_memory_appended_doc_ids": info.get("emr_memory_appended_doc_ids", []),
                "pre_hit_paths": info.get("pre_hit_paths", []),
                "pre_hit_doc_ids": info.get("pre_hit_doc_ids", []),
                "local_paths": info.get("eval_paths", []),
                "local_doc_ids": info.get("eval_doc_ids", []),
                "local_doc_ids_eval": info.get("eval_doc_ids_eval", []),
                "active_eval_paths": info.get("active_eval_paths", []),
                "active_eval_doc_ids": info.get("active_eval_doc_ids", []),
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
