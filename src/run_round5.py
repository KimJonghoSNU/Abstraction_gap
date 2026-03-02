import asyncio
import json
import logging
import os
import pickle as pkl
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from json_repair import repair_json
from tqdm.autonotebook import tqdm

from cache_utils import _prompt_cache_key, append_jsonl
from flat_then_tree import FlatHit
from history_prompts import prepend_history_to_prompt
from hyperparams import HyperParams
from llm_apis import GenAIAPI, VllmAPI
from retrievers.diver import DiverEmbeddingModel
from rewrite_prompts import REWRITE_PROMPT_TEMPLATES
from tree_objects import SemanticNode
from utils import (
    compute_ndcg,
    compute_node_registry,
    compute_recall,
    get_node_id,
    normalize_embeddings,
    save_exp,
    setup_logger,
)


@dataclass
class Round3Sample:
    original_query: str
    gold_paths: List[Tuple[int, ...]]
    gold_doc_ids: List[str]
    excluded_ids: List[str]
    last_rewrite: str = ""
    last_action: str = "exploit"
    last_actions: Dict[str, str] = None
    last_possible_docs: Dict[str, str] = None
    last_category_support_scores: Dict[str, float] = None
    last_category_decision_mode: str = ""
    last_category_decision_signal: Dict[str, float] = None
    last_category_drop_risk_scores: Dict[str, float] = None
    last_category_lock_applied: bool = False
    last_category_lock_source: List[str] = None
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
        if self.last_category_support_scores is None:
            self.last_category_support_scores = {}
        if self.last_category_decision_signal is None:
            self.last_category_decision_signal = {}
        if self.last_category_drop_risk_scores is None:
            self.last_category_drop_risk_scores = {}
        if self.last_category_lock_source is None:
            self.last_category_lock_source = []

    def to_dict(self) -> Dict:
        return {
            "original_query": self.original_query,
            "gold_paths": self.gold_paths,
            "gold_doc_ids": self.gold_doc_ids,
            "excluded_ids": self.excluded_ids,
            "last_rewrite": self.last_rewrite,
            "last_action": self.last_action,
            "last_actions": self.last_actions,
            "last_possible_docs": self.last_possible_docs,
            "last_category_support_scores": self.last_category_support_scores,
            "last_category_decision_mode": self.last_category_decision_mode,
            "last_category_decision_signal": self.last_category_decision_signal,
            "last_category_drop_risk_scores": self.last_category_drop_risk_scores,
            "last_category_lock_applied": self.last_category_lock_applied,
            "last_category_lock_source": self.last_category_lock_source,
            "rewrite_history": self.rewrite_history,
            "iter_records": self.iter_records,
        }


CATEGORY_ORDER = ["Theory", "Entity", "Example", "Other"]
STACKEXCHANGE_SUBSETS = {
    "biology",
    "earth_science",
    "economics",
    "psychology",
    "robotics",
    "sustainable_living",
}
CODING_SUBSETS = {"leetcode", "pony"}
THEOREM_SUBSETS = {"aops", "theoq", "theot"}


def _build_domain_route_hint(subset_name: str) -> str:
    name = str(subset_name or "").strip().lower()
    # Intent: enforce subset-driven routing so rewriter follows one domain-specific evidence pattern.
    if name in CODING_SUBSETS:
        return (
            "Route: coding\n"
            "- Mandatory roles: ReferenceDocumentation and WorkedExample.\n"
            "- Prefer API signatures, error strings, and implementation patterns.\n"
            "- Avoid theorem-proof-only phrasing unless the query explicitly requires it."
        )
    if name in THEOREM_SUBSETS:
        return (
            "Route: theorem-based\n"
            "- Mandatory roles: ConceptTheory and ProcedureRecipe.\n"
            "- Prefer theorem statements, lemma chains, and proof-strategy terms.\n"
            "- Avoid code-documentation-heavy phrasing."
        )
    return (
        "Route: stackexchange science/why\n"
        "- Mandatory role: ConceptTheory.\n"
        "- Include mechanism-oriented evidence via DefinitionNaming and/or EmpiricalEvidence.\n"
        "- Avoid coding-specific documentation focus unless explicitly requested."
    )


def _build_corpus_categories_hint(base_dir: str, dataset: str, subset: str, max_per_level1: int = 6) -> str:
    registry_path = os.path.join(
        base_dir,
        "trees",
        str(dataset),
        str(subset),
        "category_registry_category_assign_v2.json",
    )
    if not os.path.exists(registry_path):
        return "- Theory (no registry found)\n- Method (no registry found)\n- Evidence (no registry found)"
    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return "- Theory (registry load failed)\n- Method (registry load failed)\n- Evidence (registry load failed)"

    categories = payload.get("categories", [])
    by_level1: Dict[str, List[Tuple[str, int]]] = {}
    for row in categories:
        if not isinstance(row, dict):
            continue
        level1 = str(row.get("level1", "")).strip()
        level2 = str(row.get("level2", row.get("label", ""))).strip()
        try:
            count = int(row.get("count", 0))
        except Exception:
            count = 0
        if not level1 or not level2:
            continue
        by_level1.setdefault(level1, []).append((level2, count))

    if not by_level1:
        return "- Theory (empty)\n- Method (empty)\n- Evidence (empty)"

    lines: List[str] = []
    ordered_level1 = sorted(
        by_level1.items(),
        key=lambda kv: (-sum(int(x[1]) for x in kv[1]), str(kv[0])),
    )
    for level1, pairs in ordered_level1:
        seen: set[str] = set()
        top_level2: List[str] = []
        for level2, _ in sorted(pairs, key=lambda x: (-int(x[1]), str(x[0]))):
            if level2 in seen:
                continue
            seen.add(level2)
            top_level2.append(level2)
            if len(top_level2) >= int(max_per_level1):
                break
        if not top_level2:
            continue
        # Intent: present compact level1->level2 exemplars so the rewriter uses role taxonomies, not surface topics.
        lines.append(f"- {level1} ({', '.join(top_level2)})")

    return "\n".join(lines) if lines else "- Theory (empty)\n- Method (empty)\n- Evidence (empty)"


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


def _format_action_prompt(
    template: str,
    original_query: str,
    previous_rewrite: str,
    previous_docs: Dict[str, str],
    leaf_descs: List[str],
    branch_descs: List[str],
    subset_name: str = "",
    retrieval_history: str = "",
    corpus_categories: str = "",
) -> str:
    leaf_blob = "\n".join([x for x in leaf_descs if x])
    branch_blob = "\n".join([x for x in branch_descs if x])
    gate_blob = "\n".join([x for x in (leaf_descs + branch_descs) if x])
    domain_route_hint = _build_domain_route_hint(subset_name)
    prev_lines = []
    for key in _ordered_doc_keys(previous_docs or {}):
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
            domain_route_hint=domain_route_hint,
            corpus_categories=corpus_categories,
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
            .replace("{domain_route_hint}", domain_route_hint)
            .replace("{corpus_categories}", corpus_categories)
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
            for key, val in raw_actions.items():
                key_s = str(key or "").strip()
                if not key_s:
                    continue
                actions[key_s] = _normalize_action(val)
        if isinstance(raw_docs, dict) and raw_docs:
            for key, val in raw_docs.items():
                key_s = str(key or "").strip()
                if not key_s:
                    continue
                if isinstance(val, (list, dict)):
                    val = json.dumps(val, ensure_ascii=False)
                if isinstance(val, str) and val.strip():
                    docs[key_s] = val.strip()
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
        return None, (docs if docs else None), action, rewrite
    return None, None, "exploit", cleaned.strip()


def _flatten_docs_by_action(docs: Dict[str, str], actions: Dict[str, str]) -> str:
    pieces: List[str] = []
    for key in _ordered_doc_keys(docs):
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


def _select_categories_by_policy(
    actions: Dict[str, str],
    docs: Dict[str, str],
    policy: str,
    soft_keep: int,
) -> List[str]:
    available = _ordered_doc_keys(docs)
    if not available:
        return []
    if policy == "none":
        return available
    selected = [key for key in available if actions.get(key, "EXPLOIT") != "PRUNE"]
    if selected:
        # Intent: when round4 selector already marks PRUNE, use that set directly without extra soft_keep trimming.
        return selected
    keep = max(1, int(soft_keep))
    # Intent: focus the query on a small set of categories to implement exploit-style narrowing.
    exploit = [k for k in available if actions.get(k, "EXPLOIT") == "EXPLOIT"]
    explore = [k for k in available if actions.get(k, "EXPLOIT") == "EXPLORE"]
    if exploit:
        ordered = exploit + [k for k in explore if k not in exploit]
    elif explore:
        # Intent: when all active categories are explore, keep broad coverage instead of trimming to soft_keep.
        return explore
        # ordered = explore  # previous behavior
    else:
        ordered = available
    return ordered[:keep]


def _compose_query_from_docs_with_policy(
    original_query: str,
    docs: Dict[str, str],
    actions: Dict[str, str],
    policy: str,
    soft_keep: int,
) -> Tuple[str, List[str]]:
    keep_keys = _select_categories_by_policy(actions, docs, policy, soft_keep)
    pieces: List[str] = []
    for key in keep_keys:
        if actions.get(key, "EXPLOIT") == "PRUNE":
            continue
        text = docs.get(key, "")
        if text:
            pieces.append(text)
    blob = "\n".join(pieces).strip()
    if not blob:
        return original_query, keep_keys
    return (original_query + " " + blob).strip(), keep_keys


def _clean_docs_map(docs: Dict[str, str]) -> Dict[str, str]:
    return {
        key: str(val).strip()
        for key, val in (docs or {}).items()
        if str(key or "").strip() and str(val or "").strip()
    }


def _build_actions_keep_all(docs: Dict[str, str]) -> Dict[str, str]:
    return {
        key: "EXPLOIT"
        for key in _ordered_doc_keys(docs)
        if str(docs.get(key, "")).strip()
    }


def _merge_best_docs_by_support(
    *,
    previous_best_docs: Dict[str, str],
    previous_best_support: Dict[str, float],
    candidate_docs: Dict[str, str],
    candidate_support: Dict[str, float],
) -> Tuple[Dict[str, str], Dict[str, float], List[str]]:
    merged_docs = _clean_docs_map(previous_best_docs or {})
    merged_support = {
        str(key): float(val)
        for key, val in (previous_best_support or {}).items()
        if str(key or "").strip() in merged_docs
    }
    updated: List[str] = []
    candidate_clean = _clean_docs_map(candidate_docs or {})
    for key, text in candidate_clean.items():
        score = float(candidate_support.get(key, 0.0))
        prev_score = float(merged_support.get(key, float("-inf")))
        # Intent: update per-category best rewrite only when the same-category support strictly improves.
        if (key not in merged_docs) or (score > prev_score):
            merged_docs[key] = text
            merged_support[key] = score
            updated.append(key)
    for key in list(merged_support.keys()):
        if key not in merged_docs:
            merged_support.pop(key, None)
    return merged_docs, merged_support, updated


def _embed_texts_cached(
    texts: Sequence[str],
    retriever: DiverEmbeddingModel,
    emb_cache: Dict[str, np.ndarray],
) -> np.ndarray:
    cleaned: List[str] = []
    for text in texts:
        s = str(text or "").strip()
        if s:
            cleaned.append(s)
    if not cleaned:
        return np.zeros((0, 0), dtype=np.float32)
    missing: List[str] = [text for text in cleaned if text not in emb_cache]
    if missing:
        missing_embs = retriever.encode_docs(missing, batch_size=4)
        for text, emb in zip(missing, missing_embs):
            emb_cache[text] = emb.astype(np.float32, copy=False)
    rows = [emb_cache[text] for text in cleaned]
    return np.vstack(rows).astype(np.float32, copy=False)


def _compute_category_support(
    *,
    docs: Dict[str, str],
    evidence_descs: Sequence[str],
    support_topm: int,
    retriever: DiverEmbeddingModel,
    emb_cache: Dict[str, np.ndarray],
) -> Tuple[List[str], Dict[str, float], np.ndarray]:
    keys = _ordered_doc_keys(docs)
    if not keys:
        return [], {}, np.zeros((0, 0), dtype=np.float32)
    doc_texts = [str(docs[key]).strip() for key in keys]
    doc_embs = _embed_texts_cached(doc_texts, retriever, emb_cache)
    evidence_texts = [str(text or "").strip() for text in evidence_descs if str(text or "").strip()]
    sim = np.zeros((len(keys), 0), dtype=np.float32)
    if evidence_texts and doc_embs.size > 0:
        evidence_embs = _embed_texts_cached(evidence_texts, retriever, emb_cache)
        if evidence_embs.size > 0:
            sim = doc_embs @ evidence_embs.T
    support_by_key: Dict[str, float] = {key: 0.0 for key in keys}
    if sim.size > 0:
        topm = max(1, min(int(support_topm), sim.shape[1]))
        support_vals = np.mean(np.sort(sim, axis=1)[:, -topm:], axis=1)
        support_by_key = {key: float(val) for key, val in zip(keys, support_vals.tolist())}
    key_order = {key: idx for idx, key in enumerate(keys)}
    ordered = sorted(
        keys,
        key=lambda key: (-support_by_key[key], key_order[key]),
    )
    if sim.size > 0:
        key_to_idx = {key: idx for idx, key in enumerate(keys)}
        sim = sim[[key_to_idx[key] for key in ordered], :]
    return ordered, support_by_key, sim


def _reorder_anchor_topk_with_category_and_query_mean(
    *,
    anchor_paths: Sequence[Tuple[int, ...]],
    query_docs: Dict[str, str],
    query_categories: Sequence[str],
    node_by_path: Dict[Tuple[int, ...], object],
    path_to_registry_idx: Dict[Tuple[int, ...], int],
    scores_all: np.ndarray,
    topk: int,
    max_desc_len: int | None,
    retriever: DiverEmbeddingModel,
    emb_cache: Dict[str, np.ndarray],
) -> Tuple[List[Tuple[int, ...]], List[Dict[str, object]]]:
    if not anchor_paths or topk <= 0:
        return list(anchor_paths), []

    ordered_keys = _ordered_doc_keys(query_docs)
    if query_categories:
        category_keys = [k for k in query_categories if k in query_docs and str(query_docs.get(k, "")).strip()]
    else:
        category_keys = [k for k in ordered_keys if str(query_docs.get(k, "")).strip()]
    if not category_keys:
        return list(anchor_paths), []

    prefix_paths = [tuple(p) for p in anchor_paths[:topk]]
    suffix_paths = [tuple(p) for p in anchor_paths[topk:]]
    if not prefix_paths:
        return list(anchor_paths), []

    category_texts = [str(query_docs[key]).strip() for key in category_keys]
    category_embs = _embed_texts_cached(category_texts, retriever, emb_cache)
    if category_embs.size == 0:
        return list(anchor_paths), []

    desc_indices: List[int] = []
    desc_texts: List[str] = []
    query_scores: List[float] = []
    for idx, path in enumerate(prefix_paths):
        ridx = path_to_registry_idx.get(path, -1)
        q_score = float(scores_all[ridx]) if ridx >= 0 else 0.0
        query_scores.append(q_score)

        node = node_by_path.get(path)
        if not node:
            continue
        desc = str(getattr(node, "desc", "") or "").strip()
        if not desc:
            continue
        if max_desc_len:
            desc = desc[:max_desc_len]
        if not desc:
            continue
        desc_indices.append(idx)
        desc_texts.append(desc)

    category_score_matrix = np.zeros((len(prefix_paths), len(category_keys)), dtype=np.float32)
    if desc_texts:
        doc_embs = _embed_texts_cached(desc_texts, retriever, emb_cache)
        if doc_embs.size > 0:
            sim_part = doc_embs @ category_embs.T
            for row_idx, prefix_idx in enumerate(desc_indices):
                category_score_matrix[prefix_idx, :] = sim_part[row_idx, :]

    combined_scores = np.mean(
        np.concatenate(
            [category_score_matrix, np.array(query_scores, dtype=np.float32).reshape(-1, 1)],
            axis=1,
        ),
        axis=1,
    )

    reorder_idx = sorted(
        range(len(prefix_paths)),
        key=lambda idx: (-float(combined_scores[idx]), idx),
    )
    reordered_prefix = [prefix_paths[idx] for idx in reorder_idx]
    reordered_paths = reordered_prefix + suffix_paths

    score_rows: List[Dict[str, object]] = []
    for rank_after, idx in enumerate(reorder_idx, start=1):
        category_scores = {
            key: float(category_score_matrix[idx, col_idx])
            for col_idx, key in enumerate(category_keys)
        }
        score_rows.append({
            "path": list(prefix_paths[idx]),
            "rank_after": rank_after,
            "combined_mean_score": float(combined_scores[idx]),
            "query_score": float(query_scores[idx]),
            "category_scores": category_scores,
        })
    return reordered_paths, score_rows


def _select_categories_rule_a_simple(
    *,
    ordered: Sequence[str],
    support_by_key: Dict[str, float],
    margin_tau: float,
    bootstrap_keep_all: bool,
) -> Tuple[List[str], str, Dict[str, float], Dict[str, float]]:
    if not ordered:
        return [], "explore", {"name": "empty", "value": 0.0}, {}
    if bootstrap_keep_all:
        # Intent: iteration-0 keeps all categories to match the bootstrap behavior discussed in the plan.
        return list(ordered), "explore", {"name": "bootstrap", "value": 0.0}, {}
    s1 = float(support_by_key.get(ordered[0], 0.0))
    s2 = float(support_by_key.get(ordered[1], s1)) if len(ordered) > 1 else s1
    margin = float(s1 - s2)
    if margin >= float(margin_tau):
        # Intent: exploit uses fixed worst-one-drop to reduce category noise conservatively.
        selected = list(ordered[:-1]) if len(ordered) > 1 else list(ordered)
        return selected, "exploit", {"name": "margin", "value": margin}, {}
    return list(ordered), "explore", {"name": "margin", "value": margin}, {}


def _select_categories_rule_b_counterfactual(
    *,
    ordered: Sequence[str],
    support_by_key: Dict[str, float],
    sim: np.ndarray,
    drop_tau: float,
    bootstrap_keep_all: bool,
) -> Tuple[List[str], str, Dict[str, float], Dict[str, float]]:
    if not ordered:
        return [], "explore", {"name": "empty", "value": 0.0}, {}
    if bootstrap_keep_all:
        # Intent: iteration-0 keeps all categories to preserve first-step exploration coverage.
        return list(ordered), "explore", {"name": "bootstrap", "value": 0.0}, {}
    if len(ordered) <= 1:
        return list(ordered), "exploit", {"name": "min_drop_risk", "value": 0.0}, {}
    if sim.size == 0 or sim.shape[1] == 0:
        # Intent: with no support evidence, avoid arbitrary pruning and keep categories broad.
        return list(ordered), "explore", {"name": "min_drop_risk", "value": 1.0}, {key: 1.0 for key in ordered}

    full_util = float(np.mean(np.max(sim, axis=0)))
    denom = max(abs(full_util), 1e-6)
    risk_by_key: Dict[str, float] = {}
    ordered_list = list(ordered)
    order_idx = {key: idx for idx, key in enumerate(ordered_list)}
    for idx, key in enumerate(ordered_list):
        keep_idx = [i for i in range(len(ordered_list)) if i != idx]
        if not keep_idx:
            risk_by_key[key] = 1.0
            continue
        ablated_util = float(np.mean(np.max(sim[keep_idx, :], axis=0)))
        raw_risk = max(0.0, full_util - ablated_util)
        risk_by_key[key] = float(raw_risk / denom)

    drop_key = min(
        ordered_list,
        key=lambda key: (risk_by_key.get(key, 1.0), support_by_key.get(key, 0.0), order_idx.get(key, 10_000)),
    )
    min_risk = float(risk_by_key.get(drop_key, 1.0))
    if min_risk <= float(drop_tau):
        selected = [key for key in ordered_list if key != drop_key]
        return selected, "exploit", {"name": "min_drop_risk", "value": min_risk}, risk_by_key
    return ordered_list, "explore", {"name": "min_drop_risk", "value": min_risk}, risk_by_key


def _select_categories_round4_policy(
    *,
    docs: Dict[str, str],
    evidence_descs: Sequence[str],
    support_topm: int,
    rule_name: str,
    rule_a_margin_tau: float,
    rule_b_drop_tau: float,
    analysis_category_mode: str,
    bootstrap_keep_all: bool,
    retriever: DiverEmbeddingModel,
    emb_cache: Dict[str, np.ndarray],
) -> Tuple[List[str], Dict[str, float], str, Dict[str, float], Dict[str, float]]:
    ordered, support_by_key, sim = _compute_category_support(
        docs=docs,
        evidence_descs=evidence_descs,
        support_topm=support_topm,
        retriever=retriever,
        emb_cache=emb_cache,
    )
    if rule_name == "rule_c":
        top_key = ordered[0] if ordered else ""
        top_val = float(support_by_key.get(top_key, 0.0)) if top_key else 0.0
        selected = list(ordered)
        decision_mode = "rule_c"
        decision_signal = {"name": "top1_support", "value": top_val, "top_category": top_key}
        drop_risk_scores = {}
    elif rule_name == "rule_b":
        selected, decision_mode, decision_signal, drop_risk_scores = _select_categories_rule_b_counterfactual(
            ordered=ordered,
            support_by_key=support_by_key,
            sim=sim,
            drop_tau=rule_b_drop_tau,
            bootstrap_keep_all=bootstrap_keep_all,
        )
    else:
        selected, decision_mode, decision_signal, drop_risk_scores = _select_categories_rule_a_simple(
            ordered=ordered,
            support_by_key=support_by_key,
            margin_tau=rule_a_margin_tau,
            bootstrap_keep_all=bootstrap_keep_all,
        )
    if analysis_category_mode == "force_full":
        # Intent: analysis mode isolates the effect of no-pruning retrieval with the same rewrite outputs.
        selected = list(ordered)
        decision_mode = "explore"
        decision_signal = {"name": "analysis_force_full", "value": float(len(selected))}
    elif analysis_category_mode == "force_drop_one":
        # Intent: analysis mode isolates the effect of always dropping one category when possible.
        selected = list(ordered[:-1]) if len(ordered) > 1 else list(ordered)
        decision_mode = "exploit"
        decision_signal = {"name": "analysis_force_drop_one", "value": float(max(0, len(ordered) - len(selected)))}
    return selected, support_by_key, decision_mode, decision_signal, drop_risk_scores


def _build_actions_from_selected(
    docs: Dict[str, str],
    selected_categories: Sequence[str],
    decision_mode: str,
) -> Dict[str, str]:
    action_label = "EXPLORE" if str(decision_mode).lower() == "explore" else "EXPLOIT"
    selected = set(selected_categories)
    return {
        key: (action_label if key in selected else "PRUNE")
        for key in _ordered_doc_keys(docs)
        if str(docs.get(key, "")).strip()
    }


def _active_categories_from_docs_actions(
    docs: Dict[str, str],
    actions: Dict[str, str],
) -> List[str]:
    available = _ordered_doc_keys(docs)
    if not available:
        return []
    selected = [key for key in available if actions.get(key, "EXPLOIT") != "PRUNE"]
    return selected if selected else available


def _apply_previous_exploit_lock(
    docs: Dict[str, str],
    previous_docs: Dict[str, str],
    previous_actions: Dict[str, str],
    previous_decision_mode: str,
) -> Tuple[Dict[str, str], bool, List[str]]:
    clean_docs = {
        key: str(val).strip()
        for key, val in (docs or {}).items()
        if str(key or "").strip() and str(val or "").strip()
    }
    if str(previous_decision_mode or "").lower() != "exploit":
        return clean_docs, False, []
    locked_categories = _active_categories_from_docs_actions(previous_docs or {}, previous_actions or {})
    if not locked_categories:
        return clean_docs, False, []
    filtered_docs = {key: clean_docs[key] for key in locked_categories if key in clean_docs}
    if filtered_docs:
        # Intent: carry exploit decision across iterations by constraining next rewrite to prior active categories.
        return filtered_docs, True, locked_categories
    fallback_docs = {
        key: str((previous_docs or {}).get(key, "")).strip()
        for key in locked_categories
        if str((previous_docs or {}).get(key, "")).strip()
    }
    if fallback_docs:
        # Intent: keep exploit continuity even when the new rewrite omits previously locked categories.
        return fallback_docs, True, locked_categories
    return clean_docs, False, locked_categories


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
    leaf_ancestor_paths: Dict[Tuple[int, ...], List[Tuple[int, ...]]],
) -> Tuple[List[Tuple[int, ...]], Dict[Tuple[int, ...], float], Dict[Tuple[int, ...], float]]:
    leaf_prefix_counts: Dict[Tuple[int, ...], int] = {}
    for h in leaf_hits:
        for prefix in leaf_ancestor_paths.get(tuple(h.path), []):
            leaf_prefix_counts[prefix] = leaf_prefix_counts.get(prefix, 0) + 1
    branch_scores: Dict[Tuple[int, ...], float] = {}
    for h in branch_hits:
        branch_scores[h.path] = max(branch_scores.get(h.path, float("-inf")), h.score)

    active = set(branch_scores.keys()) | set(leaf_prefix_counts.keys())
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
    leaf_indices_by_prefix: Dict[Tuple[int, ...], List[int]],
) -> List[int]:
    prefix_set = set(prefixes)
    if not prefix_set:
        return []
    local_indices: List[int] = []
    seen: Set[int] = set()
    for prefix in prefix_set:
        for idx in leaf_indices_by_prefix.get(prefix, []):
            if idx in seen:
                continue
            seen.add(idx)
            local_indices.append(idx)
    return local_indices


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
        hits.append(FlatHit(registry_idx=registry_idx, path=tuple(node.path), score=float(score), is_leaf=node.is_leaf))
    return hits


def _anchor_ordered_local_hits(
    anchor_hits: Sequence[FlatHit],
    anchor_topk: int,
    scores_all: np.ndarray,
    node_registry: Sequence[object],
    leaf_indices_by_prefix: Dict[Tuple[int, ...], List[int]],
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


hp = HyperParams.from_args()
if not hp.REWRITE_PROMPT_NAME and not hp.REWRITE_PROMPT_PATH and not hp.REWRITE_CACHE_PATH:
    hp.add_param("rewrite_prompt_name", "round3_action_v1")
# Intent: round4 baseline scope supports only none/v2 anchor modes to keep code path auditable.
anchor_mode = str(hp.ROUND3_ANCHOR_LOCAL_RANK or "none").lower()
if anchor_mode not in {"none", "v2"}:
    raise ValueError(
        f'Unsupported --round3_anchor_local_rank="{hp.ROUND3_ANCHOR_LOCAL_RANK}" in run_round5.py. '
        'Allowed: none|v2'
    )
if bool(hp.ROUND3_REWRITE_USE_HISTORY):
    print("Ignoring --round3_rewrite_use_history in run_round5.py (fixed to False).")
if hp.ROUND3_ROUTER_PROMPT_NAME:
    print("Ignoring --round3_router_prompt_name in run_round5.py (router is disabled).")
if str(hp.ROUND3_SUMMARIZED_CONTEXT or "off").lower() != "off":
    print("Ignoring --round3_summarized_context in run_round5.py (fixed to off).")
hp.add_param("round3_anchor_local_rank", anchor_mode)
hp.add_param("round3_rewrite_use_history", False)
hp.add_param("round3_router_prompt_name", None)
hp.add_param("round3_summarized_context", "off")
round4_rule_name = str(getattr(hp, "ROUND4_RULE_NAME", "rule_a") or "rule_a").lower()
if round4_rule_name not in {"rule_a", "rule_b", "rule_c"}:
    raise ValueError(f'Unsupported --round4_rule_name="{round4_rule_name}". Allowed: rule_a|rule_b|rule_c')
round4_support_topm = max(1, int(getattr(hp, "ROUND4_SUPPORT_TOPM", 10) or 10))
round4_rule_a_margin_tau = float(getattr(hp, "ROUND4_RULE_A_MARGIN_TAU", 0.02) or 0.02)
round4_rule_b_drop_tau = float(getattr(hp, "ROUND4_RULE_B_DROP_TAU", 0.01) or 0.01)
round4_analysis_category_mode = str(getattr(hp, "ROUND4_ANALYSIS_CATEGORY_MODE", "default") or "default").lower()
if round4_analysis_category_mode not in {"default", "force_full", "force_drop_one"}:
    raise ValueError(
        f'Unsupported --round4_analysis_category_mode="{round4_analysis_category_mode}". '
        'Allowed: default|force_full|force_drop_one'
    )
category_fusion = str(getattr(hp, "CATEGORY_FUSION", "off") or "off").lower()
if category_fusion not in {"off", "category_query_mean"}:
    raise ValueError(
        f'Unsupported --category_fusion="{category_fusion}". '
        'Allowed: off|category_query_mean'
    )
hp.add_param("round4_rule_name", round4_rule_name)
hp.add_param("round4_support_topm", round4_support_topm)
hp.add_param("round4_rule_a_margin_tau", round4_rule_a_margin_tau)
hp.add_param("round4_rule_b_drop_tau", round4_rule_b_drop_tau)
hp.add_param("round4_analysis_category_mode", round4_analysis_category_mode)
hp.add_param("category_fusion", category_fusion)
round4_iter0_prompt_name = str(getattr(hp, "ROUND4_ITER0_PROMPT_NAME", "") or "").strip()
if round4_iter0_prompt_name and round4_iter0_prompt_name not in REWRITE_PROMPT_TEMPLATES:
    raise ValueError(
        f'Unsupported --round4_iter0_prompt_name="{round4_iter0_prompt_name}". '
        f"Available: {sorted(REWRITE_PROMPT_TEMPLATES.keys())}"
    )
hp.add_param("round4_iter0_prompt_name", round4_iter0_prompt_name or None)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
corpus_categories_hint = _build_corpus_categories_hint(BASE_DIR, hp.DATASET, hp.SUBSET)
exp_dir_name = str(hp)
# Intent: isolate round5 artifacts under a dedicated directory so DAG-native runs do not overwrite round4 baselines.
RESULTS_DIR = f"{BASE_DIR}/results/{hp.DATASET}/{hp.SUBSET}/round5/{exp_dir_name}/"
if os.path.exists(RESULTS_DIR) and os.listdir(RESULTS_DIR):
    print(f"Results already exist at {RESULTS_DIR}. Skipping run.")
    raise SystemExit(0)
os.makedirs(RESULTS_DIR, exist_ok=True)
log_path = f"{RESULTS_DIR}/run.log"
os.makedirs(os.path.dirname(log_path), exist_ok=True)
logger = setup_logger("lattice_runner_round5", log_path, logging.INFO)
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

query_source = str(hp.QUERY_SOURCE or "original").lower()
if query_source not in {"original", "gpt4"}:
    raise ValueError(f"Unknown --query_source '{hp.QUERY_SOURCE}'. Expected: original|gpt4")

if os.path.exists(f"{BASE_DIR}/data/{hp.DATASET}/{hp.SUBSET}/documents.jsonl"):
    logger.info(f"Loading dataset {hp.DATASET} split={hp.SUBSET} from local JSONL files")
    docs_df = pd.read_json(f"{BASE_DIR}/data/{hp.DATASET}/{hp.SUBSET}/documents.jsonl", lines=True, dtype={"id": str})
    local_examples_name = "gpt4_reason" if query_source == "gpt4" else "examples"
    local_examples_path = f"{BASE_DIR}/data/{hp.DATASET}/{hp.SUBSET}/{local_examples_name}.jsonl"
    if not os.path.exists(local_examples_path):
        raise FileNotFoundError(
            f"Requested --query_source={query_source}, but local file not found: {local_examples_path}"
        )
    examples_df = pd.read_json(local_examples_path, lines=True, dtype={"gold_ids": List[str]})
    examples_df["gold_ids"] = examples_df["gold_ids"].apply(lambda x: [str(i) for i in x])
else:
    logger.info(f"Loading dataset xlangai/BRIGHT split={hp.SUBSET} from HuggingFace Datasets")
    docs_df = pd.DataFrame(load_dataset("xlangai/BRIGHT", "documents", split=hp.SUBSET))
    examples_config = "gpt4_reason" if query_source == "gpt4" else "examples"
    examples_df = pd.DataFrame(load_dataset("xlangai/BRIGHT", examples_config, split=hp.SUBSET))


doc_id_to_content = {docs_df.iloc[i].id: docs_df.iloc[i].content for i in range(len(docs_df))}

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
    node_by_path = dag_runtime["node_by_path"]
    path_to_registry_idx = dag_runtime["path_to_registry_idx"]
    leaf_indices = dag_runtime["leaf_indices"]
    leaf_paths = dag_runtime["leaf_paths"]
    leaf_indices_by_prefix = dag_runtime["leaf_indices_by_prefix"]
    leaf_ancestor_paths = dag_runtime["leaf_ancestor_paths"]
else:
    tree_dict = pkl.load(open(f"{BASE_DIR}/trees/{hp.DATASET}/{hp.SUBSET}/tree-{hp.TREE_VERSION}.pkl", "rb"))
    semantic_root_node = SemanticNode().load_dict(tree_dict) if isinstance(tree_dict, dict) else tree_dict
    node_registry = compute_node_registry(semantic_root_node)
    node_by_path = {tuple(node.path): node for node in node_registry}
    path_to_registry_idx = {tuple(node.path): int(idx) for idx, node in enumerate(node_registry)}
    leaf_indices, leaf_paths, leaf_indices_by_prefix, leaf_ancestor_paths = _build_tree_leaf_support_maps(node_registry)

doc_id_to_paths: Dict[str, List[Tuple[int, ...]]] = defaultdict(list)
for leaf_idx in leaf_indices:
    node = node_registry[int(leaf_idx)]
    doc_id = get_node_id(node.id, docs_df)
    if not doc_id:
        continue
    doc_id_to_paths[str(doc_id)].append(tuple(node.path))
for doc_id in list(doc_id_to_paths.keys()):
    seen_paths: Set[Tuple[int, ...]] = set()
    deduped: List[Tuple[int, ...]] = []
    for path in doc_id_to_paths[doc_id]:
        if path in seen_paths:
            continue
        seen_paths.add(path)
        deduped.append(path)
    doc_id_to_paths[doc_id] = deduped
path_to_doc_id: Dict[Tuple[int, ...], str] = {}
for doc_id, paths in doc_id_to_paths.items():
    for path in paths:
        path_to_doc_id[tuple(path)] = str(doc_id)

if not hp.RETRIEVER_MODEL_PATH:
    raise ValueError("--retriever_model_path is required")
retriever = DiverEmbeddingModel(hp.RETRIEVER_MODEL_PATH, local_files_only=True)
node_embs: Optional[np.ndarray] = None
if hp.NODE_EMB_PATH:
    if not os.path.exists(hp.NODE_EMB_PATH):
        logger.warning("node_emb_path not found: %s; fallback to on-the-fly encoding", hp.NODE_EMB_PATH)
    else:
        loaded = np.load(hp.NODE_EMB_PATH, allow_pickle=False)
        if loaded.shape[0] == len(node_registry):
            node_embs = normalize_embeddings(loaded)
        else:
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

category_emb_cache: Dict[str, np.ndarray] = {}

rewrite_enabled = True
rewrite_template = None
rewrite_template_name = ""
rewrite_iter0_template = None
rewrite_map: Dict[str, str] = {}
action_map: Dict[str, str] = {}
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
    rewrite_template_name = str(hp.REWRITE_PROMPT_NAME)
if hp.REWRITE_PROMPT_PATH:
    if not os.path.exists(hp.REWRITE_PROMPT_PATH):
        raise ValueError(f"--rewrite_prompt_path not found: {hp.REWRITE_PROMPT_PATH}")
    with open(hp.REWRITE_PROMPT_PATH, "r", encoding="utf-8") as f:
        rewrite_template = f.read()
    rewrite_template_name = f"path:{os.path.basename(hp.REWRITE_PROMPT_PATH)}"
rewrite_map, action_map, docs_map = _load_rewrite_action_cache(hp.REWRITE_CACHE_PATH, hp.REWRITE_FORCE_REFRESH)

if rewrite_template is None:
    rewrite_template = REWRITE_PROMPT_TEMPLATES["round3_action_v1"]
    rewrite_template_name = "round3_action_v1"
if round4_iter0_prompt_name:
    rewrite_iter0_template = REWRITE_PROMPT_TEMPLATES[round4_iter0_prompt_name]

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
    # Keep rewrite/controller generation deterministic for reproducible round4 analysis runs.
    "temperature": 0.0,
}

num_samples = min(examples_df.shape[0], hp.NUM_EVAL_SAMPLES)
all_eval_samples: List[Round3Sample] = []
for i in range(num_samples):
    raw_gold_ids = [str(x) for x in examples_df.iloc[i]["gold_ids"]]
    gold_doc_ids = [doc_id for doc_id in raw_gold_ids if doc_id in doc_id_to_paths]
    gold_paths: List[Tuple[int, ...]] = []
    for doc_id in gold_doc_ids:
        gold_paths.extend(doc_id_to_paths.get(doc_id, []))
    dedup_gold_paths: List[Tuple[int, ...]] = []
    seen_gold_paths: Set[Tuple[int, ...]] = set()
    for path in gold_paths:
        path_t = tuple(path)
        if path_t in seen_gold_paths:
            continue
        seen_gold_paths.add(path_t)
        dedup_gold_paths.append(path_t)
    if len(gold_doc_ids) < len(raw_gold_ids):
        logger.warning(f"Some gold IDs for example {i} not found in document paths.")
    sample = Round3Sample(
        original_query=examples_df.iloc[i]['query'][: hp.MAX_QUERY_CHAR_LEN],
        gold_paths=dedup_gold_paths,
        gold_doc_ids=gold_doc_ids,
        excluded_ids=list(examples_df.iloc[i]["excluded_ids"]),
    )
    all_eval_samples.append(sample)

anchor_topk = hp.ROUND3_ANCHOR_TOPK or hp.FLAT_TOPK
local_topk = hp.ROUND3_LOCAL_TOPK or hp.FLAT_TOPK
global_topk = hp.ROUND3_GLOBAL_TOPK

all_eval_metric_dfs: List[pd.DataFrame] = []
for iter_idx in range(hp.NUM_ITERS):
    logger.info("Round5 iteration %d", iter_idx)
    iter_rewrite_template = rewrite_template
    iter_rewrite_prompt_name = rewrite_template_name
    if iter_idx == 0 and rewrite_iter0_template is not None:
        # Intent: bootstrap iter 0 with broader prompt, then switch to corpus-grounded prompt from iter>=1.
        iter_rewrite_template = rewrite_iter0_template
        iter_rewrite_prompt_name = round4_iter0_prompt_name
    logger.info("Iter %d: rewrite prompt template = %s", iter_idx, iter_rewrite_prompt_name)
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
    retrieval_query_categories_by_sample: List[List[str]] = []
    retrieval_category_support_scores_by_sample: List[Dict[str, float]] = []
    retrieval_category_decision_mode_by_sample: List[str] = []
    retrieval_category_decision_signal_by_sample: List[Dict[str, float]] = []
    retrieval_category_drop_risk_scores_by_sample: List[Dict[str, float]] = []
    retrieval_category_lock_applied_by_sample: List[bool] = []
    retrieval_category_lock_source_by_sample: List[List[str]] = []
    history_summaries_by_sample: List[List[str]] = [[] for _ in all_eval_samples]

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
        #   - Category-policy mode → _compose_query_from_docs_with_policy is used once category docs are available.
        if sample.last_possible_docs:
            # Intent: snapshot the exact query-control state used for this retrieval before rewrite updates it.
            query_action_t = "multi"
            query_actions_t = dict(sample.last_actions or {})
            query_docs_t = dict(sample.last_possible_docs or {})
            query_support_scores_t = {str(k): float(v) for k, v in (sample.last_category_support_scores or {}).items()}
            query_decision_mode_t = str(sample.last_category_decision_mode or "")
            query_decision_signal_t = dict(sample.last_category_decision_signal or {})
            query_drop_risk_scores_t = {str(k): float(v) for k, v in (sample.last_category_drop_risk_scores or {}).items()}
            query_lock_applied_t = bool(sample.last_category_lock_applied)
            query_lock_source_t = list(sample.last_category_lock_source or [])
            anchor_query, query_categories_t = _compose_query_from_docs_with_policy(
                sample.original_query,
                sample.last_possible_docs,
                sample.last_actions,
                str(hp.ROUND3_CATEGORY_POLICY or "none").lower(),
                hp.ROUND3_CATEGORY_SOFT_KEEP,
            )
        else:
            query_action_t = str(sample.last_action or "exploit").strip().lower()
            query_actions_t = {}
            query_docs_t = {}
            query_categories_t = []
            query_support_scores_t = {}
            query_decision_mode_t = ""
            query_decision_signal_t = {}
            query_drop_risk_scores_t = {}
            query_lock_applied_t = False
            query_lock_source_t = []
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
        if hp.ROUND3_ANCHOR_LOCAL_RANK != "none":
            anchor_eval_paths = _anchor_local_context_paths(
                anchor_hits=hits,
                anchor_topk=anchor_topk,
                scores_all=scores_all,
                node_registry=node_registry,
                leaf_indices_by_prefix=leaf_indices_by_prefix,
                topk=anchor_topk,
            )
        else:
            # Intent: reproduce round3 none-baseline anchor evaluation (leaf-only from anchor ranking).
            anchor_eval_paths = [h.path for h in hits if h.is_leaf][:anchor_topk]

        if category_fusion == "category_query_mean" and query_docs_t:
            reordered_anchor_paths, reorder_rows = _reorder_anchor_topk_with_category_and_query_mean(
                anchor_paths=anchor_eval_paths,
                query_docs=query_docs_t,
                query_categories=query_categories_t,
                node_by_path=node_by_path,
                path_to_registry_idx=path_to_registry_idx,
                scores_all=scores_all,
                topk=10,
                max_desc_len=hp.MAX_DOC_DESC_CHAR_LEN,
                retriever=retriever,
                emb_cache=category_emb_cache,
            )
            if reorder_rows:
                # Intent: category_fusion reranks anchor top-10 by averaging per-category support and base query score.
                anchor_eval_paths = reordered_anchor_paths
                query_decision_signal_t = dict(query_decision_signal_t or {})
                query_decision_signal_t["top10_reorder"] = {
                    "name": "category_plus_query_mean",
                    "k": 10,
                    "rows": reorder_rows,
                }

        anchor_eval_paths_by_sample.append(anchor_eval_paths)
        leaf_hits = [h for h in hits if h.is_leaf]
        branch_hits = [h for h in hits if not h.is_leaf]
        leaf_hits_by_sample.append(leaf_hits)
        branch_hits_by_sample.append(branch_hits)

        active_branches, densities, branch_scores = _build_active_branches(
            leaf_hits,
            branch_hits,
            leaf_ancestor_paths,
        )
        ranked_branches = _rank_branch_paths(active_branches, densities, branch_scores)
        branch_paths_by_sample.append(ranked_branches)
        densities_by_sample.append(densities)
        retrieval_queries_by_sample.append(anchor_query)
        retrieval_query_actions_by_sample.append(query_action_t)
        retrieval_query_actions_map_by_sample.append(query_actions_t)
        retrieval_query_docs_by_sample.append(query_docs_t)
        retrieval_query_categories_by_sample.append(query_categories_t)
        retrieval_category_support_scores_by_sample.append(query_support_scores_t)
        retrieval_category_decision_mode_by_sample.append(query_decision_mode_t)
        retrieval_category_decision_signal_by_sample.append(query_decision_signal_t)
        retrieval_category_drop_risk_scores_by_sample.append(query_drop_risk_scores_t)
        retrieval_category_lock_applied_by_sample.append(query_lock_applied_t)
        retrieval_category_lock_source_by_sample.append(query_lock_source_t)

        local_leaf_indices = _filter_leaf_indices_by_prefixes(
            leaf_indices,
            leaf_paths,
            ranked_branches,
            leaf_indices_by_prefix,
        )
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
                fallback_hits=local_hits,
                local_topk=local_topk,
            )
        elif anchor_topk_logged > 0:
            # Intent: reproduce round3 none-baseline local expansion by adding one best descendant per anchor branch.
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
            if hp.ROUND3_ANCHOR_LOCAL_RANK == "v2":
                # Intent: v2 rewrite context follows anchor branch->best-leaf replacement.
                context_paths = _anchor_local_context_paths(
                    anchor_hits=hits,
                    anchor_topk=anchor_topk,
                    scores_all=scores_all,
                    node_registry=node_registry,
                    leaf_indices_by_prefix=leaf_indices_by_prefix,
                    topk=hp.REWRITE_CONTEXT_TOPK,
                )
                leaf_descs = _paths_to_context_descs(
                    context_paths,
                    node_by_path,
                    hp.REWRITE_CONTEXT_TOPK,
                    hp.MAX_DOC_DESC_CHAR_LEN,
                )
            elif category_fusion == "category_query_mean":
                # Intent: when fusion is enabled, next-iteration rewrite context follows fused anchor_eval_paths order.
                leaf_descs = _paths_to_context_descs(
                    anchor_eval_paths,
                    node_by_path,
                    hp.REWRITE_CONTEXT_TOPK,
                    hp.MAX_DOC_DESC_CHAR_LEN,
                )
            else:
                # Intent: none-baseline rewrite context uses direct anchor leaf evidence.
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

            rewrite_candidates.append({
                "sample_idx": sample_idx,
                "sample": sample,
                "leaf_descs": leaf_descs,
                "branch_descs": branch_descs,
                "support_leaf_descs": _hits_to_context_descs(
                    leaf_hits,
                    node_registry,
                    hp.ROUND3_CATEGORY_SUPPORT_TOPK,
                    hp.MAX_DOC_DESC_CHAR_LEN,
                ),
                "retrieval_history": "",
            })

    if rewrite_candidates:
        # Intent: run_round4 keeps summarized context disabled to avoid extra summary-generation complexity.
        for meta in rewrite_candidates:
            sample_idx = int(meta.get("sample_idx", -1))
            if 0 <= sample_idx < len(history_summaries_by_sample):
                history_summaries_by_sample[sample_idx] = []

    if rewrite_candidates:
        rewrite_prompts: List[str] = []
        rewrite_meta: List[Dict] = []
        category_policy_mode = str(hp.ROUND3_CATEGORY_POLICY or "none").lower()
        for meta in rewrite_candidates:
            sample = meta["sample"]
            leaf_descs = meta["leaf_descs"]
            branch_descs = meta["branch_descs"] if hp.ROUND3_REWRITE_CONTEXT == "leaf_branch" else []
            retrieval_history = meta.get("retrieval_history", "")
            prompt = _format_action_prompt(
                iter_rewrite_template,
                sample.original_query,
                sample.last_rewrite,
                sample.last_possible_docs,
                leaf_descs,
                branch_descs,
                subset_name=hp.SUBSET,
                retrieval_history=retrieval_history,
                corpus_categories=corpus_categories_hint,
            )
            cache_key = _prompt_cache_key("round3", prompt)
            if (not hp.REWRITE_FORCE_REFRESH) and (cache_key in rewrite_map):
                rewrite = rewrite_map[cache_key]
                category_scores: Dict[str, float] = {}
                support_scores: Dict[str, float] = {}
                drop_risk_scores: Dict[str, float] = {}
                decision_mode = ""
                decision_signal: Dict[str, float] = {}
                selected_categories: List[str] = []
                lock_applied = False
                lock_source: List[str] = []
                prev_selected_categories: List[str] = _active_categories_from_docs_actions(
                    sample.last_possible_docs or {},
                    sample.last_actions or {},
                )
                cached_actions = action_map.get(cache_key)
                cached_docs = docs_map.get(cache_key, {})
                if category_policy_mode == "soft" and cached_docs:
                    policy_docs = _clean_docs_map(cached_docs)
                    if round4_rule_name == "rule_c":
                        _, current_support_scores, _ = _compute_category_support(
                            docs=policy_docs,
                            evidence_descs=meta.get("support_leaf_descs", []),
                            support_topm=round4_support_topm,
                            retriever=retriever,
                            emb_cache=category_emb_cache,
                        )
                        merged_docs, merged_support, updated_categories = _merge_best_docs_by_support(
                            previous_best_docs=sample.last_possible_docs or {},
                            previous_best_support=sample.last_category_support_scores or {},
                            candidate_docs=policy_docs,
                            candidate_support=current_support_scores,
                        )
                        selected_categories = _ordered_doc_keys(merged_docs)
                        support_scores = dict(merged_support)
                        category_scores = dict(support_scores)
                        decision_mode = "rule_c"
                        top_category = max(current_support_scores, key=current_support_scores.get) if current_support_scores else ""
                        decision_signal = {
                            "name": "rule_c_best_concat",
                            "top_category": str(top_category),
                            "top_support": float(current_support_scores.get(top_category, 0.0)) if top_category else 0.0,
                            "updated_categories": list(updated_categories),
                        }
                        drop_risk_scores = {}
                        lock_applied = False
                        lock_source = []
                        # Intent: rule_c retrieval always concatenates per-category best rewrites across all categories.
                        sample.last_possible_docs = dict(merged_docs)
                        sample.last_actions = _build_actions_keep_all(sample.last_possible_docs)
                        sample.last_rewrite = _flatten_docs_by_action(sample.last_possible_docs, sample.last_actions)
                        rewrite = sample.last_rewrite
                        sample.last_action = "exploit"
                        sample.last_category_support_scores = dict(merged_support)
                        sample.last_category_decision_mode = decision_mode
                        sample.last_category_decision_signal = dict(decision_signal)
                        sample.last_category_drop_risk_scores = {}
                        sample.last_category_lock_applied = False
                        sample.last_category_lock_source = []
                    else:
                        if round4_analysis_category_mode == "default":
                            policy_docs, lock_applied, lock_source = _apply_previous_exploit_lock(
                                policy_docs,
                                sample.last_possible_docs or {},
                                sample.last_actions or {},
                                sample.last_category_decision_mode,
                            )
                        else:
                            # Intent: analysis overrides compare category-drop effects without controller lock confounds.
                            lock_applied = False
                            lock_source = []
                        selected_categories, support_scores, decision_mode, decision_signal, drop_risk_scores = _select_categories_round4_policy(
                            docs=policy_docs,
                            evidence_descs=meta.get("support_leaf_descs", []),
                            support_topm=round4_support_topm,
                            rule_name=round4_rule_name,
                            rule_a_margin_tau=round4_rule_a_margin_tau,
                            rule_b_drop_tau=round4_rule_b_drop_tau,
                            analysis_category_mode=round4_analysis_category_mode,
                            bootstrap_keep_all=(iter_idx == 0),
                            retriever=retriever,
                            emb_cache=category_emb_cache,
                        )
                        category_scores = dict(support_scores)
                        sample.last_possible_docs = dict(cached_docs)
                        sample.last_actions = _build_actions_from_selected(
                            cached_docs,
                            selected_categories,
                            decision_mode,
                        )
                        # Intent: policy-aware query uses only selected categories while preserving original category texts.
                        sample.last_rewrite = _flatten_docs_by_action(sample.last_possible_docs, sample.last_actions)
                        rewrite = sample.last_rewrite
                        sample.last_action = decision_mode
                        sample.last_category_support_scores = dict(support_scores)
                        sample.last_category_decision_mode = decision_mode
                        sample.last_category_decision_signal = dict(decision_signal)
                        sample.last_category_drop_risk_scores = dict(drop_risk_scores)
                        sample.last_category_lock_applied = bool(lock_applied)
                        sample.last_category_lock_source = list(lock_source)
                elif isinstance(cached_actions, dict):
                    sample.last_actions = cached_actions
                    sample.last_possible_docs = cached_docs
                    sample.last_rewrite = rewrite
                    sample.last_action = "exploit"
                    sample.last_category_support_scores = {}
                    sample.last_category_decision_mode = ""
                    sample.last_category_decision_signal = {}
                    sample.last_category_drop_risk_scores = {}
                    sample.last_category_lock_applied = False
                    sample.last_category_lock_source = []
                else:
                    action = str(cached_actions or "exploit").strip().lower()
                    if action == "hold" and sample.last_rewrite:
                        rewrite = sample.last_rewrite
                    else:
                        sample.last_rewrite = rewrite
                    sample.last_action = action
                    sample.last_actions = {}
                    sample.last_possible_docs = {}
                    sample.last_category_support_scores = {}
                    sample.last_category_decision_mode = ""
                    sample.last_category_decision_signal = {}
                    sample.last_category_drop_risk_scores = {}
                    sample.last_category_lock_applied = False
                    sample.last_category_lock_source = []
                sample.rewrite_history.append({
                    "iter": iter_idx,
                    "cache_hit": True,
                    "prompt_name": iter_rewrite_prompt_name,
                    "action": sample.last_action,
                    "actions": sample.last_actions,
                    "possible_docs": sample.last_possible_docs,
                    "rewrite": rewrite,
                    "prev_selected_categories": prev_selected_categories if (category_policy_mode == "soft" and cached_docs) else [],
                    "selected_categories": selected_categories if (category_policy_mode == "soft" and cached_docs) else [],
                    "category_scores": category_scores if (category_policy_mode == "soft" and cached_docs) else {},
                    "category_support_scores": support_scores if (category_policy_mode == "soft" and cached_docs) else {},
                    "category_penalty_scores": drop_risk_scores if (category_policy_mode == "soft" and cached_docs) else {},
                    "category_drop_risk_scores": drop_risk_scores if (category_policy_mode == "soft" and cached_docs) else {},
                    "category_decision_mode": decision_mode if (category_policy_mode == "soft" and cached_docs) else "",
                    "category_decision_signal": decision_signal if (category_policy_mode == "soft" and cached_docs) else {},
                    "category_lock_applied": lock_applied if (category_policy_mode == "soft" and cached_docs) else False,
                    "category_lock_source": lock_source if (category_policy_mode == "soft" and cached_docs) else [],
                })
            else:
                rewrite_prompts.append(prompt)
                rewrite_meta.append({
                    "sample": sample,
                    "cache_key": cache_key,
                    "prompt_name": iter_rewrite_prompt_name,
                    "leaf_descs": leaf_descs,
                    "branch_descs": branch_descs,
                    "support_leaf_descs": meta.get("support_leaf_descs", []),
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
                category_scores: Dict[str, float] = {}
                support_scores: Dict[str, float] = {}
                drop_risk_scores: Dict[str, float] = {}
                decision_mode = ""
                decision_signal: Dict[str, float] = {}
                selected_categories: List[str] = []
                lock_applied = False
                lock_source: List[str] = []
                prev_selected_categories: List[str] = _active_categories_from_docs_actions(
                    sample.last_possible_docs or {},
                    sample.last_actions or {},
                )
                output_docs = docs or {}
                if category_policy_mode == "soft" and output_docs:
                    policy_docs = _clean_docs_map(output_docs)
                    if round4_rule_name == "rule_c":
                        _, current_support_scores, _ = _compute_category_support(
                            docs=policy_docs,
                            evidence_descs=meta.get("support_leaf_descs", []),
                            support_topm=round4_support_topm,
                            retriever=retriever,
                            emb_cache=category_emb_cache,
                        )
                        merged_docs, merged_support, updated_categories = _merge_best_docs_by_support(
                            previous_best_docs=sample.last_possible_docs or {},
                            previous_best_support=sample.last_category_support_scores or {},
                            candidate_docs=policy_docs,
                            candidate_support=current_support_scores,
                        )
                        selected_categories = _ordered_doc_keys(merged_docs)
                        support_scores = dict(merged_support)
                        category_scores = dict(support_scores)
                        decision_mode = "rule_c"
                        top_category = max(current_support_scores, key=current_support_scores.get) if current_support_scores else ""
                        decision_signal = {
                            "name": "rule_c_best_concat",
                            "top_category": str(top_category),
                            "top_support": float(current_support_scores.get(top_category, 0.0)) if top_category else 0.0,
                            "updated_categories": list(updated_categories),
                        }
                        drop_risk_scores = {}
                        lock_applied = False
                        lock_source = []
                        # Intent: rule_c retrieval always concatenates per-category best rewrites across all categories.
                        sample.last_possible_docs = dict(merged_docs)
                        sample.last_actions = _build_actions_keep_all(sample.last_possible_docs)
                        sample.last_rewrite = _flatten_docs_by_action(sample.last_possible_docs, sample.last_actions)
                        rewrite = sample.last_rewrite
                        sample.last_action = "exploit"
                        sample.last_category_support_scores = dict(merged_support)
                        sample.last_category_decision_mode = decision_mode
                        sample.last_category_decision_signal = dict(decision_signal)
                        sample.last_category_drop_risk_scores = {}
                        sample.last_category_lock_applied = False
                        sample.last_category_lock_source = []
                    else:
                        if round4_analysis_category_mode == "default":
                            policy_docs, lock_applied, lock_source = _apply_previous_exploit_lock(
                                policy_docs,
                                sample.last_possible_docs or {},
                                sample.last_actions or {},
                                sample.last_category_decision_mode,
                            )
                        else:
                            # Intent: analysis overrides compare category-drop effects without controller lock confounds.
                            lock_applied = False
                            lock_source = []
                        selected_categories, support_scores, decision_mode, decision_signal, drop_risk_scores = _select_categories_round4_policy(
                            docs=policy_docs,
                            evidence_descs=meta.get("support_leaf_descs", []),
                            support_topm=round4_support_topm,
                            rule_name=round4_rule_name,
                            rule_a_margin_tau=round4_rule_a_margin_tau,
                            rule_b_drop_tau=round4_rule_b_drop_tau,
                            analysis_category_mode=round4_analysis_category_mode,
                            bootstrap_keep_all=(iter_idx == 0),
                            retriever=retriever,
                            emb_cache=category_emb_cache,
                        )
                        category_scores = dict(support_scores)
                        sample.last_possible_docs = dict(output_docs)
                        sample.last_actions = _build_actions_from_selected(
                            output_docs,
                            selected_categories,
                            decision_mode,
                        )
                        # Intent: policy-aware query uses only selected categories while preserving original category texts.
                        sample.last_rewrite = _flatten_docs_by_action(sample.last_possible_docs, sample.last_actions)
                        rewrite = sample.last_rewrite
                        sample.last_action = decision_mode
                        sample.last_category_support_scores = dict(support_scores)
                        sample.last_category_decision_mode = decision_mode
                        sample.last_category_decision_signal = dict(decision_signal)
                        sample.last_category_drop_risk_scores = dict(drop_risk_scores)
                        sample.last_category_lock_applied = bool(lock_applied)
                        sample.last_category_lock_source = list(lock_source)
                elif actions:
                    sample.last_actions = actions
                    sample.last_possible_docs = output_docs
                    sample.last_rewrite = _flatten_docs_by_action(sample.last_possible_docs, sample.last_actions)
                    sample.last_action = "exploit"
                    rewrite = sample.last_rewrite
                    sample.last_category_support_scores = {}
                    sample.last_category_decision_mode = ""
                    sample.last_category_decision_signal = {}
                    sample.last_category_drop_risk_scores = {}
                    sample.last_category_lock_applied = False
                    sample.last_category_lock_source = []
                else:
                    if action == "hold" and sample.last_rewrite:
                        rewrite = sample.last_rewrite
                    else:
                        sample.last_rewrite = rewrite
                    sample.last_action = action
                    sample.last_actions = {}
                    sample.last_possible_docs = {}
                    sample.last_category_support_scores = {}
                    sample.last_category_decision_mode = ""
                    sample.last_category_decision_signal = {}
                    sample.last_category_drop_risk_scores = {}
                    sample.last_category_lock_applied = False
                    sample.last_category_lock_source = []
                sample.rewrite_history.append({
                    "iter": iter_idx,
                    "cache_hit": False,
                    "prompt_name": meta.get("prompt_name", iter_rewrite_prompt_name),
                    "action": sample.last_action,
                    "actions": sample.last_actions,
                    "possible_docs": sample.last_possible_docs,
                    "rewrite": rewrite,
                    "prev_selected_categories": prev_selected_categories,
                    "selected_categories": selected_categories,
                    "category_scores": category_scores,
                    "category_support_scores": support_scores,
                    "category_penalty_scores": drop_risk_scores,
                    "category_drop_risk_scores": drop_risk_scores,
                    "category_decision_mode": decision_mode,
                    "category_decision_signal": decision_signal,
                    "category_lock_applied": lock_applied,
                    "category_lock_source": lock_source,
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
                    "prompt_name": meta.get("prompt_name", iter_rewrite_prompt_name),
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
        anchor_branch_hits.append(
            1.0 if _gate_hit_dag(branches, sample.gold_paths, leaf_ancestor_paths) else 0.0
        )
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
    for sample_idx, (
        sample,
        anchor_eval_paths,
        leaf_hits,
        branch_hits,
        local_hits,
        global_hits,
        active_paths,
        densities,
        query_t,
        query_action_t,
        query_actions_t,
        query_docs_t,
        query_categories_t,
        query_support_scores_t,
        query_decision_mode_t,
        query_decision_signal_t,
        query_drop_risk_scores_t,
        query_lock_applied_t,
        query_lock_source_t,
    ) in enumerate(tqdm(
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
            retrieval_query_categories_by_sample,
            retrieval_category_support_scores_by_sample,
            retrieval_category_decision_mode_by_sample,
            retrieval_category_decision_signal_by_sample,
            retrieval_category_drop_risk_scores_by_sample,
            retrieval_category_lock_applied_by_sample,
            retrieval_category_lock_source_by_sample,
        ),
        desc=f"Iter {iter_idx} local/global scoring",
        total=len(all_eval_samples),
        leave=False,
    )):
        local_paths = [tuple(h.path) for h in local_hits]
        global_paths = [tuple(h.path) for h in global_hits]
        anchor_paths = [tuple(p) for p in anchor_eval_paths]
        local_doc_ids = _paths_to_ranked_doc_ids(local_paths, path_to_doc_id)
        global_doc_ids = _paths_to_ranked_doc_ids(global_paths, path_to_doc_id)
        anchor_doc_ids = _paths_to_ranked_doc_ids(anchor_paths, path_to_doc_id)
        gold_doc_ids = [str(x) for x in sample.gold_doc_ids]
        local_metrics = {
            "nDCG@10": compute_ndcg(local_doc_ids[:10], gold_doc_ids, k=10) * 100,
            "Recall@10": compute_recall(local_doc_ids[:10], gold_doc_ids, k=10) * 100,
            "Recall@100": compute_recall(local_doc_ids[:100], gold_doc_ids, k=100) * 100,
            "Recall@all": compute_recall(local_doc_ids, gold_doc_ids, k=len(local_doc_ids)) * 100,
            "Coverage": len(local_doc_ids),
        }
        global_metrics = {
            "nDCG@10": compute_ndcg(global_doc_ids[:10], gold_doc_ids, k=10) * 100,
            "Recall@10": compute_recall(global_doc_ids[:10], gold_doc_ids, k=10) * 100,
            "Recall@100": compute_recall(global_doc_ids[:100], gold_doc_ids, k=100) * 100,
            "Recall@all": compute_recall(global_doc_ids, gold_doc_ids, k=len(global_doc_ids)) * 100,
            "Coverage": len(global_doc_ids),
        }
        anchor_metrics = {
            "nDCG@10": compute_ndcg(anchor_doc_ids[:10], gold_doc_ids, k=10) * 100,
            "Recall@10": compute_recall(anchor_doc_ids[:10], gold_doc_ids, k=10) * 100,
            "Recall@100": compute_recall(anchor_doc_ids[:100], gold_doc_ids, k=100) * 100,
            "Recall@all": compute_recall(anchor_doc_ids, gold_doc_ids, k=len(anchor_doc_ids)) * 100,
            "Coverage": len(anchor_doc_ids),
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
            "query_categories": query_categories_t,
            "query_category_support_scores": query_support_scores_t,
            "query_category_decision_mode": query_decision_mode_t,
            "query_category_decision_signal": query_decision_signal_t,
            "query_category_drop_risk_scores": query_drop_risk_scores_t,
            "query_category_lock_applied": query_lock_applied_t,
            "query_category_lock_source": query_lock_source_t,
            "context_summaries": history_summaries_by_sample[sample_idx],
            "anchor_eval_paths": [list(p) for p in anchor_paths],
            "anchor_leaf_paths": [h.path for h in leaf_hits],
            "anchor_branch_paths": [h.path for h in branch_hits],
            "active_branch_paths": active_paths,
            "density": {str(k): v for k, v in densities.items()},
            "local_paths": [list(p) for p in local_paths],
            "global_paths": [list(p) for p in global_paths],
            "anchor_doc_ids": anchor_doc_ids,
            "local_doc_ids": local_doc_ids,
            "global_doc_ids": global_doc_ids,
            "gold_doc_ids": gold_doc_ids,
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
    else:
        logger.info(
            "Iter %d | Anchor nDCG@10=0.00 | Local nDCG@10=0.00 | "
            "Anchor Recall@100=0.00 | Local Recall@100=0.00",
            iter_idx,
        )

save_exp(RESULTS_DIR, hp, llm_api, all_eval_samples, all_eval_metric_dfs, allow_overwrite=True, save_llm_api_history=True)
logger.info("Saved Round5 results to %s", RESULTS_DIR)
