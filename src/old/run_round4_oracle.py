import asyncio
import json
import logging
import os
import pickle as pkl
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from json_repair import repair_json
from tqdm.autonotebook import tqdm

from cache_utils import _prompt_cache_key, append_jsonl
from flat_then_tree import FlatHit, gate_hit, is_prefix
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


def _compose_query_from_selected_keys(
    original_query: str,
    docs: Dict[str, str],
    selected_keys: Sequence[str],
) -> str:
    selected = set(str(key or "").strip() for key in selected_keys if str(key or "").strip())
    pieces: List[str] = []
    for key in _ordered_doc_keys(docs):
        if key not in selected:
            continue
        text = str((docs or {}).get(key, "")).strip()
        if text:
            pieces.append(text)
    blob = "\n".join(pieces).strip()
    if not blob:
        return (original_query or "").strip()
    return ((original_query or "") + " " + blob).strip()


def _retrieve_with_query(
    *,
    query: str,
    retriever: DiverEmbeddingModel,
    node_embs: np.ndarray,
    node_registry: Sequence[object],
    leaf_indices: Sequence[int],
    leaf_paths: Sequence[Tuple[int, ...]],
    leaf_indices_by_prefix: Dict[Tuple[int, ...], List[int]],
    anchor_topk: int,
    anchor_topk_logged: int,
    local_topk: int,
    global_topk: int,
    anchor_local_rank_mode: str,
) -> Tuple[np.ndarray, List[FlatHit], List[Tuple[int, ...]], List[FlatHit], List[FlatHit], List[FlatHit], List[FlatHit], List[Tuple[int, ...]], Dict[Tuple[int, ...], float]]:
    q_emb = retriever.encode_query(query)
    scores_all = (node_embs @ q_emb).astype(np.float32, copy=False)
    hits = _hits_from_scores(
        scores=scores_all,
        subset_indices=None,
        node_registry=node_registry,
        topk=anchor_topk,
    )
    if anchor_local_rank_mode != "none":
        anchor_eval_paths = _anchor_local_context_paths(
            anchor_hits=hits,
            anchor_topk=anchor_topk,
            scores_all=scores_all,
            node_registry=node_registry,
            leaf_indices_by_prefix=leaf_indices_by_prefix,
            topk=anchor_topk,
        )
    else:
        # Intent: oracle runner preserves the none-baseline anchor evaluation semantics (leaf-only from anchor ranking).
        anchor_eval_paths = [h.path for h in hits if h.is_leaf][:anchor_topk]
    leaf_hits = [h for h in hits if h.is_leaf]
    branch_hits = [h for h in hits if not h.is_leaf]
    active_branches, densities, branch_scores = _build_active_branches(leaf_hits, branch_hits)
    ranked_branches = _rank_branch_paths(active_branches, densities, branch_scores)

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
    if anchor_local_rank_mode != "none":
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
        # Intent: keep none-mode local expansion parity by adding one best descendant per anchor branch.
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
    return scores_all, hits, anchor_eval_paths, leaf_hits, branch_hits, local_hits, global_hits, ranked_branches, densities


def _oracle_candidate_masks(
    docs: Dict[str, str],
    actions: Dict[str, str],
) -> List[Tuple[str, List[str]]]:
    active = _active_categories_from_docs_actions(docs or {}, actions or {})
    if not active:
        active = _ordered_doc_keys(docs or {})
    if not active:
        return [("keep_all", [])]
    candidates: List[Tuple[str, List[str]]] = [("keep_all", list(active))]
    if len(active) > 1:
        for key in active:
            kept = [k for k in active if k != key]
            candidates.append((f"drop_{key}", kept))
    return candidates


def _carry_oracle_categories_forward(
    previous_docs: Dict[str, str],
    previous_actions: Dict[str, str],
    new_docs: Dict[str, str],
) -> Tuple[Dict[str, str], List[str], bool]:
    clean_new = {
        str(key).strip(): str(val).strip()
        for key, val in (new_docs or {}).items()
        if str(key or "").strip() and str(val or "").strip()
    }
    prev_active = _active_categories_from_docs_actions(previous_docs or {}, previous_actions or {})
    if not prev_active:
        return clean_new, _ordered_doc_keys(clean_new), False
    carried: Dict[str, str] = {}
    for key in prev_active:
        if key in clean_new:
            carried[key] = clean_new[key]
            continue
        old_val = str((previous_docs or {}).get(key, "")).strip()
        if old_val:
            # Intent: preserve previous active category context when current rewrite omits a carried key.
            carried[key] = old_val
    if carried:
        return carried, _ordered_doc_keys(carried), True
    return clean_new, _ordered_doc_keys(clean_new), False


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
    if rule_name == "rule_b":
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


hp = HyperParams.from_args()
if not hp.REWRITE_PROMPT_NAME and not hp.REWRITE_PROMPT_PATH and not hp.REWRITE_CACHE_PATH:
    hp.add_param("rewrite_prompt_name", "round3_action_v1")
# Intent: round4-oracle scope supports only none/v2 anchor modes to keep code path auditable.
anchor_mode = str(hp.ROUND3_ANCHOR_LOCAL_RANK or "none").lower()
if anchor_mode not in {"none", "v2"}:
    raise ValueError(
        f'Unsupported --round3_anchor_local_rank="{hp.ROUND3_ANCHOR_LOCAL_RANK}" in run_round4_oracle.py. '
        'Allowed: none|v2'
    )
if bool(hp.ROUND3_REWRITE_USE_HISTORY):
    print("Ignoring --round3_rewrite_use_history in run_round4_oracle.py (fixed to False).")
if hp.ROUND3_ROUTER_PROMPT_NAME:
    print("Ignoring --round3_router_prompt_name in run_round4_oracle.py (router is disabled).")
if str(hp.ROUND3_SUMMARIZED_CONTEXT or "off").lower() != "off":
    print("Ignoring --round3_summarized_context in run_round4_oracle.py (fixed to off).")
hp.add_param("round3_anchor_local_rank", anchor_mode)
hp.add_param("round3_rewrite_use_history", False)
hp.add_param("round3_router_prompt_name", None)
hp.add_param("round3_summarized_context", "off")
# Intent: oracle runner uses hardcoded selection behavior (keep-all vs drop-one) and does not expose rule hyperparameters.
ORACLE_ANCHOR_TOPK = 1000
ORACLE_METRIC_K = 10
ORACLE_BOOTSTRAP_KEEP_ALL = True
hp.add_param("round4_oracle_mode", "drop_one_ndcg")
hp.add_param("round4_oracle_anchor_topk", ORACLE_ANCHOR_TOPK)
hp.add_param("round4_oracle_metric_k", ORACLE_METRIC_K)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
exp_dir_name = str(hp)
# Intent: isolate round4-oracle artifacts under a dedicated directory for clean comparison with round4 policy runs.
RESULTS_DIR = f"{BASE_DIR}/results/{hp.DATASET}/{hp.SUBSET}/round4_oracle/{exp_dir_name}/"
if os.path.exists(RESULTS_DIR) and os.listdir(RESULTS_DIR):
    print(f"Results already exist at {RESULTS_DIR}. Skipping run.")
    raise SystemExit(0)
os.makedirs(RESULTS_DIR, exist_ok=True)
log_path = f"{RESULTS_DIR}/run.log"
os.makedirs(os.path.dirname(log_path), exist_ok=True)
logger = setup_logger("lattice_runner_round4_oracle", log_path, logging.INFO)
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
category_emb_cache: Dict[str, np.ndarray] = {}

leaf_indices = [idx for idx, node in enumerate(node_registry) if node.is_leaf]
leaf_paths = [tuple(node_registry[idx].path) for idx in leaf_indices]
leaf_indices_by_prefix: Dict[Tuple[int, ...], List[int]] = {}
for idx, path in zip(leaf_indices, leaf_paths):
    for d in range(1, len(path)):
        prefix = path[:d]
        leaf_indices_by_prefix.setdefault(prefix, []).append(idx)

rewrite_enabled = True
rewrite_template = None
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
if hp.REWRITE_PROMPT_PATH:
    if not os.path.exists(hp.REWRITE_PROMPT_PATH):
        raise ValueError(f"--rewrite_prompt_path not found: {hp.REWRITE_PROMPT_PATH}")
    with open(hp.REWRITE_PROMPT_PATH, "r", encoding="utf-8") as f:
        rewrite_template = f.read()
rewrite_map, action_map, docs_map = _load_rewrite_action_cache(hp.REWRITE_CACHE_PATH, hp.REWRITE_FORCE_REFRESH)

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
    # Keep rewrite/controller generation deterministic for reproducible round4 analysis runs.
    "temperature": 0.0,
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

# Intent: oracle comparison is fixed to top-1000 anchor retrieval for candidate query evaluation.
anchor_topk = ORACLE_ANCHOR_TOPK
local_topk = hp.ROUND3_LOCAL_TOPK or hp.FLAT_TOPK
global_topk = hp.ROUND3_GLOBAL_TOPK

all_eval_metric_dfs: List[pd.DataFrame] = []
for iter_idx in range(hp.NUM_ITERS):
    logger.info("Round4-oracle iteration %d", iter_idx)
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
        # Intent: oracle compares keep-all vs drop-one candidates and carries the best mask to next iteration.
        if sample.last_possible_docs:
            query_action_t = "multi"
            query_docs_t = dict(sample.last_possible_docs or {})
            # Intent: expose last rewrite's support scores in iter_records so policy analysis can use query-time signals.
            query_support_scores_t = {str(k): float(v) for k, v in (sample.last_category_support_scores or {}).items()}
            query_drop_risk_scores_t = {}
            query_lock_applied_t = False
            query_lock_source_t = []
            candidate_masks = _oracle_candidate_masks(sample.last_possible_docs, sample.last_actions)
            candidate_scores: Dict[str, float] = {}
            candidate_payloads: List[Tuple[str, List[str], Dict[str, str], str, np.ndarray, List[FlatHit], List[Tuple[int, ...]], List[FlatHit], List[FlatHit], List[FlatHit], List[FlatHit], List[Tuple[int, ...]], Dict[Tuple[int, ...], float]]] = []
            for cand_name, cand_keys in candidate_masks:
                cand_actions = _build_actions_from_selected(
                    sample.last_possible_docs,
                    cand_keys,
                    "explore" if cand_name == "keep_all" else "exploit",
                )
                cand_query = _compose_query_from_selected_keys(
                    sample.original_query,
                    sample.last_possible_docs,
                    cand_keys,
                )
                (
                    cand_scores_all,
                    cand_hits,
                    cand_anchor_eval_paths,
                    cand_leaf_hits,
                    cand_branch_hits,
                    cand_local_hits,
                    cand_global_hits,
                    cand_ranked_branches,
                    cand_densities,
                ) = _retrieve_with_query(
                    query=cand_query,
                    retriever=retriever,
                    node_embs=node_embs,
                    node_registry=node_registry,
                    leaf_indices=leaf_indices,
                    leaf_paths=leaf_paths,
                    leaf_indices_by_prefix=leaf_indices_by_prefix,
                    anchor_topk=anchor_topk,
                    anchor_topk_logged=anchor_topk_logged,
                    local_topk=local_topk,
                    global_topk=global_topk,
                    anchor_local_rank_mode=hp.ROUND3_ANCHOR_LOCAL_RANK,
                )
                cand_ndcg = compute_ndcg(
                    [list(p) for p in cand_anchor_eval_paths[:ORACLE_METRIC_K]],
                    [list(p) for p in sample.gold_paths],
                    k=ORACLE_METRIC_K,
                ) * 100.0
                candidate_scores[cand_name] = float(cand_ndcg)
                candidate_payloads.append((
                    cand_name,
                    list(cand_keys),
                    cand_actions,
                    cand_query,
                    cand_scores_all,
                    cand_hits,
                    cand_anchor_eval_paths,
                    cand_leaf_hits,
                    cand_branch_hits,
                    cand_local_hits,
                    cand_global_hits,
                    cand_ranked_branches,
                    cand_densities,
                ))

            def _oracle_rank_key(payload: Tuple[str, List[str], Dict[str, str], str, np.ndarray, List[FlatHit], List[Tuple[int, ...]], List[FlatHit], List[FlatHit], List[FlatHit], List[FlatHit], List[Tuple[int, ...]], Dict[Tuple[int, ...], float]]) -> Tuple[float, int, int, str]:
                name = payload[0]
                keys = payload[1]
                return (
                    float(candidate_scores.get(name, -1.0)),
                    1 if name == "keep_all" else 0,
                    len(keys),
                    name,
                )

            best_payload = max(candidate_payloads, key=_oracle_rank_key)
            (
                best_name,
                best_keys,
                query_actions_t,
                anchor_query,
                scores_all,
                hits,
                anchor_eval_paths,
                leaf_hits,
                branch_hits,
                local_hits,
                global_hits,
                ranked_branches,
                densities,
            ) = best_payload
            query_categories_t = list(best_keys)
            query_decision_mode_t = "explore" if best_name == "keep_all" else "exploit"
            query_decision_signal_t = {
                "name": "oracle_ndcg@10",
                "value": float(candidate_scores.get(best_name, 0.0)),
                "best_candidate": best_name,
                "candidate_scores": dict(sorted(candidate_scores.items())),
            }
            sample.last_actions = dict(query_actions_t)
            sample.last_action = query_decision_mode_t
            sample.last_category_decision_mode = query_decision_mode_t
            sample.last_category_decision_signal = dict(query_decision_signal_t)
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
            (
                scores_all,
                hits,
                anchor_eval_paths,
                leaf_hits,
                branch_hits,
                local_hits,
                global_hits,
                ranked_branches,
                densities,
            ) = _retrieve_with_query(
                query=anchor_query,
                retriever=retriever,
                node_embs=node_embs,
                node_registry=node_registry,
                leaf_indices=leaf_indices,
                leaf_paths=leaf_paths,
                leaf_indices_by_prefix=leaf_indices_by_prefix,
                anchor_topk=anchor_topk,
                anchor_topk_logged=anchor_topk_logged,
                local_topk=local_topk,
                global_topk=global_topk,
                anchor_local_rank_mode=hp.ROUND3_ANCHOR_LOCAL_RANK,
            )

        anchor_hits_by_sample.append(hits)
        anchor_eval_paths_by_sample.append(anchor_eval_paths)
        leaf_hits_by_sample.append(leaf_hits)
        branch_hits_by_sample.append(branch_hits)
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
        # Intent: oracle runner keeps summarized context disabled to avoid extra summary-generation complexity.
        for meta in rewrite_candidates:
            sample_idx = int(meta.get("sample_idx", -1))
            if 0 <= sample_idx < len(history_summaries_by_sample):
                history_summaries_by_sample[sample_idx] = []

    if rewrite_candidates:
        rewrite_prompts: List[str] = []
        rewrite_meta: List[Dict] = []
        for meta in rewrite_candidates:
            sample = meta["sample"]
            leaf_descs = meta["leaf_descs"]
            branch_descs = meta["branch_descs"] if hp.ROUND3_REWRITE_CONTEXT == "leaf_branch" else []
            retrieval_history = meta.get("retrieval_history", "")
            prompt = _format_action_prompt(
                rewrite_template,
                sample.original_query,
                sample.last_rewrite,
                sample.last_possible_docs,
                leaf_descs,
                branch_descs,
                subset_name=hp.SUBSET,
                retrieval_history=retrieval_history,
            )
            cache_key = _prompt_cache_key("round3", prompt)
            if (not hp.REWRITE_FORCE_REFRESH) and (cache_key in rewrite_map):
                rewrite = rewrite_map[cache_key]
                prev_selected_categories: List[str] = _active_categories_from_docs_actions(
                    sample.last_possible_docs or {},
                    sample.last_actions or {},
                )
                cached_actions = action_map.get(cache_key)
                cached_docs = docs_map.get(cache_key, {})
                clean_cached_docs = {
                    str(key).strip(): str(val).strip()
                    for key, val in (cached_docs or {}).items()
                    if str(key or "").strip() and str(val or "").strip()
                }
                if clean_cached_docs:
                    carried_docs, selected_categories, used_prev_mask = _carry_oracle_categories_forward(
                        sample.last_possible_docs or {},
                        sample.last_actions or {},
                        clean_cached_docs,
                    )
                    sample.last_possible_docs = carried_docs
                    sample.last_actions = _build_actions_from_selected(
                        sample.last_possible_docs,
                        selected_categories,
                        "explore",
                    )
                    sample.last_rewrite = _flatten_docs_by_action(sample.last_possible_docs, sample.last_actions)
                    rewrite = sample.last_rewrite
                    sample.last_action = "exploit" if used_prev_mask and (len(selected_categories) < len(_ordered_doc_keys(clean_cached_docs))) else "explore"
                    decision_signal = {
                        "name": "oracle_carry_forward",
                        "value": float(len(selected_categories)),
                        "used_prev_mask": bool(used_prev_mask),
                    }
                elif isinstance(cached_actions, dict):
                    sample.last_actions = cached_actions
                    sample.last_possible_docs = cached_docs
                    sample.last_rewrite = rewrite
                    sample.last_action = "exploit"
                    selected_categories = _active_categories_from_docs_actions(sample.last_possible_docs or {}, sample.last_actions or {})
                    decision_signal = {"name": "cache_actions", "value": float(len(selected_categories))}
                else:
                    action = str(cached_actions or "exploit").strip().lower()
                    if action == "hold" and sample.last_rewrite:
                        rewrite = sample.last_rewrite
                    else:
                        sample.last_rewrite = rewrite
                    sample.last_action = action
                    sample.last_actions = {}
                    sample.last_possible_docs = {}
                    selected_categories = []
                    decision_signal = {"name": "cache_text_only", "value": 0.0}
                support_leaf_descs = meta.get("support_leaf_descs", [])
                computed_support_scores: Dict[str, float] = {}
                if sample.last_possible_docs:
                    _, computed_support_scores, _ = _compute_category_support(
                        docs=sample.last_possible_docs,
                        evidence_descs=support_leaf_descs,
                        support_topm=hp.ROUND4_SUPPORT_TOPM,
                        retriever=retriever,
                        emb_cache=category_emb_cache,
                    )
                # Intent: persist per-category support so oracle traces can be reused for policy analysis.
                sample.last_category_support_scores = {str(k): float(v) for k, v in computed_support_scores.items()}
                sample.last_category_decision_mode = str(sample.last_action or "")
                sample.last_category_decision_signal = dict(decision_signal)
                sample.last_category_drop_risk_scores = {}
                sample.last_category_lock_applied = False
                sample.last_category_lock_source = []
                sample.rewrite_history.append({
                    "iter": iter_idx,
                    "cache_hit": True,
                    "action": sample.last_action,
                    "actions": sample.last_actions,
                    "possible_docs": sample.last_possible_docs,
                    "rewrite": rewrite,
                    "prev_selected_categories": prev_selected_categories,
                    "selected_categories": selected_categories,
                    "category_scores": dict(sample.last_category_support_scores),
                    "category_support_scores": dict(sample.last_category_support_scores),
                    "category_penalty_scores": {},
                    "category_drop_risk_scores": {},
                    "category_decision_mode": str(sample.last_action or ""),
                    "category_decision_signal": decision_signal,
                    "category_lock_applied": False,
                    "category_lock_source": [],
                })
            else:
                rewrite_prompts.append(prompt)
                rewrite_meta.append({
                    "sample": sample,
                    "cache_key": cache_key,
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
                prev_selected_categories: List[str] = _active_categories_from_docs_actions(
                    sample.last_possible_docs or {},
                    sample.last_actions or {},
                )
                output_docs = docs or {}
                clean_output_docs = {
                    str(key).strip(): str(val).strip()
                    for key, val in (output_docs or {}).items()
                    if str(key or "").strip() and str(val or "").strip()
                }
                if clean_output_docs:
                    carried_docs, selected_categories, used_prev_mask = _carry_oracle_categories_forward(
                        sample.last_possible_docs or {},
                        sample.last_actions or {},
                        clean_output_docs,
                    )
                    sample.last_possible_docs = carried_docs
                    sample.last_actions = _build_actions_from_selected(
                        sample.last_possible_docs,
                        selected_categories,
                        "explore",
                    )
                    sample.last_rewrite = _flatten_docs_by_action(sample.last_possible_docs, sample.last_actions)
                    rewrite = sample.last_rewrite
                    sample.last_action = "exploit" if used_prev_mask and (len(selected_categories) < len(_ordered_doc_keys(clean_output_docs))) else "explore"
                    decision_signal = {
                        "name": "oracle_carry_forward",
                        "value": float(len(selected_categories)),
                        "used_prev_mask": bool(used_prev_mask),
                    }
                elif actions:
                    sample.last_actions = actions
                    sample.last_possible_docs = output_docs
                    sample.last_rewrite = _flatten_docs_by_action(sample.last_possible_docs, sample.last_actions)
                    sample.last_action = "exploit"
                    rewrite = sample.last_rewrite
                    selected_categories = _active_categories_from_docs_actions(sample.last_possible_docs or {}, sample.last_actions or {})
                    decision_signal = {"name": "llm_actions", "value": float(len(selected_categories))}
                else:
                    if action == "hold" and sample.last_rewrite:
                        rewrite = sample.last_rewrite
                    else:
                        sample.last_rewrite = rewrite
                    sample.last_action = action
                    sample.last_actions = {}
                    sample.last_possible_docs = {}
                    selected_categories = []
                    decision_signal = {"name": "text_only_rewrite", "value": 0.0}
                support_leaf_descs = meta.get("support_leaf_descs", [])
                computed_support_scores: Dict[str, float] = {}
                if sample.last_possible_docs:
                    _, computed_support_scores, _ = _compute_category_support(
                        docs=sample.last_possible_docs,
                        evidence_descs=support_leaf_descs,
                        support_topm=hp.ROUND4_SUPPORT_TOPM,
                        retriever=retriever,
                        emb_cache=category_emb_cache,
                    )
                # Intent: persist per-category support so oracle traces can be reused for policy analysis.
                sample.last_category_support_scores = {str(k): float(v) for k, v in computed_support_scores.items()}
                sample.last_category_decision_mode = str(sample.last_action or "")
                sample.last_category_decision_signal = dict(decision_signal)
                sample.last_category_drop_risk_scores = {}
                sample.last_category_lock_applied = False
                sample.last_category_lock_source = []
                sample.rewrite_history.append({
                    "iter": iter_idx,
                    "cache_hit": False,
                    "action": sample.last_action,
                    "actions": sample.last_actions,
                    "possible_docs": sample.last_possible_docs,
                    "rewrite": rewrite,
                    "prev_selected_categories": prev_selected_categories,
                    "selected_categories": selected_categories,
                    "category_scores": dict(sample.last_category_support_scores),
                    "category_support_scores": dict(sample.last_category_support_scores),
                    "category_penalty_scores": {},
                    "category_drop_risk_scores": {},
                    "category_decision_mode": str(sample.last_action or ""),
                    "category_decision_signal": decision_signal,
                    "category_lock_applied": False,
                    "category_lock_source": [],
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
            "query_categories": query_categories_t,
            "query_category_support_scores": query_support_scores_t,
            "query_category_decision_mode": query_decision_mode_t,
            "query_category_decision_signal": query_decision_signal_t,
            "query_category_drop_risk_scores": query_drop_risk_scores_t,
            "query_category_lock_applied": query_lock_applied_t,
            "query_category_lock_source": query_lock_source_t,
            "context_summaries": history_summaries_by_sample[sample_idx],
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
    else:
        logger.info(
            "Iter %d | Anchor nDCG@10=0.00 | Local nDCG@10=0.00 | "
            "Anchor Recall@100=0.00 | Local Recall@100=0.00",
            iter_idx,
        )

save_exp(RESULTS_DIR, hp, llm_api, all_eval_samples, all_eval_metric_dfs, allow_overwrite=True, save_llm_api_history=True)
logger.info("Saved Round4 results to %s", RESULTS_DIR)
