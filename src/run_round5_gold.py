import asyncio
import json
import logging
import os
import pickle as pkl
import re
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
from retrievers.diver import DiverEmbeddingModel
from rewrite_prompts import REWRITE_PROMPT_TEMPLATES
from tree_objects import InferSample, SemanticNode
from utils import (
    compute_ndcg,
    compute_node_registry,
    compute_recall,
    get_node_id,
    normalize_embeddings,
    save_exp,
    setup_logger,
)


CATEGORY_ORDER = ["Theory", "Entity", "Example", "Other"]
DEFAULT_CATEGORY_LABELS = ["theory", "entity", "example"]
STACKEXCHANGE_SUBSETS = {
    "biology",
    "earth_science",
    "economics",
    "psychology",
    "robotics",
    "sustainable_living",
}
CODING_SUBSETS = {"leetcode", "pony", "stackoverflow"}
THEOREM_SUBSETS = {"aops", "theoq", "theot", "theoremqa_questions", "theoremqa_theorems"}


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


def _normalize_category_label(text: str) -> str:
    label = str(text or "").strip().lower()
    label = re.sub(r"[^a-z0-9_\-\s]", "", label)
    label = label.replace("-", "_")
    label = re.sub(r"\s+", "_", label)
    label = re.sub(r"_+", "_", label).strip("_")
    if not label:
        return ""
    parts = [p for p in label.split("_") if p]
    if not parts:
        return ""
    if len(parts) > 2:
        parts = parts[:2]
    return "_".join(parts)


def _parse_category_output(text: str, max_k: int) -> List[Dict[str, str]]:
    candidate = _extract_json_candidate(text)
    obj: Any = None
    try:
        obj = json.loads(candidate)
    except Exception:
        try:
            obj = repair_json(candidate, return_objects=True)
        except Exception:
            obj = None

    raw_categories: Any = None
    if isinstance(obj, dict):
        raw_categories = (
            _recursive_get(obj, "Categories")
            or _recursive_get(obj, "categories")
            or _recursive_get(obj, "Selected_Categories")
            or _recursive_get(obj, "selected_categories")
        )

    parsed: List[Dict[str, str]] = []
    if isinstance(raw_categories, list):
        for item in raw_categories:
            if isinstance(item, dict):
                label = _normalize_category_label(item.get("label", ""))
                hint = str(item.get("hint", "") or "").strip()
                if not label:
                    continue
                parsed.append({"label": label, "hint": hint})
            else:
                label = _normalize_category_label(item)
                if label:
                    parsed.append({"label": label, "hint": ""})
    elif isinstance(raw_categories, dict):
        for key, val in raw_categories.items():
            label = _normalize_category_label(key)
            hint = str(val or "").strip()
            if label:
                parsed.append({"label": label, "hint": hint})

    out: List[Dict[str, str]] = []
    seen: Set[str] = set()
    for row in parsed:
        label = str(row.get("label", "")).strip()
        if (not label) or (label in seen):
            continue
        seen.add(label)
        out.append({"label": label, "hint": str(row.get("hint", "") or "").strip()})
        if len(out) >= max(1, int(max_k)):
            break
    return out


def _build_domain_route_hint(subset_name: str) -> str:
    name = str(subset_name or "").strip().lower()
    relevance_definitions = {
        "leetcode": (
            "The relevance between queries and positive documents is defined by whether the coding problem "
            "(i.e., query) involves the same algorithm and/or data structure. The queries and documents are "
            "problems and solutions from LeetCode. The problem descriptions are used as queries Q, and the "
            "positive documents D+Q are solved problems (with solutions) that were annotated as similar "
            "problems by LeetCode."
        ),
        "theoremqa_questions": (
            "A query is relevant to a document if the document references the same/similar theorem used in the query."
        ),
        "pony": (
            "The relevance between queries and positive documents is defined by whether the coding problem "
            "(i.e., query) requires the corresponding syntax documentation."
        ),
        "stackexchange": (
            "A document is considered relevant to a query if it can be cited in an accepted or highly voted "
            "answer that helps reason through the query with critical concepts or theories."
        ),
    }

    if name in {"leetcode"}:
        key = "leetcode"
    elif name in {"theoremqa_questions", "theoremqa_theorems", "aops", "theoq", "theot"}:
        key = "theoremqa_questions"
    elif name in {"pony"}:
        key = "pony"
    else:
        # Intent: all remaining BRIGHT subsets follow StackExchange-style relevance routing in this hint.
        key = "stackexchange"

    return (
        "Use this relevance definition as your routing prior for category generation:\n"
        f"{relevance_definitions[key]}"
    )


def _format_category_history(
    category_bank: Sequence[Dict[str, Any]],
    scope: str,
    max_rows: int = 12,
) -> str:
    if str(scope or "full").lower() == "none":
        return "None"
    rows = list(category_bank or [])
    if not rows:
        return "None"
    clipped = rows[-max_rows:]
    lines: List[str] = []
    for row in clipped:
        label = str(row.get("label", "")).strip()
        hint = str(row.get("hint", "")).strip()
        iter_idx = row.get("iter")
        if not label:
            continue
        if hint:
            lines.append(f"- iter={iter_idx} label={label}: {hint}")
        else:
            lines.append(f"- iter={iter_idx} label={label}")
    return "\n".join(lines) if lines else "None"


def _format_selected_categories(categories: Sequence[Dict[str, str]]) -> str:
    lines: List[str] = []
    for row in categories:
        label = str(row.get("label", "")).strip()
        hint = str(row.get("hint", "")).strip()
        if not label:
            continue
        if hint:
            lines.append(f"- {label}: {hint}")
        else:
            lines.append(f"- {label}")
    return "\n".join(lines) if lines else "- theory\n- entity\n- example"


def _format_selected_category_schema(categories: Sequence[Dict[str, str]]) -> str:
    labels = [str(row.get("label", "")).strip() for row in categories if str(row.get("label", "")).strip()]
    if not labels:
        labels = list(DEFAULT_CATEGORY_LABELS)
    lines: List[str] = []
    for idx, label in enumerate(labels):
        suffix = "," if idx < (len(labels) - 1) else ""
        lines.append(f'    "{label}": "..."{suffix}')
    return "\n".join(lines)


def _collect_branch_descs_from_selected(
    selected_paths: Sequence[Tuple[int, ...]],
    path_to_node: Dict[Tuple[int, ...], object],
    max_desc_len: int | None,
    topk: int,
) -> List[str]:
    descs: List[str] = []
    for path in selected_paths:
        node = path_to_node.get(tuple(path))
        if node is None:
            continue
        desc = str(getattr(node, "desc", "") or "").strip()
        if not desc:
            continue
        if max_desc_len:
            desc = desc[:max_desc_len]
        descs.append(desc)
        if len(descs) >= topk:
            break
    return descs


def _is_leaf_cluster_trigger(
    selected_paths: Sequence[Tuple[int, ...]],
    path_to_node: Dict[Tuple[int, ...], object],
) -> bool:
    for path in selected_paths:
        node = path_to_node.get(tuple(path))
        if node is None:
            continue
        children = list(getattr(node, "child", []) or [])
        if children and all(bool(getattr(child, "is_leaf", False)) for child in children):
            return True
    return False


def _align_docs_to_categories(
    docs: Dict[str, str],
    categories: Sequence[Dict[str, str]],
) -> Dict[str, str]:
    if not docs:
        return {}
    categories_by_norm: Dict[str, str] = {}
    for row in categories:
        label = str(row.get("label", "")).strip()
        if label:
            categories_by_norm[_normalize_category_label(label)] = label

    aligned: Dict[str, str] = {}
    for key, val in docs.items():
        norm = _normalize_category_label(key)
        if (not norm) or (norm not in categories_by_norm):
            continue
        canonical = categories_by_norm[norm]
        text = str(val or "").strip()
        if text:
            aligned[canonical] = text
    return aligned


def _format_rewrite_prompt(
    template: str,
    *,
    original_query: str,
    previous_rewrite: str,
    leaf_descs: Sequence[str],
    branch_descs: Sequence[str],
    domain_route_hint: str = "",
    category_history: str = "",
    selected_categories: str = "",
    selected_category_schema: str = "",
    stability_hint: str = "",
    category_k: int = 3,
) -> str:
    leaf_blob = "\n".join([x for x in leaf_descs if x])
    branch_blob = "\n".join([x for x in branch_descs if x])
    gate_blob = "\n".join([x for x in list(leaf_descs) + list(branch_descs) if x])

    if not branch_blob:
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
            branch_descs=branch_blob,
            gate_descs=gate_blob,
            domain_route_hint=domain_route_hint or "",
            corpus_categories="",
            category_history=category_history or "",
            selected_categories=selected_categories or "",
            selected_category_schema=selected_category_schema or "",
            stability_hint=stability_hint or "",
            drift_hint=stability_hint or "",
            category_k=int(category_k),
        )
    except KeyError:
        return (
            template
            .replace("{original_query}", original_query or "")
            .replace("{previous_rewrite}", previous_rewrite or "")
            .replace("{previous_docs}", "")
            .replace("{leaf_descs}", leaf_blob)
            .replace("{branch_descs}", branch_blob)
            .replace("{gate_descs}", gate_blob)
            .replace("{domain_route_hint}", domain_route_hint or "")
            .replace("{corpus_categories}", "")
            .replace("{category_history}", category_history or "")
            .replace("{selected_categories}", selected_categories or "")
            .replace("{selected_category_schema}", selected_category_schema or "")
            .replace("{stability_hint}", stability_hint or "")
            .replace("{drift_hint}", stability_hint or "")
            .replace("{category_k}", str(int(category_k)))
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
) -> List[int]:
    if not selected_branches:
        return list(all_leaf_indices)

    out: List[int] = []
    seen: Set[int] = set()
    for branch_path in selected_branches:
        for idx in leaf_indices_by_prefix.get(tuple(branch_path), []):
            if idx in seen:
                continue
            seen.add(idx)
            out.append(int(idx))

    if not out:
        # Intent: if selected branches have no descendants (or are stale), fail soft to full leaf pool.
        return list(all_leaf_indices)
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


def _collect_candidate_child_branches(
    *,
    selected_before: Sequence[Tuple[int, ...]],
    child_branch_paths_by_path: Dict[Tuple[int, ...], List[Tuple[int, ...]]],
    root_branch_children: Sequence[Tuple[int, ...]],
    beam_size: int,
) -> List[Tuple[int, ...]]:
    candidates: List[Tuple[int, ...]] = []
    seen: Set[Tuple[int, ...]] = set()
    for path in selected_before:
        for child_path in child_branch_paths_by_path.get(tuple(path), []):
            child_t = tuple(child_path)
            if child_t in seen:
                continue
            seen.add(child_t)
            candidates.append(child_t)
    if not candidates:
        for child_path in root_branch_children:
            child_t = tuple(child_path)
            if child_t in seen:
                continue
            seen.add(child_t)
            candidates.append(child_t)
    return candidates[: max(1, int(beam_size))]


def _resolve_oracle_branch_paths(
    *,
    oracle_mode: str,
    selected_before: Sequence[Tuple[int, ...]],
    gold_paths: Sequence[Tuple[int, ...]],
    leaf_ancestor_paths: Dict[Tuple[int, ...], List[Tuple[int, ...]]],
    child_branch_paths_by_path: Dict[Tuple[int, ...], List[Tuple[int, ...]]],
    root_branch_children: Sequence[Tuple[int, ...]],
    beam_size: int,
) -> Tuple[List[Tuple[int, ...]], str, str]:
    default_paths = [tuple(path) for path in selected_before if path]
    if oracle_mode == "none":
        return default_paths, "llm_selected_branches", ""

    if oracle_mode == "gold_branch_v1":
        base_candidates = _collect_candidate_child_branches(
            selected_before=selected_before,
            child_branch_paths_by_path=child_branch_paths_by_path,
            root_branch_children=root_branch_children,
            beam_size=beam_size,
        )
        fallback_reason = "no_gold_in_candidate_children"
        source_tag = "oracle_gold_branch_v1"
    elif oracle_mode == "gold_branch_v2":
        # Intent: v2 follows existing semantics and treats selected_before as the current top-B source.
        base_candidates = [tuple(path) for path in selected_before if path][: max(1, int(beam_size))]
        fallback_reason = "no_gold_in_selected_before"
        source_tag = "oracle_gold_branch_v2"
    else:
        raise ValueError(f"Unsupported oracle mode: {oracle_mode}")

    gold_candidates: List[Tuple[int, ...]] = []
    for path in base_candidates:
        if _branch_is_gold_ancestor(tuple(path), gold_paths, leaf_ancestor_paths):
            gold_candidates.append(tuple(path))
            if len(gold_candidates) >= max(1, int(beam_size)):
                break
    if gold_candidates:
        return gold_candidates, source_tag, ""
    return default_paths, f"{source_tag}_fallback", fallback_reason


hp = HyperParams.from_args()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

round5_mode = str(getattr(hp, "ROUND5_MODE", "legacy") or "legacy").strip().lower()
if round5_mode not in {"legacy", "category"}:
    raise ValueError(f'Unsupported --round5_mode "{round5_mode}". Allowed: legacy|category')
if round5_mode != "category":
    raise ValueError('run_round5_gold.py supports only --round5_mode "category".')
hp.add_param("round5_mode", round5_mode)

if hp.REWRITE_PROMPT_PATH:
    print("Ignoring --rewrite_prompt_path in run_round5_gold.py (template path override is disabled).")

round5_rewrite_prompt_name = str(
    getattr(hp, "ROUND5_CATEGORY_REWRITE_PROMPT_NAME", "round5_agent_executor_category_v1")
    or "round5_agent_executor_category_v1"
).strip()
round5_category_generator_prompt_name = str(
    getattr(hp, "ROUND5_CATEGORY_GENERATOR_PROMPT_NAME", "round5_category_generator_v1")
    or "round5_category_generator_v1"
).strip()
round5_category_k = max(1, int(getattr(hp, "ROUND5_CATEGORY_K", 3) or 3))
round5_category_history_scope = str(getattr(hp, "ROUND5_CATEGORY_HISTORY_SCOPE", "full") or "full").lower()
round5_category_drift_trigger = str(getattr(hp, "ROUND5_CATEGORY_DRIFT_TRIGGER", "leaf_cluster") or "leaf_cluster").lower()
round5_category_oracle = str(getattr(hp, "ROUND5_CATEGORY_ORACLE", "none") or "none").lower()
round5_category_fallback_on_parse_fail = True
round5_category_partial_ok = True

if round5_category_generator_prompt_name not in REWRITE_PROMPT_TEMPLATES:
    raise ValueError(
        f'Unknown --round5_category_generator_prompt_name "{round5_category_generator_prompt_name}". '
        f'Available: {sorted(REWRITE_PROMPT_TEMPLATES.keys())}'
    )
if round5_rewrite_prompt_name not in REWRITE_PROMPT_TEMPLATES:
    raise ValueError(
        f'Unknown --round5_category_rewrite_prompt_name "{round5_rewrite_prompt_name}". '
        f'Available: {sorted(REWRITE_PROMPT_TEMPLATES.keys())}'
    )
if round5_category_history_scope not in {"full", "none"}:
    raise ValueError(
        f'Unsupported --round5_category_history_scope "{round5_category_history_scope}". Allowed: full|none'
    )
if round5_category_drift_trigger not in {"leaf_cluster", "none"}:
    raise ValueError(
        f'Unsupported --round5_category_drift_trigger "{round5_category_drift_trigger}". Allowed: leaf_cluster|none'
    )
if round5_category_oracle not in {"none", "gold_branch_v1", "gold_branch_v2"}:
    raise ValueError(
        f'Unsupported --round5_category_oracle "{round5_category_oracle}". '
        "Allowed: none|gold_branch_v1|gold_branch_v2"
    )
hp.add_param("round5_category_fallback_on_parse_fail", round5_category_fallback_on_parse_fail)
hp.add_param("round5_category_partial_ok", round5_category_partial_ok)

hp.add_param("rewrite_prompt_name", round5_rewrite_prompt_name)
hp.add_param("round5_category_k", round5_category_k)
hp.add_param("round5_category_history_scope", round5_category_history_scope)
hp.add_param("round5_category_drift_trigger", round5_category_drift_trigger)
hp.add_param("round5_category_oracle", round5_category_oracle)
hp.add_param("round5_category_generator_prompt_name", round5_category_generator_prompt_name)
hp.add_param("round5_category_rewrite_prompt_name", round5_rewrite_prompt_name)

if str(hp.ROUND3_SUMMARIZED_CONTEXT or "off").lower() != "off":
    print("Ignoring --round3_summarized_context in run_round5_gold.py (fixed to off).")
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

exp_dir_name = str(hp)
RESULTS_DIR = f"{BASE_DIR}/results/{hp.DATASET}/{hp.SUBSET}/round5_gold/{exp_dir_name}/"
if os.path.exists(RESULTS_DIR) and os.listdir(RESULTS_DIR):
    print(f"Results already exist at {RESULTS_DIR}. Skipping run.")
    raise SystemExit(0)
os.makedirs(RESULTS_DIR, exist_ok=True)

logger = setup_logger("lattice_runner_round5", f"{RESULTS_DIR}/run.log", logging.INFO)
with open(f"{RESULTS_DIR}/hparams.json", "w", encoding="utf-8") as f:
    json.dump(vars(hp), f, indent=2, ensure_ascii=True, sort_keys=True)
logger.info(
    "Round5 config | mode=%s | rewrite_prompt=%s | category_generator_prompt=%s | category_k=%d | category_oracle=%s | selector_mode=%s | selector_topk=%d",
    round5_mode,
    round5_rewrite_prompt_name,
    str(round5_category_generator_prompt_name or "none"),
    int(round5_category_k),
    round5_category_oracle,
    round5_selector_mode,
    int(round5_mrr_pool_k),
)

if not hp.REWRITE_CACHE_PATH:
    cache_root = os.path.join(BASE_DIR, "cache", "rewrite")
    os.makedirs(cache_root, exist_ok=True)
    cache_mode_tag = f"round5_gold_category_{round5_category_oracle}"
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

rewrite_template = REWRITE_PROMPT_TEMPLATES[round5_rewrite_prompt_name]
category_generator_template = (
    REWRITE_PROMPT_TEMPLATES[round5_category_generator_prompt_name]
    if round5_mode == "category" and round5_category_generator_prompt_name
    else ""
)
subset_domain_route_hint = _build_domain_route_hint(hp.SUBSET)
rewrite_map, docs_map = _load_rewrite_cache(hp.REWRITE_CACHE_PATH, hp.REWRITE_FORCE_REFRESH)

if hp.LLM_API_BACKEND == "genai":
    llm_api = GenAIAPI(hp.LLM, logger=logger, timeout=hp.LLM_API_TIMEOUT, max_retries=hp.LLM_API_MAX_RETRIES)
elif hp.LLM_API_BACKEND == "vllm":
    vllm_base_url, vllm_base_url_src = _resolve_vllm_base_url(BASE_DIR)
    logger.info("Round5 vLLM endpoints source: %s", vllm_base_url_src)
    logger.info("Round5 vLLM endpoints: %s", vllm_base_url)
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
    sample.last_category_labels = []
    sample.category_bank = []
    sample.rewrite_history = []
    sample.iter_records = []
    if "original_query" not in sample.SAVE_LIST:
        sample.SAVE_LIST.append("original_query")
    if "gold_doc_ids" not in sample.SAVE_LIST:
        sample.SAVE_LIST.append("gold_doc_ids")
    if "last_rewrite_raw" not in sample.SAVE_LIST:
        sample.SAVE_LIST.append("last_rewrite_raw")
    if "last_category_labels" not in sample.SAVE_LIST:
        sample.SAVE_LIST.append("last_category_labels")
    if "category_bank" not in sample.SAVE_LIST:
        sample.SAVE_LIST.append("category_bank")
    if "iter_records" not in sample.SAVE_LIST:
        sample.SAVE_LIST.append("iter_records")
    all_eval_samples.append(sample)

all_eval_metric_dfs: List[pd.DataFrame] = []
cumulative_leaf_indices_by_sample: List[Set[int]] = [set() for _ in all_eval_samples]
for iter_idx in range(hp.NUM_ITERS):
    logger.info("Round5 iteration %d", iter_idx)

    # Intent: keep cumulative leaf pools from all previously reached leaves per sample.
    for sample_idx, sample in enumerate(all_eval_samples):
        reached = sample.get_top_predictions(k=None, rel_fn=sample.get_rel_fn(leaf=True))
        for node, _ in reached:
            ridx = int(getattr(node, "registry_idx", -1))
            if ridx >= 0:
                cumulative_leaf_indices_by_sample[sample_idx].add(ridx)

    rewrite_prompts: List[str] = []
    rewrite_meta: List[Dict[str, Any]] = []
    category_prompts: List[str] = []
    category_meta: List[Dict[str, Any]] = []
    rewrite_state_by_sample_idx: Dict[int, Dict[str, Any]] = {}
    retrieval_rows: List[Dict[str, float]] = []

    for sample_idx, sample in enumerate(tqdm(all_eval_samples, desc=f"Iter {iter_idx} rewrite prep", leave=False)):
        query_pre = str(sample.query or sample.original_query).strip()
        cumulative_pool = sorted(cumulative_leaf_indices_by_sample[sample_idx])
        if not cumulative_pool:
            cumulative_pool = list(leaf_indices)

        pre_hits = _retrieve_leaf_hits(
            query=query_pre,
            leaf_pool_indices=cumulative_pool,
            retriever=retriever,
            node_embs=node_embs,
            node_registry=node_registry,
            topk=round5_mrr_pool_k,
        )
        leaf_descs = _hits_to_context_descs(
            pre_hits,
            node_registry,
            topk=hp.REWRITE_CONTEXT_TOPK,
            max_desc_len=hp.MAX_DOC_DESC_CHAR_LEN,
        )
        selected_before = _selected_branch_paths_from_sample(sample)
        oracle_branch_paths, oracle_branch_source, oracle_branch_fallback_reason = _resolve_oracle_branch_paths(
            oracle_mode=round5_category_oracle,
            selected_before=selected_before,
            gold_paths=[tuple(x) for x in getattr(sample, "gold_paths", []) if x],
            leaf_ancestor_paths=leaf_ancestor_paths,
            child_branch_paths_by_path=child_branch_paths_by_path,
            root_branch_children=root_branch_children,
            beam_size=hp.MAX_BEAM_SIZE,
        )
        # Intent: gold runner controls category generator context with branch-level oracle, while traversal stays unchanged.
        branch_descs = _collect_branch_descs_from_selected(
            selected_paths=oracle_branch_paths,
            path_to_node=path_to_node,
            max_desc_len=hp.MAX_DOC_DESC_CHAR_LEN,
            topk=hp.REWRITE_CONTEXT_TOPK,
        )
        # Intent: keep full historical category memory available to generator so labels remain stable over iterations.
        category_history_text = _format_category_history(
            category_bank=getattr(sample, "category_bank", []),
            scope=round5_category_history_scope,
        )
        leaf_cluster_triggered = (
            round5_mode == "category"
            and round5_category_drift_trigger == "leaf_cluster"
            and _is_leaf_cluster_trigger(selected_before, path_to_node)
        )
        # Intent: when near leaf-cluster regions, add a soft category-stability reminder instead of hard constraints.
        stability_hint = (
            "- Keep the last category labels unless Topic Cluster Summaries clearly indicate a new, query-relevant perspective not covered by the current labels.\n"
            if leaf_cluster_triggered
            else "\n"
        )

        rewrite_state_by_sample_idx[sample_idx] = {
            "cache_key": "",
            "cache_hit": False,
            "cached_rewrite": "",
            "cached_docs": {},
            "query_pre": query_pre,
            "query_post": query_pre,
            "leaf_descs": leaf_descs,
            "branch_descs": branch_descs,
            "rewrite": "",
            "rewrite_docs": {},
            "raw_output": "",
            "cumulative_pool_pre_size": int(len(cumulative_pool)),
            "eval_paths": [],
            "eval_doc_ids": [],
            "selected_categories": [],
            "category_prompt": "",
            "category_raw_output": "",
            "category_history": category_history_text,
            "leaf_cluster_triggered": bool(leaf_cluster_triggered),
            "stability_hint": stability_hint,
            "category_branch_source": oracle_branch_source,
            "category_branch_fallback_reason": oracle_branch_fallback_reason,
            "category_branch_paths_used": [list(path) for path in oracle_branch_paths],
        }

        if round5_mode == "category":
            cat_prompt = _format_rewrite_prompt(
                category_generator_template,
                original_query=str(sample.original_query),
                previous_rewrite=str(getattr(sample, "last_rewrite_raw", "") or ""),
                leaf_descs=leaf_descs,
                branch_descs=branch_descs,
                domain_route_hint=subset_domain_route_hint,
                category_history=category_history_text,
                stability_hint=stability_hint,
                category_k=round5_category_k,
            )
            rewrite_state_by_sample_idx[sample_idx]["category_prompt"] = cat_prompt
            category_prompts.append(cat_prompt)
            category_meta.append({"sample_idx": sample_idx})
        else:
            prompt = _format_rewrite_prompt(
                rewrite_template,
                original_query=str(sample.original_query),
                previous_rewrite=str(getattr(sample, "last_rewrite_raw", "") or ""),
                leaf_descs=leaf_descs,
                branch_descs=[],
                domain_route_hint=subset_domain_route_hint,
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
                        "branch_descs": [],
                        "query_pre": query_pre,
                        "mode": round5_mode,
                    }
                )

    if round5_mode == "category":
        if category_prompts:
            logger.info(
                "Iter %d: starting category batch (%d prompts) | oracle_mode=%s",
                iter_idx,
                len(category_prompts),
                round5_category_oracle,
            )
            cat_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(cat_loop)
            try:
                category_outputs = cat_loop.run_until_complete(llm_api.run_batch(category_prompts, **llm_api_kwargs))
            finally:
                cat_loop.close()
                asyncio.set_event_loop(None)
        else:
            category_outputs = []

        for meta, out in zip(category_meta, category_outputs):
            sample_idx = int(meta["sample_idx"])
            sample = all_eval_samples[sample_idx]
            info = rewrite_state_by_sample_idx[sample_idx]
            parsed_categories = _parse_category_output(out, round5_category_k)
            if not parsed_categories and round5_category_fallback_on_parse_fail:
                prev_labels = [str(x).strip() for x in (getattr(sample, "last_category_labels", []) or []) if str(x).strip()]
                fallback_labels = prev_labels[:round5_category_k] if prev_labels else list(DEFAULT_CATEGORY_LABELS[:round5_category_k])
                parsed_categories = [{"label": _normalize_category_label(x), "hint": ""} for x in fallback_labels if _normalize_category_label(x)]
            if (not round5_category_partial_ok) and len(parsed_categories) < round5_category_k:
                missing = [x for x in DEFAULT_CATEGORY_LABELS if x not in {c.get("label", "") for c in parsed_categories}]
                for label in missing:
                    parsed_categories.append({"label": label, "hint": ""})
                    if len(parsed_categories) >= round5_category_k:
                        break
            parsed_categories = parsed_categories[:round5_category_k]
            if not parsed_categories:
                parsed_categories = [{"label": x, "hint": ""} for x in DEFAULT_CATEGORY_LABELS[:round5_category_k]]
            info["selected_categories"] = parsed_categories
            info["category_raw_output"] = str(out or "")
            sample.last_category_labels = [str(row.get("label", "")).strip() for row in parsed_categories if str(row.get("label", "")).strip()]
            for row in parsed_categories:
                label = str(row.get("label", "")).strip()
                if not label:
                    continue
                # Intent: keep full per-iter category bank to stabilize later generation near leaf clusters.
                sample.category_bank.append(
                    {
                        "iter": int(iter_idx),
                        "label": label,
                        "hint": str(row.get("hint", "")).strip(),
                    }
                )

        for sample_idx, sample in enumerate(all_eval_samples):
            info = rewrite_state_by_sample_idx[sample_idx]
            selected_categories = info.get("selected_categories", []) or []
            selected_categories_text = _format_selected_categories(selected_categories)
            selected_category_schema = _format_selected_category_schema(selected_categories)
            prompt = _format_rewrite_prompt(
                rewrite_template,
                original_query=str(sample.original_query),
                previous_rewrite=str(getattr(sample, "last_rewrite_raw", "") or ""),
                leaf_descs=info.get("leaf_descs", []),
                branch_descs=[],
                domain_route_hint=subset_domain_route_hint,
                category_history=info.get("category_history", "None"),
                selected_categories=selected_categories_text,
                selected_category_schema=selected_category_schema,
                stability_hint=info.get("stability_hint", ""),
                category_k=round5_category_k,
            )
            cache_key = _prompt_cache_key("round5_category", prompt)
            cache_hit = (not hp.REWRITE_FORCE_REFRESH) and (cache_key in rewrite_map)
            info["cache_key"] = cache_key
            info["cache_hit"] = bool(cache_hit)
            info["cached_rewrite"] = str(rewrite_map.get(cache_key, "") or "")
            info["cached_docs"] = _clean_docs_map(docs_map.get(cache_key, {})) if cache_hit else {}
            if not cache_hit:
                rewrite_prompts.append(prompt)
                rewrite_meta.append(
                    {
                        "sample_idx": sample_idx,
                        "cache_key": cache_key,
                        "prompt": prompt,
                        "leaf_descs": info.get("leaf_descs", []),
                        "branch_descs": [],
                        "query_pre": info.get("query_pre", ""),
                        "mode": round5_mode,
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
            info = rewrite_state_by_sample_idx.get(sample_idx, {})
            docs_out, rewrite_out = _parse_possible_answer_docs(out)
            docs_out = _clean_docs_map(docs_out)
            if round5_mode == "category":
                aligned_docs = _align_docs_to_categories(
                    docs=docs_out,
                    categories=info.get("selected_categories", []),
                )
                if aligned_docs:
                    docs_out = aligned_docs
                elif round5_category_fallback_on_parse_fail:
                    selected_categories = info.get("selected_categories", [])
                    fallback_text = str(rewrite_out or _flatten_docs(docs_out) or "").strip()
                    if selected_categories:
                        primary_label = str(selected_categories[0].get("label", "theory")).strip() or "theory"
                        docs_out = {primary_label: fallback_text} if fallback_text else {}
                    else:
                        docs_out = {}
                else:
                    docs_out = {}

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
                    "prompt_name": round5_rewrite_prompt_name,
                    "llm": hp.LLM,
                    "leaf_descs": meta.get("leaf_descs", []),
                    "branch_descs": meta.get("branch_descs", []),
                    "query_pre": meta.get("query_pre", ""),
                    "mode": meta.get("mode", round5_mode),
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
        retrieval_rows.append(
            {
                "nDCG@10": compute_ndcg(eval_doc_ids[:10], gold_doc_ids, k=10) * 100,
                "Recall@10": compute_recall(eval_doc_ids[:10], gold_doc_ids, k=10) * 100,
                "Recall@100": compute_recall(eval_doc_ids[:100], gold_doc_ids, k=100) * 100,
                "Recall@all": compute_recall(eval_doc_ids, gold_doc_ids, k=len(eval_doc_ids)) * 100,
                "Coverage": float(len(eval_doc_ids)),
            }
        )
        info["cumulative_pool_eval_size"] = int(len(cumulative_pool_eval))
        info["eval_paths"] = [list(p) for p in eval_paths]
        info["eval_doc_ids"] = list(eval_doc_ids)
        info["gold_doc_ids"] = list(gold_doc_ids)
        sample.rewrite_history.append(
            {
                "iter": iter_idx,
                "cache_hit": bool(info.get("cache_hit", False)),
                "prompt_name": round5_rewrite_prompt_name,
                "query_pre": query_pre,
                "query_post": query_post,
                "rewrite": rewrite_blob,
                "possible_answer_docs": rewrite_docs,
                "selected_categories": info.get("selected_categories", []),
                "category_prompt": info.get("category_prompt", ""),
                "category_raw_output": info.get("category_raw_output", ""),
                "category_source": str(info.get("category_branch_source", "llm_selected_branches")),
                "category_oracle_mode": round5_category_oracle,
                "category_branch_fallback_reason": str(info.get("category_branch_fallback_reason", "")),
                "category_branch_paths_used": info.get("category_branch_paths_used", []),
                "leaf_cluster_triggered": bool(info.get("leaf_cluster_triggered", False)),
                "category_stability_hint": str(info.get("stability_hint", "")),
                "leaf_descs": info.get("leaf_descs", []),
                "branch_descs": info.get("branch_descs", []),
                "raw_output": raw_output,
            }
        )

    inputs = [sample.get_step_prompts() for sample in all_eval_samples]
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
    selector_candidate_paths_by_idx: Dict[int, List[Tuple[int, ...]]] = {}
    selector_scored_rows_by_idx: Dict[int, List[Dict[str, Any]]] = {}
    for sample_idx, sample in enumerate(all_eval_samples):
        selected_before = _selected_branch_paths_from_sample(sample)
        selected_before_by_idx[sample_idx] = selected_before
        sample.update(slates[sample_idx], response_jsons[sample_idx], rel_fn=sample.get_rel_fn())
        selector_pick_reason = "retriever_slate_default"
        selector_candidate_paths: List[Tuple[int, ...]] = []
        selector_scored_rows: List[Dict[str, Any]] = []
        if round5_selector_mode in {"maxscore_global", "meanscore_global", "max_hit_global"}:
            if round5_selector_mode == "maxscore_global":
                score_mode = "max"
            elif round5_selector_mode == "meanscore_global":
                score_mode = "mean"
            else:
                score_mode = "hit"
            selector_candidate_paths = _collect_candidate_child_branches(
                selected_branches=selected_before,
                child_branch_paths_by_path=child_branch_paths_by_path,
                root_branch_children=root_branch_children,
            )
            if not selector_candidate_paths:
                # Intent: if there is no expandable child-branch candidate, keep retriever-slate beam unchanged.
                selector_pick_reason = "no_candidate_children"
            else:
                selector_local_pool = _collect_leaf_pool(
                    selected_branches=selected_before,
                    leaf_indices_by_prefix=leaf_indices_by_prefix,
                    all_leaf_indices=leaf_indices,
                )
                selector_query = str(
                    rewrite_state_by_sample_idx.get(sample_idx, {}).get("query_post", "")
                    or sample.query
                    or getattr(sample, "original_query", "")
                )
                selector_local_hits = _retrieve_leaf_hits(
                    query=selector_query,
                    leaf_pool_indices=selector_local_pool,
                    retriever=retriever,
                    node_embs=node_embs,
                    node_registry=node_registry,
                    topk=round5_mrr_pool_k,
                )
                if not selector_local_hits:
                    selector_pick_reason = "no_local_hits"
                else:
                    selector_scored_rows = _score_candidate_branches_score(
                        local_hits=selector_local_hits,
                        candidate_child_paths=selector_candidate_paths,
                        leaf_ancestor_paths=leaf_ancestor_paths,
                        score_mode=score_mode,
                    )
                    if not selector_scored_rows:
                        selector_pick_reason = "no_candidate_match"
                    else:
                        selector_target_paths = [
                            tuple(row.get("path", []))
                            for row in selector_scored_rows[: max(1, int(hp.MAX_BEAM_SIZE))]
                            if row.get("path", [])
                        ]
                        expandable_state_path_map = _build_expandable_state_path_map(sample)
                        selected_state_paths: List[List[object]] = []
                        selected_endpoints: Set[Tuple[int, ...]] = set()
                        for path_t in selector_target_paths:
                            matched_state_path = expandable_state_path_map.get(tuple(path_t))
                            if not matched_state_path:
                                continue
                            endpoint = tuple(getattr(matched_state_path[-1], "path", ()) or ())
                            if not endpoint or endpoint in selected_endpoints:
                                continue
                            selected_endpoints.add(endpoint)
                            selected_state_paths.append(matched_state_path)
                        if not selected_state_paths:
                            selector_pick_reason = "no_expandable_match"
                        else:
                            fallback_state_paths = list(getattr(sample, "beam_state_paths", None) or [])
                            merged_state_paths = list(selected_state_paths)
                            merged_endpoints = {
                                tuple(getattr(path[-1], "path", ()) or ())
                                for path in merged_state_paths
                                if path
                            }
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
                            selector_pick_reason = (
                                f"{round5_selector_mode}_applied_full"
                                if len(selected_state_paths) >= max(1, int(hp.MAX_BEAM_SIZE))
                                else f"{round5_selector_mode}_applied_partial_fill"
                            )
        selector_pick_reason_by_idx[sample_idx] = selector_pick_reason
        selector_candidate_paths_by_idx[sample_idx] = selector_candidate_paths
        selector_scored_rows_by_idx[sample_idx] = [
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
            for row in selector_scored_rows[:3]
        ]
        reached_after = sample.get_top_predictions(k=None, rel_fn=sample.get_rel_fn(leaf=True))
        for node, _ in reached_after:
            ridx = int(getattr(node, "registry_idx", -1))
            if ridx >= 0:
                cumulative_leaf_indices_by_sample[sample_idx].add(ridx)

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
                "rewrite_branch_descs_count": int(len(info.get("branch_descs", []))),
                "possible_answer_docs": info.get("rewrite_docs", {}),
                "selected_categories": info.get("selected_categories", []),
                "category_source": str(info.get("category_branch_source", "llm_selected_branches")),
                "category_oracle_mode": round5_category_oracle,
                "category_branch_fallback_reason": str(info.get("category_branch_fallback_reason", "")),
                "category_branch_paths_used": info.get("category_branch_paths_used", []),
                "leaf_cluster_triggered": bool(info.get("leaf_cluster_triggered", False)),
                "selected_branches_before": [list(p) for p in selected_before_by_idx.get(sample_idx, [])],
                "selected_branches_after": [list(p) for p in selected_after],
                "selector_mode": round5_selector_mode,
                "selector_pick_reason": str(selector_pick_reason_by_idx.get(sample_idx, "retriever_slate_default")),
                "selector_candidate_branch_count": int(len(selector_candidate_paths_by_idx.get(sample_idx, []))),
                "selector_scored_top": selector_scored_rows_by_idx.get(sample_idx, []),
                "candidate_branch_count": int(sum(len(x) for x in slates[sample_idx])) if sample_idx < len(slates) else 0,
                "cumulative_pool_pre_size": int(info.get("cumulative_pool_pre_size", 0)),
                "cumulative_pool_eval_size": int(info.get("cumulative_pool_eval_size", 0)),
                "local_paths": info.get("eval_paths", []),
                "local_doc_ids": info.get("eval_doc_ids", []),
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
logger.info("Saved Round5 results to %s", RESULTS_DIR)
