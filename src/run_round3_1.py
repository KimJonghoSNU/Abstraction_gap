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

from cache_utils import _rewrite_cache_key, append_jsonl
from flat_then_tree import FlatHit, gate_hit, is_prefix, rrf_fuse_ranked_paths
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
) -> str:
    leaf_blob = "\n".join([x for x in leaf_descs if x])
    branch_blob = "\n".join([x for x in branch_descs if x])
    prev_lines = []
    for key in CATEGORY_ORDER:
        val = (previous_docs or {}).get(key, "")
        if val:
            prev_lines.append(f"- {key}: {val}")
    prev_blob = "\n".join(prev_lines) if prev_lines else "None"
    if not branch_blob:
        template = template.replace("Branch Context:\n{branch_descs}\n", "")
    try:
        return template.format(
            original_query=(original_query or ""),
            previous_rewrite=(previous_rewrite or ""),
            previous_docs=prev_blob,
            leaf_descs=leaf_blob,
            branch_descs=branch_blob,
        )
    except KeyError:
        return (
            template
            .replace("{original_query}", original_query or "")
            .replace("{previous_rewrite}", previous_rewrite or "")
            .replace("{previous_docs}", prev_blob)
            .replace("{leaf_descs}", leaf_blob)
            .replace("{branch_descs}", branch_blob)
        )


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
node_by_path = {tuple(node.path): node for node in node_registry}

if not hp.RETRIEVER_MODEL_PATH:
    raise ValueError("--retriever_model_path is required")
if not hp.NODE_EMB_PATH:
    raise ValueError("--node_emb_path is required")

node_embs = np.load(hp.NODE_EMB_PATH, allow_pickle=False)
if node_embs.shape[0] != len(node_registry):
    raise ValueError(f"node_embs rows ({node_embs.shape[0]}) must match node_registry size ({len(node_registry)})")

retriever = DiverEmbeddingModel(hp.RETRIEVER_MODEL_PATH, local_files_only=True)

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

    rewrite_prompts: List[str] = []
    rewrite_meta: List[Dict] = []

    anchor_topk_logged = max(1, min(10, int(anchor_topk)))
    anchor_hits_by_sample: List[List[FlatHit]] = []
    leaf_hits_by_sample: List[List[FlatHit]] = []
    branch_hits_by_sample: List[List[FlatHit]] = []
    local_hits_by_sample: List[List[FlatHit]] = []
    global_hits_by_sample: List[List[FlatHit]] = []
    branch_paths_by_sample: List[List[Tuple[int, ...]]] = []
    densities_by_sample: List[Dict[Tuple[int, ...], float]] = []
    retrieval_queries_by_sample: List[str] = []

    # tqdm shows per-iteration retrieval progress in terminal runs.
    for sample in tqdm(
        all_eval_samples,
        desc=f"Iter {iter_idx} anchor retrieval",
        total=len(all_eval_samples),
        leave=False,
    ):
        if iter_idx == 0 and sample is all_eval_samples[0]:
            logger.info("Iter %d: starting anchor retrieval", iter_idx)
        #   - Single‑action mode → _apply_action_rewrite is used every iter.
        #   - Per‑level actions mode → _compose_query_from_docs is used instead, once last_possible_docs is filled.
        if sample.last_possible_docs:
            anchor_query = _compose_query_from_docs(sample.original_query, sample.last_possible_docs, sample.last_actions)
        else:
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
        leaf_hits = [h for h in hits if h.is_leaf]
        branch_hits = [h for h in hits if not h.is_leaf]
        leaf_hits_by_sample.append(leaf_hits)
        branch_hits_by_sample.append(branch_hits)

        active_branches, densities, branch_scores = _build_active_branches(leaf_hits, branch_hits)
        ranked_branches = _rank_branch_paths(active_branches, densities, branch_scores)
        branch_paths_by_sample.append(ranked_branches)
        densities_by_sample.append(densities)
        retrieval_queries_by_sample.append(anchor_query)

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
            if hp.ROUND3_ANCHOR_LOCAL_RANK == "v2":
                context_paths = _anchor_local_context_paths(
                    anchor_hits=hits,
                    anchor_topk=anchor_topk_logged,
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
                leaf_descs = _hits_to_context_descs(
                    leaf_hits,
                    node_registry,
                    hp.REWRITE_CONTEXT_TOPK,
                    hp.MAX_DOC_DESC_CHAR_LEN,
                )
            if hp.ROUND3_REWRITE_CONTEXT == "leaf_branch":
                branch_descs = []
                for path in ranked_branches:
                    node = node_by_path.get(tuple(path))
                    if not node:
                        continue
                    desc = node.desc
                    if hp.MAX_DOC_DESC_CHAR_LEN:
                        desc = desc[: hp.MAX_DOC_DESC_CHAR_LEN]
                    branch_descs.append(desc)
                    if len(branch_descs) >= hp.REWRITE_CONTEXT_TOPK:
                        break
            else:
                branch_descs = []

            cache_descs = [f"LEAF: {d}" for d in leaf_descs] + [f"BRANCH: {d}" for d in branch_descs]
            prev_blob = "\n".join([f"{k}: {v}" for k, v in sample.last_possible_docs.items() if v])
            cache_key = _rewrite_cache_key(
                "round3",
                f"{sample.original_query}||{prev_blob}",
                cache_descs,
                iter_idx=iter_idx,
            )
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
                prompt = _format_action_prompt(
                    rewrite_template,
                    sample.original_query,
                    sample.last_rewrite,
                    sample.last_possible_docs,
                    leaf_descs,
                    branch_descs,
                )
                rewrite_prompts.append(prompt)
                rewrite_meta.append({
                    "sample": sample,
                    "cache_key": cache_key,
                    "leaf_descs": leaf_descs,
                    "branch_descs": branch_descs,
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
    for sample, leaf_hits, branch_hits, local_hits, global_hits, active_paths, densities, query_t in tqdm(
        zip(
            all_eval_samples,
            leaf_hits_by_sample,
            branch_hits_by_sample,
            local_hits_by_sample,
            global_hits_by_sample,
            branch_paths_by_sample,
            densities_by_sample,
            retrieval_queries_by_sample,
        ),
        desc=f"Iter {iter_idx} local/global scoring",
        total=len(all_eval_samples),
        leave=False,
    ):
        local_ranked = [(h.path, h.score) for h in local_hits]
        global_ranked = [(h.path, h.score) for h in global_hits]
        ranked_lists = [lst for lst in (local_ranked, global_ranked) if lst]
        fused_ranked = rrf_fuse_ranked_paths(ranked_lists, k=hp.ROUND3_RRF_K) if ranked_lists else []

        fused_paths = [list(p) for p, _ in fused_ranked]
        local_paths = [list(h.path) for h in local_hits]
        global_paths = [list(h.path) for h in global_hits]
        gold_paths = [list(p) for p in sample.gold_paths]
        rrf_metrics = {
            "nDCG@10": compute_ndcg(fused_paths[:10], gold_paths, k=10) * 100,
            "Recall@10": compute_recall(fused_paths[:10], gold_paths, k=10) * 100,
            "Recall@100": compute_recall(fused_paths[:100], gold_paths, k=100) * 100,
            "Recall@all": compute_recall(fused_paths, gold_paths, k=len(fused_paths)) * 100,
            "Coverage": len(fused_paths),
        }
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
        metrics = {
            "nDCG@10": rrf_metrics["nDCG@10"],
            "Recall@10": rrf_metrics["Recall@10"],
            "Recall@100": rrf_metrics["Recall@100"],
            "Recall@all": rrf_metrics["Recall@all"],
            "Coverage": rrf_metrics["Coverage"],
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
            "anchor_leaf_paths": [h.path for h in leaf_hits],
            "anchor_branch_paths": [h.path for h in branch_hits],
            "active_branch_paths": active_paths,
            "density": {str(k): v for k, v in densities.items()},
            "local_paths": local_paths,
            "global_paths": global_paths,
            "fused_paths": [p for p, _ in fused_ranked[:100]],
            "local_metrics": local_metrics,
            "global_metrics": global_metrics,
            "rrf_metrics": rrf_metrics,
        })

    iter_df = pd.DataFrame(iter_rows)
    all_eval_metric_dfs.append(iter_df)
    if not iter_df.empty:
        logger.info(
            "Iter %d | RRF nDCG@10=%.2f | Local nDCG@10=%.2f | Global nDCG@10=%.2f | "
            "RRF Recall@100=%.2f | Local Recall@100=%.2f | Global Recall@100=%.2f",
            iter_idx,
            iter_df["nDCG@10"].mean(),
            iter_df["Local_nDCG@10"].mean(),
            iter_df["Global_nDCG@10"].mean(),
            iter_df["Recall@100"].mean(),
            iter_df["Local_Recall@100"].mean(),
            iter_df["Global_Recall@100"].mean(),
        )
    else:
        logger.info(
            "Iter %d | RRF nDCG@10=0.00 | Local nDCG@10=0.00 | Global nDCG@10=0.00 | "
            "RRF Recall@100=0.00 | Local Recall@100=0.00 | Global Recall@100=0.00",
            iter_idx,
        )

save_exp(RESULTS_DIR, hp, llm_api, all_eval_samples, all_eval_metric_dfs, allow_overwrite=True, save_llm_api_history=True)
logger.info("Saved Round3 results to %s", RESULTS_DIR)
