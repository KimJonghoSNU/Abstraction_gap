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
from flat_then_tree import FlatHit, flat_retrieve_hits, is_prefix, rrf_fuse_ranked_paths
from hyperparams import HyperParams
from llm_apis import GenAIAPI, VllmAPI
from retrievers.diver import DiverEmbeddingModel, cosine_topk
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
    rewrite_history: List[Dict] = None
    iter_records: List[Dict] = None

    def __post_init__(self) -> None:
        if self.rewrite_history is None:
            self.rewrite_history = []
        if self.iter_records is None:
            self.iter_records = []

    def to_dict(self) -> Dict:
        return {
            "original_query": self.original_query,
            "gold_paths": self.gold_paths,
            "excluded_ids": self.excluded_ids,
            "last_rewrite": self.last_rewrite,
            "last_action": self.last_action,
            "rewrite_history": self.rewrite_history,
            "iter_records": self.iter_records,
        }


def _format_action_prompt(
    template: str,
    original_query: str,
    previous_rewrite: str,
    leaf_descs: List[str],
    branch_descs: List[str],
) -> str:
    leaf_blob = "\n".join([x for x in leaf_descs if x])
    branch_blob = "\n".join([x for x in branch_descs if x])
    if not branch_blob:
        template = template.replace("Branch Context:\n{branch_descs}\n", "")
    try:
        return template.format(
            original_query=(original_query or ""),
            previous_rewrite=(previous_rewrite or ""),
            leaf_descs=leaf_blob,
            branch_descs=branch_blob,
        )
    except KeyError:
        return (
            template
            .replace("{original_query}", original_query or "")
            .replace("{previous_rewrite}", previous_rewrite or "")
            .replace("{leaf_descs}", leaf_blob)
            .replace("{branch_descs}", branch_blob)
        )


def _parse_action_rewrite(text: str) -> Tuple[str, str]:
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
        action = str(obj.get("action", "exploit")).strip().lower()
        if action not in {"explore", "exploit"}:
            action = "exploit"
        if "Possible_Answer_Docs" in obj:
            docs_map = obj.get("Possible_Answer_Docs")
            if isinstance(docs_map, dict):
                flattened = "\n".join([str(v) for v in docs_map.values() if v])
                return action, flattened.strip()
            if isinstance(docs_map, list):
                flattened = "\n".join([str(v) for v in docs_map if v])
                return action, flattened.strip()
        rewrite = obj.get("rewrite", "")
        if isinstance(rewrite, (list, dict)):
            rewrite = json.dumps(rewrite, ensure_ascii=False)
        return action, str(rewrite or "").strip()
    return "exploit", cleaned.strip()


def _apply_action_rewrite(original_query: str, action: str, rewrite: str, explore_mode: str) -> str:
    rewrite = (rewrite or "").strip()
    if not rewrite:
        return original_query
    if action == "explore":
        if explore_mode == "original":
            return original_query
        return rewrite
    return (original_query + " " + rewrite).strip() # action == exploit


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

    active = set(leaf_prefix_counts.keys()) | set(branch_scores.keys())
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


def _retrieve_hits_subset(
    *,
    retriever: DiverEmbeddingModel,
    query: str,
    node_embs: np.ndarray,
    subset_indices: Sequence[int],
    node_registry: Sequence[object],
    topk: int,
) -> List[FlatHit]:
    if node_embs.size == 0 or len(subset_indices) == 0:
        return []
    q_emb = retriever.encode_query(query)
    k = min(topk, node_embs.shape[0])
    res = cosine_topk(q_emb, node_embs, k)
    hits: List[FlatHit] = []
    for ridx, score in zip(res.indices.tolist(), res.scores.tolist()):
        registry_idx = int(subset_indices[int(ridx)])
        node = node_registry[registry_idx]
        hits.append(FlatHit(registry_idx=registry_idx, path=tuple(node.path), score=float(score), is_leaf=node.is_leaf))
    return hits


def _load_rewrite_action_cache(path: str, force_refresh: bool) -> Tuple[Dict[str, str], Dict[str, str]]:
    rewrite_map: Dict[str, str] = {}
    action_map: Dict[str, str] = {}
    if not path or force_refresh or (not os.path.exists(path)):
        return rewrite_map, action_map
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
            if "action" in rec:
                action_map[str(key)] = str(rec.get("action", "exploit")).strip().lower()
    return rewrite_map, action_map


hp = HyperParams.from_args()
if not hp.REWRITE_PROMPT_NAME and not hp.REWRITE_PROMPT_PATH and not hp.REWRITE_CACHE_PATH:
    hp.add_param("rewrite_prompt_name", "round3_action_v1")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
exp_dir_name = str(hp)
RESULTS_DIR = f"{BASE_DIR}/results/{hp.DATASET}/{hp.SUBSET}/{exp_dir_name}/"
os.makedirs(RESULTS_DIR, exist_ok=True)
log_path = f"{RESULTS_DIR}/run.log"
os.makedirs(os.path.dirname(log_path), exist_ok=True)
logger = setup_logger("lattice_runner_round3", log_path, logging.INFO)
with open(f"{RESULTS_DIR}/hparams.json", "w", encoding="utf-8") as f:
    json.dump(vars(hp), f, indent=2, ensure_ascii=True, sort_keys=True)

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
leaf_embs = node_embs[leaf_indices] if leaf_indices else np.zeros((0, node_embs.shape[1]), dtype=node_embs.dtype)

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
    rewrite_template = REWRITE_PROMPT_TEMPLATES[hp.REWRITE_PROMPT_NAME]
if hp.REWRITE_PROMPT_PATH:
    if not os.path.exists(hp.REWRITE_PROMPT_PATH):
        raise ValueError(f"--rewrite_prompt_path not found: {hp.REWRITE_PROMPT_PATH}")
    with open(hp.REWRITE_PROMPT_PATH, "r", encoding="utf-8") as f:
        rewrite_template = f.read()
rewrite_map, action_map = _load_rewrite_action_cache(hp.REWRITE_CACHE_PATH, hp.REWRITE_FORCE_REFRESH)

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

    anchor_hits_by_sample: List[List[FlatHit]] = []
    leaf_hits_by_sample: List[List[FlatHit]] = []
    branch_hits_by_sample: List[List[FlatHit]] = []
    branch_paths_by_sample: List[List[Tuple[int, ...]]] = []
    densities_by_sample: List[Dict[Tuple[int, ...], float]] = []

    for sample in all_eval_samples:
        if iter_idx == 0 and sample is all_eval_samples[0]:
            logger.info("Iter %d: starting anchor retrieval", iter_idx)
        anchor_query = _apply_action_rewrite(sample.original_query, sample.last_action, sample.last_rewrite, hp.ROUND3_EXPLORE_MODE)
        hits = flat_retrieve_hits(
            retriever=retriever,
            query=anchor_query,
            node_embs=node_embs,
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

        # NOTE: In this dataset, leaf depth is always > 1. If depth-1 leaves appear later,
        # they will not contribute prefixes to B_active.
        if hp.REWRITE_EVERY <= 0:
            do_rewrite = False
        elif hp.ROUND3_REWRITE_ONCE:
            do_rewrite = (iter_idx == 0) and (not sample.last_rewrite)
        else:
            do_rewrite = (iter_idx % hp.REWRITE_EVERY == 0) or (not sample.last_rewrite)
        if rewrite_enabled and do_rewrite:
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
            cache_key = _rewrite_cache_key(
                "round3",
                sample.original_query,
                cache_descs,
                iter_idx=iter_idx,
            )
            if (not hp.REWRITE_FORCE_REFRESH) and (cache_key in rewrite_map):
                rewrite = rewrite_map[cache_key]
                action = action_map.get(cache_key, "exploit")
                sample.last_rewrite = rewrite
                sample.last_action = action
                sample.rewrite_history.append({
                    "iter": iter_idx,
                    "cache_hit": True,
                    "action": action,
                    "rewrite": rewrite,
                })
            else:
                prompt = _format_action_prompt(
                    rewrite_template,
                    sample.original_query,
                    sample.last_rewrite,
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
            action, rewrite = _parse_action_rewrite(out)
            sample = meta["sample"]
            sample.last_rewrite = rewrite
            sample.last_action = action
            sample.rewrite_history.append({
                "iter": iter_idx,
                "cache_hit": False,
                "action": action,
                "rewrite": rewrite,
            })
            rewrite_map[meta["cache_key"]] = rewrite
            action_map[meta["cache_key"]] = action
            new_records.append({
                "key": meta["cache_key"],
                "rewritten_query": rewrite,
                "action": action,
                "prompt_name": hp.REWRITE_PROMPT_NAME,
                "llm": hp.LLM,
                "leaf_descs": meta.get("leaf_descs", []),
                "branch_descs": meta.get("branch_descs", []),
            })
        if hp.REWRITE_CACHE_PATH and new_records:
            append_jsonl(hp.REWRITE_CACHE_PATH, new_records)

    logger.info("Iter %d: starting local/global retrieval", iter_idx)
    for sample, leaf_hits, branch_hits, active_paths, densities in zip(
        all_eval_samples,
        leaf_hits_by_sample,
        branch_hits_by_sample,
        branch_paths_by_sample,
        densities_by_sample,
    ):
        query_t = _apply_action_rewrite(sample.original_query, sample.last_action, sample.last_rewrite, hp.ROUND3_EXPLORE_MODE)
        local_leaf_indices = _filter_leaf_indices_by_prefixes(leaf_indices, leaf_paths, active_paths)
        local_leaf_embs = node_embs[local_leaf_indices] if local_leaf_indices else np.zeros((0, node_embs.shape[1]), dtype=node_embs.dtype)
        local_hits = _retrieve_hits_subset(
            retriever=retriever,
            query=query_t,
            node_embs=local_leaf_embs,
            subset_indices=local_leaf_indices,
            node_registry=node_registry,
            topk=local_topk,
        )
        global_hits = _retrieve_hits_subset(
            retriever=retriever,
            query=query_t,
            node_embs=leaf_embs,
            subset_indices=leaf_indices,
            node_registry=node_registry,
            topk=global_topk,
        )

        local_ranked = [(h.path, h.score) for h in local_hits]
        global_ranked = [(h.path, h.score) for h in global_hits]
        ranked_lists = [lst for lst in (local_ranked, global_ranked) if lst]
        fused_ranked = rrf_fuse_ranked_paths(ranked_lists, k=hp.ROUND3_RRF_K) if ranked_lists else []

        fused_paths = [list(p) for p, _ in fused_ranked]
        gold_paths = [list(p) for p in sample.gold_paths]
        metrics = {
            "nDCG@10": compute_ndcg(fused_paths[:10], gold_paths, k=10) * 100,
            "Recall@10": compute_recall(fused_paths[:10], gold_paths, k=10) * 100,
            "Recall@100": compute_recall(fused_paths[:100], gold_paths, k=100) * 100,
            "Recall@all": compute_recall(fused_paths, gold_paths, k=len(fused_paths)) * 100,
            "Coverage": len(fused_paths),
        }
        iter_rows.append(metrics)

        sample.iter_records.append({
            "iter": iter_idx,
            "action": sample.last_action,
            "rewrite": sample.last_rewrite,
            "query_t": query_t,
            "anchor_leaf_paths": [h.path for h in leaf_hits],
            "anchor_branch_paths": [h.path for h in branch_hits],
            "active_branch_paths": active_paths,
            "density": {str(k): v for k, v in densities.items()},
            "local_paths": [h.path for h in local_hits],
            "global_paths": [h.path for h in global_hits],
            "fused_paths": [p for p, _ in fused_ranked[:100]],
        })

    iter_df = pd.DataFrame(iter_rows)
    all_eval_metric_dfs.append(iter_df)
    if not iter_df.empty:
        logger.info(
            "Iter %d | nDCG@10 mean=%.2f | Recall@100 mean=%.2f",
            iter_idx,
            iter_df["nDCG@10"].mean(),
            iter_df["Recall@100"].mean(),
        )
    else:
        logger.info("Iter %d | nDCG@10 mean=0.00 | Recall@100 mean=0.00", iter_idx)

save_exp(RESULTS_DIR, hp, llm_api, all_eval_samples, all_eval_metric_dfs, allow_overwrite=True, save_llm_api_history=True)
logger.info("Saved Round3 results to %s", RESULTS_DIR)
