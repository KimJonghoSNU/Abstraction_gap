#region Imports
import numpy as np
import pandas as pd
import pickle as pkl
import asyncio
import json
import hashlib
from tqdm.autonotebook import tqdm
import os
import logging
from typing import List
from datasets import load_dataset
from google.genai import types
from json_repair import repair_json
from hyperparams import HyperParams
from tree_objects import SemanticNode, InferSample
from llm_apis import GenAIAPI, VllmAPI
from prompts import get_traversal_prompt_response_constraint, get_reranking_prompt
from rewrite_prompts import REWRITE_PROMPT_TEMPLATES
from qe_prompts import QE_PROMPT_TEMPLATES
from rewrite_pipeline import SchemaGenerator
from cache_utils import _rewrite_cache_key, load_rewrite_cache, append_jsonl
from flat_then_tree import (
    ancestor_hit,
    build_gates_and_leaf_candidates,
    flat_retrieve_hits,
    gate_hit,
    rrf_fuse_ranked_paths,
)
from utils import (
    setup_logger, 
    compute_ndcg,
    compute_recall,
    compute_node_registry,
    get_all_leaf_nodes_with_path, 
    get_node_id, 
    post_process, 
    save_exp, 
    load_exp,
    init_wandb_logging,
    finish_wandb_logging,
    wandb_log_iteration_metrics,
    wandb_log_reranking_metrics,
    wandb_log_final_summary,
)
np.random.seed(42)
#endregion


def _format_qe_prompt(template: str, query: str) -> str:
    if "{query}" in template:
        try:
            return template.format(query=query)
        except KeyError:
            # Fallback for templates containing JSON braces without escaping.
            return template.replace("{query}", query)
    return template.rstrip() + "\n\n" + query

def _clean_qe_text(text: str) -> str:
    text = text.split("</think>\n")[-1].strip()
    if "```" in text:
        try:
            parts = text.split("```")
            fenced = [parts[i] for i in range(1, len(parts), 2)]
            if fenced:
                text = fenced[-1].strip()
        except Exception:
            pass
    # If JSON is returned, flatten Possible_Answer_Docs into a single query string.
    try:
        obj = json.loads(text)
    except Exception:
        try:
            obj = repair_json(text, return_objects=True)
        except Exception:
            obj = None
    if isinstance(obj, dict):
        docs_map = obj.get("Possible_Answer_Docs")
        if isinstance(docs_map, dict):
            flattened = "\n".join([str(v) for v in docs_map.values() if v])
            return flattened.strip()
        if isinstance(docs_map, list):
            flattened = "\n".join([str(v) for v in docs_map if v])
            return flattened.strip()
    return text.strip()

def _parse_rewrite_output(text: str) -> tuple[str, object | None]:
    cleaned = _clean_qe_text(text)
    raw = text.split("</think>\n")[-1].strip()
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
    if isinstance(obj, dict) and "Possible_Answer_Docs" in obj:
        return cleaned, obj.get("Possible_Answer_Docs")
    return cleaned, None

def _format_rewrite_prompt(
    template: str,
    original_query: str,
    previous_rewrite: str,
    context_descs: List[str],
    schema_labels: List[str] | None = None,
) -> str:
    context_blob = "\n".join([x for x in context_descs if x])
    schema_blob = ""
    schema_kv_template = ""
    if schema_labels:
        schema_items = [label for label in schema_labels if label]
        schema_blob = "\n".join([f"- {label}" for label in schema_items])
        schema_lines = []
        for idx, label in enumerate(schema_items):
            comma = "," if idx < len(schema_items) - 1 else ""
            schema_lines.append(f"    \"{label}\": \"...\"{comma}")
        schema_kv_template = "\n".join(schema_lines)
    try:
        return template.format(
            gate_descs=context_blob,
            original_query=(original_query or ""),
            previous_rewrite=(previous_rewrite or ""),
            schema_labels=schema_blob,
            schema_kv_template=schema_kv_template,
        )
    except KeyError:
        # Fallback for templates containing JSON braces without escaping.
        return (
            template
            .replace("{gate_descs}", context_blob)
            .replace("{original_query}", original_query or "")
            .replace("{previous_rewrite}", previous_rewrite or "")
            .replace("{schema_labels}", schema_blob)
            .replace("{schema_kv_template}", schema_kv_template)
        )

def _apply_rewrite(mode: str, base_query: str, rewrite: str) -> str:
    rewrite = (rewrite or "").strip()
    if not rewrite:
        return base_query
    if mode == "replace":
        return rewrite
    return (base_query + " " + rewrite).strip()

def _compose_query(original_query: str, rewrite: str | None) -> str:
    rewrite = (rewrite or "").strip()
    if not rewrite:
        return original_query
    return (original_query + " " + rewrite).strip()

def _get_prev_rewrite(sample) -> str:
    history = getattr(sample, "rewrite_history", [])
    if history:
        return str(history[-1].get("rewrite", "")).strip()
    return ""

def _slates_to_context_descs(slates: List[List[int]], node_registry: List[object], topk: int, max_desc_len: int | None) -> List[str]:
    descs: List[str] = []
    seen: set[int] = set()
    for slate in slates:
        for ridx in slate:
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

def _hits_to_context_descs(hits: List[object], node_registry: List[object], topk: int, max_desc_len: int | None) -> List[str]:
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

def _ranked_paths_to_context_descs(
    ranked_paths: List[tuple[tuple[int, ...], float]],
    node_by_path: dict[tuple[int, ...], object],
    topk: int,
    max_desc_len: int | None,
) -> List[str]:
    descs: List[str] = []
    for path, _score in ranked_paths[:topk]:
        node = node_by_path.get(tuple(path))
        if not node:
            continue
        desc = node.desc
        if max_desc_len:
            desc = desc[:max_desc_len]
        descs.append(desc)
    return descs

def _normalize_schema_label(label: str) -> str:
    return " ".join((label or "").strip().lower().split())

def _get_rewrite_context(
    sample,
    sample_slates: List[List[int]] | None,
    source: str,
    node_by_path: dict[tuple[int, ...], object],
    node_registry: List[object],
    topk: int,
    max_desc_len: int | None,
) -> List[str]:
    if source == "flat":
        context = getattr(sample, "flat_context_descs", [])
        if context:
            return context[:topk]
    if source == "fused":
        flat_leaf_ranked = getattr(sample, "flat_leaf_ranked", [])
        if flat_leaf_ranked:
            traversal_ranked = [
                (tuple(x.path), float(s))
                for x, s in sample.get_top_predictions(200, rel_fn=sample.get_rel_fn(leaf=True))
            ]
            fused = rrf_fuse_ranked_paths([traversal_ranked, flat_leaf_ranked], k=100)
            context = _ranked_paths_to_context_descs(fused, node_by_path, topk, max_desc_len)
            if context:
                return context
    if source == "mixed" and sample_slates is not None:
        fused_paths = []
        flat_leaf_ranked = getattr(sample, "flat_leaf_ranked", [])
        if flat_leaf_ranked:
            traversal_ranked = [
                (tuple(x.path), float(s))
                for x, s in sample.get_top_predictions(200, rel_fn=sample.get_rel_fn(leaf=True))
            ]
            fused_paths = rrf_fuse_ranked_paths([traversal_ranked, flat_leaf_ranked], k=100)
        slate_ranked = []
        seen = set()
        for slate in sample_slates:
            for ridx in slate:
                if ridx in seen:
                    continue
                seen.add(ridx)
                slate_ranked.append((tuple(node_registry[ridx].path), 1.0))
        if slate_ranked:
            mixed = rrf_fuse_ranked_paths([fused_paths, slate_ranked] if fused_paths else [slate_ranked], k=100)
            context = _ranked_paths_to_context_descs(mixed, node_by_path, topk, max_desc_len)
            if context:
                return context
    if source == "slate" and sample_slates is not None:
        return _slates_to_context_descs(sample_slates, node_registry, topk, max_desc_len)
    # fallback
    context = getattr(sample, "flat_context_descs", [])
    if context:
        return context[:topk]
    if sample_slates is not None:
        return _slates_to_context_descs(sample_slates, node_registry, topk, max_desc_len)
    return []

def _count_slate_depths(slates: List[List[int]], node_registry: List[object], topk: int) -> dict[int, int]:
    depth_counts: dict[int, int] = {}
    seen: set[int] = set()
    for slate in slates:
        for ridx in slate:
            if ridx in seen:
                continue
            seen.add(ridx)
            depth = len(node_registry[ridx].path)
            depth_counts[depth] = depth_counts.get(depth, 0) + 1
            if topk and len(seen) >= topk:
                return depth_counts
    return depth_counts


def _ranked_paths_from_context_source(
    sample,
    sample_slates: List[List[int]] | None,
    source: str,
    node_registry: List[object],
) -> List[tuple[tuple[int, ...], float]]:
    if source == "flat":
        return getattr(sample, "flat_leaf_ranked", [])
    if source == "fused":
        flat_leaf_ranked = getattr(sample, "flat_leaf_ranked", [])
        traversal_ranked = [
            (tuple(x.path), float(s))
            for x, s in sample.get_top_predictions(200, rel_fn=sample.get_rel_fn(leaf=True))
        ]
        if flat_leaf_ranked:
            return rrf_fuse_ranked_paths([traversal_ranked, flat_leaf_ranked], k=100)
        return traversal_ranked
    if source == "slate":
        if not sample_slates:
            return []
        slate_ranked = []
        seen = set()
        for slate in sample_slates:
            for ridx in slate:
                if ridx in seen:
                    continue
                seen.add(ridx)
                slate_ranked.append((tuple(node_registry[ridx].path), 1.0))
        return slate_ranked
    if source == "mixed":
        if not sample_slates:
            return []
        fused_paths = []
        flat_leaf_ranked = getattr(sample, "flat_leaf_ranked", [])
        if flat_leaf_ranked:
            traversal_ranked = [
                (tuple(x.path), float(s))
                for x, s in sample.get_top_predictions(200, rel_fn=sample.get_rel_fn(leaf=True))
            ]
            fused_paths = rrf_fuse_ranked_paths([traversal_ranked, flat_leaf_ranked], k=100)
        slate_ranked = []
        seen = set()
        for slate in sample_slates:
            for ridx in slate:
                if ridx in seen:
                    continue
                seen.add(ridx)
                slate_ranked.append((tuple(node_registry[ridx].path), 1.0))
        if not slate_ranked:
            return fused_paths
        return rrf_fuse_ranked_paths([fused_paths, slate_ranked] if fused_paths else [slate_ranked], k=100)
    return []

#region Setup
hp = HyperParams.from_args()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
exp_dir_name = str(hp)
RESULTS_DIR = f'{BASE_DIR}/results/{hp.DATASET}/{hp.SUBSET}/{exp_dir_name}/'
os.makedirs(RESULTS_DIR, exist_ok=True)
log_path = f"{RESULTS_DIR}/run.log"
os.makedirs(os.path.dirname(log_path), exist_ok=True)
logger = setup_logger('lattice_runner', log_path, logging.INFO)
with open(f"{RESULTS_DIR}/hparams.json", "w", encoding="utf-8") as f:
    json.dump(vars(hp), f, indent=2, ensure_ascii=True, sort_keys=True)

# Initialize wandb logging
# run_name = init_wandb_logging(hp, RESULTS_DIR)
# logger.info(f"Initialized wandb run: {run_name}")
#endregion

#region Data loading
if os.path.exists(f'{BASE_DIR}/data/{hp.DATASET}/{hp.SUBSET}/documents.jsonl'):
    docs_df = pd.read_json(f'{BASE_DIR}/data/{hp.DATASET}/{hp.SUBSET}/documents.jsonl', lines=True, dtype={'id': str})
    examples_df = pd.read_json(f'{BASE_DIR}/data/{hp.DATASET}/{hp.SUBSET}/examples.jsonl', lines=True, dtype={'gold_ids': List[str]})
    examples_df['gold_ids'] = examples_df['gold_ids'].apply(lambda x: [str(i) for i in x])
else:
    docs_df = pd.DataFrame(load_dataset('xlangai/BRIGHT', 'documents', split=hp.SUBSET))
    examples_df = pd.DataFrame(load_dataset('xlangai/BRIGHT', 'examples', split=hp.SUBSET))
    
doc_id_to_content = {docs_df.iloc[i].id: docs_df.iloc[i].content for i in range(len(docs_df))}

tree_dict = pkl.load(open(f'{BASE_DIR}/trees/{hp.DATASET}/{hp.SUBSET}/tree-{hp.TREE_VERSION}.pkl', 'rb'))
semantic_root_node = SemanticNode().load_dict(tree_dict) if isinstance(tree_dict, dict) else tree_dict
node_registry = compute_node_registry(semantic_root_node)
all_leaf_nodes = get_all_leaf_nodes_with_path(semantic_root_node)
doc_id_to_path = {get_node_id(leaf.id, docs_df): path for leaf, path in all_leaf_nodes}
#endregion

#region Setup LLM API and Eval Samples
if hp.LLM_API_BACKEND == 'genai': 
    llm_api = GenAIAPI(hp.LLM, logger=logger, timeout=hp.LLM_API_TIMEOUT, max_retries=hp.LLM_API_MAX_RETRIES)
elif hp.LLM_API_BACKEND == 'vllm': 
    llm_api = VllmAPI(hp.LLM, logger=logger, timeout=hp.LLM_API_TIMEOUT, max_retries=hp.LLM_API_MAX_RETRIES, base_url=','.join([f"http://localhost:{8000+i}/v1" for i in range(4)]))
else: raise ValueError(f'Unknown LM API backend: {hp.LLM_API_BACKEND}')

llm_api_kwargs = {
    'max_concurrent_calls': hp.LLM_MAX_CONCURRENT_CALLS,
    'response_mime_type': 'application/json',
    'response_schema': get_traversal_prompt_response_constraint(bool(hp.REASONING_IN_TRAVERSAL_PROMPT)),
    'staggering_delay': hp.LLM_API_STAGGERING_DELAY,
    'temperature': 0.7,
    'thinking_config': types.ThinkingConfig(thinking_budget=hp.REASONING_IN_TRAVERSAL_PROMPT),
}

if hp.LLM_API_BACKEND == 'vllm':
    llm_api_kwargs.pop('response_mime_type')
    llm_api_kwargs.pop('thinking_config')
    llm_api_kwargs.pop('response_schema') # response_schema not supported in vLLM API

rewrite_enabled = False
rewrite_map = {}
schema_cache_map: dict[str, List[str]] = {}
rewrite_template = None
qe_enabled = False
preflat_rewrite_map = {}

loaded_existing = False
# if hp.LOAD_EXISTING and os.path.exists(f'{RESULTS_DIR}/all_eval_sample_dicts.pkl'):
#     all_eval_samples, all_eval_metric_dfs = load_exp(RESULTS_DIR, hp, semantic_root_node, node_registry, logger)
#     logger.info(f'Loaded existing experiment with {len(all_eval_samples)} eval samples and {len(all_eval_metric_dfs)} eval metric dfs')
#     if len(all_eval_samples) > 0:
#         eval_metric_df = pd.DataFrame([sample.compute_eval_metrics(k=10) for sample in all_eval_samples])
#         logger.info('; '.join([f'{k}: {eval_metric_df[k].mean():.2f}' for k in eval_metric_df.columns]))
#     loaded_existing = True
# else: 
if True:
    all_eval_samples, all_eval_metric_dfs = [], []
    # Optional: load precomputed node embeddings for flat retrieval -> gated traversal
    node_embs = None
    retriever = None
    qe_map = {}
    rewrite_map = {}
    schema_cache_map = {}
    rewrite_template = None
    rewrite_enabled = bool(hp.REWRITE_PROMPT_NAME or hp.REWRITE_PROMPT_PATH or hp.REWRITE_CACHE_PATH)
    pending_gate_idx = None
    if hp.FLAT_THEN_TREE:
        if not hp.RETRIEVER_MODEL_PATH:
            raise ValueError('--retriever_model_path is required when --flat_then_tree is set')
        if not hp.NODE_EMB_PATH:
            raise ValueError('--node_emb_path is required when --flat_then_tree is set')
        from retrievers.diver import DiverEmbeddingModel
        node_embs = np.load(hp.NODE_EMB_PATH, allow_pickle=False)
        if node_embs.shape[0] != len(node_registry):
            raise ValueError(f'node_embs rows ({node_embs.shape[0]}) must match node_registry size ({len(node_registry)})')
        retriever = DiverEmbeddingModel(hp.RETRIEVER_MODEL_PATH, local_files_only=True)
        schema_depth1_indices = [
            idx for idx, node in enumerate(node_registry)
            if (len(node.path) == 1 and (not node.is_leaf))
        ]
        schema_depth1_embs = node_embs[schema_depth1_indices] if schema_depth1_indices else None

        qe_enabled = bool(hp.QE_PROMPT_NAME or hp.QE_CACHE_PATH)
        if hp.PRE_FLAT_REWRITE and qe_enabled:
            logger.warning('QE is ignored when --pre_flat_rewrite is set. Using original query for initial flat retrieval.')
            qe_enabled = False
        if rewrite_enabled:
            if hp.REWRITE_PROMPT_NAME:
                if hp.REWRITE_PROMPT_NAME not in REWRITE_PROMPT_TEMPLATES:
                    raise ValueError(
                        f'Unknown --rewrite_prompt_name "{hp.REWRITE_PROMPT_NAME}". '
                        f'Available: {sorted(REWRITE_PROMPT_TEMPLATES.keys())}'
                    )
                rewrite_template = REWRITE_PROMPT_TEMPLATES[hp.REWRITE_PROMPT_NAME]
            if hp.REWRITE_PROMPT_PATH:
                if not os.path.exists(hp.REWRITE_PROMPT_PATH):
                    raise ValueError(f'--rewrite_prompt_path not found: {hp.REWRITE_PROMPT_PATH}')
                with open(hp.REWRITE_PROMPT_PATH, 'r', encoding='utf-8') as f:
                    rewrite_template = f.read()
            if rewrite_template is None and not hp.REWRITE_CACHE_PATH:
                raise ValueError('--rewrite_prompt_name or --rewrite_prompt_path is required when rewrite is enabled')
            if hp.REWRITE_CACHE_PATH:
                rewrite_map, schema_cache_map = load_rewrite_cache(hp.REWRITE_CACHE_PATH, hp.REWRITE_FORCE_REFRESH)
        if qe_enabled:
            logger.info(f'Starting precomputation of query expansions for flat->tree retrieval {hp.QE_CACHE_PATH} {hp.QE_PROMPT_NAME}')
            if hp.QE_CACHE_PATH and os.path.exists(hp.QE_CACHE_PATH) and (not hp.QE_FORCE_REFRESH):
                with open(hp.QE_CACHE_PATH, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        if 'query' in rec and 'expanded_query' in rec:
                            qe_map[str(rec['query'])] = str(rec['expanded_query'])

            # Expand only the subset we will evaluate (and only for cache misses)
            eval_queries = [examples_df.iloc[i]['query'][:hp.MAX_QUERY_CHAR_LEN] for i in range(min(examples_df.shape[0], hp.NUM_EVAL_SAMPLES))]
            missing = [q for q in eval_queries if (hp.QE_FORCE_REFRESH or (q not in qe_map))]

            if len(missing) > 0:
                qe_template = None
                if hp.QE_PROMPT_NAME:
                    if hp.QE_PROMPT_NAME not in QE_PROMPT_TEMPLATES:
                        raise ValueError(f'Unknown --qe_prompt_name "{hp.QE_PROMPT_NAME}". Available: {sorted(QE_PROMPT_TEMPLATES.keys())}')
                    qe_template = QE_PROMPT_TEMPLATES[hp.QE_PROMPT_NAME]

                if qe_template is None:
                    raise ValueError(
                        f'QE cache is missing {len(missing)} queries. Provide --qe_prompt_name.'
                    )

                qe_prompts = [_format_qe_prompt(qe_template, q) for q in missing]
                qe_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(qe_loop)
                try:
                    qe_outputs = qe_loop.run_until_complete(llm_api.run_batch(qe_prompts, max_concurrent_calls=hp.LLM_MAX_CONCURRENT_CALLS))
                finally:
                    qe_loop.close()
                    asyncio.set_event_loop(None)

                for q, out in zip(missing, qe_outputs):
                    qe_map[q] = _clean_qe_text(out)

                if hp.QE_CACHE_PATH:
                    os.makedirs(os.path.dirname(hp.QE_CACHE_PATH) or '.', exist_ok=True)
                    with open(hp.QE_CACHE_PATH, 'w', encoding='utf-8') as f:
                        for q, eq in qe_map.items():
                            f.write(json.dumps({
                                'query': q,
                                'expanded_query': eq,
                                'prompt_name': hp.QE_PROMPT_NAME,
                                'llm': hp.LLM,
                            }, ensure_ascii=False) + '\n')
        schema_map: dict[str, List[str]] = {}
        if rewrite_enabled:
            eval_queries = [examples_df.iloc[i]['query'][:hp.MAX_QUERY_CHAR_LEN] for i in range(min(examples_df.shape[0], hp.NUM_EVAL_SAMPLES))]
            schema_generator = SchemaGenerator(
                retriever=retriever,
                node_registry=node_registry,
                node_embs=node_embs,
                llm_api=llm_api,
                cache_path=hp.REWRITE_CACHE_PATH,
                schema_cache_map=schema_cache_map,
                max_concurrent_calls=hp.LLM_MAX_CONCURRENT_CALLS,
                topk=10,
                depth=1,
                force_refresh=hp.REWRITE_FORCE_REFRESH,
                logger=logger,
            )
            schema_map = schema_generator.build_schema_map(eval_queries)
        preflat_rewrite_map = {}
        preflat_cache_hits: dict[str, bool] = {}
        if hp.PRE_FLAT_REWRITE:
            if not rewrite_enabled:
                raise ValueError('--pre_flat_rewrite requires rewrite prompt or cache')
            eval_queries = [examples_df.iloc[i]['query'][:hp.MAX_QUERY_CHAR_LEN] for i in range(min(examples_df.shape[0], hp.NUM_EVAL_SAMPLES))]
            preflat_prompts = []
            preflat_meta = []
            for q in eval_queries:
                hits = flat_retrieve_hits(
                    retriever=retriever,
                    query=q,
                    node_embs=node_embs,
                    node_registry=node_registry,
                    topk=hp.FLAT_TOPK,
                )
                schema_labels = schema_map.get(q, [])
                if hp.PRE_FLAT_REWRITE_SOURCE == "branch":
                    context_hits = [h for h in hits if not h.is_leaf]
                elif hp.PRE_FLAT_REWRITE_SOURCE == "leaf":
                    context_hits = [h for h in hits if h.is_leaf]
                else:
                    context_hits = hits
                context_descs = _hits_to_context_descs(
                    context_hits,
                    node_registry,
                    hp.REWRITE_CONTEXT_TOPK,
                    hp.MAX_DOC_DESC_CHAR_LEN,
                )
                cache_key = _rewrite_cache_key(
                    "preflat",
                    q,
                    context_descs,
                    iter_idx=None,
                    schema_labels=schema_labels,
                )
                if (not hp.REWRITE_FORCE_REFRESH) and (cache_key in rewrite_map):
                    preflat_rewrite_map[q] = rewrite_map[cache_key]
                    preflat_cache_hits[q] = True
                    continue
                if rewrite_template is None:
                    raise ValueError('Rewrite enabled but no prompt template is available.')
                preflat_prompts.append(_format_rewrite_prompt(
                    rewrite_template,
                    q,
                    "",
                    context_descs,
                    schema_labels,
                ))
                preflat_meta.append({
                    'cache_key': cache_key,
                    'base_query': q,
                    'context_descs': context_descs,
                    'schema_labels': schema_labels,
                })
            if preflat_prompts:
                preflat_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(preflat_loop)
                try:
                    preflat_outputs = preflat_loop.run_until_complete(
                        llm_api.run_batch(preflat_prompts, max_concurrent_calls=hp.LLM_MAX_CONCURRENT_CALLS)
                    )
                finally:
                    preflat_loop.close()
                    asyncio.set_event_loop(None)
                new_rewrite_records = []
                for meta, out in zip(preflat_meta, preflat_outputs):
                    rewrite, possible_docs = _parse_rewrite_output(out)
                    rewrite_map[meta['cache_key']] = rewrite
                    preflat_rewrite_map[meta['base_query']] = rewrite
                    preflat_cache_hits[meta['base_query']] = False
                    new_rewrite_records.append({
                        'key': meta['cache_key'],
                        'rewritten_query': rewrite,
                        'prompt_name': hp.REWRITE_PROMPT_NAME,
                        'llm': hp.LLM,
                        'context_descs': meta.get('context_descs', []),
                        'schema_labels': meta.get('schema_labels', []),
                        'possible_answer_docs': possible_docs,
                    })
                if hp.REWRITE_CACHE_PATH and new_rewrite_records:
                    append_jsonl(hp.REWRITE_CACHE_PATH, new_rewrite_records)

if not loaded_existing:
    node_by_path = {tuple(node.path): node for node in node_registry}
    for i in range(min(examples_df.shape[0], hp.NUM_EVAL_SAMPLES)):
        gold_paths = [doc_id_to_path[doc_id] for doc_id in examples_df.iloc[i]['gold_ids'] if doc_id in doc_id_to_path]
        if len(gold_paths) < len(examples_df.iloc[i]['gold_ids']):
            logger.warning(f"Some gold IDs for example {i} not found in document paths.")

        query = examples_df.iloc[i]['query'][:hp.MAX_QUERY_CHAR_LEN]
        allowed_prefixes = None
        flat_leaf_ranked = None
        schema_labels: List[str] = []
        if hp.FLAT_THEN_TREE:
            if hp.PRE_FLAT_REWRITE:
                expanded_query = preflat_rewrite_map.get(query, "")
            elif qe_enabled:
                expanded_query = qe_map[query]  # raw rewrite output
            else:
                expanded_query = ""
            flat_query = _compose_query(query, expanded_query)
            hits = flat_retrieve_hits(
                retriever=retriever,
                query=flat_query,
                node_embs=node_embs,
                node_registry=node_registry,
                topk=hp.FLAT_TOPK,
            )
            if rewrite_enabled:
                schema_labels = schema_map.get(query, [])
            else:
                schema_labels = []
            allowed_prefixes, flat_leaf_ranked, gate_scores = build_gates_and_leaf_candidates(
                hits=hits,
                gate_branches_topb=hp.GATE_BRANCHES_TOPB,
            )
            gate_descs = []
            if allowed_prefixes:
                for gate in allowed_prefixes:
                    node = node_by_path.get(tuple(gate))
                    if node and node.desc:
                        gate_descs.append(node.desc)
            context_descs = _hits_to_context_descs(
                hits,
                node_registry,
                hp.REWRITE_CONTEXT_TOPK,
                hp.MAX_DOC_DESC_CHAR_LEN,
            )
            query = flat_query

        sample = InferSample(
            semantic_root_node,
            node_registry,
            hp=hp,
            logger=logger,
            query=query,
            gold_paths=gold_paths,
            excluded_ids_set=set(examples_df.iloc[i]['excluded_ids']),
            allowed_prefixes=allowed_prefixes,
        )
        sample.original_query = examples_df.iloc[i]['query']
        sample.last_rewrite_raw = expanded_query if hp.FLAT_THEN_TREE else ""
        sample.schema_labels = schema_labels if hp.FLAT_THEN_TREE else []
        if hp.PRE_FLAT_REWRITE:
            if not hasattr(sample, 'rewrite_history'):
                sample.rewrite_history = []
            sample.rewrite_history.append({
                'iter': None,
                'phase': 'preflat',
                'cache_hit': preflat_cache_hits.get(sample.original_query, False),
                'rewrite': expanded_query,
                'schema_labels': schema_labels,
            })
        if flat_leaf_ranked is not None:
            sample.flat_leaf_ranked = flat_leaf_ranked
            sample.flat_gates = allowed_prefixes
            sample.flat_gate_scores = gate_scores
            sample.flat_query = flat_query
            sample.gold_paths_tuples = [tuple(p) for p in gold_paths]
            sample.flat_retrieved_paths = [h.path for h in hits]
            sample.gate_descs = gate_descs
            sample.flat_context_descs = context_descs
            sample.schema_labels = schema_labels
        if allowed_prefixes and hp.SEED_FROM_FLAT_GATES:
            sample.seed_beam_from_gate_paths(allowed_prefixes, gate_scores)
        all_eval_samples.append(sample)
    if rewrite_enabled and hp.REWRITE_AT_START:
        start_prompts = []
        start_meta = []
        for sample in all_eval_samples:
            context_descs = _get_rewrite_context(
                sample,
                None,
                hp.REWRITE_CONTEXT_SOURCE,
                node_by_path,
                node_registry,
                hp.REWRITE_CONTEXT_TOPK,
                hp.MAX_DOC_DESC_CHAR_LEN,
            )
            prev_rewrite = getattr(sample, "last_rewrite_raw", "")
            schema_labels = getattr(sample, "schema_labels", [])
            cache_key = _rewrite_cache_key(
                "start",
                f"{sample.original_query}||{prev_rewrite}",
                context_descs,
                iter_idx=None,
                schema_labels=schema_labels,
            )
            if (not hp.REWRITE_FORCE_REFRESH) and (cache_key in rewrite_map):
                rewrite = rewrite_map[cache_key]
                sample.last_rewrite_raw = rewrite
                sample.query = _compose_query(sample.original_query, rewrite)
                if not hasattr(sample, 'rewrite_history'):
                    sample.rewrite_history = []
                sample.rewrite_history.append({
                    'iter': None,
                    'phase': 'start',
                    'cache_hit': True,
                    'rewrite': rewrite,
                    'schema_labels': schema_labels,
                })
                continue
            if rewrite_template is None:
                raise ValueError('Rewrite enabled but no prompt template is available.')
            start_prompts.append(_format_rewrite_prompt(
                rewrite_template,
                sample.original_query,
                prev_rewrite,
                context_descs,
                schema_labels,
            ))
            start_meta.append({
                'sample': sample,
                'cache_key': cache_key,
                'base_query': sample.original_query,
                'prev_rewrite': prev_rewrite,
                'context_descs': context_descs,
                'schema_labels': schema_labels,
            })
        if start_prompts:
            start_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(start_loop)
            try:
                start_outputs = start_loop.run_until_complete(
                    llm_api.run_batch(start_prompts, max_concurrent_calls=hp.LLM_MAX_CONCURRENT_CALLS)
                )
            finally:
                start_loop.close()
                asyncio.set_event_loop(None)
            new_rewrite_records = []
            for meta, out in zip(start_meta, start_outputs):
                sample = meta['sample']
                rewrite, possible_docs = _parse_rewrite_output(out)
                rewrite_map[meta['cache_key']] = rewrite
                sample.last_rewrite_raw = rewrite
                sample.query = _compose_query(meta['base_query'], rewrite)
                if not hasattr(sample, 'rewrite_history'):
                    sample.rewrite_history = []
                sample.rewrite_history.append({
                    'iter': None,
                    'phase': 'start',
                    'cache_hit': False,
                    'rewrite': rewrite,
                    'schema_labels': meta.get('schema_labels', []),
                    'possible_answer_docs': possible_docs,
                })
                new_rewrite_records.append({
                    'key': meta['cache_key'],
                    'rewritten_query': rewrite,
                    'prompt_name': hp.REWRITE_PROMPT_NAME,
                    'llm': hp.LLM,
                    'context_descs': meta.get('context_descs', []),
                    'schema_labels': meta.get('schema_labels', []),
                    'possible_answer_docs': possible_docs,
                })
            if hp.REWRITE_CACHE_PATH and new_rewrite_records:
                append_jsonl(hp.REWRITE_CACHE_PATH, new_rewrite_records)
assert not any([sample.prediction_tree.excluded for sample in tqdm(all_eval_samples)])
  
logger.info('Hyperparams:\n'+'\n'.join([f'{k}:\t{v}' for k, v in vars(hp).items()]))
#endregion

#region Run Retrieval Loop
async def retrieval_loop_step(iter_idx: int):  # Make the function asynchronous
    inputs = [sample.get_step_prompts() for sample in all_eval_samples]
    indptr = np.cumsum([0, *[len(x) for x in inputs]])
    flat_inputs = [y for x in inputs for y in x]
    flat_prompts, flat_slates = list(zip(*flat_inputs))
    slates = [flat_slates[indptr[j]:indptr[j+1]] for j in range(len(inputs))]
    flat_slates_all = [slate for sample_slates in slates for slate in sample_slates]
    depth_counts = _count_slate_depths(flat_slates_all, node_registry, hp.FLAT_TOPK)
    logger.info(f"Iter {iter_idx}: slate depth counts (topk={hp.FLAT_TOPK}) {depth_counts}")

    flat_responses = await llm_api.run_batch(flat_prompts, **llm_api_kwargs)
    flat_response_jsons = [post_process(output, return_json=True) for output in tqdm(flat_responses)]
    response_jsons = [flat_response_jsons[indptr[j]:indptr[j+1]] for j in range(len(inputs))]

    for sample, sample_slates, sample_response_jsons in tqdm(zip(all_eval_samples, slates, response_jsons), total=len(all_eval_samples), desc='Updating samples'):
      sample.update(sample_slates, sample_response_jsons)
    if iter_idx == 0:
      for sample in all_eval_samples:
        if getattr(sample, 'seeded_gate_paths', None):
          sample.allowed_prefixes = None

    if rewrite_enabled and (hp.REWRITE_EVERY > 0) and ((iter_idx + 1) % hp.REWRITE_EVERY == 0):
      rewrite_prompts = []
      rewrite_meta = []
      for sample, sample_slates in zip(all_eval_samples, slates):
        context_descs = _get_rewrite_context(
            sample,
            sample_slates,
            hp.REWRITE_CONTEXT_SOURCE,
            node_by_path,
            node_registry,
            hp.REWRITE_CONTEXT_TOPK,
            hp.MAX_DOC_DESC_CHAR_LEN,
        )
        prev_rewrite = getattr(sample, "last_rewrite_raw", "")
        schema_labels = getattr(sample, "schema_labels", [])
        cache_key = _rewrite_cache_key(
            "iter",
            f"{sample.original_query}||{prev_rewrite}",
            context_descs,
            iter_idx=iter_idx,
            schema_labels=schema_labels,
        )
        if (not hp.REWRITE_FORCE_REFRESH) and (cache_key in rewrite_map):
          rewrite = rewrite_map[cache_key]
          sample.last_rewrite_raw = rewrite
          sample.query = _compose_query(sample.original_query, rewrite)
          if not hasattr(sample, 'rewrite_history'):
            sample.rewrite_history = []
          sample.rewrite_history.append({
              'iter': iter_idx,
              'phase': 'iter',
              'cache_hit': True,
              'rewrite': rewrite,
              'schema_labels': schema_labels,
          })
          continue
        if rewrite_template is None:
          raise ValueError('Rewrite enabled but no prompt template is available.')
        rewrite_prompts.append(_format_rewrite_prompt(
            rewrite_template,
            sample.original_query,
            prev_rewrite,
            context_descs,
            schema_labels,
        ))
        rewrite_meta.append({
            'sample': sample,
            'cache_key': cache_key,
            'base_query': sample.original_query,
            'prev_rewrite': prev_rewrite,
            'context_descs': context_descs,
            'schema_labels': schema_labels,
        })
      if rewrite_prompts:
        rewrite_outputs = await llm_api.run_batch(
            rewrite_prompts,
            max_concurrent_calls=hp.LLM_MAX_CONCURRENT_CALLS,
            staggering_delay=hp.LLM_API_STAGGERING_DELAY,
        )
        new_rewrite_records = []
        for meta, out in zip(rewrite_meta, rewrite_outputs):
          sample = meta['sample']
          rewrite, possible_docs = _parse_rewrite_output(out)
          rewrite_map[meta['cache_key']] = rewrite
          sample.last_rewrite_raw = rewrite
          sample.query = _compose_query(meta['base_query'], rewrite)
          if not hasattr(sample, 'rewrite_history'):
            sample.rewrite_history = []
          sample.rewrite_history.append({
              'iter': iter_idx,
              'phase': 'iter',
              'cache_hit': False,
              'rewrite': rewrite,
              'schema_labels': meta.get('schema_labels', []),
              'possible_answer_docs': possible_docs,
          })
          new_rewrite_records.append({
              'key': meta['cache_key'],
              'rewritten_query': rewrite,
              'prompt_name': hp.REWRITE_PROMPT_NAME,
              'llm': hp.LLM,
              'context_descs': meta.get('context_descs', []),
              'schema_labels': meta.get('schema_labels', []),
              'possible_answer_docs': possible_docs,
          })
        if hp.REWRITE_CACHE_PATH and new_rewrite_records:
          append_jsonl(hp.REWRITE_CACHE_PATH, new_rewrite_records)
    return slates

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
try:
    for i in tqdm(range(len(all_eval_metric_dfs), hp.NUM_ITERS)):
        logger.info(f'-------------------- Iter {i} --------------------')
        slates = loop.run_until_complete(retrieval_loop_step(i))
        if hp.FLAT_THEN_TREE:
            fused_metric_rows = []
            leaf_counts_200 = []
            leaf_counts_10 = []
            for sample, sample_slates in zip(all_eval_samples, slates):
                traversal_ranked = [(tuple(x.path), float(s)) for x, s in sample.get_top_predictions(200, rel_fn=sample.get_rel_fn(leaf=True))]
                leaf_counts_200.append(len(traversal_ranked))
                leaf_counts_10.append(min(len(traversal_ranked), 10))
                flat_leaf_ranked = getattr(sample, 'flat_leaf_ranked', [])
                fused = rrf_fuse_ranked_paths([traversal_ranked, flat_leaf_ranked], k=100) # RRF에서 쓰는 상수 k는 rank 1의 영향력을 얼마나 강하게 줄지를 조절해. k가 클수록 top‑rank 가중치가 완만해지고, 작을수록 top‑rank가 더 강하게 우세해져. 60은 IR에서 흔히 쓰는 기본값이라 그대로 쓴 거야.
                fused_paths = [list(p) for p, _ in fused]
                gold_paths = sample.gold_paths
                row = {
                    f'nDCG@10': compute_ndcg(fused_paths[:10], gold_paths, k=10)*100,
                    f'Recall@10': compute_recall(fused_paths[:10], gold_paths, k=10)*100,
                    f'Recall@{100}': compute_recall(fused_paths[:100], gold_paths, k=100)*100,
                    'Coverage': len(fused_paths),
                    'AncestorHit@flatK': 100.0 if ancestor_hit(getattr(sample, 'flat_retrieved_paths', []), getattr(sample, 'gold_paths_tuples', [])) else 0.0,
                    'GateHit': 100.0 if gate_hit(getattr(sample, 'flat_gates', []) or [], getattr(sample, 'gold_paths_tuples', [])) else 0.0,
                }
                for source in ("flat", "slate", "fused", "mixed"):
                    ranked = _ranked_paths_from_context_source(sample, sample_slates, source, node_registry)
                    ranked_paths = [list(p) for p, _ in ranked]
                    if ranked_paths:
                        row[f'nDCG@10_ctx_{source}'] = compute_ndcg(ranked_paths[:10], gold_paths, k=10)*100
                    else:
                        row[f'nDCG@10_ctx_{source}'] = 0.0
                traversal_paths = [list(p) for p, _ in traversal_ranked]
                if traversal_paths:
                    row['nDCG@10_ctx_treeonly'] = compute_ndcg(traversal_paths[:10], gold_paths, k=10)*100
                else:
                    row['nDCG@10_ctx_treeonly'] = 0.0
                fused_metric_rows.append(row)
            eval_metric_df = pd.DataFrame(fused_metric_rows)
            if leaf_counts_200:
                zero_200 = sum(1 for c in leaf_counts_200 if c == 0)
                zero_10 = sum(1 for c in leaf_counts_10 if c == 0)
                logger.info(
                    "Traversal leaf@200 mean=%.1f, median=%.1f, zero=%d | leaf@10 mean=%.1f, median=%.1f, zero=%d",
                    float(np.mean(leaf_counts_200)),
                    float(np.median(leaf_counts_200)),
                    zero_200,
                    float(np.mean(leaf_counts_10)),
                    float(np.median(leaf_counts_10)),
                    zero_10,
                )
            ctx_cols = [f'nDCG@10_ctx_{s}' for s in ("flat", "slate", "fused", "mixed", "treeonly") if f'nDCG@10_ctx_{s}' in eval_metric_df.columns]
            if ctx_cols:
                ctx_summary = "; ".join([f"{c}={eval_metric_df[c].mean():.2f}" for c in ctx_cols])
                logger.info(f"Rewrite context nDCG@10: {ctx_summary}")
        else:
          eval_metric_df = pd.DataFrame([sample.compute_eval_metrics(k=10) for sample in all_eval_samples])
        all_eval_metric_dfs.append(eval_metric_df)
        
        # Log metrics
        # wandb_log_iteration_metrics(eval_metric_df, i)
        logger.info('; '.join([f'{k}: {eval_metric_df[k].mean():.2f}' for k in eval_metric_df.columns]))
        save_exp(RESULTS_DIR, hp, llm_api, all_eval_samples, all_eval_metric_dfs, allow_overwrite=True)  
        logger.info('-'*50)
finally:
    loop.close()
#endregion

#region Reranking (Optional)
# async def rerank_predictions():
#     logger.info('Starting reranking process...')
    
#     def get_sample_rerank_prompt(sample):
#         return get_reranking_prompt(sample.query, [x.desc for x, s in sample.get_top_predictions(100)], hp=hp, logger=logger, topk=10)

#     def process_sample_rerank_response(sample, response):
#         try:
#           ranking = post_process(response, return_json=True)['ranking']
#           top_preds = [x[0] for x in sample.get_top_predictions(100)]
#           for rank, idx in enumerate(ranking):
#               top_preds[idx].inverse_rank = 1/(rank+1)
#         except Exception as e:
#           logger.error(f'Error processing rerank response for query "{sample.query}": {e}')

#     all_rerank_prompts, all_rerank_constraints = list(zip(*[get_sample_rerank_prompt(sample) for sample in all_eval_samples]))
#     all_rerank_responses = await llm_api.run_batch(
#         all_rerank_prompts, 
#         max_concurrent_calls=hp.LLM_MAX_CONCURRENT_CALLS, 
#         response_mime_type='application/json', 
#         response_schema=all_rerank_constraints[0]
#     )
    
#     for sample, response in zip(all_eval_samples, all_rerank_responses):
#         process_sample_rerank_response(sample, response)
    
#     default_rel_fn = all_eval_samples[0].get_rel_fn(leaf=True)
#     rerank_rel_fn = lambda x: (x.inverse_rank if hasattr(x, 'inverse_rank') else 0, default_rel_fn(x))
#     rerank_eval_metric_df = pd.DataFrame([sample.compute_eval_metrics(k=10, rel_fn=rerank_rel_fn) for sample in all_eval_samples])
    
#     # Log reranking metrics to wandb
#     # wandb_log_reranking_metrics(rerank_eval_metric_df)
    
#     logger.info('After reranking: '+'; '.join([f'{k}: {rerank_eval_metric_df[k].mean():.2f}' for k in rerank_eval_metric_df.columns]))
    
#     return rerank_eval_metric_df

# # Run reranking if enabled
# if hasattr(hp, 'RERANK') and hp.RERANK:
#     rerank_eval_metric_df = asyncio.run(rerank_predictions())
#     save_exp(RESULTS_DIR, hp, llm_api, all_eval_samples, all_eval_metric_dfs + [rerank_eval_metric_df], allow_overwrite=True)
# else:
#     logger.info('Reranking disabled, skipping...')

# Log final summary metrics and finish wandb run
# if all_eval_metric_dfs and all_eval_samples:
#     wandb_log_final_summary(all_eval_samples)

# finish_wandb_logging(logger)
#endregion
