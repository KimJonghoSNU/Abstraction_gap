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

QE_PROMPT_TEMPLATES = {
    # Bridge-focused: explicitly try to jump abstraction levels without assuming a fixed taxonomy.
    "bridge_v1": (
        "You are helping retrieval for a reasoning-intensive question where the answer may live at a different abstraction level.\n"
        "Produce a expanded retrieval query that includes:\n"
        "- the original query intent,\n"
        "- possible underlying mechanisms / theories / examples / canonical terms,\n"
        "- alternative formulations and related concepts that could bridge to the correct topic cluster.\n"
        "Output ONLY the expanded query text.\n\n"
        "User query:\n{query}\n"
    ),
    # Adapted from previous/shortcut_reranker/utils/agent_prompts.py (executor-style).
    "agent_executor_v1": (
        "Your goal is to generate the next set of 'Possible Answer Documents' (Search Queries).\n"
        "No prior retrieved docs are given, so propose diverse hypotheses that could bridge abstraction gaps.\n\n"
        "**Thinking Process (Execute inside 'Plan'):**\n"
        "1. Identify User Intent & Answer Type.\n"
        "2. Propose multiple plausible theories/mechanisms.\n"
        "3. Propose concrete entities/phenomena.\n"
        "4. Propose examples/analogies.\n"
        "5. Ensure each item is directly useful for retrieval.\n\n"
        "**Output Format:**\n"
        "Output a single JSON object. The values will be concatenated to form the next query.\n"
        "```json\n"
        "{\n"
        "  \"Plan\": \"...\",\n"
        "  \"Possible_Answer_Docs\": {\n"
        "    \"Theory\": \"...text...\",\n"
        "    \"Entity\": \"...text...\",\n"
        "    \"Example\": \"...text...\",\n"
        "    \"Other\": \"...text...\"\n"
        "  }\n"
        "}\n"
        "```\n\n"
        "Original Query: {query}\n"
    ),
    "pre_flat_rewrite_v1": (
        "You are rewriting a search query for reasoning-intensive retrieval.\n\n"
        "Key idea:\n"
        "- Assume you already have the correct answer to the user query.\n"
        "- The documents we want are the ones that would be used as core evidence or justification for that answer.\n"
        "- These evidence documents are often more abstract than the surface query.\n\n"
        "Task:\n"
        "- First, write a 1-2 sentence Plan that states the user's core intent and what kind of evidence would justify the answer.\n"
        "- Produce 2-5 distinct Possible_Answer_Docs that could serve as evidence for the assumed correct answer.\n"
        "- Avoid near-duplicates; keep each item short and retrieval-friendly.\n"
        "- Evidence forms may include (not exhaustive): theory/mechanism, entity/fact, analogy/example, method/metric, canonical reference.\n\n"
        "Output JSON only:\n"
        "{\n"
        "  \"Plan\": \"short reasoning\",\n"
        "  \"Possible_Answer_Docs\": [\n"
        "    \"...\",\n"
        "    \"...\",\n"
        "    \"...\"\n"
        "  ]\n"
        "}\n\n"
        "User Query:\n{query}\n"
    ),
    "stepback_json_pre": (
        "Your goal is to generate answer documents for the query without any retrieved passages.\n"
        "Each generated document must be strictly relevant: it should either contain the exact answer or "
        "provide abstractive-level theory, evidence, or background that is necessary for that answer.\n\n"
        "Output Format:\n"
        "You must output a single JSON object with the following keys:\n"
        "- \"Plan\": A detailed string where you analyze availability of information and plan your selection. Follow the planning steps below.\n"
        "- \"Possible_Answer_Docs\": A JSON object with keys as document types (\"Theory\", \"Entity\", \"Example\", \"Other\") and values as strings representing distinct answer documents.\n\n"
        "Planning Steps (for the \"Plan\" field):\n"
        "1. Precisely identify what information the user is seeking "
        "(e.g., why / how / what / what is this called / which part) and what type of "
        "answer would satisfy them (e.g., concept name, theory, explanation, concrete "
        "entity, worked example).\n"
        "2. If the query is explicitly or implicitly asking 'what is this called?', "
        "generate diverse hypotheses for the name of the phenomenon, object, or concept.\n"
        "3. Abstraction: infer which academic terms, scientific theories, mathematical "
        "models, canonical methods, or standard resources lie behind this question and "
        "would be cited in a strictly correct answer.\n"
        "4. Consider alternative ways the answer might be supported: a direct definition, "
        "background theory, canonical examples, or reference websites.\n"
        "5. For causal or explanatory questions (why/how), identify multiple theoretical "
        "frameworks that offer different explanations, including mainstream, alternative, "
        "and controversial perspectives if mentioned in the query.\n"
        "6. Anticipate common wrong directions (topic, answer type, or framing) suggested by the query "
        "and treat them as negative constraints.\n"
        "Ensure that every generated document contributes directly and strictly to such an answer "
        "(no loosely related or merely interesting content).\n\n"
        "Final Answer Requirements (for the \"Answer documents\" field):\n"
        "Include 3-5 distinct answer-document entries:\n"
        "- 1: Concept/Theory-focused (academic terms, principles, models).\n"
        "- 2: Entity/Fact-focused (specific names, objects, or concrete facts).\n"
        "- 3: Broad Context/Evidence-focused (surveys, experiments, or background "
        "that support the answer).\n"
        "- Document 4–5 (optional): Any additional strictly relevant documents that provide "
        "alternative but correct perspectives or examples.\n\n"
        "Query: {query}\n\n"
        "Provide the JSON output.\n"
        "Example Output Format:\n"
        "```json\n"
        "{\n"
        "  \"Plan\": \"Detailed plan applying Step 6 logic...\",\n"
        "  \"Possible_Answer_Docs\": {\n"
        "    \"Theory\": \"...text...\",\n"
        "    \"Entity\": \"...text...\",\n"
        "    \"Example\": \"...text...\",\n"
        "    \"Other\": \"...text...\"\n"
        "  }\n"
        "}\n"
        "```\n"
    ),
}

REWRITE_PROMPT_TEMPLATES = {
    "gate_rewrite_v1": (
        "You are rewriting a search query for reasoning-intensive retrieval.\n\n"
        "Key idea:\n"
        "- Assume you already have the correct answer to the user query.\n"
        "- The documents we want are the ones that would be used as core evidence or justification for that answer.\n"
        "- These evidence documents are often more abstract than the surface query.\n\n"
        "Task:\n"
        "- First, write a 1-2 sentence Plan that states the user's core intent and what kind of evidence would justify the answer.\n"
        "- Use the context summaries only as hints (not ground truth).\n"
        "- Produce 2-5 distinct Possible_Answer_Docs that could serve as evidence for the assumed correct answer.\n"
        "- Avoid near-duplicates; keep each item short and retrieval-friendly.\n"
        "- Evidence forms may include (not exhaustive): theory/mechanism, entity/fact, analogy/example, method/metric, canonical reference.\n\n"
        "Output JSON only:\n"
        "{\n"
        "  \"Plan\": \"short reasoning\",\n"
        "  \"Possible_Answer_Docs\": [\n"
        "    \"...\",\n"
        "    \"...\",\n"
        "    \"...\"\n"
        "  ]\n"
        "}\n\n"
        "Original Query:\n{original_query}\n\n"
        "Previous Rewritten Query:\n{previous_rewrite}\n\n"
        "Context Summaries:\n{gate_descs}\n"
    ),
    "stepback_json": (
        "Given a query, the provided passages (most of which may be incorrect or irrelevant), and the previous round's answer (if any), "
        "your goal is to analyze the candidate passages retrieved for the query, "
        "identify which of them are strictly relevant to the user's true intent, "
        "and then refine the set of answer documents accordingly. "
        "Each refined document must be strictly relevant: it should either contain the "
        "exact answer or provide abstractive-level theory, evidence, or background that is "
        "necessary for that answer.\n\n"
        "Output Format:\n"
        "You must output a single JSON object with the following keys:\n"
        "- \"Plan\": A detailed string where you analyze availability of information and plan your selection. Follow the planning steps below.\n"
        "- \"Possible_Answer_Docs\": A JSON object with keys as document types (\"Theory\", \"Entity\", \"Example\", \"Other\") and values as strings representing distinct answer documents.\n\n"
        "Planning Steps (for the \"Plan\" field):\n"
        "1. Precisely identify what information the user is seeking "
        "(e.g., why / how / what / what is this called / which part) and what type of "
        "answer would satisfy them (e.g., concept name, theory, explanation, concrete "
        "entity, worked example).\n"
        "2. If the query is explicitly or implicitly asking 'what is this called?', "
        "generate diverse hypotheses for the name of the phenomenon, object, or concept.\n"
        "3. Abstraction: infer which academic terms, scientific theories, mathematical "
        "models, canonical methods, or standard resources lie behind this question and "
        "would be cited in a strictly correct answer.\n"
        "4. Consider alternative ways the answer might be supported: a direct definition, "
        "background theory, canonical examples, or reference websites.\n"
        "5. For causal or explanatory questions (why/how), identify multiple theoretical "
        "frameworks that offer different explanations, including mainstream, alternative, "
        "and controversial perspectives if mentioned in the query.\n"
        "6. For each Candidate Passage, judge which part of the user's query does the passage address. Then, "
        "identify the common wrong direction (topic, answer type, or framing) suggested by "
        "the Candidate Passages, and treat this as a negative constraint. Plan how to "
        "refine the answer documents to avoid that direction and instead explore a different, more "
        "plausible interpretation aligned with the user's wording.\n"
        "Ensure that every generated document contributes directly and strictly to such an "
        "answer (no loosely related or merely interesting content).\n\n"
        "Final Answer Requirements (for the \"Answer documents\" field):\n"
        "Include 3-5 distinct answer-document entries:\n"
        "- 1: Concept/Theory-focused (academic terms, principles, models).\n"
        "- 2: Entity/Fact-focused (specific names, objects, or concrete facts).\n"
        "- 3: Broad Context/Evidence-focused (surveys, experiments, or background "
        "that support the answer).\n"
        "- Document 4–5 (optional): Any additional strictly relevant documents that provide "
        "alternative but correct perspectives or examples.\n\n"
        "Original Query:\n{original_query}\n\n"
        "Candidate Passages:\n{gate_descs}\n\n"
        "Previous rewritten query (if any):\n{previous_rewrite}\n\n"
        "Provide the JSON output.\n"
        "Example Output Format:\n"
        "```json\n"
        "{\n"
        "  \"Plan\": \"Detailed plan applying Step 6 logic...\",\n"
        "  \"Possible_Answer_Docs\": {\n"
        "    \"Theory\": \"...text...\",\n"
        "    \"Entity\": \"...text...\",\n"
        "    \"Example\": \"...text...\",\n"
        "    \"Other\": \"...text...\"\n"
        "  }\n"
        "}\n"
        "```\n"
    ),
}

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

def _format_rewrite_prompt(
    template: str,
    original_query: str,
    previous_rewrite: str,
    context_descs: List[str],
) -> str:
    context_blob = "\n".join([x for x in context_descs if x])
    try:
        return template.format(
            gate_descs=context_blob,
            original_query=(original_query or ""),
            previous_rewrite=(previous_rewrite or ""),
        )
    except KeyError:
        # Fallback for templates containing JSON braces without escaping.
        return (
            template
            .replace("{gate_descs}", context_blob)
            .replace("{original_query}", original_query or "")
            .replace("{previous_rewrite}", previous_rewrite or "")
        )

def _rewrite_cache_key(prefix: str, query: str, context_descs: List[str], iter_idx: int | None = None) -> str:
    context_blob = "\n".join([x for x in context_descs if x]).strip()
    context_sig = hashlib.md5(context_blob.encode('utf-8')).hexdigest() if context_blob else "none"
    iter_tag = f"||iter={iter_idx}" if iter_idx is not None else ""
    return f"{prefix}||{query}||{context_sig}{iter_tag}"

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

#region Setup
hp = HyperParams.from_args()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = f'{BASE_DIR}/results/{hp.DATASET}/{hp.SUBSET}/'
os.makedirs(RESULTS_DIR, exist_ok=True)
log_path = f"{RESULTS_DIR}/{hp}.log"
os.makedirs(os.path.dirname(log_path), exist_ok=True)
logger = setup_logger('lattice_runner', log_path, logging.INFO)

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
    # 'temperature': 0.8,
    'thinking_config': types.ThinkingConfig(thinking_budget=hp.REASONING_IN_TRAVERSAL_PROMPT),
}

if hp.LLM_API_BACKEND == 'vllm':
    llm_api_kwargs.pop('response_mime_type')
    llm_api_kwargs.pop('thinking_config')
    llm_api_kwargs.pop('response_schema') # response_schema not supported in vLLM API

rewrite_enabled = False
rewrite_map = {}
rewrite_template = None

if hp.LOAD_EXISTING and os.path.exists(f'{RESULTS_DIR}/all_eval_sample_dicts-{hp}.pkl'):
    all_eval_samples, all_eval_metric_dfs = load_exp(RESULTS_DIR, hp, semantic_root_node, node_registry, logger)
    logger.info(f'Loaded existing experiment with {len(all_eval_samples)} eval samples and {len(all_eval_metric_dfs)} eval metric dfs')
    if len(all_eval_samples) > 0:
        eval_metric_df = pd.DataFrame([sample.compute_eval_metrics(k=10) for sample in all_eval_samples])
        logger.info('; '.join([f'{k}: {eval_metric_df[k].mean():.2f}' for k in eval_metric_df.columns]))
else: 
    all_eval_samples, all_eval_metric_dfs = [], []
    # Optional: load precomputed node embeddings for flat retrieval -> gated traversal
    node_embs = None
    retriever = None
    qe_map = {}
    rewrite_map = {}
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

        qe_enabled = bool(hp.QE_PROMPT_NAME or hp.QE_CACHE_PATH)
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
            if hp.REWRITE_CACHE_PATH and os.path.exists(hp.REWRITE_CACHE_PATH) and (not hp.REWRITE_FORCE_REFRESH):
                with open(hp.REWRITE_CACHE_PATH, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        if 'key' in rec and 'rewritten_query' in rec:
                            rewrite_map[str(rec['key'])] = str(rec['rewritten_query'])
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

node_by_path = {tuple(node.path): node for node in node_registry}
for i in range(min(examples_df.shape[0], hp.NUM_EVAL_SAMPLES)):
    gold_paths = [doc_id_to_path[doc_id] for doc_id in examples_df.iloc[i]['gold_ids'] if doc_id in doc_id_to_path]
    if len(gold_paths) < len(examples_df.iloc[i]['gold_ids']):
        logger.warning(f"Some gold IDs for example {i} not found in document paths.")

    query = examples_df.iloc[i]['query'][:hp.MAX_QUERY_CHAR_LEN]
    allowed_prefixes = None
    flat_leaf_ranked = None
    if hp.FLAT_THEN_TREE:
        if qe_enabled:
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
        allowed_prefixes, flat_leaf_ranked = build_gates_and_leaf_candidates(
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
    if flat_leaf_ranked is not None:
        sample.flat_leaf_ranked = flat_leaf_ranked
        sample.flat_gates = allowed_prefixes
        sample.flat_query = flat_query
        sample.gold_paths_tuples = [tuple(p) for p in gold_paths]
        sample.flat_retrieved_paths = [h.path for h in hits]
        sample.gate_descs = gate_descs
        sample.flat_context_descs = context_descs
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
        cache_key = _rewrite_cache_key("start", f"{sample.original_query}||{prev_rewrite}", context_descs, iter_idx=None)
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
            })
            continue
        if rewrite_template is None:
            raise ValueError('Rewrite enabled but no prompt template is available.')
        start_prompts.append(_format_rewrite_prompt(
            rewrite_template,
            sample.original_query,
            prev_rewrite,
            context_descs,
        ))
        start_meta.append({
            'sample': sample,
            'cache_key': cache_key,
            'base_query': sample.original_query,
            'prev_rewrite': prev_rewrite,
            'context_descs': context_descs,
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
            rewrite = _clean_qe_text(out)
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
            })
            new_rewrite_records.append({
                'key': meta['cache_key'],
                'rewritten_query': rewrite,
                'prompt_name': hp.REWRITE_PROMPT_NAME,
                'llm': hp.LLM,
                'context_descs': meta.get('context_descs', []),
            })
        if hp.REWRITE_CACHE_PATH and new_rewrite_records:
            os.makedirs(os.path.dirname(hp.REWRITE_CACHE_PATH) or '.', exist_ok=True)
            with open(hp.REWRITE_CACHE_PATH, 'a', encoding='utf-8') as f:
                for rec in new_rewrite_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
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

    flat_responses = await llm_api.run_batch(flat_prompts, **llm_api_kwargs)
    flat_response_jsons = [post_process(output, return_json=True) for output in tqdm(flat_responses)]
    response_jsons = [flat_response_jsons[indptr[j]:indptr[j+1]] for j in range(len(inputs))]

    for sample, sample_slates, sample_response_jsons in tqdm(zip(all_eval_samples, slates, response_jsons), total=len(all_eval_samples), desc='Updating samples'):
      sample.update(sample_slates, sample_response_jsons)

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
        cache_key = _rewrite_cache_key("iter", f"{sample.original_query}||{prev_rewrite}", context_descs, iter_idx=iter_idx)
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
          })
          continue
        if rewrite_template is None:
          raise ValueError('Rewrite enabled but no prompt template is available.')
        rewrite_prompts.append(_format_rewrite_prompt(
            rewrite_template,
            sample.original_query,
            prev_rewrite,
            context_descs,
        ))
        rewrite_meta.append({
            'sample': sample,
            'cache_key': cache_key,
            'base_query': sample.original_query,
            'prev_rewrite': prev_rewrite,
            'context_descs': context_descs,
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
          rewrite = _clean_qe_text(out)
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
          })
          new_rewrite_records.append({
              'key': meta['cache_key'],
              'rewritten_query': rewrite,
              'prompt_name': hp.REWRITE_PROMPT_NAME,
              'llm': hp.LLM,
              'context_descs': meta.get('context_descs', []),
          })
        if hp.REWRITE_CACHE_PATH and new_rewrite_records:
          os.makedirs(os.path.dirname(hp.REWRITE_CACHE_PATH) or '.', exist_ok=True)
          with open(hp.REWRITE_CACHE_PATH, 'a', encoding='utf-8') as f:
            for rec in new_rewrite_records:
              f.write(json.dumps(rec, ensure_ascii=False) + '\n')

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
try:
    for i in tqdm(range(len(all_eval_metric_dfs), hp.NUM_ITERS)):
        logger.info(f'-------------------- Iter {i} --------------------')
        loop.run_until_complete(retrieval_loop_step(i))
        if hp.FLAT_THEN_TREE:
            fused_metric_rows = []
            leaf_counts_200 = []
            leaf_counts_10 = []
            for sample in all_eval_samples:
                traversal_ranked = [(tuple(x.path), float(s)) for x, s in sample.get_top_predictions(200, rel_fn=sample.get_rel_fn(leaf=True))]
                leaf_counts_200.append(len(traversal_ranked))
                leaf_counts_10.append(min(len(traversal_ranked), 10))
                flat_leaf_ranked = getattr(sample, 'flat_leaf_ranked', [])
                fused = rrf_fuse_ranked_paths([traversal_ranked, flat_leaf_ranked], k=100) # RRF에서 쓰는 상수 k는 rank 1의 영향력을 얼마나 강하게 줄지를 조절해. k가 클수록 top‑rank 가중치가 완만해지고, 작을수록 top‑rank가 더 강하게 우세해져. 60은 IR에서 흔히 쓰는 기본값이라 그대로 쓴 거야.
                fused_paths = [list(p) for p, _ in fused]
                gold_paths = sample.gold_paths
                fused_metric_rows.append({
                f'nDCG@10': compute_ndcg(fused_paths[:10], gold_paths, k=10)*100,
                f'Recall@10': compute_recall(fused_paths[:10], gold_paths, k=10)*100,
                f'Recall@{100}': compute_recall(fused_paths[:100], gold_paths, k=100)*100,
                'Coverage': len(fused_paths),
                'AncestorHit@flatK': 100.0 if ancestor_hit(getattr(sample, 'flat_retrieved_paths', []), getattr(sample, 'gold_paths_tuples', [])) else 0.0,
                'GateHit': 100.0 if gate_hit(getattr(sample, 'flat_gates', []) or [], getattr(sample, 'gold_paths_tuples', [])) else 0.0,
                })
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
