#region Imports
import numpy as np
import pandas as pd
import pickle as pkl
import asyncio
import json
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
}

def _format_qe_prompt(template: str, query: str) -> str:
    if "{query}" in template:
        return template.format(query=query)
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
    return text.strip()

#region Setup
hp = HyperParams.from_args()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = f'{BASE_DIR}/results/{hp.DATASET}/{hp.SUBSET}/'
os.makedirs(RESULTS_DIR, exist_ok=True)
logger = setup_logger('lattice_runner', f"{RESULTS_DIR}/{hp}.log", logging.INFO)

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

for i in range(min(examples_df.shape[0], hp.NUM_EVAL_SAMPLES)):
    gold_paths = [doc_id_to_path[doc_id] for doc_id in examples_df.iloc[i]['gold_ids'] if doc_id in doc_id_to_path]
    if len(gold_paths) < len(examples_df.iloc[i]['gold_ids']):
        logger.warning(f"Some gold IDs for example {i} not found in document paths.")

    query = examples_df.iloc[i]['query'][:hp.MAX_QUERY_CHAR_LEN]
    allowed_prefixes = None
    flat_leaf_ranked = None
    if hp.FLAT_THEN_TREE:
        if qe_enabled:
            expanded_query = query + " " + qe_map[query]  # must exist due to precomputation above
        else:
            expanded_query = query
        # expanded_query = qe_map.get(query, query)
        flat_query = expanded_query
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
        # stash flat leaf candidates on the sample later; keep traversal query configurable
        query = expanded_query

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
    if flat_leaf_ranked is not None:
        sample.flat_leaf_ranked = flat_leaf_ranked
        sample.flat_gates = allowed_prefixes
        sample.flat_query = flat_query
        sample.original_query = examples_df.iloc[i]['query']
        sample.gold_paths_tuples = [tuple(p) for p in gold_paths]
        sample.flat_retrieved_paths = [h.path for h in hits]
    all_eval_samples.append(sample)
assert not any([sample.prediction_tree.excluded for sample in tqdm(all_eval_samples)])
  
logger.info('Hyperparams:\n'+'\n'.join([f'{k}:\t{v}' for k, v in vars(hp).items()]))
#endregion

#region Run Retrieval Loop
async def retrieval_loop_step():  # Make the function asynchronous
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

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
try:
    for i in tqdm(range(len(all_eval_metric_dfs), hp.NUM_ITERS)):
        logger.info(f'-------------------- Iter {i} --------------------')
        loop.run_until_complete(retrieval_loop_step())
        if hp.FLAT_THEN_TREE:
            fused_metric_rows = []
            for sample in all_eval_samples:
                traversal_ranked = [(tuple(x.path), float(s)) for x, s in sample.get_top_predictions(200, rel_fn=sample.get_rel_fn(leaf=True))]
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
