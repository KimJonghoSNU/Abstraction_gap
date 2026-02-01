#region Imports
import numpy as np
import pandas as pd
import pickle as pkl
import asyncio
from tqdm.autonotebook import tqdm
import os
import logging
from typing import List
from datasets import load_dataset
from google.genai import types
from hyperparams import HyperParams
from tree_objects import SemanticNode, InferSample
from llm_apis import GenAIAPI, VllmAPI
from prompts import get_traversal_prompt_response_constraint, get_reranking_prompt
from retrievers.diver import DiverEmbeddingModel
from utils import (
    setup_logger, 
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
    normalize_embeddings,
)
np.random.seed(42)
#endregion

#region Setup
hp = HyperParams.from_args()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
exp_dir_name = str(hp)
RESULTS_DIR = f'{BASE_DIR}/results/{hp.DATASET}/{hp.SUBSET}/{exp_dir_name}/'
os.makedirs(RESULTS_DIR, exist_ok=True)
logger = setup_logger('lattice_runner', f"{RESULTS_DIR}/run.log", logging.INFO)

# Initialize wandb logging
run_name = init_wandb_logging(hp, RESULTS_DIR)
logger.info(f"Initialized wandb run: {run_name}")
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

#region Optional retriever for traversal scoring
node_embs = None
retriever = None
if hp.USE_RETRIEVER_TRAVERSAL:
    if not hp.RETRIEVER_MODEL_PATH or not hp.NODE_EMB_PATH:
        raise ValueError('--use_retriever_traversal requires --retriever_model_path and --node_emb_path')
    node_embs = np.load(hp.NODE_EMB_PATH, allow_pickle=False)
    if node_embs.shape[0] != len(node_registry):
        raise ValueError(f"node_embs rows ({node_embs.shape[0]}) must match node_registry size ({len(node_registry)})")
    node_embs = normalize_embeddings(node_embs)
    retriever = DiverEmbeddingModel(hp.RETRIEVER_MODEL_PATH, local_files_only=True)
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
    llm_api_kwargs.pop('response_schema')

if hp.LOAD_EXISTING and os.path.exists(f'{RESULTS_DIR}/all_eval_sample_dicts.pkl'):
  all_eval_samples, all_eval_metric_dfs = load_exp(RESULTS_DIR, hp, semantic_root_node, node_registry, logger)
  logger.info(f'Loaded existing experiment with {len(all_eval_samples)} eval samples and {len(all_eval_metric_dfs)} eval metric dfs')
  if len(all_eval_samples) > 0:
    eval_metric_df = pd.DataFrame([sample.compute_eval_metrics(k=10) for sample in all_eval_samples])
    logger.info('; '.join([f'{k}: {eval_metric_df[k].mean():.2f}' for k in eval_metric_df.columns]))
else: 
  all_eval_samples, all_eval_metric_dfs = [], []
  for i in range(min(examples_df.shape[0], hp.NUM_EVAL_SAMPLES)):
    gold_paths = [doc_id_to_path[doc_id] for doc_id in examples_df.iloc[i]['gold_ids'] if doc_id in doc_id_to_path]
    if len(gold_paths) < len(examples_df.iloc[i]['gold_ids']):
        logger.warning(f"Some gold IDs for example {i} not found in document paths.")
    sample = InferSample(
        semantic_root_node,
        node_registry,
        hp=hp,
        logger=logger,
        query=examples_df.iloc[i]['query'][:hp.MAX_QUERY_CHAR_LEN],
        gold_paths=gold_paths,
        excluded_ids_set=set(examples_df.iloc[i]['excluded_ids']),
        )
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
    if hp.USE_RETRIEVER_TRAVERSAL:
      # Intent: replace traversal LLM scoring with retriever cosine scores over the slate nodes.
      response_jsons = []
      for sample, sample_slates in zip(all_eval_samples, slates):
        per_sample = []
        q_emb = retriever.encode_query(sample.query)
        for slate in sample_slates:
          if not slate:
            per_sample.append({"ranking": [], "relevance_scores": []})
            continue
          slate_indices = list(slate)
          slate_embs = node_embs[slate_indices]
          # Intent: compute cosine similarity via explicit L2 normalization for traversal scoring.
          q_norm = np.linalg.norm(q_emb)
          if q_norm == 0.0:
            q_norm = 1.0
          q_vec = q_emb / q_norm
          slate_norms = np.linalg.norm(slate_embs, axis=1, keepdims=True)
          slate_norms[slate_norms == 0.0] = 1.0
          slate_embs = slate_embs / slate_norms
          scores = (slate_embs @ q_vec).astype(np.float32, copy=False)
          order = np.argsort(-scores)
          ranking = [int(slate_indices[i]) for i in order.tolist()]
          relevance_scores = [[int(slate_indices[i]), float(scores[i])] for i in order.tolist()]
          per_sample.append({"ranking": ranking, "relevance_scores": relevance_scores})
        response_jsons.append(per_sample)
    else:
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
        eval_metric_df = pd.DataFrame([sample.compute_eval_metrics(k=10) for sample in all_eval_samples])
        all_eval_metric_dfs.append(eval_metric_df)
        
        # Log metrics
        wandb_log_iteration_metrics(eval_metric_df, i)
        logger.info('; '.join([f'{k}: {eval_metric_df[k].mean():.2f}' for k in eval_metric_df.columns]))
        save_exp(RESULTS_DIR, hp, llm_api, all_eval_samples, all_eval_metric_dfs, allow_overwrite=True)  
        logger.info('-'*50)
finally:
    loop.close()
#endregion



# Log final summary metrics and finish wandb run
if all_eval_metric_dfs and all_eval_samples:
    wandb_log_final_summary(all_eval_samples)

finish_wandb_logging(logger)
#endregion
