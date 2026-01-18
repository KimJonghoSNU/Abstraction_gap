import asyncio
import hashlib
import json
import logging
import os
import pickle as pkl
from typing import List

import numpy as np
import pandas as pd
from datasets import load_dataset
from json_repair import repair_json
from tqdm.autonotebook import tqdm

from flat_then_tree import flat_retrieve_hits
from hyperparams import HyperParams
from llm_apis import GenAIAPI, VllmAPI
from tree_objects import SemanticNode
from utils import (
    compute_ndcg,
    compute_node_registry,
    compute_recall,
    get_all_leaf_nodes_with_path,
    get_node_id,
    setup_logger,
)


# - 초기화: 기존 문서 임베딩 전체를 대상으로 flat retrieval을 수행한다.
#   - iter=0 (첫 반복):
#       - flat retrieval 결과에서 branch 노드만 추려서 rewrite context를 만든다.
#       - 이 context로 query rewrite를 1회 수행한다.
#   - iter>=1 (이후 반복):
#       - 매 iter마다 flat retrieval을 다시 수행한다.
#       - retrieval 결과 전체(leaf+branch)를 그대로 rewrite context로 사용한다.
#       - rewrite는 rewrite_every 주기마다 수행한다.
#   - 평가(nDCG 등):
#       - retrieval 결과에서 leaf 노드만 필터링해서 평가한다.
#       - 즉, branch는 rewrite context에는 들어가지만 평가에서는 제외된다.
#   - 결과 저장:
#       - iter별로 leaf_iter_metrics.jsonl에 기록한다.
#       - 요약 로그는 iter 평균을 출력한다.

#   핵심 포인트

#   - 검색 자체는 항상 “전체 노드(flat)” 대상으로 수행한다.
#   - branch는 “rewrite에만” 쓰이고, 평가는 “leaf만”으로 계산한다.
#   - rewrite cadence는 rewrite_every에 따르고, rewrite_at_start는 사용하지 않는다.


QE_PROMPT_TEMPLATES = {}

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
        "Previous rewritten query (if any):\n{previous_rewrite}\n\n"
        "Candidate passages:\n{gate_descs}\n"
    ),
}


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


def _compute_leaf_metrics(pred_paths: List[List[int]], gold_paths: List[List[int]]) -> dict[str, float]:
    return {
        "nDCG@10": compute_ndcg(pred_paths, gold_paths, k=10) * 100,
        "Recall@10": compute_recall(pred_paths, gold_paths, k=10) * 100,
        "Recall@100": compute_recall(pred_paths, gold_paths, k=100) * 100,
        "Recall@all": compute_recall(pred_paths, gold_paths, k=len(pred_paths)) * 100,
    }


hp = HyperParams.from_args()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
exp_dir_name = str(hp)
RESULTS_DIR = f"{BASE_DIR}/results/{hp.DATASET}/{hp.SUBSET}/{exp_dir_name}/"
os.makedirs(RESULTS_DIR, exist_ok=True)
log_path = f"{RESULTS_DIR}/run.log"
os.makedirs(os.path.dirname(log_path), exist_ok=True)
logger = setup_logger("leaf_rank_runner", log_path, logging.INFO)
with open(f"{RESULTS_DIR}/hparams.json", "w", encoding="utf-8") as f:
    json.dump(vars(hp), f, indent=2, ensure_ascii=True, sort_keys=True)

if os.path.exists(f"{BASE_DIR}/data/{hp.DATASET}/{hp.SUBSET}/documents.jsonl"):
    docs_df = pd.read_json(
        f"{BASE_DIR}/data/{hp.DATASET}/{hp.SUBSET}/documents.jsonl",
        lines=True,
        dtype={"id": str},
    )
    examples_df = pd.read_json(
        f"{BASE_DIR}/data/{hp.DATASET}/{hp.SUBSET}/examples.jsonl",
        lines=True,
        dtype={"gold_ids": List[str]},
    )
    examples_df["gold_ids"] = examples_df["gold_ids"].apply(lambda x: [str(i) for i in x])
else:
    docs_df = pd.DataFrame(load_dataset("xlangai/BRIGHT", "documents", split=hp.SUBSET))
    examples_df = pd.DataFrame(load_dataset("xlangai/BRIGHT", "examples", split=hp.SUBSET))

tree_dict = pkl.load(open(f"{BASE_DIR}/trees/{hp.DATASET}/{hp.SUBSET}/tree-{hp.TREE_VERSION}.pkl", "rb"))
semantic_root_node = SemanticNode().load_dict(tree_dict) if isinstance(tree_dict, dict) else tree_dict
node_registry = compute_node_registry(semantic_root_node)
all_leaf_nodes = get_all_leaf_nodes_with_path(semantic_root_node)
doc_id_to_path = {get_node_id(leaf.id, docs_df): path for leaf, path in all_leaf_nodes}

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

if not hp.FLAT_THEN_TREE:
    raise ValueError("--flat_then_tree is required for run_leaf_rank.py")
if not hp.RETRIEVER_MODEL_PATH:
    raise ValueError("--retriever_model_path is required when --flat_then_tree is set")
if not hp.NODE_EMB_PATH:
    raise ValueError("--node_emb_path is required when --flat_then_tree is set")

rewrite_enabled = bool(hp.REWRITE_PROMPT_NAME or hp.REWRITE_PROMPT_PATH or hp.REWRITE_CACHE_PATH)
if not rewrite_enabled:
    raise ValueError("rewrite prompt or cache is required for run_leaf_rank.py")

rewrite_template = None
if hp.REWRITE_PROMPT_NAME:
    if hp.REWRITE_PROMPT_NAME not in REWRITE_PROMPT_TEMPLATES:
        raise ValueError(
            f'Unknown --rewrite_prompt_name "{hp.REWRITE_PROMPT_NAME}". '
            f"Available: {sorted(REWRITE_PROMPT_TEMPLATES.keys())}"
        )
    rewrite_template = REWRITE_PROMPT_TEMPLATES[hp.REWRITE_PROMPT_NAME]
if hp.REWRITE_PROMPT_PATH:
    if not os.path.exists(hp.REWRITE_PROMPT_PATH):
        raise ValueError(f"--rewrite_prompt_path not found: {hp.REWRITE_PROMPT_PATH}")
    with open(hp.REWRITE_PROMPT_PATH, "r", encoding="utf-8") as f:
        rewrite_template = f.read()
if rewrite_template is None and not hp.REWRITE_CACHE_PATH:
    raise ValueError("--rewrite_prompt_name or --rewrite_prompt_path is required when rewrite is enabled")

rewrite_map: dict[str, str] = {}
if hp.REWRITE_CACHE_PATH and os.path.exists(hp.REWRITE_CACHE_PATH) and (not hp.REWRITE_FORCE_REFRESH):
    with open(hp.REWRITE_CACHE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "key" in rec and "rewritten_query" in rec:
                rewrite_map[str(rec["key"])] = str(rec["rewritten_query"])

from retrievers.diver import DiverEmbeddingModel

node_embs = np.load(hp.NODE_EMB_PATH, allow_pickle=False)
if node_embs.shape[0] != len(node_registry):
    raise ValueError(f"node_embs rows ({node_embs.shape[0]}) must match node_registry size ({len(node_registry)})")
retriever = DiverEmbeddingModel(hp.RETRIEVER_MODEL_PATH, local_files_only=True)

samples = []
for i in range(min(examples_df.shape[0], hp.NUM_EVAL_SAMPLES)):
    gold_paths = [doc_id_to_path[doc_id] for doc_id in examples_df.iloc[i]["gold_ids"] if doc_id in doc_id_to_path]
    if len(gold_paths) < len(examples_df.iloc[i]["gold_ids"]):
        logger.warning(f"Some gold IDs for example {i} not found in document paths.")
    original_query = examples_df.iloc[i]["query"][:hp.MAX_QUERY_CHAR_LEN]
    samples.append({
        "index": i,
        "original_query": original_query,
        "current_query": original_query,
        "last_rewrite": "",
        "gold_paths": gold_paths,
    })

logger.info(f"Loaded {len(samples)} eval samples.")

metrics_path = f"{RESULTS_DIR}/leaf_iter_metrics.jsonl"
rewrite_records = []

logger.info("Starting initial branch-only rewrite.")
init_prompts = []
init_meta = []
for sample in samples:
    hits = flat_retrieve_hits(
        retriever=retriever,
        query=sample["original_query"],
        node_embs=node_embs,
        node_registry=node_registry,
        topk=hp.FLAT_TOPK,
    )
    branch_hits = [h for h in hits if not h.is_leaf]
    context_descs = _hits_to_context_descs(branch_hits, node_registry, hp.REWRITE_CONTEXT_TOPK, hp.MAX_DOC_DESC_CHAR_LEN)
    cache_key = _rewrite_cache_key("leaf_init", sample["original_query"], context_descs, iter_idx=None)
    if (not hp.REWRITE_FORCE_REFRESH) and (cache_key in rewrite_map):
        rewrite = rewrite_map[cache_key]
        sample["last_rewrite"] = rewrite
        sample["current_query"] = _apply_rewrite(hp.REWRITE_MODE, sample["original_query"], rewrite)
        continue
    if rewrite_template is None:
        raise ValueError("Rewrite enabled but no prompt template is available.")
    init_prompts.append(_format_rewrite_prompt(
        rewrite_template,
        sample["original_query"],
        "",
        context_descs,
    ))
    init_meta.append({
        "sample": sample,
        "cache_key": cache_key,
        "context_descs": context_descs,
    })

if init_prompts:
    init_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(init_loop)
    try:
        init_outputs = init_loop.run_until_complete(
            llm_api.run_batch(init_prompts, max_concurrent_calls=hp.LLM_MAX_CONCURRENT_CALLS)
        )
    finally:
        init_loop.close()
        asyncio.set_event_loop(None)
    for meta, out in zip(init_meta, init_outputs):
        rewrite = _clean_qe_text(out)
        rewrite_map[meta["cache_key"]] = rewrite
        meta["sample"]["last_rewrite"] = rewrite
        meta["sample"]["current_query"] = _apply_rewrite(hp.REWRITE_MODE, meta["sample"]["original_query"], rewrite)
        rewrite_records.append({
            "key": meta["cache_key"],
            "rewritten_query": rewrite,
            "prompt_name": hp.REWRITE_PROMPT_NAME,
            "llm": hp.LLM,
            "context_descs": meta.get("context_descs", []),
        })

if hp.REWRITE_CACHE_PATH and rewrite_records:
    os.makedirs(os.path.dirname(hp.REWRITE_CACHE_PATH) or ".", exist_ok=True)
    with open(hp.REWRITE_CACHE_PATH, "a", encoding="utf-8") as f:
        for rec in rewrite_records:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")

for iter_idx in range(hp.NUM_ITERS):
    iter_metrics = []
    iter_hits = []
    for sample in tqdm(samples, desc=f"Iter {iter_idx} retrieval"):
        hits = flat_retrieve_hits(
            retriever=retriever,
            query=sample["current_query"],
            node_embs=node_embs,
            node_registry=node_registry,
            topk=hp.FLAT_TOPK,
        )
        iter_hits.append(hits)
        leaf_hits = [h for h in hits if h.is_leaf]
        pred_paths = [list(h.path) for h in leaf_hits[:hp.FLAT_TOPK]]
        metrics = _compute_leaf_metrics(pred_paths, sample["gold_paths"])
        metrics["iter"] = iter_idx
        metrics["query_idx"] = sample["index"]
        metrics["query"] = sample["original_query"]
        iter_metrics.append(metrics)

    with open(metrics_path, "a", encoding="utf-8") as f:
        for metrics in iter_metrics:
            f.write(json.dumps(metrics, ensure_ascii=True) + "\n")

    mean_ndcg = float(np.mean([m["nDCG@10"] for m in iter_metrics])) if iter_metrics else 0.0
    mean_r10 = float(np.mean([m["Recall@10"] for m in iter_metrics])) if iter_metrics else 0.0
    mean_r100 = float(np.mean([m["Recall@100"] for m in iter_metrics])) if iter_metrics else 0.0
    logger.info(
        f"Iter {iter_idx} mean metrics: nDCG@10={mean_ndcg:.2f}, "
        f"Recall@10={mean_r10:.2f}, Recall@100={mean_r100:.2f}"
    )

    if hp.REWRITE_EVERY > 0 and ((iter_idx + 1) % hp.REWRITE_EVERY == 0):
        rewrite_prompts = []
        rewrite_meta = []
        for sample, hits in zip(samples, iter_hits):
            context_descs = _hits_to_context_descs(
                hits,
                node_registry,
                hp.REWRITE_CONTEXT_TOPK,
                hp.MAX_DOC_DESC_CHAR_LEN,
            )
            cache_key = _rewrite_cache_key(
                "leaf_iter",
                f"{sample['original_query']}||{sample['last_rewrite']}",
                context_descs,
                iter_idx=iter_idx,
            )
            if (not hp.REWRITE_FORCE_REFRESH) and (cache_key in rewrite_map):
                rewrite = rewrite_map[cache_key]
                sample["last_rewrite"] = rewrite
                sample["current_query"] = _apply_rewrite(hp.REWRITE_MODE, sample["original_query"], rewrite)
                continue
            if rewrite_template is None:
                raise ValueError("Rewrite enabled but no prompt template is available.")
            rewrite_prompts.append(_format_rewrite_prompt(
                rewrite_template,
                sample["original_query"],
                sample["last_rewrite"],
                context_descs,
            ))
            rewrite_meta.append({
                "sample": sample,
                "cache_key": cache_key,
                "context_descs": context_descs,
            })
        if rewrite_prompts:
            rewrite_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(rewrite_loop)
            try:
                rewrite_outputs = rewrite_loop.run_until_complete(
                    llm_api.run_batch(
                        rewrite_prompts,
                        max_concurrent_calls=hp.LLM_MAX_CONCURRENT_CALLS,
                        staggering_delay=hp.LLM_API_STAGGERING_DELAY,
                    )
                )
            finally:
                rewrite_loop.close()
                asyncio.set_event_loop(None)
            rewrite_records = []
            for meta, out in zip(rewrite_meta, rewrite_outputs):
                rewrite = _clean_qe_text(out)
                rewrite_map[meta["cache_key"]] = rewrite
                meta["sample"]["last_rewrite"] = rewrite
                meta["sample"]["current_query"] = _apply_rewrite(hp.REWRITE_MODE, meta["sample"]["original_query"], rewrite)
                rewrite_records.append({
                    "key": meta["cache_key"],
                    "rewritten_query": rewrite,
                    "prompt_name": hp.REWRITE_PROMPT_NAME,
                    "llm": hp.LLM,
                    "context_descs": meta.get("context_descs", []),
                })
            if hp.REWRITE_CACHE_PATH and rewrite_records:
                os.makedirs(os.path.dirname(hp.REWRITE_CACHE_PATH) or ".", exist_ok=True)
                with open(hp.REWRITE_CACHE_PATH, "a", encoding="utf-8") as f:
                    for rec in rewrite_records:
                        f.write(json.dumps(rec, ensure_ascii=True) + "\n")

logger.info("Completed run_leaf_rank.py.")
