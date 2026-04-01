import asyncio
import hashlib
import json
import logging
import os
import pickle as pkl
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from json_repair import repair_json
from tqdm.autonotebook import tqdm

from emr_memory_utils import (
    EMR_CROSS_ENCODER_MODEL_NAME,
    EmrMemoryCompressor,
    RewritePromptTokenBudgeter,
    append_emr_history_entry as _shared_append_emr_history_entry,
    build_current_doc_memory_items as _shared_build_current_doc_memory_items,
    build_doc_text_map as _shared_build_doc_text_map,
    build_emr_prompt_memory as _shared_build_emr_prompt_memory,
    compact_text as _shared_compact_text,
    format_emr_doc_memory as _shared_format_emr_doc_memory,
    format_emr_history as _shared_format_emr_history,
    materialize_accumulated_doc_pool as _shared_materialize_accumulated_doc_pool,
    update_accumulated_doc_bank as _shared_update_accumulated_doc_bank,
)
from flat_then_tree import flat_retrieve_hits
from hyperparams import HyperParams
from llm_apis import GenAIAPI, VllmAPI
from retrievers import build_retriever
from tree_objects import SemanticNode
from utils import (
    compute_ndcg,
    compute_node_registry,
    compute_recall,
    filter_excluded_predictions,
    get_all_leaf_nodes_with_path,
    get_node_id,
    resolve_node_emb_registry_indices,
    setup_logger,
)


# - 초기화: flat retrieval을 수행한다.
#   - iter=0 (첫 반복):
#       - rewrite context는 leaf-only로 구성한다.
#       - 이 context로 query rewrite를 1회 수행한다.
#   - iter>=1 (이후 반복):
#       - 매 iter마다 flat retrieval을 다시 수행한다.
#       - rewrite context는 leaf-only로 구성한다.
#       - rewrite는 rewrite_every 주기마다 수행한다.
#   - 평가(nDCG 등):
#       - retrieval 결과를 leaf 기준으로 모은 뒤, document ID ranking으로 평가한다.
#   - 결과 저장:
#       - iter별로 leaf_iter_metrics.jsonl에 기록한다.
#       - 요약 로그는 iter 평균을 출력한다.
#
#   핵심 포인트
#
#   - 검색은 leaf-only 옵션에 따라 “leaf만” 또는 “전체 노드(flat)” 대상으로 수행한다.
#   - rewrite context는 leaf-only로 구성한다.
#   - rewrite cadence는 rewrite_every에 따르고, rewrite_at_start는 사용하지 않는다.


QE_PROMPT_TEMPLATES = {}

from rewrite_prompts import REWRITE_PROMPT_TEMPLATES

def _extract_qe_candidate(text: str) -> str:
    text = text.split("</think>\n")[-1].strip()
    if "```" in text:
        try:
            parts = text.split("```")
            fenced = [parts[i] for i in range(1, len(parts), 2)]
            if fenced:
                text = fenced[-1].strip()
        except Exception:
            pass
    return text.strip()


def _parse_rewrite_output(text: str) -> str:
    text = _extract_qe_candidate(text)
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


def _clean_qe_text(text: str) -> str:
    return _parse_rewrite_output(text)


def _parse_stop_decision(text: str) -> str:
    candidate = _extract_qe_candidate(text).strip().lower()
    if candidate == "stop":
        return "stop"
    if candidate == "continue":
        return "continue"
    return "continue"


def _get_relevance_definition(subset_name: str) -> str:
    name = str(subset_name or "").strip().lower()
    relevance_definitions = {
        "leetcode": "The relevant documents share the core algorithm or data structure needed to solve the coding problem.",
        "theoremqa_questions": "A document is relevant if it references the same or a closely related theorem needed to answer the question.",
        "pony": "A document is relevant if it provides the exact syntax, concept, or reference needed to answer the coding or configuration question.",
        "stackexchange": "A document is relevant if it can directly support an accepted or high-quality answer to the query, not just a semantically nearby topic.",
        "scifact": "A document is relevant if it contains sufficient evidence to support or refute the scientific claim.",
        "nq": "A document is relevant if it directly contains the answer to the question.",
        "fiqa": "A document is relevant if it contains the specific information needed to answer the financial question.",
        "scidocs": "A document is relevant if it is directly cited by or strongly tied to the query paper's core scientific content.",
    }
    relevance_definitions["aops"] = relevance_definitions["theoremqa_questions"]
    relevance_definitions["theoremqa_theorems"] = relevance_definitions["theoremqa_questions"]
    relevance_definitions["theoq"] = relevance_definitions["theoremqa_questions"]
    relevance_definitions["theot"] = relevance_definitions["theoremqa_questions"]
    return relevance_definitions.get(name, relevance_definitions["stackexchange"])


def _format_rewrite_prompt(
    template: str,
    original_query: str,
    previous_rewrite: str,
    context_descs: List[str],
    relevance_definition: str = "",
    history_actions: str = "",
    memory_docs: str = "",
) -> str:
    context_blob = "\n".join([x for x in context_descs if x])
    try:
        return template.format(
            gate_descs=context_blob,
            leaf_descs=context_blob,
            original_query=(original_query or ""),
            previous_rewrite=(previous_rewrite or ""),
            relevance_definition=(relevance_definition or ""),
            history_actions=(history_actions or ""),
            memory_docs=(memory_docs or ""),
        )
    except KeyError:
        return (
            template
            .replace("{gate_descs}", context_blob)
            .replace("{leaf_descs}", context_blob)
            .replace("{original_query}", original_query or "")
            .replace("{previous_rewrite}", previous_rewrite or "")
            .replace("{relevance_definition}", relevance_definition or "")
            .replace("{history_actions}", history_actions or "")
            .replace("{memory_docs}", memory_docs or "")
        )


def _build_stop_prompt(
    template: str,
    *,
    original_query: str,
    previous_rewrite: str,
    context_descs: List[str],
    relevance_definition: str,
) -> str:
    return _format_rewrite_prompt(
        template,
        original_query,
        previous_rewrite,
        context_descs,
        relevance_definition=relevance_definition,
    )


def _rewrite_cache_key(
    prefix: str,
    query: str,
    context_descs: List[str],
    iter_idx: Optional[int] = None,
    *,
    prompt_name: str = "",
    history_actions: str = "",
    memory_docs: str = "",
) -> str:
    context_blob = "\n".join([x for x in context_descs if x]).strip()
    context_sig = hashlib.md5(context_blob.encode("utf-8")).hexdigest() if context_blob else "none"
    memory_blob = "||".join([
        str(prompt_name or ""),
        str(history_actions or "").strip(),
        str(memory_docs or "").strip(),
    ])
    memory_sig = hashlib.md5(memory_blob.encode("utf-8")).hexdigest() if memory_blob.strip() else "none"
    iter_tag = f"||iter={iter_idx}" if iter_idx is not None else ""
    return f"{prefix}||{query}||ctx={context_sig}||mem={memory_sig}{iter_tag}"


def _mark_sample_stopped(sample: Dict[str, Any], *, stop_iter: int, stop_phase: str) -> None:
    # Intent: true early exit removes the sample from future retrieval and rewrite iterations once stop is decided.
    sample["active"] = False
    sample["stop_iter"] = int(stop_iter)
    sample["stop_phase"] = str(stop_phase)


def _apply_rewrite(mode: str, base_query: str, rewrite: str) -> str:
    rewrite = (rewrite or "").strip()
    if not rewrite:
        return base_query
    if mode == "replace":
        return rewrite
    return (base_query + " " + rewrite).strip()


def _apply_leaf_rewrite(sample: Dict[str, Any], mode: str, rewrite: str, *, thinkqe_style: bool) -> str:
    rewrite = (rewrite or "").strip()
    if thinkqe_style:
        # Intent: ThinkQE-style query evolution accumulates the new rewrite onto the live query state instead of rebuilding from the original query.
        base_query = str(sample.get("current_query", sample.get("original_query", "")))
        return (base_query + " " + rewrite).strip() if rewrite else base_query
    return _apply_rewrite(mode, str(sample.get("original_query", "")), rewrite)


def _hits_to_context_hits(hits: List[object], topk: int) -> List[object]:
    out: List[object] = []
    seen: set[int] = set()
    for h in hits:
        ridx = int(h.registry_idx)
        if ridx in seen:
            continue
        seen.add(ridx)
        out.append(h)
        if len(out) >= topk:
            break
    return out


def _hits_to_context_descs(hits: List[object], node_registry: List[object], topk: int, max_desc_len: Optional[int]) -> List[str]:
    descs: List[str] = []
    context_hits = _hits_to_context_hits(hits, topk)
    for h in context_hits:
        ridx = int(h.registry_idx)
        desc = node_registry[ridx].desc
        if max_desc_len:
            desc = desc[:max_desc_len]
        descs.append(desc)
    return descs


def _hits_to_doc_context_rows(
    hits: List[object],
    node_registry: List[object],
    path_to_doc_id: Dict[Tuple[int, ...], str],
    max_desc_len: Optional[int],
    excluded_doc_ids: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen_doc_ids: set[str] = set()
    excluded = {str(x) for x in (excluded_doc_ids or [])}
    for h in hits:
        path = tuple(h.path)
        doc_id = path_to_doc_id.get(path)
        if not doc_id or doc_id in seen_doc_ids or doc_id in excluded:
            continue
        seen_doc_ids.add(doc_id)
        desc = node_registry[int(h.registry_idx)].desc
        if max_desc_len:
            desc = desc[:max_desc_len]
        rows.append({
            "doc_id": str(doc_id),
            "path": path,
            "desc": desc,
        })
    return rows


def _extend_unique_doc_ids(base_doc_ids: Sequence[str], appended_doc_ids: Sequence[str]) -> List[str]:
    merged: List[str] = [str(x) for x in base_doc_ids]
    seen = set(merged)
    for doc_id in appended_doc_ids:
        doc_id = str(doc_id)
        if doc_id in seen:
            continue
        seen.add(doc_id)
        merged.append(doc_id)
    return merged


def _select_thinkqe_context_rows(
    sample: Dict[str, Any],
    candidate_rows: Sequence[Dict[str, Any]],
    topk: int,
) -> Tuple[List[Dict[str, Any]], List[str], List[str], List[str]]:
    prev_context_doc_ids = [str(x) for x in sample.get("thinkqe_prev_context_doc_ids", [])]
    prev_context_set = set(prev_context_doc_ids)
    blocklist_before = [str(x) for x in sample.get("thinkqe_blocklist_doc_ids", [])]
    blocklist_set = set(blocklist_before)
    selected_rows: List[Dict[str, Any]] = []
    filtered_doc_ids: List[str] = []
    for row in candidate_rows:
        doc_id = str(row["doc_id"])
        # Intent: ThinkQE-style filtering drops previous-context and blocklisted docs before selecting fresh rewrite evidence.
        if doc_id in prev_context_set or doc_id in blocklist_set:
            filtered_doc_ids.append(doc_id)
            continue
        selected_rows.append(row)
        if len(selected_rows) >= topk:
            break
    blocklist_after = _extend_unique_doc_ids(blocklist_before, filtered_doc_ids)
    selected_doc_ids = [str(row["doc_id"]) for row in selected_rows]
    return selected_rows, selected_doc_ids, filtered_doc_ids, blocklist_after


def _compute_leaf_metrics(
    pred_doc_ids: Sequence[str],
    gold_doc_ids: Sequence[str],
    excluded_doc_ids: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    # Intent: remove BRIGHT excluded_ids before metric cutoff so leaf-only scores follow the benchmark protocol.
    ranked_doc_ids = filter_excluded_predictions(list(pred_doc_ids), excluded_doc_ids)
    return {
        "nDCG@10": compute_ndcg(ranked_doc_ids, list(gold_doc_ids), k=10) * 100,
        "Recall@10": compute_recall(ranked_doc_ids, list(gold_doc_ids), k=10) * 100,
        "Recall@100": compute_recall(ranked_doc_ids, list(gold_doc_ids), k=100) * 100,
        "Recall@all": compute_recall(ranked_doc_ids, list(gold_doc_ids), k=len(ranked_doc_ids)) * 100,
    }


def _paths_to_ranked_doc_ids(paths: Sequence[Tuple[int, ...]], path_to_doc_id: Dict[Tuple[int, ...], str]) -> List[str]:
    ranked: List[str] = []
    seen: set[str] = set()
    for path in paths:
        doc_id = path_to_doc_id.get(tuple(path))
        if not doc_id:
            continue
        if doc_id in seen:
            continue
        seen.add(doc_id)
        ranked.append(doc_id)
    return ranked


def _compact_text(text: str) -> str:
    return _shared_compact_text(text)


def _build_doc_text_map(docs_df: pd.DataFrame) -> Dict[str, str]:
    return _shared_build_doc_text_map(docs_df)


def _format_emr_history(history_entries: Sequence[Dict[str, Any]]) -> str:
    return _shared_format_emr_history(history_entries)


def _format_emr_doc_memory(memory_items: Sequence[Dict[str, Any]]) -> str:
    return _shared_format_emr_doc_memory(memory_items)


def _build_current_doc_memory_items(
    *,
    query_for_memory: str,
    retrieved_doc_ids: Sequence[str],
    doc_text_by_id: Dict[str, str],
    doc_topk: int,
) -> List[str]:
    del query_for_memory
    return _shared_build_current_doc_memory_items(
        retrieved_doc_ids=retrieved_doc_ids,
        doc_text_by_id=doc_text_by_id,
        doc_topk=doc_topk,
    )


def _update_accumulated_doc_bank(
    sample: Dict[str, Any],
    *,
    current_doc_ids: Sequence[str],
    iter_idx: int,
    query_for_memory: str,
) -> None:
    bank = sample.setdefault("emr_doc_bank", {})
    next_order = int(sample.get("emr_doc_bank_next_order", 0) or 0)
    sample["emr_doc_bank_next_order"] = _shared_update_accumulated_doc_bank(
        bank,
        next_order=next_order,
        current_doc_ids=current_doc_ids,
        iter_idx=iter_idx,
        query_for_memory=query_for_memory,
    )


def _materialize_accumulated_doc_pool(sample: Dict[str, Any]) -> List[str]:
    return _shared_materialize_accumulated_doc_pool(sample.get("emr_doc_bank", {}))


def _build_emr_prompt_memory(
    *,
    sample: Dict[str, Any],
    mode: str,
    query_for_memory: str,
    retrieved_doc_ids: Sequence[str],
    compressor: Optional[EmrMemoryCompressor],
    doc_text_by_id: Dict[str, str],
    doc_topk: int,
    iter_idx: int,
    rewrite_template: Optional[str],
    original_query: str,
    previous_rewrite: str,
    context_descs: List[str],
    relevance_definition: str,
    prompt_budgeter: Optional[RewritePromptTokenBudgeter],
    compression_mode: str,
    max_memory_tokens: Optional[int],
) -> Dict[str, Any]:
    current_doc_ids = _build_current_doc_memory_items(
        query_for_memory=query_for_memory,
        retrieved_doc_ids=retrieved_doc_ids,
        doc_text_by_id=doc_text_by_id,
        doc_topk=doc_topk,
    )
    if str(mode).lower() == "accumulated":
        _update_accumulated_doc_bank(
            sample,
            current_doc_ids=current_doc_ids,
            iter_idx=iter_idx,
            query_for_memory=query_for_memory,
        )
    elif str(mode).lower() != "off":
        raise ValueError(f"Unknown --leaf_emr_memory_mode: {mode}")
    return _shared_build_emr_prompt_memory(
        mode=mode,
        history_entries=sample.get("emr_history", []),
        doc_bank=sample.get("emr_doc_bank", {}),
        query_for_memory=query_for_memory,
        compressor=compressor,
        doc_text_by_id=doc_text_by_id,
        render_prompt=lambda history_text, memory_text: _format_rewrite_prompt(
            rewrite_template or "{original_query}\n{previous_rewrite}\n{history_actions}\n{memory_docs}\n{gate_descs}",
            original_query,
            previous_rewrite,
            context_descs,
            relevance_definition=relevance_definition,
            history_actions=history_text,
            memory_docs=memory_text,
        ),
        prompt_budgeter=prompt_budgeter,
        compression_mode=compression_mode,
        max_memory_tokens=max_memory_tokens,
        memory_source_label="accumulated",
    )


def _append_emr_history_entry(
    sample: Dict[str, Any],
    *,
    applied_query: str,
    retrieved_doc_ids: Sequence[str],
    history_rank_topk: int,
) -> None:
    history = sample.setdefault("emr_history", [])
    _shared_append_emr_history_entry(
        history,
        applied_query=applied_query,
        retrieved_doc_ids=retrieved_doc_ids,
        history_rank_topk=history_rank_topk,
    )


hp = HyperParams.from_args()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
exp_dir_name = str(hp)
RESULTS_DIR = f"{BASE_DIR}/results/{hp.DATASET}/{hp.SUBSET}/{exp_dir_name}/"
os.makedirs(RESULTS_DIR, exist_ok=True)
metrics_path = f"{RESULTS_DIR}/leaf_iter_metrics.jsonl"
iter_records_path = f"{RESULTS_DIR}/leaf_iter_records.jsonl"
done_marker_path = f"{RESULTS_DIR}/leaf_iter_done.json"
# Intent: let batch launchers skip only completed runs while allowing incomplete runs to restart cleanly.
if os.path.exists(done_marker_path):
    print(f"Skipping run because completion marker already exists: {done_marker_path}")
    raise SystemExit(0)
stale_leaf_outputs = [path for path in [metrics_path, iter_records_path] if os.path.exists(path)]
if stale_leaf_outputs:
    print(f"Removing stale leaf output files before restart: {stale_leaf_outputs}")
    for stale_path in stale_leaf_outputs:
        os.remove(stale_path)
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
full_node_registry = compute_node_registry(semantic_root_node)
node_registry = full_node_registry
all_leaf_nodes = get_all_leaf_nodes_with_path(semantic_root_node)
doc_id_to_path = {get_node_id(leaf.id, docs_df): path for leaf, path in all_leaf_nodes}
path_to_doc_id = {tuple(path): str(doc_id) for doc_id, path in doc_id_to_path.items()}
doc_text_by_id = _build_doc_text_map(docs_df)

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
stop_enabled = bool(getattr(hp, "LEAF_ALLOW_STOP", False))
leaf_thinkqe_style = bool(getattr(hp, "LEAF_THINKQE_STYLE", False))
if leaf_thinkqe_style:
    logger.info(
        "ThinkQE-style leaf loop enabled: previous rewrite-context exclusion + cumulative blocklist + accumulated concat updates."
    )
    if str(hp.REWRITE_MODE).lower() != "concat":
        logger.warning("Ignoring --rewrite_mode=%s because --leaf_thinkqe_style always uses concat accumulation.", hp.REWRITE_MODE)

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

stop_template = None
if stop_enabled:
    stop_prompt_name = str(getattr(hp, "LEAF_STOP_PROMPT_NAME", "leaf_stop_judge_v1") or "leaf_stop_judge_v1").strip()
    if stop_prompt_name not in REWRITE_PROMPT_TEMPLATES:
        raise ValueError(
            f'Unknown --leaf_stop_prompt_name "{stop_prompt_name}". '
            f"Available: {sorted(REWRITE_PROMPT_TEMPLATES.keys())}"
        )
    stop_template = REWRITE_PROMPT_TEMPLATES[stop_prompt_name]
else:
    stop_prompt_name = ""

rewrite_map: Dict[str, str] = {}
if hp.REWRITE_CACHE_PATH and os.path.exists(hp.REWRITE_CACHE_PATH) and (not hp.REWRITE_FORCE_REFRESH):
    with open(hp.REWRITE_CACHE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "key" in rec and "rewritten_query" in rec:
                rewrite_map[str(rec["key"])] = str(rec["rewritten_query"])

stop_cache_path = str(getattr(hp, "LEAF_STOP_CACHE_PATH", "") or "").strip()
if not stop_cache_path and hp.REWRITE_CACHE_PATH:
    stop_cache_path = f"{hp.REWRITE_CACHE_PATH}.stop"
stop_map: Dict[str, str] = {}
if stop_enabled and stop_cache_path and os.path.exists(stop_cache_path) and (not hp.REWRITE_FORCE_REFRESH):
    with open(stop_cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "key" in rec and "decision" in rec:
                stop_map[str(rec["key"])] = str(rec["decision"])

node_embs = np.load(hp.NODE_EMB_PATH, allow_pickle=False)
emb_registry_indices = resolve_node_emb_registry_indices(full_node_registry, int(node_embs.shape[0]))
if node_embs.shape[0] != len(full_node_registry):
    logger.warning(
        "Resolved node embedding row mismatch via legacy registry alignment: emb_rows=%d registry_rows=%d",
        int(node_embs.shape[0]),
        len(full_node_registry),
    )
aligned_node_registry = [full_node_registry[int(idx)] for idx in emb_registry_indices.tolist()]
if hp.LEAF_ONLY_RETRIEVAL:
    leaf_row_indices = [
        row_idx for row_idx, node in enumerate(aligned_node_registry)
        if (not node.child) or (len(node.child) == 0)
    ]
    if not leaf_row_indices:
        raise ValueError("leaf-only retrieval requested but no leaf nodes were found.")
    # Intent: keep row-to-node alignment stable even when legacy embeddings skipped blank-description nodes.
    node_embs = node_embs[leaf_row_indices]
    node_registry = [aligned_node_registry[row_idx] for row_idx in leaf_row_indices]
else:
    node_registry = aligned_node_registry
retriever = build_retriever(hp.RETRIEVER_MODEL_PATH, subset=hp.SUBSET, local_files_only=True)
relevance_definition = _get_relevance_definition(hp.SUBSET)

memory_mode = str(getattr(hp, "LEAF_EMR_MEMORY_MODE", "off") or "off").lower()
memory_compression = str(getattr(hp, "LEAF_EMR_COMPRESSION", "on") or "on").lower()
emr_compressor: Optional[EmrMemoryCompressor] = None
rewrite_prompt_budgeter: Optional[RewritePromptTokenBudgeter] = None
if memory_mode != "off":
    emr_compressor = EmrMemoryCompressor(
        model_name=EMR_CROSS_ENCODER_MODEL_NAME,
        sent_topk=int(getattr(hp, "LEAF_EMR_SENT_TOPK", 10) or 10),
        logger=logger,
    )
    rewrite_prompt_budgeter = RewritePromptTokenBudgeter(
        model_name=str(hp.LLM),
        backend=str(hp.LLM_API_BACKEND),
        logger=logger,
    )
    logger.info(
        "EMR rewrite memory enabled | mode=%s | compression=%s | history_topk=%d | doc_topk=%d | sent_topk=%d | max_memory_tokens=%d | model_context_tokens=%s",
        memory_mode,
        memory_compression,
        int(getattr(hp, "LEAF_EMR_HISTORY_RANK_TOPK", 10) or 10),
        int(getattr(hp, "LEAF_EMR_DOC_TOPK", 10) or 10),
        int(getattr(hp, "LEAF_EMR_SENT_TOPK", 10) or 10),
        int(getattr(hp, "LEAF_EMR_MEMORY_MAX_TOKENS", 0) or 0),
        str(rewrite_prompt_budgeter.max_context_tokens if rewrite_prompt_budgeter is not None else None),
    )

samples = []
for i in range(min(examples_df.shape[0], hp.NUM_EVAL_SAMPLES)):
    gold_doc_ids = [str(doc_id) for doc_id in examples_df.iloc[i]["gold_ids"] if doc_id in doc_id_to_path]
    gold_paths = [doc_id_to_path[doc_id] for doc_id in gold_doc_ids]
    if len(gold_doc_ids) < len(examples_df.iloc[i]["gold_ids"]):
        logger.warning(f"Some gold IDs for example {i} not found in document paths.")
    original_query = examples_df.iloc[i]["query"][:hp.MAX_QUERY_CHAR_LEN]
    samples.append({
        "index": i,
        "original_query": original_query,
        "current_query": original_query,
        "last_rewrite": "",
        "active": True,
        "gold_doc_ids": gold_doc_ids,
        "gold_paths": gold_paths,
        "excluded_ids": [str(x) for x in examples_df.iloc[i].get("excluded_ids", [])],
        "stop_iter": None,
        "stop_phase": "",
        "emr_history": [],
        "emr_doc_bank": {},
        "emr_doc_bank_next_order": 0,
        "thinkqe_prev_context_doc_ids": [],
        "thinkqe_blocklist_doc_ids": [],
    })

logger.info(f"Loaded {len(samples)} eval samples.")
rewrite_records = []

# Intent: preserve legacy behavior by default, while allowing no-initial-rewrite ablation via a flag.
if hp.LEAF_NO_INITIAL_REWRITE:
    logger.info("Skipping initial rewrite due to --leaf_no_initial_rewrite.")
else:
    logger.info("Starting initial rewrite.")
    init_candidates = []
    init_iter_records = []
    stop_cache_records = []
    rewrite_records = []
    for sample in samples:
        if not bool(sample.get("active", True)):
            continue
        hits = flat_retrieve_hits(
            retriever=retriever,
            query=sample["original_query"],
            node_embs=node_embs,
            node_registry=node_registry,
            topk=hp.FLAT_TOPK,
        )
        rewrite_hits = [h for h in hits if h.is_leaf] if hp.LEAF_ONLY_RETRIEVAL else [h for h in hits if not h.is_leaf]
        if leaf_thinkqe_style:
            context_rows = _hits_to_doc_context_rows(
                rewrite_hits,
                node_registry,
                path_to_doc_id,
                hp.MAX_DOC_DESC_CHAR_LEN,
                excluded_doc_ids=sample.get("excluded_ids", []),
            )
            context_rows = context_rows[:int(hp.REWRITE_CONTEXT_TOPK)]
            context_descs = [str(row["desc"]) for row in context_rows]
            context_path_tuples = [tuple(row["path"]) for row in context_rows]
            context_doc_ids = [str(row["doc_id"]) for row in context_rows]
        else:
            context_hits = _hits_to_context_hits(rewrite_hits, hp.REWRITE_CONTEXT_TOPK)
            context_descs = _hits_to_context_descs(rewrite_hits, node_registry, hp.REWRITE_CONTEXT_TOPK, hp.MAX_DOC_DESC_CHAR_LEN)
            context_path_tuples = [tuple(h.path) for h in context_hits]
            context_doc_ids = _paths_to_ranked_doc_ids(context_path_tuples, path_to_doc_id)
        retrieved_path_tuples = [tuple(h.path) for h in rewrite_hits[:hp.FLAT_TOPK]]
        retrieved_doc_ids = _paths_to_ranked_doc_ids(retrieved_path_tuples, path_to_doc_id)
        retrieved_doc_ids_eval = filter_excluded_predictions(retrieved_doc_ids, sample.get("excluded_ids", []))
        stop_cache_key = _rewrite_cache_key(
            "leaf_stop_init",
            sample["current_query"] if leaf_thinkqe_style else sample["original_query"],
            context_descs,
            iter_idx=None,
            prompt_name=stop_prompt_name,
        ) if stop_enabled else ""
        init_candidates.append({
            "sample": sample,
            "context_descs": context_descs,
            "retrieved_doc_ids": retrieved_doc_ids,
            "thinkqe_context_doc_ids": context_doc_ids,
            "base_record": {
            "phase": "initial_rewrite",
            "iter": -1,
            "query_idx": int(sample["index"]),
            "query": sample["original_query"],
            "query_for_retrieval": sample["original_query"],
            "retrieval_topk": int(hp.FLAT_TOPK),
            "retrieved_paths": [list(p) for p in retrieved_path_tuples],
            "retrieved_doc_ids": retrieved_doc_ids,
            "retrieved_doc_ids_eval": retrieved_doc_ids_eval,
            "rewrite_context_topk": int(hp.REWRITE_CONTEXT_TOPK),
            "rewrite_context_paths": [list(p) for p in context_path_tuples],
            "rewrite_context_doc_ids": list(context_doc_ids),
            "rewrite_triggered": False,
            "rewrite_stop_decision": "",
            "stop_iter": None,
            "stop_phase": "",
            "stop_judge_triggered": bool(stop_enabled),
            "stop_judge_cache_hit": False,
            "thinkqe_style_enabled": bool(leaf_thinkqe_style),
            "thinkqe_context_doc_ids": list(context_doc_ids),
            "thinkqe_prev_context_doc_ids_before": [],
            "thinkqe_filtered_doc_ids_this_round": [],
            "thinkqe_blocklist_doc_ids_after": list(sample.get("thinkqe_blocklist_doc_ids", [])),
            "thinkqe_rewrite_skipped_no_new_context": False,
            "emr_memory_mode": memory_mode,
            "emr_memory_compression": memory_compression,
            "emr_history": list(sample.get("emr_history", [])),
            "emr_history_prompt_text": "",
            "emr_memory_source": "off" if memory_mode == "off" else memory_mode,
            "emr_memory_doc_ids": [],
            "emr_memory_docs_count": 0,
            "emr_memory_prompt_text": "",
            "emr_memory_overflow_dropped": [],
            "emr_memory_pool_doc_ids": [],
            "emr_memory_selected_sentences": [],
            "emr_memory_prompt_tokens": None,
            "emr_prompt_total_tokens": None,
            "emr_prompt_budget_tokens": None,
            "emr_model_context_tokens": rewrite_prompt_budgeter.max_context_tokens if rewrite_prompt_budgeter is not None else None,
            },
            "stop_cache_key": stop_cache_key,
            "stop_decision": "continue",
        })

    if stop_enabled:
        stop_prompts = []
        stop_meta = []
        for candidate in init_candidates:
            stop_cache_key = str(candidate.get("stop_cache_key", ""))
            if (not hp.REWRITE_FORCE_REFRESH) and stop_cache_key and (stop_cache_key in stop_map):
                candidate["stop_decision"] = _parse_stop_decision(stop_map[stop_cache_key])
                candidate["base_record"]["stop_judge_cache_hit"] = True
                continue
            stop_prompts.append(_build_stop_prompt(
                stop_template,
                original_query=candidate["sample"]["original_query"],
                previous_rewrite="",
                context_descs=candidate["context_descs"],
                relevance_definition=relevance_definition,
            ))
            stop_meta.append(candidate)
        if stop_prompts:
            stop_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(stop_loop)
            try:
                stop_outputs = stop_loop.run_until_complete(
                    llm_api.run_batch(
                        stop_prompts,
                        max_concurrent_calls=hp.LLM_MAX_CONCURRENT_CALLS,
                        temperature=0.0,
                    )
                )
            finally:
                stop_loop.close()
                asyncio.set_event_loop(None)
            for candidate, out in zip(stop_meta, stop_outputs):
                decision = _parse_stop_decision(out)
                candidate["stop_decision"] = decision
                if candidate["stop_cache_key"]:
                    stop_map[candidate["stop_cache_key"]] = decision
                    stop_cache_records.append({
                        "key": candidate["stop_cache_key"],
                        "decision": decision,
                        "prompt_name": stop_prompt_name,
                        "llm": hp.LLM,
                        "context_descs": candidate["context_descs"],
                    })

    init_prompts = []
    init_meta = []
    for candidate in init_candidates:
        sample = candidate["sample"]
        base_record = dict(candidate["base_record"])
        if stop_enabled and str(candidate.get("stop_decision", "continue")) == "stop":
            _mark_sample_stopped(sample, stop_iter=-1, stop_phase="initial_rewrite")
            base_record["rewrite_stop_decision"] = "stop"
            base_record["stop_iter"] = -1
            base_record["stop_phase"] = "initial_rewrite"
            base_record["applied_rewrite"] = ""
            base_record["applied_query"] = sample["current_query"]
            base_record["emr_history_after_write"] = list(sample.get("emr_history", []))
            init_iter_records.append(base_record)
            continue

        emr_state = _build_emr_prompt_memory(
            sample=sample,
            mode=memory_mode,
            query_for_memory=sample["original_query"],
            retrieved_doc_ids=candidate["retrieved_doc_ids"],
            compressor=emr_compressor,
            doc_text_by_id=doc_text_by_id,
            doc_topk=int(getattr(hp, "LEAF_EMR_DOC_TOPK", 10) or 10),
            iter_idx=-1,
            rewrite_template=rewrite_template,
            original_query=sample["original_query"],
            previous_rewrite="",
            context_descs=candidate["context_descs"],
            relevance_definition=relevance_definition,
            prompt_budgeter=rewrite_prompt_budgeter,
            compression_mode=memory_compression,
            max_memory_tokens=int(getattr(hp, "LEAF_EMR_MEMORY_MAX_TOKENS", 0) or 0),
        )
        base_record["emr_memory_compression"] = emr_state["compression_mode"]
        base_record["emr_history_prompt_text"] = emr_state["history_text"]
        base_record["emr_memory_source"] = emr_state["memory_source"]
        base_record["emr_memory_doc_ids"] = list(emr_state["memory_doc_ids"])
        base_record["emr_memory_docs_count"] = int(len(emr_state["memory_doc_ids"]))
        base_record["emr_memory_prompt_text"] = emr_state["memory_text"]
        base_record["emr_memory_overflow_dropped"] = list(emr_state["overflow_dropped"])
        base_record["emr_memory_pool_doc_ids"] = list(emr_state["memory_pool_doc_ids"])
        base_record["emr_memory_selected_sentences"] = list(emr_state["selected_sentences"])
        base_record["emr_memory_prompt_tokens"] = emr_state["memory_prompt_tokens"]
        base_record["emr_prompt_total_tokens"] = emr_state["prompt_total_tokens"]
        base_record["emr_prompt_budget_tokens"] = emr_state["prompt_budget_tokens"]
        base_record["emr_model_context_tokens"] = emr_state["model_context_tokens"]

        cache_key = _rewrite_cache_key(
            "leaf_init",
            sample["current_query"] if leaf_thinkqe_style else sample["original_query"],
            candidate["context_descs"],
            iter_idx=None,
            prompt_name=str(hp.REWRITE_PROMPT_NAME or hp.REWRITE_PROMPT_PATH or ""),
            history_actions=emr_state["history_text"],
            memory_docs=emr_state["memory_text"],
        )
        if (not hp.REWRITE_FORCE_REFRESH) and (cache_key in rewrite_map):
            rewrite = _parse_rewrite_output(rewrite_map[cache_key])
            sample["last_rewrite"] = rewrite
            sample["current_query"] = _apply_leaf_rewrite(sample, hp.REWRITE_MODE, rewrite, thinkqe_style=leaf_thinkqe_style)
            if leaf_thinkqe_style:
                sample["thinkqe_prev_context_doc_ids"] = list(candidate.get("thinkqe_context_doc_ids", []))
                sample["thinkqe_blocklist_doc_ids"] = []
            _append_emr_history_entry(
                sample,
                applied_query=sample["current_query"],
                retrieved_doc_ids=candidate["retrieved_doc_ids"],
                history_rank_topk=int(getattr(hp, "LEAF_EMR_HISTORY_RANK_TOPK", 10) or 10),
            )
            base_record["cache_hit"] = True
            base_record["rewrite_triggered"] = True
            base_record["applied_rewrite"] = rewrite
            base_record["applied_query"] = sample["current_query"]
            base_record["emr_history_after_write"] = list(sample.get("emr_history", []))
            init_iter_records.append(base_record)
            continue

        if rewrite_template is None:
            raise ValueError("Rewrite enabled but no prompt template is available.")
        init_prompts.append(_format_rewrite_prompt(
            rewrite_template,
            sample["original_query"],
            "",
            candidate["context_descs"],
            relevance_definition=relevance_definition,
            history_actions=emr_state["history_text"],
            memory_docs=emr_state["memory_text"],
        ))
        init_meta.append({
            "sample": sample,
            "cache_key": cache_key,
            "context_descs": candidate["context_descs"],
            "base_record": base_record,
            "retrieved_doc_ids": candidate["retrieved_doc_ids"],
            "thinkqe_context_doc_ids": list(candidate.get("thinkqe_context_doc_ids", [])),
        })

    if init_prompts:
        init_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(init_loop)
        try:
            init_outputs = init_loop.run_until_complete(
                llm_api.run_batch(
                    init_prompts,
                    max_concurrent_calls=hp.LLM_MAX_CONCURRENT_CALLS,
                    # Intent: keep leaf-only rewrite generation deterministic so baseline comparisons match round5/6.
                    temperature=0.0,
                )
            )
        finally:
            init_loop.close()
            asyncio.set_event_loop(None)
        for meta, out in zip(init_meta, init_outputs):
            rewrite = _parse_rewrite_output(out)
            rewrite_map[meta["cache_key"]] = rewrite
            meta["sample"]["last_rewrite"] = rewrite
            meta["sample"]["current_query"] = _apply_leaf_rewrite(meta["sample"], hp.REWRITE_MODE, rewrite, thinkqe_style=leaf_thinkqe_style)
            if leaf_thinkqe_style:
                meta["sample"]["thinkqe_prev_context_doc_ids"] = list(meta.get("thinkqe_context_doc_ids", []))
                meta["sample"]["thinkqe_blocklist_doc_ids"] = []
            _append_emr_history_entry(
                meta["sample"],
                applied_query=meta["sample"]["current_query"],
                retrieved_doc_ids=meta["retrieved_doc_ids"],
                history_rank_topk=int(getattr(hp, "LEAF_EMR_HISTORY_RANK_TOPK", 10) or 10),
            )
            generated_record = dict(meta.get("base_record", {}))
            generated_record["cache_hit"] = False
            generated_record["rewrite_triggered"] = True
            generated_record["applied_rewrite"] = rewrite
            generated_record["applied_query"] = meta["sample"]["current_query"]
            generated_record["emr_history_after_write"] = list(meta["sample"].get("emr_history", []))
            init_iter_records.append(generated_record)
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
    if stop_enabled and stop_cache_path and stop_cache_records:
        os.makedirs(os.path.dirname(stop_cache_path) or ".", exist_ok=True)
        with open(stop_cache_path, "a", encoding="utf-8") as f:
            for rec in stop_cache_records:
                f.write(json.dumps(rec, ensure_ascii=True) + "\n")
    if init_iter_records:
        with open(iter_records_path, "a", encoding="utf-8") as f:
            for rec in init_iter_records:
                f.write(json.dumps(rec, ensure_ascii=True) + "\n")

completed_iters = 0
for iter_idx in range(hp.NUM_ITERS):
    active_samples = [sample for sample in samples if bool(sample.get("active", True))]
    if not active_samples:
        logger.info("No active samples remain at iter %d; stopping early.", iter_idx)
        break
    iter_metrics = []
    iter_records = []
    iter_hits = []
    for sample in tqdm(active_samples, desc=f"Iter {iter_idx} retrieval"):
        hits = flat_retrieve_hits(
            retriever=retriever,
            query=sample["current_query"],
            node_embs=node_embs,
            node_registry=node_registry,
            topk=hp.FLAT_TOPK,
        )
        iter_hits.append(hits)
        leaf_hits = [h for h in hits if h.is_leaf]
        pred_path_tuples = [tuple(h.path) for h in leaf_hits[:hp.FLAT_TOPK]]
        pred_doc_ids = _paths_to_ranked_doc_ids(pred_path_tuples, path_to_doc_id)
        pred_doc_ids_eval = filter_excluded_predictions(pred_doc_ids, sample.get("excluded_ids", []))
        rewrite_hits_for_context = [h for h in hits if h.is_leaf] if hp.LEAF_ONLY_RETRIEVAL else hits
        context_hits = _hits_to_context_hits(rewrite_hits_for_context, hp.REWRITE_CONTEXT_TOPK)
        context_path_tuples = [tuple(h.path) for h in context_hits]
        metrics = _compute_leaf_metrics(
            pred_doc_ids,
            sample["gold_doc_ids"],
            excluded_doc_ids=sample.get("excluded_ids", []),
        )
        metrics["iter"] = iter_idx
        metrics["query_idx"] = sample["index"]
        metrics["query"] = sample["original_query"]
        iter_metrics.append(metrics)
        iter_records.append({
            # Intent: keep per-iter retrieval evidence IDs/paths for downstream noise attribution analysis.
            "phase": "iter_retrieval",
            "iter": int(iter_idx),
            "query_idx": int(sample["index"]),
            "query": sample["original_query"],
            "query_for_retrieval": sample["current_query"],
            "last_rewrite": sample["last_rewrite"],
            "retrieval_topk": int(hp.FLAT_TOPK),
            "retrieved_paths": [list(p) for p in pred_path_tuples],
            "retrieved_doc_ids": pred_doc_ids,
            "retrieved_doc_ids_eval": pred_doc_ids_eval,
            "rewrite_context_topk": int(hp.REWRITE_CONTEXT_TOPK),
            "rewrite_context_paths": [list(p) for p in context_path_tuples],
            "rewrite_context_doc_ids": _paths_to_ranked_doc_ids(context_path_tuples, path_to_doc_id),
            "rewrite_triggered": False,
            "rewrite_stop_decision": "",
            "stop_iter": None,
            "stop_phase": "",
            "stop_judge_triggered": bool(stop_enabled),
            "stop_judge_cache_hit": False,
            "thinkqe_style_enabled": bool(leaf_thinkqe_style),
            "thinkqe_context_doc_ids": [],
            "thinkqe_prev_context_doc_ids_before": list(sample.get("thinkqe_prev_context_doc_ids", [])),
            "thinkqe_filtered_doc_ids_this_round": [],
            "thinkqe_blocklist_doc_ids_after": list(sample.get("thinkqe_blocklist_doc_ids", [])),
            "thinkqe_rewrite_skipped_no_new_context": False,
            "emr_memory_mode": memory_mode,
            "emr_memory_compression": memory_compression,
            "emr_history": list(sample.get("emr_history", [])),
            "emr_history_prompt_text": "",
            "emr_memory_source": "off" if memory_mode == "off" else memory_mode,
            "emr_memory_doc_ids": [],
            "emr_memory_docs_count": 0,
            "emr_memory_prompt_text": "",
            "emr_memory_overflow_dropped": [],
            "emr_memory_pool_doc_ids": [],
            "emr_memory_selected_sentences": [],
            "emr_memory_prompt_tokens": None,
            "emr_prompt_total_tokens": None,
            "emr_prompt_budget_tokens": None,
            "emr_model_context_tokens": rewrite_prompt_budgeter.max_context_tokens if rewrite_prompt_budgeter is not None else None,
        })

    if hp.REWRITE_EVERY > 0 and ((iter_idx + 1) % hp.REWRITE_EVERY == 0):
        rewrite_candidates = []
        stop_cache_records = []
        rewrite_records = []
        for row_idx, (sample, hits) in enumerate(zip(active_samples, iter_hits)):
            rewrite_hits = [h for h in hits if h.is_leaf] if hp.LEAF_ONLY_RETRIEVAL else hits
            context_doc_ids: List[str] = []
            filtered_doc_ids: List[str] = []
            blocklist_after = list(sample.get("thinkqe_blocklist_doc_ids", []))
            rewrite_skipped_no_new_context = False
            if leaf_thinkqe_style:
                context_rows = _hits_to_doc_context_rows(
                    rewrite_hits,
                    node_registry,
                    path_to_doc_id,
                    hp.MAX_DOC_DESC_CHAR_LEN,
                    excluded_doc_ids=sample.get("excluded_ids", []),
                )
                selected_rows, context_doc_ids, filtered_doc_ids, blocklist_after = _select_thinkqe_context_rows(
                    sample,
                    context_rows,
                    int(hp.REWRITE_CONTEXT_TOPK),
                )
                context_rows = selected_rows
                sample["thinkqe_blocklist_doc_ids"] = list(blocklist_after)
                sample["thinkqe_prev_context_doc_ids"] = list(context_doc_ids)
                if not context_rows:
                    # Intent: if redundancy filtering leaves no fresh evidence, keep the query unchanged rather than recycling stale context.
                    rewrite_skipped_no_new_context = True
                context_descs = [str(row["desc"]) for row in context_rows]
                context_path_tuples = [tuple(row["path"]) for row in context_rows]
            else:
                context_descs = _hits_to_context_descs(
                    rewrite_hits,
                    node_registry,
                    hp.REWRITE_CONTEXT_TOPK,
                    hp.MAX_DOC_DESC_CHAR_LEN,
                )
                context_hits = _hits_to_context_hits(rewrite_hits, hp.REWRITE_CONTEXT_TOPK)
                context_path_tuples = [tuple(h.path) for h in context_hits]
                context_doc_ids = _paths_to_ranked_doc_ids(context_path_tuples, path_to_doc_id)
            retrieved_doc_ids = iter_records[row_idx]["retrieved_doc_ids"]
            iter_records[row_idx]["rewrite_context_paths"] = [list(p) for p in context_path_tuples]
            iter_records[row_idx]["rewrite_context_doc_ids"] = list(context_doc_ids)
            iter_records[row_idx]["thinkqe_context_doc_ids"] = list(context_doc_ids)
            iter_records[row_idx]["thinkqe_filtered_doc_ids_this_round"] = list(filtered_doc_ids)
            iter_records[row_idx]["thinkqe_blocklist_doc_ids_after"] = list(blocklist_after)
            iter_records[row_idx]["thinkqe_rewrite_skipped_no_new_context"] = bool(rewrite_skipped_no_new_context)
            stop_cache_key = _rewrite_cache_key(
                "leaf_stop_iter",
                sample["current_query"] if leaf_thinkqe_style else f"{sample['original_query']}||{sample['last_rewrite']}",
                context_descs,
                iter_idx=iter_idx,
                prompt_name=stop_prompt_name,
            ) if stop_enabled else ""
            rewrite_candidates.append({
                "sample": sample,
                "context_descs": context_descs,
                "row_idx": row_idx,
                "retrieved_doc_ids": retrieved_doc_ids,
                "thinkqe_context_doc_ids": list(context_doc_ids),
                "stop_cache_key": stop_cache_key,
                "stop_decision": "continue",
                "rewrite_skipped_no_new_context": bool(rewrite_skipped_no_new_context),
            })

        if stop_enabled:
            stop_prompts = []
            stop_meta = []
            for candidate in rewrite_candidates:
                if bool(candidate.get("rewrite_skipped_no_new_context", False)):
                    continue
                stop_cache_key = str(candidate.get("stop_cache_key", ""))
                if (not hp.REWRITE_FORCE_REFRESH) and stop_cache_key and (stop_cache_key in stop_map):
                    candidate["stop_decision"] = _parse_stop_decision(stop_map[stop_cache_key])
                    iter_records[candidate["row_idx"]]["stop_judge_cache_hit"] = True
                    continue
                stop_prompts.append(_build_stop_prompt(
                    stop_template,
                    original_query=candidate["sample"]["original_query"],
                    previous_rewrite=candidate["sample"]["last_rewrite"],
                    context_descs=candidate["context_descs"],
                    relevance_definition=relevance_definition,
                ))
                stop_meta.append(candidate)
            if stop_prompts:
                stop_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(stop_loop)
                try:
                    stop_outputs = stop_loop.run_until_complete(
                        llm_api.run_batch(
                            stop_prompts,
                            max_concurrent_calls=hp.LLM_MAX_CONCURRENT_CALLS,
                            staggering_delay=hp.LLM_API_STAGGERING_DELAY,
                            temperature=0.0,
                        )
                    )
                finally:
                    stop_loop.close()
                    asyncio.set_event_loop(None)
                for candidate, out in zip(stop_meta, stop_outputs):
                    decision = _parse_stop_decision(out)
                    candidate["stop_decision"] = decision
                    if candidate["stop_cache_key"]:
                        stop_map[candidate["stop_cache_key"]] = decision
                        stop_cache_records.append({
                            "key": candidate["stop_cache_key"],
                            "decision": decision,
                            "prompt_name": stop_prompt_name,
                            "llm": hp.LLM,
                            "context_descs": candidate["context_descs"],
                        })

        rewrite_prompts = []
        rewrite_meta = []
        for candidate in rewrite_candidates:
            sample = candidate["sample"]
            row_idx = int(candidate["row_idx"])
            if bool(candidate.get("rewrite_skipped_no_new_context", False)):
                iter_records[row_idx]["applied_rewrite"] = ""
                iter_records[row_idx]["applied_query"] = sample["current_query"]
                iter_records[row_idx]["emr_history_after_write"] = list(sample.get("emr_history", []))
                continue
            if stop_enabled and str(candidate.get("stop_decision", "continue")) == "stop":
                _mark_sample_stopped(sample, stop_iter=int(iter_idx), stop_phase="iter_rewrite")
                iter_records[row_idx]["rewrite_stop_decision"] = "stop"
                iter_records[row_idx]["stop_iter"] = int(iter_idx)
                iter_records[row_idx]["stop_phase"] = "iter_rewrite"
                iter_records[row_idx]["applied_rewrite"] = ""
                iter_records[row_idx]["applied_query"] = sample["current_query"]
                iter_records[row_idx]["emr_history_after_write"] = list(sample.get("emr_history", []))
                continue

            emr_state = _build_emr_prompt_memory(
                sample=sample,
                mode=memory_mode,
                query_for_memory=sample["current_query"],
                retrieved_doc_ids=candidate["retrieved_doc_ids"],
                compressor=emr_compressor,
                doc_text_by_id=doc_text_by_id,
                doc_topk=int(getattr(hp, "LEAF_EMR_DOC_TOPK", 10) or 10),
                iter_idx=int(iter_idx),
                rewrite_template=rewrite_template,
                original_query=sample["original_query"],
                previous_rewrite=sample["last_rewrite"],
                context_descs=candidate["context_descs"],
                relevance_definition=relevance_definition,
                prompt_budgeter=rewrite_prompt_budgeter,
                compression_mode=memory_compression,
                max_memory_tokens=int(getattr(hp, "LEAF_EMR_MEMORY_MAX_TOKENS", 0) or 0),
            )
            iter_records[row_idx]["rewrite_triggered"] = True
            iter_records[row_idx]["emr_memory_compression"] = emr_state["compression_mode"]
            iter_records[row_idx]["emr_history_prompt_text"] = emr_state["history_text"]
            iter_records[row_idx]["emr_memory_source"] = emr_state["memory_source"]
            iter_records[row_idx]["emr_memory_doc_ids"] = list(emr_state["memory_doc_ids"])
            iter_records[row_idx]["emr_memory_docs_count"] = int(len(emr_state["memory_doc_ids"]))
            iter_records[row_idx]["emr_memory_prompt_text"] = emr_state["memory_text"]
            iter_records[row_idx]["emr_memory_overflow_dropped"] = list(emr_state["overflow_dropped"])
            iter_records[row_idx]["emr_memory_pool_doc_ids"] = list(emr_state["memory_pool_doc_ids"])
            iter_records[row_idx]["emr_memory_selected_sentences"] = list(emr_state["selected_sentences"])
            iter_records[row_idx]["emr_memory_prompt_tokens"] = emr_state["memory_prompt_tokens"]
            iter_records[row_idx]["emr_prompt_total_tokens"] = emr_state["prompt_total_tokens"]
            iter_records[row_idx]["emr_prompt_budget_tokens"] = emr_state["prompt_budget_tokens"]
            iter_records[row_idx]["emr_model_context_tokens"] = emr_state["model_context_tokens"]

            cache_key = _rewrite_cache_key(
                "leaf_iter",
                sample["current_query"] if leaf_thinkqe_style else f"{sample['original_query']}||{sample['last_rewrite']}",
                candidate["context_descs"],
                iter_idx=iter_idx,
                prompt_name=str(hp.REWRITE_PROMPT_NAME or hp.REWRITE_PROMPT_PATH or ""),
                history_actions=emr_state["history_text"],
                memory_docs=emr_state["memory_text"],
            )
            if (not hp.REWRITE_FORCE_REFRESH) and (cache_key in rewrite_map):
                rewrite = _parse_rewrite_output(rewrite_map[cache_key])
                sample["last_rewrite"] = rewrite
                sample["current_query"] = _apply_leaf_rewrite(sample, hp.REWRITE_MODE, rewrite, thinkqe_style=leaf_thinkqe_style)
                _append_emr_history_entry(
                    sample,
                    applied_query=sample["current_query"],
                    retrieved_doc_ids=candidate["retrieved_doc_ids"],
                    history_rank_topk=int(getattr(hp, "LEAF_EMR_HISTORY_RANK_TOPK", 10) or 10),
                )
                iter_records[row_idx]["cache_hit"] = True
                iter_records[row_idx]["rewrite_triggered"] = True
                iter_records[row_idx]["applied_rewrite"] = rewrite
                iter_records[row_idx]["applied_query"] = sample["current_query"]
                iter_records[row_idx]["emr_history_after_write"] = list(sample.get("emr_history", []))
                continue

            if rewrite_template is None:
                raise ValueError("Rewrite enabled but no prompt template is available.")
            rewrite_prompts.append(_format_rewrite_prompt(
                rewrite_template,
                sample["original_query"],
                sample["last_rewrite"],
                candidate["context_descs"],
                relevance_definition=relevance_definition,
                history_actions=emr_state["history_text"],
                memory_docs=emr_state["memory_text"],
            ))
            rewrite_meta.append({
                "sample": sample,
                "cache_key": cache_key,
                "context_descs": candidate["context_descs"],
                "row_idx": row_idx,
                "retrieved_doc_ids": candidate["retrieved_doc_ids"],
                "thinkqe_context_doc_ids": list(candidate.get("thinkqe_context_doc_ids", [])),
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
                        # Intent: keep iterative rewrite generation deterministic so leaf-only runs are reproducible.
                        temperature=0.0,
                    )
                )
            finally:
                rewrite_loop.close()
                asyncio.set_event_loop(None)
            for meta, out in zip(rewrite_meta, rewrite_outputs):
                rewrite = _parse_rewrite_output(out)
                rewrite_map[meta["cache_key"]] = rewrite
                meta["sample"]["last_rewrite"] = rewrite
                meta["sample"]["current_query"] = _apply_leaf_rewrite(meta["sample"], hp.REWRITE_MODE, rewrite, thinkqe_style=leaf_thinkqe_style)
                _append_emr_history_entry(
                    meta["sample"],
                    applied_query=meta["sample"]["current_query"],
                    retrieved_doc_ids=meta["retrieved_doc_ids"],
                    history_rank_topk=int(getattr(hp, "LEAF_EMR_HISTORY_RANK_TOPK", 10) or 10),
                )
                iter_records[meta["row_idx"]]["cache_hit"] = False
                iter_records[meta["row_idx"]]["rewrite_triggered"] = True
                iter_records[meta["row_idx"]]["applied_rewrite"] = rewrite
                iter_records[meta["row_idx"]]["applied_query"] = meta["sample"]["current_query"]
                iter_records[meta["row_idx"]]["emr_history_after_write"] = list(meta["sample"].get("emr_history", []))
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
        if stop_enabled and stop_cache_path and stop_cache_records:
            os.makedirs(os.path.dirname(stop_cache_path) or ".", exist_ok=True)
            with open(stop_cache_path, "a", encoding="utf-8") as f:
                for rec in stop_cache_records:
                    f.write(json.dumps(rec, ensure_ascii=True) + "\n")

    with open(metrics_path, "a", encoding="utf-8") as f:
        for metrics in iter_metrics:
            f.write(json.dumps(metrics, ensure_ascii=True) + "\n")
    with open(iter_records_path, "a", encoding="utf-8") as f:
        for rec in iter_records:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")

    mean_ndcg = float(np.mean([m["nDCG@10"] for m in iter_metrics])) if iter_metrics else 0.0
    mean_r10 = float(np.mean([m["Recall@10"] for m in iter_metrics])) if iter_metrics else 0.0
    mean_r100 = float(np.mean([m["Recall@100"] for m in iter_metrics])) if iter_metrics else 0.0
    active_count = int(sum(1 for sample in samples if bool(sample.get("active", True))))
    stopped_count = int(sum(1 for sample in samples if not bool(sample.get("active", True))))
    logger.info(
        f"Iter {iter_idx} mean metrics: nDCG@10={mean_ndcg:.2f}, "
        f"Recall@10={mean_r10:.2f}, Recall@100={mean_r100:.2f}, ActiveSamples={active_count}, StoppedSamples={stopped_count}"
    )
    completed_iters = int(iter_idx + 1)

logger.info("Saved metrics to %s", metrics_path)
logger.info("Saved retrieval records to %s", iter_records_path)
with open(done_marker_path, "w", encoding="utf-8") as f:
    json.dump(
        {"status": "completed", "num_iters_requested": int(hp.NUM_ITERS), "num_iters_completed": int(completed_iters)},
        f,
        ensure_ascii=True,
        indent=2,
    )
logger.info("Wrote completion marker to %s", done_marker_path)
logger.info("Completed run_leaf_rank.py.")
