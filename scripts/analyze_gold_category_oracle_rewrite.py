#!/usr/bin/env python3
import argparse
import asyncio
import json
import logging
import os
import pickle as pkl
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from json_repair import repair_json


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from cache_utils import _prompt_cache_key, append_jsonl  # noqa: E402
from llm_apis import GenAIAPI, VllmAPI  # noqa: E402
from retrievers.diver import DiverEmbeddingModel  # noqa: E402
from tree_objects import SemanticNode  # noqa: E402


CATEGORY_CLASSIFIER_PROMPT = (
    "You are an oracle category annotator for retrieval analysis.\n\n"
    "Goal:\n"
    "- Read the original query and gold documents (known relevant).\n"
    "- Assign retrieval-role categories using ONLY allowed labels.\n"
    "- Output a union category set that will be used as rewrite constraints.\n\n"
    "Rules:\n"
    "- Focus on retrieval role type, not topic names.\n"
    "- Multi-label is allowed.\n"
    "- If uncertain, keep categories broad but still from the allowed labels.\n"
    "- Do not invent labels outside the allowed label pool.\n"
    "- Use at most {max_categories} selected categories.\n\n"
    "Allowed category labels:\n"
    "{category_guide}\n\n"
    "Original Query:\n{original_query}\n\n"
    "Gold Documents (relevant references):\n{gold_docs}\n\n"
    "Output JSON only:\n"
    "{{\n"
    "  \"Doc_Categories\": {{\"doc_id_1\": [\"label1\", \"label2\"]}},\n"
    "  \"Selected_Categories\": [\"label1\", \"label2\"]\n"
    "}}\n"
)

ORACLE_REWRITE_PROMPT = (
    "You are rewriting a search query for reasoning-intensive retrieval.\n\n"
    "Goal:\n"
    "- You must use only the selected category labels as routing constraints.\n"
    "- Produce concrete retrieval hypotheses per selected category.\n\n"
    "Rules:\n"
    "- Use ONLY keys from Selected Categories.\n"
    "- Each value must be concrete, retrieval-friendly text (not category echo).\n"
    "- Keep each value short (one sentence).\n"
    "- Do not include labels not in Selected Categories.\n\n"
    "Original Query:\n{original_query}\n\n"
    "Selected Categories:\n{selected_categories}\n\n"
    "Output JSON only:\n"
    "{{\n"
    "  \"Possible_Answer_Docs\": {{\n"
    "    \"CategoryA\": \"concrete retrieval text\",\n"
    "    \"CategoryB\": \"concrete retrieval text\"\n"
    "  }}\n"
    "}}\n"
)

CONTROL_REWRITE_PROMPT = (
    "You are rewriting a search query for reasoning-intensive retrieval.\n\n"
    "Goal:\n"
    "- Propose concrete retrieval hypotheses without category constraints.\n\n"
    "Rules:\n"
    "- Produce 3 to 5 short retrieval-friendly lines.\n"
    "- Focus on evidence that would justify a correct answer.\n\n"
    "Original Query:\n{original_query}\n\n"
    "Output JSON only:\n"
    "{{\n"
    "  \"Possible_Answer_Docs\": [\"...\", \"...\", \"...\"]\n"
    "}}\n"
)


@dataclass
class SampleItem:
    sample_idx: int
    query: str
    gold_ids: List[str]
    gold_paths: List[Tuple[int, ...]]


def _setup_logger(out_dir: str) -> logging.Logger:
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "run.log")
    logger = logging.getLogger("gold_category_oracle_analysis")
    logger.setLevel(logging.INFO)
    while logger.handlers:
        handler = logger.handlers.pop()
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.StreamHandler(open(log_path, "a", encoding="utf-8"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _normalize_embeddings(embs: np.ndarray) -> np.ndarray:
    if embs.size == 0:
        return embs.astype(np.float32, copy=False)
    embs = embs.astype(np.float32, copy=False)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return embs / norms


def _compute_ndcg(sorted_preds: List[List[int]], gold: List[List[int]], k: int = 10) -> float:
    if not sorted_preds or not gold:
        return 0.0
    sorted_preds = sorted_preds[:k]
    dcg = 0.0
    for i, pred in enumerate(sorted_preds):
        if pred in gold:
            dcg += 1.0 / np.log2(i + 2)
    ideal_k = min(k, len(gold))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_k))
    if idcg <= 0.0:
        return 0.0
    return float(dcg / idcg)


def _compute_recall(sorted_preds: List[List[int]], gold: List[List[int]], k: int = 10) -> float:
    if not sorted_preds or not gold:
        return 0.0
    topk = sorted_preds[:k]
    hits = sum(1 for pred in topk if pred in gold)
    return float(hits / max(1, len(gold)))


def _get_node_id(node_id: object, docs_df: pd.DataFrame) -> Optional[str]:
    try:
        if isinstance(node_id, str):
            if node_id.startswith("["):
                return str(node_id.split(" ", 1)[1])
            return str(node_id)
        if isinstance(node_id, (int, np.integer)):
            return str(docs_df.id.iloc[int(node_id)])
    except Exception:
        return None
    return None


def _get_all_leaf_nodes_with_path(node: object, path: Optional[List[int]] = None) -> List[Tuple[object, List[int]]]:
    if path is None:
        path = []
    if not getattr(node, "child", None):
        return [(node, path)]
    leaves: List[Tuple[object, List[int]]] = []
    for idx, child in enumerate(node.child):
        leaves.extend(_get_all_leaf_nodes_with_path(child, path + [idx]))
    return leaves


def _compute_node_registry(semantic_root_node: object) -> List[object]:
    def _set_path(node: object, path: Tuple[int, ...] = ()) -> None:
        node.path = path
        for idx, child in enumerate(node.child):
            _set_path(child, (*path, idx))

    def _set_num_leaves(node: object) -> int:
        if (not node.child) or (len(node.child) == 0):
            node.num_leaves = 1
        else:
            node.num_leaves = sum(_set_num_leaves(child) for child in node.child)
        return node.num_leaves

    def _collect(node: object, bag: List[object]) -> None:
        bag.append(node)
        for child in node.child:
            _collect(child, bag)

    _set_path(semantic_root_node)
    _set_num_leaves(semantic_root_node)
    bag: List[object] = []
    _collect(semantic_root_node, bag)
    for idx, node in enumerate(bag):
        node.registry_idx = idx
    return bag


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or scores.size == 0:
        return np.array([], dtype=np.int64)
    k = min(k, scores.shape[0])
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx.astype(np.int64, copy=False)


def _read_cache(path: str) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    if not path or (not os.path.exists(path)):
        return cache
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
            cache[key] = rec
    return cache


def _parse_json_obj(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    raw = raw.split("</think>\n")[-1].strip()
    if "```" in raw:
        try:
            parts = raw.split("```")
            fenced = [parts[i] for i in range(1, len(parts), 2)]
            if fenced:
                raw = fenced[-1].strip()
        except Exception:
            pass
    if raw.startswith("json"):
        raw = raw[4:].strip()
    obj: Any = {}
    try:
        obj = json.loads(raw)
    except Exception:
        try:
            obj = repair_json(raw, return_objects=True)
        except Exception:
            obj = {}
    return obj if isinstance(obj, dict) else {}


def _to_clean_list(value: object, max_items: int) -> List[str]:
    if max_items <= 0:
        return []
    items: List[str] = []
    if isinstance(value, list):
        items = [str(x).strip() for x in value if str(x).strip()]
    elif isinstance(value, str):
        items = [str(x).strip() for x in re.split(r"[,\n]", value) if str(x).strip()]
    dedup: List[str] = []
    seen: set[str] = set()
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(item)
        if len(dedup) >= max_items:
            break
    return dedup


def _parse_doc_categories(
    obj: Dict[str, Any],
    allowed_labels: Sequence[str],
    max_categories: int,
) -> Tuple[Dict[str, List[str]], List[str]]:
    allowed_map = {str(label).strip().lower(): str(label).strip() for label in allowed_labels if str(label).strip()}

    def _normalize_labels(raw_labels: object) -> List[str]:
        labels = _to_clean_list(raw_labels, max_categories)
        out: List[str] = []
        seen: set[str] = set()
        for label in labels:
            key = str(label).strip().lower()
            if key not in allowed_map:
                continue
            canon = allowed_map[key]
            canon_key = canon.lower()
            if canon_key in seen:
                continue
            seen.add(canon_key)
            out.append(canon)
            if len(out) >= max_categories:
                break
        return out

    doc_categories: Dict[str, List[str]] = {}
    raw_doc_map = obj.get("Doc_Categories") or obj.get("doc_categories") or {}
    if isinstance(raw_doc_map, dict):
        for doc_id, labels in raw_doc_map.items():
            doc_key = str(doc_id or "").strip()
            if not doc_key:
                continue
            normalized = _normalize_labels(labels)
            if normalized:
                doc_categories[doc_key] = normalized

    selected = _normalize_labels(obj.get("Selected_Categories") or obj.get("selected_categories") or [])
    if not selected and doc_categories:
        union: List[str] = []
        seen_union: set[str] = set()
        for labels in doc_categories.values():
            for label in labels:
                key = label.lower()
                if key in seen_union:
                    continue
                seen_union.add(key)
                union.append(label)
                if len(union) >= max_categories:
                    break
            if len(union) >= max_categories:
                break
        selected = union
    return doc_categories, selected


def _parse_rewrite_docs(
    obj: Dict[str, Any],
    selected_categories: Sequence[str],
    max_docs: int,
) -> Dict[str, str]:
    selected_set = {str(x).strip().lower() for x in selected_categories if str(x).strip()}
    docs = obj.get("Possible_Answer_Docs") or obj.get("possible_answer_docs") or {}
    out: Dict[str, str] = {}
    if isinstance(docs, dict):
        for key, val in docs.items():
            k = str(key or "").strip()
            if not k:
                continue
            if selected_set and (k.lower() not in selected_set):
                continue
            v = str(val or "").strip()
            if not v:
                continue
            out[k] = v
            if len(out) >= max_docs:
                break
        return out
    if isinstance(docs, list):
        for idx, val in enumerate(docs):
            v = str(val or "").strip()
            if not v:
                continue
            out[f"Doc{idx+1}"] = v
            if len(out) >= max_docs:
                break
    return out


def _compose_query(original_query: str, docs: Dict[str, str]) -> str:
    pieces = [str(v).strip() for v in docs.values() if str(v).strip()]
    if not pieces:
        return str(original_query or "").strip()
    return ((original_query or "").strip() + " " + "\n".join(pieces)).strip()


def _load_dataset_frames(base_dir: str, dataset: str, subset: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    docs_path = os.path.join(base_dir, "data", dataset, subset, "documents.jsonl")
    examples_path = os.path.join(base_dir, "data", dataset, subset, "examples.jsonl")
    if os.path.exists(docs_path) and os.path.exists(examples_path):
        docs_df = pd.read_json(docs_path, lines=True, dtype={"id": str})
        examples_df = pd.read_json(examples_path, lines=True, dtype={"gold_ids": List[str]})
        examples_df["gold_ids"] = examples_df["gold_ids"].apply(lambda x: [str(i) for i in x])
        return docs_df, examples_df
    docs_df = pd.DataFrame(load_dataset("xlangai/BRIGHT", "documents", split=subset))
    examples_df = pd.DataFrame(load_dataset("xlangai/BRIGHT", "examples", split=subset))
    examples_df["gold_ids"] = examples_df["gold_ids"].apply(lambda x: [str(i) for i in x])
    return docs_df, examples_df


def _build_category_guide(
    registry_path: str,
    category_level: str,
    max_items: int,
) -> List[str]:
    if not os.path.exists(registry_path):
        raise FileNotFoundError(f"Category registry not found: {registry_path}")
    with open(registry_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    categories = payload.get("categories", [])
    rows_level1: List[Tuple[str, int]] = []
    rows_level2: List[Tuple[str, int]] = []
    for row in categories:
        if not isinstance(row, dict):
            continue
        try:
            count = int(row.get("count", 0))
        except Exception:
            count = 0
        level1 = str(row.get("level1", "")).strip()
        level2 = str(row.get("level2", row.get("label", ""))).strip()
        if level1:
            rows_level1.append((level1, count))
        if level2:
            rows_level2.append((level2, count))

    merged_level1: Dict[str, int] = {}
    merged_level2: Dict[str, int] = {}
    for label, count in rows_level1:
        merged_level1[label] = merged_level1.get(label, 0) + int(count)
    for label, count in rows_level2:
        merged_level2[label] = merged_level2.get(label, 0) + int(count)

    ordered_level1 = [label for label, _ in sorted(merged_level1.items(), key=lambda x: (-x[1], x[0]))]
    ordered_level2 = [label for label, _ in sorted(merged_level2.items(), key=lambda x: (-x[1], x[0]))]

    if category_level == "level1":
        labels = ordered_level1[:max_items]
    else:
        # Intent: level2 mode uses both level1 and level2 labels so oracle routing can select broad+fine keys together.
        labels = []
        seen: set[str] = set()
        for label in ordered_level1 + ordered_level2:
            key = label.lower()
            if key in seen:
                continue
            seen.add(key)
            labels.append(label)
            if len(labels) >= max_items:
                break
    if not labels:
        raise ValueError(f"No labels found in registry: {registry_path}")
    return labels


def _format_gold_docs_block(
    gold_ids: Sequence[str],
    doc_id_to_content: Dict[str, str],
    max_gold_docs: int,
    max_doc_chars: int,
) -> str:
    def _tail_title_from_doc_id(doc_id: str) -> str:
        # Intent: use only the suffix after "/" so split-document IDs stay interpretable without fragile "_" parsing.
        return str(doc_id or "").split("/")[-1].strip()

    lines: List[str] = []
    shown = 0
    for doc_id in gold_ids:
        if shown >= max_gold_docs:
            break
        content = str(doc_id_to_content.get(str(doc_id), "")).strip()
        if not content:
            continue
        snippet = content[:max_doc_chars]
        title = _tail_title_from_doc_id(str(doc_id))
        lines.append(f"Title: {title}\n{snippet}")
        shown += 1
    return "\n\n".join(lines).strip()


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.array(values, dtype=np.float32)))


def _build_llm_api(args: argparse.Namespace, logger: logging.Logger):
    if args.llm_api_backend == "genai":
        return GenAIAPI(
            args.llm,
            logger=logger,
            timeout=args.llm_api_timeout,
            max_retries=args.llm_api_max_retries,
        )
    if args.llm_api_backend == "vllm":
        return VllmAPI(
            args.llm,
            logger=logger,
            timeout=args.llm_api_timeout,
            max_retries=args.llm_api_max_retries,
            base_url=args.vllm_base_url,
        )
    raise ValueError(f"Unknown --llm_api_backend: {args.llm_api_backend}")


def _run_batch(llm_api, prompts: List[str], args: argparse.Namespace) -> List[str]:
    if not prompts:
        return []
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            llm_api.run_batch(
                prompts,
                max_concurrent_calls=args.llm_max_concurrent_calls,
                staggering_delay=args.llm_api_staggering_delay,
                temperature=0.0,
            )
        )
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gold-document category oracle rewrite analysis.")
    parser.add_argument("--dataset", type=str, default="BRIGHT")
    parser.add_argument("--subset", type=str, required=True)
    parser.add_argument("--tree_version", type=str, required=True)
    parser.add_argument("--retriever_model_path", type=str, required=True)
    parser.add_argument("--node_emb_path", type=str, required=True)
    parser.add_argument("--llm_api_backend", type=str, default="vllm", choices=["vllm", "genai"])
    parser.add_argument("--llm", type=str, required=True)
    parser.add_argument("--vllm_base_url", type=str, default="http://localhost:8000/v1,http://localhost:8001/v1,http://localhost:8002/v1,http://localhost:8003/v1")
    parser.add_argument("--llm_max_concurrent_calls", type=int, default=20)
    parser.add_argument("--llm_api_timeout", type=int, default=60)
    parser.add_argument("--llm_api_max_retries", type=int, default=3)
    parser.add_argument("--llm_api_staggering_delay", type=float, default=0.02)
    parser.add_argument("--num_eval_samples", type=int, default=300)
    parser.add_argument("--anchor_topk", type=int, default=1000)
    parser.add_argument("--global_topk", type=int, default=1000)
    parser.add_argument("--metric_k", type=int, default=10)
    parser.add_argument("--max_query_char_len", type=int, default=1024)
    parser.add_argument("--max_gold_docs_in_prompt", type=int, default=3)
    parser.add_argument("--max_doc_chars", type=int, default=1800)
    parser.add_argument(
        "--category_level",
        type=str,
        default="level1",
        choices=["level1", "level2"],
        help="level1=use only level1 labels; level2=use merged level1+level2 labels",
    )
    parser.add_argument("--max_category_guide_items", type=int, default=80)
    parser.add_argument("--max_selected_categories", type=int, default=5)
    parser.add_argument("--max_rewrite_docs", type=int, default=5)
    parser.add_argument("--skip_control_rewrite", default=False, action="store_true")
    parser.add_argument("--cache_path", type=str, default="")
    parser.add_argument("--force_refresh", default=False, action="store_true")
    parser.add_argument("--out_dir", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.out_dir:
        suffix_blob = (
            f"S={args.subset}"
            f"-TV={args.tree_version}"
            f"-CL={args.category_level}"
            f"-NumES={args.num_eval_samples}"
            f"-Llm={os.path.basename(args.llm)}"
        )
        args.out_dir = os.path.join(BASE_DIR, "results", "analysis", "gold_category_oracle", suffix_blob)
    os.makedirs(args.out_dir, exist_ok=True)
    logger = _setup_logger(args.out_dir)
    logger.info("Starting gold-category oracle analysis")

    if not args.cache_path:
        # Intent: keep oracle analysis cache separated by subset and category granularity.
        cache_file = f"{args.subset}_gold_category_oracle_{args.category_level}.jsonl"
        args.cache_path = os.path.join(BASE_DIR, "cache", "analysis", cache_file)
    os.makedirs(os.path.dirname(args.cache_path), exist_ok=True)

    registry_path = os.path.join(
        BASE_DIR,
        "trees",
        args.dataset,
        args.subset,
        "category_registry_category_assign_v2.json",
    )
    allowed_labels = _build_category_guide(
        registry_path=registry_path,
        category_level=args.category_level,
        max_items=args.max_category_guide_items,
    )
    category_guide_blob = "\n".join([f"- {label}" for label in allowed_labels])
    logger.info("Loaded %d allowed category labels (%s)", len(allowed_labels), args.category_level)

    docs_df, examples_df = _load_dataset_frames(BASE_DIR, args.dataset, args.subset)
    tree_path = os.path.join(BASE_DIR, "trees", args.dataset, args.subset, f"tree-{args.tree_version}.pkl")
    tree_dict = pkl.load(open(tree_path, "rb"))
    semantic_root_node = SemanticNode().load_dict(tree_dict) if isinstance(tree_dict, dict) else tree_dict
    node_registry = _compute_node_registry(semantic_root_node)
    all_leaf_nodes = _get_all_leaf_nodes_with_path(semantic_root_node)
    doc_id_to_path: Dict[str, Tuple[int, ...]] = {}
    for leaf, path in all_leaf_nodes:
        doc_id = _get_node_id(leaf.id, docs_df)
        if doc_id:
            doc_id_to_path[str(doc_id)] = tuple(path)
    doc_id_to_content = {str(docs_df.iloc[i].id): str(docs_df.iloc[i].content) for i in range(len(docs_df))}

    node_embs = np.load(args.node_emb_path, allow_pickle=False)
    if node_embs.shape[0] != len(node_registry):
        raise ValueError(
            f"node_emb rows ({node_embs.shape[0]}) must match node_registry size ({len(node_registry)})"
        )
    node_embs = _normalize_embeddings(node_embs)
    leaf_indices = [idx for idx, node in enumerate(node_registry) if bool(getattr(node, "is_leaf", False))]

    retriever = DiverEmbeddingModel(args.retriever_model_path, local_files_only=True)
    llm_api = _build_llm_api(args, logger)

    num_samples = min(len(examples_df), int(args.num_eval_samples))
    samples: List[SampleItem] = []
    for i in range(num_samples):
        query = str(examples_df.iloc[i]["query"])[: args.max_query_char_len]
        gold_ids = [str(x) for x in list(examples_df.iloc[i]["gold_ids"])]
        gold_paths = [doc_id_to_path[g] for g in gold_ids if g in doc_id_to_path]
        if not gold_paths:
            continue
        samples.append(
            SampleItem(
                sample_idx=i,
                query=query,
                gold_ids=gold_ids,
                gold_paths=[tuple(p) for p in gold_paths],
            )
        )
    logger.info("Prepared %d samples (from requested %d)", len(samples), num_samples)
    if not samples:
        raise ValueError("No evaluable samples found after gold_path mapping.")

    cache = _read_cache(args.cache_path)
    cache_updates: List[Dict[str, Any]] = []

    # Phase 1: gold-doc category oracle classification
    classify_meta: List[Tuple[int, str, str]] = []
    classify_prompts: List[str] = []
    doc_categories_by_sample: List[Dict[str, List[str]]] = [{} for _ in samples]
    selected_categories_by_sample: List[List[str]] = [[] for _ in samples]
    for idx, sample in enumerate(samples):
        gold_docs_blob = _format_gold_docs_block(
            sample.gold_ids,
            doc_id_to_content,
            max_gold_docs=args.max_gold_docs_in_prompt,
            max_doc_chars=args.max_doc_chars,
        )
        prompt = CATEGORY_CLASSIFIER_PROMPT.format(
            max_categories=args.max_selected_categories,
            category_guide=category_guide_blob,
            original_query=sample.query,
            gold_docs=gold_docs_blob,
        )
        cache_key = _prompt_cache_key("gold_category_oracle_classifier", prompt)
        if (not args.force_refresh) and (cache_key in cache):
            obj = cache.get(cache_key, {}).get("parsed_obj", {})
            if not isinstance(obj, dict):
                obj = {}
            doc_map, selected = _parse_doc_categories(obj, allowed_labels, args.max_selected_categories)
            doc_categories_by_sample[idx] = doc_map
            selected_categories_by_sample[idx] = selected
            continue
        classify_meta.append((idx, cache_key, prompt))
        classify_prompts.append(prompt)

    if classify_prompts:
        logger.info("LLM classify phase: %d prompts", len(classify_prompts))
        classify_outputs = _run_batch(llm_api, classify_prompts, args)
        for (idx, cache_key, prompt), output in zip(classify_meta, classify_outputs):
            obj = _parse_json_obj(output)
            doc_map, selected = _parse_doc_categories(obj, allowed_labels, args.max_selected_categories)
            doc_categories_by_sample[idx] = doc_map
            selected_categories_by_sample[idx] = selected
            cache_updates.append({
                "key": cache_key,
                "phase": "classify",
                "prompt": prompt,
                "response": output,
                "parsed_obj": obj,
            })

    # Phase 2: category-only rewrite
    oracle_meta: List[Tuple[int, str, str]] = []
    oracle_prompts: List[str] = []
    oracle_docs_by_sample: List[Dict[str, str]] = [{} for _ in samples]
    oracle_query_by_sample: List[str] = [sample.query for sample in samples]
    for idx, sample in enumerate(samples):
        selected = selected_categories_by_sample[idx]
        if not selected:
            continue
        selected_blob = "\n".join([f"- {label}" for label in selected])
        prompt = ORACLE_REWRITE_PROMPT.format(
            original_query=sample.query,
            selected_categories=selected_blob,
        )
        cache_key = _prompt_cache_key("gold_category_oracle_rewrite", prompt)
        if (not args.force_refresh) and (cache_key in cache):
            obj = cache.get(cache_key, {}).get("parsed_obj", {})
            if not isinstance(obj, dict):
                obj = {}
            docs_map = _parse_rewrite_docs(obj, selected, args.max_rewrite_docs)
            oracle_docs_by_sample[idx] = docs_map
            oracle_query_by_sample[idx] = _compose_query(sample.query, docs_map)
            continue
        oracle_meta.append((idx, cache_key, prompt))
        oracle_prompts.append(prompt)

    if oracle_prompts:
        logger.info("LLM oracle rewrite phase: %d prompts", len(oracle_prompts))
        oracle_outputs = _run_batch(llm_api, oracle_prompts, args)
        for (idx, cache_key, prompt), output in zip(oracle_meta, oracle_outputs):
            selected = selected_categories_by_sample[idx]
            obj = _parse_json_obj(output)
            docs_map = _parse_rewrite_docs(obj, selected, args.max_rewrite_docs)
            oracle_docs_by_sample[idx] = docs_map
            oracle_query_by_sample[idx] = _compose_query(samples[idx].query, docs_map)
            cache_updates.append({
                "key": cache_key,
                "phase": "oracle_rewrite",
                "prompt": prompt,
                "response": output,
                "parsed_obj": obj,
            })

    # Phase 3: control rewrite (without categories)
    control_docs_by_sample: List[Dict[str, str]] = [{} for _ in samples]
    control_query_by_sample: List[str] = [sample.query for sample in samples]
    control_outputs_enabled = not bool(args.skip_control_rewrite)
    if control_outputs_enabled:
        control_meta: List[Tuple[int, str, str]] = []
        control_prompts: List[str] = []
        for idx, sample in enumerate(samples):
            prompt = CONTROL_REWRITE_PROMPT.format(original_query=sample.query)
            cache_key = _prompt_cache_key("gold_category_oracle_control_rewrite", prompt)
            if (not args.force_refresh) and (cache_key in cache):
                obj = cache.get(cache_key, {}).get("parsed_obj", {})
                if not isinstance(obj, dict):
                    obj = {}
                docs_map = _parse_rewrite_docs(obj, [], args.max_rewrite_docs)
                control_docs_by_sample[idx] = docs_map
                control_query_by_sample[idx] = _compose_query(sample.query, docs_map)
                continue
            control_meta.append((idx, cache_key, prompt))
            control_prompts.append(prompt)

        if control_prompts:
            logger.info("LLM control rewrite phase: %d prompts", len(control_prompts))
            control_outputs = _run_batch(llm_api, control_prompts, args)
            for (idx, cache_key, prompt), output in zip(control_meta, control_outputs):
                obj = _parse_json_obj(output)
                docs_map = _parse_rewrite_docs(obj, [], args.max_rewrite_docs)
                control_docs_by_sample[idx] = docs_map
                control_query_by_sample[idx] = _compose_query(samples[idx].query, docs_map)
                cache_updates.append({
                    "key": cache_key,
                    "phase": "control_rewrite",
                    "prompt": prompt,
                    "response": output,
                    "parsed_obj": obj,
                })

    if cache_updates:
        append_jsonl(args.cache_path, cache_updates)
        logger.info("Cache updated: %d records -> %s", len(cache_updates), args.cache_path)

    # Retrieval/evaluation
    per_sample_rows: List[Dict[str, Any]] = []
    orig_anchor_scores: List[float] = []
    oracle_anchor_scores: List[float] = []
    control_anchor_scores: List[float] = []
    orig_global_scores: List[float] = []
    oracle_global_scores: List[float] = []
    control_global_scores: List[float] = []

    for idx, sample in enumerate(samples):
        def _eval_query(query: str) -> Dict[str, float]:
            q_emb = retriever.encode_query(query)
            scores_all = (node_embs @ q_emb).astype(np.float32, copy=False)
            anchor_idx = _topk_indices(scores_all, args.anchor_topk)
            anchor_paths = [list(node_registry[i].path) for i in anchor_idx.tolist() if bool(node_registry[i].is_leaf)]
            leaf_scores = scores_all[leaf_indices]
            global_rel_idx = _topk_indices(leaf_scores, args.global_topk)
            global_paths = [list(node_registry[leaf_indices[int(i)]].path) for i in global_rel_idx.tolist()]
            gold_paths = [list(p) for p in sample.gold_paths]
            return {
                "anchor_ndcg@10": _compute_ndcg(anchor_paths[: args.metric_k], gold_paths, k=args.metric_k) * 100.0,
                "anchor_recall@10": _compute_recall(anchor_paths[: args.metric_k], gold_paths, k=args.metric_k) * 100.0,
                "anchor_recall@100": _compute_recall(anchor_paths[:100], gold_paths, k=100) * 100.0,
                "global_ndcg@10": _compute_ndcg(global_paths[: args.metric_k], gold_paths, k=args.metric_k) * 100.0,
                "global_recall@10": _compute_recall(global_paths[: args.metric_k], gold_paths, k=args.metric_k) * 100.0,
                "global_recall@100": _compute_recall(global_paths[:100], gold_paths, k=100) * 100.0,
            }

        m_orig = _eval_query(sample.query)
        m_oracle = _eval_query(oracle_query_by_sample[idx])
        m_control = _eval_query(control_query_by_sample[idx]) if control_outputs_enabled else {}

        orig_anchor_scores.append(float(m_orig["anchor_ndcg@10"]))
        oracle_anchor_scores.append(float(m_oracle["anchor_ndcg@10"]))
        orig_global_scores.append(float(m_orig["global_ndcg@10"]))
        oracle_global_scores.append(float(m_oracle["global_ndcg@10"]))
        if control_outputs_enabled:
            control_anchor_scores.append(float(m_control["anchor_ndcg@10"]))
            control_global_scores.append(float(m_control["global_ndcg@10"]))

        per_sample_rows.append({
            "sample_idx": sample.sample_idx,
            "query": sample.query,
            "gold_ids": sample.gold_ids,
            "selected_categories": selected_categories_by_sample[idx],
            "doc_categories": doc_categories_by_sample[idx],
            "oracle_possible_docs": oracle_docs_by_sample[idx],
            "oracle_query": oracle_query_by_sample[idx],
            "control_possible_docs": control_docs_by_sample[idx] if control_outputs_enabled else {},
            "control_query": control_query_by_sample[idx] if control_outputs_enabled else sample.query,
            "metrics_original": m_orig,
            "metrics_oracle": m_oracle,
            "metrics_control": m_control,
        })

    def _win_tie_loss(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
        wins = sum(1 for x, y in zip(a, b) if y > x)
        ties = sum(1 for x, y in zip(a, b) if y == x)
        losses = sum(1 for x, y in zip(a, b) if y < x)
        total = max(1, len(a))
        return {
            "wins": int(wins),
            "ties": int(ties),
            "losses": int(losses),
            "win_rate": float(wins / total),
            "tie_rate": float(ties / total),
            "loss_rate": float(losses / total),
        }

    summary: Dict[str, Any] = {
        "config": vars(args),
        "num_samples": len(samples),
        "category_level": args.category_level,
        "num_allowed_labels": len(allowed_labels),
        "anchor": {
            "mean_ndcg@10_original": _mean(orig_anchor_scores),
            "mean_ndcg@10_oracle": _mean(oracle_anchor_scores),
            "delta_oracle_minus_original": _mean(oracle_anchor_scores) - _mean(orig_anchor_scores),
            "oracle_vs_original_wtl": _win_tie_loss(orig_anchor_scores, oracle_anchor_scores),
        },
        "global": {
            "mean_ndcg@10_original": _mean(orig_global_scores),
            "mean_ndcg@10_oracle": _mean(oracle_global_scores),
            "delta_oracle_minus_original": _mean(oracle_global_scores) - _mean(orig_global_scores),
            "oracle_vs_original_wtl": _win_tie_loss(orig_global_scores, oracle_global_scores),
        },
    }
    if control_outputs_enabled:
        summary["anchor"]["mean_ndcg@10_control"] = _mean(control_anchor_scores)
        summary["anchor"]["delta_oracle_minus_control"] = _mean(oracle_anchor_scores) - _mean(control_anchor_scores)
        summary["anchor"]["oracle_vs_control_wtl"] = _win_tie_loss(control_anchor_scores, oracle_anchor_scores)
        summary["global"]["mean_ndcg@10_control"] = _mean(control_global_scores)
        summary["global"]["delta_oracle_minus_control"] = _mean(oracle_global_scores) - _mean(control_global_scores)
        summary["global"]["oracle_vs_control_wtl"] = _win_tie_loss(control_global_scores, oracle_global_scores)

    sample_path = os.path.join(args.out_dir, "per_sample.jsonl")
    with open(sample_path, "w", encoding="utf-8") as f:
        for row in per_sample_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("Saved per-sample rows: %s", sample_path)
    logger.info("Saved summary: %s", summary_path)
    logger.info(
        "Anchor nDCG@10 original=%.2f oracle=%.2f delta=%.2f",
        summary["anchor"]["mean_ndcg@10_original"],
        summary["anchor"]["mean_ndcg@10_oracle"],
        summary["anchor"]["delta_oracle_minus_original"],
    )
    if control_outputs_enabled:
        logger.info(
            "Anchor nDCG@10 control=%.2f oracle-control delta=%.2f",
            summary["anchor"]["mean_ndcg@10_control"],
            summary["anchor"]["delta_oracle_minus_control"],
        )


if __name__ == "__main__":
    main()
