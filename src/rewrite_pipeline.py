import asyncio
from typing import List

from json_repair import repair_json
import json

from retrievers.diver import cosine_topk
from cache_utils import _rewrite_cache_key, append_jsonl
from rewrite_prompts import SCHEMA_PROMPT_TEMPLATE


def _format_schema_prompt(template: str, original_query: str, branch_descs: List[str]) -> str:
    branch_blob = "\n".join([x for x in branch_descs if x])
    try:
        return template.format(
            original_query=(original_query or ""),
            branch_descs=branch_blob,
        )
    except KeyError:
        return (
            template
            .replace("{original_query}", original_query or "")
            .replace("{branch_descs}", branch_blob)
        )


def _parse_schema_output(text: str) -> List[str]:
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
    if not isinstance(obj, dict):
        return []
    labels = obj.get("schema_labels")
    if not isinstance(labels, list):
        return []
    cleaned: List[str] = []
    seen: set[str] = set()
    for label in labels:
        if not isinstance(label, str):
            continue
        text = label.strip()
        if not text:
            continue
        norm = " ".join(text.lower().split())
        if norm in seen:
            continue
        seen.add(norm)
        cleaned.append(text)
        if len(cleaned) >= 5:
            break
    return cleaned


class SchemaGenerator:
    def __init__(
        self,
        *,
        retriever,
        node_registry,
        node_embs,
        llm_api,
        cache_path: str | None,
        schema_cache_map: dict,
        max_concurrent_calls: int,
        topk: int = 10,
        depth: int = 1,
        force_refresh: bool = False,
        logger=None,
    ):
        self.retriever = retriever
        self.node_registry = node_registry
        self.node_embs = node_embs
        self.llm_api = llm_api
        self.cache_path = cache_path
        self.schema_cache_map = schema_cache_map
        self.max_concurrent_calls = max_concurrent_calls
        self.topk = topk
        self.depth = depth
        self.force_refresh = force_refresh
        self.logger = logger

        self.depth_indices = [
            idx for idx, node in enumerate(node_registry)
            if (len(node.path) == depth and (not node.is_leaf))
        ]
        self.depth_embs = node_embs[self.depth_indices] if self.depth_indices else None

    def _schema_retrieve_descs(self, query: str) -> List[str]:
        if self.depth_embs is None or not self.depth_indices:
            return []
        q_emb = self.retriever.encode_query(query)
        k = min(self.topk, len(self.depth_indices))
        res = cosine_topk(q_emb, self.depth_embs, k)
        descs: List[str] = []
        for ridx in res.indices.tolist():
            registry_idx = self.depth_indices[int(ridx)]
            desc = self.node_registry[registry_idx].desc
            if desc:
                descs.append(desc)
        return descs

    def build_schema_map(self, queries: List[str]) -> dict:
        schema_map: dict[str, List[str]] = {}
        schema_prompts = []
        schema_meta = []
        for q in queries:
            branch_descs = self._schema_retrieve_descs(q)
            cache_key = _rewrite_cache_key("schema", q, branch_descs, iter_idx=None)
            if (not self.force_refresh) and (cache_key in self.schema_cache_map):
                schema_map[q] = self.schema_cache_map[cache_key]
                continue
            if not branch_descs:
                schema_map[q] = []
                continue
            schema_prompts.append(_format_schema_prompt(SCHEMA_PROMPT_TEMPLATE, q, branch_descs))
            schema_meta.append({
                "cache_key": cache_key,
                "base_query": q,
                "branch_descs": branch_descs,
            })

        if schema_prompts:
            if self.logger:
                self.logger.info("Generating schema labels via LLM.")
            schema_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(schema_loop)
            try:
                schema_outputs = schema_loop.run_until_complete(
                    self.llm_api.run_batch(schema_prompts, max_concurrent_calls=self.max_concurrent_calls)
                )
            finally:
                schema_loop.close()
                asyncio.set_event_loop(None)

            new_schema_records = []
            for meta, out in zip(schema_meta, schema_outputs):
                labels = _parse_schema_output(out)
                schema_map[meta["base_query"]] = labels
                self.schema_cache_map[meta["cache_key"]] = labels
                new_schema_records.append({
                    "key": meta["cache_key"],
                    "schema_labels": labels,
                    "prompt_name": "schema_branch_v1",
                    "branch_descs": meta.get("branch_descs", []),
                })

            append_jsonl(self.cache_path, new_schema_records)

        return schema_map
