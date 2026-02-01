import json
import os
import hashlib
from typing import Iterable, List, Tuple


def _rewrite_cache_key(
    prefix: str,
    query: str,
    context_descs: List[str],
    iter_idx: int | None = None,
    schema_labels: List[str] | None = None,
) -> str:
    context_blob = "\n".join([x for x in context_descs if x]).strip()
    schema_blob = "\n".join([x for x in schema_labels if x]).strip() if schema_labels else ""
    combined_blob = "\n".join([b for b in [context_blob, schema_blob] if b])
    context_sig = hashlib.md5(combined_blob.encode("utf-8")).hexdigest() if combined_blob else "none"
    iter_tag = f"||iter={iter_idx}" if iter_idx is not None else ""
    return f"{prefix}||{query}||{context_sig}{iter_tag}"


def _prompt_cache_key(prefix: str, prompt: str) -> str:
    prompt_sig = hashlib.md5((prompt or "").encode("utf-8")).hexdigest() if prompt else "none"
    return f"{prefix}||prompt={prompt_sig}"


def load_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_rewrite_cache(path: str, force_refresh: bool) -> Tuple[dict, dict]:
    rewrite_map: dict[str, str] = {}
    schema_cache_map: dict[str, List[str]] = {}
    if not path or force_refresh or (not os.path.exists(path)):
        return rewrite_map, schema_cache_map
    for rec in load_jsonl(path):
        if "key" in rec and "rewritten_query" in rec:
            rewrite_map[str(rec["key"])] = str(rec["rewritten_query"])
        if "key" in rec and "schema_labels" in rec:
            if str(rec["key"]).startswith("schema||"):
                labels = rec.get("schema_labels")
                if isinstance(labels, list):
                    schema_cache_map[str(rec["key"])] = [str(x) for x in labels if str(x).strip()]
    return rewrite_map, schema_cache_map


def append_jsonl(path: str, records: Iterable[dict]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
