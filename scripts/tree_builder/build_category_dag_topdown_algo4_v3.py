import argparse
import datetime
import json
import os
import pickle
import re
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import pyarrow.parquet as pq
from json_repair import repair_json

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:
    _tqdm = None

_THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path = [
    path
    for path in sys.path
    if os.path.abspath(path or os.getcwd()) != _THIS_SCRIPT_DIR
]

try:
    from scripts.tree_builder.snowflake_utils import (
        SnowflakeCortexRouter,
        _load_snowflake_account_configs,
    )
except ModuleNotFoundError:
    # Intent: keep direct script execution working even when project root is not on sys.path.
    if _THIS_SCRIPT_DIR not in sys.path:
        sys.path.append(_THIS_SCRIPT_DIR)
    from snowflake_utils import (  # type: ignore
        SnowflakeCortexRouter,
        _load_snowflake_account_configs,
    )


SUMMARY_PROMPT_TEMPLATE = (
    "You are an expert in information retrieval and keyword generation.\n\n"
    "Your task is to analyze ONE informational passage and generate hierarchically sorted "
    "retrieval keywords, strictly following the 5-level rubric.\n\n"
    "Important constraints:\n"
    "- Keep keywords concise and actionable for retrieval.\n"
    "- Move from broad abstraction (L1) to specific passage meaning (L5).\n"
    "- Output must be actionable search phrases.\n"
    "- Return only one hierarchical path for this passage.\n\n"
    "Keyword Generation Rules (5 Levels):\n"
    "Level 1: 1-2 words, core subject/domain (broadest).\n"
    "Level 2: 3-4 words, general topic/sub-domain.\n"
    "Level 3: 4-6 words, key concepts/main themes.\n"
    "Level 4: 7-10 words, very concise passage summary.\n"
    "Level 5: 11-20 words, concise and specific passage summary.\n\n"
    "Output JSON only:\n"
    "{{\n"
    "  \"passage_id\": \"{passage_id}\",\n"
    "  \"hierarchical_keywords\": [\"L1\", \"L2\", \"L3\", \"L4\", \"L5\"]\n"
    "}}\n\n"
    "Input Passage ID:\n"
    "{passage_id}\n\n"
    "Website Title:\n"
    "{website_title}\n\n"
    "Passage:\n"
    "{passage}\n"
)

CLUSTER_PROMPT_TEMPLATE = (
    "You are an expert data analyst and taxonomist.\n\n"
    "Task: Cluster summary items into coherent groups for top-down divisive tree construction.\n"
    "Goal constraints:\n"
    "- Build semantically coherent clusters that preserve information coverage.\n"
    "- Use counts only as weak importance hints.\n"
    "- Avoid one dominant catch-all cluster unless unavoidable.\n"
    "- Produce between {min_k} and {max_k} clusters.\n"
    "- Every input summary_id must be assigned exactly once.\n\n"
    "Output JSON only:\n"
    "{{\n"
    "  \"clusters\": [\n"
    "    {{\n"
    "      \"name\": \"topic name\",\n"
    "      \"description\": \"dense conceptual description\",\n"
    "      \"summary_ids\": [\"S1\", \"S2\"]\n"
    "    }}\n"
    "  ]\n"
    "}}\n\n"
    "Input summaries with counts:\n"
    "{summary_items}\n"
)


DEFAULT_LEVEL_LABELS = {
    1: "General Domain",
    2: "General Topic",
    3: "Core Concept",
    4: "Concise Passage Summary",
    5: "Specific Passage Summary",
}

LEVEL_WORD_MAX = {1: 2, 2: 4, 3: 6, 4: 10, 5: 20}
DOC_LEAF_PATTERN = re.compile(r".+/.+\.txt$", flags=re.IGNORECASE)


class _NullProgress:
    def update(self, n: int = 1) -> None:
        return

    def set_postfix(self, ordered_dict=None, refresh: bool = True, **kwargs) -> None:
        return

    def close(self) -> None:
        return


def _make_progress(*, total: Optional[int], desc: str, unit: str):
    if _tqdm is None:
        return _NullProgress()
    return _tqdm(total=total, desc=desc, unit=unit, leave=True)


def _is_doc_leaf_id(node_id: object) -> bool:
    text = str(node_id or "").strip()
    if not text:
        return False
    # Intent: pruning should keep only true split-document leaves, not dangling category nodes.
    return DOC_LEAF_PATTERN.fullmatch(text) is not None


@dataclass
class LongDocSummaryResult:
    doc_id: str
    website_title: str
    summaries: List[str]
    parse_success: bool
    parse_retry_count: int
    was_token_trimmed: bool


@dataclass
class LongDocResult:
    doc_id: str
    website_title: str
    paths: List[List[str]]
    parse_success: bool
    parse_retry_count: int
    was_token_trimmed: bool


@dataclass
class ClusterLLMOutput:
    name: str
    description: str
    summary_ids: List[str]


def _list_subsets(data_dir: str) -> List[str]:
    long_dir = os.path.join(data_dir, "long_documents")
    if not os.path.isdir(long_dir):
        return []
    subsets: List[str] = []
    for name in sorted(os.listdir(long_dir)):
        if not name.endswith("-00000-of-00001.parquet"):
            continue
        subsets.append(name.replace("-00000-of-00001.parquet", ""))
    return subsets


def _resolve_subsets(subset_arg: str, data_dir: str) -> List[str]:
    available = _list_subsets(data_dir)
    if subset_arg.strip().lower() == "all":
        return available
    requested = [x.strip() for x in subset_arg.split(",") if x.strip()]
    missing = [x for x in requested if x not in available]
    if missing:
        raise ValueError(f"Unknown subset(s): {missing}. Available: {available}")
    return requested


def _read_parquet_rows(path: str) -> List[Dict[str, str]]:
    table = pq.read_table(path, columns=["id", "content"])
    rows = table.to_pylist()
    out: List[Dict[str, str]] = []
    for row in rows:
        out.append(
            {
                "id": str(row.get("id", "")).strip(),
                "content": str(row.get("content", "") or "").strip(),
            }
        )
    return out


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _safe_title_case(text: str) -> str:
    words = [w for w in text.split(" ") if w]
    if not words:
        return ""
    return " ".join([w[:1].upper() + w[1:].lower() if w else w for w in words])


def _strip_extension(name: str) -> str:
    return re.sub(r"\.(txt|html?)$", "", name, flags=re.IGNORECASE)


def _website_title_from_doc_id(doc_id: str) -> str:
    tail = doc_id.split("/", 1)[-1] if "/" in doc_id else doc_id
    title = _strip_extension(tail)
    title = re.sub(r"[_\-]+", " ", title)
    title = re.sub(r"\b\d+\b", " ", title)
    title = _normalize_space(title)
    if not title:
        return "Untitled Document"
    return title


def _truncate_words(text: str, max_words: int) -> str:
    if max_words <= 0:
        return _normalize_space(text)
    words = _normalize_space(text).split(" ")
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def _clean_json_candidate(text: str) -> str:
    cleaned = str(text or "")
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1]
    cleaned = cleaned.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        fenced = [parts[i].strip() for i in range(1, len(parts), 2)]
        if fenced:
            cleaned = fenced[-1]
    return cleaned.strip()


def _normalize_label(label: str, level: int) -> str:
    raw = _normalize_space(label)
    raw = re.sub(r"\.(txt|html?)\b", "", raw, flags=re.IGNORECASE)
    raw = raw.strip(" \t\r\n\"'`.,:;|[]{}()")
    words = [w for w in raw.split(" ") if w]
    normalized = " ".join(words).strip()
    if not normalized:
        normalized = DEFAULT_LEVEL_LABELS[level]
    if level <= 3:
        normalized = _safe_title_case(normalized)
    elif normalized:
        normalized = normalized[0].upper() + normalized[1:]
    return normalized


def _normalize_path(raw_path: Iterable[str]) -> Optional[List[str]]:
    if not isinstance(raw_path, list):
        return None
    if len(raw_path) < 5:
        return None
    norm: List[str] = []
    for level in range(1, 6):
        value = raw_path[level - 1] if level - 1 < len(raw_path) else ""
        norm.append(_normalize_label(str(value or ""), level))
    return norm


def _parse_paths_from_output(
    text: str,
    doc_id: str,
    max_alt_paths: int,
) -> Tuple[List[List[str]], bool]:
    cleaned = _clean_json_candidate(text)
    obj = None
    try:
        obj = json.loads(cleaned)
    except Exception:
        try:
            obj = repair_json(cleaned, return_objects=True)
        except Exception:
            obj = None

    parsed_paths: List[List[str]] = []
    if isinstance(obj, dict):
        primary_raw = obj.get("hierarchical_keywords", obj.get("keywords", []))
        primary = _normalize_path(primary_raw)
        if primary:
            parsed_paths.append(primary)
        alt_raw = obj.get("alternate_hierarchical_keywords", [])
        if isinstance(alt_raw, list):
            for item in alt_raw[: max(0, int(max_alt_paths))]:
                alt = _normalize_path(item)
                if alt:
                    parsed_paths.append(alt)

    dedup: List[List[str]] = []
    seen: Set[str] = set()
    for path in parsed_paths:
        key = "||".join(path)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(path)

    if dedup:
        return dedup, True

    fallback_title = _website_title_from_doc_id(doc_id)
    fallback = [
        DEFAULT_LEVEL_LABELS[1],
        DEFAULT_LEVEL_LABELS[2],
        DEFAULT_LEVEL_LABELS[3],
        DEFAULT_LEVEL_LABELS[4],
        _normalize_label(f"Specific support details from {fallback_title}", 5),
    ]
    return [fallback], False


def _build_prompt(
    *,
    passage_id: str,
    website_title: str,
    passage: str,
) -> str:
    return SUMMARY_PROMPT_TEMPLATE.format(
        passage_id=passage_id,
        website_title=website_title,
        passage=passage,
    )


def _build_guarded_prompt(
    *,
    passage_id: str,
    website_title: str,
    passage: str,
    tokenizer=None,
    prompt_token_limit: Optional[int] = None,
) -> Tuple[str, bool]:
    prompt = _build_prompt(
        passage_id=passage_id,
        website_title=website_title,
        passage=passage,
    )
    if tokenizer is None or prompt_token_limit is None:
        return prompt, False
    if prompt_token_limit <= 0:
        return prompt, False
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(token_ids) <= prompt_token_limit:
        return prompt, False

    words = passage.split(" ")
    lo = 0
    hi = len(words)
    best_prompt = prompt
    best_len = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        cand_passage = " ".join(words[:mid])
        cand_prompt = _build_prompt(
            passage_id=passage_id,
            website_title=website_title,
            passage=cand_passage,
        )
        cand_len = len(tokenizer.encode(cand_prompt, add_special_tokens=False))
        if cand_len <= prompt_token_limit:
            best_prompt = cand_prompt
            best_len = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best_prompt, best_len < len(words)


def _label_key(level: int, label: str) -> str:
    # Intent: keep token boundaries in node IDs by replacing separators with underscores instead of dropping them.
    norm = re.sub(r"[^a-z0-9]+", "_", str(label or "").lower()).strip("_")
    norm = re.sub(r"_+", "_", norm)
    if not norm:
        norm = "misc"
    return f"L{level}|{norm}"


def _make_internal_node_id(level: int, label: str) -> str:
    return _label_key(level, label)


def _split_prefix(doc_id: str) -> str:
    return doc_id.split("/", 1)[0] if "/" in doc_id else ""


def _strip_chunk_suffix(name: str) -> str:
    base = _strip_extension(name)
    # Intent: split-doc leaf IDs often end with chunk indices; removing numeric tails aligns them to long-document titles.
    while re.search(r"_[0-9]+$", base):
        base = re.sub(r"_[0-9]+$", "", base)
    return base


def _normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _build_internal_desc(level: int, label: str, num_children: int, num_parents: int) -> str:
    return (
        f"Level {level} category node. Label: {label}. "
        # f"Parent links: {num_parents}. Child links: {num_children}."
    )


def _build_projection_root_desc(subset: str, version: str) -> str:
    return (
        f"ROOT Node: paper-reproduction top-down tree for subset={subset}, version={version}. "
        "Built from long_documents hierarchy and attached split-document leaves."
    )


def _generate_long_doc_summaries(
    *,
    generate_fn: Callable[[List[str]], List[str]],
    tokenizer=None,
    prompt_token_limit: Optional[int] = None,
    long_rows: List[Dict[str, str]],
    batch_size: int,
    max_desc_words: int,
    parse_retry_max: int,
) -> List[LongDocSummaryResult]:
    results: List[LongDocSummaryResult] = []
    pending: List[Dict[str, str]] = []
    progress = _make_progress(
        total=len(long_rows),
        desc="LLM summary generation",
        unit="doc",
    )

    def flush_batch(batch_rows: List[Dict[str, str]]) -> None:
        if not batch_rows:
            return
        prompts: List[str] = []
        trimmed_flags: List[bool] = []
        for row in batch_rows:
            doc_id = row["id"]
            website_title = _website_title_from_doc_id(doc_id)
            desc = _truncate_words(row["content"], max_desc_words)
            prompt, was_trimmed = _build_guarded_prompt(
                tokenizer=tokenizer,
                prompt_token_limit=prompt_token_limit,
                passage_id=doc_id,
                website_title=website_title,
                passage=desc,
            )
            prompts.append(prompt)
            trimmed_flags.append(was_trimmed)

        outputs = generate_fn(prompts)
        if len(outputs) != len(prompts):
            raise RuntimeError(
                f"Generation function returned {len(outputs)} outputs for {len(prompts)} prompts."
            )

        for row, text, was_trimmed, base_prompt in zip(batch_rows, outputs, trimmed_flags, prompts):
            doc_id = row["id"]
            paths, parse_ok = _parse_paths_from_output(
                text=text,
                doc_id=doc_id,
                max_alt_paths=0,
            )
            retry_count = 0
            if (not parse_ok) and parse_retry_max > 0:
                strict_prompt = (
                    base_prompt
                    + "\n\nSTRICT OUTPUT FORMAT: Return exactly one valid JSON object only."
                )
                for _ in range(parse_retry_max):
                    retry_out = generate_fn([strict_prompt])
                    retry_text = retry_out[0] if retry_out else ""
                    retry_count += 1
                    paths, parse_ok = _parse_paths_from_output(
                        text=retry_text,
                        doc_id=doc_id,
                        max_alt_paths=0,
                    )
                    if parse_ok:
                        break

            primary = paths[0] if paths else [
                DEFAULT_LEVEL_LABELS[1],
                DEFAULT_LEVEL_LABELS[2],
                DEFAULT_LEVEL_LABELS[3],
                DEFAULT_LEVEL_LABELS[4],
                DEFAULT_LEVEL_LABELS[5],
            ]
            results.append(
                LongDocSummaryResult(
                    doc_id=doc_id,
                    website_title=_website_title_from_doc_id(doc_id),
                    summaries=list(primary[:5]),
                    parse_success=parse_ok,
                    parse_retry_count=retry_count,
                    was_token_trimmed=was_trimmed,
                )
            )

    try:
        for row in long_rows:
            pending.append(row)
            if len(pending) < batch_size:
                continue
            batch_count = len(pending)
            flush_batch(pending)
            progress.update(batch_count)
            pending = []
        if pending:
            batch_count = len(pending)
            flush_batch(pending)
            progress.update(batch_count)
    finally:
        progress.close()
    return results


def _build_cluster_prompt(
    *,
    summary_rows: List[Dict[str, object]],
    min_k: int,
    max_k: int,
) -> str:
    lines: List[str] = []
    for row in summary_rows:
        sid = str(row.get("summary_id", ""))
        cnt = int(row.get("count", 1))
        txt = _normalize_space(str(row.get("summary", "")))
        lines.append(f"- {sid} | count={cnt} | summary={txt}")
    summary_items = "\n".join(lines)
    return CLUSTER_PROMPT_TEMPLATE.format(
        min_k=max(1, int(min_k)),
        max_k=max(1, int(max_k)),
        summary_items=summary_items,
    )


def _parse_cluster_output(
    *,
    text: str,
    valid_summary_ids: Set[str],
    summary_text_to_ids: Dict[str, List[str]],
    min_k: int,
    max_k: int,
) -> Optional[List[ClusterLLMOutput]]:
    cleaned = _clean_json_candidate(text)
    obj = None
    try:
        obj = json.loads(cleaned)
    except Exception:
        try:
            obj = repair_json(cleaned, return_objects=True)
        except Exception:
            obj = None
    if not isinstance(obj, dict):
        return None

    raw_clusters = obj.get("clusters", [])
    if not isinstance(raw_clusters, list):
        return None

    out: List[ClusterLLMOutput] = []
    assigned_once: Set[str] = set()
    for item in raw_clusters:
        if not isinstance(item, dict):
            continue
        name = _normalize_space(str(item.get("name", "")))
        desc = _normalize_space(str(item.get("description", "")))
        raw_ids = item.get("summary_ids", item.get("keywords", []))
        if not isinstance(raw_ids, list):
            raw_ids = []

        mapped_ids: List[str] = []
        local_seen: Set[str] = set()
        for raw_id in raw_ids:
            token = _normalize_space(str(raw_id))
            if not token:
                continue
            if token in valid_summary_ids:
                mapped = [token]
            else:
                mapped = summary_text_to_ids.get(token, [])
            for sid in mapped:
                # Intent: v3 follows paper-style tree partitioning, so each summary_id is assigned to exactly one cluster.
                if sid in local_seen or sid in assigned_once:
                    continue
                local_seen.add(sid)
                assigned_once.add(sid)
                mapped_ids.append(sid)
        if (not mapped_ids) and (not name) and (not desc):
            continue
        if not mapped_ids:
            continue
        out.append(
            ClusterLLMOutput(
                name=name,
                description=desc,
                summary_ids=mapped_ids,
            )
        )
    if not out:
        return None

    missing = [sid for sid in sorted(valid_summary_ids) if sid not in assigned_once]
    if missing:
        # Intent: enforce full coverage while preserving single-assignment by placing each missing id in exactly one smallest cluster.
        bucket_sizes = [len(x.summary_ids) for x in out]
        for sid in missing:
            min_idx = min(range(len(bucket_sizes)), key=lambda i: bucket_sizes[i])
            if sid not in out[min_idx].summary_ids:
                out[min_idx].summary_ids.append(sid)
                assigned_once.add(sid)
            bucket_sizes[min_idx] += 1
    # Intent: enforce paper-style branching width (M in [10, 20] when feasible) instead of accepting under-clustered outputs.
    if len(out) < max(1, int(min_k)) or len(out) > max(1, int(max_k)):
        return None
    return out


def _fallback_cluster_output(
    *,
    summary_rows: List[Dict[str, object]],
    max_k: int,
) -> List[ClusterLLMOutput]:
    if not summary_rows:
        return []
    ordered = sorted(
        summary_rows,
        key=lambda x: (-int(x.get("count", 1)), str(x.get("summary_id", ""))),
    )
    k = min(max(1, int(max_k)), len(ordered))
    if len(ordered) >= 2:
        k = max(2, k)
    seeds = ordered[:k]
    clusters: List[ClusterLLMOutput] = []
    for seed in seeds:
        name = _normalize_label(str(seed.get("summary", "Topic")), 2)
        clusters.append(
            ClusterLLMOutput(
                name=name,
                description=f"Fallback cluster around summary seed: {name}",
                summary_ids=[str(seed.get("summary_id", ""))],
            )
        )
    for row in ordered[k:]:
        sid = str(row.get("summary_id", ""))
        txt = str(row.get("summary", ""))
        best_idx = 0
        best_score = -1.0
        for idx, seed in enumerate(seeds):
            score = _similarity(_normalize_key(txt), _normalize_key(str(seed.get("summary", ""))))
            if score > best_score:
                best_score = score
                best_idx = idx
        clusters[best_idx].summary_ids.append(sid)
    return clusters


def _select_summary_level(
    *,
    doc_ids: List[str],
    summaries_by_doc: Dict[str, List[str]],
    max_branching: int,
    cluster_prompt_word_budget: int,
) -> int:
    first_valid_level: Optional[int] = None
    for level in range(1, 6):
        values: List[str] = []
        for doc_id in doc_ids:
            summaries = summaries_by_doc.get(doc_id, [])
            if (not summaries) or (len(summaries) < level):
                continue
            values.append(_normalize_space(str(summaries[level - 1])))
        unique_values = sorted(set([x for x in values if x]))
        unique_count = len(unique_values)
        if unique_count <= 1:
            continue
        est_words = sum([len(v.split(" ")) for v in unique_values]) + (10 * unique_count)
        if est_words > max(1, int(cluster_prompt_word_budget)):
            continue
        if first_valid_level is None:
            first_valid_level = level
        # Intent: follow LATTICE SelectSummaryLevel heuristic by picking the earliest abstract level
        # that has enough unique summaries (>M), instead of maximizing uniqueness across levels.
        if unique_count > max(1, int(max_branching)):
            return level
    return first_valid_level if first_valid_level is not None else 1


def _build_topdown_doc_paths_algo4(
    *,
    generate_fn: Callable[[List[str]], List[str]],
    summary_results: List[LongDocSummaryResult],
    max_branching: int,
    max_depth: int,
    cluster_prompt_word_budget: int,
    parse_retry_max: int,
) -> Tuple[List[LongDocResult], Dict[str, object], Dict[str, Dict[str, object]]]:
    summaries_by_doc: Dict[str, List[str]] = {
        x.doc_id: list(x.summaries[:5]) for x in summary_results
    }
    # Intent: keep one canonical path per document to match strict top-down tree behavior in v3.
    best_path_by_doc: Dict[str, List[str]] = {}
    path_desc_votes: Dict[str, Counter] = defaultdict(Counter)
    path_term_votes: Dict[str, Counter] = defaultdict(Counter)

    queue: deque = deque()
    all_doc_ids = [x.doc_id for x in summary_results]
    if len(all_doc_ids) > max(1, int(max_branching)):
        queue.append({"path": [], "doc_ids": list(all_doc_ids)})

    partition_steps = 0
    llm_cluster_calls = 0
    # Intent: queue length changes dynamically as clusters are expanded, so unbounded progress tracks processed nodes.
    queue_progress = _make_progress(total=None, desc="Top-down partition", unit="node")
    try:
        while queue:
            node = queue.popleft()
            queue_progress.update(1)
            parent_path = list(node.get("path", []))
            doc_ids = list(node.get("doc_ids", []))
            if len(doc_ids) <= max(1, int(max_branching)):
                queue_progress.set_postfix(
                    ordered_dict={
                        "queue": len(queue),
                        "steps": partition_steps,
                        "calls": llm_cluster_calls,
                    },
                    refresh=False,
                )
                continue
            if len(parent_path) >= max(1, int(max_depth)):
                queue_progress.set_postfix(
                    ordered_dict={
                        "queue": len(queue),
                        "steps": partition_steps,
                        "calls": llm_cluster_calls,
                    },
                    refresh=False,
                )
                continue

            level = _select_summary_level(
                doc_ids=doc_ids,
                summaries_by_doc=summaries_by_doc,
                max_branching=max_branching,
                cluster_prompt_word_budget=cluster_prompt_word_budget,
            )
            summary_counter: Counter = Counter()
            for doc_id in doc_ids:
                summary_list = summaries_by_doc.get(doc_id, [])
                if len(summary_list) < level:
                    continue
                summary_counter[_normalize_space(summary_list[level - 1])] += 1
            unique_rows: List[Dict[str, object]] = []
            for idx, (summary_text, count) in enumerate(
                sorted(summary_counter.items(), key=lambda x: (-x[1], x[0])),
                start=1,
            ):
                unique_rows.append(
                    {
                        "summary_id": f"S{idx}",
                        "summary": summary_text,
                        "count": int(count),
                    }
                )
            if len(unique_rows) <= 1:
                queue_progress.set_postfix(
                    ordered_dict={
                        "queue": len(queue),
                        "steps": partition_steps,
                        "calls": llm_cluster_calls,
                    },
                    refresh=False,
                )
                continue

            sid_by_summary: Dict[str, str] = {
                str(x["summary"]): str(x["summary_id"]) for x in unique_rows
            }
            summary_text_to_ids: Dict[str, List[str]] = defaultdict(list)
            for row in unique_rows:
                summary_text_to_ids[str(row["summary"])].append(str(row["summary_id"]))
            row_by_sid: Dict[str, Dict[str, object]] = {
                str(row["summary_id"]): row for row in unique_rows
            }
            prompt = _build_cluster_prompt(
                summary_rows=unique_rows,
                min_k=max(2, min(10, len(unique_rows))),
                max_k=max(2, min(20, len(unique_rows))),
            )
            cluster_min_k = max(2, min(10, len(unique_rows)))
            cluster_max_k = max(cluster_min_k, min(20, len(unique_rows)))
            llm_out = generate_fn([prompt])
            llm_cluster_calls += 1
            parsed = _parse_cluster_output(
                text=llm_out[0] if llm_out else "",
                valid_summary_ids=set([str(x["summary_id"]) for x in unique_rows]),
                summary_text_to_ids=summary_text_to_ids,
                min_k=cluster_min_k,
                max_k=cluster_max_k,
            )
            retry_count = 0
            if (parsed is None) and parse_retry_max > 0:
                strict_prompt = prompt + "\n\nSTRICT OUTPUT FORMAT: Return exactly one valid JSON object only."
                while retry_count < parse_retry_max:
                    retry_out = generate_fn([strict_prompt])
                    llm_cluster_calls += 1
                    retry_count += 1
                    parsed = _parse_cluster_output(
                        text=retry_out[0] if retry_out else "",
                        valid_summary_ids=set([str(x["summary_id"]) for x in unique_rows]),
                        summary_text_to_ids=summary_text_to_ids,
                        min_k=cluster_min_k,
                        max_k=cluster_max_k,
                    )
                    if parsed is not None:
                        break
            if parsed is None:
                parsed = _fallback_cluster_output(
                    summary_rows=unique_rows,
                    max_k=cluster_max_k,
                )

            sid_to_cluster_idx: Dict[str, int] = {}
            for idx, c in enumerate(parsed):
                for sid in c.summary_ids:
                    if sid not in sid_to_cluster_idx:
                        sid_to_cluster_idx[sid] = idx
            if not sid_to_cluster_idx:
                queue_progress.set_postfix(
                    ordered_dict={
                        "queue": len(queue),
                        "steps": partition_steps,
                        "calls": llm_cluster_calls,
                    },
                    refresh=False,
                )
                continue

            cluster_members: Dict[int, Set[str]] = defaultdict(set)
            for doc_id in doc_ids:
                summary_list = summaries_by_doc.get(doc_id, [])
                if len(summary_list) < level:
                    continue
                sid = sid_by_summary.get(_normalize_space(summary_list[level - 1]), "")
                cidx = sid_to_cluster_idx.get(sid)
                if cidx is None:
                    continue
                cluster_members[cidx].add(doc_id)
            if not cluster_members:
                queue_progress.set_postfix(
                    ordered_dict={
                        "queue": len(queue),
                        "steps": partition_steps,
                        "calls": llm_cluster_calls,
                    },
                    refresh=False,
                )
                continue

            for cidx, member_set in sorted(cluster_members.items(), key=lambda x: (-len(x[1]), x[0])):
                members = sorted(member_set)
                cluster = parsed[cidx] if cidx < len(parsed) else ClusterLLMOutput("", "", [])
                raw_label = cluster.name or cluster.description or f"Topic {cidx + 1}"
                level_for_norm = min(5, len(parent_path) + 1)
                child_label = _normalize_label(raw_label, level_for_norm)
                child_path = [*parent_path, child_label]
                path_key = "||".join(child_path)
                raw_desc = _normalize_space(cluster.description or cluster.name or "")
                if raw_desc:
                    path_desc_votes[path_key][raw_desc] += max(1, len(members))
                for sid in cluster.summary_ids:
                    row = row_by_sid.get(str(sid))
                    if row is None:
                        continue
                    term = _truncate_words(str(row.get("summary", "")), 8)
                    if term:
                        path_term_votes[path_key][term] += int(row.get("count", 1))
                for doc_id in members:
                    best_path_by_doc[doc_id] = list(child_path)
                if (
                    len(members) > max(1, int(max_branching))
                    and len(child_path) < max(1, int(max_depth))
                ):
                    queue.append(
                        {
                            "path": child_path,
                            "doc_ids": list(members),
                        }
                    )
            partition_steps += 1
            queue_progress.set_postfix(
                ordered_dict={
                    "queue": len(queue),
                    "steps": partition_steps,
                    "calls": llm_cluster_calls,
                },
                refresh=False,
            )
    finally:
        queue_progress.close()

    out: List[LongDocResult] = []
    for item in summary_results:
        seed_path = list(best_path_by_doc.get(item.doc_id, []))
        norm_path: List[str] = []
        for idx, label in enumerate(seed_path, start=1):
            # Intent: strict paper reproduction uses only partition-generated topic paths; do not force-fill to 5 levels.
            norm_path.append(_normalize_label(label, min(5, idx)))
        final_paths: List[List[str]] = [norm_path] if norm_path else [[]]
        out.append(
            LongDocResult(
                doc_id=item.doc_id,
                website_title=item.website_title,
                paths=final_paths,
                parse_success=item.parse_success,
                parse_retry_count=item.parse_retry_count,
                was_token_trimmed=item.was_token_trimmed,
            )
        )
    multi_path_doc_count = sum([1 for x in out if len(x.paths) > 1])
    meta = {
        "partition_steps": int(partition_steps),
        "cluster_llm_calls": int(llm_cluster_calls),
        "max_depth": int(max_depth),
        "multi_path_doc_count": int(multi_path_doc_count),
    }
    path_desc_payload: Dict[str, Dict[str, object]] = {}
    for path_key in set(list(path_desc_votes.keys()) + list(path_term_votes.keys())):
        desc_counter = path_desc_votes.get(path_key, Counter())
        term_counter = path_term_votes.get(path_key, Counter())
        best_desc = ""
        if desc_counter:
            best_desc = str(desc_counter.most_common(1)[0][0])
        top_terms = [term for term, _ in term_counter.most_common(8)]
        path_desc_payload[path_key] = {
            "desc": best_desc,
            "key_terms": top_terms,
        }
    return out, meta, path_desc_payload


def _compute_num_leaves(tree_node: Dict) -> int:
    children = tree_node.get("child") or []
    if not children:
        tree_node["num_leaves"] = 1
        return 1
    total = 0
    for child in children:
        total += _compute_num_leaves(child)
    tree_node["num_leaves"] = total
    return total


def _export_node_catalog(tree_dict: Dict, out_jsonl: str) -> None:
    _compute_num_leaves(tree_dict)
    records: List[Dict] = []

    def walk(node: Dict, path: Tuple[int, ...]) -> None:
        child = node.get("child") or []
        node_id = node.get("id")
        raw_desc = _normalize_space(str(node.get("desc", "") or ""))
        if node_id is None:
            desc_with_id = raw_desc
        else:
            # Intent: include node id in node-catalog text because embedding consumes desc field only.
            desc_with_id = _normalize_space(f"ID: {node_id}. {raw_desc}")
        rec = {
            "path": list(path),
            "depth": len(path),
            "is_leaf": len(child) == 0,
            "num_children": len(child),
            "num_leaves": int(node.get("num_leaves", 1)),
            "id": node_id,
            "desc": desc_with_id,
        }
        records.append(rec)
        for idx, c in enumerate(child):
            walk(c, (*path, idx))

    walk(tree_dict, ())
    for idx, rec in enumerate(records):
        rec["registry_idx"] = idx

    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    depth1_path = os.path.splitext(out_jsonl)[0] + "_depth1.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f_all, open(depth1_path, "w", encoding="utf-8") as f_d1:
        for rec in records:
            line = json.dumps(rec, ensure_ascii=False)
            f_all.write(line + "\n")
            if rec["depth"] == 1:
                f_d1.write(line + "\n")


def _build_subset_artifacts(
    *,
    dataset: str,
    subset: str,
    long_rows: List[Dict[str, str]],
    split_rows: List[Dict[str, str]],
    long_results: List[LongDocResult],
    version: str,
    out_dir: str,
    algo4_meta: Optional[Dict[str, object]] = None,
    path_desc_payload: Optional[Dict[str, Dict[str, object]]] = None,
    category_desc_key_terms_topk: int = 0,
    category_desc_max_chars: int = 320,
    category_desc_child_anchor_topk: int = 0,
    category_desc_child_anchor_min_ratio: float = 0.12,
    category_desc_tail_anchor_topk: int = 0,
) -> Dict[str, object]:
    nodes: Dict[str, Dict[str, object]] = {}
    edge_weights: Dict[Tuple[str, str], float] = defaultdict(float)
    edge_types: Dict[Tuple[str, str], str] = {}
    parent_of: Dict[str, str] = {}

    root_id = "L0|root"
    nodes[root_id] = {
        "id": root_id,
        "level": 0,
        "kind": "root",
        "label": "Root",
        "display_id": None,
        "desc": _build_projection_root_desc(subset=subset, version=version),
        "long_support_ids": set(),
        "split_support_ids": set(),
    }

    long_doc_terminal_node: Dict[str, str] = {}
    long_doc_title_key: Dict[str, str] = {}
    long_doc_by_prefix: Dict[str, List[str]] = defaultdict(list)
    node_desc_votes: Dict[str, Counter] = defaultdict(Counter)
    node_term_votes: Dict[str, Counter] = defaultdict(Counter)

    def _path_scoped_node_id(path_labels: List[str]) -> str:
        parts: List[str] = []
        for idx, label in enumerate(path_labels, start=1):
            parts.append(_label_key(idx, label).split("|", 1)[-1])
        suffix = "__".join([p for p in parts if p]) or "misc"
        return f"L{len(path_labels)}|path|{suffix}"

    def ensure_internal_node(level: int, label: str, path_labels: List[str]) -> str:
        # Intent: path-scoped IDs prevent accidental cross-branch merges, preserving strict tree structure.
        node_id = _path_scoped_node_id(path_labels[:level])
        if node_id not in nodes:
            nodes[node_id] = {
                "id": node_id,
                "level": level,
                "kind": "category",
                "label": label,
                "display_id": label,
                "desc": "",
                "long_support_ids": set(),
                "split_support_ids": set(),
            }
        return node_id

    def ensure_leaf_node(doc_id: str, content: str, parent_level: int) -> str:
        if doc_id not in nodes:
            nodes[doc_id] = {
                "id": doc_id,
                # Intent: leaf depth should follow the actual parent depth in strict top-down reproduction.
                "level": int(parent_level) + 1,
                "kind": "leaf",
                "label": doc_id,
                "display_id": doc_id,
                "desc": content,
                "long_support_ids": set(),
                "split_support_ids": {doc_id},
            }
        return doc_id

    def add_tree_edge(parent_id: str, child_id: str, weight: float, edge_type: str) -> None:
        prev_parent = parent_of.get(child_id)
        # Intent: v3 paper reproduction enforces exactly one parent per node.
        if (prev_parent is not None) and (prev_parent != parent_id):
            return
        if prev_parent is None:
            parent_of[child_id] = parent_id
        key = (parent_id, child_id)
        edge_weights[key] += float(weight)
        edge_types[key] = edge_type

    for result in long_results:
        primary_path = list(result.paths[0]) if result.paths else []
        parent_id = root_id
        path_labels: List[str] = []
        # Intent: keep full generated category depth; max depth is controlled upstream by top-down partition settings.
        for level, label in enumerate(primary_path, start=1):
            path_labels.append(label)
            node_id = ensure_internal_node(level, label, path_labels)
            add_tree_edge(parent_id, node_id, weight=1.0, edge_type="category")
            nodes[node_id]["long_support_ids"].add(result.doc_id)
            prefix_key = "||".join(path_labels[:level])
            payload = (path_desc_payload or {}).get(prefix_key, {})
            vote_desc = _normalize_space(str(payload.get("desc", "")))
            if vote_desc:
                node_desc_votes[node_id][vote_desc] += 1
            for term in payload.get("key_terms", []) if isinstance(payload, dict) else []:
                norm_term = _normalize_space(str(term))
                if norm_term:
                    node_term_votes[node_id][norm_term] += 1
            if str(label):
                node_term_votes[node_id][_normalize_space(str(label))] += 1
            parent_id = node_id
        long_doc_terminal_node[result.doc_id] = parent_id
        long_doc_title_key[result.doc_id] = _normalize_key(result.website_title)
        long_doc_by_prefix[_split_prefix(result.doc_id)].append(result.doc_id)

    leaf_membership_rows: List[Dict[str, object]] = []
    leaf_progress = _make_progress(total=len(split_rows), desc="Attach split leaves", unit="leaf")
    try:
        for row in split_rows:
            split_id = row["id"]
            split_content = row["content"]
            split_prefix = _split_prefix(split_id)
            split_tail = split_id.split("/", 1)[-1] if "/" in split_id else split_id
            split_base_key = _normalize_key(_website_title_from_doc_id(_strip_chunk_suffix(split_tail)))

            selected_long_doc = ""
            selected_parent = root_id
            selected_score = 0.0
            candidates = long_doc_by_prefix.get(split_prefix, [])
            for long_doc_id in candidates:
                terminal = long_doc_terminal_node.get(long_doc_id, root_id)
                sim = _similarity(split_base_key, long_doc_title_key.get(long_doc_id, ""))
                rank_key = (sim, terminal, long_doc_id)
                best_key = (selected_score, selected_parent, selected_long_doc)
                if rank_key > best_key:
                    selected_score = float(sim)
                    selected_parent = terminal
                    selected_long_doc = long_doc_id

            leaf_node_id = ensure_leaf_node(
                split_id,
                split_content,
                parent_level=int(nodes.get(selected_parent, {}).get("level", 0)),
            )
            add_tree_edge(
                selected_parent,
                leaf_node_id,
                weight=max(0.01, float(selected_score)),
                edge_type="leaf_attach",
            )
            nodes[selected_parent]["split_support_ids"].add(split_id)
            if selected_long_doc:
                nodes[leaf_node_id]["long_support_ids"].add(selected_long_doc)
            leaf_membership_rows.append(
                {
                    "doc_id": split_id,
                    "selected_parent_node_ids": [selected_parent],
                    "selected_parent_long_doc_ids": [selected_long_doc],
                    "selected_scores": [round(float(selected_score), 6)],
                }
            )
            leaf_progress.update(1)
    finally:
        leaf_progress.close()

    while True:
        has_children: Set[str] = set(parent_of.values())
        removable: List[str] = []
        for node_id in list(nodes.keys()):
            if node_id == root_id:
                continue
            if node_id in has_children:
                continue
            if _is_doc_leaf_id(node_id):
                continue
            removable.append(node_id)
        if not removable:
            break
        removable_set = set(removable)
        # Intent: prune dangling non-document category leaves so only split-document ids remain terminal leaves.
        for node_id in removable:
            nodes.pop(node_id, None)
            parent_of.pop(node_id, None)
        for edge_key in list(edge_weights.keys()):
            parent_id, child_id = edge_key
            if parent_id in removable_set or child_id in removable_set:
                edge_weights.pop(edge_key, None)
                edge_types.pop(edge_key, None)

    for node_id in list(nodes.keys()):
        if node_id == root_id:
            continue
        if node_id in parent_of:
            continue
        # Intent: reconnect rare orphans directly under root so exported tree is always connected.
        add_tree_edge(root_id, node_id, weight=0.01, edge_type="reconnect")

    incoming: Dict[str, Set[str]] = defaultdict(set)
    outgoing: Dict[str, Set[str]] = defaultdict(set)
    for child_id, parent_id in parent_of.items():
        incoming[child_id].add(parent_id)
        outgoing[parent_id].add(child_id)

    tree_children: Dict[str, List[str]] = {
        parent_id: list(child_ids) for parent_id, child_ids in outgoing.items()
    }

    projection_leaf_count_cache: Dict[str, int] = {}

    def _projection_leaf_count(node_id: str) -> int:
        cached = projection_leaf_count_cache.get(node_id)
        if cached is not None:
            return int(cached)
        node = nodes.get(node_id, {})
        if str(node.get("kind", "")) == "leaf":
            projection_leaf_count_cache[node_id] = 1
            return 1
        child_ids = tree_children.get(node_id, [])
        if not child_ids:
            projection_leaf_count_cache[node_id] = 0
            return 0
        total = 0
        for child_id in child_ids:
            total += _projection_leaf_count(child_id)
        projection_leaf_count_cache[node_id] = int(total)
        return int(total)

    for node_id, node in nodes.items():
        if node_id == root_id:
            continue
        in_deg = len(incoming.get(node_id, set()))
        out_deg = len(outgoing.get(node_id, set()))
        if node["kind"] == "category":
            desc_counter = node_desc_votes.get(node_id, Counter())
            term_counter = node_term_votes.get(node_id, Counter())
            base_desc = str(desc_counter.most_common(1)[0][0]) if desc_counter else ""
            if not base_desc:
                base_desc = _build_internal_desc(
                    level=int(node["level"]),
                    label=str(node["label"]),
                    num_children=out_deg,
                    num_parents=in_deg,
                )
            topk = max(0, int(category_desc_key_terms_topk))
            top_terms = [term for term, _ in term_counter.most_common(topk)] if topk > 0 else []
            composed_desc = base_desc
            if top_terms:
                composed_desc = f"{base_desc} Key terms: {', '.join(top_terms)}."
            if int(node["level"]) <= 3 and max(0, int(category_desc_child_anchor_topk)) > 0:
                child_category_scores: List[Tuple[int, str]] = []
                for child_id in tree_children.get(node_id, []):
                    child_node = nodes.get(child_id, {})
                    if str(child_node.get("kind", "")) != "category":
                        continue
                    child_label = _normalize_space(str(child_node.get("label", "")))
                    if not child_label:
                        continue
                    leaf_count = int(_projection_leaf_count(child_id))
                    child_category_scores.append((leaf_count, child_label))
                if child_category_scores:
                    child_category_scores.sort(key=lambda x: (-x[0], x[1]))
                    max_leaf_count = max(1, int(child_category_scores[0][0]))
                    min_ratio = max(0.0, float(category_desc_child_anchor_min_ratio))
                    min_leaf_count = max(1, int(max_leaf_count * min_ratio))
                    topk_child = max(0, int(category_desc_child_anchor_topk))
                    selected_labels: List[str] = []
                    for leaf_count, child_label in child_category_scores:
                        if leaf_count < min_leaf_count:
                            continue
                        selected_labels.append(_truncate_words(child_label, 8))
                        if len(selected_labels) >= topk_child:
                            break
                    if len(selected_labels) < min(2, len(child_category_scores)):
                        for _, child_label in child_category_scores:
                            short_label = _truncate_words(child_label, 8)
                            if short_label in selected_labels:
                                continue
                            selected_labels.append(short_label)
                            if len(selected_labels) >= min(topk_child, len(child_category_scores)):
                                break
                    tail_labels: List[str] = []
                    tail_topk = max(0, int(category_desc_tail_anchor_topk))
                    if tail_topk > 0:
                        lower_half_start = max(0, len(child_category_scores) // 2)
                        tail_pool = child_category_scores[lower_half_start:]
                        if not tail_pool:
                            tail_pool = child_category_scores
                        # Intent: retain minority branches in parent desc so traversal can keep less frequent anchors.
                        for _, child_label in reversed(tail_pool):
                            short_label = _truncate_words(child_label, 8)
                            if (not short_label) or (short_label in selected_labels) or (short_label in tail_labels):
                                continue
                            tail_labels.append(short_label)
                            if len(tail_labels) >= tail_topk:
                                break
                    if selected_labels:
                        composed_desc = (
                            f"{composed_desc} Representative child topics: "
                            f"{'; '.join(selected_labels)}."
                        )
                    if tail_labels:
                        composed_desc = (
                            f"{composed_desc} Tail child anchors: "
                            f"{'; '.join(tail_labels)}."
                        )
            composed_desc = _normalize_space(composed_desc)
            max_chars = max(64, int(category_desc_max_chars))
            if len(composed_desc) > max_chars:
                truncated = composed_desc[:max_chars].rstrip()
                if " " in truncated:
                    truncated = truncated.rsplit(" ", 1)[0]
                composed_desc = truncated
            node["desc"] = composed_desc

    def child_sort_key(node_id: str) -> Tuple[int, int, str]:
        node = nodes[node_id]
        is_leaf = 1 if node["kind"] == "leaf" else 0
        return (int(node["level"]), is_leaf, str(node["display_id"]))

    def build_tree_dict(node_id: str) -> Dict:
        node = nodes[node_id]
        children_ids = sorted(tree_children.get(node_id, []), key=child_sort_key)
        child_nodes = [build_tree_dict(cid) for cid in children_ids]
        return {
            "id": node["display_id"],
            "desc": node["desc"],
            "child": child_nodes if child_nodes else None,
        }

    tree_dict = build_tree_dict(root_id)
    tree_dict["id"] = None

    dag_nodes_out: List[Dict[str, object]] = []
    for node_id, node in sorted(nodes.items(), key=lambda x: (int(x[1]["level"]), str(x[0]))):
        dag_nodes_out.append(
            {
                "id": node_id,
                "display_id": node["display_id"],
                "level": int(node["level"]),
                "kind": node["kind"],
                "label": node["label"],
                "desc": node["desc"],
                "num_parents": len(incoming.get(node_id, set())),
                "num_children": len(tree_children.get(node_id, [])),
                "long_support_count": len(node["long_support_ids"]),
                "split_support_count": len(node["split_support_ids"]),
            }
        )

    dag_edges_out: List[Dict[str, object]] = []
    for (parent_id, child_id), weight in sorted(edge_weights.items(), key=lambda x: (x[0][0], x[0][1])):
        dag_edges_out.append(
            {
                "parent_id": parent_id,
                "child_id": child_id,
                "weight": float(weight),
                "edge_type": edge_types.get((parent_id, child_id), "unknown"),
                "is_projection_parent": True,
            }
        )

    level_counter = Counter([int(x["level"]) for x in dag_nodes_out])
    multi_parent_counter = Counter(
        [
            int(node["level"])
            for node_id, node in nodes.items()
            if len(incoming.get(node_id, set())) > 1
        ]
    )
    leaf_parent_hist = Counter([len(x["selected_parent_node_ids"]) for x in leaf_membership_rows])
    tree_edge_count = len(parent_of)

    report = {
        "meta": {
            "dataset": dataset,
            "subset": subset,
            "version": version,
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
            "builder": "topdown_algo4_v3",
            "mode": "paper_repro_tree_only",
        },
        "counts": {
            "num_long_documents": len(long_rows),
            "num_split_documents": len(split_rows),
            "num_leaf_split_documents": len(leaf_membership_rows),
            "num_tree_nodes": len(nodes),
            "num_tree_edges": int(tree_edge_count),
            "num_dag_nodes": len(nodes),
            "num_dag_edges": int(tree_edge_count),
            "num_projection_nodes": len(nodes),
            "num_projection_edges": int(tree_edge_count),
        },
        "level_distribution": {str(k): int(v) for k, v in sorted(level_counter.items())},
        "multi_parent_level_distribution": {
            str(k): int(v) for k, v in sorted(multi_parent_counter.items())
        },
        "leaf_parent_histogram": {str(k): int(v) for k, v in sorted(leaf_parent_hist.items())},
    }
    if isinstance(algo4_meta, dict):
        report["meta"]["algo4"] = algo4_meta

    os.makedirs(out_dir, exist_ok=True)
    version_u = version.replace("-", "_")
    dag_json_path = os.path.join(out_dir, f"category_dag_topdown_algo4_{version_u}.json")
    dag_edge_jsonl_path = os.path.join(out_dir, f"category_dag_edges_topdown_algo4_{version_u}.jsonl")
    leaf_membership_path = os.path.join(out_dir, f"category_leaf_membership_topdown_algo4_{version_u}.jsonl")
    report_path = os.path.join(out_dir, f"category_build_report_topdown_algo4_{version_u}.json")
    projection_tree_path = os.path.join(out_dir, f"category_tree_projection_topdown_algo4_{version_u}.pkl")
    runtime_tree_path = os.path.join(out_dir, f"tree-category-topdown-algo4-{version}.pkl")
    node_catalog_path = os.path.join(out_dir, f"category_node_catalog_topdown_algo4_{version_u}.jsonl")

    with open(dag_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": report["meta"],
                "nodes": dag_nodes_out,
                "edges": dag_edges_out,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(dag_edge_jsonl_path, "w", encoding="utf-8") as f:
        for row in dag_edges_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(leaf_membership_path, "w", encoding="utf-8") as f:
        for row in leaf_membership_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    pickle.dump(tree_dict, open(projection_tree_path, "wb"))
    pickle.dump(tree_dict, open(runtime_tree_path, "wb"))
    _export_node_catalog(tree_dict, node_catalog_path)

    return {
        "dag_json_path": dag_json_path,
        "dag_edge_jsonl_path": dag_edge_jsonl_path,
        "leaf_membership_path": leaf_membership_path,
        "report_path": report_path,
        "projection_tree_path": projection_tree_path,
        "runtime_tree_path": runtime_tree_path,
        "node_catalog_path": node_catalog_path,
        "report": report,
    }


def _save_long_results(
    out_dir: str,
    version: str,
    long_results: List[LongDocResult],
) -> str:
    version_u = version.replace("-", "_")
    out_path = os.path.join(out_dir, f"category_longdoc_paths_topdown_algo4_{version_u}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for row in long_results:
            f.write(
                json.dumps(
                    {
                        "doc_id": row.doc_id,
                        "website_title": row.website_title,
                        "paths": row.paths,
                        "parse_success": row.parse_success,
                        "parse_retry_count": row.parse_retry_count,
                        "was_token_trimmed": row.was_token_trimmed,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return out_path


def _save_long_summary_results(
    out_dir: str,
    version: str,
    summary_results: List[LongDocSummaryResult],
    out_path: Optional[str] = None,
) -> str:
    if out_path is None:
        version_u = version.replace("-", "_")
        out_path = os.path.join(out_dir, f"category_longdoc_summaries_topdown_algo4_{version_u}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for row in summary_results:
            # Intent: persist raw 5-level summary outputs before top-down repartitioning for auditing and ablation analysis.
            f.write(
                json.dumps(
                    {
                        "doc_id": row.doc_id,
                        "website_title": row.website_title,
                        "summaries": row.summaries,
                        "parse_success": row.parse_success,
                        "parse_retry_count": row.parse_retry_count,
                        "was_token_trimmed": row.was_token_trimmed,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return out_path


def _load_long_summary_results(cache_path: str) -> List[LongDocSummaryResult]:
    out: List[LongDocSummaryResult] = []
    with open(cache_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = str(obj.get("doc_id", "")).strip()
            if not doc_id:
                continue
            raw_summaries = obj.get("summaries", [])
            if not isinstance(raw_summaries, list):
                raw_summaries = []
            summaries: List[str] = []
            for level in range(1, 6):
                fallback = DEFAULT_LEVEL_LABELS[level]
                raw_value = raw_summaries[level - 1] if level - 1 < len(raw_summaries) else fallback
                summaries.append(_normalize_label(str(raw_value or ""), level))
            out.append(
                LongDocSummaryResult(
                    doc_id=doc_id,
                    website_title=str(obj.get("website_title", _website_title_from_doc_id(doc_id))),
                    summaries=summaries,
                    parse_success=bool(obj.get("parse_success", True)),
                    parse_retry_count=int(obj.get("parse_retry_count", 0)),
                    was_token_trimmed=bool(obj.get("was_token_trimmed", False)),
                )
            )
    return out


def _is_summary_cache_compatible(
    *,
    summary_results: List[LongDocSummaryResult],
    long_rows: List[Dict[str, str]],
) -> bool:
    if not summary_results:
        return False
    expected_ids = set([str(row.get("id", "")) for row in long_rows if str(row.get("id", ""))])
    cached_ids = set([str(row.doc_id) for row in summary_results if str(row.doc_id)])
    return expected_ids == cached_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build paper-style top-down tree partition (Algorithm 4) "
            "from BRIGHT long_documents and export runtime-compatible tree artifacts."
        )
    )
    parser.add_argument("--dataset", type=str, default="BRIGHT")
    parser.add_argument("--subset", type=str, required=True, help="Subset name, comma list, or 'all'")
    parser.add_argument("--data_dir", type=str, default="data/BRIGHT")
    parser.add_argument("--trees_root", type=str, default="trees")
    parser.add_argument("--version", type=str, default="v3")
    parser.add_argument("--llm", type=str, required=True, help="Snowflake Cortex model name")
    parser.add_argument("--env_file", type=str, default=".env")
    parser.add_argument("--snowflake_request_timeout", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--summary_max_tokens", type=int, default=0)
    parser.add_argument("--cluster_max_tokens", type=int, default=0)
    parser.add_argument("--max_input_prompt_chars", type=int, default=120000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_desc_words", type=int, default=4096)
    parser.add_argument("--parse_retry_max", type=int, default=1)
    parser.add_argument("--max_branching", type=int, default=20)
    parser.add_argument("--max_partition_depth", type=int, default=5)
    parser.add_argument("--cluster_prompt_word_budget", type=int, default=12000)
    parser.add_argument("--category_desc_key_terms_topk", type=int, default=0)
    parser.add_argument("--category_desc_max_chars", type=int, default=320)
    parser.add_argument("--category_desc_child_anchor_topk", type=int, default=0)
    parser.add_argument("--category_desc_child_anchor_min_ratio", type=float, default=0.12)
    parser.add_argument("--category_desc_tail_anchor_topk", type=int, default=0)
    parser.add_argument("--summary_cache_path", type=str, default="")
    parser.add_argument("--disable_summary_cache", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    subsets = _resolve_subsets(args.subset, args.data_dir)
    if not subsets:
        raise ValueError(f"No subsets available under {args.data_dir}/long_documents")

    account_configs = _load_snowflake_account_configs(args.env_file)
    print(
        f"[Snowflake] model={args.llm} "
        f"accounts={len(account_configs)} env_file={args.env_file}"
    )
    summary_max_tokens = int(args.summary_max_tokens) if int(args.summary_max_tokens) > 0 else int(args.max_tokens)
    cluster_max_tokens = int(args.cluster_max_tokens) if int(args.cluster_max_tokens) > 0 else int(args.max_tokens)
    print(
        f"[Snowflake] max_tokens summary={summary_max_tokens} "
        f"cluster={cluster_max_tokens}"
    )

    # Intent: keep prompt templates fixed while allocating larger output budget to cluster JSON, which carries many summary_ids.
    summary_router = SnowflakeCortexRouter(
        account_configs=account_configs,
        model=args.llm,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=summary_max_tokens,
        request_timeout=args.snowflake_request_timeout,
        max_input_prompt_chars=args.max_input_prompt_chars,
    )
    if cluster_max_tokens == summary_max_tokens:
        cluster_router = summary_router
    else:
        cluster_router = SnowflakeCortexRouter(
            account_configs=account_configs,
            model=args.llm,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=cluster_max_tokens,
            request_timeout=args.snowflake_request_timeout,
            max_input_prompt_chars=args.max_input_prompt_chars,
        )

    def generate_summary_fn(prompts: List[str]) -> List[str]:
        return [summary_router.complete(prompt) for prompt in prompts]

    def generate_cluster_fn(prompts: List[str]) -> List[str]:
        return [cluster_router.complete(prompt) for prompt in prompts]

    try:
        for subset in subsets:
            out_dir = os.path.join(args.trees_root, args.dataset, subset)
            version_u = args.version.replace("-", "_")
            report_path = os.path.join(out_dir, f"category_build_report_topdown_algo4_{version_u}.json")
            if os.path.exists(report_path) and (not args.overwrite):
                print(f"[Skip] {subset}: {report_path} already exists. Use --overwrite to rebuild.")
                continue

            long_path = os.path.join(args.data_dir, "long_documents", f"{subset}-00000-of-00001.parquet")
            split_path = os.path.join(args.data_dir, "documents", f"{subset}-00000-of-00001.parquet")
            if not os.path.exists(long_path):
                raise FileNotFoundError(f"Missing long_documents parquet: {long_path}")
            if not os.path.exists(split_path):
                raise FileNotFoundError(f"Missing documents parquet: {split_path}")

            print(f"[Build] subset={subset}")
            long_rows = _read_parquet_rows(long_path)
            split_rows = _read_parquet_rows(split_path)
            print(f"[Load] long_docs={len(long_rows)} split_docs={len(split_rows)}")

            summary_cache_path = args.summary_cache_path.strip()
            if not summary_cache_path:
                summary_cache_path = os.path.join(
                    out_dir,
                    f"category_longdoc_summaries_topdown_algo4_{version_u}.jsonl",
                )

            summary_results: List[LongDocSummaryResult] = []
            loaded_from_cache = False
            if (not args.disable_summary_cache) and os.path.exists(summary_cache_path):
                try:
                    cached_summaries = _load_long_summary_results(summary_cache_path)
                    if _is_summary_cache_compatible(
                        summary_results=cached_summaries,
                        long_rows=long_rows,
                    ):
                        summary_results = cached_summaries
                        loaded_from_cache = True
                        print(
                            f"[LLM-Summary] loaded cache docs={len(summary_results)} "
                            f"path={summary_cache_path}"
                        )
                    else:
                        print(
                            f"[LLM-Summary] cache mismatch, regenerating "
                            f"path={summary_cache_path}"
                        )
                except Exception as exc:
                    print(
                        f"[LLM-Summary] cache load failed, regenerating "
                        f"path={summary_cache_path} error={exc}"
                    )

            if not loaded_from_cache:
                summary_results = _generate_long_doc_summaries(
                    generate_fn=generate_summary_fn,
                    long_rows=long_rows,
                    batch_size=args.batch_size,
                    max_desc_words=args.max_desc_words,
                    parse_retry_max=args.parse_retry_max,
                )
                os.makedirs(out_dir, exist_ok=True)
                _save_long_summary_results(
                    out_dir=out_dir,
                    version=args.version,
                    summary_results=summary_results,
                    out_path=summary_cache_path,
                )
                print(
                    f"[LLM-Summary] generated and cached docs={len(summary_results)} "
                    f"path={summary_cache_path}"
                )
            # Intent: implement paper-style top-down queue partition first, then project to fixed 5-level paths for runtime compatibility.
            long_results, algo4_meta, path_desc_payload = _build_topdown_doc_paths_algo4(
                generate_fn=generate_cluster_fn,
                summary_results=summary_results,
                max_branching=args.max_branching,
                max_depth=args.max_partition_depth,
                cluster_prompt_word_budget=args.cluster_prompt_word_budget,
                parse_retry_max=args.parse_retry_max,
            )

            parse_ok = sum([1 for x in summary_results if x.parse_success])
            parse_fail = len(summary_results) - parse_ok
            print(f"[LLM-Summary] parse_success={parse_ok} parse_fail={parse_fail}")
            print(f"[Algo4] {json.dumps(algo4_meta, ensure_ascii=False)}")

            os.makedirs(out_dir, exist_ok=True)
            long_summary_results_path = summary_cache_path
            long_results_path = _save_long_results(out_dir=out_dir, version=args.version, long_results=long_results)
            artifact_paths = _build_subset_artifacts(
                dataset=args.dataset,
                subset=subset,
                long_rows=long_rows,
                split_rows=split_rows,
                long_results=long_results,
                version=args.version,
                out_dir=out_dir,
                algo4_meta=algo4_meta,
                path_desc_payload=path_desc_payload,
                category_desc_key_terms_topk=args.category_desc_key_terms_topk,
                category_desc_max_chars=args.category_desc_max_chars,
                category_desc_child_anchor_topk=args.category_desc_child_anchor_topk,
                category_desc_child_anchor_min_ratio=args.category_desc_child_anchor_min_ratio,
                category_desc_tail_anchor_topk=args.category_desc_tail_anchor_topk,
            )

            report = artifact_paths["report"]
            report_meta = report["meta"] if isinstance(report, dict) else {}
            report_counts = report["counts"] if isinstance(report, dict) else {}
            print(
                "[Done] subset={subset} tree_nodes={tree_nodes} tree_edges={tree_edges} tree={tree_path}".format(
                    subset=subset,
                    tree_nodes=report_counts.get("num_tree_nodes", report_counts.get("num_dag_nodes", -1)),
                    tree_edges=report_counts.get("num_tree_edges", report_counts.get("num_dag_edges", -1)),
                    tree_path=artifact_paths["runtime_tree_path"],
                )
            )
            print(f"[Meta] {json.dumps(report_meta, ensure_ascii=False)}")
            print(f"[LongDocSummaries] {long_summary_results_path}")
            print(f"[LongDocPaths] {long_results_path}")
    finally:
        summary_router.close_all()
        if cluster_router is not summary_router:
            cluster_router.close_all()


if __name__ == "__main__":
    main()
