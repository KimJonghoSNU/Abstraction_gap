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
    "Your task is to analyze ONE informational passage and generate a category-based hierarchy "
    "hierarchical retrieval keywords, strictly following the 5-level rubric.\n\n"
    "Important constraints:\n"
    "- Level 1 must be a support role/category label (e.g. Theory, Entity, Example, Evidence, Principle, Mechanism).\n"
    "- Levels 2-5 gradually focus contents that refine the passage semantics under the Level-1 role.\n"
    "- Output must be actionable search phrases.\n"
    "- If the passage reasonably supports two paths, provide one alternate path.\n\n"
    "Keyword Generation Rules (5 Levels):\n"
    "Level 1: 1-2 words, broadest support role/category label.\n"
    "Level 2: 3-4 words, general content topic within that role (usually with title).\n"
    "Level 3: 4-6 words, key content concepts/themes.\n"
    "Level 4: 7-10 words, concise content summary.\n"
    "Level 5: 11-20 words, most specific content summary.\n\n"
    "Output JSON only:\n"
    "{{\n"
    "  \"passage_id\": \"{passage_id}\",\n"
    "  \"hierarchical_keywords\": [\"L1\", \"L2\", \"L3\", \"L4\", \"L5\"],\n"
    "  \"alternate_hierarchical_keywords\": [[\"L1\", \"L2\", \"L3\", \"L4\", \"L5\"]]\n"
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
    "Task: Group input summary items into coherent categories for top-down tree partitioning.\n"
    "Goal constraints:\n"
    "- This stage performs category typing; infer SUPPORT ROLE / FUNCTION from summary semantics.\n"
    "- Prefer cluster name that reflects the underlying support role concept, such as Theory/Principle, Method/Procedure, Entity/Definition, Evidence/Observation, Mechanism/Causality.\n"
    "- It is acceptable to keep a larger cluster if splitting would lose information fidelity.\n"
    "- Category-first grouping by support role/conceptual function, not by its topics (e.g. group mathematical 'theory' with domain-specific 'theory' under 'theory').\n"
    "- Avoid one dominant catch-all cluster when possible. But if it seems you lose information due to balanced cluster, you should add one more cluster not to lose the information \n"
    "- Use item counts as importance hints.\n"
    "- Try to maximize cluster count within range without sacrificing semantic coherence.\n"
    "- Produce between {min_k} and {max_k} clusters.\n"
    "- Every input summary_id must be assigned at least once.\n"
    "- Multi-assignment is allowed when semantically justified.\n\n"
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
    1: "General Role",
    2: "General Supporting Category",
    3: "Core Supporting Role Concept",
    4: "This passage provides broad support for related information needs",
    5: "This passage provides specific supporting details useful for retrieval and answer grounding",
}

LEVEL_WORD_MAX = {1: 2, 2: 4, 3: 6, 4: 10, 5: 20}


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
    max_words = LEVEL_WORD_MAX[level]
    if len(words) > max_words:
        words = words[:max_words]
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
        f"ROOT Node: category-first top-down projection tree for subset={subset}, version={version}. "
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
    covered: Set[str] = set()
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
                if sid in local_seen:
                    continue
                local_seen.add(sid)
                covered.add(sid)
                mapped_ids.append(sid)
        if (not name) and (not desc) and (not mapped_ids):
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

    missing = [sid for sid in sorted(valid_summary_ids) if sid not in covered]
    if missing:
        # Intent: keep Algorithm-4 style full coverage by forcing every summary id to belong to at least one cluster.
        bucket_sizes = [len(x.summary_ids) for x in out]
        for sid in missing:
            min_idx = min(range(len(bucket_sizes)), key=lambda i: bucket_sizes[i])
            if sid not in out[min_idx].summary_ids:
                out[min_idx].summary_ids.append(sid)
            bucket_sizes[min_idx] += 1
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
    best_paths_by_doc: Dict[str, List[List[str]]] = {x.doc_id: [] for x in summary_results}
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
                min_k=2,
                max_k=min(max(2, len(unique_rows)), max(2, int(max_branching))),
            )
            llm_out = generate_fn([prompt])
            llm_cluster_calls += 1
            parsed = _parse_cluster_output(
                text=llm_out[0] if llm_out else "",
                valid_summary_ids=set([str(x["summary_id"]) for x in unique_rows]),
                summary_text_to_ids=summary_text_to_ids,
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
                    )
                    if parsed is not None:
                        break
            if parsed is None:
                parsed = _fallback_cluster_output(
                    summary_rows=unique_rows,
                    max_k=min(max(2, len(unique_rows)), max(2, int(max_branching))),
                )

            sid_to_cluster_idxs: Dict[str, List[int]] = defaultdict(list)
            for idx, c in enumerate(parsed):
                for sid in c.summary_ids:
                    if idx not in sid_to_cluster_idxs[sid]:
                        sid_to_cluster_idxs[sid].append(idx)
            if not sid_to_cluster_idxs:
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
                candidate_idxs = sid_to_cluster_idxs.get(sid, [])
                if not candidate_idxs:
                    continue
                for cidx in candidate_idxs:
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
                    existing_paths = best_paths_by_doc.setdefault(doc_id, [])
                    if child_path not in existing_paths:
                        existing_paths.append(list(child_path))
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
        seed_paths = [list(path) for path in best_paths_by_doc.get(item.doc_id, [])]
        # Intent: keep fixed 5-level paths for runtime compatibility while retaining multiple parent memberships for DAG behavior.
        final_paths: List[List[str]] = []
        seen_final_paths: Set[str] = set()
        if not seed_paths:
            seed_paths = [[]]
        for seed_path in seed_paths:
            final_path: List[str] = []
            for idx in range(5):
                if idx < len(seed_path):
                    final_path.append(_normalize_label(seed_path[idx], idx + 1))
                    continue
                fallback = item.summaries[idx] if idx < len(item.summaries) else DEFAULT_LEVEL_LABELS[idx + 1]
                final_path.append(_normalize_label(fallback, idx + 1))
            path_key = "||".join(final_path)
            if path_key in seen_final_paths:
                continue
            seen_final_paths.add(path_key)
            final_paths.append(final_path)
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
        rec = {
            "path": list(path),
            "depth": len(path),
            "is_leaf": len(child) == 0,
            "num_children": len(child),
            "num_leaves": int(node.get("num_leaves", 1)),
            "id": node.get("id"),
            "desc": node.get("desc", ""),
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
    leaf_parent_cap: int,
    version: str,
    out_dir: str,
    algo4_meta: Optional[Dict[str, object]] = None,
    path_desc_payload: Optional[Dict[str, Dict[str, object]]] = None,
    category_desc_key_terms_topk: int = 3,
    category_desc_max_chars: int = 320,
    category_desc_child_anchor_topk: int = 3,
    category_desc_child_anchor_min_ratio: float = 0.12,
    category_desc_tail_anchor_topk: int = 2,
) -> Dict[str, object]:
    nodes: Dict[str, Dict[str, object]] = {}
    edge_weights: Dict[Tuple[str, str], float] = defaultdict(float)
    edge_types: Dict[Tuple[str, str], str] = {}
    incoming: Dict[str, Set[str]] = defaultdict(set)
    outgoing: Dict[str, Set[str]] = defaultdict(set)

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

    split_content_by_id = {row["id"]: row["content"] for row in split_rows}

    long_doc_l5_candidates: Dict[str, List[Tuple[str, float, int]]] = defaultdict(list)
    long_doc_title_key: Dict[str, str] = {}
    node_desc_votes: Dict[str, Counter] = defaultdict(Counter)
    node_term_votes: Dict[str, Counter] = defaultdict(Counter)

    def ensure_internal_node(level: int, label: str) -> str:
        node_id = _make_internal_node_id(level, label)
        if node_id not in nodes:
            nodes[node_id] = {
                "id": node_id,
                "level": level,
                "kind": "category",
                "label": label,
                "display_id": f"[L{level}] {label}",
                "desc": "",
                "long_support_ids": set(),
                "split_support_ids": set(),
            }
        return node_id

    def ensure_leaf_node(doc_id: str, content: str) -> str:
        if doc_id not in nodes:
            nodes[doc_id] = {
                "id": doc_id,
                "level": 6,
                "kind": "leaf",
                "label": doc_id,
                "display_id": doc_id,
                "desc": content,
                "long_support_ids": set(),
                "split_support_ids": {doc_id},
            }
        return doc_id

    def add_edge(parent_id: str, child_id: str, weight: float, edge_type: str) -> None:
        key = (parent_id, child_id)
        edge_weights[key] += float(weight)
        edge_types[key] = edge_type
        incoming[child_id].add(parent_id)
        outgoing[parent_id].add(child_id)

    for result in long_results:
        long_doc_title_key[result.doc_id] = _normalize_key(result.website_title)
        path_score_base = 1.0
        for path_idx, path in enumerate(result.paths):
            parent_id = root_id
            for level, label in enumerate(path, start=1):
                node_id = ensure_internal_node(level, label)
                add_edge(parent_id, node_id, weight=path_score_base, edge_type="category")
                nodes[node_id]["long_support_ids"].add(result.doc_id)
                prefix_key = "||".join(path[:level])
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
            # Intent: slight penalty keeps alternate paths for DAG coverage but preserves primary path preference.
            score = 1.0 - (0.05 * path_idx)
            long_doc_l5_candidates[result.doc_id].append((parent_id, score, path_idx))
            path_score_base = max(0.5, score)

    long_doc_by_prefix: Dict[str, List[str]] = defaultdict(list)
    for result in long_results:
        long_doc_by_prefix[_split_prefix(result.doc_id)].append(result.doc_id)

    leaf_membership_rows: List[Dict[str, object]] = []
    leaf_progress = _make_progress(total=len(split_rows), desc="Attach split leaves", unit="leaf")
    try:
        for row in split_rows:
            split_id = row["id"]
            split_content = row["content"]
            split_pref = _split_prefix(split_id)
            split_tail = split_id.split("/", 1)[-1] if "/" in split_id else split_id
            split_base_key = _normalize_key(_website_title_from_doc_id(_strip_chunk_suffix(split_tail)))

            long_candidates = long_doc_by_prefix.get(split_pref, [])
            scored_parent_candidates: List[Tuple[float, str, str]] = []
            for long_doc_id in long_candidates:
                title_key = long_doc_title_key.get(long_doc_id, "")
                sim = _similarity(split_base_key, title_key)
                for l5_node_id, base_score, path_idx in long_doc_l5_candidates.get(long_doc_id, []):
                    cand_score = (sim * 0.9) + (base_score * 0.1) - (0.01 * path_idx)
                    scored_parent_candidates.append((cand_score, l5_node_id, long_doc_id))

            scored_parent_candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
            selected_parents: List[Tuple[float, str, str]] = []
            seen_l5: Set[str] = set()
            for item in scored_parent_candidates:
                if item[1] in seen_l5:
                    continue
                seen_l5.add(item[1])
                selected_parents.append(item)
                if len(selected_parents) >= max(1, int(leaf_parent_cap)):
                    break

            if not selected_parents:
                # Intent: keep every split document reachable by attaching to at least one category parent in the same subset.
                fallback_parent = None
                if long_candidates:
                    first_long = long_candidates[0]
                    l5_options = long_doc_l5_candidates.get(first_long, [])
                    if l5_options:
                        fallback_parent = l5_options[0][0]
                if fallback_parent is None:
                    fallback_parent = root_id
                selected_parents = [(0.0, fallback_parent, long_candidates[0] if long_candidates else "")]

            leaf_node_id = ensure_leaf_node(split_id, split_content)
            for score, parent_l5, parent_long_doc_id in selected_parents:
                add_edge(parent_l5, leaf_node_id, weight=max(0.01, score), edge_type="leaf_attach")
                nodes[parent_l5]["split_support_ids"].add(split_id)
                nodes[leaf_node_id]["long_support_ids"].add(parent_long_doc_id)

            leaf_membership_rows.append(
                {
                    "doc_id": split_id,
                    "selected_parent_node_ids": [x[1] for x in selected_parents],
                    "selected_parent_long_doc_ids": [x[2] for x in selected_parents],
                    "selected_scores": [round(float(x[0]), 6) for x in selected_parents],
                }
            )
            leaf_progress.update(1)
    finally:
        leaf_progress.close()

    parent_choice: Dict[str, str] = {}
    for child_id, parents in incoming.items():
        if child_id == root_id:
            continue
        best_parent = None
        best_tuple = None
        for parent_id in parents:
            key = (parent_id, child_id)
            weight = edge_weights.get(key, 0.0)
            parent_level = int(nodes.get(parent_id, {}).get("level", -1))
            rank_key = (weight, -parent_level, parent_id)
            if (best_tuple is None) or (rank_key > best_tuple):
                best_tuple = rank_key
                best_parent = parent_id
        if best_parent is None:
            best_parent = root_id
        parent_choice[child_id] = best_parent

    proj_children: Dict[str, List[str]] = defaultdict(list)
    for child_id, parent_id in parent_choice.items():
        proj_children[parent_id].append(child_id)

    visited: Set[str] = set()
    queue: List[str] = [root_id]
    while queue:
        cur = queue.pop(0)
        if cur in visited:
            continue
        visited.add(cur)
        for nxt in proj_children.get(cur, []):
            if nxt not in visited:
                queue.append(nxt)

    for node_id in nodes:
        if node_id == root_id:
            continue
        if node_id in visited:
            continue
        proj_children[root_id].append(node_id)
        parent_choice[node_id] = root_id

    projection_leaf_count_cache: Dict[str, int] = {}

    def _projection_leaf_count(node_id: str) -> int:
        cached = projection_leaf_count_cache.get(node_id)
        if cached is not None:
            return int(cached)
        node = nodes.get(node_id, {})
        if str(node.get("kind", "")) == "leaf":
            projection_leaf_count_cache[node_id] = 1
            return 1
        child_ids = proj_children.get(node_id, [])
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
            topk = max(1, int(category_desc_key_terms_topk))
            top_terms = [term for term, _ in term_counter.most_common(topk)]
            # Intent: preserve lexical anchors in upper-node desc so run_original traversal prompts do not lose key terms.
            # Do not. top terms are already in base_desc.
            if top_terms:
                composed_desc = f"{base_desc}"  #Key terms: {', '.join(top_terms)}."
            else:
                composed_desc = base_desc
            if int(node["level"]) <= 3:
                child_category_scores: List[Tuple[int, str]] = []
                for child_id in proj_children.get(node_id, []):
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
                    topk_child = max(1, int(category_desc_child_anchor_topk))
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
                        # Intent: promote substantial child topics into upper-level desc to preserve minority-but-important anchors (e.g., Poisson) during traversal.
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
        children_ids = sorted(proj_children.get(node_id, []), key=child_sort_key)
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
                "num_children": len(outgoing.get(node_id, set())),
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
                "is_projection_parent": parent_choice.get(child_id) == parent_id,
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

    report = {
        "meta": {
            "dataset": dataset,
            "subset": subset,
            "version": version,
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
            "leaf_parent_cap": int(leaf_parent_cap),
            "builder": "topdown_algo4",
        },
        "counts": {
            "num_long_documents": len(long_rows),
            "num_split_documents": len(split_rows),
            "num_dag_nodes": len(dag_nodes_out),
            "num_dag_edges": len(dag_edges_out),
            "num_projection_nodes": len(dag_nodes_out),
            "num_projection_edges": max(0, len(dag_nodes_out) - 1),
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
            "Build category-first top-down DAG with Algorithm-4 style queue partitioning "
            "from BRIGHT long_documents and export runtime-compatible projection tree."
        )
    )
    parser.add_argument("--dataset", type=str, default="BRIGHT")
    parser.add_argument("--subset", type=str, required=True, help="Subset name, comma list, or 'all'")
    parser.add_argument("--data_dir", type=str, default="data/BRIGHT")
    parser.add_argument("--trees_root", type=str, default="trees")
    parser.add_argument("--version", type=str, default="v2")
    parser.add_argument("--llm", type=str, required=True, help="Snowflake Cortex model name")
    parser.add_argument("--env_file", type=str, default=".env")
    parser.add_argument("--snowflake_request_timeout", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--max_input_prompt_chars", type=int, default=120000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_desc_words", type=int, default=4096)
    parser.add_argument("--parse_retry_max", type=int, default=1)
    parser.add_argument("--leaf_parent_cap", type=int, default=2)
    parser.add_argument("--max_branching", type=int, default=20)
    parser.add_argument("--max_partition_depth", type=int, default=5)
    parser.add_argument("--cluster_prompt_word_budget", type=int, default=12000)
    parser.add_argument("--category_desc_key_terms_topk", type=int, default=3)
    parser.add_argument("--category_desc_max_chars", type=int, default=320)
    parser.add_argument("--category_desc_child_anchor_topk", type=int, default=3)
    parser.add_argument("--category_desc_child_anchor_min_ratio", type=float, default=0.12)
    parser.add_argument("--category_desc_tail_anchor_topk", type=int, default=2)
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
    router = SnowflakeCortexRouter(
        account_configs=account_configs,
        model=args.llm,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        request_timeout=args.snowflake_request_timeout,
        max_input_prompt_chars=args.max_input_prompt_chars,
    )

    def generate_fn(prompts: List[str]) -> List[str]:
        return [router.complete(prompt) for prompt in prompts]

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
                    generate_fn=generate_fn,
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
                generate_fn=generate_fn,
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
                leaf_parent_cap=args.leaf_parent_cap,
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
                "[Done] subset={subset} dag_nodes={dag_nodes} dag_edges={dag_edges} tree={tree_path}".format(
                    subset=subset,
                    dag_nodes=report_counts.get("num_dag_nodes", -1),
                    dag_edges=report_counts.get("num_dag_edges", -1),
                    tree_path=artifact_paths["runtime_tree_path"],
                )
            )
            print(f"[Meta] {json.dumps(report_meta, ensure_ascii=False)}")
            print(f"[LongDocSummaries] {long_summary_results_path}")
            print(f"[LongDocPaths] {long_results_path}")
    finally:
        router.close_all()


if __name__ == "__main__":
    main()
