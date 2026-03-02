import argparse
import datetime
import json
import os
import re
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd
from json_repair import repair_json
from vllm import LLM, SamplingParams


V2_BASE_CATEGORY_SEEDS = [
    ("Theory", "Domain Theory"),
    ("Theory", "Cross-Domain Theory"),
    ("Theory", "Mathematical Theory"),
    ("Theory", "Concept Definition"),
    ("Rule", "Normative Rule"),
    ("Evidence", "Causal Evidence"),
    ("Method", "Procedural Method"),
    ("Resource", "Technical Resource"),
    ("Example", "Worked Example"),
    ("Background", "General Background"),
]
V2_DEFAULT_LEVEL1 = "Background"
V2_DEFAULT_LEVEL2 = "General Background"
V2_MAX_LEVEL1_ITEMS = 7
V2_MAX_LEVEL2_ITEMS = 10

V3_BASE_CATEGORY_SEEDS = [
    ("Theory", "Theory"),
    ("Method/Protocol", "Method/Protocol"),
    ("Evidence/Analogy/Example", "Evidence/Analogy/Example"),
]
V3_DEFAULT_LEVEL1 = "Evidence/Analogy/Example"
V3_DEFAULT_LEVEL2 = "Evidence/Analogy/Example"
V3_MAX_LEVEL1_ITEMS = 3
V3_MAX_LEVEL2_ITEMS = 10

DATASET_RELEVANCE_DEFINITION = {
    "biology": (
        "Given a query (biology post) and a document (passage), the document is relevant to the query "
        "if the critical concepts or theories discussed in the document can provide references for "
        "domain experts to draft an answer to the query."
    ),
    "earth_science": (
        "Given a query (earth science post) and a document (passage), the document is relevant to the query "
        "if the critical concepts or theories discussed in the document can provide references for "
        "domain experts to draft an answer to the query."
    ),
    "economics": (
        "Given a query (economics post) and a document (passage), the document is relevant to the query "
        "if the critical concepts or theories discussed in the document can provide references for "
        "domain experts to draft an answer to the query."
    ),
    "psychology": (
        "Given a query (psychology post) and a document (passage), the document is relevant to the query "
        "if the critical concepts or theories discussed in the document can provide references for "
        "domain experts to draft an answer to the query."
    ),
    "robotics": (
        "Given a query (robotics post) and a document (passage), the document is relevant to the query "
        "if the critical concepts or theories discussed in the document can provide references for "
        "domain experts to draft an answer to the query."
    ),
    "stackoverflow": (
        "Given a query (Stack Overflow post) and a document (passage), the document is relevant to the query "
        "if the critical concepts or theories discussed in the document can provide references for "
        "domain experts to draft an answer to the query."
    ),
    "sustainable_living": (
        "Given a query (sustainable living post) and a document (passage), the document is relevant to the query "
        "if the critical concepts or theories discussed in the document can provide references for "
        "domain experts to draft an answer to the query."
    ),
    "leetcode": (
        "Given a query (LeetCode problem) and a document (coding problem solution), the document is relevant "
        "to the query if the underlying algorithms or data structures used in the document can provide "
        "helpful insights for solving the problem in the query."
    ),
    "pony": (
        "Given a query (Pony coding instruction) and a document (Pony documentation passage), the document is "
        "relevant to the query if the Pony syntax described in the document is necessary for beginners with "
        "no prior knowledge of Pony to complete the coding instruction in the query."
    ),
    "aops": (
        "Given a query (math problem) and a document (math problem solution), the document is relevant to the "
        "query if the theorems used in the document can provide helpful insights for solving the problem in the query."
    ),
    "theoremqa_questions": (
        "Given a query (math problem) and a document (math problem solution), the document is relevant to the "
        "query if the theorems used in the document can provide helpful insights for solving the problem in the query."
    ),
    "theoremqa_theorems": (
        "Given a query (math problem) and a document (math-related passage), the document is relevant to the "
        "query if the theorem described in the document can help solve the problem in the query."
    ),
}

SUBSET_DOMAIN_LABEL = {
    "biology": "Biology",
    "earth_science": "Earth Science",
    "economics": "Economics",
    "psychology": "Psychology",
    "robotics": "Robotics",
    "stackoverflow": "Software Engineering",
    "sustainable_living": "Sustainable Living",
    "leetcode": "Algorithmic Programming",
    "pony": "Programming Language Syntax",
    "aops": "Mathematics",
    "theoremqa_questions": "Mathematics",
    "theoremqa_theorems": "Mathematics",
}

V2_PROMPT_TEMPLATE = (
    "You are assigning hierarchical category keywords for one document.\n\n"
    "Subset: {subset}\n"
    "Subset Domain: {subset_domain}\n"
    "Relevance Definition:\n{relevance_definition}\n\n"
    "Goal:\n"
    "- Output retrieval-role category keywords (not content/topic keywords) in 2 levels.\n"
    "- A document may have multiple categories at each level.\n"
    "- The category should describe document role for retrieval support.\n"
    "- Example: a passage explaining Darwin's evolution should map to Theory-related category, not \"Evolution\".\n"
    "Level-1 Category (broad, 1-2 words): choose one or more from:\n"
    "- Theory\n"
    "- Rule\n"
    "- Method\n"
    "- Evidence\n"
    "- Resource\n"
    "- Example\n"
    "- Background\n\n"
    "Level-2 Category (more specific, 2-4 words):\n"
    "- Must be abstract and reusable across many documents.\n"
    "- Must align with one of selected level-1 categories.\n"
    "- You may output multiple level-2 categories when the document serves multiple citation roles.\n"
    "- Avoid concrete topics/entities.\n"
    "- Bad: Evolution, Maxwell-Boltzmann Distribution, DSM-5\n"
    "- Good: Domain Theory, Causal Evidence, Procedural Method, Technical Resource\n\n"
    "Known Level-2 Categories (canonical):\n"
    "{known_labels_block}\n\n"
    "If your new level-2 category can absorb old labels, list those old labels in merge_from.\n\n"
    "{website_title}\n"
    "Document Snippet:\n{doc_desc}\n\n"
    "Output JSON only:\n"
    "{{\n"
    "  \"level1_categories\": [\"one or more labels\"],\n"
    "  \"level2_categories\": [\"one or more labels\"],\n"
    "  \"canonical_level2\": [\"optional existing known level-2 labels if reused\"],\n"
    "  \"merge_from\": [\"optional known labels to alias into category\"]\n"
    "}}"
)

V3_PROMPT_TEMPLATE = (
    "You are assigning hierarchical category keywords for one document.\n\n"
    "Subset: {subset}\n"
    "Subset Domain: {subset_domain}\n"
    "Relevance Definition:\n{relevance_definition}\n\n"
    "Goal:\n"
    "- Output retrieval-role category keywords (not content/topic keywords) in 2 levels.\n"
    "- A document may have multiple categories at each level.\n"
    "- The category should describe document role for retrieval support.\n"
    "- Example: a passage about Darwin's evolution should map by role, not by topic name.\n"
    "- Reuse existing categories when possible; create a new one only if truly needed.\n\n"
    "Level-1 Category (broad role): choose one or more from:\n"
    "- Theory\n"
    "- Method/Protocol\n"
    "- Evidence/Analogy/Example\n\n"
    "Level-2 Category (more specific, 2-4 words):\n"
    "- Must be role-abstract and reusable across many documents.\n"
    "- Must align with one of selected level-1 categories.\n"
    "- You may output multiple level-2 categories when the document serves multiple retrieval roles.\n"
    "- Avoid concrete subset-specific terms.\n"
    "- Bad: Evolution Theory, Behavioral Theory, Maxwell-Boltzmann Example\n"
    "- Good: Theory, Method/Protocol, Evidence/Analogy/Example\n\n"
    "Gap rule:\n"
    "- If 3 level-1 categories are insufficient to express role distinction, set is_gap=true.\n"
    "- gap_candidates must be role taxonomy labels, not topic labels.\n"
    "- If gap reason is topic-based, set is_gap=false and leave gap_candidates empty.\n\n"
    "Known Level-2 Categories (canonical):\n"
    "{known_labels_block}\n\n"
    "If your new level-2 category can absorb old labels, list those old labels in merge_from.\n\n"
    "{website_title}\n"
    "Document Snippet:\n{doc_desc}\n\n"
    "Output JSON only:\n"
    "{{\n"
    "  \"level1_categories\": [\"one or more of [Theory, Method/Protocol, Evidence/Analogy/Example]\"],\n"
    "  \"level2_categories\": [\"one or more labels\"],\n"
    "  \"canonical_level2\": [\"optional existing known level-2 labels if reused\"],\n"
    "  \"merge_from\": [\"optional known labels to alias into category\"],\n"
    "  \"is_gap\": false,\n"
    "  \"gap_reason\": \"\",\n"
    "  \"gap_candidates\": [\"optional role labels\"]\n"
    "}}"
)

V2_CATEGORY_NORMALIZATION = {
    "domaintheory": "Domain Theory",
    "crossdomaintheory": "Cross-Domain Theory",
    "mathematicaltheory": "Mathematical Theory",
    "maththeory": "Mathematical Theory",
    "statisticaltheory": "Mathematical Theory",
    "probabilistictheory": "Mathematical Theory",
    "conceptdefinition": "Concept Definition",
    "normativerule": "Normative Rule",
    "method": "Procedural Method",
    "proceduralmethod": "Procedural Method",
    "technicalresource": "Technical Resource",
    "resource": "Technical Resource",
    "evidence": "Causal Evidence",
    "causalevidence": "Causal Evidence",
    "example": "Worked Example",
    "workedexample": "Worked Example",
    "background": "General Background",
    "generalbackground": "General Background",
    "auxiliarycontext": "General Background",
    "backgroundcontext": "General Background",
    # Intent: v2에서도 Entity를 제거하고 Background로 흡수한다.
    "entity": "General Background",
}

V3_CATEGORY_NORMALIZATION = {
    "theory": "Theory",
    "domaintheory": "Theory",
    "crossdomaintheory": "Theory",
    "mathematicaltheory": "Theory",
    "maththeory": "Theory",
    "statisticaltheory": "Theory",
    "probabilistictheory": "Theory",
    "conceptdefinition": "Theory",
    "normativerule": "Theory",
    "method": "Method/Protocol",
    "proceduralmethod": "Method/Protocol",
    "technicalresource": "Method/Protocol",
    "resource": "Method/Protocol",
    "protocol": "Method/Protocol",
    "evidence": "Evidence/Analogy/Example",
    "causalevidence": "Evidence/Analogy/Example",
    "example": "Evidence/Analogy/Example",
    "workedexample": "Evidence/Analogy/Example",
    "analogy": "Evidence/Analogy/Example",
    "background": "Evidence/Analogy/Example",
    "generalbackground": "Evidence/Analogy/Example",
    "auxiliarycontext": "Evidence/Analogy/Example",
    "backgroundcontext": "Evidence/Analogy/Example",
    # Intent: remove Entity role from v2/v3 taxonomy by folding legacy entity outputs into evidence-like support role.
    "entity": "Evidence/Analogy/Example",
}

V2_LEVEL1_NORMALIZATION = {
    "theory": "Theory",
    "domain": "Theory",
    "definition": "Theory",
    "concept": "Theory",
    "rule": "Rule",
    "law": "Rule",
    "constraint": "Rule",
    "method": "Method",
    "protocol": "Method",
    "procedure": "Method",
    "mechanism": "Method",
    "evidence": "Evidence",
    "causal": "Evidence",
    "analogy": "Evidence",
    "resource": "Resource",
    "reference": "Resource",
    "example": "Example",
    "background": "Background",
    "context": "Background",
    # Intent: v2에서도 Entity를 허용 라벨에서 제거해 Background로 수렴시킨다.
    "entity": "Background",
}

V3_LEVEL1_NORMALIZATION = {
    "theory": "Theory",
    "domain": "Theory",
    # Intent: fold legacy definition/rule outputs into Theory for v3 simplified role taxonomy.
    "definition": "Theory",
    "concept": "Theory",
    "rule": "Theory",
    "law": "Theory",
    "constraint": "Theory",
    "method": "Method/Protocol",
    "protocol": "Method/Protocol",
    "procedure": "Method/Protocol",
    "mechanism": "Method/Protocol",
    "evidence": "Evidence/Analogy/Example",
    "causal": "Evidence/Analogy/Example",
    "analogy": "Evidence/Analogy/Example",
    "example": "Evidence/Analogy/Example",
    "resource": "Evidence/Analogy/Example",
    "reference": "Evidence/Analogy/Example",
    "background": "Evidence/Analogy/Example",
    "context": "Evidence/Analogy/Example",
    "entity": "Evidence/Analogy/Example",
}

# Runtime-selected taxonomy config (set in main via --category_version).
CATEGORY_VERSION = "v3"
BASE_CATEGORY_SEEDS = V3_BASE_CATEGORY_SEEDS
DEFAULT_LEVEL1 = V3_DEFAULT_LEVEL1
DEFAULT_LEVEL2 = V3_DEFAULT_LEVEL2
MAX_LEVEL1_ITEMS = V3_MAX_LEVEL1_ITEMS
MAX_LEVEL2_ITEMS = V3_MAX_LEVEL2_ITEMS
PROMPT_TEMPLATE = V3_PROMPT_TEMPLATE
CATEGORY_NORMALIZATION = V3_CATEGORY_NORMALIZATION
LEVEL1_NORMALIZATION = V3_LEVEL1_NORMALIZATION


def _set_category_version(category_version: str) -> str:
    global CATEGORY_VERSION
    global BASE_CATEGORY_SEEDS
    global DEFAULT_LEVEL1
    global DEFAULT_LEVEL2
    global MAX_LEVEL1_ITEMS
    global MAX_LEVEL2_ITEMS
    global PROMPT_TEMPLATE
    global CATEGORY_NORMALIZATION
    global LEVEL1_NORMALIZATION

    version = str(category_version or "v3").strip().lower()
    if version not in {"v2", "v3"}:
        raise ValueError(f"Unsupported category_version={category_version!r}. Use one of: v2, v3")

    if version == "v2":
        # Intent: v2는 기존(엔티티 제거) taxonomy를 사용해 v3와 실험 축을 명확히 분리한다.
        BASE_CATEGORY_SEEDS = V2_BASE_CATEGORY_SEEDS
        DEFAULT_LEVEL1 = V2_DEFAULT_LEVEL1
        DEFAULT_LEVEL2 = V2_DEFAULT_LEVEL2
        MAX_LEVEL1_ITEMS = V2_MAX_LEVEL1_ITEMS
        MAX_LEVEL2_ITEMS = V2_MAX_LEVEL2_ITEMS
        PROMPT_TEMPLATE = V2_PROMPT_TEMPLATE
        CATEGORY_NORMALIZATION = V2_CATEGORY_NORMALIZATION
        LEVEL1_NORMALIZATION = V2_LEVEL1_NORMALIZATION
    else:
        BASE_CATEGORY_SEEDS = V3_BASE_CATEGORY_SEEDS
        DEFAULT_LEVEL1 = V3_DEFAULT_LEVEL1
        DEFAULT_LEVEL2 = V3_DEFAULT_LEVEL2
        MAX_LEVEL1_ITEMS = V3_MAX_LEVEL1_ITEMS
        MAX_LEVEL2_ITEMS = V3_MAX_LEVEL2_ITEMS
        PROMPT_TEMPLATE = V3_PROMPT_TEMPLATE
        CATEGORY_NORMALIZATION = V3_CATEGORY_NORMALIZATION
        LEVEL1_NORMALIZATION = V3_LEVEL1_NORMALIZATION

    CATEGORY_VERSION = version
    return version


def _default_long_documents_path(dataset: str, subset: str) -> str:
    return os.path.join("data", dataset, "long_documents", f"{subset}-00000-of-00001.parquet")


def _default_documents_path(dataset: str, subset: str) -> str:
    return os.path.join("data", dataset, "documents", f"{subset}-00000-of-00001.parquet")


def _default_output_path(dataset: str, subset: str, prompt_name: str) -> str:
    return os.path.join("trees", dataset, subset, f"document_categories_{prompt_name}.jsonl")


def _default_registry_path(dataset: str, subset: str, prompt_name: str) -> str:
    return os.path.join("trees", dataset, subset, f"category_registry_{prompt_name}.json")


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _normalize_label_key(label: str) -> str:
    return "".join(ch for ch in str(label or "").lower() if ch.isalnum())


def _title_case_words(text: str, max_words: int = 3) -> str:
    words = [w for w in re.split(r"\s+", str(text or "").strip()) if w]
    if not words:
        return ""
    return " ".join(words[:max_words]).title()


def _truncate_words(text: str, max_words: int) -> str:
    if max_words <= 0:
        return ""
    words = str(text or "").split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def _count_prompt_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _build_guarded_prompt(
    *,
    subset: str,
    website_title: str,
    doc_desc: str,
    known_labels_block: str,
    tokenizer,
    prompt_token_limit: int,
) -> Tuple[str, bool]:
    prompt = _build_prompt(
        subset=subset,
        website_title=website_title,
        doc_desc=doc_desc,
        known_labels_block=known_labels_block,
    )
    if _count_prompt_tokens(prompt, tokenizer) <= prompt_token_limit:
        return prompt, False

    words = str(doc_desc or "").split()
    lo = 0
    hi = len(words)
    best_prompt = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        cand_desc = " ".join(words[:mid])
        cand_prompt = _build_prompt(
            subset=subset,
            website_title=website_title,
            doc_desc=cand_desc,
            known_labels_block=known_labels_block,
        )
        cand_len = _count_prompt_tokens(cand_prompt, tokenizer)
        if cand_len <= prompt_token_limit:
            best_prompt = cand_prompt
            lo = mid + 1
        else:
            hi = mid - 1

    if best_prompt:
        return best_prompt, True

    empty_prompt = _build_prompt(
        subset=subset,
        website_title=website_title,
        doc_desc="",
        known_labels_block=known_labels_block,
    )
    if _count_prompt_tokens(empty_prompt, tokenizer) <= prompt_token_limit:
        return empty_prompt, True

    # Intent: final hard guard to prevent vLLM max-length crash even when prompt prefix alone is too long.
    ids = tokenizer.encode(empty_prompt, add_special_tokens=False)
    trimmed = tokenizer.decode(ids[:prompt_token_limit], skip_special_tokens=True)
    return trimmed, True


def _iter_document_units_from_parquet(parquet_path: str) -> Iterable[Tuple[int, Dict]]:
    df = pd.read_parquet(parquet_path, columns=["id", "content"])
    for doc_idx, row in enumerate(df.itertuples(index=False)):
        doc_id = str(getattr(row, "id", "")).strip()
        if not doc_id:
            continue
        yield doc_idx, {
            "doc_id": doc_id,
            "desc": str(getattr(row, "content", "")),
            "is_long_document": True,
        }


def _website_title_from_doc_id(doc_id: str) -> str:
    related_query, document_title = doc_id.split("/")[:2]
    return f"Related query: {related_query}, Document title: {document_title}"


def _resolve_input_parquet_path(
    *,
    long_documents_path: str,
    documents_path: str,
) -> Tuple[str, str]:
    if os.path.exists(long_documents_path):
        return long_documents_path, "long_documents"
    if os.path.exists(documents_path):
        # Intent: fall back to documents split when long_documents is unavailable for a subset.
        return documents_path, "documents"
    raise FileNotFoundError(
        "Missing both long_documents and documents parquet files. "
        f"Checked: {long_documents_path} and {documents_path}. "
        "Download with: python -c \"from huggingface_hub import snapshot_download; "
        "snapshot_download(repo_id='xlangai/BRIGHT', repo_type='dataset', "
        "allow_patterns=['long_documents/*', 'documents/*'], local_dir='data/BRIGHT')\""
    )


def _clean_json_candidate(text: str) -> str:
    cleaned = text.split("</think>\n")[-1].strip()
    if "```" in cleaned:
        try:
            parts = cleaned.split("```")
            fenced = [parts[i] for i in range(1, len(parts), 2)]
            if fenced:
                cleaned = fenced[-1].strip()
        except Exception:
            pass
    return cleaned.strip()


def _normalize_list(value: object, max_items: int) -> List[str]:
    if isinstance(value, str):
        items = [v.strip() for v in value.split(",")]
    elif isinstance(value, list):
        items = [str(v or "").strip() for v in value]
    else:
        items = []
    out: List[str] = []
    for item in items:
        if not item:
            continue
        normalized = _title_case_words(re.sub(r"\s+", " ", item).strip(), max_words=4)
        if normalized and normalized not in out:
            out.append(normalized)
        if len(out) >= max_items:
            break
    return out


def _normalize_level1_label(raw_label: object) -> str:
    label = _title_case_words(str(raw_label or ""), max_words=2)
    key = _normalize_label_key(label)
    if not label:
        return DEFAULT_LEVEL1
    if key in LEVEL1_NORMALIZATION:
        return LEVEL1_NORMALIZATION[key]
    return DEFAULT_LEVEL1


def _normalize_level1_list(value: object, max_items: int = None) -> List[str]:
    if max_items is None:
        max_items = int(MAX_LEVEL1_ITEMS)
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        items = []
    out: List[str] = []
    for item in items:
        normalized = _normalize_level1_label(item)
        if normalized not in out:
            out.append(normalized)
        if len(out) >= max_items:
            break
    return out


def _normalize_category_label(raw_label: object) -> str:
    label = _title_case_words(str(raw_label or ""), max_words=4)
    key = _normalize_label_key(label)
    if not label:
        return DEFAULT_LEVEL2
    if key in CATEGORY_NORMALIZATION:
        return CATEGORY_NORMALIZATION[key]
    return label


def _normalize_category_list(value: object, max_items: int = None) -> List[str]:
    if max_items is None:
        max_items = int(MAX_LEVEL2_ITEMS)
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        items = []
    out: List[str] = []
    for item in items:
        normalized = _normalize_category_label(item)
        if normalized not in out:
            out.append(normalized)
        if len(out) >= max_items:
            break
    return out


def _canonicalize_label(label: str, alias_map: Dict[str, str]) -> str:
    current = _title_case_words(label, max_words=4)
    if not current:
        return ""
    visited: Set[str] = set()
    while True:
        key = _normalize_label_key(current)
        if not key or key in visited:
            return current
        visited.add(key)
        nxt = alias_map.get(key, "")
        if not nxt:
            return current
        current = _title_case_words(nxt, max_words=4)


def _register_alias(old_label: str, new_label: str, alias_map: Dict[str, str], label_counts: Dict[str, int]) -> bool:
    old_can = _canonicalize_label(old_label, alias_map)
    new_can = _canonicalize_label(new_label, alias_map)
    if not old_can or not new_can or old_can == new_can:
        return False
    alias_map[_normalize_label_key(old_can)] = new_can
    if old_can in label_counts:
        label_counts[new_can] = int(label_counts.get(new_can, 0)) + int(label_counts.get(old_can, 0))
        del label_counts[old_can]
    for key, val in list(alias_map.items()):
        if _normalize_label_key(val) == _normalize_label_key(old_can):
            alias_map[key] = new_can
    return True


def _get_canonical_label_counts(label_counts: Dict[str, int], alias_map: Dict[str, str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for label, count in label_counts.items():
        can = _canonicalize_label(label, alias_map) or _title_case_words(label, max_words=4)
        if not can:
            continue
        out[can] = int(out.get(can, 0)) + int(count)
    return out


def _build_shortlist(
    label_counts: Dict[str, int],
    alias_map: Dict[str, str],
    level2_to_level1: Dict[str, str],
    topk: int,
    max_total_categories: int,
) -> List[Tuple[str, str, int]]:
    canonical_counts = _get_canonical_label_counts(label_counts, alias_map)
    seed_labels = [seed_l2 for _, seed_l2 in BASE_CATEGORY_SEEDS]
    seed_entries: List[Tuple[str, str, int]] = []
    for level1, seed_l2 in BASE_CATEGORY_SEEDS:
        seed_entries.append((level1, seed_l2, int(canonical_counts.get(seed_l2, 0))))
    non_seed = [(label, count) for label, count in canonical_counts.items() if label not in seed_labels]
    non_seed = sorted(non_seed, key=lambda x: (-x[1], x[0]))
    non_seed_entries: List[Tuple[str, str, int]] = []
    for label, count in non_seed:
        level1 = level2_to_level1.get(label, DEFAULT_LEVEL1)
        non_seed_entries.append((level1, label, count))
    if topk <= 0:
        return seed_entries + non_seed_entries
    budget = max(0, int(max_total_categories) - len(seed_entries))
    return seed_entries + non_seed_entries[:min(topk, budget)]


def _load_existing_state(
    out_jsonl_path: str,
) -> Tuple[Set[str], Dict[str, int], Dict[str, str], Dict[str, str]]:
    if not os.path.exists(out_jsonl_path):
        return set(), {}, {}, {}
    seen: Set[str] = set()
    counts: Dict[str, int] = {}
    alias_map: Dict[str, str] = {}
    level2_to_level1: Dict[str, str] = {}
    with open(out_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            doc_id = str(row.get("doc_id", "")).strip()
            if doc_id:
                seen.add(doc_id)
            category_keywords = _normalize_category_list(
                row.get("category_level_2_keywords", row.get("category_level_2", row.get("category", "")))
            )
            category_raw = _normalize_category_label(row.get("category_raw", ""))
            level1_keywords = _normalize_level1_list(
                row.get("category_level_1_keywords", row.get("category_level_1", ""))
            )
            level1_fallback = level1_keywords[0] if level1_keywords else DEFAULT_LEVEL1
            for idx, category in enumerate(category_keywords):
                counts[category] = int(counts.get(category, 0)) + 1
                mapped_level1 = level1_keywords[idx] if idx < len(level1_keywords) else level1_fallback
                if category not in level2_to_level1:
                    level2_to_level1[category] = mapped_level1
            if category_keywords and category_raw and _normalize_label_key(category_keywords[0]) != _normalize_label_key(category_raw):
                alias_map[_normalize_label_key(category_raw)] = category_keywords[0]
    return seen, counts, alias_map, level2_to_level1


def _build_prompt(subset: str, website_title: str, doc_desc: str, known_labels_block: str) -> str:
    subset_key = str(subset or "").strip().lower()
    relevance_definition = DATASET_RELEVANCE_DEFINITION.get(
        subset_key,
        DATASET_RELEVANCE_DEFINITION["stackoverflow"],
    )
    subset_domain = SUBSET_DOMAIN_LABEL.get(subset_key, subset_key.replace("_", " ").title())
    return PROMPT_TEMPLATE.format(
        subset=subset,
        subset_domain=subset_domain,
        relevance_definition=relevance_definition,
        known_labels_block=known_labels_block,
        website_title=website_title,
        doc_desc=doc_desc,
    )


def _normalize_gap_reason(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _normalize_gap_candidates(value: object, max_items: int = 3) -> List[str]:
    return _normalize_list(value, max_items=max_items)


def _tokens_for_overlap(text: str) -> Set[str]:
    return {
        tok
        for tok in re.findall(r"[a-zA-Z]{3,}", str(text or "").lower())
        if tok not in {"the", "and", "for", "with", "from", "that", "this"}
    }


def _filter_topic_like_gap_candidates(gap_candidates: List[str], website_title: str) -> Tuple[List[str], List[str]]:
    title_tokens = _tokens_for_overlap(website_title)
    kept: List[str] = []
    leaked: List[str] = []
    for cand in gap_candidates:
        cand_tokens = _tokens_for_overlap(cand)
        if title_tokens and cand_tokens and (title_tokens & cand_tokens):
            leaked.append(cand)
            continue
        kept.append(cand)
    return kept, leaked


def _parse_output(text: str) -> Tuple[bool, List[str], List[str], List[str], List[str], bool, str, List[str]]:
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
        return False, [DEFAULT_LEVEL1], [DEFAULT_LEVEL2], [], [], False, "", []
    level1_raw = _normalize_level1_list(
        obj.get("level1_categories", obj.get("level1_category", obj.get("category_level_1", obj.get("level1", ""))))
    )
    category_raw = _normalize_category_list(
        obj.get("level2_categories", obj.get("level2_category", obj.get("category_level_2", obj.get("category", ""))))
    )
    canonical_hint_raw = obj.get("canonical_level2", obj.get("canonical_label", ""))
    if isinstance(canonical_hint_raw, str):
        canonical_hint_items = [canonical_hint_raw] if canonical_hint_raw.strip() else []
    elif isinstance(canonical_hint_raw, list):
        canonical_hint_items = canonical_hint_raw
    else:
        canonical_hint_items = []
    canonical_hint: List[str] = []
    for item in canonical_hint_items:
        item_text = str(item or "").strip()
        if not item_text:
            continue
        normalized_item = _normalize_category_label(item_text)
        if normalized_item and normalized_item not in canonical_hint:
            canonical_hint.append(normalized_item)
    merge_from = _normalize_list(obj.get("merge_from", []), max_items=6)
    if CATEGORY_VERSION == "v3":
        is_gap = bool(obj.get("is_gap", False))
        gap_reason = _normalize_gap_reason(obj.get("gap_reason", ""))
        gap_candidates = _normalize_gap_candidates(obj.get("gap_candidates", []), max_items=3)
    else:
        is_gap = False
        gap_reason = ""
        gap_candidates = []
    if not level1_raw:
        level1_raw = [DEFAULT_LEVEL1]
    if not category_raw:
        category_raw = [DEFAULT_LEVEL2]
    return True, level1_raw, category_raw, canonical_hint, merge_from, is_gap, gap_reason, gap_candidates


def _classify_batch_rows(
    *,
    llm: LLM,
    sampling_params: SamplingParams,
    tokenizer,
    prompt_token_limit: int,
    subset: str,
    max_desc_words: int,
    batch_rows: List[Tuple[int, Dict]],
    label_counts: Dict[str, int],
    alias_map: Dict[str, str],
    level2_to_level1: Dict[str, str],
    label_shortlist_topk: int,
    label_alias_threshold: float,
    max_total_categories: int,
    parse_retry_max: int,
) -> List[Dict]:
    shortlist_with_counts = _build_shortlist(
        label_counts,
        alias_map,
        level2_to_level1,
        topk=label_shortlist_topk,
        max_total_categories=max_total_categories,
    )
    shortlist_labels = [label for _, label, _ in shortlist_with_counts]
    if shortlist_with_counts:
        known_labels_block = "\n".join(
            [f"- [{level1}] {label} (count={count})" for level1, label, count in shortlist_with_counts]
        )
    else:
        known_labels_block = "- None yet (create a new abstract category if needed)"

    prompts: List[str] = []
    trimmed_count = 0
    for _, row in batch_rows:
        doc_id = str(row.get("doc_id", "")).strip()
        website_title = _website_title_from_doc_id(doc_id)
        desc = _truncate_words(row.get("desc", ""), max_desc_words)
        # Intent: pass website title (not raw id) so category decisions are anchored to human-readable source context.
        desc = f"{website_title}\n{desc}"
        guarded_prompt, was_trimmed = _build_guarded_prompt(
            subset=subset,
            website_title=website_title,
            doc_desc=desc,
            known_labels_block=known_labels_block,
            tokenizer=tokenizer,
            prompt_token_limit=prompt_token_limit,
        )
        prompts.append(guarded_prompt)
        if was_trimmed:
            trimmed_count += 1

    if trimmed_count > 0:
        print(f"[Guard] token-trimmed prompts in batch: {trimmed_count}/{len(batch_rows)}")

    outputs = llm.generate(prompts, sampling_params)
    classified: List[Dict] = []
    for row_idx, ((_, row), output) in enumerate(zip(batch_rows, outputs)):
        text = output.outputs[0].text if output.outputs else ""
        doc_id = str(row.get("doc_id", "")).strip()
        website_title = _website_title_from_doc_id(doc_id)
        (
            parse_ok,
            level1_raw_list,
            category_raw_list,
            canonical_hint_list,
            merge_from,
            is_gap,
            gap_reason,
            gap_candidates,
        ) = _parse_output(text)
        parse_retry_count = 0
        if (not parse_ok) and int(parse_retry_max) > 0:
            # Intent: regenerate on parse failure to avoid default fallback caused by transient malformed JSON output.
            retry_prompt = (
                prompts[row_idx]
                + "\n\nSTRICT OUTPUT FORMAT: Return exactly one valid JSON object. No markdown fences. No extra text."
            )
            for _ in range(int(parse_retry_max)):
                retry_output = llm.generate([retry_prompt], sampling_params)
                retry_text = (
                    retry_output[0].outputs[0].text
                    if retry_output and retry_output[0].outputs
                    else ""
                )
                parse_retry_count += 1
                (
                    parse_ok,
                    level1_raw_list,
                    category_raw_list,
                    canonical_hint_list,
                    merge_from,
                    is_gap,
                    gap_reason,
                    gap_candidates,
                ) = _parse_output(retry_text)
                if parse_ok:
                    text = retry_text
                    break
        if CATEGORY_VERSION == "v3":
            filtered_gap_candidates, leaked_gap_candidates = _filter_topic_like_gap_candidates(gap_candidates, website_title)
            if leaked_gap_candidates and not filtered_gap_candidates:
                is_gap = False
                gap_reason = ""
        else:
            filtered_gap_candidates, leaked_gap_candidates = [], []
            is_gap = False
            gap_reason = ""
        primary_level1_raw = level1_raw_list[0] if level1_raw_list else DEFAULT_LEVEL1
        primary_category_raw = category_raw_list[0] if category_raw_list else DEFAULT_LEVEL2
        primary_canonical_hint = canonical_hint_list[0] if canonical_hint_list else ""
        level1_category, category, label_decision, merge_applied = _resolve_category(
            generated_level1=primary_level1_raw,
            generated_category=primary_category_raw,
            canonical_hint=primary_canonical_hint,
            merge_from=merge_from,
            shortlist_labels=shortlist_labels,
            label_counts=label_counts,
            alias_map=alias_map,
            level2_to_level1=level2_to_level1,
            alias_threshold=label_alias_threshold,
            max_total_categories=max_total_categories,
        )
        known_set = set(_get_canonical_label_counts(label_counts, alias_map).keys())
        known_set.update(shortlist_labels)

        resolved_level2_keywords: List[str] = [category]
        resolved_level1_keywords: List[str] = [level1_category]
        for idx in range(1, len(category_raw_list)):
            raw_level2 = _canonicalize_label(category_raw_list[idx], alias_map)
            if not raw_level2:
                continue
            chosen_extra = raw_level2
            if raw_level2 not in known_set and known_set:
                best_label = ""
                best_score = -1.0
                key_gen = _normalize_label_key(raw_level2)
                for label in known_set:
                    score = SequenceMatcher(None, key_gen, _normalize_label_key(label)).ratio()
                    if score > best_score:
                        best_score = score
                        best_label = label
                if best_label and best_score >= float(label_alias_threshold):
                    chosen_extra = best_label
            if chosen_extra not in known_set:
                if len(_get_canonical_label_counts(label_counts, alias_map)) >= int(max_total_categories) and known_set:
                    best_label = ""
                    best_score = -1.0
                    key_extra = _normalize_label_key(chosen_extra)
                    for label in known_set:
                        score = SequenceMatcher(None, key_extra, _normalize_label_key(label)).ratio()
                        if score > best_score:
                            best_score = score
                            best_label = label
                    if best_label:
                        chosen_extra = best_label
            if chosen_extra in resolved_level2_keywords:
                continue
            if chosen_extra not in level2_to_level1:
                mapped_level1 = (
                    level1_raw_list[idx] if idx < len(level1_raw_list) else level1_category
                )
                level2_to_level1[chosen_extra] = _normalize_level1_label(mapped_level1)
            resolved_level2_keywords.append(chosen_extra)
            resolved_level1_keywords.append(level2_to_level1.get(chosen_extra, level1_category))
            known_set.add(chosen_extra)

        for keyword in resolved_level2_keywords:
            level2_to_level1[keyword] = level2_to_level1.get(keyword, level1_category)
            label_counts[keyword] = int(label_counts.get(keyword, 0)) + 1
        resolved_level1_keywords = list(dict.fromkeys(resolved_level1_keywords))

        classified.append({
            "row": row,
            "category_level_1": level1_category,
            "category_level_2": category,
            "category_level_1_keywords": resolved_level1_keywords,
            "category_level_2_keywords": resolved_level2_keywords,
            "category": category,
            "category_raw": primary_category_raw,
            "category_level_1_raw": primary_level1_raw,
            "label_decision": label_decision,
            "canonical_hint": primary_canonical_hint,
            "merge_from": merge_from,
            "merge_applied": merge_applied,
            "is_gap": bool(is_gap),
            "gap_reason": gap_reason,
            "gap_candidates": filtered_gap_candidates,
            "gap_topic_leak_candidates": leaked_gap_candidates,
            "parse_success": bool(parse_ok),
            "parse_retry_count": int(parse_retry_count),
            "shortlist_labels": shortlist_labels,
        })
    return classified


def _resolve_category(
    *,
    generated_level1: str,
    generated_category: str,
    canonical_hint: str,
    merge_from: List[str],
    shortlist_labels: List[str],
    label_counts: Dict[str, int],
    alias_map: Dict[str, str],
    level2_to_level1: Dict[str, str],
    alias_threshold: float,
    max_total_categories: int,
) -> Tuple[str, str, str, List[str]]:
    level1 = _normalize_level1_label(generated_level1)
    generated = _canonicalize_label(generated_category, alias_map)
    hint = _canonicalize_label(canonical_hint, alias_map) if canonical_hint else ""
    known_set = set(_get_canonical_label_counts(label_counts, alias_map).keys())
    known_set.update(shortlist_labels)

    decision = "new"
    chosen = generated
    if hint and hint in known_set:
        chosen = hint
        decision = "reuse"
    elif generated in known_set:
        chosen = generated
        decision = "reuse"
    elif generated and known_set:
        best_label = ""
        best_score = -1.0
        key_gen = _normalize_label_key(generated)
        for label in known_set:
            score = SequenceMatcher(None, key_gen, _normalize_label_key(label)).ratio()
            if score > best_score:
                best_score = score
                best_label = label
        if best_label and best_score >= float(alias_threshold):
            chosen = best_label
            decision = "alias"
            _register_alias(generated, chosen, alias_map, label_counts)

    if chosen and chosen not in known_set:
        current_num_categories = len(_get_canonical_label_counts(label_counts, alias_map))
        if current_num_categories >= int(max_total_categories) and known_set:
            best_label = ""
            best_score = -1.0
            key_gen = _normalize_label_key(chosen)
            for label in known_set:
                score = SequenceMatcher(None, key_gen, _normalize_label_key(label)).ratio()
                if score > best_score:
                    best_score = score
                    best_label = label
            if best_label:
                # Intent: hard-cap total role categories for stable retrieval behavior across runs.
                chosen = best_label
                decision = "cap_reuse"

    merge_applied: List[str] = []
    if merge_from:
        for old in merge_from:
            old_can = _canonicalize_label(old, alias_map)
            if not old_can or old_can == chosen:
                continue
            if old_can in known_set and _register_alias(old_can, chosen, alias_map, label_counts):
                merge_applied.append(old_can)
        if merge_applied:
            decision = f"{decision}_merge"

    if not chosen:
        chosen = DEFAULT_LEVEL2
        decision = "fallback_default"

    # Intent: keep level-1 stable for existing level-2 labels to avoid semantic drift in registry.
    chosen_level1 = level2_to_level1.get(chosen, level1 or DEFAULT_LEVEL1)
    if chosen not in level2_to_level1:
        level2_to_level1[chosen] = chosen_level1
    return chosen_level1, chosen, decision, merge_applied


def _flush_batch(
    *,
    llm: LLM,
    sampling_params: SamplingParams,
    tokenizer,
    prompt_token_limit: int,
    subset: str,
    max_desc_words: int,
    batch_rows: List[Tuple[int, Dict]],
    out_file,
    debug_output: bool,
    seen_doc_ids: Set[str],
    label_counts: Dict[str, int],
    alias_map: Dict[str, str],
    level2_to_level1: Dict[str, str],
    label_shortlist_topk: int,
    label_alias_threshold: float,
    max_total_categories: int,
    parse_retry_max: int,
    limit: int,
    generated_so_far: int,
) -> int:
    if not batch_rows:
        return 0

    if limit > 0:
        remaining = int(limit - generated_so_far)
        if remaining <= 0:
            return 0
        batch_rows = batch_rows[:remaining]

    classified = _classify_batch_rows(
        llm=llm,
        sampling_params=sampling_params,
        tokenizer=tokenizer,
        prompt_token_limit=prompt_token_limit,
        subset=subset,
        max_desc_words=max_desc_words,
        batch_rows=batch_rows,
        label_counts=label_counts,
        alias_map=alias_map,
        level2_to_level1=level2_to_level1,
        label_shortlist_topk=label_shortlist_topk,
        label_alias_threshold=label_alias_threshold,
        max_total_categories=max_total_categories,
        parse_retry_max=parse_retry_max,
    )
    new_count = 0
    for item in classified:
        row = item["row"]
        doc_id = str(row.get("doc_id", ""))
        category_level_1 = str(item["category_level_1"])
        category_level_2 = str(item["category_level_2"])
        record = {
            "doc_id": doc_id,
            # Intent: keep legacy `category` key for backward compatibility while adding explicit 2-level fields.
            "category": category_level_2,
            "category_level_1": category_level_1,
            "category_level_2": category_level_2,
            "category_level_1_keywords": item["category_level_1_keywords"],
            "category_level_2_keywords": item["category_level_2_keywords"],
            # Intent: always persist gap signals so v3 category expansion can be analyzed offline without debug mode.
            "is_gap": item["is_gap"],
            "gap_reason": item["gap_reason"],
            "gap_candidates": item["gap_candidates"],
            "gap_topic_leak_candidates": item["gap_topic_leak_candidates"],
            "parse_success": item["parse_success"],
            "parse_retry_count": item["parse_retry_count"],
        }
        if debug_output:
            record.update({
                "category_raw": item["category_raw"],
                "category_level_1_raw": item["category_level_1_raw"],
                "label_decision": item["label_decision"],
                "canonical_label_hint": item["canonical_hint"],
                "merge_from": item["merge_from"],
                "merge_applied": item["merge_applied"],
                "is_chunk_merged": bool(row.get("is_chunk_merged", False)),
                "num_merged_chunks": len(row.get("source_doc_ids", [])),
                "shortlist_labels": item["shortlist_labels"],
            })
        out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        seen_doc_ids.add(record["doc_id"])
        new_count += 1
    return new_count


def _save_registry(
    path: str,
    label_counts: Dict[str, int],
    alias_map: Dict[str, str],
    level2_to_level1: Dict[str, str],
    meta: Dict[str, object],
) -> None:
    canonical_counts = _get_canonical_label_counts(label_counts, alias_map)
    categories = []
    level1_counts: Dict[str, int] = {}
    for label, count in sorted(canonical_counts.items(), key=lambda x: (-x[1], x[0])):
        level1 = level2_to_level1.get(label, DEFAULT_LEVEL1)
        level1_counts[level1] = int(level1_counts.get(level1, 0)) + int(count)
        categories.append({
            "label": label,
            "level1": level1,
            "level2": label,
            "count": int(count),
        })
    payload = {
        "meta": meta,
        "num_categories": len(categories),
        "num_level1_categories": len(level1_counts),
        "level1_categories": [
            {"level1": level1, "count": int(count)}
            for level1, count in sorted(level1_counts.items(), key=lambda x: (-x[1], x[0]))
        ],
        "categories": categories,
        "alias_map": alias_map,
        "level2_to_level1": level2_to_level1,
    }
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="One-pass abstract category assignment per document (local vLLM).")
    parser.add_argument("--dataset", type=str, default="BRIGHT")
    parser.add_argument("--subset", type=str, required=True)
    parser.add_argument("--category_version", type=str, default=None, choices=["v2", "v3"])
    parser.add_argument("--long_documents_path", type=str, default=None)
    parser.add_argument("--documents_path", type=str, default=None)
    parser.add_argument("--out_jsonl", type=str, default=None)
    parser.add_argument("--out_registry_json", type=str, default=None)
    parser.add_argument("--prompt_name", type=str, default=None)
    parser.add_argument("--llm", type=str, required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=24576)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--max_tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_desc_words", type=int, default=12000)
    # Intent: keep one range interface (doc_idx) to reduce CLI complexity and maintenance overhead.
    parser.add_argument("--start_doc_idx", type=int, default=0)
    parser.add_argument("--end_doc_idx", type=int, default=-1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--print_every", type=int, default=200)
    parser.add_argument("--label_shortlist_topk", type=int, default=7)
    parser.add_argument("--label_alias_threshold", type=float, default=0.90)
    parser.add_argument("--max_total_categories", type=int, default=10)
    parser.add_argument("--parse_retry_max", type=int, default=2)
    parser.add_argument("--hard_prompt_token_limit", type=int, default=16384)
    parser.add_argument("--prompt_token_margin", type=int, default=32)
    parser.add_argument("--debug_output", action="store_true", default=False)
    args = parser.parse_args()

    inferred_version = str(args.prompt_name or "").strip().lower()
    if args.category_version is not None:
        category_version = _set_category_version(args.category_version)
    elif "v2" in inferred_version:
        category_version = _set_category_version("v2")
    else:
        category_version = _set_category_version("v3")
    prompt_name = args.prompt_name or f"category_assign_{category_version}"

    long_documents_path = args.long_documents_path or _default_long_documents_path(args.dataset, args.subset)
    documents_path = args.documents_path or _default_documents_path(args.dataset, args.subset)
    source_path, source_name = _resolve_input_parquet_path(
        long_documents_path=long_documents_path,
        documents_path=documents_path,
    )
    out_jsonl = args.out_jsonl or _default_output_path(args.dataset, args.subset, prompt_name)
    out_registry_json = args.out_registry_json or _default_registry_path(args.dataset, args.subset, prompt_name)
    print(f"[Data] using {source_name}: {source_path}")
    print(f"[Category] version={category_version} prompt_name={prompt_name}")

    if args.overwrite and os.path.exists(out_jsonl):
        os.remove(out_jsonl)

    _ensure_parent_dir(out_jsonl)
    if args.overwrite:
        seen_doc_ids, label_counts, alias_map, level2_to_level1 = set(), {}, {}, {}
        for level1, level2 in BASE_CATEGORY_SEEDS:
            level2_to_level1[level2] = level1
    else:
        seen_doc_ids, label_counts, alias_map, level2_to_level1 = _load_existing_state(out_jsonl)
        for level1, level2 in BASE_CATEGORY_SEEDS:
            if level2 not in level2_to_level1:
                level2_to_level1[level2] = level1

    llm = LLM(
        model=args.llm,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    reserved_generation_tokens = max(1, int(args.max_tokens))
    reserved_margin_tokens = max(0, int(args.prompt_token_margin))
    # Intent: reserve generation + safety margin so prompt+output never exceeds vLLM max_model_len.
    prompt_budget_from_model = int(args.max_model_len) - reserved_generation_tokens - reserved_margin_tokens
    prompt_token_limit = int(min(prompt_budget_from_model, int(args.hard_prompt_token_limit)))
    if prompt_token_limit <= 0:
        raise ValueError(
            "Prompt budget is non-positive. Increase --max_model_len, or reduce --max_tokens / --prompt_token_margin."
        )
    print(
        "[Config] prompt_token_limit="
        f"{prompt_token_limit} (max_model_len={int(args.max_model_len)}, "
        f"max_tokens={reserved_generation_tokens}, margin={reserved_margin_tokens}, "
        f"hard_limit={int(args.hard_prompt_token_limit)})"
    )

    generated = 0
    skipped_existing = 0
    skipped_range = 0
    batch_rows: List[Tuple[int, Dict]] = []

    with open(out_jsonl, "a", encoding="utf-8") as out_file:
        # Intent: prefer long_documents, then fall back to documents to preserve subset coverage.
        for doc_idx, row in _iter_document_units_from_parquet(source_path):
            if doc_idx < args.start_doc_idx:
                skipped_range += 1
                continue
            if args.end_doc_idx >= 0 and doc_idx >= args.end_doc_idx:
                break

            doc_id = str(row.get("doc_id", "")).strip()
            if not doc_id:
                continue
            if doc_id in seen_doc_ids:
                skipped_existing += 1
                continue

            batch_rows.append((doc_idx, row))
            if len(batch_rows) < args.batch_size:
                continue

            generated += _flush_batch(
                llm=llm,
                sampling_params=sampling_params,
                tokenizer=tokenizer,
                prompt_token_limit=prompt_token_limit,
                subset=args.subset,
                max_desc_words=args.max_desc_words,
                batch_rows=batch_rows,
                out_file=out_file,
                debug_output=args.debug_output,
                seen_doc_ids=seen_doc_ids,
                label_counts=label_counts,
                alias_map=alias_map,
                level2_to_level1=level2_to_level1,
                label_shortlist_topk=args.label_shortlist_topk,
                label_alias_threshold=args.label_alias_threshold,
                max_total_categories=args.max_total_categories,
                parse_retry_max=args.parse_retry_max,
                limit=args.limit,
                generated_so_far=generated,
            )
            batch_rows = []

            if args.print_every > 0 and generated > 0 and generated % args.print_every == 0:
                print(
                    f"[Progress] generated={generated} "
                    f"skipped_existing={skipped_existing} "
                    f"skipped_range={skipped_range} "
                    f"num_categories={len(_get_canonical_label_counts(label_counts, alias_map))}"
                )

            if args.limit > 0 and generated >= args.limit:
                break

        if batch_rows and (args.limit <= 0 or generated < args.limit):
            generated += _flush_batch(
                llm=llm,
                sampling_params=sampling_params,
                tokenizer=tokenizer,
                prompt_token_limit=prompt_token_limit,
                subset=args.subset,
                max_desc_words=args.max_desc_words,
                batch_rows=batch_rows,
                out_file=out_file,
                debug_output=args.debug_output,
                seen_doc_ids=seen_doc_ids,
                label_counts=label_counts,
                alias_map=alias_map,
                level2_to_level1=level2_to_level1,
                label_shortlist_topk=args.label_shortlist_topk,
                label_alias_threshold=args.label_alias_threshold,
                max_total_categories=args.max_total_categories,
                parse_retry_max=args.parse_retry_max,
                limit=args.limit,
                generated_so_far=generated,
            )

    created_at = datetime.datetime.utcnow().isoformat() + "Z"
    _save_registry(
        out_registry_json,
        label_counts,
        alias_map,
        level2_to_level1,
        meta={
            "dataset": args.dataset,
            "subset": args.subset,
            "prompt_name": prompt_name,
            "category_version": category_version,
            "model": args.llm,
            "created_at": created_at,
        },
    )

    print(f"[Done] output={out_jsonl}")
    print(f"[Done] registry={out_registry_json}")
    print(
        f"[Stats] generated={generated} "
        f"skipped_existing={skipped_existing} "
        f"skipped_range={skipped_range} "
        f"num_categories={len(_get_canonical_label_counts(label_counts, alias_map))}"
    )


if __name__ == "__main__":
    main()
