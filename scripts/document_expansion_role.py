import argparse
import datetime
import json
import os
import re
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Set, Tuple

from json_repair import repair_json
from vllm import LLM, SamplingParams


CHUNK_MERGE_SUBSETS = {
    "biology",
    "earth_science",
    "economics",
    "psychology",
    "robotics",
    "sustainable_living",
    "pony",
    "stackoverflow",
}

BASE_CATEGORY_SEEDS = [
    "Domain Theory",
    "Cross-Domain Theory",
    "Mathematical Theory",
    "Entity",
    "Example",
]

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

PROMPT_TEMPLATE = (
    "You are assigning an abstract evidence-category for one document.\n\n"
    "Subset: {subset}\n"
    "Subset Domain: {subset_domain}\n"
    "Relevance Definition:\n{relevance_definition}\n\n"
    "Goal:\n"
    "- Output one abstract category label for how this document can support reasoning answers.\n"
    "- First try to use one of the core categories below.\n"
    "- Reuse an existing label when it fits.\n"
    "- Only create a new label when no existing label fits.\n"
    "- Category labels must be abstract (1-3 words), not concrete named theories/entities.\n"
    "- Bad: Evolution Theory, Poisson Distribution, DSM-5\n"
    "- Good: Domain Theory, Mathematical Theory, Causal Evidence, Procedural Method\n\n"
    "Core Categories:\n"
    "- Domain Theory\n"
    "- Cross-Domain Theory\n"
    "- Mathematical Theory\n"
    "- Entity\n"
    "- Example\n\n"
    "Expansion rule:\n"
    "- If no core category fits, you may create one new abstract category.\n\n"
    "Known Categories (canonical):\n"
    "{known_labels_block}\n\n"
    "If your new category can absorb old labels, list those old labels in merge_from.\n\n"
    "Document ID: {doc_id}\n"
    "Document Snippet:\n{doc_desc}\n\n"
    "Output JSON only:\n"
    "{{\n"
    "  \"category\": \"...\",\n"
    "  \"canonical_label\": \"optional existing known label if reused, else empty\",\n"
    "  \"merge_from\": [\"optional known labels to alias into category\"]\n"
    "}}"
)

CONCRETE_TO_ABSTRACT = {
}

CATEGORY_NORMALIZATION = {
    "domaintheory": "Domain Theory",
    "crossdomaintheory": "Cross-Domain Theory",
    "mathematicaltheory": "Mathematical Theory",
    "maththeory": "Mathematical Theory",
    "statisticaltheory": "Mathematical Theory",
    "probabilistictheory": "Mathematical Theory",
    "entity": "Entity",
    "example": "Example",
}


def _default_node_catalog_path(dataset: str, subset: str) -> str:
    return os.path.join("trees", dataset, subset, "node_catalog.jsonl")


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
    doc_id: str,
    doc_desc: str,
    known_labels_block: str,
    tokenizer,
    prompt_token_limit: int,
) -> Tuple[str, bool]:
    prompt = _build_prompt(subset=subset, doc_id=doc_id, doc_desc=doc_desc, known_labels_block=known_labels_block)
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
            doc_id=doc_id,
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

    empty_prompt = _build_prompt(subset=subset, doc_id=doc_id, doc_desc="", known_labels_block=known_labels_block)
    if _count_prompt_tokens(empty_prompt, tokenizer) <= prompt_token_limit:
        return empty_prompt, True

    # Intent: final hard guard to prevent vLLM max-length crash even when prompt prefix alone is too long.
    ids = tokenizer.encode(empty_prompt, add_special_tokens=False)
    trimmed = tokenizer.decode(ids[:prompt_token_limit], skip_special_tokens=True)
    return trimmed, True


def _merge_chunked_doc_id(doc_id: str) -> Tuple[str, int]:
    raw = str(doc_id or "").strip()
    m = re.match(r"^(.*)_(\d+)\.txt$", raw)
    if not m:
        return raw, -1
    base = m.group(1)
    # Intent: after removing the last _<num>, also trim trailing 1-2 digits (without underscore).
    base = re.sub(r"\d{1,2}$", "", base)
    base = re.sub(r"[_-]+$", "", base)
    return f"{base}.txt", int(m.group(2))


def _iter_leaf_rows(node_catalog_path: str) -> Iterable[Tuple[int, Dict]]:
    leaf_idx = 0
    with open(node_catalog_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not bool(row.get("is_leaf", False)):
                continue
            yield leaf_idx, row
            leaf_idx += 1


def _iter_document_units(node_catalog_path: str, subset: str) -> Iterable[Tuple[int, Dict]]:
    subset_key = str(subset or "").strip().lower()
    if subset_key not in CHUNK_MERGE_SUBSETS:
        for leaf_idx, row in _iter_leaf_rows(node_catalog_path):
            doc_id = str(row.get("id", "")).strip()
            yield leaf_idx, {
                "doc_id": doc_id,
                "desc": str(row.get("desc", "")),
                "is_chunk_merged": False,
                "source_doc_ids": [doc_id] if doc_id else [],
            }
        return

    groups: Dict[str, Dict] = {}
    for leaf_idx, row in _iter_leaf_rows(node_catalog_path):
        raw_doc_id = str(row.get("id", "")).strip()
        if not raw_doc_id:
            continue
        merged_doc_id, chunk_idx = _merge_chunked_doc_id(raw_doc_id)
        if merged_doc_id not in groups:
            groups[merged_doc_id] = {
                "first_leaf_idx": int(leaf_idx),
                "parts": [],
            }
        groups[merged_doc_id]["parts"].append({
            "chunk_idx": int(chunk_idx),
            "leaf_idx": int(leaf_idx),
            "doc_id": raw_doc_id,
            "desc": str(row.get("desc", "")),
        })

    ordered_groups = sorted(groups.items(), key=lambda x: int(x[1].get("first_leaf_idx", 10**9)))
    for unit_idx, (merged_doc_id, payload) in enumerate(ordered_groups):
        parts = payload.get("parts", [])
        parts_sorted = sorted(
            parts,
            key=lambda p: (
                10**9 if int(p.get("chunk_idx", -1)) < 0 else int(p.get("chunk_idx", -1)),
                int(p.get("leaf_idx", -1)),
            ),
        )
        merged_desc = "\n".join(
            [str(p.get("desc", "")).strip() for p in parts_sorted if str(p.get("desc", "")).strip()]
        ).strip()
        yield unit_idx, {
            "doc_id": merged_doc_id,
            "desc": merged_desc,
            "is_chunk_merged": True,
            "source_doc_ids": [str(p.get("doc_id", "")).strip() for p in parts_sorted if str(p.get("doc_id", "")).strip()],
        }


def _parse_depths(raw: str) -> Set[int]:
    out: Set[int] = set()
    for token in str(raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            depth = int(token)
        except Exception:
            continue
        if depth >= 0:
            out.add(depth)
    return out


def _iter_nonleaf_units(node_catalog_path: str, depths: Set[int]) -> Iterable[Tuple[int, Dict]]:
    unit_idx = 0
    with open(node_catalog_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if bool(row.get("is_leaf", False)):
                continue
            depth = int(row.get("depth", -1))
            if depth not in depths:
                continue
            node_id = str(row.get("id", "")).strip()
            if not node_id:
                node_id = f"path:{row.get('path', [])}"
            yield unit_idx, {
                "doc_id": f"NONLEAF_D{depth}:{node_id}",
                "desc": str(row.get("desc", "")),
                "depth": depth,
                "is_nonleaf_warmup": True,
            }
            unit_idx += 1


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
        normalized = _title_case_words(re.sub(r"\s+", " ", item).strip(), max_words=3)
        if normalized and normalized not in out:
            out.append(normalized)
        if len(out) >= max_items:
            break
    return out


def _normalize_category_label(raw_label: object) -> str:
    label = _title_case_words(str(raw_label or ""), max_words=3)
    key = _normalize_label_key(label)
    if not label:
        return "Auxiliary Context"
    if key in CATEGORY_NORMALIZATION:
        return CATEGORY_NORMALIZATION[key]
    for cue, mapped in CONCRETE_TO_ABSTRACT.items():
        if cue in key:
            return mapped
    return label


def _canonicalize_label(label: str, alias_map: Dict[str, str]) -> str:
    current = _title_case_words(label, max_words=3)
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
        current = _title_case_words(nxt, max_words=3)


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
        can = _canonicalize_label(label, alias_map) or _title_case_words(label, max_words=3)
        if not can:
            continue
        out[can] = int(out.get(can, 0)) + int(count)
    return out


def _build_shortlist(label_counts: Dict[str, int], alias_map: Dict[str, str], topk: int) -> List[Tuple[str, int]]:
    canonical_counts = _get_canonical_label_counts(label_counts, alias_map)
    seed_entries: List[Tuple[str, int]] = []
    for seed in BASE_CATEGORY_SEEDS:
        seed_entries.append((seed, int(canonical_counts.get(seed, 0))))
    non_seed = [(label, count) for label, count in canonical_counts.items() if label not in BASE_CATEGORY_SEEDS]
    non_seed = sorted(non_seed, key=lambda x: (-x[1], x[0]))
    if topk <= 0:
        return seed_entries + non_seed
    return seed_entries + non_seed[:topk]


def _load_existing_state(out_jsonl_path: str) -> Tuple[Set[str], Dict[str, int], Dict[str, str]]:
    if not os.path.exists(out_jsonl_path):
        return set(), {}, {}
    seen: Set[str] = set()
    counts: Dict[str, int] = {}
    alias_map: Dict[str, str] = {}
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
            category = _title_case_words(row.get("category", ""), max_words=3)
            category_raw = _title_case_words(row.get("category_raw", ""), max_words=3)
            if category:
                counts[category] = int(counts.get(category, 0)) + 1
            if category and category_raw and _normalize_label_key(category) != _normalize_label_key(category_raw):
                alias_map[_normalize_label_key(category_raw)] = category
    return seen, counts, alias_map


def _build_prompt(subset: str, doc_id: str, doc_desc: str, known_labels_block: str) -> str:
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
        doc_id=doc_id,
        doc_desc=doc_desc,
    )


def _parse_output(text: str) -> Tuple[str, str, List[str]]:
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
        return "Auxiliary Context", "", []
    category_raw = _normalize_category_label(obj.get("category", ""))
    canonical_hint = _title_case_words(obj.get("canonical_label", ""), max_words=3)
    merge_from = _normalize_list(obj.get("merge_from", []), max_items=6)
    return category_raw, canonical_hint, merge_from


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
    label_shortlist_topk: int,
    label_alias_threshold: float,
) -> Tuple[List[Dict], List[str]]:
    shortlist_with_counts = _build_shortlist(label_counts, alias_map, topk=label_shortlist_topk)
    shortlist_labels = [label for label, _ in shortlist_with_counts]
    if shortlist_with_counts:
        known_labels_block = "\n".join([f"- {label} (count={count})" for label, count in shortlist_with_counts])
    else:
        known_labels_block = "- None yet (create a new abstract category if needed)"

    prompts: List[str] = []
    trimmed_count = 0
    for _, row in batch_rows:
        doc_id = str(row.get("doc_id", "")).strip()
        desc = _truncate_words(row.get("desc", ""), max_desc_words)
        guarded_prompt, was_trimmed = _build_guarded_prompt(
            subset=subset,
            doc_id=doc_id,
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
    for (_, row), output in zip(batch_rows, outputs):
        text = output.outputs[0].text if output.outputs else ""
        category_raw, canonical_hint, merge_from = _parse_output(text)
        category, label_decision, merge_applied = _resolve_category(
            generated_category=category_raw,
            canonical_hint=canonical_hint,
            merge_from=merge_from,
            shortlist_labels=shortlist_labels,
            label_counts=label_counts,
            alias_map=alias_map,
            alias_threshold=label_alias_threshold,
        )
        label_counts[category] = int(label_counts.get(category, 0)) + 1
        classified.append({
            "row": row,
            "category": category,
            "category_raw": category_raw,
            "label_decision": label_decision,
            "canonical_hint": canonical_hint,
            "merge_from": merge_from,
            "merge_applied": merge_applied,
            "shortlist_labels": shortlist_labels,
        })
    return classified, shortlist_labels


def _resolve_category(
    *,
    generated_category: str,
    canonical_hint: str,
    merge_from: List[str],
    shortlist_labels: List[str],
    label_counts: Dict[str, int],
    alias_map: Dict[str, str],
    alias_threshold: float,
) -> Tuple[str, str, List[str]]:
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

    return chosen, decision, merge_applied


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
    label_shortlist_topk: int,
    label_alias_threshold: float,
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

    classified, _ = _classify_batch_rows(
        llm=llm,
        sampling_params=sampling_params,
        tokenizer=tokenizer,
        prompt_token_limit=prompt_token_limit,
        subset=subset,
        max_desc_words=max_desc_words,
        batch_rows=batch_rows,
        label_counts=label_counts,
        alias_map=alias_map,
        label_shortlist_topk=label_shortlist_topk,
        label_alias_threshold=label_alias_threshold,
    )
    new_count = 0
    for item in classified:
        row = item["row"]
        category = str(item["category"])
        record = {
            "doc_id": str(row.get("doc_id", "")),
            "category": category,
        }
        if debug_output:
            record.update({
                "category_raw": item["category_raw"],
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


def _run_nonleaf_warmup(
    *,
    llm: LLM,
    sampling_params: SamplingParams,
    tokenizer,
    prompt_token_limit: int,
    node_catalog_path: str,
    subset: str,
    depths: Set[int],
    max_nonleaf_desc_words: int,
    batch_size: int,
    label_counts: Dict[str, int],
    alias_map: Dict[str, str],
    label_shortlist_topk: int,
    label_alias_threshold: float,
    max_nodes: int,
    print_every: int,
) -> int:
    if not depths:
        return 0
    warmup_batch: List[Tuple[int, Dict]] = []
    warmup_processed = 0
    for node_idx, row in _iter_nonleaf_units(node_catalog_path, depths):
        if max_nodes > 0 and warmup_processed >= max_nodes:
            break
        warmup_batch.append((node_idx, row))
        warmup_processed += 1
        if len(warmup_batch) < batch_size:
            continue
        _classify_batch_rows(
            llm=llm,
            sampling_params=sampling_params,
            tokenizer=tokenizer,
            prompt_token_limit=prompt_token_limit,
            subset=subset,
            max_desc_words=max_nonleaf_desc_words,
            batch_rows=warmup_batch,
            label_counts=label_counts,
            alias_map=alias_map,
            label_shortlist_topk=label_shortlist_topk,
            label_alias_threshold=label_alias_threshold,
        )
        warmup_batch = []
        if print_every > 0 and warmup_processed % print_every == 0:
            print(
                f"[Warmup] processed_nonleaf={warmup_processed} "
                f"num_categories={len(_get_canonical_label_counts(label_counts, alias_map))}"
            )

    if warmup_batch:
        _classify_batch_rows(
            llm=llm,
            sampling_params=sampling_params,
            tokenizer=tokenizer,
            prompt_token_limit=prompt_token_limit,
            subset=subset,
            max_desc_words=max_nonleaf_desc_words,
            batch_rows=warmup_batch,
            label_counts=label_counts,
            alias_map=alias_map,
            label_shortlist_topk=label_shortlist_topk,
            label_alias_threshold=label_alias_threshold,
        )
    return warmup_processed


def _save_registry(path: str, label_counts: Dict[str, int], alias_map: Dict[str, str], meta: Dict[str, object]) -> None:
    canonical_counts = _get_canonical_label_counts(label_counts, alias_map)
    categories = [{"label": label, "count": int(count)} for label, count in sorted(canonical_counts.items(), key=lambda x: (-x[1], x[0]))]
    payload = {
        "meta": meta,
        "num_categories": len(categories),
        "categories": categories,
        "alias_map": alias_map,
    }
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="One-pass abstract category assignment per document (local vLLM).")
    parser.add_argument("--dataset", type=str, default="BRIGHT")
    parser.add_argument("--subset", type=str, required=True)
    parser.add_argument("--node_catalog_path", type=str, default=None)
    parser.add_argument("--out_jsonl", type=str, default=None)
    parser.add_argument("--out_registry_json", type=str, default=None)
    parser.add_argument("--prompt_name", type=str, default="category_assign_v1")
    parser.add_argument("--llm", type=str, required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_desc_words", type=int, default=12000)
    parser.add_argument("--max_nonleaf_desc_words", type=int, default=12000)
    parser.add_argument("--hybrid_depths", type=str, default="1,2,3")
    parser.add_argument("--hybrid_warmup_max_nodes", type=int, default=0)
    parser.add_argument("--no_hybrid_warmup", action="store_true", default=False)
    parser.add_argument("--start_doc_idx", type=int, default=0)
    parser.add_argument("--end_doc_idx", type=int, default=-1)
    parser.add_argument("--start_leaf_idx", type=int, default=None)
    parser.add_argument("--end_leaf_idx", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--print_every", type=int, default=200)
    parser.add_argument("--label_shortlist_topk", type=int, default=12)
    parser.add_argument("--label_alias_threshold", type=float, default=0.90)
    parser.add_argument("--hard_prompt_token_limit", type=int, default=16384)
    parser.add_argument("--prompt_token_margin", type=int, default=32)
    parser.add_argument("--debug_output", action="store_true", default=False)
    args = parser.parse_args()

    if args.start_leaf_idx is not None:
        # Intent: backward compatibility for existing scripts that still pass leaf-index ranges.
        args.start_doc_idx = int(args.start_leaf_idx)
    if args.end_leaf_idx is not None:
        # Intent: backward compatibility for existing scripts that still pass leaf-index ranges.
        args.end_doc_idx = int(args.end_leaf_idx)

    node_catalog_path = args.node_catalog_path or _default_node_catalog_path(args.dataset, args.subset)
    out_jsonl = args.out_jsonl or _default_output_path(args.dataset, args.subset, args.prompt_name)
    out_registry_json = args.out_registry_json or _default_registry_path(args.dataset, args.subset, args.prompt_name)
    if not os.path.exists(node_catalog_path):
        raise FileNotFoundError(f"Missing node catalog: {node_catalog_path}")

    if args.overwrite and os.path.exists(out_jsonl):
        os.remove(out_jsonl)

    _ensure_parent_dir(out_jsonl)
    if args.overwrite:
        seen_doc_ids, label_counts, alias_map = set(), {}, {}
    else:
        seen_doc_ids, label_counts, alias_map = _load_existing_state(out_jsonl)

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

    hybrid_depths = _parse_depths(args.hybrid_depths)
    warmup_nodes = 0
    if not args.no_hybrid_warmup and hybrid_depths:
        # Intent: warm up abstract category shortlist from non-leaf summaries before leaf-level assignment.
        warmup_nodes = _run_nonleaf_warmup(
            llm=llm,
            sampling_params=sampling_params,
            tokenizer=tokenizer,
            prompt_token_limit=prompt_token_limit,
            node_catalog_path=node_catalog_path,
            subset=args.subset,
            depths=hybrid_depths,
            max_nonleaf_desc_words=args.max_nonleaf_desc_words,
            batch_size=args.batch_size,
            label_counts=label_counts,
            alias_map=alias_map,
            label_shortlist_topk=args.label_shortlist_topk,
            label_alias_threshold=args.label_alias_threshold,
            max_nodes=args.hybrid_warmup_max_nodes,
            print_every=args.print_every,
        )
        print(
            f"[Warmup Done] processed_nonleaf={warmup_nodes} "
            f"num_categories={len(_get_canonical_label_counts(label_counts, alias_map))}"
        )

    generated = 0
    skipped_existing = 0
    skipped_range = 0
    batch_rows: List[Tuple[int, Dict]] = []

    with open(out_jsonl, "a", encoding="utf-8") as out_file:
        for doc_idx, row in _iter_document_units(node_catalog_path, args.subset):
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
                label_shortlist_topk=args.label_shortlist_topk,
                label_alias_threshold=args.label_alias_threshold,
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
                label_shortlist_topk=args.label_shortlist_topk,
                label_alias_threshold=args.label_alias_threshold,
                limit=args.limit,
                generated_so_far=generated,
            )

    created_at = datetime.datetime.utcnow().isoformat() + "Z"
    _save_registry(
        out_registry_json,
        label_counts,
        alias_map,
        meta={
            "dataset": args.dataset,
            "subset": args.subset,
            "prompt_name": args.prompt_name,
            "model": args.llm,
            "hybrid_depths": sorted(hybrid_depths),
            "warmup_nonleaf_nodes": int(warmup_nodes),
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
