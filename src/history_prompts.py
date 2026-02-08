from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple


Path = Tuple[int, ...]


def prepend_history_to_prompt(prompt: str, history_block: str) -> str:
    history_block = (history_block or "").strip()
    if not history_block:
        return prompt
    return f"{history_block}\n\n{prompt}"


def _paths_from_iter_record(iter_record: Mapping[str, object]) -> List[Path]:
    for key in ("anchor_eval_paths", "anchor_leaf_paths", "local_paths", "global_paths"):
        raw_paths = iter_record.get(key, [])
        if not raw_paths:
            continue
        return [tuple(int(x) for x in path) for path in raw_paths if path]
    return []


def _format_action_field(iter_record: Mapping[str, object]) -> str:
    query_action = str(iter_record.get("query_action", "") or "").strip().upper()
    if query_action:
        return query_action
    query_actions = iter_record.get("query_actions", {})
    if isinstance(query_actions, Mapping) and query_actions:
        parts_query: List[str] = []
        for key in ("Theory", "Entity", "Example", "Other"):
            if key in query_actions:
                parts_query.append(f"{key}:{str(query_actions.get(key, '')).strip().upper()}")
        if parts_query:
            return "; ".join(parts_query)
    action = str(iter_record.get("action", "") or "").strip().upper()
    if action:
        return action
    actions = iter_record.get("actions", {})
    if isinstance(actions, Mapping) and actions:
        parts: List[str] = []
        for key in ("Theory", "Entity", "Example", "Other"):
            if key in actions:
                parts.append(f"{key}:{str(actions.get(key, '')).strip().upper()}")
        if parts:
            return "; ".join(parts)
    return "(none)"


def _sanitize_cell(text: object) -> str:
    s = str(text or "").replace("\n", " ").replace("|", "/").strip()
    return s if s else "(none)"


def _format_summary_cell(iter_record: Mapping[str, object], topk: int) -> str:
    raw = iter_record.get("context_summaries", [])
    if (not isinstance(raw, Sequence)) or isinstance(raw, (str, bytes)):
        return ""
    pieces: List[str] = []
    for item in raw:
        s = _sanitize_cell(item)
        if s == "(none)":
            continue
        if len(s) > 220:
            s = s[:220].rstrip()
            if " " in s:
                s = s.rsplit(" ", 1)[0].rstrip()
            s = (s + " ...").strip()
        pieces.append(s)
        if len(pieces) >= topk:
            break
    return " || ".join(pieces)


def build_retrieval_history_block(
    iter_records: Sequence[Mapping[str, object]],
    path_to_doc_id: Dict[Path, str],
    topk: int = 10,
) -> str:
    if not iter_records:
        return ""
    topk = max(1, int(topk))
    lines = [
        "Retrieval History (all previous iterations; summarized evidence)",
        "| Iter | Action | Query Used | Evidence Summary (Top-K) |",
        "| --- | --- | --- | --- |",
    ]
    for rec in iter_records:
        if "iter" not in rec:
            continue
        iter_idx = rec.get("iter")
        query_t = _sanitize_cell(rec.get("query_t_history", rec.get("query_t", "")))
        action_t = _sanitize_cell(_format_action_field(rec))
        summarized = _format_summary_cell(rec, topk=topk)
        if summarized:
            lines.append(f"| {iter_idx} | {action_t} | {query_t} | {summarized} |")
            continue
        path_list = _paths_from_iter_record(rec)
        # Intent: keep only compact doc IDs (not document content) to avoid prompt bloat.
        doc_ids: List[str] = []
        seen: set[str] = set()
        for path in path_list:
            doc_id = path_to_doc_id.get(tuple(path))
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)
            doc_ids.append(str(doc_id))
            if len(doc_ids) >= topk:
                break
        joined = ", ".join(doc_ids) if doc_ids else "(none)"
        lines.append(f"| {iter_idx} | {action_t} | {query_t} | {joined} |")
    if len(lines) == 3:
        return ""
    return "\n".join(lines)
