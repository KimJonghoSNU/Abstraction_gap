from __future__ import annotations

import re


_WS_RE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    cleaned = str(text or "")
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = cleaned.replace("\\n", " ").replace("\n", " ")
    return _WS_RE.sub(" ", cleaned).strip()


def _truncate_chars(text: str, max_chars: int) -> str:
    s = str(text or "").strip()
    if max_chars <= 0:
        return ""
    if len(s) <= max_chars:
        return s
    cut = s[:max_chars].rstrip()
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0].rstrip()
    return (cut + " ...").strip()


def _truncate_words(text: str, max_words: int) -> str:
    if max_words <= 0:
        return ""
    words = str(text or "").split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def build_abstractive_summary_prompt(
    query: str,
    text: str,
    max_summary_words: int = 64,
    max_source_words: int = 420,
) -> str:
    source = _truncate_words(_normalize_text(text), max_source_words)
    return (
        "You compress retrieved evidence into a concise support summary.\n"
        "Rules:\n"
        "- Focus only on information that can support answering the query.\n"
        "- Keep key technical nouns, entities, and causal relations.\n"
        "- Do not add new facts.\n"
        "- No bullets, no JSON, no prefacing words.\n"
        f"- Maximum {max(16, int(max_summary_words))} words.\n\n"
        f"Query:\n{str(query or '').strip()}\n\n"
        f"Evidence:\n{source}\n\n"
        "Summary:"
    )


def parse_abstractive_summary_output(
    text: str,
    max_chars: int = 320,
) -> str:
    raw = str(text or "").strip()
    if "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()
    if "```" in raw:
        try:
            parts = raw.split("```")
            fenced = [parts[i] for i in range(1, len(parts), 2)]
            if fenced:
                raw = fenced[-1].strip()
        except Exception:
            pass
    raw = raw.replace("Summary:", "").replace("summary:", "").strip()
    return _truncate_chars(_normalize_text(raw), max_chars)


def fallback_extractive_summary(
    text: str,
    max_words: int = 64,
    max_chars: int = 320,
) -> str:
    normalized = _normalize_text(text)
    if not normalized:
        return ""
    return _truncate_chars(_truncate_words(normalized, max_words), max_chars)
