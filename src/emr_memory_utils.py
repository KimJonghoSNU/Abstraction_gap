import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


EMR_CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
EMR_REWRITE_OUTPUT_RESERVE_TOKENS = 1024
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def compact_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def build_doc_text_map(docs_df) -> Dict[str, str]:
    text_col = None
    for candidate in ["content", "contents", "text"]:
        if candidate in docs_df.columns:
            text_col = candidate
            break
    if text_col is None:
        return {}
    return {str(row["id"]): str(row[text_col] or "") for _, row in docs_df.iterrows() if str(row.get("id", "")).strip()}


class EmrMemoryCompressor:
    def __init__(self, model_name: str, sent_topk: int, logger: logging.Logger):
        self.model_name = str(model_name)
        self.sent_topk = max(1, int(sent_topk))
        self.logger = logger
        self._tokenizer = None
        self._model = None
        self._device = "cpu"
        self._nlp = None
        self._sentence_cache: Dict[str, List[str]] = {}

    def _ensure_model(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        self.logger.info("Loading EMR cross-encoder model: %s", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.eval()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)

    def _ensure_sentence_splitter(self):
        if self._nlp is not None:
            return self._nlp
        try:
            import spacy

            self._nlp = spacy.blank("en")
            if "sentencizer" not in self._nlp.pipe_names:
                self._nlp.add_pipe("sentencizer")
        except Exception as exc:
            self.logger.warning("spaCy unavailable for EMR sentence segmentation; using regex fallback (%s)", exc)
            self._nlp = False
        return self._nlp

    def _split_sentences(self, text: str) -> List[str]:
        normalized = compact_text(text)
        if not normalized:
            return []
        nlp = self._ensure_sentence_splitter()
        if nlp:
            try:
                doc = nlp(normalized)
                sentences = [sent.text.strip() for sent in doc.sents if sent.text and sent.text.strip()]
                if sentences:
                    return sentences
            except Exception as exc:
                self.logger.warning("spaCy sentencizer failed; using regex fallback (%s)", exc)
        return [piece.strip() for piece in SENTENCE_SPLIT_RE.split(normalized) if piece and piece.strip()]

    def _score_sentences(self, query: str, sentences: Sequence[str]) -> List[float]:
        self._ensure_model()
        if self._tokenizer is None or self._model is None:
            raise RuntimeError("EMR cross-encoder model failed to initialize")
        scores: List[float] = []
        batch_size = 32
        for start in range(0, len(sentences), batch_size):
            sent_batch = list(sentences[start:start + batch_size])
            query_batch = [query for _ in sent_batch]
            encoded = self._tokenizer(
                query_batch,
                sent_batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: val.to(self._device) for key, val in encoded.items()}
            with torch.no_grad():
                outputs = self._model(**encoded)
            logits = outputs.logits
            if logits.ndim == 2 and logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
            elif logits.ndim == 2:
                logits = logits[:, 0]
            scores.extend([float(x) for x in logits.detach().cpu().tolist()])
        return scores

    def _get_doc_sentences(self, doc_id: str, document_text: str) -> List[str]:
        doc_key = str(doc_id or "").strip()
        cached = self._sentence_cache.get(doc_key)
        if cached is not None:
            return cached
        sentences = self._split_sentences(document_text)
        self._sentence_cache[doc_key] = sentences
        return sentences

    def build_global_memory_items(
        self,
        query: str,
        doc_ids: Sequence[str],
        doc_text_by_id: Dict[str, str],
        compression_mode: str = "on",
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        mode = str(compression_mode or "on").lower()
        if mode == "off":
            memory_items: List[Dict[str, Any]] = []
            for doc_order, raw_doc_id in enumerate(doc_ids):
                doc_id = str(raw_doc_id or "").strip()
                if not doc_id:
                    continue
                doc_text = compact_text(str(doc_text_by_id.get(doc_id, "") or ""))
                if not doc_text:
                    continue
                memory_items.append({
                    "doc_id": doc_id,
                    "text": doc_text,
                    "last_rank": int(doc_order),
                })
            return memory_items, []

        sentence_rows: List[Dict[str, Any]] = []
        doc_order_map: Dict[str, int] = {}
        doc_sentences_map: Dict[str, List[str]] = {}
        for doc_order, raw_doc_id in enumerate(doc_ids):
            doc_id = str(raw_doc_id or "").strip()
            if not doc_id:
                continue
            doc_text = str(doc_text_by_id.get(doc_id, "") or "")
            if not doc_text.strip():
                continue
            sentences = self._get_doc_sentences(doc_id, doc_text)
            if not sentences:
                continue
            doc_order_map[doc_id] = int(doc_order)
            doc_sentences_map[doc_id] = sentences
            for sent_idx, sentence in enumerate(sentences):
                sentence_rows.append({
                    "doc_id": doc_id,
                    "sent_idx": int(sent_idx),
                    "text": sentence,
                    "doc_order": int(doc_order),
                })
        if not sentence_rows:
            return [], []

        sentence_texts = [str(row["text"]) for row in sentence_rows]
        scores = self._score_sentences(str(query or "").strip(), sentence_texts)
        scored_rows: List[Dict[str, Any]] = []
        for row, score in zip(sentence_rows, scores):
            scored = dict(row)
            scored["score"] = float(score)
            scored_rows.append(scored)
        scored_rows.sort(key=lambda row: row["score"], reverse=True)
        top_rows = scored_rows[: self.sent_topk]

        selected_by_doc: Dict[str, List[Dict[str, Any]]] = {}
        for row in top_rows:
            selected_by_doc.setdefault(str(row["doc_id"]), []).append(row)

        doc_best_score = {
            doc_id: max(float(row["score"]) for row in rows)
            for doc_id, rows in selected_by_doc.items()
        }
        ordered_doc_ids = sorted(
            selected_by_doc.keys(),
            key=lambda doc_id: (-doc_best_score[doc_id], int(doc_order_map.get(doc_id, 10**9))),
        )

        memory_items: List[Dict[str, Any]] = []
        for doc_id in ordered_doc_ids:
            rows = selected_by_doc[doc_id]
            selected_indices = {int(row["sent_idx"]) for row in rows}
            sentences = doc_sentences_map.get(doc_id, [])
            composed: List[str] = []
            for sent_idx, sentence in enumerate(sentences):
                if sent_idx in selected_indices:
                    composed.append(sentence)
                elif (not composed) or (composed[-1] != "..."):
                    composed.append("...")
            snippet = " ".join(composed).strip()
            if not snippet:
                continue
            memory_items.append({
                "doc_id": doc_id,
                "text": snippet,
                "best_score": float(doc_best_score[doc_id]),
                "last_rank": int(doc_order_map.get(doc_id, 10**9)),
            })
        return memory_items, [str(row["text"]) for row in top_rows]


class RewritePromptTokenBudgeter:
    def __init__(self, model_name: str, backend: str, logger: logging.Logger):
        self.model_name = str(model_name)
        self.backend = str(backend or "").lower()
        self.logger = logger
        self.tokenizer = None
        self.max_context_tokens: Optional[int] = None
        self._load()

    def _load(self) -> None:
        local_only = os.path.exists(self.model_name)
        config = None
        try:
            config = AutoConfig.from_pretrained(self.model_name, local_files_only=local_only)
        except Exception as exc:
            self.logger.warning("Could not load rewrite tokenizer config for %s (%s)", self.model_name, exc)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=local_only, use_fast=False)
        except Exception as exc:
            self.logger.warning("Could not load rewrite tokenizer for %s (%s)", self.model_name, exc)
            self.tokenizer = None
        candidates: List[int] = []
        if self.tokenizer is not None:
            tok_max = getattr(self.tokenizer, "model_max_length", None)
            if isinstance(tok_max, int) and 0 < tok_max < 10**7:
                candidates.append(int(tok_max))
        if config is not None:
            cfg_max = getattr(config, "max_position_embeddings", None)
            if isinstance(cfg_max, int) and cfg_max > 0:
                candidates.append(int(cfg_max))
        self.max_context_tokens = max(candidates) if candidates else None
        if self.tokenizer is not None and self.max_context_tokens is not None:
            self.logger.info(
                "Loaded rewrite tokenizer budgeter | model=%s | backend=%s | max_context_tokens=%d",
                self.model_name,
                self.backend,
                int(self.max_context_tokens),
            )

    def count_text_tokens(self, text: str) -> Optional[int]:
        if self.tokenizer is None:
            return None
        try:
            token_ids = self.tokenizer.encode(str(text or ""), add_special_tokens=False)
            return int(len(token_ids))
        except Exception as exc:
            self.logger.warning("Failed to tokenize rewrite text (%s)", exc)
            return None

    def count_prompt_tokens(self, prompt_text: str) -> Optional[int]:
        if self.tokenizer is None:
            return None
        try:
            if self.backend == "vllm" and hasattr(self.tokenizer, "apply_chat_template"):
                token_ids = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": str(prompt_text or "")}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                return int(len(token_ids))
            token_ids = self.tokenizer.encode(str(prompt_text or ""), add_special_tokens=True)
            return int(len(token_ids))
        except Exception as exc:
            self.logger.warning("Failed to tokenize rewrite prompt (%s)", exc)
            return None


def format_emr_history(history_entries: Sequence[Dict[str, Any]]) -> str:
    if not history_entries:
        return "No actions have been taken yet."
    lines: List[str] = []
    for idx, entry in enumerate(history_entries, start=1):
        query = compact_text(str(entry.get("query", "") or ""))
        ranks = ", ".join([str(x) for x in list(entry.get("ranks", [])) if str(x).strip()])
        lines.append(f"[{idx}] Query : {query} Ranks : {ranks}")
    return "\n".join(lines)


def format_emr_doc_memory(memory_items: Sequence[Dict[str, Any]]) -> str:
    if not memory_items:
        return "No documents have been stored yet."
    return "\n".join([
        f"[{str(item.get('doc_id', '')).strip()}] {compact_text(str(item.get('text', '') or ''))}"
        for item in memory_items
        if str(item.get("doc_id", "")).strip() and str(item.get("text", "")).strip()
    ])


def build_current_doc_memory_items(
    *,
    retrieved_doc_ids: Sequence[str],
    doc_text_by_id: Dict[str, str],
    doc_topk: int,
) -> List[str]:
    return [
        str(doc_id)
        for doc_id in list(retrieved_doc_ids)[: max(1, int(doc_topk))]
        if str(doc_id).strip() and str(doc_text_by_id.get(str(doc_id), "") or "").strip()
    ]


def update_accumulated_doc_bank(
    doc_bank: Dict[str, Dict[str, Any]],
    *,
    next_order: int,
    current_doc_ids: Sequence[str],
    iter_idx: int,
    query_for_memory: str,
) -> int:
    for rank, raw_doc_id in enumerate(current_doc_ids, start=1):
        doc_id = str(raw_doc_id or "")
        if not doc_id:
            continue
        existing = dict(doc_bank.get(doc_id, {}))
        if not existing:
            existing["first_seen_iter"] = int(iter_idx)
            existing["insertion_order"] = int(next_order)
            next_order += 1
        # Intent: cumulative EMR memory tracks a growing document pool; sentence composition is done later with paper-style global filtering.
        existing.update({
            "doc_id": doc_id,
            "last_seen_iter": int(iter_idx),
            "last_query": str(query_for_memory or ""),
            "last_rank": int(rank),
        })
        doc_bank[doc_id] = existing
    return int(next_order)


def materialize_accumulated_doc_pool(doc_bank: Dict[str, Dict[str, Any]]) -> List[str]:
    items = list(dict(doc_bank or {}).values())
    items.sort(
        key=lambda item: (
            -int(item.get("last_seen_iter", -1) or -1),
            int(item.get("last_rank", 10**9) or 10**9),
            int(item.get("insertion_order", 10**9) or 10**9),
        )
    )
    return [str(item.get("doc_id", "") or "") for item in items if str(item.get("doc_id", "")).strip()]


def fit_emr_memory_to_token_budget(
    *,
    render_prompt: Callable[[str, str], str],
    history_text: str,
    memory_items: Sequence[Dict[str, Any]],
    prompt_budgeter: Optional[RewritePromptTokenBudgeter],
    output_reserve_tokens: int,
    max_memory_tokens: Optional[int],
) -> Dict[str, Any]:
    base_memory_text = format_emr_doc_memory([])
    prompt_without_docs = render_prompt(history_text, base_memory_text)
    if prompt_budgeter is None:
        return {
            "memory_items": list(memory_items),
            "memory_text": format_emr_doc_memory(memory_items),
            "overflow_dropped": [],
            "prompt_total_tokens": None,
            "memory_prompt_tokens": None,
            "prompt_budget_tokens": None,
            "model_context_tokens": None,
        }

    prompt_budget_tokens = None
    if prompt_budgeter.max_context_tokens is not None:
        prompt_budget_tokens = max(1, int(prompt_budgeter.max_context_tokens) - max(0, int(output_reserve_tokens)))
    base_prompt_tokens = prompt_budgeter.count_prompt_tokens(prompt_without_docs)
    if prompt_budget_tokens is None or base_prompt_tokens is None:
        final_memory_text = format_emr_doc_memory(memory_items)
        final_prompt = render_prompt(history_text, final_memory_text)
        return {
            "memory_items": list(memory_items),
            "memory_text": final_memory_text,
            "overflow_dropped": [],
            "prompt_total_tokens": prompt_budgeter.count_prompt_tokens(final_prompt),
            "memory_prompt_tokens": prompt_budgeter.count_text_tokens(final_memory_text),
            "prompt_budget_tokens": prompt_budget_tokens,
            "model_context_tokens": prompt_budgeter.max_context_tokens,
        }

    if base_prompt_tokens > prompt_budget_tokens:
        # Intent: if the non-document prompt already exceeds the window, drop document memory first rather than silently truncating the whole prompt.
        return {
            "memory_items": [],
            "memory_text": base_memory_text,
            "overflow_dropped": [str(item.get("doc_id", "") or "") for item in memory_items if str(item.get("doc_id", "")).strip()],
            "prompt_total_tokens": base_prompt_tokens,
            "memory_prompt_tokens": prompt_budgeter.count_text_tokens(base_memory_text),
            "prompt_budget_tokens": prompt_budget_tokens,
            "model_context_tokens": prompt_budgeter.max_context_tokens,
        }

    kept: List[Dict[str, Any]] = []
    dropped: List[str] = []
    current_memory_text = base_memory_text
    current_prompt_tokens = base_prompt_tokens
    current_memory_tokens = prompt_budgeter.count_text_tokens(current_memory_text)
    token_cap = int(max_memory_tokens) if max_memory_tokens and int(max_memory_tokens) > 0 else None
    for idx, item in enumerate(memory_items):
        candidate_items = kept + [item]
        candidate_memory_text = format_emr_doc_memory(candidate_items)
        candidate_prompt = render_prompt(history_text, candidate_memory_text)
        candidate_prompt_tokens = prompt_budgeter.count_prompt_tokens(candidate_prompt)
        candidate_memory_tokens = prompt_budgeter.count_text_tokens(candidate_memory_text)
        fits_prompt = candidate_prompt_tokens is not None and candidate_prompt_tokens <= prompt_budget_tokens
        fits_memory = token_cap is None or (candidate_memory_tokens is not None and candidate_memory_tokens <= token_cap)
        if fits_prompt and fits_memory:
            kept = candidate_items
            current_memory_text = candidate_memory_text
            current_prompt_tokens = candidate_prompt_tokens
            current_memory_tokens = candidate_memory_tokens
            continue
        dropped.extend([
            str(rem_item.get("doc_id", "") or "")
            for rem_item in memory_items[idx:]
            if str(rem_item.get("doc_id", "")).strip()
        ])
        break
    return {
        "memory_items": kept,
        "memory_text": current_memory_text,
        "overflow_dropped": dropped,
        "prompt_total_tokens": current_prompt_tokens,
        "memory_prompt_tokens": current_memory_tokens,
        "prompt_budget_tokens": prompt_budget_tokens,
        "model_context_tokens": prompt_budgeter.max_context_tokens,
    }


def build_emr_prompt_memory(
    *,
    mode: str,
    history_entries: Sequence[Dict[str, Any]],
    doc_bank: Dict[str, Dict[str, Any]],
    query_for_memory: str,
    compressor: Optional[EmrMemoryCompressor],
    doc_text_by_id: Dict[str, str],
    render_prompt: Callable[[str, str], str],
    prompt_budgeter: Optional[RewritePromptTokenBudgeter],
    compression_mode: str,
    max_memory_tokens: Optional[int],
    memory_source_label: str = "accumulated",
) -> Dict[str, Any]:
    compression_mode_s = str(compression_mode or "on").lower()
    history_text = format_emr_history(history_entries)
    if str(mode or "off").lower() == "off":
        return {
            "history_text": history_text,
            "memory_text": "",
            "memory_doc_ids": [],
            "memory_items": [],
            "overflow_dropped": [],
            "memory_source": "off",
            "memory_pool_doc_ids": [],
            "selected_sentences": [],
            "compression_mode": compression_mode_s,
            "prompt_total_tokens": None,
            "memory_prompt_tokens": None,
            "prompt_budget_tokens": None,
            "model_context_tokens": prompt_budgeter.max_context_tokens if prompt_budgeter is not None else None,
        }
    if compressor is None:
        raise ValueError("EMR memory mode requested but compressor is not initialized")

    mode_s = str(mode).lower()
    if mode_s == "accumulated":
        memory_pool_doc_ids = materialize_accumulated_doc_pool(doc_bank)
        memory_source = str(memory_source_label or "accumulated")
    else:
        raise ValueError(f"Unknown EMR memory mode: {mode}")
    memory_items, selected_sentences = compressor.build_global_memory_items(
        query=query_for_memory,
        doc_ids=memory_pool_doc_ids,
        doc_text_by_id=doc_text_by_id,
        compression_mode=compression_mode_s,
    )
    fitted = fit_emr_memory_to_token_budget(
        render_prompt=render_prompt,
        history_text=history_text,
        memory_items=memory_items,
        prompt_budgeter=prompt_budgeter,
        output_reserve_tokens=EMR_REWRITE_OUTPUT_RESERVE_TOKENS,
        max_memory_tokens=max_memory_tokens,
    )
    return {
        "history_text": history_text,
        "memory_text": fitted["memory_text"],
        "memory_doc_ids": [str(item.get("doc_id", "") or "") for item in fitted["memory_items"] if str(item.get("doc_id", "")).strip()],
        "memory_items": fitted["memory_items"],
        "overflow_dropped": list(fitted["overflow_dropped"]),
        "memory_source": memory_source,
        "memory_pool_doc_ids": list(memory_pool_doc_ids),
        "selected_sentences": list(selected_sentences),
        "compression_mode": compression_mode_s,
        "prompt_total_tokens": fitted["prompt_total_tokens"],
        "memory_prompt_tokens": fitted["memory_prompt_tokens"],
        "prompt_budget_tokens": fitted["prompt_budget_tokens"],
        "model_context_tokens": fitted["model_context_tokens"],
    }


def append_emr_history_entry(
    history_entries: List[Dict[str, Any]],
    *,
    applied_query: str,
    retrieved_doc_ids: Sequence[str],
    history_rank_topk: int,
) -> None:
    # Intent: write the current rewrite step only after the rewrite is applied so the next prompt sees a completed state trace.
    history_entries.append({
        "query": str(applied_query or ""),
        "ranks": [str(x) for x in list(retrieved_doc_ids)[: max(1, int(history_rank_topk))]],
    })
