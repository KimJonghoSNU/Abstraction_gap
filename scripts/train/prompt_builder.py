#!/usr/bin/env python3
"""
Prompt builder for traversal policy training.

Supports two modes:
1) Default mode: delegate to src/prompts.py:get_traversal_prompt
2) Custom template mode: read template text file and replace tokens
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np

from prompts import get_desc_str_from_list, get_relevance_definition, get_traversal_prompt


class TraversalPromptBuilder:
    def __init__(
        self,
        *,
        max_prompt_proto_size: int,
        max_desc_char_len: int,
        prompt_template_file: str,
    ) -> None:
        self.max_prompt_proto_size = int(max_prompt_proto_size)
        self.max_desc_char_len = int(max_desc_char_len)
        self.prompt_template_file = str(prompt_template_file or "").strip()

        self.template_text = ""
        if self.prompt_template_file:
            path = Path(self.prompt_template_file)
            if not path.exists():
                raise FileNotFoundError(f"Prompt template file not found: {path}")
            self.template_text = path.read_text(encoding="utf-8")

    @property
    def using_custom_template(self) -> bool:
        return bool(self.template_text)

    @staticmethod
    def _get_content_proto_size(text: str) -> int:
        return len(text.encode("utf-8"))

    def _build_with_custom_template(
        self,
        *,
        query: str,
        candidate_descs: Sequence[str],
        subset: str,
    ) -> str:
        cur_desc_char_len = self.max_desc_char_len if self.max_desc_char_len > 0 else None

        while True:
            candidate_block = get_desc_str_from_list(candidate_descs, cur_desc_char_len)
            prompt = self.template_text
            prompt = prompt.replace("{{QUERY}}", str(query).replace("\n", "  "))
            prompt = prompt.replace("{{CANDIDATES}}", candidate_block)
            prompt = prompt.replace("{{RELEVANCE_DEFINITION}}", get_relevance_definition(subset))

            if (
                self.max_prompt_proto_size > 0
                and self._get_content_proto_size(prompt) > self.max_prompt_proto_size
                and cur_desc_char_len is not None
                and cur_desc_char_len > 100
            ):
                # Intent: keep prompt size bounded by trimming candidate descriptions while preserving template structure.
                cur_desc_char_len -= 100
                continue
            return prompt

    def _build_with_default_prompt(
        self,
        *,
        query: str,
        candidate_descs: Sequence[str],
        subset: str,
    ) -> str:
        hp = SimpleNamespace(
            SUBSET=subset,
            MAX_PROMPT_PROTO_SIZE=self.max_prompt_proto_size,
            MAX_DOC_DESC_CHAR_LEN=self.max_desc_char_len,
        )
        logger = SimpleNamespace(debug=lambda *args, **kwargs: None)
        prompt = get_traversal_prompt(
            query=query,
            child_desc_list=list(candidate_descs),
            hp=hp,
            logger=logger,
            return_constraint=False,
            leaf_cluster=False,
        )
        return str(prompt)

    def build(
        self,
        *,
        query: str,
        candidate_descs: Sequence[str],
        subset: str,
    ) -> str:
        if self.using_custom_template:
            return self._build_with_custom_template(
                query=query,
                candidate_descs=candidate_descs,
                subset=subset,
            )
        return self._build_with_default_prompt(
            query=query,
            candidate_descs=candidate_descs,
            subset=subset,
        )
