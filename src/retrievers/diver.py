from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

TextOrTexts = Union[str, Sequence[str]]


@dataclass(frozen=True)
class RetrievalResult:
    indices: np.ndarray  # int64, shape [K]
    scores: np.ndarray   # float32, shape [K]


class DiverEmbeddingModel:
    """
    Ported from `previous/shortcut_reranker/scripts/query_generator.py`.

    - left padding tokenizer
    - last-token pooling (left padding aware)
    - L2-normalized embeddings
    - query encoded with an instruction prefix
    """

    def __init__(
        self,
        model_path: str,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
        device: str | None = None,
        local_files_only: bool = True,
        task_description: str = "",  # Added task_description parameter
    ):
        # NOTE: Passing `device_map="auto"` requires `accelerate`. For this repo we default
        # to single-device loading to keep dependencies minimal.
        model_kwargs = {
            "local_files_only": local_files_only,
        }

        # transformers is transitioning from `torch_dtype` -> `dtype` in some versions.
        # Use `dtype` when available, but fall back for compatibility.
        try:
            self.model = AutoModel.from_pretrained(model_path, dtype=torch_dtype, device_map=device_map, **model_kwargs).eval()
        except:
            self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch_dtype, device_map=device_map, **model_kwargs).eval()

        if device_map is None:
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
            local_files_only=local_files_only,
        )
        self.task_description = task_description

    def _get_detailed_instruct(self, query: str) -> str:
        return f"Instruct: {self.task_description}\nQuery:{query}"

    def _last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    @torch.inference_mode()
    def encode(
        self,
        texts: TextOrTexts,
        *,
        max_length: int = 16384,
        batch_size: int = 4,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts_list = [texts]
        else:
            texts_list = list(texts)

        if len(texts_list) == 0:
            return np.zeros((0, 0), dtype=np.float32)

        out: List[np.ndarray] = []
        for start in range(0, len(texts_list), batch_size): #, desc="Encoding texts", total=(len(texts_list) + batch_size - 1) // batch_size):
            batch_texts = texts_list[start : start + batch_size]
            batch = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch = {k: v.to(self.model.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            pooled = self._last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=1)
            out.append(pooled.detach().cpu().numpy().astype(np.float32, copy=False))

        return np.vstack(out)

    def encode_query(self, query: str, *, max_length: int = 8192, batch_size: int = 4) -> np.ndarray:
        return self.encode(self._get_detailed_instruct(query), max_length=max_length, batch_size=batch_size)[0]

    def encode_docs(self, docs: Sequence[str], *, max_length: int = 16384, batch_size: int = 2) -> np.ndarray:
        docs_clean = [d for d in docs if isinstance(d, str) and d.strip()]
        return self.encode(docs_clean, max_length=max_length, batch_size=batch_size)


def _topk(scores: np.ndarray, k: int) -> RetrievalResult:
    if k <= 0:
        return RetrievalResult(indices=np.array([], dtype=np.int64), scores=np.array([], dtype=np.float32))
    k = min(k, scores.shape[0])
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return RetrievalResult(indices=idx.astype(np.int64, copy=False), scores=scores[idx].astype(np.float32, copy=False))


def cosine_topk(query_emb: np.ndarray, doc_embs: np.ndarray, topk: int) -> RetrievalResult:
    """
    Computes cosine similarity via dot product assuming L2-normalized embeddings.
    query_emb: shape [D]
    doc_embs: shape [N, D]
    """
    if query_emb.ndim != 1:
        raise ValueError(f"query_emb must be 1D, got {query_emb.shape}")
    if doc_embs.ndim != 2:
        raise ValueError(f"doc_embs must be 2D, got {doc_embs.shape}")
    if doc_embs.shape[0] == 0:
        return RetrievalResult(indices=np.array([], dtype=np.int64), scores=np.array([], dtype=np.float32))
    if doc_embs.shape[1] != query_emb.shape[0]:
        raise ValueError(f"dim mismatch: doc_embs {doc_embs.shape[1]} vs query_emb {query_emb.shape[0]}")

    scores = (doc_embs @ query_emb).astype(np.float32, copy=False)
    return _topk(scores, topk)

