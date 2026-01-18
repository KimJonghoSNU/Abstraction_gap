from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from retrievers.diver import DiverEmbeddingModel, cosine_topk


Path = Tuple[int, ...]


@dataclass(frozen=True)
class FlatHit:
    registry_idx: int
    path: Path
    score: float
    is_leaf: bool


def build_gates_and_leaf_candidates(
    *,
    hits: Sequence[FlatHit],
    gate_branches_topb: int,
) -> tuple[list[Path], list[tuple[Path, float]], Dict[Path, float]]:
    """
    Robust policy:
    - Do NOT promote-and-drop leaf hits; keep them as final candidates.
    - Gates are built primarily from retrieved branches.
    - If not enough branches, supplement with multi-depth ancestors of retrieved leaves.
    """
    leaf_hits = [h for h in hits if h.is_leaf]
    branch_hits = [h for h in hits if not h.is_leaf]

    flat_leaf_ranked = sorted([(h.path, h.score) for h in leaf_hits], key=lambda x: x[1], reverse=True)

    gates: List[Path] = []
    gate_scores: Dict[Path, float] = {}

    def add_gate(path: Path, score: float) -> None:
        nonlocal gates, gate_scores
        to_remove: List[Path] = []
        for existing in gates:
            if is_prefix(existing, path) or is_prefix(path, existing):
                if score > gate_scores.get(existing, float("-inf")):
                    to_remove.append(existing)
                else:
                    return
        for existing in to_remove:
            gates.remove(existing)
            gate_scores.pop(existing, None)
        if path not in gate_scores:
            gates.append(path)
            gate_scores[path] = score

    for h in sorted(branch_hits, key=lambda x: x.score, reverse=True):
        add_gate(h.path, h.score)
        if len(gates) >= gate_branches_topb:
            break

    if len(gates) < gate_branches_topb:
        for h in sorted(leaf_hits, key=lambda x: x.score, reverse=True):
            for d in range(1, len(h.path)):
                anc = h.path[:d]
                add_gate(anc, h.score)
                if len(gates) >= gate_branches_topb:
                    break
            if len(gates) >= gate_branches_topb:
                break

    gates = sorted(gates, key=lambda p: gate_scores.get(p, 0.0), reverse=True)[:gate_branches_topb]
    gate_scores = {p: gate_scores[p] for p in gates}
    return gates, flat_leaf_ranked, gate_scores


def flat_retrieve_hits(
    *,
    retriever: DiverEmbeddingModel,
    query: str,
    node_embs: np.ndarray,
    node_registry: Sequence[object],
    topk: int,
) -> List[FlatHit]:
    """
    Assumes `node_embs[i]` aligns with `node_registry[i]` (same `registry_idx` ordering).
    """
    q_emb = retriever.encode_query(query)
    res = cosine_topk(q_emb, node_embs, topk)

    hits: List[FlatHit] = []
    for ridx, score in zip(res.indices.tolist(), res.scores.tolist()):
        node = node_registry[int(ridx)]
        path = tuple(node.path)
        is_leaf = (not node.child) or (len(node.child) == 0)
        hits.append(FlatHit(registry_idx=int(ridx), path=path, score=float(score), is_leaf=is_leaf))
    return hits


def rrf_fuse_ranked_paths(
    ranked_lists: Sequence[Sequence[Tuple[Path, float]]],
    *,
    k: int = 60,
) -> List[Tuple[Path, float]]:
    """
    Reciprocal Rank Fusion over multiple ranked lists of (path, score).
    """
    fused: Dict[Path, float] = {}
    for lst in ranked_lists:
        for rank, (path, _score) in enumerate(lst, start=1):
            fused[path] = fused.get(path, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def is_prefix(a: Path, b: Path) -> bool:
    return len(a) <= len(b) and b[: len(a)] == a


def ancestor_hit(retrieved_paths: Sequence[Path], gold_paths: Sequence[Path]) -> bool:
    for rp in retrieved_paths:
        for gp in gold_paths:
            if is_prefix(rp, gp) or is_prefix(gp, rp):
                return True
    return False


def gate_hit(gates: Sequence[Path], gold_paths: Sequence[Path]) -> bool:
    for g in gates:
        for gp in gold_paths:
            if is_prefix(g, gp):
                return True
    return False
