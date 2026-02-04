import argparse
import json
import os
import pickle as pkl
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from retrievers.diver import DiverEmbeddingModel  # noqa: E402


Path = Tuple[int, ...]


@dataclass
class FlatHit:
    registry_idx: int
    path: Path
    score: float
    is_leaf: bool


def _normalize_rows(embs: np.ndarray) -> np.ndarray:
    if embs.size == 0:
        return embs.astype(np.float32, copy=False)
    embs = embs.astype(np.float32, copy=False)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return embs / norms


def _normalize_vector(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32, copy=False)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


def _load_node_catalog(node_catalog_path: str) -> Tuple[List[Path], List[bool]]:
    paths: List[Path] = []
    is_leaf_flags: List[bool] = []
    with open(node_catalog_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            paths.append(tuple(int(x) for x in obj["path"]))
            is_leaf_flags.append(bool(obj["is_leaf"]))
    return paths, is_leaf_flags


def _find_iter_record(sample: Dict, iter_idx: int) -> Dict:
    for rec in sample.get("iter_records", []):
        if int(rec.get("iter", -1)) == iter_idx:
            return rec
    return {}


def _topk_indices_desc(scores: np.ndarray, topk: int) -> np.ndarray:
    n = int(scores.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    k = int(max(1, min(topk, n)))
    if k == n:
        return np.argsort(-scores).astype(np.int64, copy=False)
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx.astype(np.int64, copy=False)


def _build_anchor_hits(
    scores_all: np.ndarray,
    node_paths: Sequence[Path],
    is_leaf_flags: Sequence[bool],
    retrieval_topk: int,
) -> List[FlatHit]:
    idx = _topk_indices_desc(scores_all, retrieval_topk)
    hits: List[FlatHit] = []
    for ridx in idx.tolist():
        hits.append(
            FlatHit(
                registry_idx=int(ridx),
                path=tuple(node_paths[int(ridx)]),
                score=float(scores_all[int(ridx)]),
                is_leaf=bool(is_leaf_flags[int(ridx)]),
            )
        )
    return hits


def _build_leaf_indices_by_prefix(node_paths: Sequence[Path], is_leaf_flags: Sequence[bool]) -> Dict[Path, List[int]]:
    leaf_indices_by_prefix: Dict[Path, List[int]] = {}
    for idx, (path, is_leaf) in enumerate(zip(node_paths, is_leaf_flags)):
        if not is_leaf:
            continue
        for d in range(1, len(path)):
            prefix = tuple(path[:d])
            leaf_indices_by_prefix.setdefault(prefix, []).append(int(idx))
    return leaf_indices_by_prefix


def _build_graph_on_v2_paths(
    anchor_hits: Sequence[FlatHit],
    scores_all: np.ndarray,
    node_paths: Sequence[Path],
    leaf_indices_by_prefix: Dict[Path, List[int]],
    topk: int,
) -> List[Path]:
    ordered: List[Path] = []
    seen: set[int] = set()
    for hit in anchor_hits:
        if hit.is_leaf:
            if hit.registry_idx in seen:
                continue
            ordered.append(hit.path)
            seen.add(hit.registry_idx)
            if len(ordered) >= topk:
                return ordered
            continue
        candidate_indices = leaf_indices_by_prefix.get(hit.path, [])
        if not candidate_indices:
            continue
        candidate_scores = scores_all[candidate_indices]
        if candidate_scores.size == 0:
            continue
        order = np.argsort(-candidate_scores)
        for ridx in order.tolist():
            leaf_idx = int(candidate_indices[int(ridx)])
            if leaf_idx in seen:
                continue
            ordered.append(tuple(node_paths[leaf_idx]))
            seen.add(leaf_idx)
            if len(ordered) >= topk:
                return ordered
            break
    return ordered


def _build_graph_off_leaf_paths(anchor_hits: Sequence[FlatHit], topk: int) -> List[Path]:
    out: List[Path] = []
    for hit in anchor_hits:
        if hit.is_leaf:
            out.append(hit.path)
            if len(out) >= topk:
                return out
    return out


def _hit_at_k(pred_paths: Sequence[Path], gold_paths: Sequence[Path], k: int) -> float:
    topk = set(pred_paths[:k])
    return float(any(g in topk for g in gold_paths))


def _ndcg_at_k(pred_paths: Sequence[Path], gold_paths: Sequence[Path], k: int) -> float:
    if not gold_paths:
        return 0.0
    pred_k = list(pred_paths[:k])
    dcg = 0.0
    for i, p in enumerate(pred_k):
        if p in gold_paths:
            dcg += 1.0 / np.log2(i + 2.0)
    ideal_hits = min(k, len(gold_paths))
    idcg = sum(1.0 / np.log2(i + 2.0) for i in range(ideal_hits))
    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def _branch_contains_any_gold(branch_path: Path, gold_paths: Sequence[Path]) -> bool:
    branch_len = len(branch_path)
    for gold_path in gold_paths:
        if len(gold_path) >= branch_len and tuple(gold_path[:branch_len]) == branch_path:
            return True
    return False


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.array(values, dtype=np.float32)))


def _infer_retrieval_topk(sample_dicts: Sequence[Dict], iter_indices: Sequence[int], fallback: int) -> int:
    for sample in sample_dicts:
        for iter_idx in iter_indices:
            rec = _find_iter_record(sample, iter_idx)
            if not rec:
                continue
            leaf_n = len(rec.get("anchor_leaf_paths", []))
            branch_n = len(rec.get("anchor_branch_paths", []))
            total = int(leaf_n + branch_n)
            if total > 0:
                return total
    return int(fallback)


def _evaluate_iter(
    *,
    sample_dicts: Sequence[Dict],
    iter_idx: int,
    eval_k: int,
    retrieval_topk: int,
    retriever: DiverEmbeddingModel,
    node_embs: np.ndarray,
    node_paths: Sequence[Path],
    is_leaf_flags: Sequence[bool],
    leaf_indices_by_prefix: Dict[Path, List[int]],
) -> Dict[str, float]:
    flat_leaf_hit_list: List[float] = []
    graph_off_hit_list: List[float] = []
    graph_on_hit_list: List[float] = []
    graph_off_ndcg_list: List[float] = []
    graph_on_ndcg_list: List[float] = []
    branch_hit_any_list: List[float] = []
    branch_hit_ratio_list: List[float] = []
    opportunity_list: List[float] = []
    conversion_list: List[float] = []
    miss_flat_leaf_list: List[float] = []

    n_samples = 0
    for sample in sample_dicts:
        rec = _find_iter_record(sample, iter_idx)
        if not rec:
            continue
        gold_paths = [tuple(int(x) for x in p) for p in sample.get("gold_paths", [])]
        if not gold_paths:
            continue
        query_t = str(rec.get("query_t") or "").strip()
        if not query_t:
            query_t = str(sample.get("original_query") or "").strip()
        if not query_t:
            continue

        q_emb = retriever.encode_query(query_t)
        q_emb = _normalize_vector(np.asarray(q_emb))
        # Intent: match run_round3_1 scoring behavior (cosine via normalized node_embs and query embedding).
        scores_all = (node_embs @ q_emb).astype(np.float32, copy=False)
        anchor_hits = _build_anchor_hits(
            scores_all=scores_all,
            node_paths=node_paths,
            is_leaf_flags=is_leaf_flags,
            retrieval_topk=retrieval_topk,
        )
        if not anchor_hits:
            continue

        n_samples += 1
        flat_topk = anchor_hits[:eval_k]
        flat_topk_leaf_paths = [h.path for h in flat_topk if h.is_leaf]
        flat_topk_branch_paths = [h.path for h in flat_topk if not h.is_leaf]

        graph_off_paths = _build_graph_off_leaf_paths(anchor_hits, topk=eval_k)
        graph_on_paths = _build_graph_on_v2_paths(
            anchor_hits=anchor_hits,
            scores_all=scores_all,
            node_paths=node_paths,
            leaf_indices_by_prefix=leaf_indices_by_prefix,
            topk=eval_k,
        )

        flat_leaf_hit = _hit_at_k(flat_topk_leaf_paths, gold_paths, eval_k)
        graph_off_hit = _hit_at_k(graph_off_paths, gold_paths, eval_k)
        graph_on_hit = _hit_at_k(graph_on_paths, gold_paths, eval_k)
        graph_off_ndcg = _ndcg_at_k(graph_off_paths, gold_paths, eval_k) * 100.0
        graph_on_ndcg = _ndcg_at_k(graph_on_paths, gold_paths, eval_k) * 100.0

        if flat_topk_branch_paths:
            branch_hit_ratio = float(
                sum(1 for branch_path in flat_topk_branch_paths if _branch_contains_any_gold(branch_path, gold_paths))
            ) / float(len(flat_topk_branch_paths))
        else:
            branch_hit_ratio = 0.0
        branch_hit_any = float(branch_hit_ratio > 0.0)

        miss_flat_leaf = float(flat_leaf_hit == 0.0)
        opportunity = float((flat_leaf_hit == 0.0) and (branch_hit_any == 1.0))
        conversion = float((opportunity == 1.0) and (graph_on_hit == 1.0))

        flat_leaf_hit_list.append(flat_leaf_hit)
        graph_off_hit_list.append(graph_off_hit)
        graph_on_hit_list.append(graph_on_hit)
        graph_off_ndcg_list.append(graph_off_ndcg)
        graph_on_ndcg_list.append(graph_on_ndcg)
        branch_hit_any_list.append(branch_hit_any)
        branch_hit_ratio_list.append(branch_hit_ratio)
        opportunity_list.append(opportunity)
        conversion_list.append(conversion)
        miss_flat_leaf_list.append(miss_flat_leaf)

    miss_count = int(sum(miss_flat_leaf_list))
    opportunity_count = int(sum(opportunity_list))
    conversion_count = int(sum(conversion_list))

    return {
        "iter": int(iter_idx),
        "n_samples": int(n_samples),
        f"FlatMixedLeafHit@{eval_k}": _mean(flat_leaf_hit_list) * 100.0,
        f"GraphOffLeafHit@{eval_k}": _mean(graph_off_hit_list) * 100.0,
        f"GraphOnV2LeafHit@{eval_k}": _mean(graph_on_hit_list) * 100.0,
        f"Delta_Hit@{eval_k}_graph_on_minus_off": (_mean(graph_on_hit_list) - _mean(graph_off_hit_list)) * 100.0,
        f"GraphOff_nDCG@{eval_k}": _mean(graph_off_ndcg_list),
        f"GraphOnV2_nDCG@{eval_k}": _mean(graph_on_ndcg_list),
        f"Delta_nDCG@{eval_k}_graph_on_minus_off": _mean(graph_on_ndcg_list) - _mean(graph_off_ndcg_list),
        f"FlatTopK_BranchHitAny@{eval_k}": _mean(branch_hit_any_list) * 100.0,
        f"FlatTopK_BranchHitRatio@{eval_k}": _mean(branch_hit_ratio_list) * 100.0,
        f"FlatTopK_MissLeaf@{eval_k}": _mean(miss_flat_leaf_list) * 100.0,
        f"Opportunity@{eval_k}_overall": _mean(opportunity_list) * 100.0,
        f"Opportunity@{eval_k}_givenFlatLeafMiss": (
            100.0 * float(opportunity_count) / float(miss_count)
        ) if miss_count > 0 else 0.0,
        f"ConversionByGraphOn@{eval_k}_overall": _mean(conversion_list) * 100.0,
        f"ConversionByGraphOn@{eval_k}_givenOpportunity": (
            100.0 * float(conversion_count) / float(opportunity_count)
        ) if opportunity_count > 0 else 0.0,
        "FlatLeafMiss_count": miss_count,
        "Opportunity_count": opportunity_count,
        "Conversion_count": conversion_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_pkl", type=str, required=True, help="Path to v2 run all_eval_sample_dicts.pkl")
    parser.add_argument("--dataset", type=str, default="BRIGHT")
    parser.add_argument("--subset", type=str, required=True)
    parser.add_argument("--node_catalog_path", type=str, default=None)
    parser.add_argument("--node_emb_path", type=str, default=None)
    parser.add_argument("--retriever_model_path", type=str, required=True)
    parser.add_argument("--eval_k", type=int, default=10)
    parser.add_argument("--retrieval_topk", type=int, default=-1, help="If <=0, infer from saved anchor leaf+branch counts")
    parser.add_argument("--iter_indices", type=str, default="0,1")
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    eval_pkl = os.path.abspath(args.eval_pkl)
    if not os.path.exists(eval_pkl):
        raise FileNotFoundError(f"eval_pkl not found: {eval_pkl}")

    node_catalog_path = args.node_catalog_path
    if not node_catalog_path:
        node_catalog_path = os.path.join(REPO_ROOT, "trees", args.dataset, args.subset, "node_catalog.jsonl")
    node_catalog_path = os.path.abspath(node_catalog_path)
    if not os.path.exists(node_catalog_path):
        raise FileNotFoundError(f"node_catalog_path not found: {node_catalog_path}")

    node_emb_path = args.node_emb_path
    if not node_emb_path:
        node_emb_path = os.path.join(REPO_ROOT, "trees", args.dataset, args.subset, "node_embs.diver.npy")
    node_emb_path = os.path.abspath(node_emb_path)
    if not os.path.exists(node_emb_path):
        raise FileNotFoundError(f"node_emb_path not found: {node_emb_path}")

    sample_dicts = pkl.load(open(eval_pkl, "rb"))
    iter_indices = [int(x.strip()) for x in args.iter_indices.split(",") if x.strip()]

    node_paths, is_leaf_flags = _load_node_catalog(node_catalog_path)
    node_embs = np.load(node_emb_path, allow_pickle=False)
    if node_embs.shape[0] != len(node_paths):
        raise ValueError(
            f"node_embs rows ({node_embs.shape[0]}) must match node_catalog size ({len(node_paths)})"
        )
    node_embs = _normalize_rows(node_embs)
    leaf_indices_by_prefix = _build_leaf_indices_by_prefix(node_paths, is_leaf_flags)

    retrieval_topk = int(args.retrieval_topk)
    if retrieval_topk <= 0:
        retrieval_topk = _infer_retrieval_topk(sample_dicts, iter_indices, fallback=1000)

    retriever = DiverEmbeddingModel(args.retriever_model_path, local_files_only=True)

    iter_metrics: Dict[str, Dict[str, float]] = {}
    for iter_idx in iter_indices:
        iter_metrics[str(iter_idx)] = _evaluate_iter(
            sample_dicts=sample_dicts,
            iter_idx=iter_idx,
            eval_k=int(args.eval_k),
            retrieval_topk=retrieval_topk,
            retriever=retriever,
            node_embs=node_embs,
            node_paths=node_paths,
            is_leaf_flags=is_leaf_flags,
            leaf_indices_by_prefix=leaf_indices_by_prefix,
        )

    report = {
        "meta": {
            "dataset": args.dataset,
            "subset": args.subset,
            "eval_k": int(args.eval_k),
            "retrieval_topk": int(retrieval_topk),
            "iter_indices": iter_indices,
            "note": (
                "Within the same v2 run, compare graph-off (leaf-only from mixed flat ranking) "
                "vs graph-on (v2 branch->best leaf replacement)."
            ),
        },
        "eval_pkl_path": eval_pkl,
        "node_catalog_path": node_catalog_path,
        "node_emb_path": node_emb_path,
        "iter_metrics": iter_metrics,
    }

    output_path = os.path.abspath(args.output_json)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved report: {output_path}")


if __name__ == "__main__":
    main()
