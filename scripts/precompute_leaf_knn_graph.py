import argparse
import os
import pickle as pkl
import sys
from typing import Tuple

import numpy as np
from tqdm.auto import tqdm


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from tree_objects import SemanticNode  # noqa: E402
from utils import compute_node_registry, normalize_embeddings  # noqa: E402


def _topk_rows_desc(scores: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    if scores.size == 0 or topk <= 0:
        empty_idx = np.zeros((scores.shape[0], 0), dtype=np.int64)
        empty_vals = np.zeros((scores.shape[0], 0), dtype=np.float32)
        return empty_idx, empty_vals
    k = min(topk, scores.shape[1])
    if k == scores.shape[1]:
        idx = np.argsort(-scores, axis=1)
        vals = np.take_along_axis(scores, idx, axis=1)
        return idx.astype(np.int64, copy=False), vals.astype(np.float32, copy=False)
    idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-vals, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    vals = np.take_along_axis(vals, order, axis=1)
    return idx.astype(np.int64, copy=False), vals.astype(np.float32, copy=False)


def _resolve_emb_registry_indices(node_registry, num_emb_rows: int) -> np.ndarray:
    num_registry = len(node_registry)
    if num_emb_rows == num_registry:
        return np.arange(num_registry, dtype=np.int64)

    non_empty_desc_indices = np.array(
        [idx for idx, node in enumerate(node_registry) if str(getattr(node, "desc", "") or "").strip()],
        dtype=np.int64,
    )
    if num_emb_rows == int(non_empty_desc_indices.shape[0]):
        # Intent: align rows when embeddings were built with encode_docs(), which drops empty descriptions.
        return non_empty_desc_indices

    if num_emb_rows == (num_registry - 1):
        first_path = tuple(getattr(node_registry[0], "path", ()))
        if len(first_path) == 0:
            return np.arange(1, num_registry, dtype=np.int64)

    raise ValueError(
        f"Cannot align embeddings to registry: emb_rows={num_emb_rows}, "
        f"registry_rows={num_registry}, non_empty_desc_rows={int(non_empty_desc_indices.shape[0])}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute leaf-to-leaf top-K cosine neighbors.")
    parser.add_argument("--dataset", type=str, default="BRIGHT")
    parser.add_argument("--subset", type=str, required=True)
    parser.add_argument("--tree_version", type=str, required=True)
    parser.add_argument("--node_emb_path", type=str, default=None)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    if args.topk <= 0:
        raise ValueError("--topk must be > 0")
    if args.block_size <= 0:
        raise ValueError("--block_size must be > 0")

    tree_path = os.path.join(REPO_ROOT, "trees", args.dataset, args.subset, f"tree-{args.tree_version}.pkl")
    if not os.path.exists(tree_path):
        raise FileNotFoundError(f"Tree file not found: {tree_path}")
    node_emb_path = args.node_emb_path or os.path.join(REPO_ROOT, "trees", args.dataset, args.subset, "node_embs.diver.npy")
    if not os.path.exists(node_emb_path):
        raise FileNotFoundError(f"Node embedding file not found: {node_emb_path}")
    output_path = args.output_path or os.path.join(
        REPO_ROOT,
        "trees",
        args.dataset,
        args.subset,
        f"leaf_knn_top{args.topk}.npz",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    tree_dict = pkl.load(open(tree_path, "rb"))
    semantic_root_node = SemanticNode().load_dict(tree_dict) if isinstance(tree_dict, dict) else tree_dict
    node_registry = compute_node_registry(semantic_root_node)

    node_embs = np.load(node_emb_path, allow_pickle=False)
    emb_registry_indices = _resolve_emb_registry_indices(node_registry, int(node_embs.shape[0]))
    registry_to_emb_row = {
        int(registry_idx): int(row_idx)
        for row_idx, registry_idx in enumerate(emb_registry_indices.tolist())
    }
    node_embs = normalize_embeddings(node_embs)

    leaf_registry_indices = np.array(
        [idx for idx, node in enumerate(node_registry) if node.is_leaf and idx in registry_to_emb_row],
        dtype=np.int64,
    )
    if leaf_registry_indices.size == 0:
        raise ValueError("No leaf nodes found in node registry.")
    leaf_emb_rows = np.array(
        [registry_to_emb_row[int(registry_idx)] for registry_idx in leaf_registry_indices.tolist()],
        dtype=np.int64,
    )
    leaf_embs = node_embs[leaf_emb_rows]
    print(
        f"[INFO] aligned embedding rows={node_embs.shape[0]} to registry rows={len(node_registry)} "
        f"(usable leaf rows={leaf_embs.shape[0]})"
    )
    num_leaf = int(leaf_registry_indices.shape[0])
    k = min(args.topk, max(0, num_leaf - 1))

    neighbor_registry_indices = np.full((num_leaf, args.topk), -1, dtype=np.int32)
    neighbor_scores = np.full((num_leaf, args.topk), -1.0, dtype=np.float16)

    for start in tqdm(range(0, num_leaf, args.block_size), desc="Computing leaf kNN"):
        end = min(start + args.block_size, num_leaf)
        block_embs = leaf_embs[start:end]
        sim = block_embs @ leaf_embs.T
        for i in range(end - start):
            sim[i, start + i] = -np.inf
        if k <= 0:
            continue
        idx_local, val_local = _topk_rows_desc(sim, k)
        idx_global = leaf_registry_indices[idx_local]
        # Intent: store only compact top-K neighbors so v6 can skip full leaf×leaf similarity at runtime.
        neighbor_registry_indices[start:end, :k] = idx_global.astype(np.int32, copy=False)
        neighbor_scores[start:end, :k] = val_local.astype(np.float16, copy=False)

    np.savez_compressed(
        output_path,
        leaf_registry_indices=leaf_registry_indices.astype(np.int32, copy=False),
        neighbor_registry_indices=neighbor_registry_indices,
        neighbor_scores=neighbor_scores,
        topk=np.array([args.topk], dtype=np.int32),
    )
    print(f"[OK] saved leaf-kNN graph to {output_path}")
    print(f"[INFO] num_leaf={num_leaf}, requested_topk={args.topk}, effective_topk={k}")


if __name__ == "__main__":
    main()
