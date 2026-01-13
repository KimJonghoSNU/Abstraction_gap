import argparse
import os
import pickle as pkl
import sys
from collections import defaultdict

import numpy as np
import logging

#   python scripts/ancestor_hit_by_depth.py \
#     --tree_pkl trees/BRIGHT/biology/tree-bottom-up.pkl \
#     --eval_samples_pkl results/BRIGHT/biology/all_eval_sample_dicts-<hp>.pkl \
#     --topk 10 \
#     --out_csv results/BRIGHT/biology/ancestor_hit_by_depth.csv

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from tree_objects import InferSample, SemanticNode  # noqa: E402
from utils import compute_node_registry  # noqa: E402


def is_prefix(a, b):
    return len(a) <= len(b) and tuple(b[: len(a)]) == tuple(a)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute AncestorHit@depth for LATTICE top-10 leaves.")
    parser.add_argument("--tree_pkl", type=str, required=True)
    parser.add_argument("--eval_samples_pkl", type=str, required=True, help="Path to all_eval_sample_dicts-*.pkl")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--out_csv", type=str, default=None)
    args = parser.parse_args()

    tree_obj = pkl.load(open(args.tree_pkl, "rb"))
    root = SemanticNode().load_dict(tree_obj) if isinstance(tree_obj, dict) else tree_obj
    node_registry = compute_node_registry(root)
    max_depth = max(len(n.path) for n in node_registry)

    sample_dicts = pkl.load(open(args.eval_samples_pkl, "rb"))

    # Minimal hp namespace to allow InferSample to work
    class HP:
        MAX_BEAM_SIZE = 2
        SEARCH_WITH_PATH_RELEVANCE = True
        NUM_LEAF_CALIB = 10
        PL_TAU = 5.0
        RELEVANCE_CHAIN_FACTOR = 0.5

    hits = defaultdict(int)
    eligible = defaultdict(int)

    logger = logging.getLogger("ancestor_hit_by_depth")
    for sample_dict in sample_dicts:
        excluded_ids_set = sample_dict.get("excluded_ids_set")
        if excluded_ids_set is not None and not isinstance(excluded_ids_set, set):
            excluded_ids_set = set(excluded_ids_set)
        sample = InferSample(
            root,
            node_registry,
            hp=HP,
            logger=logger,
            query="",
            gold_paths=[],
            excluded_ids_set=excluded_ids_set,
        )
        sample.load_dict(sample_dict)
        sample.post_load_processing()

        top_preds = sample.get_top_predictions(k=args.topk, rel_fn=sample.get_rel_fn(leaf=True))
        pred_paths = [tuple(x.path) for x, _ in top_preds]

        gold_paths = [tuple(p) for p in sample.gold_paths]
        if len(gold_paths) == 0:
            continue

        for d in range(1, max_depth + 1):
            # only count queries where a gold path is deep enough
            if not any(len(g) >= d for g in gold_paths):
                continue
            eligible[d] += 1
            hit = False
            for pred in pred_paths:
                for gold in gold_paths:
                    if len(pred) >= d and len(gold) >= d and pred[:d] == gold[:d]:
                        hit = True
                        break
                if hit:
                    break
            if hit:
                hits[d] += 1

    rows = []
    for d in range(1, max_depth + 1):
        if eligible[d] == 0:
            continue
        rate = hits[d] / eligible[d]
        rows.append((d, eligible[d], hits[d], rate))

    header = "depth,eligible_queries,hit_queries,hit_rate"
    lines = [header] + [f"{d},{e},{h},{r:.4f}" for d, e, h, r in rows]
    output = "\n".join(lines)
    print(output)

    if args.out_csv:
        with open(args.out_csv, "w", encoding="utf-8") as f:
            f.write(output + "\n")


if __name__ == "__main__":
    main()
