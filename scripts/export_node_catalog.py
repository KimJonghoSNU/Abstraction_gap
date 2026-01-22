import argparse
import json
import os
import pickle as pkl
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from tree_objects import SemanticNode
from utils import compute_node_registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Export flattened node catalog from a semantic tree")
    parser.add_argument("--tree_pkl", type=str, required=True)
    parser.add_argument("--out_jsonl", type=str, required=True)
    args = parser.parse_args()

    try:
        tree_obj = pkl.load(open(args.tree_pkl, "rb"))
    except Exception as e:
        args.tree_pkl = args.tree_pkl.replace("bottom-up", "top-down")
        tree_obj = pkl.load(open(args.tree_pkl, "rb"))
    root = SemanticNode().load_dict(tree_obj) if isinstance(tree_obj, dict) else tree_obj
    node_registry = compute_node_registry(root)
    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    depth1_path = os.path.splitext(args.out_jsonl)[0] + "_depth1.jsonl"
    os.makedirs(os.path.dirname(depth1_path) or ".", exist_ok=True)
    depth1_count = 0
    with open(args.out_jsonl, "w", encoding="utf-8") as f, open(depth1_path, "w", encoding="utf-8") as f_depth1:
        for node in node_registry:
            rec = {
                "registry_idx": int(node.registry_idx),
                "path": list(node.path),
                "depth": int(len(node.path)),
                "is_leaf": (not node.child) or (len(node.child) == 0),
                "num_children": int(len(node.child)) if node.child else 0,
                "num_leaves": int(getattr(node, "num_leaves", 0)),
                "id": str(node.id),
                "desc": node.desc,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if rec["depth"] == 1:
                f_depth1.write(json.dumps(rec, ensure_ascii=False) + "\n")
                depth1_count += 1
        print(f"Exported {len(node_registry)} nodes to {args.out_jsonl}")
        print(f"Exported {depth1_count} depth=1 nodes to {depth1_path}")


if __name__ == "__main__":
    main()
