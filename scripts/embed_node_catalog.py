import argparse
import json
import os
import sys
from typing import List
import torch
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from retrievers.diver import DiverEmbeddingModel


def read_jsonl(path: str) -> List[dict]:
    items: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed node catalog (desc field) using Diver-Retriever style embeddings")
    parser.add_argument("--node_catalog_jsonl", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--out_npy", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=16384)
    args = parser.parse_args()

    if os.path.exists(args.out_npy):
        print(f"Output file {args.out_npy} already exists. Skipping embedding.")
        return
    catalog = read_jsonl(args.node_catalog_jsonl)
    texts = [str(rec.get("desc", "")) for rec in catalog]
    model = DiverEmbeddingModel(args.model_path, local_files_only=True)
    # with torch.infererence_mode():
    print("Computing embeddings...")
    embs = model.encode_docs(texts, max_length=args.max_length, batch_size=args.batch_size)

    os.makedirs(os.path.dirname(args.out_npy) or ".", exist_ok=True)
    np.save(args.out_npy, embs.astype(np.float32, copy=False))


if __name__ == "__main__":
    main()
