#!/usr/bin/env python3
"""
Train branch traversal reward model with TRL RewardTrainer.

Input:
- branch supervision jsonl files produced by
  scripts/train/build_dataset/build_traversal_rl_dataset.py

Training objective:
- Build pairwise preferences from candidate rewards in each branch step.
- Train reward model with TRL's pairwise reward loss on (chosen, rejected).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from datasets import Dataset
from trl import RewardConfig, RewardTrainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_subsets(raw: str) -> List[str]:
    text = raw.strip()
    if text.lower() == "all":
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def discover_branch_files(data_root: Path, subsets: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    if subsets:
        for subset in subsets:
            p = data_root / subset / "branch_steps.jsonl"
            if p.exists():
                files.append(p)
    else:
        files.extend(sorted(data_root.glob("*/branch_steps.jsonl")))
    return files


def format_frontier(frontier: Sequence[dict], max_items: int) -> str:
    rows: List[str] = []
    for node in list(frontier)[:max_items]:
        desc = str(node.get("desc", "")).strip()
        if desc:
            rows.append(f"- {desc}")
    return "\n".join(rows) if rows else "- (none)"


def format_candidate(candidate: dict) -> str:
    desc = str(candidate.get("desc", "")).strip()
    path = candidate.get("path", [])
    leaf_count = int(candidate.get("subtree_leaf_count", 0))
    return (
        f"[Candidate]\n{desc}\n"
        f"[Path]\n{path}\n"
        f"[Estimated Leaf Count]\n{leaf_count}\n"
    )


def format_candidate_text(
    *,
    subset: str,
    query: str,
    depth: int,
    frontier: Sequence[dict],
    candidate: dict,
    max_frontier_items: int,
) -> str:
    frontier_text = format_frontier(frontier=frontier, max_items=max_frontier_items)
    return (
        f"[Subset]\n{subset}\n"
        f"[Depth]\n{depth}\n"
        f"[Query]\n{query}\n"
        f"[Frontier]\n{frontier_text}\n"
        f"{format_candidate(candidate)}"
    )


def build_preference_pairs_from_record(
    record: dict,
    *,
    max_pairs_per_step: int,
    min_reward_gap: float,
    max_frontier_items: int,
) -> List[dict]:
    subset = str(record.get("subset", ""))
    query = str(record.get("query", ""))
    depth = int(record.get("depth", -1))
    query_idx = int(record.get("query_idx", -1))
    frontier = record.get("frontier", []) or []
    candidates = record.get("candidates", []) or []
    if len(candidates) < 2:
        return []

    pairs: List[dict] = []
    for i, ci in enumerate(candidates):
        ri = float(ci.get("reward", 0.0))
        for j, cj in enumerate(candidates):
            if i == j:
                continue
            rj = float(cj.get("reward", 0.0))
            gap = ri - rj
            if gap <= 0.0:
                continue
            if gap < min_reward_gap:
                continue

            chosen = format_candidate_text(
                subset=subset,
                query=query,
                depth=depth,
                frontier=frontier,
                candidate=ci,
                max_frontier_items=max_frontier_items,
            )
            rejected = format_candidate_text(
                subset=subset,
                query=query,
                depth=depth,
                frontier=frontier,
                candidate=cj,
                max_frontier_items=max_frontier_items,
            )
            pairs.append(
                {
                    "subset": subset,
                    "query_idx": query_idx,
                    "depth": depth,
                    "chosen": chosen,
                    "rejected": rejected,
                    "reward_gap": gap,
                }
            )

    if max_pairs_per_step > 0 and len(pairs) > max_pairs_per_step:
        random.shuffle(pairs)
        pairs = pairs[:max_pairs_per_step]
    return pairs


def load_pairs(
    branch_files: Sequence[Path],
    *,
    max_pairs_per_step: int,
    min_reward_gap: float,
    max_frontier_items: int,
    max_steps_per_subset: int,
) -> List[dict]:
    pairs: List[dict] = []
    per_subset_steps: Dict[str, int] = {}

    for fp in branch_files:
        subset = fp.parent.name
        per_subset_steps.setdefault(subset, 0)
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                if max_steps_per_subset > 0 and per_subset_steps[subset] >= max_steps_per_subset:
                    break
                record = json.loads(line)
                step_pairs = build_preference_pairs_from_record(
                    record,
                    max_pairs_per_step=max_pairs_per_step,
                    min_reward_gap=min_reward_gap,
                    max_frontier_items=max_frontier_items,
                )
                if step_pairs:
                    pairs.extend(step_pairs)
                per_subset_steps[subset] += 1
    return pairs


def split_train_eval_by_query(
    pairs: Sequence[dict],
    *,
    eval_ratio: float,
    seed: int,
) -> Tuple[List[dict], List[dict]]:
    grouped: Dict[str, List[dict]] = {}
    for row in pairs:
        key = f"{row['subset']}::{row['query_idx']}"
        grouped.setdefault(key, []).append(row)

    keys = list(grouped.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)

    n_eval = int(round(len(keys) * eval_ratio))
    n_eval = min(max(n_eval, 1), max(len(keys) - 1, 1))
    eval_keys = set(keys[:n_eval])

    train_rows: List[dict] = []
    eval_rows: List[dict] = []
    for k, rows in grouped.items():
        if k in eval_keys:
            eval_rows.extend(rows)
        else:
            train_rows.extend(rows)
    return train_rows, eval_rows


@torch.no_grad()
def evaluate_pairwise_accuracy(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    rows: Sequence[dict],
    *,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> dict:
    if len(rows) == 0:
        return {"pair_acc": 0.0, "mean_margin": 0.0, "num_pairs": 0}

    model.eval()
    total = 0
    correct = 0
    margin_sum = 0.0

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        chosen_texts = [x["chosen"] for x in batch_rows]
        rejected_texts = [x["rejected"] for x in batch_rows]

        chosen_tok = tokenizer(
            chosen_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        rejected_tok = tokenizer(
            rejected_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        chosen_tok = {k: v.to(device) for k, v in chosen_tok.items()}
        rejected_tok = {k: v.to(device) for k, v in rejected_tok.items()}

        chosen_scores = model(**chosen_tok).logits.squeeze(-1)
        rejected_scores = model(**rejected_tok).logits.squeeze(-1)
        margins = (chosen_scores - rejected_scores).detach().cpu().numpy()

        correct += int((margins > 0).sum())
        total += int(margins.shape[0])
        margin_sum += float(margins.sum())

    return {
        "pair_acc": float(correct / total) if total > 0 else 0.0,
        "mean_margin": float(margin_sum / total) if total > 0 else 0.0,
        "num_pairs": int(total),
    }


def rows_to_dataset(rows: Sequence[dict], margin: float) -> Dataset:
    hf_rows: List[dict] = []
    for row in rows:
        item = {
            "chosen": row["chosen"],
            "rejected": row["rejected"],
        }
        if margin > 0.0:
            # Intent: keep margin explicit so TRL RewardTrainer applies pairwise margin loss variant.
            item["margin"] = float(margin)
        hf_rows.append(item)
    return Dataset.from_list(hf_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train branch traversal reward model with TRL RewardTrainer.")
    parser.add_argument("--data_root", type=str, required=True, help="Path containing subset/*/branch_steps.jsonl.")
    parser.add_argument("--subsets", type=str, default="all", help='Comma-separated subsets or "all".')
    parser.add_argument("--model_name_or_path", type=str, required=True, help="HF model path/name for reward model.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints and logs.")

    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_frontier_items", type=int, default=2)
    parser.add_argument("--max_pairs_per_step", type=int, default=16)
    parser.add_argument("--max_steps_per_subset", type=int, default=-1)
    parser.add_argument("--min_reward_gap", type=float, default=0.05)
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--fp16", action="store_true", default=False)
    args = parser.parse_args()

    seed_everything(args.seed)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subsets = parse_subsets(args.subsets)
    branch_files = discover_branch_files(data_root=data_root, subsets=subsets)
    if not branch_files:
        raise FileNotFoundError(f"No branch_steps.jsonl found under {data_root} for subsets={subsets or 'all'}")

    print(f"[info] found branch files: {len(branch_files)}")
    for fp in branch_files:
        print(f"  - {fp}")

    pairs = load_pairs(
        branch_files=branch_files,
        max_pairs_per_step=args.max_pairs_per_step,
        min_reward_gap=args.min_reward_gap,
        max_frontier_items=args.max_frontier_items,
        max_steps_per_subset=args.max_steps_per_subset,
    )
    if not pairs:
        raise RuntimeError("No preference pairs generated. Check filters or source data.")
    print(f"[info] total pairs: {len(pairs)}")

    train_rows, eval_rows = split_train_eval_by_query(
        pairs=pairs,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )
    print(f"[info] train pairs: {len(train_rows)}")
    print(f"[info] eval pairs: {len(eval_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    eos_token = tokenizer.eos_token or tokenizer.sep_token or tokenizer.pad_token
    if eos_token is None:
        raise ValueError(
            f"Tokenizer for {args.model_name_or_path} has no eos/sep/pad token; cannot build TRL reward dataset safely."
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=1,
        torch_dtype=(torch.bfloat16 if args.bf16 else None),
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    train_ds = rows_to_dataset(train_rows, margin=args.margin)
    eval_ds = rows_to_dataset(eval_rows, margin=args.margin)

    train_args = RewardConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=[],
        max_length=args.max_length,
        eos_token=eos_token,
        pad_token=tokenizer.pad_token,
        remove_unused_columns=False,
        dataset_num_proc=1,
    )

    trainer = RewardTrainer(
        model=model,
        args=train_args,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )
    try:
        trainer.train()
    except ValueError as e:
        if "num_samples=0" in str(e):
            # Intent: fail fast with actionable guidance when TRL length filtering removes all pairs.
            raise ValueError(
                "No training samples left after TRL length filtering. "
                "Increase --max_length (or reduce prompt size via frontier/pair settings) and retry."
            ) from e
        raise

    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    device = trainer.model.device
    train_metrics = evaluate_pairwise_accuracy(
        model=trainer.model,
        tokenizer=tokenizer,
        rows=train_rows,
        max_length=args.max_length,
        batch_size=args.per_device_eval_batch_size,
        device=device,
    )
    eval_metrics = evaluate_pairwise_accuracy(
        model=trainer.model,
        tokenizer=tokenizer,
        rows=eval_rows,
        max_length=args.max_length,
        batch_size=args.per_device_eval_batch_size,
        device=device,
    )

    summary = {
        "data_root": str(data_root),
        "subsets": subsets if subsets else "all",
        "num_branch_files": len(branch_files),
        "num_pairs_total": len(pairs),
        "num_pairs_train": len(train_rows),
        "num_pairs_eval": len(eval_rows),
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "args": vars(args),
    }
    with (output_dir / "train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    print("[done] saved model and summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
