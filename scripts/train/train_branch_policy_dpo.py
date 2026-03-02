#!/usr/bin/env python3
"""
Train traversal ranking policy with TRL DPOTrainer.

Expected input columns per row:
- prompt: str
- chosen: str (JSON response)
- rejected: str (JSON response)
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def rows_to_dataset(rows: Sequence[dict]) -> Dataset:
    out_rows = []
    for row in rows:
        out_rows.append(
            {
                "prompt": str(row["prompt"]),
                "chosen": str(row["chosen"]),
                "rejected": str(row["rejected"]),
            }
        )
    return Dataset.from_list(out_rows)


def maybe_get_bnb_config(load_in_4bit: bool, bf16: bool) -> BitsAndBytesConfig | None:
    if not load_in_4bit:
        return None
    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train traversal policy with DPO.")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--eval_jsonl", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--fp16", action="store_true", default=False)

    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    args = parser.parse_args()

    seed_everything(args.seed)
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank >= 0 and torch.cuda.is_available():
        # Intent: bind each distributed rank to its dedicated CUDA device before loading quantized weights.
        torch.cuda.set_device(local_rank)

    train_jsonl = Path(args.train_jsonl)
    eval_jsonl = Path(args.eval_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(train_jsonl)
    eval_rows = load_jsonl(eval_jsonl)
    if len(train_rows) == 0:
        raise RuntimeError(f"No training rows in {train_jsonl}")

    train_ds = rows_to_dataset(train_rows)
    eval_ds = rows_to_dataset(eval_rows) if len(eval_rows) > 0 else None

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.pad_token is None:
        raise ValueError("Tokenizer has no pad/eos/unk token; cannot run DPO training safely.")

    bnb_config = maybe_get_bnb_config(load_in_4bit=args.load_in_4bit, bf16=args.bf16)
    model_kwargs = {
        "trust_remote_code": True,
    }
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
        if local_rank >= 0:
            current_device = int(torch.cuda.current_device()) if torch.cuda.is_available() else 0
            # Intent: bind quantized weights to the process-local CUDA device (safe for both per-rank and shared visibility launch modes).
            model_kwargs["device_map"] = {"": current_device}
        else:
            model_kwargs["device_map"] = "auto"
    elif args.bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif args.fp16:
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    if bnb_config is not None:
        # Intent: enable stable QLoRA optimization on 4-bit base weights.
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    train_args = DPOConfig(
        output_dir=str(output_dir),
        beta=args.beta,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=[],
        remove_unused_columns=False,
    )

    # Intent: keep a single quantized base model in memory by using PEFT adapters instead of a separate full ref model.
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(final_dir))
        summary = {
            "train_jsonl": str(train_jsonl),
            "eval_jsonl": str(eval_jsonl),
            "num_train_rows": len(train_rows),
            "num_eval_rows": len(eval_rows),
            "model_name_or_path": args.model_name_or_path,
            "output_dir": str(output_dir),
            "args": vars(args),
        }
        (output_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=4), encoding="utf-8")
        print(f"[done] saved model to {final_dir}")


if __name__ == "__main__":
    main()
