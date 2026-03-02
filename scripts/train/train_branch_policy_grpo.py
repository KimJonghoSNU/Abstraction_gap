#!/usr/bin/env python3
"""
Train traversal ranking policy with TRL GRPOTrainer using subtree nDCG@10 reward.

This script builds per-step prompts from branch_steps.jsonl and computes reward from
model-generated ranking top2 against candidate subtree nDCG@10 values.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import AutoPeftModelForCausalLM, LoraConfig, prepare_model_for_kbit_training
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer


def find_repo_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "src").exists() and (parent / "scripts").exists() and (parent / "trees").exists():
            return parent
    raise RuntimeError(f"Failed to locate repository root from {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
SRC_DIR = REPO_ROOT / "src"
TRAIN_DIR = REPO_ROOT / "scripts" / "train"
TRAIN_BUILD_DIR = REPO_ROOT / "scripts" / "train" / "build_dataset"
for target in [SRC_DIR, TRAIN_DIR, TRAIN_BUILD_DIR]:
    if str(target) not in sys.path:
        sys.path.insert(0, str(target))

from build_traversal_rl_dataset import (  # noqa: E402
    DEFAULT_TREE_VERSION_MAP,
    build_node_cache,
    compute_ndcg,
    load_tree_for_subset,
    parse_tree_map_from_baseline_script,
)
from prompt_builder import TraversalPromptBuilder  # noqa: E402


EPS = 1e-6


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


def discover_branch_files(data_root: Path, subsets: Sequence[str]) -> List[Tuple[str, Path]]:
    files: List[Tuple[str, Path]] = []
    if subsets:
        for subset in subsets:
            fp = data_root / subset / "branch_steps.jsonl"
            if fp.exists():
                files.append((subset, fp))
    else:
        for fp in sorted(data_root.glob("*/branch_steps.jsonl")):
            files.append((fp.parent.name, fp))
    return files


def parse_tree_version_map(tree_map_source: str, baseline_run_script: Path) -> Dict[str, str]:
    if tree_map_source == "baseline_script":
        mapping = parse_tree_map_from_baseline_script(baseline_run_script)
        if len(mapping) == 0:
            raise RuntimeError(f"No subset->tree_version mapping found in {baseline_run_script}")
        return mapping
    if tree_map_source == "default":
        return dict(DEFAULT_TREE_VERSION_MAP)
    raise ValueError(f"Unknown tree_map_source: {tree_map_source}")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def count_non_empty_jsonl_rows(path: Path, max_rows: int = -1) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            count += 1
            if max_rows > 0 and count >= max_rows:
                break
    return count


def load_gold_map(terminal_rewards_path: Path) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = {}
    if not terminal_rewards_path.exists():
        return out
    for row in iter_jsonl(terminal_rewards_path):
        qidx = int(row.get("query_idx", -1))
        if qidx < 0:
            continue
        gold_ids = [str(x) for x in (row.get("gold_doc_ids", []) or [])]
        out[qidx] = gold_ids
    return out


def build_path_to_uid(node_cache: Dict[int, object]) -> Dict[Tuple[int, ...], int]:
    mapping: Dict[Tuple[int, ...], int] = {}
    for uid, info in node_cache.items():
        mapping[tuple(int(x) for x in info.path)] = uid
    return mapping


def resolve_docs_parquet_path(docs_root: Path, subset: str) -> Path:
    exact = docs_root / f"{subset}.parquet"
    if exact.exists():
        return exact
    pattern_matches = sorted(docs_root.glob(f"{subset}-*.parquet"))
    if len(pattern_matches) > 0:
        # Intent: support HF shard naming (e.g., subset-00000-of-00001.parquet) without forcing manual rename.
        return pattern_matches[0]
    raise FileNotFoundError(
        f"Documents parquet not found for subset={subset} under {docs_root}. "
        f"Tried: {exact.name} and {subset}-*.parquet"
    )


def cap_candidates_for_grpo(
    entries: Sequence[dict],
    *,
    candidate_cap: int,
    gold_min_keep: int,
) -> List[dict]:
    if candidate_cap <= 0 or len(entries) <= candidate_cap:
        return list(entries)

    gold_min_keep = max(0, int(gold_min_keep))

    def _path_key(entry: dict) -> Tuple[int, ...]:
        return tuple(int(x) for x in entry.get("path", []))

    def _gold_key(entry: dict) -> Tuple[float, int, Tuple[int, ...]]:
        return (-float(entry.get("candidate_ndcg", 0.0)), -int(entry.get("pos_overlap_count", 0)), _path_key(entry))

    def _nongold_key(entry: dict) -> Tuple[float, float, Tuple[int, ...]]:
        return (-float(entry.get("reward", 0.0)), -float(entry.get("candidate_ndcg", 0.0)), _path_key(entry))

    gold_entries = [x for x in entries if bool(x.get("has_pos", False))]
    nongold_entries = [x for x in entries if not bool(x.get("has_pos", False))]

    selected: List[dict] = []
    selected_keys: set[Tuple[int, ...]] = set()

    def _append_if_room(cands: Sequence[dict], *, sorter, limit: Optional[int] = None) -> None:
        if len(selected) >= candidate_cap:
            return
        used = 0
        for item in sorted(cands, key=sorter):
            if len(selected) >= candidate_cap:
                break
            key = _path_key(item)
            if key in selected_keys:
                continue
            selected.append(item)
            selected_keys.add(key)
            used += 1
            if (limit is not None) and (used >= limit):
                break

    if gold_entries and gold_min_keep > 0:
        # Intent: preserve at least one gold-supporting branch after capping to keep supervision aligned with target evidence path.
        _append_if_room(gold_entries, sorter=_gold_key, limit=min(gold_min_keep, len(gold_entries)))

    # Intent: after minimum gold keep, prioritize hard negatives to maintain discriminative training signal.
    _append_if_room(nongold_entries, sorter=_nongold_key)
    _append_if_room(gold_entries, sorter=_gold_key)
    _append_if_room(entries, sorter=_nongold_key)
    return selected[:candidate_cap]


def build_grpo_rows(
    *,
    data_root: Path,
    subsets: Sequence[str],
    docs_root: Path,
    tree_version_map: Dict[str, str],
    max_steps_per_subset: int,
    candidate_cap: int,
    gold_min_keep: int,
    prompt_builder: TraversalPromptBuilder,
) -> Tuple[List[dict], Dict[str, dict]]:
    rows: List[dict] = []
    per_subset_stats: Dict[str, dict] = {}

    branch_files = discover_branch_files(data_root=data_root, subsets=subsets)
    if len(branch_files) == 0:
        raise FileNotFoundError(f"No branch_steps.jsonl found under {data_root}")
    print(f"[build] found {len(branch_files)} subset files under {data_root}")

    for subset, branch_path in tqdm(branch_files, desc="GRPO rows | subsets", unit="subset"):
        tree_version = tree_version_map.get(subset)
        if not tree_version:
            raise KeyError(f"Tree version not found for subset={subset}")

        docs_parquet_path = resolve_docs_parquet_path(docs_root=docs_root, subset=subset)
        print(f"[build:{subset}] docs={docs_parquet_path.name} tree_version={tree_version}")

        docs_df = pd.read_parquet(docs_parquet_path, columns=["id"])
        docs_df["id"] = docs_df["id"].astype(str)

        tree_root = load_tree_for_subset(subset=subset, tree_version=tree_version)
        _, node_cache = build_node_cache(tree_root, docs_df)
        path_to_uid = build_path_to_uid(node_cache)

        terminal_rewards_path = branch_path.parent / "terminal_rewards.jsonl"
        gold_map = load_gold_map(terminal_rewards_path)

        stats = {
            "num_steps_seen": 0,
            "num_steps_used": 0,
            "num_steps_skipped_no_gold": 0,
            "num_steps_skipped_no_candidates": 0,
            "num_steps_skipped_missing_path": 0,
            "num_steps_capped": 0,
            "num_steps_dropped_after_cap": 0,
            "num_steps_gold_present_before_cap": 0,
            "num_steps_gold_kept_after_cap": 0,
            "sum_candidates_before": 0,
            "sum_candidates_after": 0,
        }

        total_rows = count_non_empty_jsonl_rows(branch_path, max_rows=max_steps_per_subset)
        row_iter = tqdm(
            iter_jsonl(branch_path),
            total=total_rows,
            desc=f"GRPO rows | {subset}",
            unit="step",
            leave=False,
        )
        for idx, row in enumerate(row_iter):
            if max_steps_per_subset > 0 and idx >= max_steps_per_subset:
                break

            stats["num_steps_seen"] += 1
            candidates = row.get("candidates", []) or []
            if len(candidates) < 2:
                stats["num_steps_skipped_no_candidates"] += 1
                continue

            query_idx = int(row.get("query_idx", -1))
            gold_doc_ids = gold_map.get(query_idx, [])
            if len(gold_doc_ids) == 0:
                stats["num_steps_skipped_no_gold"] += 1
                continue

            gold_doc_id_set = set(str(x) for x in gold_doc_ids)
            candidate_entries: List[dict] = []
            missing_path = False

            for cand in candidates:
                path = [int(x) for x in cand.get("path", [])]
                uid = path_to_uid.get(tuple(path))
                if uid is None:
                    missing_path = True
                    break

                docs_top10 = node_cache[uid].doc_list[:10]
                ndcg = compute_ndcg(docs_top10, gold_doc_ids, k=10)
                pos_overlap_count = len(node_cache[uid].doc_set & gold_doc_id_set)
                candidate_entries.append(
                    {
                        "desc": str(cand.get("desc", "")).strip(),
                        "path": path,
                        "reward": float(cand.get("reward", 0.0)),
                        "has_pos": bool(pos_overlap_count > 0),
                        "has_neg": bool(cand.get("has_neg", False)),
                        "pos_overlap_count": int(pos_overlap_count),
                        "candidate_ndcg": float(ndcg),
                    }
                )

            if missing_path:
                stats["num_steps_skipped_missing_path"] += 1
                continue

            before_count = len(candidate_entries)
            stats["sum_candidates_before"] += int(before_count)
            if any(bool(x["has_pos"]) for x in candidate_entries):
                stats["num_steps_gold_present_before_cap"] += 1

            capped_entries = cap_candidates_for_grpo(
                candidate_entries,
                candidate_cap=candidate_cap,
                gold_min_keep=gold_min_keep,
            )
            after_count = len(capped_entries)
            stats["sum_candidates_after"] += int(after_count)
            if before_count > after_count:
                stats["num_steps_capped"] += 1
            if any(bool(x["has_pos"]) for x in capped_entries):
                stats["num_steps_gold_kept_after_cap"] += 1
            if after_count < 2:
                stats["num_steps_dropped_after_cap"] += 1
                continue

            candidate_descs = [str(x["desc"]) for x in capped_entries]
            candidate_paths = [list(x["path"]) for x in capped_entries]
            candidate_rewards = [float(x["reward"]) for x in capped_entries]
            candidate_ndcgs = [float(x["candidate_ndcg"]) for x in capped_entries]

            prompt = prompt_builder.build(
                query=str(row.get("query", "")),
                candidate_descs=candidate_descs,
                subset=subset,
            )

            rows.append(
                {
                    "prompt": prompt,
                    "subset": subset,
                    "query_idx": query_idx,
                    "depth": int(row.get("depth", -1)),
                    "candidate_ndcgs": candidate_ndcgs,
                    "candidate_rewards": candidate_rewards,
                    "candidate_paths": candidate_paths,
                    "gold_doc_ids": gold_doc_ids,
                }
            )
            stats["num_steps_used"] += 1

        per_subset_stats[subset] = stats
        print(
            "[build:{subset}] seen={seen} used={used} skip_no_gold={skip_no_gold} "
            "skip_no_candidates={skip_no_cands} skip_missing_path={skip_missing} "
            "capped={capped} dropped_after_cap={dropped_cap} avg_cands={avg_before:.1f}->{avg_after:.1f} "
            "gold_kept={gold_kept}/{gold_seen}".format(
                subset=subset,
                seen=stats["num_steps_seen"],
                used=stats["num_steps_used"],
                skip_no_gold=stats["num_steps_skipped_no_gold"],
                skip_no_cands=stats["num_steps_skipped_no_candidates"],
                skip_missing=stats["num_steps_skipped_missing_path"],
                capped=stats["num_steps_capped"],
                dropped_cap=stats["num_steps_dropped_after_cap"],
                avg_before=(float(stats["sum_candidates_before"]) / max(1, int(stats["num_steps_seen"]))),
                avg_after=(float(stats["sum_candidates_after"]) / max(1, int(stats["num_steps_seen"]))),
                gold_kept=stats["num_steps_gold_kept_after_cap"],
                gold_seen=stats["num_steps_gold_present_before_cap"],
            )
        )

    return rows, per_subset_stats


def split_by_query(rows: Sequence[dict], eval_ratio: float, seed: int) -> Tuple[List[dict], List[dict]]:
    if eval_ratio <= 0.0:
        # Intent: allow eval-free training runs to reduce GRPO wall-clock and memory overhead when needed.
        return list(rows), []

    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        key = f"{row['subset']}::{row['query_idx']}"
        grouped[key].append(row)

    keys = list(grouped.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)

    if len(keys) <= 1:
        return list(rows), []

    n_eval = int(round(len(keys) * eval_ratio))
    n_eval = min(max(n_eval, 1), len(keys) - 1)
    eval_keys = set(keys[:n_eval])

    train_rows: List[dict] = []
    eval_rows: List[dict] = []
    for key, bucket in grouped.items():
        if key in eval_keys:
            eval_rows.extend(bucket)
        else:
            train_rows.extend(bucket)
    return train_rows, eval_rows


def try_parse_json_blob(text: str) -> Optional[dict]:
    if not text:
        return None
    stripped = text.strip()
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if code_block_match:
        candidate = code_block_match.group(1)
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    left = stripped.find("{")
    right = stripped.rfind("}")
    if left >= 0 and right > left:
        candidate = stripped[left : right + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def recursive_find_ranking(obj) -> Optional[List[int]]:
    if isinstance(obj, dict):
        if "ranking" in obj and isinstance(obj["ranking"], list):
            out: List[int] = []
            for x in obj["ranking"]:
                try:
                    out.append(int(x))
                except Exception:
                    continue
            return out
        for value in obj.values():
            found = recursive_find_ranking(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = recursive_find_ranking(value)
            if found is not None:
                return found
    return None


def completion_to_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        if "content" in completion:
            return str(completion.get("content", ""))
        return json.dumps(completion, ensure_ascii=False)
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(completion)


def compute_top2_ndcg_reward(ranking: Sequence[int], candidate_ndcgs: Sequence[float]) -> float:
    valid: List[int] = []
    max_idx = len(candidate_ndcgs)
    for idx in ranking:
        if idx < 0 or idx >= max_idx:
            continue
        if idx in valid:
            continue
        valid.append(idx)
        if len(valid) == 2:
            break

    if len(valid) < 2:
        return 0.0
    # Intent: optimize the actual beam action by rewarding the mean subtree nDCG@10 of the selected top2 branches.
    return float((candidate_ndcgs[valid[0]] + candidate_ndcgs[valid[1]]) / 2.0)


def maybe_get_bnb_config(load_in_4bit: bool, bf16: bool) -> Optional[BitsAndBytesConfig]:
    if not load_in_4bit:
        return None
    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_policy_model(
    *,
    model_name_or_path: str,
    bnb_config: Optional[BitsAndBytesConfig],
    bf16: bool,
    fp16: bool,
    local_rank: int,
    enable_peft: bool,
) -> Tuple[torch.nn.Module, Optional[LoraConfig]]:
    model_path = Path(model_name_or_path)
    common_kwargs = {"trust_remote_code": True}
    if bnb_config is not None:
        common_kwargs["quantization_config"] = bnb_config
        if torch.cuda.is_available():
            current_device = int(torch.cuda.current_device())
            # Intent: always pin 4bit weights to a single active CUDA device to avoid cross-device training errors.
            common_kwargs["device_map"] = {"": current_device}
        else:
            common_kwargs["device_map"] = "auto"
    elif bf16:
        common_kwargs["torch_dtype"] = torch.bfloat16
    elif fp16:
        common_kwargs["torch_dtype"] = torch.float16

    if (not enable_peft) and model_path.exists() and (model_path / "adapter_config.json").exists():
        # Intent: server-compatible full-weight training requires a base/merged model, not adapter-only checkpoints.
        raise ValueError(
            "disable_peft/server_compatible_mode is enabled, but model path looks like adapter-only "
            f"({model_name_or_path}). Use a base model or a merged full checkpoint."
        )

    if enable_peft and model_path.exists() and (model_path / "adapter_config.json").exists():
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_name_or_path,
            is_trainable=True,
            **common_kwargs,
        )
        return model, None

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **common_kwargs)
    if not enable_peft:
        # Intent: avoid LoRA injection path to keep trainer/vLLM parameter sync shape-compatible.
        return model, None
    if bnb_config is not None:
        # Intent: prepare quantized base model for low-rank adapter training in GRPO fine-tuning.
        model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    return model, peft_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train traversal policy with GRPO and subtree nDCG reward.")
    parser.add_argument("--data_root", type=str, required=True, help="Path containing subset/branch_steps.jsonl.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--subsets", type=str, default="all")
    parser.add_argument("--tree_map_source", type=str, choices=["baseline_script", "default"], default="baseline_script")
    parser.add_argument(
        "--baseline_run_script",
        type=str,
        default=str(REPO_ROOT / "src" / "bash" / "baselines" / "run_baseline1_tree_only.sh"),
    )
    parser.add_argument("--docs_root", type=str, default=str(REPO_ROOT / "data" / "BRIGHT" / "documents"))

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--max_steps_per_subset", type=int, default=-1)
    parser.add_argument("--candidate_cap", type=int, default=20)
    parser.add_argument("--gold_min_keep", type=int, default=1)
    parser.add_argument("--max_prompt_proto_size", type=int, default=0)
    parser.add_argument("--max_desc_char_len", type=int, default=1200)
    parser.add_argument(
        "--prompt_template_file",
        type=str,
        default="",
        help=(
            "Optional custom prompt template file. "
            "Supported tokens: {{QUERY}}, {{CANDIDATES}}, {{RELEVANCE_DEFINITION}}, {{SUBSET}}, {{PROMPT_ID}}."
        ),
    )

    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--generation_batch_size", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--kl_beta", type=float, default=0.01)
    parser.add_argument("--use_vllm", action="store_true", default=False)
    parser.add_argument("--vllm_server_host", type=str, default="127.0.0.1")
    parser.add_argument("--vllm_server_port", type=int, default=8000)
    parser.add_argument("--vllm_server_base_url", type=str, default="")
    parser.add_argument("--vllm_group_port", type=int, default=51216)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--disable_peft", action="store_true", default=False)
    parser.add_argument(
        "--server_compatible_mode",
        action="store_true",
        default=False,
        help="Disable LoRA/4bit for safer trainer-vLLM weight sync compatibility.",
    )
    parser.add_argument("--cache_dataset_jsonl", type=str, default="")
    args = parser.parse_args()

    disable_peft = bool(args.disable_peft or args.server_compatible_mode)
    effective_load_in_4bit = bool(args.load_in_4bit)
    if args.server_compatible_mode and args.load_in_4bit:
        # Intent: 4bit+LoRA has shown unstable sync behavior with external vLLM server updates in this project setup.
        print("[warn] server_compatible_mode enabled: overriding load_in_4bit=False")
        effective_load_in_4bit = False
    if disable_peft and effective_load_in_4bit:
        raise ValueError("disable_peft/server_compatible_mode cannot be combined with load_in_4bit.")

    print("[config] parsed args:")
    print(json.dumps(vars(args), ensure_ascii=False, indent=4))
    print(
        "[config] effective flags: "
        f"disable_peft={disable_peft}, load_in_4bit={effective_load_in_4bit}, "
        f"server_compatible_mode={args.server_compatible_mode}"
    )

    seed_everything(args.seed)
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank >= 0 and torch.cuda.is_available():
        # Intent: each distributed rank must select its own CUDA device before loading quantized weights.
        torch.cuda.set_device(local_rank)

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    docs_root = Path(args.docs_root)

    subsets = parse_subsets(args.subsets)
    tree_version_map = parse_tree_version_map(
        tree_map_source=args.tree_map_source,
        baseline_run_script=Path(args.baseline_run_script),
    )
    prompt_builder = TraversalPromptBuilder(
        max_prompt_proto_size=args.max_prompt_proto_size,
        max_desc_char_len=args.max_desc_char_len,
        prompt_template_file=args.prompt_template_file,
    )

    rows, per_subset_stats = build_grpo_rows(
        data_root=data_root,
        subsets=subsets,
        docs_root=docs_root,
        tree_version_map=tree_version_map,
        max_steps_per_subset=args.max_steps_per_subset,
        candidate_cap=args.candidate_cap,
        gold_min_keep=args.gold_min_keep,
        prompt_builder=prompt_builder,
    )
    if len(rows) == 0:
        raise RuntimeError("No GRPO training rows constructed from branch_steps.jsonl")
    print(f"[build] total rows={len(rows)}")

    if args.cache_dataset_jsonl:
        cache_path = Path(args.cache_dataset_jsonl)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as f:
            for row in tqdm(rows, desc="GRPO rows | cache write", unit="row"):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[build] cached rows -> {cache_path}")

    train_rows, eval_rows = split_by_query(rows, eval_ratio=args.eval_ratio, seed=args.seed)
    print(f"[split] train_rows={len(train_rows)} eval_rows={len(eval_rows)}")
    for subset_name, subset_stats in per_subset_stats.items():
        print(f"[split:{subset_name}] {subset_stats}")
    train_ds = Dataset.from_list(train_rows)
    eval_ds = Dataset.from_list(eval_rows) if len(eval_rows) > 0 else None

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.pad_token is None:
        raise ValueError("Tokenizer has no pad/eos/unk token; cannot run GRPO training safely.")
    tokenizer.padding_side = "left"

    bnb_config = maybe_get_bnb_config(load_in_4bit=effective_load_in_4bit, bf16=args.bf16)
    model, peft_config = load_policy_model(
        model_name_or_path=args.model_name_or_path,
        bnb_config=bnb_config,
        bf16=args.bf16,
        fp16=args.fp16,
        local_rank=local_rank,
        enable_peft=(not disable_peft),
    )

    def branch_ndcg_reward_func(prompts, completions, candidate_ndcgs, **kwargs):
        rewards: List[float] = []
        for completion, ndcgs in zip(completions, candidate_ndcgs, strict=True):
            text = completion_to_text(completion)
            parsed = try_parse_json_blob(text)
            if parsed is None:
                rewards.append(0.0)
                continue
            ranking = recursive_find_ranking(parsed)
            if ranking is None:
                rewards.append(0.0)
                continue
            rewards.append(compute_top2_ndcg_reward(ranking, ndcgs))
        return rewards

    train_args_kwargs = dict(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        temperature=args.temperature,
        beta=args.kl_beta,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to=[],
    )
    if args.generation_batch_size is not None and args.generation_batch_size > 0:
        # Intent: only force generation_batch_size when explicitly set; otherwise keep TRL default batching behavior.
        train_args_kwargs["generation_batch_size"] = int(args.generation_batch_size)
    grpo_config_params = set(inspect.signature(GRPOConfig.__init__).parameters.keys())
    if args.use_vllm:
        # Intent: offload GRPO generation calls to an external vLLM server to reduce trainer-side generation memory pressure.
        train_args_kwargs["use_vllm"] = True
        if "vllm_server_host" in grpo_config_params:
            # Intent: force IPv4 localhost rendezvous to avoid NCCL socket failures on IPv6-disabled systems.
            train_args_kwargs["vllm_server_host"] = str(args.vllm_server_host).strip() or "127.0.0.1"
        if "vllm_group_port" in grpo_config_params:
            # Intent: pass communicator group port only when installed TRL version supports this field.
            train_args_kwargs["vllm_group_port"] = int(args.vllm_group_port)
        else:
            print("[warn] Installed TRL does not support vllm_group_port; skipping it.")

        if str(args.vllm_server_base_url).strip():
            if "vllm_server_base_url" in grpo_config_params:
                train_args_kwargs["vllm_server_base_url"] = str(args.vllm_server_base_url).strip()
            else:
                print("[warn] Installed TRL does not support vllm_server_base_url; falling back to vllm_server_port.")
                if "vllm_server_port" in grpo_config_params:
                    train_args_kwargs["vllm_server_port"] = int(args.vllm_server_port)
        else:
            if "vllm_server_port" in grpo_config_params:
                train_args_kwargs["vllm_server_port"] = int(args.vllm_server_port)
            else:
                raise ValueError("Installed TRL does not expose vllm_server_port in GRPOConfig; cannot use external vLLM server.")

    train_args = GRPOConfig(**train_args_kwargs)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[branch_ndcg_reward_func],
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("[train] starting GRPOTrainer.train()")
    trainer.train()
    print("[train] finished GRPOTrainer.train()")

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    summary = {
        "data_root": str(data_root),
        "subsets": subsets if subsets else "all",
        "num_rows_total": len(rows),
        "num_rows_train": len(train_rows),
        "num_rows_eval": len(eval_rows),
        "model_name_or_path": args.model_name_or_path,
        "output_dir": str(output_dir),
        "args": vars(args),
        "stats_by_subset": per_subset_stats,
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=4), encoding="utf-8")
    print(f"[done] saved model to {final_dir}")


if __name__ == "__main__":
    main()
