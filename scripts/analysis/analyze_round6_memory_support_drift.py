#!/usr/bin/env python3
"""
Analyze whether unsupported hypothesis slots are automatically dropped or revised
across iterations in the frontiercum_qstate round6 run.

This script treats `possible_answer_docs` as the hypothesis decomposition because
it is the cleanest structured view of the query state saved in the artifacts.
Support is measured against the current iteration's top-10 retrieved leaf texts
using the same reason-embed model family used by retrieval.

Outputs:
- <out_prefix>_slot_rows.csv
- <out_prefix>_subset_iter_summary.csv
- <out_prefix>_overall_iter_summary.csv
- <out_prefix>_examples.csv
"""

from __future__ import annotations

import argparse
import math
import pickle
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT / "src"))

from retrievers import get_reasonembed_task_description  # noqa: E402
from retrievers.reasonembed import ReasonEmbedEmbeddingModel  # noqa: E402
DEFAULT_MODEL_PATH = "/data2/pretrained_models/reason-embed-qwen3-8b-0928"
DEFAULT_TOPK = 10
DEFAULT_RETAIN_SIM_THRESHOLD = 0.90
DEFAULT_DROP_SIM_THRESHOLD = 0.60
DEFAULT_SUBSETS: List[str] = [
    "biology",
    "earth_science",
    "economics",
    "pony",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
    "theoremqa_theorems",
]


@dataclass(frozen=True)
class SlotRecord:
    subset: str
    sample_idx: int
    iter_idx: int
    slot_key: str
    slot_text: str
    query_post: str
    next_query_post: str
    top10_paths: Tuple[Tuple[int, ...], ...]
    next_top10_paths: Tuple[Tuple[int, ...], ...]
    top10_doc_texts: Tuple[str, ...]
    next_top10_doc_texts: Tuple[str, ...]
    next_slots: Tuple[Tuple[str, str], ...]


def _normalize_text(text: Any) -> str:
    value = str(text or "").strip()
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _truncate(text: str, limit: int = 220) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _mean(values: Iterable[float]) -> float:
    filtered: List[float] = []
    for value in values:
        if value is None:
            continue
        value_f = float(value)
        if math.isnan(value_f):
            continue
        filtered.append(value_f)
    if not filtered:
        return float("nan")
    return float(sum(filtered) / len(filtered))


def _find_run_paths(base_dir: Path, subsets: Sequence[str]) -> Dict[str, Path]:
    run_paths: Dict[str, Path] = {}
    for subset in subsets:
        candidates = sorted((base_dir / subset / "round6").glob("**/all_eval_sample_dicts.pkl"))
        matches: List[Path] = []
        for path in candidates:
            path_str = str(path)
            if "frontiercum_qstate" not in path_str:
                continue
            if "reason/embed-qwen3-8b-0928" not in path_str:
                continue
            if "round6_mrr_selector_accum_meanscore_global_expandable_ended_reseat" not in path_str:
                continue
            if "RERP=random" in path_str:
                continue
            if "REmrM=True" in path_str:
                continue
            matches.append(path)
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one score frontiercum_qstate run for subset={subset}, "
                f"got {len(matches)}: {[str(path) for path in matches]}"
            )
        run_paths[subset] = matches[0]
    return run_paths


def _normalize_paths(paths: Sequence[Any], *, limit: int | None = None) -> List[Tuple[int, ...]]:
    normalized: List[Tuple[int, ...]] = []
    for path in paths or []:
        if not path:
            continue
        normalized.append(tuple(int(x) for x in list(path)))
        if limit is not None and len(normalized) >= limit:
            break
    return normalized


def _load_leaf_catalog(subset_dir: Path) -> Tuple[Dict[Tuple[int, ...], str], Dict[Tuple[int, ...], np.ndarray]]:
    import json

    catalog_path = subset_dir / "node_catalog.jsonl"
    emb_path = subset_dir / "node_embs.reasonembed8b.npy"
    node_embs = np.load(emb_path)

    leaf_desc_by_path: Dict[Tuple[int, ...], str] = {}
    leaf_emb_by_path: Dict[Tuple[int, ...], np.ndarray] = {}
    with open(catalog_path, "r") as handle:
        for line in handle:
            row = json.loads(line)
            if not bool(row.get("is_leaf", False)):
                continue
            path = tuple(int(x) for x in (row.get("path", []) or []))
            registry_idx = int(row["registry_idx"])
            leaf_desc_by_path[path] = _normalize_text(row.get("desc", ""))
            leaf_emb_by_path[path] = node_embs[registry_idx].astype(np.float32, copy=False)
    return leaf_desc_by_path, leaf_emb_by_path


def _extract_slots(iter_record: Mapping[str, Any]) -> List[Tuple[str, str]]:
    pad = iter_record.get("possible_answer_docs")
    if isinstance(pad, dict):
        slots: List[Tuple[str, str]] = []
        for key, value in pad.items():
            normalized = _normalize_text(value)
            if normalized:
                slots.append((str(key), normalized))
        if slots:
            return slots
    query_state_after = _normalize_text(iter_record.get("query_state_after", ""))
    if query_state_after:
        # Intent: keep the fallback coarse, because query_state_after has already
        # collapsed multiple hypotheses into one string and cannot support clean slot tracking.
        return [("query_state_after", query_state_after)]
    return []


def _collect_subset_records(
    *,
    subset: str,
    sample_path: Path,
    leaf_desc_by_path: Mapping[Tuple[int, ...], str],
    topk: int,
    max_queries_per_subset: int | None,
    sample_seed: int,
) -> Tuple[List[SlotRecord], List[str]]:
    with open(sample_path, "rb") as handle:
        samples = pickle.load(handle)
    if max_queries_per_subset is not None and len(samples) > max_queries_per_subset:
        rng = random.Random(sample_seed)
        sampled_indices = sorted(rng.sample(range(len(samples)), max_queries_per_subset))
        samples = [samples[idx] for idx in sampled_indices]

    slot_records: List[SlotRecord] = []
    unique_slot_texts: List[str] = []
    seen_slot_texts: set[str] = set()

    for sample_idx, sample in enumerate(samples):
        iter_records = sample.get("iter_records", []) or []
        if len(iter_records) < 2:
            continue
        for iter_idx in range(len(iter_records) - 1):
            rec = iter_records[iter_idx]
            next_rec = iter_records[iter_idx + 1]
            cur_slots = _extract_slots(rec)
            next_slots = _extract_slots(next_rec)
            if not cur_slots:
                continue

            top10_paths = _normalize_paths(rec.get("active_eval_paths", []) or [], limit=topk)
            next_top10_paths = _normalize_paths(next_rec.get("active_eval_paths", []) or [], limit=topk)
            top10_doc_texts = tuple(
                leaf_desc_by_path.get(path, "")
                for path in top10_paths
                if leaf_desc_by_path.get(path, "")
            )
            next_top10_doc_texts = tuple(
                leaf_desc_by_path.get(path, "")
                for path in next_top10_paths
                if leaf_desc_by_path.get(path, "")
            )

            for slot_key, slot_text in cur_slots:
                if slot_text not in seen_slot_texts:
                    seen_slot_texts.add(slot_text)
                    unique_slot_texts.append(slot_text)
                for _, next_slot_text in next_slots:
                    if next_slot_text not in seen_slot_texts:
                        seen_slot_texts.add(next_slot_text)
                        unique_slot_texts.append(next_slot_text)
                slot_records.append(
                    SlotRecord(
                        subset=subset,
                        sample_idx=sample_idx,
                        iter_idx=iter_idx,
                        slot_key=slot_key,
                        slot_text=slot_text,
                        query_post=_normalize_text(rec.get("query_post", "")),
                        next_query_post=_normalize_text(next_rec.get("query_post", "")),
                        top10_paths=tuple(top10_paths),
                        next_top10_paths=tuple(next_top10_paths),
                        top10_doc_texts=top10_doc_texts,
                        next_top10_doc_texts=next_top10_doc_texts,
                        next_slots=tuple(next_slots),
                    )
                )
    return slot_records, unique_slot_texts


def _encode_query_texts(
    retriever: ReasonEmbedEmbeddingModel,
    *,
    subset: str,
    texts: Sequence[str],
    max_length: int,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    if not texts:
        return {}
    task_description = get_reasonembed_task_description(subset)
    prompted_texts = [f"Instruct: {task_description}\nQuery: {text}" for text in texts]
    embeddings = retriever.encode(prompted_texts, max_length=max_length, batch_size=batch_size)
    return {text: embeddings[idx] for idx, text in enumerate(texts)}


def _best_doc_support(
    slot_emb: np.ndarray,
    doc_paths: Sequence[Tuple[int, ...]],
    doc_text_by_path: Mapping[Tuple[int, ...], str],
    doc_emb_by_path: Mapping[Tuple[int, ...], np.ndarray],
) -> Tuple[float, float, str]:
    best_score = float("nan")
    best_rank = float("nan")
    best_doc_text = ""
    if not doc_paths:
        return best_score, best_rank, best_doc_text
    for rank_idx, doc_path in enumerate(doc_paths, start=1):
        doc_emb = doc_emb_by_path.get(doc_path)
        if doc_emb is None:
            continue
        score = float(np.dot(slot_emb, doc_emb))
        if math.isnan(best_score) or score > best_score:
            best_score = score
            best_rank = float(rank_idx)
            best_doc_text = doc_text_by_path.get(doc_path, "")
    return best_score, best_rank, best_doc_text


def _best_next_slot_match(
    slot_emb: np.ndarray,
    next_slots: Sequence[Tuple[str, str]],
    slot_emb_map: Mapping[str, np.ndarray],
) -> Tuple[str, str, float]:
    best_key = ""
    best_text = ""
    best_score = float("nan")
    for next_key, next_text in next_slots:
        next_emb = slot_emb_map.get(next_text)
        if next_emb is None:
            continue
        score = float(np.dot(slot_emb, next_emb))
        if math.isnan(best_score) or score > best_score:
            best_key = next_key
            best_text = next_text
            best_score = score
    return best_key, best_text, best_score


def _classify_transition(
    *,
    best_next_similarity: float,
    best_next_key: str,
    current_key: str,
    retain_threshold: float,
    drop_threshold: float,
) -> str:
    if math.isnan(best_next_similarity) or best_next_similarity < drop_threshold:
        return "dropped"
    if best_next_key == current_key and best_next_similarity >= retain_threshold:
        return "retained"
    return "rewritten"


def _assign_support_bucket(rows_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    q25 = float(rows_df["support_score"].quantile(0.25))
    q50 = float(rows_df["support_score"].quantile(0.50))
    q75 = float(rows_df["support_score"].quantile(0.75))

    def _bucket(score: float) -> str:
        if score <= q25:
            return "q1_lowest"
        if score <= q50:
            return "q2"
        if score <= q75:
            return "q3"
        return "q4_highest"

    rows_df = rows_df.copy()
    rows_df["support_bucket"] = rows_df["support_score"].map(_bucket)
    return rows_df, {"q25": q25, "q50": q50, "q75": q75}


def _build_summary(rows_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    subset_rows: List[Dict[str, Any]] = []
    for (subset, iter_idx), group in rows_df.groupby(["subset", "iter_idx"], sort=True):
        low_group = group[group["support_bucket"] == "q1_lowest"]
        high_group = group[group["support_bucket"] == "q4_highest"]
        subset_rows.append(
            {
                "subset": subset,
                "iter": int(iter_idx),
                "slot_count": int(len(group)),
                "mean_support": _mean(group["support_score"]),
                "low_slot_count": int(len(low_group)),
                "low_drop_rate": _mean(low_group["transition"] == "dropped"),
                "low_rewrite_rate": _mean(low_group["transition"] == "rewritten"),
                "low_retain_rate": _mean(low_group["transition"] == "retained"),
                "low_mean_next_support_delta": _mean(low_group["matched_next_support_delta"]),
                "low_mean_best_next_similarity": _mean(low_group["best_next_slot_similarity"]),
                "high_slot_count": int(len(high_group)),
                "high_drop_rate": _mean(high_group["transition"] == "dropped"),
                "high_rewrite_rate": _mean(high_group["transition"] == "rewritten"),
                "high_retain_rate": _mean(high_group["transition"] == "retained"),
                "high_mean_next_support_delta": _mean(high_group["matched_next_support_delta"]),
            }
        )
    subset_df = pd.DataFrame(subset_rows).sort_values(["subset", "iter"]).reset_index(drop=True)

    overall_rows: List[Dict[str, Any]] = []
    for iter_idx, group in subset_df.groupby("iter", sort=True):
        overall_rows.append(
            {
                "iter": int(iter_idx),
                "slot_count_mean": _mean(group["slot_count"]),
                "mean_support": _mean(group["mean_support"]),
                "low_slot_count_mean": _mean(group["low_slot_count"]),
                "low_drop_rate": _mean(group["low_drop_rate"]),
                "low_rewrite_rate": _mean(group["low_rewrite_rate"]),
                "low_retain_rate": _mean(group["low_retain_rate"]),
                "low_mean_next_support_delta": _mean(group["low_mean_next_support_delta"]),
                "low_mean_best_next_similarity": _mean(group["low_mean_best_next_similarity"]),
                "high_slot_count_mean": _mean(group["high_slot_count"]),
                "high_drop_rate": _mean(group["high_drop_rate"]),
                "high_rewrite_rate": _mean(group["high_rewrite_rate"]),
                "high_retain_rate": _mean(group["high_retain_rate"]),
                "high_mean_next_support_delta": _mean(group["high_mean_next_support_delta"]),
            }
        )
    overall_df = pd.DataFrame(overall_rows).sort_values("iter").reset_index(drop=True)
    return subset_df, overall_df


def _build_examples(rows_df: pd.DataFrame) -> pd.DataFrame:
    low_rows = rows_df[rows_df["support_bucket"] == "q1_lowest"].copy()
    low_rows["sort_gain"] = low_rows["matched_next_support_delta"].fillna(-999.0)
    retained = low_rows[low_rows["transition"] == "retained"].sort_values(
        ["support_score", "best_next_slot_similarity"], ascending=[True, False]
    ).head(10)
    rewritten = low_rows[low_rows["transition"] == "rewritten"].sort_values(
        ["sort_gain"], ascending=[False]
    ).head(10)
    dropped = low_rows[low_rows["transition"] == "dropped"].sort_values(
        ["support_score"], ascending=[True]
    ).head(10)
    examples = pd.concat([retained, rewritten, dropped], ignore_index=True)
    return examples[
        [
            "subset",
            "sample_idx",
            "iter_idx",
            "slot_key",
            "support_score",
            "best_doc_rank",
            "best_doc_snippet",
            "slot_text",
            "query_post",
            "transition",
            "best_next_slot_key",
            "best_next_slot_similarity",
            "matched_next_support_score",
            "matched_next_support_delta",
            "matched_next_slot_text",
            "next_query_post",
            "top10_doc_snippets",
        ]
    ].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_dir", type=Path, default=Path("results/BRIGHT"))
    parser.add_argument("--tree_dir", type=Path, default=Path("trees/BRIGHT"))
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--query_max_length", type=int, default=384)
    parser.add_argument("--retain_sim_threshold", type=float, default=DEFAULT_RETAIN_SIM_THRESHOLD)
    parser.add_argument("--drop_sim_threshold", type=float, default=DEFAULT_DROP_SIM_THRESHOLD)
    parser.add_argument("--max_queries_per_subset", type=int, default=None)
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument(
        "--subsets",
        nargs="*",
        default=None,
        help="Override subset list. Default is the 9 frontiercum_qstate score subsets.",
    )
    parser.add_argument(
        "--out_prefix",
        type=Path,
        default=Path("results/BRIGHT/analysis/round6_memory_support_drift"),
    )
    args = parser.parse_args()

    subsets = list(args.subsets or DEFAULT_SUBSETS)
    run_paths = _find_run_paths(args.base_dir, subsets)

    retriever = ReasonEmbedEmbeddingModel(
        args.model_path,
        local_files_only=True,
        task_description="",
    )

    all_rows: List[Dict[str, Any]] = []
    for subset in subsets:
        print(f"[subset] {subset}")
        sample_path = run_paths[subset]
        subset_tree_dir = args.tree_dir / subset
        leaf_desc_by_path, leaf_emb_by_path = _load_leaf_catalog(subset_tree_dir)
        slot_records, unique_slot_texts = _collect_subset_records(
            subset=subset,
            sample_path=sample_path,
            leaf_desc_by_path=leaf_desc_by_path,
            topk=args.topk,
            max_queries_per_subset=args.max_queries_per_subset,
            sample_seed=args.sample_seed,
        )
        print(
            f"  slot_records={len(slot_records)} "
            f"unique_slot_texts={len(unique_slot_texts)}"
        )
        slot_emb_map = _encode_query_texts(
            retriever,
            subset=subset,
            texts=unique_slot_texts,
            max_length=args.query_max_length,
            batch_size=args.batch_size,
        )
        print("  encoded slot texts")

        for record in slot_records:
            slot_emb = slot_emb_map.get(record.slot_text)
            if slot_emb is None:
                continue
            support_score, best_doc_rank, best_doc_text = _best_doc_support(
                slot_emb,
                record.top10_paths,
                leaf_desc_by_path,
                leaf_emb_by_path,
            )
            best_next_slot_key, best_next_slot_text, best_next_slot_similarity = _best_next_slot_match(
                slot_emb,
                record.next_slots,
                slot_emb_map,
            )
            matched_next_support_score = float("nan")
            if best_next_slot_text:
                matched_next_emb = slot_emb_map.get(best_next_slot_text)
                if matched_next_emb is not None:
                    matched_next_support_score, _, _ = _best_doc_support(
                        matched_next_emb,
                        record.next_top10_paths,
                        leaf_desc_by_path,
                        leaf_emb_by_path,
                    )
            matched_next_support_delta = (
                matched_next_support_score - support_score
                if not math.isnan(support_score) and not math.isnan(matched_next_support_score)
                else float("nan")
            )
            transition = _classify_transition(
                best_next_similarity=best_next_slot_similarity,
                best_next_key=best_next_slot_key,
                current_key=record.slot_key,
                retain_threshold=args.retain_sim_threshold,
                drop_threshold=args.drop_sim_threshold,
            )
            all_rows.append(
                {
                    "subset": record.subset,
                    "sample_idx": record.sample_idx,
                    "iter_idx": record.iter_idx,
                    "slot_key": record.slot_key,
                    "slot_text": record.slot_text,
                    "support_score": support_score,
                    "best_doc_rank": best_doc_rank,
                    "best_doc_snippet": _truncate(best_doc_text),
                    "query_post": record.query_post,
                    "next_query_post": record.next_query_post,
                    "best_next_slot_key": best_next_slot_key,
                    "matched_next_slot_text": best_next_slot_text,
                    "best_next_slot_similarity": best_next_slot_similarity,
                    "matched_next_support_score": matched_next_support_score,
                    "matched_next_support_delta": matched_next_support_delta,
                    "transition": transition,
                    # Intent: keep examples readable without storing the full 10 raw leaf texts inline.
                    "top10_doc_snippets": " || ".join(_truncate(text, limit=120) for text in record.top10_doc_texts[:3]),
                }
            )
        # Intent: keep long-running multi-subset analysis stable on the shared GPUs used for reason-embed.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows_df = pd.DataFrame(all_rows)
    rows_df, quantiles = _assign_support_bucket(rows_df)
    rows_df["q25_support_threshold"] = quantiles["q25"]
    rows_df["q50_support_threshold"] = quantiles["q50"]
    rows_df["q75_support_threshold"] = quantiles["q75"]

    subset_iter_df, overall_iter_df = _build_summary(rows_df)
    examples_df = _build_examples(rows_df)

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    rows_df.to_csv(f"{args.out_prefix}_slot_rows.csv", index=False)
    subset_iter_df.to_csv(f"{args.out_prefix}_subset_iter_summary.csv", index=False)
    overall_iter_df.to_csv(f"{args.out_prefix}_overall_iter_summary.csv", index=False)
    examples_df.to_csv(f"{args.out_prefix}_examples.csv", index=False)

    print(f"saved slot rows -> {args.out_prefix}_slot_rows.csv")
    print(f"saved subset iter summary -> {args.out_prefix}_subset_iter_summary.csv")
    print(f"saved overall iter summary -> {args.out_prefix}_overall_iter_summary.csv")
    print(f"saved examples -> {args.out_prefix}_examples.csv")
    print(
        "support quartiles:",
        f"q25={quantiles['q25']:.4f}",
        f"q50={quantiles['q50']:.4f}",
        f"q75={quantiles['q75']:.4f}",
    )


if __name__ == "__main__":
    main()
