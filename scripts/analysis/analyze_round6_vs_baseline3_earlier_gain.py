#!/usr/bin/env python3
"""
Analyze whether round6 reaches strong retrieval states earlier than baseline3.

Examples
--------
python scripts/analysis/analyze_round6_vs_baseline3_earlier_gain.py

python scripts/analysis/analyze_round6_vs_baseline3_earlier_gain.py \
    --window_end 3 \
    --out_dir results/BRIGHT/analysis/round6_vs_baseline3_earlier_gain
"""

import argparse
import json
import pickle as pkl
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd


DEFAULT_SUBSETS = [
    "aops",
    "biology",
    "earth_science",
    "economics",
    "leetcode",
    "pony",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
    "theoremqa_questions",
    "theoremqa_theorems",
]

DEFAULT_BASELINE_TOKENS = [
    "baseline3_leaf_only_loop",
    "agent_executor_v1_icl2",
    "PlTau=5.0",
    "RCT=10",
    "RSC=on",
]

DEFAULT_ROUND6_TOKENS = [
    "round6_mrr_selector_accum_meanscore_global_expandable_ended_reseat",
    "agent_executor_v1_icl2",
    "PlTau=5.0",
    "RCT=10",
]


def _infer_subset_from_path(path: Path) -> str:
    parts = list(path.parts)
    if "BRIGHT" in parts:
        idx = parts.index("BRIGHT")
        if idx + 1 < len(parts):
            return str(parts[idx + 1])
    return "unknown"


def _select_matching_path(
    paths: Sequence[Path],
    required_tokens: Sequence[str],
    label: str,
    subset: str,
) -> Path:
    matches = [path for path in paths if all(token in str(path) for token in required_tokens)]
    if not matches:
        raise FileNotFoundError(
            f"No {label} matched for subset={subset}. required_tokens={list(required_tokens)}"
        )

    if len(matches) == 1:
        return matches[0]

    # Intent: prefer the latest artifact when repeated reruns leave multiple matching outputs.
    latest = max(matches, key=lambda path: path.stat().st_mtime)
    print(
        f"[warn] Multiple {label} matches for subset={subset}; "
        f"using latest mtime: {latest}"
    )
    return latest


def _jaccard_similarity(a: Sequence[Tuple[int, ...]], b: Sequence[Tuple[int, ...]]) -> float:
    set_a = set(tuple(x) for x in (a or []))
    set_b = set(tuple(x) for x in (b or []))
    if not set_a and not set_b:
        return 1.0
    return float(len(set_a & set_b) / len(set_a | set_b))


def _first_true_iter(curve_df: pd.DataFrame, predicate_col: str) -> Optional[int]:
    hits = curve_df[curve_df[predicate_col].fillna(False)]
    if hits.empty:
        return None
    return int(hits["iter"].min())


def _first_sustained_true_iter(curve_df: pd.DataFrame, predicate_col: str) -> Optional[int]:
    curve_df = curve_df.sort_values("iter").reset_index(drop=True)
    if curve_df.empty:
        return None

    flags = curve_df[predicate_col].fillna(False).astype(bool).tolist()
    iters = curve_df["iter"].tolist()
    for idx, flag in enumerate(flags):
        if flag and all(flags[idx:]):
            return int(iters[idx])
    return None


def _append_overall_row(
    df: pd.DataFrame,
    subset_col: str = "subset",
    overall_label: str = "overall",
) -> pd.DataFrame:
    if df.empty:
        return df

    numeric_cols = [
        col for col in df.columns
        if col != subset_col and pd.api.types.is_numeric_dtype(df[col])
    ]
    overall_row: Dict[str, Any] = {subset_col: overall_label}
    for col in numeric_cols:
        overall_row[col] = float(df[col].mean())
    non_numeric = [col for col in df.columns if col not in numeric_cols and col != subset_col]
    for col in non_numeric:
        overall_row[col] = ""
    return pd.concat([df, pd.DataFrame([overall_row])], ignore_index=True)


def _load_baseline_metrics(metrics_path: Path, subset: str) -> pd.DataFrame:
    rows_by_key: Dict[Tuple[int, int], Dict[str, Any]] = {}
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            query_idx = int(rec["query_idx"])
            iter_idx = int(rec["iter"])
            # Intent: old baseline files may contain repeated full-run appends, so keep the latest row per query/iter.
            rows_by_key[(query_idx, iter_idx)] = {
                "subset": subset,
                "query_idx": query_idx,
                "iter": iter_idx,
                "ndcg10": float(rec["nDCG@10"]),
                "query": str(rec.get("query", "") or ""),
            }
    df = pd.DataFrame(rows_by_key.values())
    if df.empty:
        raise RuntimeError(f"Baseline metrics are empty: {metrics_path}")
    return df


def _load_baseline_doc_sequences(records_path: Path, subset: str) -> Dict[int, Dict[int, List[str]]]:
    by_query: Dict[int, Dict[int, List[str]]] = defaultdict(dict)
    if not records_path.exists():
        return by_query

    with open(records_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if str(rec.get("phase", "")) != "iter_retrieval":
                continue
            query_idx = int(rec["query_idx"])
            iter_idx = int(rec["iter"])
            by_query[query_idx][iter_idx] = [str(x) for x in (rec.get("retrieved_doc_ids", []) or [])]
    return by_query


def _load_round6_samples(eval_samples_path: Path) -> List[Dict[str, Any]]:
    with open(eval_samples_path, "rb") as f:
        samples = pkl.load(f)
    if not isinstance(samples, list):
        raise TypeError(f"Unexpected round6 sample payload type: {type(samples)}")
    return samples


def _load_round6_metrics_and_carry(
    eval_samples_path: Path,
    subset: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    samples = _load_round6_samples(eval_samples_path)
    metric_rows: List[Dict[str, Any]] = []
    carry_rows: List[Dict[str, Any]] = []

    for query_idx, sample in enumerate(samples):
        original_query = str(sample.get("original_query", "") or "")
        iter_records = sample.get("iter_records", []) or []

        for rec in iter_records:
            metrics = rec.get("metrics", {}) or {}
            metric_rows.append(
                {
                    "subset": subset,
                    "query_idx": int(query_idx),
                    "iter": int(rec.get("iter", -1)),
                    "ndcg10": float(metrics.get("nDCG@10", float("nan"))),
                    "query": original_query,
                }
            )

        for idx in range(len(iter_records) - 1):
            current_rec = iter_records[idx]
            next_rec = iter_records[idx + 1]
            current_after = [tuple(x) for x in (current_rec.get("selected_branches_after", []) or [])]
            next_before = [tuple(x) for x in (next_rec.get("selected_branches_before", []) or [])]
            next_after = [tuple(x) for x in (next_rec.get("selected_branches_after", []) or [])]

            # Intent: state carry is exact only when the next step starts from the previous selected branches.
            carry_exact = int(current_after == next_before)
            carry_rows.append(
                {
                    "subset": subset,
                    "query_idx": int(query_idx),
                    "iter_t": int(current_rec.get("iter", idx)),
                    "carry_exact": float(carry_exact),
                    "update_jaccard_next_after": float(_jaccard_similarity(current_after, next_after)),
                    "explore_effective_next": float(bool(next_rec.get("explore_effective", False))),
                }
            )

    metric_df = pd.DataFrame(metric_rows)
    carry_df = pd.DataFrame(carry_rows)
    if metric_df.empty:
        raise RuntimeError(f"Round6 metrics are empty: {eval_samples_path}")
    return metric_df, carry_df, samples


def _build_query_window_rows(
    metrics_df: pd.DataFrame,
    method_name: str,
    window_start: int,
    window_end: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for (subset, query_idx), group in metrics_df.groupby(["subset", "query_idx"], dropna=False):
        group = group.sort_values("iter").reset_index(drop=True)
        early = group[(group["iter"] >= int(window_start)) & (group["iter"] <= int(window_end))]
        if early.empty:
            continue

        best_ndcg = float(early["ndcg10"].max())
        best_iter = int(early.loc[early["ndcg10"] == best_ndcg, "iter"].min())
        full_peak_ndcg = float(group["ndcg10"].max())
        full_peak_iter = int(group.loc[group["ndcg10"] == full_peak_ndcg, "iter"].min())
        end_iter = int(group["iter"].max())
        end_ndcg = float(group.loc[group["iter"] == end_iter, "ndcg10"].iloc[0])

        # Intent: time-to-max is the earliest iteration inside the early window that attains the query-level best nDCG.
        rows.append(
            {
                "subset": str(subset),
                "query_idx": int(query_idx),
                "query": str(group["query"].iloc[0]),
                f"{method_name}_best_iter": best_iter,
                f"{method_name}_best_ndcg": best_ndcg,
                f"{method_name}_early_mean_ndcg": float(early["ndcg10"].mean()),
                f"{method_name}_early_auc_ndcg": float(early["ndcg10"].sum()),
                f"{method_name}_peak_iter_all": full_peak_iter,
                f"{method_name}_peak_ndcg_all": full_peak_ndcg,
                f"{method_name}_end_iter": end_iter,
                f"{method_name}_end_ndcg": end_ndcg,
                f"{method_name}_peak_to_end_drop": float(full_peak_ndcg - end_ndcg),
            }
        )

    return pd.DataFrame(rows)


def _build_iter_curve_df(
    baseline_metrics_df: pd.DataFrame,
    round6_metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    baseline_curve = (
        baseline_metrics_df.groupby(["subset", "iter"], dropna=False)["ndcg10"]
        .mean()
        .reset_index(name="baseline_mean_ndcg10")
    )
    round6_curve = (
        round6_metrics_df.groupby(["subset", "iter"], dropna=False)["ndcg10"]
        .mean()
        .reset_index(name="ours_mean_ndcg10")
    )
    curve_df = baseline_curve.merge(round6_curve, on=["subset", "iter"], how="outer").sort_values(
        ["subset", "iter"]
    )
    curve_df["delta_ndcg10"] = curve_df["ours_mean_ndcg10"] - curve_df["baseline_mean_ndcg10"]
    curve_df["ours_ge_baseline"] = curve_df["delta_ndcg10"] >= 0.0
    curve_df["ours_gt_baseline"] = curve_df["delta_ndcg10"] > 0.0

    overall_curve = (
        curve_df.groupby("iter", dropna=False)[["baseline_mean_ndcg10", "ours_mean_ndcg10", "delta_ndcg10"]]
        .mean()
        .reset_index()
    )
    overall_curve["subset"] = "overall"
    overall_curve["ours_ge_baseline"] = overall_curve["delta_ndcg10"] >= 0.0
    overall_curve["ours_gt_baseline"] = overall_curve["delta_ndcg10"] > 0.0

    curve_df = pd.concat([curve_df, overall_curve], ignore_index=True, sort=False)
    return curve_df.sort_values(["subset", "iter"]).reset_index(drop=True)


def _build_state_carry_summary(carry_df: pd.DataFrame) -> pd.DataFrame:
    if carry_df.empty:
        return pd.DataFrame(
            columns=[
                "subset",
                "num_transitions",
                "state_carry_exact_rate",
                "state_update_jaccard_mean",
                "explore_effective_next_rate",
            ]
        )

    summary_df = (
        carry_df.groupby("subset", dropna=False)
        .agg(
            num_transitions=("query_idx", "count"),
            state_carry_exact_rate=("carry_exact", "mean"),
            state_update_jaccard_mean=("update_jaccard_next_after", "mean"),
            explore_effective_next_rate=("explore_effective_next", "mean"),
        )
        .reset_index()
    )
    overall_row = {
        "subset": "overall",
        "num_transitions": int(summary_df["num_transitions"].sum()),
        "state_carry_exact_rate": float(summary_df["state_carry_exact_rate"].mean()),
        "state_update_jaccard_mean": float(summary_df["state_update_jaccard_mean"].mean()),
        "explore_effective_next_rate": float(summary_df["explore_effective_next_rate"].mean()),
    }
    return pd.concat([summary_df, pd.DataFrame([overall_row])], ignore_index=True)


def _compute_baseline_gold_retention_control(
    doc_sequences: Dict[int, Dict[int, List[str]]],
    gold_by_query: Dict[int, List[str]],
    top_k: int,
) -> Dict[str, float]:
    hit_transitions = 0
    lost_transitions = 0
    query_ever_lost = 0

    for query_idx, per_iter in doc_sequences.items():
        gold_set = set(gold_by_query.get(query_idx, []))
        q_lost = False
        iter_keys = sorted(per_iter.keys())
        for iter_idx in iter_keys[:-1]:
            current_docs = list(per_iter[iter_idx])[: int(top_k)]
            next_docs = list(per_iter[iter_idx + 1])[: int(top_k)] if (iter_idx + 1) in per_iter else []
            has_gold = any(doc_id in gold_set for doc_id in current_docs)
            lost_gold = has_gold and not any(doc_id in gold_set for doc_id in next_docs)
            hit_transitions += int(has_gold)
            lost_transitions += int(lost_gold)
            q_lost = q_lost or lost_gold
        query_ever_lost += int(q_lost)

    num_queries = int(len(doc_sequences))
    return {
        "baseline_gold_loss_given_hit_pct_topk": (
            100.0 * float(lost_transitions / hit_transitions) if hit_transitions else float("nan")
        ),
        "baseline_query_ever_lost_gold_pct_topk": (
            100.0 * float(query_ever_lost / num_queries) if num_queries else float("nan")
        ),
    }


def _compute_round6_gold_retention_control(
    samples: Sequence[Dict[str, Any]],
    top_k: int,
) -> Dict[str, float]:
    hit_transitions = 0
    lost_transitions = 0
    query_ever_lost = 0

    for sample in samples:
        gold_set = set(str(x) for x in (sample.get("gold_doc_ids", []) or []))
        q_lost = False
        iter_records = sample.get("iter_records", []) or []
        for idx in range(len(iter_records) - 1):
            current_docs = [str(x) for x in (iter_records[idx].get("active_eval_doc_ids", []) or [])][: int(top_k)]
            next_docs = [str(x) for x in (iter_records[idx + 1].get("active_eval_doc_ids", []) or [])][: int(top_k)]
            has_gold = any(doc_id in gold_set for doc_id in current_docs)
            lost_gold = has_gold and not any(doc_id in gold_set for doc_id in next_docs)
            hit_transitions += int(has_gold)
            lost_transitions += int(lost_gold)
            q_lost = q_lost or lost_gold
        query_ever_lost += int(q_lost)

    num_queries = int(len(samples))
    return {
        "ours_gold_loss_given_hit_pct_topk": (
            100.0 * float(lost_transitions / hit_transitions) if hit_transitions else float("nan")
        ),
        "ours_query_ever_lost_gold_pct_topk": (
            100.0 * float(query_ever_lost / num_queries) if num_queries else float("nan")
        ),
    }


def _gold_by_query_from_round6_samples(samples: Sequence[Dict[str, Any]]) -> Dict[int, List[str]]:
    return {
        int(query_idx): [str(x) for x in (sample.get("gold_doc_ids", []) or [])]
        for query_idx, sample in enumerate(samples)
    }


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze whether round6 reaches strong retrieval states earlier than baseline3."
    )
    parser.add_argument(
        "--subsets",
        nargs="*",
        default=list(DEFAULT_SUBSETS),
        help="BRIGHT subsets to analyze. Default is the current 12-subset intersection.",
    )
    parser.add_argument(
        "--baseline_run_contains",
        nargs="*",
        default=list(DEFAULT_BASELINE_TOKENS),
        help="Tokens that must all appear in the baseline run path.",
    )
    parser.add_argument(
        "--round6_run_contains",
        nargs="*",
        default=list(DEFAULT_ROUND6_TOKENS),
        help="Tokens that must all appear in the round6 run path.",
    )
    parser.add_argument(
        "--window_start",
        type=int,
        default=0,
        help="Inclusive early-window start iteration for time-to-max.",
    )
    parser.add_argument(
        "--window_end",
        type=int,
        default=3,
        help="Inclusive early-window end iteration for time-to-max.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top-k cutoff for control gold-retention metrics.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/BRIGHT/analysis/round6_vs_baseline3_earlier_gain",
        help="Directory for CSV outputs.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = (repo_root / args.out_dir).resolve()

    baseline_metric_frames: List[pd.DataFrame] = []
    round6_metric_frames: List[pd.DataFrame] = []
    round6_carry_frames: List[pd.DataFrame] = []
    baseline_query_frames: List[pd.DataFrame] = []
    round6_query_frames: List[pd.DataFrame] = []
    control_rows: List[Dict[str, Any]] = []

    for subset in args.subsets:
        subset = str(subset)
        baseline_metrics_path = _select_matching_path(
            list((repo_root / "results" / "BRIGHT" / subset).glob("**/leaf_iter_metrics.jsonl")),
            required_tokens=args.baseline_run_contains,
            label="baseline metrics",
            subset=subset,
        )
        round6_eval_path = _select_matching_path(
            list((repo_root / "results" / "BRIGHT" / subset).glob("round6/**/all_eval_sample_dicts.pkl")),
            required_tokens=args.round6_run_contains,
            label="round6 eval samples",
            subset=subset,
        )

        baseline_metrics_df = _load_baseline_metrics(baseline_metrics_path, subset=subset)
        round6_metrics_df, round6_carry_df, round6_samples = _load_round6_metrics_and_carry(
            round6_eval_path,
            subset=subset,
        )

        baseline_metric_frames.append(baseline_metrics_df)
        round6_metric_frames.append(round6_metrics_df)
        if not round6_carry_df.empty:
            round6_carry_frames.append(round6_carry_df)

        baseline_query_frames.append(
            _build_query_window_rows(
                baseline_metrics_df,
                method_name="baseline",
                window_start=int(args.window_start),
                window_end=int(args.window_end),
            )
        )
        round6_query_frames.append(
            _build_query_window_rows(
                round6_metrics_df,
                method_name="ours",
                window_start=int(args.window_start),
                window_end=int(args.window_end),
            )
        )

        gold_by_query = _gold_by_query_from_round6_samples(round6_samples)
        baseline_doc_sequences = _load_baseline_doc_sequences(
            baseline_metrics_path.with_name("leaf_iter_records.jsonl"),
            subset=subset,
        )
        baseline_controls = _compute_baseline_gold_retention_control(
            baseline_doc_sequences,
            gold_by_query=gold_by_query,
            top_k=int(args.top_k),
        )
        round6_controls = _compute_round6_gold_retention_control(
            round6_samples,
            top_k=int(args.top_k),
        )
        control_rows.append(
            {
                "subset": subset,
                **baseline_controls,
                **round6_controls,
                "baseline_peak_to_end_drop_mean": float(
                    baseline_query_frames[-1]["baseline_peak_to_end_drop"].mean()
                ),
                "ours_peak_to_end_drop_mean": float(
                    round6_query_frames[-1]["ours_peak_to_end_drop"].mean()
                ),
            }
        )

    baseline_metrics_all = pd.concat(baseline_metric_frames, ignore_index=True)
    round6_metrics_all = pd.concat(round6_metric_frames, ignore_index=True)
    round6_carry_all = pd.concat(round6_carry_frames, ignore_index=True) if round6_carry_frames else pd.DataFrame()
    baseline_query_all = pd.concat(baseline_query_frames, ignore_index=True)
    round6_query_all = pd.concat(round6_query_frames, ignore_index=True)

    query_time_to_max_df = baseline_query_all.merge(
        round6_query_all,
        on=["subset", "query_idx"],
        how="inner",
        suffixes=("_baseline", "_ours"),
    )
    query_time_to_max_df["query"] = query_time_to_max_df["query_baseline"].where(
        query_time_to_max_df["query_baseline"].astype(bool),
        query_time_to_max_df["query_ours"],
    )
    query_time_to_max_df["delta_time_to_max"] = (
        query_time_to_max_df["ours_best_iter"] - query_time_to_max_df["baseline_best_iter"]
    )
    query_time_to_max_df["ours_earlier"] = query_time_to_max_df["delta_time_to_max"] < 0
    query_time_to_max_df["same_best_iter"] = query_time_to_max_df["delta_time_to_max"] == 0
    query_time_to_max_df["baseline_earlier"] = query_time_to_max_df["delta_time_to_max"] > 0
    query_time_to_max_df["delta_best_ndcg"] = (
        query_time_to_max_df["ours_best_ndcg"] - query_time_to_max_df["baseline_best_ndcg"]
    )
    query_time_to_max_df["delta_early_mean_ndcg"] = (
        query_time_to_max_df["ours_early_mean_ndcg"] - query_time_to_max_df["baseline_early_mean_ndcg"]
    )
    query_time_to_max_df["delta_early_auc_ndcg"] = (
        query_time_to_max_df["ours_early_auc_ndcg"] - query_time_to_max_df["baseline_early_auc_ndcg"]
    )
    query_time_to_max_df["delta_peak_to_end_drop"] = (
        query_time_to_max_df["ours_peak_to_end_drop"] - query_time_to_max_df["baseline_peak_to_end_drop"]
    )
    query_time_to_max_df = query_time_to_max_df[
        [
            "subset",
            "query_idx",
            "query",
            "baseline_best_iter",
            "ours_best_iter",
            "delta_time_to_max",
            "ours_earlier",
            "same_best_iter",
            "baseline_earlier",
            "baseline_best_ndcg",
            "ours_best_ndcg",
            "delta_best_ndcg",
            "baseline_early_mean_ndcg",
            "ours_early_mean_ndcg",
            "delta_early_mean_ndcg",
            "baseline_early_auc_ndcg",
            "ours_early_auc_ndcg",
            "delta_early_auc_ndcg",
            "baseline_peak_to_end_drop",
            "ours_peak_to_end_drop",
            "delta_peak_to_end_drop",
        ]
    ].sort_values(["subset", "query_idx"]).reset_index(drop=True)

    iter_curve_df = _build_iter_curve_df(baseline_metrics_all, round6_metrics_all)
    state_carry_summary_df = _build_state_carry_summary(round6_carry_all)
    state_carry_lookup = (
        state_carry_summary_df.set_index("subset").to_dict(orient="index")
        if not state_carry_summary_df.empty
        else {}
    )

    summary_rows: List[Dict[str, Any]] = []
    for subset, group in query_time_to_max_df.groupby("subset", dropna=False):
        subset_curve = iter_curve_df[iter_curve_df["subset"] == subset].copy()
        state_row = state_carry_lookup.get(str(subset), {})

        summary_rows.append(
            {
                "subset": str(subset),
                "num_queries": int(group.shape[0]),
                "baseline_time_to_max_mean": float(group["baseline_best_iter"].mean()),
                "ours_time_to_max_mean": float(group["ours_best_iter"].mean()),
                "delta_time_to_max_mean": float(group["delta_time_to_max"].mean()),
                "baseline_time_to_max_median": float(group["baseline_best_iter"].median()),
                "ours_time_to_max_median": float(group["ours_best_iter"].median()),
                "ours_earlier_query_rate": 100.0 * float(group["ours_earlier"].mean()),
                "same_best_iter_query_rate": 100.0 * float(group["same_best_iter"].mean()),
                "baseline_earlier_query_rate": 100.0 * float(group["baseline_earlier"].mean()),
                "baseline_best_ndcg_mean": float(group["baseline_best_ndcg"].mean()),
                "ours_best_ndcg_mean": float(group["ours_best_ndcg"].mean()),
                "baseline_early_mean_ndcg": float(group["baseline_early_mean_ndcg"].mean()),
                "ours_early_mean_ndcg": float(group["ours_early_mean_ndcg"].mean()),
                "delta_early_mean_ndcg": float(group["delta_early_mean_ndcg"].mean()),
                "baseline_early_auc_ndcg": float(group["baseline_early_auc_ndcg"].mean()),
                "ours_early_auc_ndcg": float(group["ours_early_auc_ndcg"].mean()),
                "delta_early_auc_ndcg": float(group["delta_early_auc_ndcg"].mean()),
                # Intent: detect the first subset-level iteration where round6 catches up to or exceeds baseline.
                "first_iter_ours_ge_baseline": _first_true_iter(subset_curve, "ours_ge_baseline"),
                "first_iter_ours_strictly_better": _first_true_iter(subset_curve, "ours_gt_baseline"),
                "first_iter_ours_ge_baseline_sustained": _first_sustained_true_iter(
                    subset_curve,
                    "ours_ge_baseline",
                ),
                "state_carry_exact_rate": float(state_row.get("state_carry_exact_rate", float("nan"))),
                "state_update_jaccard_mean": float(state_row.get("state_update_jaccard_mean", float("nan"))),
                "explore_effective_next_rate": float(state_row.get("explore_effective_next_rate", float("nan"))),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("subset").reset_index(drop=True)
    overall_curve = iter_curve_df[iter_curve_df["subset"] == "overall"].copy()
    overall_state_row = state_carry_lookup.get("overall", {})
    overall_summary_row = {
        "subset": "overall",
        "num_queries": int(summary_df["num_queries"].sum()),
        "baseline_time_to_max_mean": float(summary_df["baseline_time_to_max_mean"].mean()),
        "ours_time_to_max_mean": float(summary_df["ours_time_to_max_mean"].mean()),
        "delta_time_to_max_mean": float(summary_df["delta_time_to_max_mean"].mean()),
        "baseline_time_to_max_median": float(summary_df["baseline_time_to_max_median"].mean()),
        "ours_time_to_max_median": float(summary_df["ours_time_to_max_median"].mean()),
        "ours_earlier_query_rate": float(summary_df["ours_earlier_query_rate"].mean()),
        "same_best_iter_query_rate": float(summary_df["same_best_iter_query_rate"].mean()),
        "baseline_earlier_query_rate": float(summary_df["baseline_earlier_query_rate"].mean()),
        "baseline_best_ndcg_mean": float(summary_df["baseline_best_ndcg_mean"].mean()),
        "ours_best_ndcg_mean": float(summary_df["ours_best_ndcg_mean"].mean()),
        "baseline_early_mean_ndcg": float(summary_df["baseline_early_mean_ndcg"].mean()),
        "ours_early_mean_ndcg": float(summary_df["ours_early_mean_ndcg"].mean()),
        "delta_early_mean_ndcg": float(summary_df["delta_early_mean_ndcg"].mean()),
        "baseline_early_auc_ndcg": float(summary_df["baseline_early_auc_ndcg"].mean()),
        "ours_early_auc_ndcg": float(summary_df["ours_early_auc_ndcg"].mean()),
        "delta_early_auc_ndcg": float(summary_df["delta_early_auc_ndcg"].mean()),
        "first_iter_ours_ge_baseline": _first_true_iter(overall_curve, "ours_ge_baseline"),
        "first_iter_ours_strictly_better": _first_true_iter(overall_curve, "ours_gt_baseline"),
        "first_iter_ours_ge_baseline_sustained": _first_sustained_true_iter(
            overall_curve,
            "ours_ge_baseline",
        ),
        "state_carry_exact_rate": float(overall_state_row.get("state_carry_exact_rate", float("nan"))),
        "state_update_jaccard_mean": float(overall_state_row.get("state_update_jaccard_mean", float("nan"))),
        "explore_effective_next_rate": float(overall_state_row.get("explore_effective_next_rate", float("nan"))),
    }
    summary_df = pd.concat([summary_df, pd.DataFrame([overall_summary_row])], ignore_index=True)

    controls_df = pd.DataFrame(control_rows).sort_values("subset").reset_index(drop=True)
    overall_controls_row = {
        "subset": "overall",
        "baseline_gold_loss_given_hit_pct_topk": float(controls_df["baseline_gold_loss_given_hit_pct_topk"].mean()),
        "baseline_query_ever_lost_gold_pct_topk": float(controls_df["baseline_query_ever_lost_gold_pct_topk"].mean()),
        "ours_gold_loss_given_hit_pct_topk": float(controls_df["ours_gold_loss_given_hit_pct_topk"].mean()),
        "ours_query_ever_lost_gold_pct_topk": float(controls_df["ours_query_ever_lost_gold_pct_topk"].mean()),
        "baseline_peak_to_end_drop_mean": float(controls_df["baseline_peak_to_end_drop_mean"].mean()),
        "ours_peak_to_end_drop_mean": float(controls_df["ours_peak_to_end_drop_mean"].mean()),
    }
    controls_df = pd.concat([controls_df, pd.DataFrame([overall_controls_row])], ignore_index=True)

    _save_csv(summary_df, out_dir / "summary_earlier_gain.csv")
    _save_csv(iter_curve_df, out_dir / "iter_curve_ndcg.csv")
    _save_csv(query_time_to_max_df, out_dir / "query_time_to_max_iter0_3.csv")
    _save_csv(state_carry_summary_df, out_dir / "round6_state_carry.csv")
    _save_csv(controls_df, out_dir / "controls_not_for_headline.csv")

    overall_summary = summary_df[summary_df["subset"] == "overall"].iloc[0]
    print("[Earlier Gain] Summary")
    print(f"window = iter {int(args.window_start)}..{int(args.window_end)}")
    print(f"mean delta_time_to_max = {float(overall_summary['delta_time_to_max_mean']):.4f}")
    print(f"mean delta_early_mean_ndcg = {float(overall_summary['delta_early_mean_ndcg']):.4f}")
    print(
        "first_iter_ours_ge_baseline = "
        f"{overall_summary['first_iter_ours_ge_baseline']}"
    )
    print(
        "first_iter_ours_strictly_better = "
        f"{overall_summary['first_iter_ours_strictly_better']}"
    )
    print(
        "first_iter_ours_ge_baseline_sustained = "
        f"{overall_summary['first_iter_ours_ge_baseline_sustained']}"
    )
    print(
        "round6 state_carry_exact_rate = "
        f"{float(overall_summary['state_carry_exact_rate']):.4f}"
    )
    print(
        "round6 state_update_jaccard_mean = "
        f"{float(overall_summary['state_update_jaccard_mean']):.4f}"
    )
    print(
        "gold_loss_given_hit@topk baseline vs ours = "
        f"{float(overall_controls_row['baseline_gold_loss_given_hit_pct_topk']):.4f} vs "
        f"{float(overall_controls_row['ours_gold_loss_given_hit_pct_topk']):.4f}"
    )
    print(f"\nSaved CSVs under: {out_dir}")


if __name__ == "__main__":
    main()
