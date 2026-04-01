#!/usr/bin/env python3
"""Analyze whether round6 reseat acts as an explicit escape operator versus baseline3.

This script compares EMR-only runs on two aligned questions:

1. Mechanism: do next-step feedback docs escape the previous local branch-like neighborhood?
2. Utility: when that escape happens, does R@10 / nDCG@10 improve?

The comparison is intentionally asymmetric but mechanism-aligned:
- baseline has no real branch state, so we proxy local neighborhood from previous rewrite context paths
- round6 has real branch state, so we use selected branches before reseat and next-step pre-hit feedback
"""

from __future__ import annotations

import argparse
import json
import math
import pickle as pkl
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


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

DEFAULT_BASELINE_TOKENS: List[str] = [
    "baseline3_leaf_only_loop_emr",
    "agent_executor_v1_icl2_emr_memory",
    "LEmrMM=accumulated",
    "RSC=on",
]

DEFAULT_QSTATE_TOKENS: List[str] = [
    "round6_mrr_selector_accum_meanscore_global_expandable_ended_reseat_emr",
    "agent_executor_v1_icl2_emr_memory",
    "frontiercum_qstate_v1",
]

DEFAULT_QSTATE_FORBID: List[str] = [
    "depthbatch",
    "random",
    "beampack",
]

DEFAULT_DEPTHBATCH_TOKENS: List[str] = [
    "round6_mrr_selector_accum_meanscore_global_expandable_ended_reseat_emr_depthbatch",
    "agent_executor_v1_icl2_emr_memory",
    "reseat_depth_batch_v1",
]


def _is_prefix(prefix: Sequence[int], full: Sequence[int]) -> bool:
    prefix_t = tuple(prefix)
    full_t = tuple(full)
    return len(prefix_t) <= len(full_t) and full_t[: len(prefix_t)] == prefix_t


def _infer_subset_from_path(path: Path) -> str:
    parts = list(path.parts)
    if "BRIGHT" in parts:
        idx = parts.index("BRIGHT")
        if idx + 1 < len(parts):
            return str(parts[idx + 1])
    return "unknown"


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _normalize_paths(paths: Sequence[Any]) -> List[Tuple[int, ...]]:
    out: List[Tuple[int, ...]] = []
    for path in paths or []:
        if not path:
            continue
        out.append(tuple(int(x) for x in list(path)))
    return out


def _normalize_path_rows(path_rows: Sequence[Any]) -> List[Tuple[int, ...]]:
    out: List[Tuple[int, ...]] = []
    for row in path_rows or []:
        if isinstance(row, dict):
            path = row.get("path", []) or []
        else:
            path = row or []
        if not path:
            continue
        out.append(tuple(int(x) for x in list(path)))
    return out


def _json_dumps_paths(paths: Sequence[Tuple[int, ...]]) -> str:
    return json.dumps([list(path) for path in paths], ensure_ascii=False)


def _context_prefix_branches(context_paths: Sequence[Tuple[int, ...]], depth: int) -> List[Tuple[int, ...]]:
    branches: List[Tuple[int, ...]] = []
    seen = set()
    for path in context_paths:
        if len(path) < 2:
            continue
        # Intent: baseline rewrite_context paths are leaf-like paths, so the proxy should stop before the terminal leaf.
        d = min(int(depth), len(path) - 1)
        if d <= 0:
            continue
        branch = tuple(path[:d])
        if branch in seen:
            continue
        seen.add(branch)
        branches.append(branch)
    return branches


def _selected_prefix_branches(selected_paths: Sequence[Tuple[int, ...]], depth: int) -> List[Tuple[int, ...]]:
    branches: List[Tuple[int, ...]] = []
    seen = set()
    for path in selected_paths:
        if not path:
            continue
        # Intent: round6 selected branches are already internal branch states, so keep the proxy on the real branch path itself.
        d = min(int(depth), len(path))
        if d <= 0:
            continue
        branch = tuple(path[:d])
        if branch in seen:
            continue
        seen.add(branch)
        branches.append(branch)
    return branches


def _outside_proxy_metrics(
    feedback_paths: Sequence[Tuple[int, ...]],
    proxy_branches: Sequence[Tuple[int, ...]],
    top_k: int,
) -> Tuple[float, float, int, int]:
    top_paths = [tuple(path) for path in (feedback_paths or [])[: int(top_k)] if path]
    if (not top_paths) or (not proxy_branches):
        return float("nan"), float("nan"), 0, 0
    outside = 0
    for path in top_paths:
        if not any(_is_prefix(branch, path) for branch in proxy_branches):
            outside += 1
    pct = 100.0 * float(outside) / float(len(top_paths))
    return (1.0 if outside > 0 else 0.0), pct, outside, len(top_paths)


def _reseat_uptake_metrics(
    feedback_paths: Sequence[Tuple[int, ...]],
    reseat_paths: Sequence[Tuple[int, ...]],
    top_k: int,
) -> Tuple[float, float, int, int]:
    top_paths = [tuple(path) for path in (feedback_paths or [])[: int(top_k)] if path]
    if (not top_paths) or (not reseat_paths):
        return float("nan"), float("nan"), 0, 0
    uptake = 0
    for path in top_paths:
        if any(_is_prefix(reseat_path, path) for reseat_path in reseat_paths):
            uptake += 1
    pct = 100.0 * float(uptake) / float(len(top_paths))
    return (1.0 if uptake > 0 else 0.0), pct, uptake, len(top_paths)


def _select_latest_match(
    candidates: Sequence[Path],
    required_tokens: Sequence[str],
    forbidden_tokens: Optional[Sequence[str]],
    label: str,
    subset: str,
) -> Path:
    matches = []
    for path in candidates:
        path_str = str(path)
        if any(token not in path_str for token in required_tokens):
            continue
        if forbidden_tokens and any(token in path_str for token in forbidden_tokens):
            continue
        matches.append(path)
    if not matches:
        raise FileNotFoundError(
            f"No {label} match for subset={subset}. required={list(required_tokens)} forbidden={list(forbidden_tokens or [])}"
        )
    if len(matches) == 1:
        return matches[0]
    # Intent: repeated reruns can leave multiple matching outputs; prefer the latest artifact for reproducibility.
    latest = max(matches, key=lambda path: path.stat().st_mtime)
    print(f"[warn] multiple {label} matches for subset={subset}; using latest: {latest}")
    return latest


@lru_cache(maxsize=None)
def _resolve_baseline_records_cached(root_str: str, subset: str) -> str:
    root = Path(root_str)
    subset_dir = root / subset
    candidates = list(subset_dir.rglob("leaf_iter_records.jsonl"))
    return str(_select_latest_match(candidates, DEFAULT_BASELINE_TOKENS, None, "baseline records", subset))


def _resolve_baseline_records(root: Path, subset: str) -> Path:
    return Path(_resolve_baseline_records_cached(str(root), subset))


@lru_cache(maxsize=None)
def _resolve_round6_eval_samples_cached(root_str: str, subset: str, depthbatch: bool) -> str:
    root = Path(root_str)
    subset_dir = root / subset
    candidates = list(subset_dir.rglob("all_eval_sample_dicts.pkl"))
    if depthbatch:
        return str(_select_latest_match(candidates, DEFAULT_DEPTHBATCH_TOKENS, None, "round6 depthbatch", subset))
    return str(_select_latest_match(candidates, DEFAULT_QSTATE_TOKENS, DEFAULT_QSTATE_FORBID, "round6 qstate", subset))


def _resolve_round6_eval_samples(root: Path, subset: str, *, depthbatch: bool) -> Path:
    return Path(_resolve_round6_eval_samples_cached(str(root), subset, depthbatch))


@lru_cache(maxsize=None)
def _load_baseline_records_cached(records_path_str: str) -> Dict[int, Dict[int, Dict[str, Any]]]:
    records_path = Path(records_path_str)
    by_q: Dict[int, Dict[int, Dict[str, Any]]] = {}
    with records_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qidx = int(rec.get("query_idx", -1))
            if qidx < 0:
                continue
            phase = str(rec.get("phase", ""))
            by_q.setdefault(qidx, {})
            if phase == "initial_rewrite":
                by_q[qidx][-1] = rec
            elif phase == "iter_retrieval":
                iter_idx = int(rec.get("iter", -1))
                if iter_idx >= 0:
                    by_q[qidx][iter_idx] = rec
    return by_q


def _load_baseline_records(records_path: Path) -> Dict[int, Dict[int, Dict[str, Any]]]:
    return _load_baseline_records_cached(str(records_path))


@lru_cache(maxsize=None)
def _load_baseline_metrics_cached(metrics_path_str: str) -> Dict[int, Dict[int, Dict[str, float]]]:
    metrics_path = Path(metrics_path_str)
    by_q: Dict[int, Dict[int, Dict[str, float]]] = {}
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qidx = int(rec.get("query_idx", -1))
            iter_idx = int(rec.get("iter", -1))
            if qidx < 0 or iter_idx < 0:
                continue
            by_q.setdefault(qidx, {})[iter_idx] = {
                "nDCG@10": _safe_float(rec.get("nDCG@10")),
                "Recall@10": _safe_float(rec.get("Recall@10")),
            }
    return by_q


def _load_baseline_metrics(metrics_path: Path) -> Dict[int, Dict[int, Dict[str, float]]]:
    return _load_baseline_metrics_cached(str(metrics_path))


def _build_baseline_rows(root: Path, subset: str, proxy_depth: int, top_k: int) -> List[Dict[str, Any]]:
    records_path = _resolve_baseline_records(root, subset)
    metrics_path = Path(str(records_path).replace("leaf_iter_records.jsonl", "leaf_iter_metrics.jsonl"))
    record_by_q = _load_baseline_records(records_path)
    metric_by_q = _load_baseline_metrics(metrics_path)

    rows: List[Dict[str, Any]] = []
    for qidx, by_iter in record_by_q.items():
        iter_keys = sorted(k for k in by_iter.keys() if k >= 0)
        metric_keys = sorted(metric_by_q.get(qidx, {}).keys())
        if not metric_keys:
            continue
        end_iter = max(metric_keys)
        end_ndcg = metric_by_q[qidx][end_iter]["nDCG@10"]
        end_recall10 = metric_by_q[qidx][end_iter]["Recall@10"]

        for iter_idx in iter_keys:
            prev = by_iter.get(iter_idx - 1)
            cur = by_iter.get(iter_idx)
            cur_metrics = metric_by_q.get(qidx, {}).get(iter_idx)
            next_metrics = metric_by_q.get(qidx, {}).get(iter_idx + 1)
            if prev is None or cur is None or cur_metrics is None:
                continue

            prev_context_paths = _normalize_paths(prev.get("rewrite_context_paths", []) or [])
            proxy_branches = _context_prefix_branches(prev_context_paths, depth=int(proxy_depth))
            feedback_paths = _normalize_paths(cur.get("retrieved_paths", []) or [])
            escape_any, escape_pct, escape_count, feedback_count = _outside_proxy_metrics(
                feedback_paths,
                proxy_branches,
                top_k=int(top_k),
            )

            ndcg_t = cur_metrics["nDCG@10"]
            recall10_t = cur_metrics["Recall@10"]
            ndcg_t1 = next_metrics["nDCG@10"] if next_metrics else float("nan")
            recall10_t1 = next_metrics["Recall@10"] if next_metrics else float("nan")

            rows.append(
                {
                    "system": "baseline3_leaf_only_loop_emr",
                    "subset": subset,
                    "query_idx": int(qidx),
                    "iter": int(iter_idx),
                    "event_scope": "baseline_iter",
                    "feedback_source": "retrieved_paths_t",
                    "proxy_depth": int(proxy_depth),
                    "feedback_topk": int(top_k),
                    "feedback_count": int(feedback_count),
                    "proxy_branch_count": int(len(proxy_branches)),
                    "proxy_branches_json": _json_dumps_paths(proxy_branches),
                    "feedback_paths_json": _json_dumps_paths(feedback_paths[: int(top_k)]),
                    "escape_any_feedback10_proxy": float(escape_any),
                    "escape_pct_feedback10_proxy": float(escape_pct),
                    "escape_count_feedback10_proxy": int(escape_count),
                    "reseat_uptake_any_feedback10": float("nan"),
                    "reseat_uptake_pct_feedback10": float("nan"),
                    "reseat_uptake_count_feedback10": int(0),
                    "reseat_path_count": int(0),
                    "ndcg10_t": float(ndcg_t),
                    "ndcg10_t1": float(ndcg_t1),
                    "ndcg10_end": float(end_ndcg),
                    "recall10_t": float(recall10_t),
                    "recall10_t1": float(recall10_t1),
                    "recall10_end": float(end_recall10),
                    "delta_ndcg10_t1": float(ndcg_t1 - ndcg_t) if not math.isnan(ndcg_t1) else float("nan"),
                    "delta_recall10_t1": float(recall10_t1 - recall10_t) if not math.isnan(recall10_t1) else float("nan"),
                    "delta_ndcg10_end": float(end_ndcg - ndcg_t),
                    "delta_recall10_end": float(end_recall10 - recall10_t),
                    "records_path": str(records_path),
                }
            )
    return rows


@lru_cache(maxsize=None)
def _load_round6_samples_cached(eval_samples_path_str: str) -> List[Dict[str, Any]]:
    eval_samples_path = Path(eval_samples_path_str)
    with eval_samples_path.open("rb") as f:
        samples = pkl.load(f)
    if not isinstance(samples, list):
        raise TypeError(f"Unexpected payload type in {eval_samples_path}: {type(samples)}")
    return samples


def _load_round6_samples(eval_samples_path: Path) -> List[Dict[str, Any]]:
    return _load_round6_samples_cached(str(eval_samples_path))


def _build_round6_rows(root: Path, subset: str, proxy_depth: int, top_k: int, *, depthbatch: bool) -> List[Dict[str, Any]]:
    eval_samples_path = _resolve_round6_eval_samples(root, subset, depthbatch=depthbatch)
    samples = _load_round6_samples(eval_samples_path)
    system_name = "reseat_depth_batch_v1_emr" if depthbatch else "frontiercum_qstate_v1_emr"

    rows: List[Dict[str, Any]] = []
    for qidx, sample in enumerate(samples):
        iter_records = sample.get("iter_records", []) or []
        if not iter_records:
            continue
        end_metrics = iter_records[-1].get("metrics", {}) or {}
        end_ndcg = _safe_float(end_metrics.get("nDCG@10"))
        end_recall10 = _safe_float(end_metrics.get("Recall@10"))

        for iter_idx in range(len(iter_records) - 1):
            rec = iter_records[iter_idx]
            next_rec = iter_records[iter_idx + 1]
            reseat_paths = _normalize_path_rows(rec.get("ended_beam_reseat_selected_paths", []) or [])
            if not reseat_paths:
                continue

            selected_before = _normalize_paths(rec.get("selected_branches_before", []) or [])
            proxy_branches = _selected_prefix_branches(selected_before, depth=int(proxy_depth))
            feedback_paths = _normalize_paths(next_rec.get("pre_hit_paths", []) or [])

            escape_any, escape_pct, escape_count, feedback_count = _outside_proxy_metrics(
                feedback_paths,
                proxy_branches,
                top_k=int(top_k),
            )
            uptake_any, uptake_pct, uptake_count, _ = _reseat_uptake_metrics(
                feedback_paths,
                reseat_paths,
                top_k=int(top_k),
            )

            cur_metrics = rec.get("metrics", {}) or {}
            next_metrics = next_rec.get("metrics", {}) or {}
            ndcg_t = _safe_float(cur_metrics.get("nDCG@10"))
            ndcg_t1 = _safe_float(next_metrics.get("nDCG@10"))
            recall10_t = _safe_float(cur_metrics.get("Recall@10"))
            recall10_t1 = _safe_float(next_metrics.get("Recall@10"))

            rows.append(
                {
                    "system": system_name,
                    "subset": subset,
                    "query_idx": int(qidx),
                    "iter": int(iter_idx),
                    "event_scope": "reseat_active_iter",
                    # Intent: use next-step pre-hit feedback because this is the first feedback surface after the reseat transition.
                    "feedback_source": "pre_hit_paths_t1",
                    "proxy_depth": int(proxy_depth),
                    "feedback_topk": int(top_k),
                    "feedback_count": int(feedback_count),
                    "proxy_branch_count": int(len(proxy_branches)),
                    "proxy_branches_json": _json_dumps_paths(proxy_branches),
                    "feedback_paths_json": _json_dumps_paths(feedback_paths[: int(top_k)]),
                    "escape_any_feedback10_proxy": float(escape_any),
                    "escape_pct_feedback10_proxy": float(escape_pct),
                    "escape_count_feedback10_proxy": int(escape_count),
                    "reseat_uptake_any_feedback10": float(uptake_any),
                    "reseat_uptake_pct_feedback10": float(uptake_pct),
                    "reseat_uptake_count_feedback10": int(uptake_count),
                    "reseat_path_count": int(len(reseat_paths)),
                    "reseat_paths_json": _json_dumps_paths(reseat_paths),
                    "ndcg10_t": float(ndcg_t),
                    "ndcg10_t1": float(ndcg_t1),
                    "ndcg10_end": float(end_ndcg),
                    "recall10_t": float(recall10_t),
                    "recall10_t1": float(recall10_t1),
                    "recall10_end": float(end_recall10),
                    "delta_ndcg10_t1": float(ndcg_t1 - ndcg_t),
                    "delta_recall10_t1": float(recall10_t1 - recall10_t),
                    "delta_ndcg10_end": float(end_ndcg - ndcg_t),
                    "delta_recall10_end": float(end_recall10 - recall10_t),
                    "eval_samples_path": str(eval_samples_path),
                }
            )
    return rows


def _aggregate_base(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for keys, grp in df.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: Dict[str, Any] = {col: val for col, val in zip(group_cols, keys)}
        row.update(
            {
                "num_events": int(len(grp)),
                "num_queries": int(grp["query_idx"].nunique()),
                "escape_any_rate": float(grp["escape_any_feedback10_proxy"].mean()),
                "escape_pct_mean": float(grp["escape_pct_feedback10_proxy"].mean()),
                "delta_ndcg10_t1_mean": float(grp["delta_ndcg10_t1"].mean()),
                "delta_recall10_t1_mean": float(grp["delta_recall10_t1"].mean()),
                "delta_ndcg10_end_mean": float(grp["delta_ndcg10_end"].mean()),
                "delta_recall10_end_mean": float(grp["delta_recall10_end"].mean()),
            }
        )
        if grp["reseat_uptake_any_feedback10"].notna().any():
            uptake_grp = grp[grp["reseat_uptake_any_feedback10"].notna()]
            row["reseat_uptake_any_rate"] = float(uptake_grp["reseat_uptake_any_feedback10"].mean())
            row["reseat_uptake_pct_mean"] = float(uptake_grp["reseat_uptake_pct_feedback10"].mean())
        else:
            row["reseat_uptake_any_rate"] = float("nan")
            row["reseat_uptake_pct_mean"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _build_utility_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    baseline = df[df["system"] == "baseline3_leaf_only_loop_emr"]
    for flag_value, grp in baseline.groupby("escape_any_feedback10_proxy", dropna=False):
        rows.append(
            {
                "system": "baseline3_leaf_only_loop_emr",
                "group_type": "escape_any_feedback10_proxy",
                "group_value": flag_value,
                "num_events": int(len(grp)),
                "delta_ndcg10_t1_mean": float(grp["delta_ndcg10_t1"].mean()),
                "delta_recall10_t1_mean": float(grp["delta_recall10_t1"].mean()),
                "delta_ndcg10_end_mean": float(grp["delta_ndcg10_end"].mean()),
                "delta_recall10_end_mean": float(grp["delta_recall10_end"].mean()),
            }
        )

    ours = df[df["system"].isin(["frontiercum_qstate_v1_emr", "reseat_depth_batch_v1_emr"])]
    for system, sys_grp in ours.groupby("system", dropna=False):
        for group_type in ["escape_any_feedback10_proxy", "reseat_uptake_any_feedback10"]:
            for flag_value, grp in sys_grp.groupby(group_type, dropna=False):
                rows.append(
                    {
                        "system": system,
                        "group_type": group_type,
                        "group_value": flag_value,
                        "num_events": int(len(grp)),
                        "delta_ndcg10_t1_mean": float(grp["delta_ndcg10_t1"].mean()),
                        "delta_recall10_t1_mean": float(grp["delta_recall10_t1"].mean()),
                        "delta_ndcg10_end_mean": float(grp["delta_ndcg10_end"].mean()),
                        "delta_recall10_end_mean": float(grp["delta_recall10_end"].mean()),
                    }
                )

    return pd.DataFrame(rows)


def _build_proxy_depth_sweep(root: Path, subsets: Sequence[str], top_k: int, depths: Sequence[int]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for proxy_depth in depths:
        query_rows = []
        for subset in subsets:
            query_rows.extend(_build_baseline_rows(root, subset, proxy_depth=proxy_depth, top_k=top_k))
            query_rows.extend(_build_round6_rows(root, subset, proxy_depth=proxy_depth, top_k=top_k, depthbatch=False))
            query_rows.extend(_build_round6_rows(root, subset, proxy_depth=proxy_depth, top_k=top_k, depthbatch=True))
        df = pd.DataFrame(query_rows)
        summary_df = _aggregate_base(df, ["system"])
        summary_df.insert(0, "proxy_depth", int(proxy_depth))
        rows.extend(summary_df.to_dict(orient="records"))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze explicit escape-operator evidence for round6 vs baseline3.")
    parser.add_argument("--results_root", default="results/BRIGHT", help="Root directory containing subset run outputs.")
    parser.add_argument(
        "--subsets",
        nargs="*",
        default=DEFAULT_SUBSETS,
        help="Subset names to include. Defaults exclude aops, leetcode, theoremqa_questions.",
    )
    parser.add_argument("--proxy_depth", type=int, default=3, help="Prefix depth for local branch-like neighborhood proxy.")
    parser.add_argument("--top_k", type=int, default=10, help="Feedback top-k used for mechanism metrics.")
    parser.add_argument(
        "--out_prefix",
        default="results/BRIGHT/analysis/round6_vs_baseline3_escape_operator",
        help="Output prefix for CSVs.",
    )
    parser.add_argument(
        "--proxy_depth_sweep",
        nargs="*",
        type=int,
        default=[2, 3, 4],
        help="Proxy depths to evaluate for robustness sweep.",
    )
    args = parser.parse_args()

    root = Path(args.results_root)
    subsets = [str(x) for x in args.subsets]

    query_rows: List[Dict[str, Any]] = []
    for subset in subsets:
        query_rows.extend(_build_baseline_rows(root, subset, proxy_depth=int(args.proxy_depth), top_k=int(args.top_k)))
        query_rows.extend(_build_round6_rows(root, subset, proxy_depth=int(args.proxy_depth), top_k=int(args.top_k), depthbatch=False))
        query_rows.extend(_build_round6_rows(root, subset, proxy_depth=int(args.proxy_depth), top_k=int(args.top_k), depthbatch=True))

    query_df = pd.DataFrame(query_rows)
    if query_df.empty:
        raise RuntimeError("No query rows produced.")

    system_summary_df = _aggregate_base(query_df, ["system"])
    subset_summary_df = _aggregate_base(query_df, ["system", "subset"])
    utility_summary_df = _build_utility_summary(query_df)
    sweep_df = _build_proxy_depth_sweep(root, subsets, top_k=int(args.top_k), depths=args.proxy_depth_sweep)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    query_df.to_csv(f"{out_prefix}_query_rows.csv", index=False)
    system_summary_df.to_csv(f"{out_prefix}_system_summary.csv", index=False)
    subset_summary_df.to_csv(f"{out_prefix}_subset_summary.csv", index=False)
    utility_summary_df.to_csv(f"{out_prefix}_utility_summary.csv", index=False)
    sweep_df.to_csv(f"{out_prefix}_proxy_depth_sweep.csv", index=False)

    print(f"Saved query rows to {out_prefix}_query_rows.csv")
    print(f"Saved system summary to {out_prefix}_system_summary.csv")
    print(f"Saved subset summary to {out_prefix}_subset_summary.csv")
    print(f"Saved utility summary to {out_prefix}_utility_summary.csv")
    print(f"Saved proxy depth sweep to {out_prefix}_proxy_depth_sweep.csv")


if __name__ == "__main__":
    main()
