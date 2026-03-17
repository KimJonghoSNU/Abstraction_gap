#!/usr/bin/env python3
"""
Analyze how often round6 method1 global escape actually replaces local beam slots.

Examples
--------
python scripts/analyze_round6_global_escape.py \
    --input results/BRIGHT/biology/round6/.../all_eval_sample_dicts.pkl

python scripts/analyze_round6_global_escape.py \
    --glob_pattern 'results/BRIGHT/*/round6/**/all_eval_sample_dicts.pkl' \
    --out_prefix results/BRIGHT/round6_global_escape
"""

import argparse
import glob
import os
import pickle as pkl
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _resolve_eval_paths(inputs: List[str], glob_pattern: str) -> List[str]:
    resolved: List[str] = []

    for path in inputs:
        if not path:
            continue
        path = os.path.abspath(path)
        if os.path.isdir(path):
            candidate = os.path.join(path, "all_eval_sample_dicts.pkl")
            if not os.path.exists(candidate):
                raise FileNotFoundError(f"Directory does not contain all_eval_sample_dicts.pkl: {path}")
            resolved.append(candidate)
            continue
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input path not found: {path}")
        resolved.append(path)

    if glob_pattern:
        resolved.extend(glob.glob(glob_pattern, recursive=True))

    resolved = sorted({os.path.abspath(path) for path in resolved})
    if not resolved:
        raise FileNotFoundError("No all_eval_sample_dicts.pkl files found.")
    return resolved


def _infer_subset_from_path(path: str) -> str:
    parts = os.path.abspath(path).split(os.sep)
    for idx, part in enumerate(parts):
        if part == "results" and (idx + 3) < len(parts):
            return str(parts[idx + 3])
    return "unknown"


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _extract_sample_iter_rows(samples_path: str) -> pd.DataFrame:
    with open(samples_path, "rb") as f:
        samples = pkl.load(f)

    run_dir = os.path.dirname(os.path.abspath(samples_path))
    subset = _infer_subset_from_path(samples_path)
    rows: List[Dict[str, Any]] = []

    for sample_idx, sample in enumerate(samples):
        iter_records = sample.get("iter_records", []) or []
        for iter_idx, rec in enumerate(iter_records):
            selected_rows = rec.get("global_escape_selected_paths", []) or []
            replaced_rows = rec.get("global_escape_replaced_local_paths", []) or []
            metrics = rec.get("metrics", {}) or {}

            # Intent: treat actual replacement as the primary event, not just "escape was enabled".
            num_selected = int(len(selected_rows))
            num_replaced = int(len(replaced_rows))
            replaced = bool(num_replaced > 0)
            enabled = bool(rec.get("global_escape_enabled", False))

            rows.append(
                {
                    "samples_path": os.path.abspath(samples_path),
                    "run_dir": run_dir,
                    "subset": subset,
                    "sample_idx": int(sample_idx),
                    "iter": int(rec.get("iter", iter_idx)),
                    "global_escape_enabled": enabled,
                    "global_escape_slots": int(rec.get("global_escape_slots", 0) or 0),
                    "global_escape_pick_reason": str(rec.get("global_escape_pick_reason", "") or ""),
                    "selector_pick_reason": str(rec.get("selector_pick_reason", "") or ""),
                    "num_escape_selected": num_selected,
                    "num_escape_replaced": num_replaced,
                    "escape_replaced": replaced,
                    "local_selector_top_count": int(len(rec.get("local_selector_scored_top", []) or [])),
                    "global_selector_top_count": int(len(rec.get("global_escape_scored_top", []) or [])),
                    "selected_branch_count": int(len(rec.get("selected_branches_after", []) or [])),
                    "ndcg10": _safe_float(metrics.get("nDCG@10")),
                    "branchhit_b": _safe_float(metrics.get("BranchHit@B")),
                }
            )

    return pd.DataFrame(rows)


def _summarize_per_iter(records_df: pd.DataFrame) -> pd.DataFrame:
    if records_df.empty:
        return pd.DataFrame()

    summary = (
        records_df.groupby(["samples_path", "subset", "iter"], dropna=False)
        .agg(
            num_sample_iters=("sample_idx", "count"),
            escape_enabled_rate=("global_escape_enabled", "mean"),
            replacement_rate=("escape_replaced", "mean"),
            avg_replaced_slots=("num_escape_replaced", "mean"),
            avg_selected_slots=("num_escape_selected", "mean"),
            avg_ndcg10=("ndcg10", "mean"),
            avg_branchhit_b=("branchhit_b", "mean"),
        )
        .reset_index()
    )
    for col in ("escape_enabled_rate", "replacement_rate"):
        summary[col] = 100.0 * summary[col]
    return summary.sort_values(["subset", "samples_path", "iter"]).reset_index(drop=True)


def _summarize_per_run(records_df: pd.DataFrame) -> pd.DataFrame:
    if records_df.empty:
        return pd.DataFrame()

    enabled_mask = records_df["global_escape_enabled"].fillna(False).astype(bool)
    replaced_mask = records_df["escape_replaced"].fillna(False).astype(bool)

    grouped_rows: List[Dict[str, Any]] = []
    for samples_path, group in records_df.groupby("samples_path", dropna=False):
        enabled_group = group[enabled_mask.loc[group.index]]
        replaced_group = group[replaced_mask.loc[group.index]]
        non_replaced_group = group[~replaced_mask.loc[group.index]]

        grouped_rows.append(
            {
                "samples_path": samples_path,
                "run_dir": str(group["run_dir"].iloc[0]),
                "subset": str(group["subset"].iloc[0]),
                "num_sample_iters": int(group.shape[0]),
                "num_enabled_sample_iters": int(enabled_group.shape[0]),
                "num_replaced_sample_iters": int(replaced_group.shape[0]),
                "enabled_rate": 100.0 * float(enabled_group.shape[0] / max(1, group.shape[0])),
                "replacement_rate_over_all": 100.0 * float(replaced_group.shape[0] / max(1, group.shape[0])),
                "replacement_rate_over_enabled": 100.0
                * float(replaced_group.shape[0] / max(1, enabled_group.shape[0])),
                "avg_replaced_slots_over_all": float(group["num_escape_replaced"].mean()),
                "avg_replaced_slots_when_replaced": float(replaced_group["num_escape_replaced"].mean())
                if not replaced_group.empty
                else 0.0,
                "avg_ndcg10_replaced": float(replaced_group["ndcg10"].mean()) if not replaced_group.empty else np.nan,
                "avg_ndcg10_not_replaced": float(non_replaced_group["ndcg10"].mean())
                if not non_replaced_group.empty
                else np.nan,
            }
        )

    return pd.DataFrame(grouped_rows).sort_values(["subset", "samples_path"]).reset_index(drop=True)


def _summarize_pick_reasons(records_df: pd.DataFrame) -> pd.DataFrame:
    if records_df.empty:
        return pd.DataFrame()

    reason_df = (
        records_df.groupby(["samples_path", "subset", "global_escape_pick_reason"], dropna=False)
        .agg(
            count=("sample_idx", "count"),
            mean_replaced_slots=("num_escape_replaced", "mean"),
            replacement_rate=("escape_replaced", "mean"),
        )
        .reset_index()
    )
    reason_df["replacement_rate"] = 100.0 * reason_df["replacement_rate"]
    return reason_df.sort_values(["subset", "samples_path", "count"], ascending=[True, True, False]).reset_index(
        drop=True
    )


def _print_summary(run_df: pd.DataFrame, iter_df: pd.DataFrame, reason_df: pd.DataFrame) -> None:
    pd.set_option("display.max_colwidth", 160)

    print("[Round6 Global Escape] Per-run summary")
    if run_df.empty:
        print("No iter_records found.")
    else:
        print(run_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n[Round6 Global Escape] Per-iteration summary")
    if iter_df.empty:
        print("No per-iteration rows found.")
    else:
        print(iter_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n[Round6 Global Escape] Pick-reason summary")
    if reason_df.empty:
        print("No reason rows found.")
    else:
        print(reason_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def _save_csv(df: pd.DataFrame, path: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze how often round6 method1 global escape actually replaces local beam slots."
    )
    parser.add_argument(
        "--input",
        nargs="*",
        default=[],
        help="One or more all_eval_sample_dicts.pkl paths or run directories containing it.",
    )
    parser.add_argument(
        "--glob_pattern",
        type=str,
        default="",
        help="Optional glob pattern to discover all_eval_sample_dicts.pkl files.",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="",
        help="Optional prefix for CSV outputs. Writes *_records.csv, *_per_iter.csv, *_per_run.csv, *_pick_reasons.csv",
    )
    args = parser.parse_args()

    eval_paths = _resolve_eval_paths(args.input, args.glob_pattern)
    record_dfs = [_extract_sample_iter_rows(path) for path in eval_paths]
    records_df = pd.concat(record_dfs, axis=0, ignore_index=True) if record_dfs else pd.DataFrame()

    per_iter_df = _summarize_per_iter(records_df)
    per_run_df = _summarize_per_run(records_df)
    pick_reason_df = _summarize_pick_reasons(records_df)

    _print_summary(per_run_df, per_iter_df, pick_reason_df)

    if args.out_prefix:
        _save_csv(records_df, f"{args.out_prefix}_records.csv")
        _save_csv(per_iter_df, f"{args.out_prefix}_per_iter.csv")
        _save_csv(per_run_df, f"{args.out_prefix}_per_run.csv")
        _save_csv(pick_reason_df, f"{args.out_prefix}_pick_reasons.csv")
        print(f"\nSaved CSVs with prefix: {args.out_prefix}")


if __name__ == "__main__":
    main()
