import argparse
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.append(SCRIPT_DIR)

from collect_ndcg_results import (  # noqa: E402
    _build_drop_map,
    _find_metrics_files,
    _normalize_glob_pattern,
    _relative_experiment_id,
)


def _extract_iter_means(
    df: pd.DataFrame,
    metric: str,
    max_iter: Optional[int] = None,
) -> List[Tuple[int, float]]:
    rows: List[Tuple[int, float]] = []
    if hasattr(df.columns, "levels"):
        iter_keys = [
            key for key in df.columns.levels[0]
            if isinstance(key, str) and key.startswith("Iter ")
        ]
        for iter_key in sorted(iter_keys, key=lambda x: int(x.split("Iter ")[-1])):
            if metric not in df[iter_key].columns:
                continue
            iter_idx = int(iter_key.split("Iter ")[-1])
            if max_iter is not None and iter_idx >= int(max_iter):
                continue
            rows.append((iter_idx, float(df[iter_key][metric].mean())))
        return rows

    if metric not in df.columns:
        return rows
    if max_iter is not None and int(max_iter) <= 0:
        return rows
    rows.append((0, float(df[metric].mean())))
    return rows


def collect_iter_results(
    base_dir: str,
    drop_map: Dict[str, str],
    exclude_subdirs: List[str],
    include_dir: Optional[str],
    metric: str,
    max_iter: Optional[int],
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for metrics_path in tqdm(sorted(_find_metrics_files(base_dir, include_dir))):
        if any(f"{os.sep}{subdir}{os.sep}" in metrics_path for subdir in exclude_subdirs):
            print(f"Skipping excluded path: {metrics_path}")
            continue
        try:
            df = pd.read_pickle(metrics_path)
        except Exception:
            print(f"Warning: Failed to read metrics file: {metrics_path}")
            continue
        category, exp_id = _relative_experiment_id(base_dir, metrics_path, drop_map)
        for iter_idx, mean_val in _extract_iter_means(df, metric, max_iter=max_iter):
            records.append(
                {
                    "category": category,
                    "experiment": exp_id,
                    "iter": int(iter_idx),
                    "metric_mean": round(float(mean_val), 2),
                }
            )
    return pd.DataFrame(records)


def _metric_iter_column_name(metric: str, iter_idx: int) -> str:
    metric_key = re.sub(r"[^a-z0-9]+", "_", str(metric).lower()).strip("_")
    if metric_key == "ndcg_10":
        metric_key = "ndcg"
    return f"avg_{metric_key}_iter{int(iter_idx)}"


def _format_flat_overall_iter_results(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if df.empty:
        return df
    overall = df.groupby(["experiment", "iter"], as_index=False)["metric_mean"].mean()
    wide = overall.pivot_table(
        index="experiment",
        columns="iter",
        values="metric_mean",
        aggfunc="first",
    )
    ordered_iters = sorted(int(x) for x in overall["iter"].dropna().unique())
    wide = wide.reindex(columns=ordered_iters)
    wide.columns = [_metric_iter_column_name(metric, int(iter_idx)) for iter_idx in wide.columns]
    wide = wide.reset_index()
    metric_cols = [col for col in wide.columns if col != "experiment"]
    if metric_cols:
        wide[metric_cols] = wide[metric_cols].apply(pd.to_numeric, errors="coerce").round(4)
    return wide


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect per-iteration mean metric summaries from all_eval_metrics.pkl files."
    )
    parser.add_argument("--base_dir", type=str, default="results/BRIGHT", help="Base results directory")
    parser.add_argument(
        "--out_csv",
        type=str,
        default="results/BRIGHT/ndcg_iter_summary.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--exclude_subdir",
        action="append",
        default=["260116", "260121"],
        help="Subdirectory name to exclude",
    )
    parser.add_argument("--metric", type=str, default="nDCG@10", help="Metric column to aggregate")
    parser.add_argument(
        "--include_dir",
        type=str,
        default=None,
        help="Only scan directories matching this glob (e.g., baseline or *baseline*)",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=None,
        help="If set, keep only iterations with index < max_iter.",
    )
    parser.add_argument(
        "--drop_param",
        action="append",
        default=[
            "tree_version=bottom-up",
            "tree_version=top-down",
            "tree_pred_version=5",
            "reasoning_in_traversal_prompt=/1",
            "num_leaf_calib=10",
            "pl_tau=5",
            "relevance_chain_factor=0",
            "llm_api_backend=vllm",
            "llm=Qwen3-4B-Instruct-2507",
            "num_iters=5",
            "num_eval_samples=1000",
            "max_beam_size=2",
            "flat_then_tree=True",
            "flat_topk=100",
            "gate_branches_topb=10",
        ],
        help="Drop params (full names) from experiment path, e.g. --drop_param tree_version=bottom-up",
    )
    args = parser.parse_args()

    base_dir = args.base_dir
    out_csv = args.out_csv
    if not os.path.isabs(base_dir):
        base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), base_dir)
    if not os.path.isabs(out_csv):
        out_csv = os.path.join(os.path.dirname(os.path.dirname(__file__)), out_csv)
    if args.include_dir:
        out_csv = out_csv[:-4] + _normalize_glob_pattern(args.include_dir).replace("*", "") + out_csv[-4:]
    if args.max_iter is not None:
        out_csv = out_csv[:-4] + f"_maxiter{int(args.max_iter)}" + out_csv[-4:]

    print(f"Collecting per-iteration {args.metric} means from {base_dir}...")
    drop_map = _build_drop_map(args.drop_param)
    df = collect_iter_results(
        base_dir,
        drop_map,
        args.exclude_subdir,
        args.include_dir,
        args.metric,
        args.max_iter,
    )
    wide_df = _format_flat_overall_iter_results(df, args.metric)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    wide_df.to_csv(out_csv, index=False)
    print(f"Wrote {len(wide_df)} rows to {out_csv}")


if __name__ == "__main__":
    main()
