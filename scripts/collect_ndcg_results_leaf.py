import argparse
import glob
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from hyperparams import abbreviate_key  # noqa: E402


categories = (
    "biology",
    "earth_science",
    "psychology",
    "leetcode",
    "economics",
    "robotics",
    "stackoverflow",
    "sustainable_living",
    "pony",
    "aops",
    "theoremqa_questions",
    "theoremqa_theorems",
)
_PARAM_START_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*=")


def _normalize_glob_pattern(pattern: str) -> str:
    if any(ch in pattern for ch in ["*", "?", "["]):
        return pattern
    return f"*{pattern}*"


def _split_param_tokens(segment: str) -> List[str]:
    tokens: List[str] = []
    if not segment:
        return tokens
    start = 0
    i = 0
    while i < len(segment):
        if segment[i] == "-" and _PARAM_START_RE.match(segment[i + 1:]):
            tokens.append(segment[start:i])
            start = i + 1
        i += 1
    tokens.append(segment[start:])
    return [t for t in tokens if t]


def _find_metrics_files(base_dir: str, include_dir: Optional[str]) -> List[str]:
    metrics_files: List[str] = []
    if include_dir:
        pattern = _normalize_glob_pattern(include_dir)
        matched_dirs = [
            path for path in glob.glob(os.path.join(base_dir, "**", pattern), recursive=True)
            if os.path.isdir(path)
        ]
        seen_roots = set()
        for root in matched_dirs:
            if root in seen_roots:
                continue
            seen_roots.add(root)
            for walk_root, _dirs, files in os.walk(root):
                if "leaf_iter_metrics.jsonl" in files:
                    metrics_files.append(os.path.join(walk_root, "leaf_iter_metrics.jsonl"))
    else:
        for root, _dirs, files in os.walk(base_dir):
            if "leaf_iter_metrics.jsonl" in files:
                metrics_files.append(os.path.join(root, "leaf_iter_metrics.jsonl"))
    return metrics_files


def _build_drop_map(drop_params: List[str]) -> Dict[str, str]:
    drop_map: Dict[str, str] = {}
    for item in drop_params:
        if "=" not in item:
            continue
        key, val = item.split("=", 1)
        key = key.strip()
        val = val.strip()
        if not key:
            continue
        drop_map[abbreviate_key(key)] = val
    return drop_map


def _relative_experiment_id(base_dir: str, metrics_path: str, drop_map: Dict[str, str]) -> Tuple[str, str]:
    rel = os.path.relpath(os.path.dirname(metrics_path), base_dir)
    tokens: List[str] = []
    for part in rel.split(os.sep):
        tokens.extend(_split_param_tokens(part))
    kept: List[str] = []
    for token in tokens:
        if "=" not in token:
            kept.append(token)
            continue
        key, val = token.split("=", 1)
        if key in drop_map and drop_map[key] == val:
            continue
        if key == "TV" and val in ("bottom-up", "top-down"):
            continue
        kept.append(token)
    cleaned = "-".join(kept)
    parts = cleaned.split(os.sep) if cleaned else []
    category = "unknown"
    for cat in categories:
        if parts[0].startswith(cat):
            category = cat
            break
    if category != "unknown":
        kept = [t for t in kept if t != f"S={category}"]
        if kept and kept[0].startswith(category):
            kept = kept[1:]
        cleaned = "-".join(kept)
    return category, cleaned


def _load_leaf_metrics(metrics_path: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return pd.DataFrame(rows)


def _extract_last_ndcg(
    grouped: pd.Series,
    max_iter: Optional[int] = None,
) -> Tuple[Optional[float], Optional[int]]:
    if max_iter is not None:
        grouped = grouped[grouped.index < int(max_iter)]
    if grouped.empty:
        return None, None
    last_iter = int(grouped.index.max())
    return float(grouped.loc[last_iter]), last_iter


def _extract_max_ndcg(
    grouped: pd.Series,
    max_iter: Optional[int] = None,
) -> Tuple[Optional[float], Optional[int]]:
    if max_iter is not None:
        grouped = grouped[grouped.index < int(max_iter)]
    if grouped.empty:
        return None, None
    max_iter = int(grouped.idxmax())
    return float(grouped.loc[max_iter]), max_iter


def _round_or_none(value: Optional[float], digits: int = 2) -> Optional[float]:
    if value is None:
        return None
    return round(value, digits)


def collect_ndcg_results(
    base_dir: str,
    drop_map: Dict[str, str],
    exclude_subdirs: List[str],
    include_dir: Optional[str],
    max_iter: Optional[int],
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for metrics_path in tqdm(sorted(_find_metrics_files(base_dir, include_dir))):
        if any(f"{os.sep}{subdir}{os.sep}" in metrics_path for subdir in exclude_subdirs):
            print(f"Skipping excluded path: {metrics_path}")
            continue
        df = _load_leaf_metrics(metrics_path)
        if df.empty:
            continue
        category, exp_id = _relative_experiment_id(base_dir, metrics_path, drop_map)
        if "iter" not in df.columns or "nDCG@10" not in df.columns:
            continue
        grouped = df.groupby("iter")["nDCG@10"].mean()
        if grouped.empty:
            continue
        # Intent: mirror the main collector by reporting the terminal leaf-ranking score, not the initial iteration score.
        end_ndcg, _end_iter = _extract_last_ndcg(grouped, max_iter=max_iter)
        max_ndcg, max_iter_idx = _extract_max_ndcg(grouped, max_iter=max_iter)
        records.append({
            "category": category,
            "experiment": exp_id,
            "ndcg_end": _round_or_none(end_ndcg),
            "ndcg_max": _round_or_none(max_ndcg),
            "ndcg_max_iter": max_iter_idx,
        })
    return pd.DataFrame(records)


def _format_wide_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    wide = df.pivot_table(
        index="experiment",
        columns="category",
        values=["ndcg_end", "ndcg_max", "ndcg_max_iter"],
        aggfunc="first",
    )
    ordered_categories = [cat for cat in categories if cat in df["category"].unique()]
    metric_order = ["ndcg_end", "ndcg_max", "ndcg_max_iter"]
    wide = wide.reindex(columns=pd.MultiIndex.from_product([metric_order, ordered_categories]))
    wide.columns = pd.MultiIndex.from_tuples([(cat, metric) for metric, cat in wide.columns])
    wide = wide.sort_index(axis=1, level=[0, 1], sort_remaining=False)
    wide = wide.reset_index()
    return wide


def _append_ndcg_mean_columns(wide_df: pd.DataFrame) -> pd.DataFrame:
    if wide_df.empty:
        return wide_df
    ndcg_end_cols: List[object] = []
    ndcg_max_cols: List[object] = []
    for col in wide_df.columns:
        if isinstance(col, tuple):
            metric_name = col[1] if len(col) > 1 else ""
        else:
            metric_name = str(col)
        if metric_name == "ndcg_end":
            ndcg_end_cols.append(col)
        if metric_name == "ndcg_max":
            ndcg_max_cols.append(col)

    out_df = wide_df.copy()
    # Intent: keep the leaf summary aligned with the main collector's experiment-level terminal/max averages.
    if ndcg_end_cols:
        ndcg_end_vals = out_df[ndcg_end_cols].apply(pd.to_numeric, errors="coerce")
        out_df[("overall", "avg_ndcg_end")] = ndcg_end_vals.mean(axis=1)
    else:
        out_df[("overall", "avg_ndcg_end")] = pd.NA
    if ndcg_max_cols:
        ndcg_max_vals = out_df[ndcg_max_cols].apply(pd.to_numeric, errors="coerce")
        out_df[("overall", "avg_ndcg_max")] = ndcg_max_vals.mean(axis=1)
    else:
        out_df[("overall", "avg_ndcg_max")] = pd.NA
    return out_df

# python scripts/collect_ndcg_results_leaf.py --include_dir *baseline*
def main() -> None:
    parser = argparse.ArgumentParser(description="Collect leaf-only nDCG@10 results (run_leaf_rank).")
    parser.add_argument("--base_dir", type=str, default="results/BRIGHT", help="Base results directory")
    parser.add_argument("--out_csv", type=str, default="results/BRIGHT/ndcg_summary_leaf.csv", help="Output CSV path")
    parser.add_argument("--exclude_subdir", action="append", default=["260116", "260121"], help="Subdirectory name to exclude")
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
        help="If set, compute ndcg_end/ndcg_max using only iterations with index < max_iter (e.g., 5 => iter 0..4).",
    )
    parser.add_argument(
        "--drop_param",
        action="append",
        default=[
            "tree_version=bottom-up",
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
        out_csv = out_csv[:-4] + args.include_dir.replace("*", "") + out_csv[-4:]
    if args.max_iter is not None:
        out_csv = out_csv[:-4] + f"_maxiter{int(args.max_iter)}" + out_csv[-4:]
    print(f"Collecting leaf-only nDCG@10 results from {base_dir}...")
    drop_map = _build_drop_map(args.drop_param)
    df = collect_ndcg_results(base_dir, drop_map, args.exclude_subdir, args.include_dir, args.max_iter)
    wide_df = _append_ndcg_mean_columns(_format_wide_results(df))
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    wide_df.to_csv(out_csv, index=False)
    print(f"Wrote {len(wide_df)} rows to {out_csv}")


if __name__ == "__main__":
    main()
