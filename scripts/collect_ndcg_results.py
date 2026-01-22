import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from hyperparams import abbreviate_key  # noqa: E402


categories=("biology", "earth_science", "psychology", "leetcode", "economics", "robotics", "stackoverflow", "sustainable_living", "pony", "aops", "theoremqa_questions", "theoremqa_theorems")

def _find_metrics_files(base_dir: str) -> List[str]:
    metrics_files: List[str] = []
    for root, _dirs, files in os.walk(base_dir):
        if "all_eval_metrics.pkl" in files:
            metrics_files.append(os.path.join(root, "all_eval_metrics.pkl"))
            continue
        legacy = [f for f in files if f.startswith("all_eval_metrics-") and f.endswith(".pkl")]
        for fname in legacy:
            metrics_files.append(os.path.join(root, fname))
    return metrics_files


def _extract_ndcg_means(df: pd.DataFrame, iter_idx: int) -> Optional[float]:
    if hasattr(df.columns, "levels"):
        iter_key = f"Iter {iter_idx}"
        if iter_key not in df.columns.levels[0]:
            return None
        if "nDCG@10" not in df[iter_key].columns:
            return None
        return float(df[iter_key]["nDCG@10"].mean())
    if iter_idx != 0:
        return None
    if "nDCG@10" not in df.columns:
        return None
    return float(df["nDCG@10"].mean())


def _extract_max_ndcg(df: pd.DataFrame) -> Tuple[Optional[float], Optional[int]]:
    if hasattr(df.columns, "levels"):
        best_val = None
        best_iter = None
        for iter_key in df.columns.levels[0]:
            if not isinstance(iter_key, str) or not iter_key.startswith("Iter "):
                continue
            if "nDCG@10" not in df[iter_key].columns:
                continue
            mean_val = float(df[iter_key]["nDCG@10"].mean())
            iter_idx = int(iter_key.split("Iter ")[-1])
            if best_val is None or mean_val > best_val:
                best_val = mean_val
                best_iter = iter_idx
        return best_val, best_iter
    if "nDCG@10" not in df.columns:
        return None, None
    return float(df["nDCG@10"].mean()), 0


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
    tokens = [t for t in rel.replace("/", "-").split("-") if t]
    kept: List[str] = []
    for token in tokens:
        if "=" not in token:
            kept.append(token)
            continue
        key, val = token.split("=", 1)
        if key in drop_map and drop_map[key] == val:
            continue
        kept.append(token)
    cleaned = "-".join(kept)
    parts = cleaned.split(os.sep) if cleaned else []
    # category = parts[0] if parts else "unknown"
    # category name should be one of predefined categories
    category = "unknown"
    for cat in categories:
        if parts[0].startswith(cat):
            category = cat
            break
    return category, cleaned


def collect_ndcg_results(base_dir: str, drop_map: Dict[str, str], exclude_subdirs: List[str]) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for metrics_path in tqdm(sorted(_find_metrics_files(base_dir))):
        if any(f"{os.sep}{subdir}{os.sep}" in metrics_path for subdir in exclude_subdirs):
            print(f"Skipping excluded path: {metrics_path}")
            continue
        try:
            df = pd.read_pickle(metrics_path)
        except Exception:
            print(f"Warning: Failed to read metrics file: {metrics_path}")
            continue
        category, exp_id = _relative_experiment_id(base_dir, metrics_path, drop_map)
        max_ndcg, max_iter = _extract_max_ndcg(df)
        records.append({
            "category": category,
            "experiment": exp_id,
            "ndcg_iter0": round(_extract_ndcg_means(df, 0), 2),
            "ndcg_max": round(max_ndcg, 2),
            "ndcg_max_iter": max_iter,
        })
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect nDCG@10 results for iter 0 and iter 3.")
    parser.add_argument("--base_dir", type=str, default="results/BRIGHT", help="Base results directory")
    parser.add_argument("--out_csv", type=str, default="results/BRIGHT/ndcg_summary.csv", help="Output CSV path")
    parser.add_argument("--exclude_subdir", action="append", default=["260116", "260121"], help="Subdirectory name to exclude")
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
    print(f"Collecting nDCG@10 results from {base_dir}...")
    drop_map = _build_drop_map(args.drop_param)
    df = collect_ndcg_results(base_dir, drop_map, args.exclude_subdir)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} rows to {out_csv}")


if __name__ == "__main__":
    main()
