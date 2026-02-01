import argparse
import glob
import math
import os
import pickle as pkl
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


DATASET_NAMES = [
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


def infer_dataset_name(path: str) -> str:
    lower = path.lower()
    for name in DATASET_NAMES:
        if name in lower:
            return name
    return "unknown"


def iter_experiment_dirs_from_pattern(pattern: str) -> List[str]:
    matches = []
    for path in glob.glob(pattern, recursive=True):
        if os.path.basename(path) != "all_eval_sample_dicts.pkl":
            continue
        matches.append(os.path.dirname(path))
    return sorted(set(matches))


def _normalize_action(action: Optional[str]) -> Optional[str]:
    if not action:
        return None
    action = str(action).strip().lower()
    if action in ("explore", "exploit"):
        return action
    return "other"


def _density_values_from_record(record: Dict) -> List[float]:
    density = record.get("density")
    if isinstance(density, dict) and density:
        values = [v for v in density.values() if isinstance(v, (int, float)) and v > 0]
        return values

    anchor_leaf_paths = record.get("anchor_leaf_paths") or []
    prefixes: List[Tuple[int, ...]] = []
    for path in anchor_leaf_paths:
        if isinstance(path, (list, tuple)) and len(path) > 1:
            prefixes.append(tuple(path[:-1]))
    if not prefixes:
        return []
    counts = Counter(prefixes)
    total = sum(counts.values())
    if not total:
        return []
    return [count / total for count in counts.values()]


def _density_stats(values: Iterable[float]) -> Tuple[float, float]:
    vals = [v for v in values if v > 0]
    if not vals:
        return 0.0, 0.0
    dmax = max(vals)
    entropy = -sum(v * math.log(v) for v in vals)
    return dmax, entropy


def _extract_hit_flags(record: Dict, metric_key: str) -> Tuple[int, int]:
    metrics = record.get(metric_key) or {}
    recall10 = metrics.get("Recall@10", 0.0)
    recall100 = metrics.get("Recall@100", 0.0)
    hit10 = 1 if isinstance(recall10, (int, float)) and recall10 > 0 else 0
    hit100 = 1 if isinstance(recall100, (int, float)) and recall100 > 0 else 0
    return hit10, hit100


def extract_signal_rows(samples: List[Dict], exp_dir: str) -> List[Dict]:
    rows: List[Dict] = []
    dataset = infer_dataset_name(exp_dir)
    for sample in samples:
        for record in sample.get("iter_records", []):
            iter_idx = record.get("iter")
            if iter_idx is None:
                continue
            values = _density_values_from_record(record)
            dmax, entropy = _density_stats(values)
            anchor_base = record.get("anchor_top_paths") or []
            anchor_set = {tuple(p) for p in anchor_base}
            cand_top10 = record.get("oracle_action_anchor_top10")
            if isinstance(cand_top10, dict) and cand_top10 and anchor_set:
                denom = float(len(anchor_set))
                for action_tag, paths in cand_top10.items():
                    action = _normalize_action(action_tag)
                    cand_set = {tuple(p) for p in (paths or [])}
                    overlap = len(anchor_set & cand_set) / denom if denom else 0.0
                    drift = 1.0 - overlap
                    rows.append({
                        "dataset": dataset,
                        "iter": int(iter_idx),
                        "action": action or "other",
                        "dmax": dmax,
                        "entropy": entropy,
                        "anchor_overlap@10": float(overlap),
                        "anchor_drift@10": float(drift),
                        "rrf_hit10": None,
                        "rrf_hit100": None,
                    })
                continue

            action = _normalize_action(record.get("oracle_action_choice") or record.get("action"))
            rrf_hit10, rrf_hit100 = _extract_hit_flags(record, "rrf_metrics")
            rows.append({
                "dataset": dataset,
                "iter": int(iter_idx),
                "action": action or "other",
                "dmax": dmax,
                "entropy": entropy,
                "anchor_overlap@10": None,
                "anchor_drift@10": None,
                "rrf_hit10": rrf_hit10,
                "rrf_hit100": rrf_hit100,
            })
    return rows


def _point_biserial(x: pd.Series, y: pd.Series) -> float:
    if x.empty or y.empty:
        return float("nan")
    if x.nunique() < 2:
        return float("nan")
    if y.nunique() < 2:
        return float("nan")
    return float(x.corr(y))



# python scripts/compute_query_signal_stats.py --pattern "results/BRIGHT/**/NumES=1000-MaxBS=2-S=oracle_round3_anchor_local_rank_round3_action_v1-FT=1000-GBT=10/**/all_eval_sample_dicts.pkl"  --output results/BRIGHT/oracle/S=oracle_round3_anchor_local_rank_round3_action_v1/query_signal_stats.csv
# python scripts/compute_query_signal_stats.py --pattern "results/BRIGHT/**/NumES=1000-MaxBS=2-S=round3_action_oracle_round3_action_v1-FT=1000-GBT=10/**/all_eval_sample_dicts.pkl"  --output results/BRIGHT/oracle/S=round3_action_oracle_round3_action_v1/query_signal_stats.csv

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pattern",
        type=str,
        default="results/BRIGHT/**/all_eval_sample_dicts.pkl",
        help="Glob pattern to scan for all_eval_sample_dicts.pkl",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output CSV path (summary). Correlations saved to *_corr.csv",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=0,
        help="Optional number of bins for dmax to compute explore ratio per bin",
    )
    args = parser.parse_args()

    exp_dirs = iter_experiment_dirs_from_pattern(args.pattern)
    if not exp_dirs:
        print(f"No experiments found for pattern {args.pattern}")
        return

    rows: List[Dict] = []
    for exp_dir in exp_dirs:
        pkl_path = os.path.join(exp_dir, "all_eval_sample_dicts.pkl")
        try:
            samples = pkl.load(open(pkl_path, "rb"))
        except Exception as exc:
            print(f"Failed to load {pkl_path}: {exc}")
            continue
        rows.extend(extract_signal_rows(samples, exp_dir))

    if not rows:
        print("No iter_records found.")
        return

    df = pd.DataFrame(rows)
    if df.empty:
        print("No rows after parsing.")
        return

    summary = (
        df.groupby(["dataset", "iter", "action"])
        .agg(
            count=("dmax", "size"),
            mean_dmax=("dmax", "mean"),
            mean_entropy=("entropy", "mean"),
            median_dmax=("dmax", "median"),
            median_entropy=("entropy", "median"),
            rrf_hit10_rate=("rrf_hit10", "mean"),
            rrf_hit100_rate=("rrf_hit100", "mean"),
            mean_anchor_overlap10=("anchor_overlap@10", "mean"),
            mean_anchor_drift10=("anchor_drift@10", "mean"),
        )
        .reset_index()
    )

    df_action = df.copy()
    df_action["is_explore"] = (df_action["action"] == "explore").astype(int)
    corr_rows = []
    for (dataset, iter_idx), sub in df_action.groupby(["dataset", "iter"]):
        corr_rows.append({
            "dataset": dataset,
            "iter": iter_idx,
            "corr_dmax": _point_biserial(sub["dmax"], sub["is_explore"]),
            "corr_entropy": _point_biserial(sub["entropy"], sub["is_explore"]),
            "explore_ratio": float(sub["is_explore"].mean()),
        })
    corr_df = pd.DataFrame(corr_rows)

    bin_df = None
    if args.bins and args.bins > 0:
        df_bins = df_action.copy()
        df_bins["dmax_bin"] = pd.qcut(df_bins["dmax"], args.bins, duplicates="drop")
        bin_df = (
            df_bins.groupby(["dataset", "iter", "dmax_bin"])
            .agg(
                count=("is_explore", "size"),
                explore_ratio=("is_explore", "mean"),
                mean_dmax=("dmax", "mean"),
                mean_entropy=("entropy", "mean"),
            )
            .reset_index()
        )

    if args.output:
        out_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        summary.to_csv(out_path, index=False, float_format="%.3f")
        corr_path = out_path.replace(".csv", "_corr.csv")
        corr_df.to_csv(corr_path, index=False, float_format="%.3f")
        if bin_df is not None:
            bin_path = out_path.replace(".csv", "_bins.csv")
            bin_df.to_csv(bin_path, index=False, float_format="%.3f")
        print(f"Wrote {len(summary)} rows to {out_path}")
        print(f"Wrote {len(corr_df)} rows to {corr_path}")
        if bin_df is not None:
            print(f"Wrote {len(bin_df)} rows to {bin_path}")
    else:
        print(summary.round(3).head(50).to_string(index=False))
        print(corr_df.round(3).head(50).to_string(index=False))
        if bin_df is not None:
            print(bin_df.round(3).head(50).to_string(index=False))


if __name__ == "__main__":
    main()
