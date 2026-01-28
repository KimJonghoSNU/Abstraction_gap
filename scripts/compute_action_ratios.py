import argparse
import glob
import os
import pickle as pkl
from typing import Dict, List, Tuple

import pandas as pd


def iter_experiment_dirs(root: str) -> List[str]:
    matches = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "all_eval_sample_dicts.pkl" in filenames:
            matches.append(dirpath)
    return sorted(matches)


def iter_experiment_dirs_from_pattern(pattern: str) -> List[str]:
    matches = []
    for path in glob.glob(pattern, recursive=True):
        if os.path.basename(path) != "all_eval_sample_dicts.pkl":
            continue
        matches.append(os.path.dirname(path))
    return sorted(set(matches))


def extract_actions(sample_dict: Dict) -> List[Tuple[int, str]]:
    actions: List[Tuple[int, str]] = []
    for rec in sample_dict.get("iter_records", []):
        iter_idx = rec.get("iter")
        if iter_idx is None:
            continue
        # Decision: prefer oracle_action_choice when present; fall back to action.
        action = rec.get("oracle_action_choice") or rec.get("action")
        if not action:
            continue
        action = str(action).strip().lower()
        actions.append((int(iter_idx), action))
    return actions


def summarize_actions(samples: List[Dict]) -> Dict[int, Dict[str, int]]:
    counts: Dict[int, Dict[str, int]] = {}
    for sample in samples:
        for iter_idx, action in extract_actions(sample):
            if iter_idx not in counts:
                counts[iter_idx] = {"exploit": 0, "explore": 0, "other": 0}
            if action in ("exploit", "explore"):
                counts[iter_idx][action] += 1
            else:
                counts[iter_idx]["other"] += 1
    return counts


def summarize_action_ndcg(metrics_path: str) -> Dict[int, Dict[str, float]]:
    if not os.path.exists(metrics_path):
        return {}
    try:
        df = pd.read_pickle(metrics_path)
    except Exception:
        return {}
    if df is None or df.empty or not isinstance(df.columns, pd.MultiIndex):
        return {}
    means: Dict[int, Dict[str, float]] = {}
    for iter_idx, subdf in df.groupby(level=0, axis=1):
        if not isinstance(iter_idx, str) or not iter_idx.startswith("Iter "):
            continue
        try:
            iter_num = int(iter_idx.split(" ", 1)[1])
        except Exception:
            continue
        sub = subdf.xs(iter_idx, axis=1, level=0)
        if "OracleExploit_nDCG@10" not in sub.columns and "OracleExplore_nDCG@10" not in sub.columns:
            continue
        means[iter_num] = {
            "exploit_ndcg": float(sub.get("OracleExploit_nDCG@10", pd.Series(dtype=float)).mean()),
            "explore_ndcg": float(sub.get("OracleExplore_nDCG@10", pd.Series(dtype=float)).mean()),
        }
    return means

# python scripts/compute_action_ratios.py --pattern "results/BRIGHT/**/all_eval_sample_dicts.pkl"
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="results/BRIGHT", help="Root results directory")
    parser.add_argument("--pattern", type=str, default="", help="Glob pattern for all_eval_sample_dicts.pkl")
    parser.add_argument("--output", type=str, default="", help="Optional output CSV path")
    args = parser.parse_args()

    if args.pattern:
        exp_dirs = iter_experiment_dirs_from_pattern(args.pattern)
    else:
        root = os.path.abspath(args.root)
        exp_dirs = iter_experiment_dirs(root)
    if not exp_dirs:
        if args.pattern:
            print(f"No experiments found for pattern {args.pattern}")
        else:
            print(f"No experiments found under {root}")
        return

    rows: List[str] = []
    header = "experiment,iter,exploit,explore,other,exploit_ratio,explore_ratio,exploit_ndcg,explore_ndcg,ndcg_diff"
    rows.append(header)

    for exp_dir in exp_dirs:
        pkl_path = os.path.join(exp_dir, "all_eval_sample_dicts.pkl")
        try:
            samples = pkl.load(open(pkl_path, "rb"))
        except Exception as exc:
            print(f"Failed to load {pkl_path}: {exc}")
            continue

        counts = summarize_actions(samples)
        # Decision: use all_eval_metrics.pkl for per-action nDCG if available.
        metrics_path = os.path.join(exp_dir, "all_eval_metrics.pkl")
        ndcg_means = summarize_action_ndcg(metrics_path)
        for iter_idx in sorted(counts.keys()):
            c = counts[iter_idx]
            total = c["exploit"] + c["explore"] + c["other"]
            exploit_ratio = (c["exploit"] / total) if total else 0.0
            explore_ratio = (c["explore"] / total) if total else 0.0
            ndcg = ndcg_means.get(iter_idx, {})
            exploit_ndcg = ndcg.get("exploit_ndcg", 0.0)
            explore_ndcg = ndcg.get("explore_ndcg", 0.0)
            ndcg_diff = exploit_ndcg - explore_ndcg
            rows.append(",".join([
                exp_dir,
                str(iter_idx),
                str(c["exploit"]),
                str(c["explore"]),
                str(c["other"]),
                f"{exploit_ratio:.4f}",
                f"{explore_ratio:.4f}",
                f"{exploit_ndcg:.4f}",
                f"{explore_ndcg:.4f}",
                f"{ndcg_diff:.4f}",
            ]))

    output = args.output
    if output:
        out_path = os.path.abspath(output)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(rows))
        print(f"Wrote {len(rows) - 1} rows to {out_path}")
    else:
        for row in rows[:50]:
            print(row)
        if len(rows) > 50:
            print(f"... ({len(rows) - 50} more rows)")


if __name__ == "__main__":
    main()
