"""
Analyze round6 method2-RRF runs by separating:
1) official nDCG@10 from the banked controller result
2) current-run nDCG@10 from the same iteration's local retrieval only

This is used to test whether degradation comes more from:
- the fusion rule itself, or
- the first explore step hurting immediate retrieval quality.

Example:
    python scripts/analyze_round6_rrf_explore_vs_fusion.py \
        --base_dir results/BRIGHT \
        --exclude_subsets stackoverflow \
        --output_prefix results/BRIGHT/analysis/round6_rrf_explore_vs_fusion_no_stackoverflow
"""

from __future__ import annotations

import argparse
import math
import pickle as pkl
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


def compute_ndcg_doc_ids(doc_ids: List[str], gold_doc_ids: List[str], k: int = 10) -> float:
    ranked = list(doc_ids)[: max(1, int(k))]
    gold = list(gold_doc_ids)
    if not gold:
        return float("nan")

    dcg = 0.0
    for rank_idx, doc_id in enumerate(ranked, start=1):
        if doc_id in gold:
            dcg += 1.0 / math.log2(rank_idx + 1)

    ideal_hits = min(len(gold), max(1, int(k)))
    idcg = sum(1.0 / math.log2(rank_idx + 1) for rank_idx in range(1, ideal_hits + 1))
    return 100.0 * dcg / idcg if idcg > 0 else float("nan")


def iter_rrf_sample_pkls(base_dir: Path, include_subsets: Iterable[str], exclude_subsets: Iterable[str]) -> List[Path]:
    include = {str(x).strip() for x in include_subsets if str(x).strip()}
    exclude = {str(x).strip() for x in exclude_subsets if str(x).strip()}

    all_paths = sorted(base_dir.glob("*/round6/**/all_eval_sample_dicts.pkl"))
    selected: List[Path] = []
    for path in all_paths:
        subset = path.parts[2]
        if include and subset not in include:
            continue
        if subset in exclude:
            continue
        if "method2_rrf" not in str(path):
            continue
        selected.append(path)
    return selected


def build_row_records(sample_dicts: List[Dict], subset: str) -> List[Dict]:
    rows: List[Dict] = []
    for sample_idx, sample in enumerate(sample_dicts):
        gold_doc_ids = [str(x) for x in sample.get("gold_doc_ids", [])]
        prev_official = None
        prev_current = None
        for rec in sample.get("iter_records", []):
            metrics = dict(rec.get("metrics", {}) or {})
            official_ndcg = float(metrics.get("nDCG@10", float("nan")))
            current_ndcg = compute_ndcg_doc_ids(
                [str(x) for x in rec.get("local_doc_ids", [])],
                gold_doc_ids,
                k=10,
            )
            rows.append(
                {
                    "subset": subset,
                    "sample_idx": int(sample_idx),
                    "iter": int(rec.get("iter", -1)),
                    "explore": bool(rec.get("explore_effective", False)),
                    "official_ndcg": official_ndcg,
                    "current_ndcg": current_ndcg,
                    "fusion_delta": official_ndcg - current_ndcg,
                    "official_step_delta": (
                        official_ndcg - prev_official if prev_official is not None else float("nan")
                    ),
                    "current_step_delta": (
                        current_ndcg - prev_current if prev_current is not None else float("nan")
                    ),
                }
            )
            prev_official = official_ndcg
            prev_current = current_ndcg
    return rows


def summarize_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for subset, subset_df in df.groupby("subset"):
        for explore_value in [False, True]:
            view = subset_df[subset_df["explore"] == explore_value]
            if view.empty:
                continue
            official_step = view["official_step_delta"].dropna()
            current_step = view["current_step_delta"].dropna()
            rows.append(
                {
                    "subset": subset,
                    "explore": bool(explore_value),
                    "n": int(len(view)),
                    "official_mean": float(view["official_ndcg"].mean()),
                    "current_mean": float(view["current_ndcg"].mean()),
                    "fusion_delta_mean": float(view["fusion_delta"].mean()),
                    "official_step_delta_mean": float(official_step.mean()) if not official_step.empty else float("nan"),
                    "current_step_delta_mean": float(current_step.mean()) if not current_step.empty else float("nan"),
                    "official_step_drop_rate": float((official_step < 0).mean()) if not official_step.empty else float("nan"),
                    "current_step_drop_rate": float((current_step < 0).mean()) if not current_step.empty else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def summarize_overall(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for explore_value in [False, True]:
        view = df[df["explore"] == explore_value]
        official_step = view["official_step_delta"].dropna()
        current_step = view["current_step_delta"].dropna()
        rows.append(
            {
                "explore": bool(explore_value),
                "n": int(len(view)),
                "official_mean": float(view["official_ndcg"].mean()),
                "current_mean": float(view["current_ndcg"].mean()),
                "fusion_delta_mean": float(view["fusion_delta"].mean()),
                "official_step_delta_mean": float(official_step.mean()) if not official_step.empty else float("nan"),
                "current_step_delta_mean": float(current_step.mean()) if not current_step.empty else float("nan"),
                "official_step_drop_rate": float((official_step < 0).mean()) if not official_step.empty else float("nan"),
                "current_step_drop_rate": float((current_step < 0).mean()) if not current_step.empty else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze round6 method2-RRF explore vs fusion behavior.")
    parser.add_argument("--base_dir", type=str, default="results/BRIGHT", help="Base BRIGHT results directory.")
    parser.add_argument("--include_subsets", nargs="*", default=[], help="Optional subset allowlist.")
    parser.add_argument("--exclude_subsets", nargs="*", default=[], help="Optional subset denylist.")
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="results/BRIGHT/analysis/round6_rrf_explore_vs_fusion",
        help="Prefix for CSV outputs.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    pkl_paths = iter_rrf_sample_pkls(base_dir, args.include_subsets, args.exclude_subsets)
    if not pkl_paths:
        raise FileNotFoundError("No method2_rrf all_eval_sample_dicts.pkl files found for the requested subsets.")

    row_records: List[Dict] = []
    for pkl_path in pkl_paths:
        subset = pkl_path.parts[2]
        with open(pkl_path, "rb") as f:
            sample_dicts = pkl.load(f)
        row_records.extend(build_row_records(sample_dicts, subset))

    row_df = pd.DataFrame(row_records)
    subset_summary_df = summarize_rows(row_df)
    overall_summary_df = summarize_overall(row_df)

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    row_df.to_csv(f"{output_prefix}_rows.csv", index=False)
    subset_summary_df.to_csv(f"{output_prefix}_subset_summary.csv", index=False)
    overall_summary_df.to_csv(f"{output_prefix}_overall_summary.csv", index=False)

    print(f"Saved row-level analysis to {output_prefix}_rows.csv")
    print(f"Saved subset summary to {output_prefix}_subset_summary.csv")
    print(f"Saved overall summary to {output_prefix}_overall_summary.csv")
    print()
    print("Overall summary")
    print(overall_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
