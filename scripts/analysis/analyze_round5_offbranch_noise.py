import argparse
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class SampleIterRow:
    subset: str
    run_path: str
    sample_idx: int
    iter_idx: int
    ndcg10: float
    off_branch_pct: float
    selected_branch_count: int
    local_doc_count: int


def _is_prefix(prefix: Sequence[int], full: Sequence[int]) -> bool:
    p = tuple(prefix)
    f = tuple(full)
    return len(p) <= len(f) and f[: len(p)] == p


def _is_descendant_of_selected(path: Sequence[int], selected_branches: Sequence[Sequence[int]]) -> bool:
    for branch in selected_branches:
        if _is_prefix(branch, path):
            return True
    return False


def _extract_subset_from_path(sample_pkl_path: str) -> str:
    parts = Path(sample_pkl_path).parts
    if "BRIGHT" in parts:
        idx = parts.index("BRIGHT")
        if idx + 1 < len(parts):
            return str(parts[idx + 1])
    return "unknown"


def _load_target_subsets_from_ndcg_csv(csv_path: str) -> List[str]:
    if not os.path.exists(csv_path):
        return []
    raw = pd.read_csv(csv_path, nrows=0)
    cols = [str(c).strip() for c in raw.columns]
    subsets = [c for c in cols if c and c.lower() != "experiment"]
    return sorted(set(subsets))


def _discover_run_paths(
    glob_pattern: str,
    run_contains: Sequence[str],
    target_subsets: Optional[Sequence[str]],
) -> List[str]:
    matched = [str(p) for p in Path(".").glob(glob_pattern)]
    out: List[str] = []
    target_subset_set = set(target_subsets or [])
    for path in matched:
        if run_contains and not all(token in path for token in run_contains):
            continue
        subset = _extract_subset_from_path(path)
        if target_subset_set and subset not in target_subset_set:
            continue
        out.append(path)
    return sorted(out)


def _iter_rows_from_samples(
    subset: str,
    run_path: str,
    samples: Sequence[Dict],
    top_k: int,
    selected_branch_field: str,
) -> List[SampleIterRow]:
    rows: List[SampleIterRow] = []
    for sample_idx, sample in enumerate(samples):
        iter_records = sample.get("iter_records", []) or []
        for iter_idx, rec in enumerate(iter_records):
            metrics = rec.get("metrics", {}) or {}
            ndcg10 = float(metrics.get("nDCG@10", np.nan))
            selected = [tuple(x) for x in rec.get(selected_branch_field, []) if x]
            local_paths = [tuple(x) for x in rec.get("local_paths", []) if x]
            local_topk = local_paths[: max(1, int(top_k))]

            if (not selected) or (not local_topk):
                off_branch_pct = float(np.nan)
            else:
                # Intent: off-branch is defined against currently selected branch descendants at the same iteration.
                off_count = 0
                for path in local_topk:
                    if not _is_descendant_of_selected(path, selected):
                        off_count += 1
                off_branch_pct = 100.0 * float(off_count) / float(len(local_topk))

            rows.append(
                SampleIterRow(
                    subset=subset,
                    run_path=run_path,
                    sample_idx=int(sample_idx),
                    iter_idx=int(iter_idx),
                    ndcg10=float(ndcg10),
                    off_branch_pct=float(off_branch_pct),
                    selected_branch_count=int(len(selected)),
                    local_doc_count=int(len(local_topk)),
                )
            )
    return rows


def _build_transition_rows(df_rows: pd.DataFrame, off_threshold_pct: float) -> pd.DataFrame:
    transitions: List[Dict] = []
    for (subset, run_path, sample_idx), grp in df_rows.groupby(["subset", "run_path", "sample_idx"], dropna=False):
        grp = grp.sort_values("iter_idx")
        ndcg_vals = grp["ndcg10"].tolist()
        off_vals = grp["off_branch_pct"].tolist()
        iter_vals = grp["iter_idx"].tolist()
        for i in range(len(grp) - 1):
            ndcg_t = float(ndcg_vals[i])
            ndcg_next = float(ndcg_vals[i + 1])
            off_t = float(off_vals[i]) if not pd.isna(off_vals[i]) else np.nan
            if pd.isna(ndcg_t) or pd.isna(ndcg_next):
                continue
            transitions.append(
                {
                    "subset": subset,
                    "run_path": run_path,
                    "sample_idx": int(sample_idx),
                    "iter_t": int(iter_vals[i]),
                    "off_branch_pct_t": float(off_t),
                    "off_event_t": bool((not pd.isna(off_t)) and (off_t > float(off_threshold_pct))),
                    "ndcg10_t": float(ndcg_t),
                    "ndcg10_t1": float(ndcg_next),
                    "ndcg10_delta_t1_minus_t": float(ndcg_next - ndcg_t),
                }
            )
    return pd.DataFrame(transitions)


def _safe_mean(series: pd.Series) -> float:
    if series.empty:
        return float(np.nan)
    return float(series.mean())


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    joined = pd.concat([a, b], axis=1).dropna()
    if len(joined) < 2:
        return float(np.nan)
    return float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))


def _summarize_runs(df_rows: pd.DataFrame, df_trans: pd.DataFrame) -> pd.DataFrame:
    out_rows: List[Dict] = []
    key_cols = ["subset", "run_path"]
    for (subset, run_path), grp in df_rows.groupby(key_cols, dropna=False):
        trans = df_trans[(df_trans["subset"] == subset) & (df_trans["run_path"] == run_path)]
        off = trans[trans["off_event_t"]]
        on = trans[~trans["off_event_t"]]
        out_rows.append(
            {
                "subset": subset,
                "run_path": run_path,
                "num_sample_iter_rows": int(len(grp)),
                "num_transition_rows": int(len(trans)),
                "OffBranchPct@K_mean": _safe_mean(grp["off_branch_pct"]),
                "OffBranchPct@K_median": float(grp["off_branch_pct"].median()) if len(grp) else float(np.nan),
                "OffBranchEventRate(>0%)": 100.0 * _safe_mean((grp["off_branch_pct"] > 0.0).astype(float)),
                "nDCG10_mean": _safe_mean(grp["ndcg10"]),
                "nDCG10_delta_mean_all": _safe_mean(trans["ndcg10_delta_t1_minus_t"]),
                "nDCG10_delta_mean_when_off": _safe_mean(off["ndcg10_delta_t1_minus_t"]),
                "nDCG10_delta_mean_when_onbranch_only": _safe_mean(on["ndcg10_delta_t1_minus_t"]),
                # Intent: negative value here means off-branch feedback is associated with larger next-iter nDCG drop.
                "nDCG10_drop_delta(off-onbranch_only)": _safe_mean(off["ndcg10_delta_t1_minus_t"])
                - _safe_mean(on["ndcg10_delta_t1_minus_t"]),
                "corr(off_branch_pct_t, ndcg10_delta_t1_minus_t)": _safe_corr(
                    trans["off_branch_pct_t"],
                    trans["ndcg10_delta_t1_minus_t"],
                ),
            }
        )
    return pd.DataFrame(out_rows).sort_values(["subset", "run_path"]).reset_index(drop=True)


def _summarize_by_iter(df_rows: pd.DataFrame) -> pd.DataFrame:
    out = (
        df_rows.groupby(["subset", "run_path", "iter_idx"], as_index=False)
        .agg(
            OffBranchPct_mean=("off_branch_pct", "mean"),
            OffBranchPct_median=("off_branch_pct", "median"),
            nDCG10_mean=("ndcg10", "mean"),
            NumSampleRows=("sample_idx", "count"),
        )
        .sort_values(["subset", "run_path", "iter_idx"])
        .reset_index(drop=True)
    )
    return out


def _to_dataframe(rows: Iterable[SampleIterRow]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subset": r.subset,
                "run_path": r.run_path,
                "sample_idx": r.sample_idx,
                "iter_idx": r.iter_idx,
                "ndcg10": r.ndcg10,
                "off_branch_pct": r.off_branch_pct,
                "selected_branch_count": r.selected_branch_count,
                "local_doc_count": r.local_doc_count,
            }
            for r in rows
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze off-branch retrieval noise and next-iter nDCG drop in round5 runs.")
    parser.add_argument(
        "--glob_pattern",
        type=str,
        default="results/BRIGHT/*/round5/**/all_eval_sample_dicts.pkl",
        help="Glob pattern for round5 sample pickles.",
    )
    parser.add_argument(
        "--run_contains",
        type=str,
        nargs="*",
        default=["round5_mrr_selector_accum_retriever_slate", "RPN=agent_executor_v1"],
        help="All tokens that must exist in path.",
    )
    parser.add_argument(
        "--ndcg_summary_csv",
        type=str,
        default="results/BRIGHT/ndcg_summary_leaf.csv",
        help="CSV used only to derive target subset list.",
    )
    parser.add_argument("--top_k", type=int, default=10, help="Top-K cutoff for Off-branch percentage.")
    parser.add_argument(
        "--selected_branch_field",
        type=str,
        default="selected_branches_before",
        choices=["selected_branches_before", "selected_branches_after"],
        help="Which selected-branch snapshot to use as reference for off-branch check.",
    )
    parser.add_argument(
        "--off_threshold_pct",
        type=float,
        default=0.0,
        help="Threshold for defining off-branch event at iter t.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/BRIGHT/analysis/round5_offbranch",
        help="Output directory for CSV artifacts.",
    )
    args = parser.parse_args()

    target_subsets = _load_target_subsets_from_ndcg_csv(args.ndcg_summary_csv)
    run_paths = _discover_run_paths(
        glob_pattern=args.glob_pattern,
        run_contains=args.run_contains,
        target_subsets=target_subsets if target_subsets else None,
    )

    if not run_paths:
        raise FileNotFoundError(
            "No run paths found. "
            f"glob_pattern={args.glob_pattern}, run_contains={args.run_contains}, target_subsets={target_subsets}"
        )

    all_rows: List[SampleIterRow] = []
    for p in run_paths:
        subset = _extract_subset_from_path(p)
        with open(p, "rb") as f:
            samples = pickle.load(f)
        all_rows.extend(
            _iter_rows_from_samples(
                subset=subset,
                run_path=p,
                samples=samples,
                top_k=int(args.top_k),
                selected_branch_field=str(args.selected_branch_field),
            )
        )

    df_rows = _to_dataframe(all_rows)
    df_trans = _build_transition_rows(df_rows, off_threshold_pct=float(args.off_threshold_pct))
    df_run = _summarize_runs(df_rows, df_trans)
    df_iter = _summarize_by_iter(df_rows)

    os.makedirs(args.out_dir, exist_ok=True)
    field_suffix = str(args.selected_branch_field).replace("selected_branches_", "")
    rows_csv = os.path.join(args.out_dir, f"offbranch_rows_top{int(args.top_k)}_{field_suffix}.csv")
    trans_csv = os.path.join(args.out_dir, f"offbranch_transitions_top{int(args.top_k)}_{field_suffix}.csv")
    run_csv = os.path.join(args.out_dir, f"offbranch_summary_top{int(args.top_k)}_{field_suffix}.csv")
    iter_csv = os.path.join(args.out_dir, f"offbranch_by_iter_top{int(args.top_k)}_{field_suffix}.csv")
    df_rows.to_csv(rows_csv, index=False)
    df_trans.to_csv(trans_csv, index=False)
    df_run.to_csv(run_csv, index=False)
    df_iter.to_csv(iter_csv, index=False)

    show_cols = [
        "subset",
        "OffBranchPct@K_mean",
        "OffBranchEventRate(>0%)",
        "nDCG10_mean",
        "nDCG10_delta_mean_when_off",
        "nDCG10_delta_mean_when_onbranch_only",
        "nDCG10_drop_delta(off-onbranch_only)",
        "corr(off_branch_pct_t, ndcg10_delta_t1_minus_t)",
    ]
    show_cols = [c for c in show_cols if c in df_run.columns]
    print(df_run[show_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved:\n- {run_csv}\n- {iter_csv}\n- {rows_csv}\n- {trans_csv}")


if __name__ == "__main__":
    main()
