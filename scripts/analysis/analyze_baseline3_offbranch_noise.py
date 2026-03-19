import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def _is_prefix(prefix: Sequence[int], full: Sequence[int]) -> bool:
    p = tuple(prefix)
    f = tuple(full)
    return len(p) <= len(f) and f[: len(p)] == p


def _extract_subset(path: str) -> str:
    parts = path.split("/")
    if "BRIGHT" in parts:
        idx = parts.index("BRIGHT")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "unknown"


def _load_iter_records(path: str) -> Dict[int, Dict[int, Dict]]:
    by_q: Dict[int, Dict[int, Dict]] = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qidx = int(rec.get("query_idx", -1))
            if qidx < 0:
                continue
            phase = str(rec.get("phase", ""))
            if phase == "initial_rewrite":
                by_q[qidx][-1] = rec
            elif phase == "iter_retrieval":
                iter_idx = int(rec.get("iter", -1))
                if iter_idx >= 0:
                    by_q[qidx][iter_idx] = rec
    return by_q


def _load_ndcg_metrics(path: str) -> Dict[int, Dict[int, float]]:
    by_q: Dict[int, Dict[int, float]] = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qidx = int(rec.get("query_idx", -1))
            iter_idx = int(rec.get("iter", -1))
            ndcg = float(rec.get("nDCG@10", np.nan))
            if qidx >= 0 and iter_idx >= 0:
                by_q[qidx][iter_idx] = ndcg
    return by_q


def _context_prefix_branches(context_paths: Sequence[Tuple[int, ...]], depth: int) -> List[Tuple[int, ...]]:
    branches = []
    seen = set()
    for path in context_paths:
        if len(path) < 2:
            continue
        d = min(int(depth), len(path) - 1)
        if d <= 0:
            continue
        branch = tuple(path[:d])
        if branch in seen:
            continue
        seen.add(branch)
        branches.append(branch)
    return branches


def _analyze_run(
    records_path: str,
    metrics_path: str,
    top_k: int,
    context_branch_depth: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    iter_records = _load_iter_records(records_path)
    ndcg_by_q = _load_ndcg_metrics(metrics_path)

    rows: List[Dict] = []
    transitions: List[Dict] = []
    subset = _extract_subset(records_path)

    for qidx, by_iter in iter_records.items():
        iter_keys = sorted([k for k in by_iter.keys() if k >= 0])
        for iter_idx in iter_keys:
            prev = by_iter.get(iter_idx - 1)
            cur = by_iter.get(iter_idx)
            if prev is None or cur is None:
                continue

            prev_context_paths = [tuple(x) for x in prev.get("rewrite_context_paths", []) if x]
            retrieved_paths = [tuple(x) for x in cur.get("retrieved_paths", [])[: int(top_k)] if x]
            if (not prev_context_paths) or (not retrieved_paths):
                continue

            selected_branches = _context_prefix_branches(prev_context_paths, depth=int(context_branch_depth))
            if not selected_branches:
                continue

            off_count = 0
            for path in retrieved_paths:
                if not any(_is_prefix(branch, path) for branch in selected_branches):
                    off_count += 1
            off_pct = 100.0 * float(off_count) / float(len(retrieved_paths))

            ndcg_t = ndcg_by_q.get(qidx, {}).get(iter_idx, np.nan)
            rows.append(
                {
                    "subset": subset,
                    "run_path": records_path,
                    "query_idx": int(qidx),
                    "iter": int(iter_idx),
                    "off_branch_pct": float(off_pct),
                    "off_event": bool(off_pct > 0.0),
                    "ndcg10": float(ndcg_t),
                    "selected_branch_count_proxy": int(len(selected_branches)),
                }
            )

            ndcg_t1 = ndcg_by_q.get(qidx, {}).get(iter_idx + 1, np.nan)
            if not np.isnan(ndcg_t) and not np.isnan(ndcg_t1):
                transitions.append(
                    {
                        "subset": subset,
                        "run_path": records_path,
                        "query_idx": int(qidx),
                        "iter_t": int(iter_idx),
                        "off_branch_pct_t": float(off_pct),
                        "off_event_t": bool(off_pct > 0.0),
                        "ndcg10_t": float(ndcg_t),
                        "ndcg10_t1": float(ndcg_t1),
                        "ndcg10_delta_t1_minus_t": float(ndcg_t1 - ndcg_t),
                    }
                )

    return pd.DataFrame(rows), pd.DataFrame(transitions)


def _safe_mean(series: pd.Series) -> float:
    if series.empty:
        return float(np.nan)
    return float(series.mean())


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    joined = pd.concat([a, b], axis=1).dropna()
    if len(joined) < 2:
        return float(np.nan)
    return float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))


def _summarize(rows_df: pd.DataFrame, trans_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    out_rows = []
    for (subset, run_path), grp in rows_df.groupby(["subset", "run_path"], dropna=False):
        tr = trans_df[(trans_df["subset"] == subset) & (trans_df["run_path"] == run_path)]
        tr_off = tr[tr["off_event_t"]]
        tr_on = tr[~tr["off_event_t"]]
        d_off = _safe_mean(tr_off["ndcg10_delta_t1_minus_t"])
        d_on = _safe_mean(tr_on["ndcg10_delta_t1_minus_t"])
        out_rows.append(
            {
                "subset": subset,
                "run_path": run_path,
                f"OffBranchPct@{top_k}_mean_proxy": _safe_mean(grp["off_branch_pct"]),
                f"OffBranchEventRate@{top_k}_proxy": 100.0 * _safe_mean(grp["off_event"].astype(float)),
                "nDCG10_delta_mean_when_off_proxy": d_off,
                "nDCG10_delta_mean_when_onbranch_only_proxy": d_on,
                # Intent: negative means off-branch feedback is associated with a larger next-iter nDCG decrease.
                "nDCG10_drop_delta_proxy(off-on)": d_off - d_on,
                "corr(off_pct, ndcg_delta)_proxy": _safe_corr(
                    tr["off_branch_pct_t"],
                    tr["ndcg10_delta_t1_minus_t"],
                ),
                "num_rows": int(len(grp)),
                "num_transitions": int(len(tr)),
            }
        )
    return pd.DataFrame(out_rows).sort_values(["subset", "run_path"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze baseline3 off-branch noise with a proxy branch definition "
            "(previous iter rewrite-context prefix branches)."
        )
    )
    parser.add_argument(
        "--records_glob",
        type=str,
        default="results/BRIGHT/**/leaf_iter_records.jsonl",
        help="Glob for baseline3 iteration records.",
    )
    parser.add_argument(
        "--run_contains",
        type=str,
        nargs="*",
        default=["baseline3_leaf_only_loop"],
        help="Path tokens all required to select runs.",
    )
    parser.add_argument("--top_k", type=int, default=10, help="Top-K retrieval cutoff for off-branch ratio.")
    parser.add_argument(
        "--context_branch_depth",
        type=int,
        default=3,
        help="Prefix depth used to convert previous rewrite-context paths into branch proxies.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/BRIGHT/analysis/baseline3_offbranch",
        help="Directory for output CSVs.",
    )
    args = parser.parse_args()

    all_rows = []
    all_transitions = []

    record_paths = sorted(glob.glob(args.records_glob, recursive=True))
    selected_paths = []
    for p in record_paths:
        if args.run_contains and not all(token in p for token in args.run_contains):
            continue
        metrics_path = p.replace("leaf_iter_records.jsonl", "leaf_iter_metrics.jsonl")
        if not os.path.exists(metrics_path):
            continue
        selected_paths.append((p, metrics_path))

    if not selected_paths:
        raise FileNotFoundError(
            f"No matching runs. records_glob={args.records_glob}, run_contains={args.run_contains}"
        )

    for records_path, metrics_path in selected_paths:
        rows_df, trans_df = _analyze_run(
            records_path=records_path,
            metrics_path=metrics_path,
            top_k=int(args.top_k),
            context_branch_depth=int(args.context_branch_depth),
        )
        if not rows_df.empty:
            all_rows.append(rows_df)
        if not trans_df.empty:
            all_transitions.append(trans_df)

    if not all_rows:
        raise RuntimeError("No analyzable rows produced from selected runs.")

    rows_df = pd.concat(all_rows, ignore_index=True)
    trans_df = pd.concat(all_transitions, ignore_index=True) if all_transitions else pd.DataFrame()
    summary_df = _summarize(rows_df, trans_df, top_k=int(args.top_k))

    os.makedirs(args.out_dir, exist_ok=True)
    summary_csv = os.path.join(
        args.out_dir,
        f"offbranch_proxy_summary_top{int(args.top_k)}_prevctx_depth{int(args.context_branch_depth)}.csv",
    )
    rows_csv = os.path.join(
        args.out_dir,
        f"offbranch_proxy_rows_top{int(args.top_k)}_prevctx_depth{int(args.context_branch_depth)}.csv",
    )
    trans_csv = os.path.join(
        args.out_dir,
        f"offbranch_proxy_transitions_top{int(args.top_k)}_prevctx_depth{int(args.context_branch_depth)}.csv",
    )
    summary_df.to_csv(summary_csv, index=False)
    rows_df.to_csv(rows_csv, index=False)
    trans_df.to_csv(trans_csv, index=False)

    print(
        summary_df[
            [
                "subset",
                f"OffBranchPct@{int(args.top_k)}_mean_proxy",
                f"OffBranchEventRate@{int(args.top_k)}_proxy",
                "nDCG10_drop_delta_proxy(off-on)",
                "corr(off_pct, ndcg_delta)_proxy",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )
    print(f"\nSaved:\n- {summary_csv}\n- {rows_csv}\n- {trans_csv}")


if __name__ == "__main__":
    main()
