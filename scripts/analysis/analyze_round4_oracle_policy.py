#!/usr/bin/env python3
import argparse
import json
import math
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class EventRow:
    sample_idx: int
    iter_idx: int
    decision_mode: str
    num_support: int
    num_selected_categories: int
    top1: float
    top2: float
    margin: float
    relative_margin: float
    anchor_delta: Optional[float]


def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _resolve_metrics_path(samples_path: str, explicit_metrics_path: Optional[str]) -> Optional[str]:
    if explicit_metrics_path:
        return explicit_metrics_path if os.path.exists(explicit_metrics_path) else None
    candidate = samples_path.replace("all_eval_sample_dicts.pkl", "all_eval_metrics.pkl")
    return candidate if os.path.exists(candidate) else None


def _extract_anchor_by_iter(metrics_df: Optional[pd.DataFrame], sample_idx: int, iter_idx: int) -> Optional[float]:
    if metrics_df is None:
        return None
    col = (f"Iter {iter_idx}", "nDCG@10")
    if col not in metrics_df.columns:
        return None
    return float(metrics_df.iloc[sample_idx][col])


def build_event_rows(samples: List[Dict], metrics_df: Optional[pd.DataFrame]) -> List[EventRow]:
    rows: List[EventRow] = []
    for sample_idx, sample in enumerate(samples):
        records = sample.get("iter_records") or []
        for iter_idx in range(1, len(records)):
            rec = records[iter_idx]
            decision_mode = str(rec.get("query_category_decision_mode") or "")
            support_scores = rec.get("query_category_support_scores") or {}
            if not isinstance(support_scores, dict):
                support_scores = {}
            values = sorted([float(v) for v in support_scores.values()], reverse=True)
            if len(values) >= 2:
                top1 = values[0]
                top2 = values[1]
            elif len(values) == 1:
                top1 = values[0]
                top2 = values[0]
            else:
                top1 = 0.0
                top2 = 0.0
            margin = top1 - top2
            relative_margin = margin / max(abs(top1), 1e-9)
            anchor_curr = _extract_anchor_by_iter(metrics_df, sample_idx, iter_idx)
            anchor_prev = _extract_anchor_by_iter(metrics_df, sample_idx, iter_idx - 1)
            anchor_delta = None if anchor_curr is None or anchor_prev is None else anchor_curr - anchor_prev
            rows.append(
                EventRow(
                    sample_idx=sample_idx,
                    iter_idx=iter_idx,
                    decision_mode=decision_mode,
                    num_support=len(values),
                    num_selected_categories=len(rec.get("query_categories") or []),
                    top1=top1,
                    top2=top2,
                    margin=margin,
                    relative_margin=relative_margin,
                    anchor_delta=anchor_delta,
                )
            )
    return rows


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    acc = (tp + tn) / max(len(y_true), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    if prec + rec == 0:
        f1 = 0.0
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return {
        "n": int(len(y_true)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float(acc),
        "precision_exploit": float(prec),
        "recall_exploit": float(rec),
        "f1_exploit": float(f1),
        "pred_exploit_rate": float(np.mean(y_pred == 1)),
        "true_exploit_rate": float(np.mean(y_true == 1)),
    }


def fit_margin_threshold(
    oracle_rows: List[EventRow],
    use_relative_margin: bool,
    min_support_categories: int,
) -> Tuple[float, Dict[str, float], List[Dict[str, float]]]:
    train = [
        r for r in oracle_rows
        if r.decision_mode in {"explore", "exploit"} and r.num_support >= min_support_categories
    ]
    if not train:
        raise ValueError("No oracle rows available for threshold fitting.")

    margin_values = np.array(
        [r.relative_margin if use_relative_margin else r.margin for r in train],
        dtype=float,
    )
    y_true = np.array([1 if r.decision_mode == "exploit" else 0 for r in train], dtype=int)

    candidates = sorted(set(float(v) for v in margin_values))
    # Intent: include boundaries so threshold search can also represent always-explore / always-exploit cases.
    candidates = [min(candidates) - 1e-12] + candidates + [max(candidates) + 1e-12]

    grid_rows: List[Dict[str, float]] = []
    best_tau = candidates[0]
    best_metrics = None
    for tau in candidates:
        y_pred = (margin_values >= tau).astype(int)
        metrics = _binary_metrics(y_true, y_pred)
        row = {"tau": float(tau), **metrics}
        grid_rows.append(row)
        if best_metrics is None:
            best_tau, best_metrics = tau, row
            continue
        if row["f1_exploit"] > best_metrics["f1_exploit"] + 1e-12:
            best_tau, best_metrics = tau, row
            continue
        if abs(row["f1_exploit"] - best_metrics["f1_exploit"]) <= 1e-12:
            if abs(row["pred_exploit_rate"] - row["true_exploit_rate"]) < abs(
                best_metrics["pred_exploit_rate"] - best_metrics["true_exploit_rate"]
            ):
                best_tau, best_metrics = tau, row

    return float(best_tau), best_metrics, grid_rows


def evaluate_threshold_on_rows(
    rows: List[EventRow],
    tau: float,
    use_relative_margin: bool,
    min_support_categories: int,
) -> Dict[str, float]:
    eval_rows = [r for r in rows if r.num_support >= min_support_categories]
    if not eval_rows:
        return {"n": 0}
    margins = np.array(
        [r.relative_margin if use_relative_margin else r.margin for r in eval_rows],
        dtype=float,
    )
    pred_exploit = margins >= tau

    anchor_deltas = np.array(
        [np.nan if r.anchor_delta is None else float(r.anchor_delta) for r in eval_rows],
        dtype=float,
    )
    expl_mask = pred_exploit
    expr = anchor_deltas[expl_mask]
    expro = anchor_deltas[~expl_mask]
    return {
        "n": int(len(eval_rows)),
        "pred_exploit_rate": float(np.mean(pred_exploit)),
        "pred_exploit_count": int(np.sum(pred_exploit)),
        "pred_explore_count": int(np.sum(~pred_exploit)),
        "mean_anchor_delta_if_pred_exploit": float(np.nanmean(expr)) if expr.size else float("nan"),
        "mean_anchor_delta_if_pred_explore": float(np.nanmean(expro)) if expro.size else float("nan"),
        "win_rate_if_pred_exploit": float(np.nanmean(expr > 0)) if expr.size else float("nan"),
        "win_rate_if_pred_explore": float(np.nanmean(expro > 0)) if expro.size else float("nan"),
        "tie_rate_if_pred_exploit": float(np.nanmean(expr == 0)) if expr.size else float("nan"),
        "tie_rate_if_pred_explore": float(np.nanmean(expro == 0)) if expro.size else float("nan"),
    }


def summarize_ndcg_consistency(samples: List[Dict], metrics_df: Optional[pd.DataFrame]) -> Dict[str, Dict[str, float]]:
    if metrics_df is None:
        return {"note": {"message": "metrics_df unavailable"}}

    max_iter = max(len(s.get("iter_records") or []) for s in samples)
    out: Dict[str, Dict[str, float]] = {}
    for iter_idx in range(max_iter):
        rec_global = []
        rec_local = []
        df_anchor = []
        df_global = []
        df_local = []
        for sample_idx, sample in enumerate(samples):
            recs = sample.get("iter_records") or []
            if iter_idx >= len(recs):
                continue
            rec = recs[iter_idx]
            rec_global.append(float((rec.get("global_metrics") or {}).get("nDCG@10")))
            rec_local.append(float((rec.get("local_metrics") or {}).get("nDCG@10")))
            c_anchor = (f"Iter {iter_idx}", "nDCG@10")
            c_global = (f"Iter {iter_idx}", "Global_nDCG@10")
            c_local = (f"Iter {iter_idx}", "Local_nDCG@10")
            if c_anchor in metrics_df.columns:
                df_anchor.append(float(metrics_df.iloc[sample_idx][c_anchor]))
            if c_global in metrics_df.columns:
                df_global.append(float(metrics_df.iloc[sample_idx][c_global]))
            if c_local in metrics_df.columns:
                df_local.append(float(metrics_df.iloc[sample_idx][c_local]))
        out[f"Iter {iter_idx}"] = {
            "anchor_mean_from_metrics_df": float(np.mean(df_anchor)) if df_anchor else float("nan"),
            "global_mean_from_iter_records": float(np.mean(rec_global)) if rec_global else float("nan"),
            "global_mean_from_metrics_df": float(np.mean(df_global)) if df_global else float("nan"),
            "local_mean_from_iter_records": float(np.mean(rec_local)) if rec_local else float("nan"),
            "local_mean_from_metrics_df": float(np.mean(df_local)) if df_local else float("nan"),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="Analyze oracle support-score signal for round4 category policy.")
    parser.add_argument("--oracle_samples", type=str, required=True, help="Path to oracle all_eval_sample_dicts.pkl")
    parser.add_argument("--real_samples", type=str, required=True, help="Path to real all_eval_sample_dicts.pkl")
    parser.add_argument("--oracle_metrics", type=str, default=None, help="Path to oracle all_eval_metrics.pkl")
    parser.add_argument("--real_metrics", type=str, default=None, help="Path to real all_eval_metrics.pkl")
    parser.add_argument("--min_support_categories", type=int, default=2, help="Minimum categories required to include an event")
    parser.add_argument("--use_relative_margin", default=False, action="store_true", help="Fit threshold on relative margin instead of raw margin")
    parser.add_argument("--out_json", type=str, default=None, help="Optional output JSON path")
    args = parser.parse_args()

    oracle_samples = _load_pickle(args.oracle_samples)
    real_samples = _load_pickle(args.real_samples)
    oracle_metrics_path = _resolve_metrics_path(args.oracle_samples, args.oracle_metrics)
    real_metrics_path = _resolve_metrics_path(args.real_samples, args.real_metrics)
    oracle_metrics_df = _load_pickle(oracle_metrics_path) if oracle_metrics_path else None
    real_metrics_df = _load_pickle(real_metrics_path) if real_metrics_path else None

    oracle_rows = build_event_rows(oracle_samples, oracle_metrics_df)
    real_rows = build_event_rows(real_samples, real_metrics_df)

    eligible_oracle_with_support = [
        r
        for r in oracle_rows
        if r.decision_mode in {"explore", "exploit"} and r.num_support >= args.min_support_categories
    ]

    training_source = "oracle_support_scores"
    fit_rows = eligible_oracle_with_support
    if not fit_rows:
        # Intent: oracle run lacks support-score logging; fall back to aligned real support + oracle decision labels.
        real_map = {(r.sample_idx, r.iter_idx): r for r in real_rows}
        paired_rows: List[EventRow] = []
        for o in oracle_rows:
            if o.decision_mode not in {"explore", "exploit"}:
                continue
            key = (o.sample_idx, o.iter_idx)
            if key not in real_map:
                continue
            rr = real_map[key]
            if rr.num_support < args.min_support_categories:
                continue
            paired_rows.append(
                EventRow(
                    sample_idx=o.sample_idx,
                    iter_idx=o.iter_idx,
                    decision_mode=o.decision_mode,
                    num_support=rr.num_support,
                    num_selected_categories=rr.num_selected_categories,
                    top1=rr.top1,
                    top2=rr.top2,
                    margin=rr.margin,
                    relative_margin=rr.relative_margin,
                    anchor_delta=rr.anchor_delta,
                )
            )
        if not paired_rows:
            raise ValueError(
                "No train rows available: oracle has no support scores and no aligned real support rows were found."
            )
        training_source = "paired_real_support_with_oracle_labels"
        fit_rows = paired_rows

    tau, oracle_fit, tau_grid = fit_margin_threshold(
        oracle_rows=fit_rows,
        use_relative_margin=args.use_relative_margin,
        min_support_categories=args.min_support_categories,
    )
    train_eval = evaluate_threshold_on_rows(
        rows=fit_rows,
        tau=tau,
        use_relative_margin=args.use_relative_margin,
        min_support_categories=args.min_support_categories,
    )
    real_eval = evaluate_threshold_on_rows(
        rows=real_rows,
        tau=tau,
        use_relative_margin=args.use_relative_margin,
        min_support_categories=args.min_support_categories,
    )

    oracle_actual_labels = [r.decision_mode == "exploit" for r in oracle_rows if r.decision_mode in {"explore", "exploit"}]
    real_actual_labels = [r.decision_mode == "exploit" for r in real_rows if r.decision_mode in {"explore", "exploit"}]
    oracle_actual_exploit_rate = float(np.mean(oracle_actual_labels)) if oracle_actual_labels else None
    real_actual_exploit_rate = float(np.mean(real_actual_labels)) if real_actual_labels else None

    ndcg_check_oracle = summarize_ndcg_consistency(oracle_samples, oracle_metrics_df)
    ndcg_check_real = summarize_ndcg_consistency(real_samples, real_metrics_df)

    result = {
        "inputs": {
            "oracle_samples": args.oracle_samples,
            "real_samples": args.real_samples,
            "oracle_metrics": oracle_metrics_path,
            "real_metrics": real_metrics_path,
            "min_support_categories": args.min_support_categories,
            "margin_type": "relative" if args.use_relative_margin else "raw",
        },
        "fitted_policy_from_oracle": {
            "training_source": training_source,
            "tau": tau,
            "fit_metrics_on_oracle_labels": oracle_fit,
            "oracle_actual_exploit_rate": oracle_actual_exploit_rate,
            "real_actual_exploit_rate": real_actual_exploit_rate,
        },
        "threshold_eval": {
            "train_eval": train_eval,
            "real_eval": real_eval,
        },
        "ndcg_consistency_check": {
            "oracle": ndcg_check_oracle,
            "real": ndcg_check_real,
        },
        "tau_grid_top5_by_f1": sorted(tau_grid, key=lambda x: x["f1_exploit"], reverse=True)[:5],
    }

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[saved] {args.out_json}")

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
