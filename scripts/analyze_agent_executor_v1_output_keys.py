#!/usr/bin/env python3
"""
Analyze Round5 rewrite output-key behavior and retrieval performance by prompt.

What this script computes
-------------------------
1) Subset-level comparison between two rewrite prompts (default: v1 vs v1_icl2)
   - best/last nDCG@10 from all_eval_metrics.pkl
   - key statistics from llm_api_history.pkl
     (mean key count, unique key count, non-standard key ratio)

2) Query-level paired analysis (same subset + same original_query)
   - delta_best = best_ndcg(prompt_b) - best_ndcg(prompt_a)
   - relation with non-standard key appearance and key-count signals

3) Optional qualitative examples and non-standard-key frequency tables
   for prompt_b (default: agent_executor_v1_icl2).

How to run
----------
# Default (v1 vs v1_icl2, 5 subsets, run_tag=meanscore_global, RCT=10)
python scripts/analyze_agent_executor_v1_output_keys.py

# Compare v1 vs v1_icl, and print prompt_b examples/frequency
python scripts/analyze_agent_executor_v1_output_keys.py \
    --prompt_b agent_executor_v1_icl \
    --show_examples \
    --show_nonstd_freq

# Explicit base dir + custom subset list
python scripts/analyze_agent_executor_v1_output_keys.py \
    --base_results_dir /data4/jongho/lattice/results/BRIGHT \
    --subsets biology earth_science economics psychology robotics
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from json_repair import repair_json
except Exception:
    repair_json = None


CANONICAL_KEYS = {
    "Theory",
    "Entity",
    "Example",
    "Other",
    "theory",
    "entity",
    "example",
    "other",
}


def extract_json_text(text: str) -> str:
    t = str(text or "").strip()
    if "```" in t:
        parts = t.split("```")
        fenced = [parts[i] for i in range(1, len(parts), 2)]
        if fenced:
            t = fenced[-1].strip()
            if t.lower().startswith("json"):
                t = t[4:].strip()
    if "{" in t and "}" in t:
        t = t[t.find("{") : t.rfind("}") + 1]
    return t


def parse_response_to_obj(response_text: str) -> Optional[Any]:
    t = extract_json_text(response_text)
    try:
        return json.loads(t)
    except Exception:
        if repair_json is not None:
            try:
                return repair_json(t, return_objects=True)
            except Exception:
                return None
        return None


def find_docs_obj(obj: Any) -> Optional[Any]:
    if isinstance(obj, dict):
        if "Possible_Answer_Docs" in obj:
            return obj["Possible_Answer_Docs"]
        if "possible_answer_docs" in obj:
            return obj["possible_answer_docs"]
        for v in obj.values():
            found = find_docs_obj(v)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for it in obj:
            found = find_docs_obj(it)
            if found is not None:
                return found
    return None


def infer_subset_from_path(path: Path) -> str:
    parts = list(path.parts)
    if "BRIGHT" in parts:
        idx = parts.index("BRIGHT")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "unknown"


def resolve_base_results_dir(explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"--base_results_dir not found: {p}")
        return p

    candidates = [
        Path("results/BRIGHT"),
        Path("../results/BRIGHT"),
        Path("/data4/jongho/lattice/results/BRIGHT"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"No results dir found. checked={candidates}")


def prompt_tag(prompt_name: str) -> str:
    return f"RPN={prompt_name}-RM=concat-RE=1"


def pick_latest_files(
    base_dir: Path,
    filename: str,
    subsets: Sequence[str],
    run_tag: str,
    rct_tag: str,
    prompt_name: str,
) -> Dict[str, Path]:
    picked: Dict[str, Path] = {}
    ptag = prompt_tag(prompt_name)
    subset_set = set(subsets)

    for p in base_dir.rglob(filename):
        s = str(p)
        if "/round5/" not in s.replace("\\", "/"):
            continue
        if run_tag not in s or rct_tag not in s or ptag not in s:
            continue
        subset = infer_subset_from_path(p)
        if subset not in subset_set:
            continue

        # Intent: use newest artifact per subset/prompt to avoid stale duplicated runs.
        if subset not in picked or p.stat().st_mtime > picked[subset].stat().st_mtime:
            picked[subset] = p

    return picked


def ndcg_from_metrics(metrics_path: Path) -> Tuple[float, float]:
    if not metrics_path.exists():
        return np.nan, np.nan
    df = pickle.load(open(metrics_path, "rb"))
    if not isinstance(df, pd.DataFrame):
        return np.nan, np.nan

    ndcg_cols = [
        c for c in df.columns
        if isinstance(c, tuple) and len(c) == 2 and c[1] == "nDCG@10"
    ]
    if not ndcg_cols:
        return np.nan, np.nan

    ndcg_df = df[ndcg_cols].astype(float)
    ndcg_best_mean = float(ndcg_df.max(axis=1).mean())
    ndcg_last_mean = float(ndcg_df.iloc[:, -1].mean())
    return ndcg_best_mean, ndcg_last_mean


def analyze_history(history_path: Path) -> Tuple[Dict[str, Any], Counter, Counter]:
    rows = pickle.load(open(history_path, "rb"))

    parsed = 0
    parse_fail = 0
    with_docs = 0
    docs_not_dict = 0
    nonstandard_responses = 0

    key_counter: Counter = Counter()
    keyset_counter: Counter = Counter()
    per_response_key_count: List[int] = []

    for row in rows:
        obj = parse_response_to_obj(row.get("response", ""))
        if obj is None:
            parse_fail += 1
            continue
        parsed += 1

        docs = find_docs_obj(obj)
        if docs is None:
            continue
        with_docs += 1

        if not isinstance(docs, dict):
            docs_not_dict += 1
            continue

        keys = [str(k).strip() for k in docs.keys() if str(k).strip()]
        per_response_key_count.append(len(keys))
        keyset_counter[tuple(sorted(keys))] += 1

        has_nonstandard = False
        for k in keys:
            key_counter[k] += 1
            if k not in CANONICAL_KEYS:
                has_nonstandard = True
        if has_nonstandard:
            nonstandard_responses += 1

    arr = np.array(per_response_key_count, dtype=float) if per_response_key_count else np.array([])

    summary = {
        "n_rows": len(rows),
        "n_parsed_json": parsed,
        "n_parse_fail": parse_fail,
        "n_with_possible_answer_docs": with_docs,
        "n_docs_not_dict": docs_not_dict,
        "unique_key_count": len(key_counter),
        "mean_keys_per_response": float(arr.mean()) if len(arr) else np.nan,
        "median_keys_per_response": float(np.median(arr)) if len(arr) else np.nan,
        "p90_keys_per_response": float(np.percentile(arr, 90)) if len(arr) else np.nan,
        "nonstandard_response_ratio": (nonstandard_responses / len(per_response_key_count)) if per_response_key_count else np.nan,
    }
    return summary, key_counter, keyset_counter


def build_subset_prompt_summary(
    base_dir: Path,
    subsets: Sequence[str],
    run_tag: str,
    rct_tag: str,
    prompt_a: str,
    prompt_b: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[Tuple[str, str], List[Dict[str, Any]]]]:
    summaries: List[Dict[str, Any]] = []
    key_rows: List[Dict[str, Any]] = []
    keyset_map: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    for prompt in [prompt_a, prompt_b]:
        hist_map = pick_latest_files(base_dir, "llm_api_history.pkl", subsets, run_tag, rct_tag, prompt)
        metric_map = pick_latest_files(base_dir, "all_eval_metrics.pkl", subsets, run_tag, rct_tag, prompt)

        for subset in subsets:
            if subset not in hist_map:
                continue
            hist_path = hist_map[subset]
            metric_path = metric_map.get(subset)

            hist_summary, key_counter, keyset_counter = analyze_history(hist_path)
            ndcg_best, ndcg_last = (ndcg_from_metrics(metric_path) if metric_path else (np.nan, np.nan))

            row = {
                "subset": subset,
                "prompt": prompt,
                "history_path": str(hist_path),
                "metrics_path": str(metric_path) if metric_path else "",
                "ndcg_best_mean": ndcg_best,
                "ndcg_last_mean": ndcg_last,
                **hist_summary,
            }
            summaries.append(row)

            total_keys = sum(key_counter.values())
            for k, cnt in key_counter.items():
                key_rows.append(
                    {
                        "subset": subset,
                        "prompt": prompt,
                        "key": k,
                        "count": int(cnt),
                        "ratio_in_all_keys": (cnt / total_keys) if total_keys else np.nan,
                        "is_nonstandard": (k not in CANONICAL_KEYS),
                    }
                )

            keyset_map[(subset, prompt)] = [
                {"key_set": " | ".join(kset), "count": int(cnt)}
                for kset, cnt in keyset_counter.most_common(10)
            ]

    df_summary = pd.DataFrame(summaries)
    if not df_summary.empty:
        df_summary = df_summary.sort_values(["subset", "prompt"]).reset_index(drop=True)

    df_keys = pd.DataFrame(key_rows)
    if not df_keys.empty:
        df_keys = df_keys.sort_values(["subset", "prompt", "count"], ascending=[True, True, False]).reset_index(drop=True)

    return df_summary, df_keys, keyset_map


def build_compare_table(df_summary: pd.DataFrame, prompt_a: str, prompt_b: str) -> pd.DataFrame:
    if df_summary.empty:
        return pd.DataFrame()

    cols = [
        "ndcg_best_mean",
        "ndcg_last_mean",
        "unique_key_count",
        "mean_keys_per_response",
        "median_keys_per_response",
        "p90_keys_per_response",
        "nonstandard_response_ratio",
    ]
    wide_parts = []
    for c in cols:
        pvt = df_summary.pivot(index="subset", columns="prompt", values=c)
        pvt.columns = [f"{c}__{x}" for x in pvt.columns]
        wide_parts.append(pvt)

    comp = pd.concat(wide_parts, axis=1).reset_index()

    # Intent: explicit prompt_b - prompt_a deltas make side-by-side prompt impact auditable.
    for c in cols:
        b = f"{c}__{prompt_b}"
        a = f"{c}__{prompt_a}"
        if b in comp.columns and a in comp.columns:
            comp[f"delta({c})_{prompt_b}_minus_{prompt_a}"] = comp[b] - comp[a]

    return comp.sort_values("subset").reset_index(drop=True)


def collect_query_level_rows(
    sample_path: Path,
    subset: str,
    prompt: str,
) -> List[Dict[str, Any]]:
    samples = pickle.load(open(sample_path, "rb"))
    rows: List[Dict[str, Any]] = []

    for sample in samples:
        query = sample.get("original_query") or sample.get("query")
        iter_records = sample.get("iter_records", [])

        best_ndcg = np.nan
        key_counts: List[int] = []
        nonstd_flags: List[bool] = []

        for rec in iter_records:
            metrics = rec.get("metrics", {}) or {}
            if "nDCG@10" in metrics:
                v = float(metrics["nDCG@10"])
                best_ndcg = v if np.isnan(best_ndcg) else max(best_ndcg, v)

            docs = rec.get("possible_answer_docs")
            if isinstance(docs, dict):
                keys = [str(k).strip() for k in docs.keys() if str(k).strip()]
                key_counts.append(len(keys))
                nonstd_flags.append(any(k not in CANONICAL_KEYS for k in keys))

        rows.append(
            {
                "subset": subset,
                "prompt": prompt,
                "query": query,
                "best_ndcg": best_ndcg,
                "has_nonstd_any": bool(any(nonstd_flags)),
                "mean_key_count": float(np.mean(key_counts)) if key_counts else np.nan,
                "all_iters_three_keys": bool(len(key_counts) > 0 and all(k == 3 for k in key_counts)),
                "any_iter_ge4_keys": bool(any(k >= 4 for k in key_counts)),
                "nonstd_iter_ratio": float(np.mean(nonstd_flags)) if nonstd_flags else np.nan,
            }
        )

    return rows


def run_query_level_paired(
    base_dir: Path,
    subsets: Sequence[str],
    run_tag: str,
    rct_tag: str,
    prompt_a: str,
    prompt_b: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    a_map = pick_latest_files(base_dir, "all_eval_sample_dicts.pkl", subsets, run_tag, rct_tag, prompt_a)
    b_map = pick_latest_files(base_dir, "all_eval_sample_dicts.pkl", subsets, run_tag, rct_tag, prompt_b)

    rows: List[Dict[str, Any]] = []
    for subset in subsets:
        if subset in a_map:
            rows.extend(collect_query_level_rows(a_map[subset], subset, prompt_a))
        if subset in b_map:
            rows.extend(collect_query_level_rows(b_map[subset], subset, prompt_b))

    qdf = pd.DataFrame(rows)
    if qdf.empty:
        return qdf, {}

    a_df = qdf[qdf.prompt == prompt_a][["subset", "query", "best_ndcg"]].rename(columns={"best_ndcg": "best_a"})
    b_df = qdf[qdf.prompt == prompt_b][
        [
            "subset",
            "query",
            "best_ndcg",
            "has_nonstd_any",
            "mean_key_count",
            "all_iters_three_keys",
            "any_iter_ge4_keys",
            "nonstd_iter_ratio",
        ]
    ].rename(columns={"best_ndcg": "best_b"})

    merged = b_df.merge(a_df, on=["subset", "query"], how="inner")
    merged["delta_best"] = merged["best_b"] - merged["best_a"]

    out: Dict[str, Any] = {}
    out["n_paired"] = int(len(merged))
    out["by_nonstd"] = (
        merged.groupby("has_nonstd_any")["delta_best"].agg(["count", "mean"]).reset_index()
        if not merged.empty
        else pd.DataFrame()
    )

    if not merged.empty:
        median_k = float(merged["mean_key_count"].median())
        merged["low_key_count"] = merged["mean_key_count"] < median_k
        out["median_mean_key_count"] = median_k
        out["by_low_key_count"] = merged.groupby("low_key_count")["delta_best"].agg(["count", "mean"]).reset_index()
        out["pearson"] = float(merged[["mean_key_count", "delta_best"]].corr(method="pearson").iloc[0, 1])
        out["spearman"] = float(merged[["mean_key_count", "delta_best"]].corr(method="spearman").iloc[0, 1])
        out["per_subset_nonstd"] = (
            merged.groupby(["subset", "has_nonstd_any"])["delta_best"].mean().unstack()
        )
    else:
        out["median_mean_key_count"] = np.nan
        out["by_low_key_count"] = pd.DataFrame()
        out["pearson"] = np.nan
        out["spearman"] = np.nan
        out["per_subset_nonstd"] = pd.DataFrame()

    return merged, out


def collect_prompt_examples(
    base_dir: Path,
    subsets: Sequence[str],
    run_tag: str,
    rct_tag: str,
    prompt_name: str,
    max_nonstd_per_subset: int = 3,
    max_canonical_per_subset: int = 1,
) -> pd.DataFrame:
    hist_map = pick_latest_files(base_dir, "llm_api_history.pkl", subsets, run_tag, rct_tag, prompt_name)

    rows: List[Dict[str, Any]] = []
    for subset in sorted(hist_map.keys()):
        hist = pickle.load(open(hist_map[subset], "rb"))
        nonstd_count = 0
        canonical_count = 0

        for idx, row in enumerate(hist):
            obj = parse_response_to_obj(row.get("response", ""))
            if obj is None:
                continue
            docs = find_docs_obj(obj)
            if not isinstance(docs, dict):
                continue

            keys = [str(k).strip() for k in docs.keys() if str(k).strip()]
            nonstd_keys = [k for k in keys if k not in CANONICAL_KEYS]
            plan = str(obj.get("Plan", "") or "").strip()
            raw = str(row.get("response", "") or "")

            if nonstd_keys and nonstd_count < max_nonstd_per_subset:
                rows.append(
                    {
                        "subset": subset,
                        "type": "nonstandard",
                        "row_idx": int(idx),
                        "keys": keys,
                        "nonstd_keys": nonstd_keys,
                        "plan_preview": plan[:240],
                        "response_preview": raw[:360],
                    }
                )
                nonstd_count += 1
            elif (not nonstd_keys) and canonical_count < max_canonical_per_subset:
                rows.append(
                    {
                        "subset": subset,
                        "type": "canonical_only",
                        "row_idx": int(idx),
                        "keys": keys,
                        "nonstd_keys": [],
                        "plan_preview": plan[:240],
                        "response_preview": raw[:360],
                    }
                )
                canonical_count += 1

            if nonstd_count >= max_nonstd_per_subset and canonical_count >= max_canonical_per_subset:
                break

    return pd.DataFrame(rows)


def collect_nonstd_frequency(
    base_dir: Path,
    subsets: Sequence[str],
    run_tag: str,
    rct_tag: str,
    prompt_name: str,
    top_n: int,
) -> Dict[str, pd.DataFrame]:
    hist_map = pick_latest_files(base_dir, "llm_api_history.pkl", subsets, run_tag, rct_tag, prompt_name)

    subset_counter: Dict[str, Counter] = {s: Counter() for s in sorted(hist_map.keys())}
    global_counter: Counter = Counter()
    subset_presence: Dict[str, set] = defaultdict(set)
    subset_total: Dict[str, int] = defaultdict(int)
    subset_nonstd: Dict[str, int] = defaultdict(int)

    for subset in sorted(hist_map.keys()):
        hist = pickle.load(open(hist_map[subset], "rb"))
        for row in hist:
            obj = parse_response_to_obj(row.get("response", ""))
            if obj is None:
                continue
            docs = find_docs_obj(obj)
            if not isinstance(docs, dict):
                continue
            keys = [str(k).strip() for k in docs.keys() if str(k).strip()]
            nonstd_keys = [k for k in keys if k not in CANONICAL_KEYS]

            subset_total[subset] += 1
            if nonstd_keys:
                subset_nonstd[subset] += 1

            for k in nonstd_keys:
                subset_counter[subset][k] += 1
                global_counter[k] += 1
                subset_presence[k].add(subset)

    ratio_rows = []
    for subset in sorted(subset_total.keys()):
        total = int(subset_total[subset])
        nonstd = int(subset_nonstd[subset])
        ratio_rows.append(
            {
                "subset": subset,
                "responses_with_docs": total,
                "responses_with_nonstd": nonstd,
                "nonstd_ratio": (nonstd / total) if total else np.nan,
            }
        )

    subset_rows = []
    for subset in sorted(subset_counter.keys()):
        for key, cnt in subset_counter[subset].most_common(top_n):
            subset_rows.append(
                {"subset": subset, "key": key, "count": int(cnt)}
            )

    global_rows = []
    for key, cnt in global_counter.most_common(top_n):
        global_rows.append(
            {
                "key": key,
                "count": int(cnt),
                "num_subsets_present": int(len(subset_presence[key])),
                "subsets": ", ".join(sorted(subset_presence[key])),
            }
        )

    return {
        "ratio": pd.DataFrame(ratio_rows).sort_values("subset"),
        "subset_top": pd.DataFrame(subset_rows).sort_values(["subset", "count"], ascending=[True, False]),
        "global_top": pd.DataFrame(global_rows),
    }


def print_df(title: str, df: pd.DataFrame, max_rows: int = 200) -> None:
    print(f"\n[{title}]")
    if df is None or df.empty:
        print("(empty)")
        return
    show = df.head(max_rows)
    print(show.to_string(index=False))
    if len(df) > len(show):
        print(f"... ({len(df) - len(show)} more rows)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Round5 prompt key behavior and performance.")
    parser.add_argument("--base_results_dir", type=str, default=None)
    parser.add_argument("--run_tag", type=str, default="round5_mrr_selector_accum_meanscore_global")
    parser.add_argument("--rct_tag", type=str, default="RCT=10")
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["biology", "earth_science", "economics", "psychology", "robotics"],
    )
    parser.add_argument("--prompt_a", type=str, default="agent_executor_v1")
    parser.add_argument("--prompt_b", type=str, default="agent_executor_v1_icl2")
    parser.add_argument("--top_n", type=int, default=30)
    parser.add_argument("--show_examples", action="store_true")
    parser.add_argument("--show_nonstd_freq", action="store_true")
    parser.add_argument("--skip_query_level", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = resolve_base_results_dir(args.base_results_dir)

    print("=== Config ===")
    print(f"base_results_dir: {base_dir}")
    print(f"run_tag: {args.run_tag}")
    print(f"rct_tag: {args.rct_tag}")
    print(f"subsets: {args.subsets}")
    print(f"prompt_a: {args.prompt_a}")
    print(f"prompt_b: {args.prompt_b}")

    df_summary, df_keys, keyset_map = build_subset_prompt_summary(
        base_dir=base_dir,
        subsets=args.subsets,
        run_tag=args.run_tag,
        rct_tag=args.rct_tag,
        prompt_a=args.prompt_a,
        prompt_b=args.prompt_b,
    )

    print_df("Summary per subset/prompt", df_summary)
    print_df("Top keys per subset/prompt", df_keys.groupby(["subset", "prompt"], as_index=False).head(15) if not df_keys.empty else df_keys)

    comp = build_compare_table(df_summary, args.prompt_a, args.prompt_b)
    print_df(f"Compare ({args.prompt_b} - {args.prompt_a})", comp)

    if not comp.empty:
        macro_col = f"delta(ndcg_best_mean)_{args.prompt_b}_minus_{args.prompt_a}"
        if macro_col in comp.columns:
            print(f"\nMacro mean {macro_col}: {comp[macro_col].mean():.6f}")

    if not args.skip_query_level:
        paired_df, paired_stats = run_query_level_paired(
            base_dir=base_dir,
            subsets=args.subsets,
            run_tag=args.run_tag,
            rct_tag=args.rct_tag,
            prompt_a=args.prompt_a,
            prompt_b=args.prompt_b,
        )
        print(f"\n[Query-level paired n] {paired_stats.get('n_paired', 0)}")
        print_df("Paired delta by has_nonstd_any", paired_stats.get("by_nonstd", pd.DataFrame()))
        print(f"median(mean_key_count): {paired_stats.get('median_mean_key_count', np.nan)}")
        print_df("Paired delta by low_key_count", paired_stats.get("by_low_key_count", pd.DataFrame()))
        print(f"pearson(mean_key_count, delta_best): {paired_stats.get('pearson', np.nan)}")
        print(f"spearman(mean_key_count, delta_best): {paired_stats.get('spearman', np.nan)}")
        print_df("Per-subset delta by nonstd", paired_stats.get("per_subset_nonstd", pd.DataFrame()).reset_index() if isinstance(paired_stats.get("per_subset_nonstd"), pd.DataFrame) and not paired_stats.get("per_subset_nonstd").empty else pd.DataFrame())

    if args.show_examples:
        examples = collect_prompt_examples(
            base_dir=base_dir,
            subsets=args.subsets,
            run_tag=args.run_tag,
            rct_tag=args.rct_tag,
            prompt_name=args.prompt_b,
        )
        print_df(f"Examples for {args.prompt_b}", examples)

    if args.show_nonstd_freq:
        freq = collect_nonstd_frequency(
            base_dir=base_dir,
            subsets=args.subsets,
            run_tag=args.run_tag,
            rct_tag=args.rct_tag,
            prompt_name=args.prompt_b,
            top_n=args.top_n,
        )
        print_df(f"{args.prompt_b} nonstd ratio by subset", freq["ratio"])
        print_df(f"{args.prompt_b} top nonstd keys per subset", freq["subset_top"])
        print_df(f"{args.prompt_b} global top nonstd keys", freq["global_top"])


if __name__ == "__main__":
    main()
