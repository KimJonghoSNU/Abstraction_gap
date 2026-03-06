import argparse
import os
import pickle as pkl
import re
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd


def _infer_subset_from_path(path: Path) -> str:
    parts = list(path.parts)
    if "BRIGHT" in parts:
        idx = parts.index("BRIGHT")
        if idx + 1 < len(parts):
            return str(parts[idx + 1])
    return "unknown_subset"


def _infer_model_from_path(path: Path) -> str:
    m = re.search(r"Llm=([^/]+)", str(path))
    return m.group(1) if m else "unknown_model"


def _collect_iter_rows(sample_dict: Dict, topk: int) -> List[Dict[str, float]]:
    iter_records = sample_dict.get("iter_records", []) or []
    sample_gold_ids = [str(x) for x in (sample_dict.get("gold_doc_ids", []) or [])]
    seen_doc_ids = set()

    rows: List[Dict[str, float]] = []
    for rec in iter_records:
        iter_idx = int(rec.get("iter", -1))
        local_doc_ids = [str(x) for x in (rec.get("local_doc_ids", []) or [])]
        gold_doc_ids = [str(x) for x in (rec.get("gold_doc_ids", []) or sample_gold_ids)]
        gold_set = set(gold_doc_ids)

        top_docs = local_doc_ids[: max(1, int(topk))]
        new_top_docs = [doc_id for doc_id in top_docs if doc_id not in seen_doc_ids]
        old_top_docs = [doc_id for doc_id in top_docs if doc_id in seen_doc_ids]

        rows.append(
            {
                "iter": float(iter_idx),
                "has_gold_topk": float(any(doc_id in gold_set for doc_id in top_docs)),
                "has_new_gold_topk": float(any(doc_id in gold_set for doc_id in new_top_docs)),
                "has_old_gold_topk": float(any(doc_id in gold_set for doc_id in old_top_docs)),
                "new_only_gold_topk": float(
                    any(doc_id in gold_set for doc_id in new_top_docs)
                    and not any(doc_id in gold_set for doc_id in old_top_docs)
                ),
                "new_slots_topk": float(len(new_top_docs)),
                "has_new_slot": float(len(new_top_docs) > 0),
            }
        )

        # Intent: keep accumulation definition identical to round5 runtime by carrying all retrieved local_doc_ids forward.
        seen_doc_ids.update(local_doc_ids)

    return rows


def _mean_or_nan(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _summarize_rows(rows: List[Dict[str, float]], min_iter: int) -> Dict[str, float]:
    scoped = [row for row in rows if int(row.get("iter", -1)) >= int(min_iter)]
    if not scoped:
        return {
            "n_rows": 0.0,
            "P(gold@k)": float("nan"),
            "P(new_gold@k)": float("nan"),
            "P(old_gold@k)": float("nan"),
            "P(new_only_gold@k)": float("nan"),
            "avg_new_slots@k": float("nan"),
            "P(has_new_slot)": float("nan"),
            "P(new_gold@k|has_new_slot)": float("nan"),
        }

    has_new_rows = [row for row in scoped if row.get("has_new_slot", 0.0) > 0.0]
    return {
        "n_rows": float(len(scoped)),
        "P(gold@k)": _mean_or_nan([row["has_gold_topk"] for row in scoped]),
        "P(new_gold@k)": _mean_or_nan([row["has_new_gold_topk"] for row in scoped]),
        "P(old_gold@k)": _mean_or_nan([row["has_old_gold_topk"] for row in scoped]),
        "P(new_only_gold@k)": _mean_or_nan([row["new_only_gold_topk"] for row in scoped]),
        "avg_new_slots@k": _mean_or_nan([row["new_slots_topk"] for row in scoped]),
        "P(has_new_slot)": _mean_or_nan([row["has_new_slot"] for row in scoped]),
        "P(new_gold@k|has_new_slot)": _mean_or_nan(
            [row["has_new_gold_topk"] for row in has_new_rows]
        ),
    }


def _discover_paths(args: argparse.Namespace) -> List[Path]:
    paths: List[Path] = []
    for raw_path in args.eval_samples_pkl:
        path = Path(raw_path)
        if path.exists() and path.is_file():
            paths.append(path)

    if args.glob_pattern:
        paths.extend(Path(".").glob(args.glob_pattern))

    dedup = []
    seen = set()
    for path in paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        dedup.append(path)

    include_filters = [x for x in (args.include or []) if str(x).strip()]
    exclude_filters = [x for x in (args.exclude or []) if str(x).strip()]

    out = []
    for path in dedup:
        s = str(path)
        if include_filters and not all(token in s for token in include_filters):
            continue
        if exclude_filters and any(token in s for token in exclude_filters):
            continue
        out.append(path)

    return sorted(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze round5 accumulation effect on new-topk gold appearance across iterations."
    )
    parser.add_argument(
        "--eval_samples_pkl",
        nargs="*",
        default=[],
        help="One or more all_eval_sample_dicts.pkl paths.",
    )
    parser.add_argument(
        "--glob_pattern",
        type=str,
        default="",
        help="Optional glob pattern to discover pkl files, e.g. 'results/BRIGHT/*/round5/**/all_eval_sample_dicts.pkl'.",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=[],
        help="Keep only paths that include all tokens.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Drop paths that include any token.",
    )
    parser.add_argument("--topk", type=int, default=10, help="Top-k cutoff for hit computation.")
    parser.add_argument("--min_iter", type=int, default=1, help="Minimum iteration index to include in summary.")
    parser.add_argument(
        "--out_csv",
        type=str,
        default="",
        help="Optional CSV output path for per-file summary rows.",
    )
    args = parser.parse_args()

    target_paths = _discover_paths(args)
    if not target_paths:
        raise FileNotFoundError("No matching all_eval_sample_dicts.pkl files found.")

    rows: List[Dict[str, object]] = []
    for path in target_paths:
        with open(path, "rb") as f:
            samples = pkl.load(f)

        all_iter_rows: List[Dict[str, float]] = []
        for sample in samples:
            all_iter_rows.extend(_collect_iter_rows(sample, topk=args.topk))

        summary = _summarize_rows(all_iter_rows, min_iter=args.min_iter)
        row = {
            "path": str(path),
            "subset": _infer_subset_from_path(path),
            "model": _infer_model_from_path(path),
            "num_samples": int(len(samples)),
            "topk": int(args.topk),
            "min_iter": int(args.min_iter),
            **summary,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    show_cols = [
        "subset",
        "model",
        "num_samples",
        "P(gold@k)",
        "P(new_gold@k)",
        "P(new_only_gold@k)",
        "avg_new_slots@k",
        "P(has_new_slot)",
        "P(new_gold@k|has_new_slot)",
    ]
    print(df[show_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    group_df = (
        df.groupby("model", dropna=False)[
            [
                "P(gold@k)",
                "P(new_gold@k)",
                "P(new_only_gold@k)",
                "avg_new_slots@k",
                "P(has_new_slot)",
                "P(new_gold@k|has_new_slot)",
            ]
        ]
        .mean()
        .reset_index()
    )
    print("\n[Mean by model]")
    print(group_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if args.out_csv:
        out_path = Path(args.out_csv)
        if out_path.parent and not out_path.parent.exists():
            os.makedirs(out_path.parent, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
