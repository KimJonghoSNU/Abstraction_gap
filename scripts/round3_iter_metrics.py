import argparse
from pathlib import Path

import pandas as pd


def summarize_metrics(metrics_path: Path) -> None:
    df = pd.read_pickle(metrics_path)
    print(f"\n== {metrics_path.parent} ==")
    if hasattr(df.columns, "levels"):
        iters = [c for c in df.columns.levels[0] if isinstance(c, str) and c.startswith("Iter ")]
        for iter_key in sorted(iters, key=lambda x: int(x.split("Iter ")[-1])):
            sub = df[iter_key]
            ndcg = sub["nDCG@10"].mean() if "nDCG@10" in sub.columns else None
            r10 = sub["Recall@10"].mean() if "Recall@10" in sub.columns else None
            r100 = sub["Recall@100"].mean() if "Recall@100" in sub.columns else None
            cov = sub["Coverage"].mean() if "Coverage" in sub.columns else None
            print(f"{iter_key}: nDCG@10={ndcg:.2f} | R@10={r10:.2f} | R@100={r100:.2f} | Cov={cov:.1f}")
    else:
        ndcg = df["nDCG@10"].mean() if "nDCG@10" in df.columns else None
        r10 = df["Recall@10"].mean() if "Recall@10" in df.columns else None
        r100 = df["Recall@100"].mean() if "Recall@100" in df.columns else None
        cov = df["Coverage"].mean() if "Coverage" in df.columns else None
        print(f"Iter 0: nDCG@10={ndcg:.2f} | R@10={r10:.2f} | R@100={r100:.2f} | Cov={cov:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize per-iter metrics for round3 runs.")
    parser.add_argument("paths", nargs="+", help="Paths to all_eval_metrics.pkl")
    args = parser.parse_args()

    for path_str in args.paths:
        path = Path(path_str)
        if not path.exists():
            print(f"Missing: {path}")
            continue
        summarize_metrics(path)


if __name__ == "__main__":
    main()
