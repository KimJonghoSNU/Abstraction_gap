import argparse
import pickle as pkl
from collections import Counter
from pathlib import Path


def load_action_rates(sample_path: Path):
    samples = pkl.load(open(sample_path, "rb"))
    action_counts = {}
    for sample in samples:
        for rec in sample.get("iter_records", []):
            iter_idx = rec.get("iter")
            action = str(rec.get("action", "exploit")).lower()
            action_counts.setdefault(iter_idx, Counter())
            action_counts[iter_idx][action] += 1
    return action_counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize explore/exploit rates per iter for round3 runs.")
    parser.add_argument("paths", nargs="+", help="Paths to all_eval_sample_dicts.pkl")
    args = parser.parse_args()

    for path_str in args.paths:
        path = Path(path_str)
        if not path.exists():
            print(f"Missing: {path}")
            continue
        action_counts = load_action_rates(path)
        print(f"\n== {path.parent} ==")
        for iter_idx in sorted(action_counts.keys()):
            counts = action_counts[iter_idx]
            total = sum(counts.values()) or 1
            explore = counts.get("explore", 0)
            exploit = counts.get("exploit", 0)
            other = total - explore - exploit
            print(
                f"Iter {iter_idx}: explore={explore/total:.3f} ({explore}) | "
                f"exploit={exploit/total:.3f} ({exploit}) | other={other}"
            )


if __name__ == "__main__":
    main()
