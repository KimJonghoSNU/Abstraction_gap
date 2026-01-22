import argparse
import json
import pickle
from pathlib import Path

def extract_schema_records(samples):
    records = []
    for idx, sample in enumerate(samples):
        rewrite_history = sample.get("rewrite_history", []) if isinstance(sample, dict) else []
        for entry in rewrite_history:
            labels = entry.get("schema_labels")
            if not labels:
                continue
            records.append({
                "sample_idx": idx,
                "iter": entry.get("iter"),
                "phase": entry.get("phase"),
                "schema_labels": labels,
                "possible_answer_docs": entry.get("possible_answer_docs"),
            })
    return records

# scripts/extract_schema_labels.py --input /data4/jongho/lattice/results/BRIGHT/biology/S=biology-TV=bottom-up-TPV=5-RInTP=/1-NumLC=10-PlTau=5-RCF=0-LlmApiB=vllm/Llm=Qwen3-4B-Instruct-2507-NumI=5/NumES=1000-MaxBS=2-S=exp1_qe_iter-FTT=True-FT=100/GBT=10-PreFRS=branch-QePN=pre_flat_rewrite_v1-QeCP=biology_pre_flat_rewrite_v1-RPN=gate_rewrite_schema_v1/RM=concat-RE=1-RCT=5-RCS=fused/all_eval_sample_dicts.pkl
# python scripts/extract_schema_labels.py --input /data4/jongho/lattice/results/BRIGHT/biology/S=biology-TV=bottom-up-TPV=5-RInTP=/1-NumLC=10-PlTau=5-RCF=0-LlmApiB=vllm/Llm=Qwen3-4B-Instruct-2507-NumI=5/NumES=1000-MaxBS=2-S=exp1_qe_iter_schema_d1-FTT=True-FT=100/GBT=10-PreFRS=branch-QePN=pre_flat_rewrite_v1-QeCP=biology_pre_flat_rewrite_v1-RPN=gate_rewrite_schema_v1/RM=concat-RE=1-RCT=5-RCS=slate/all_eval_sample_dicts.pkl
def main():
    parser = argparse.ArgumentParser(description="Extract schema labels from all_eval_sample_dicts.pkl")
    parser.add_argument("--input", required=True, help="Path to all_eval_sample_dicts.pkl")
    parser.add_argument("--output", default=None, help="Output JSONL path (default: <input_dir>/schema_labels.jsonl)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_path = Path(args.output) if args.output else input_path.parent / "schema_labels.jsonl"

    with input_path.open("rb") as f:
        samples = pickle.load(f)

    records = extract_schema_records(samples)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} rows to {output_path}")

if __name__ == "__main__":
    main()
