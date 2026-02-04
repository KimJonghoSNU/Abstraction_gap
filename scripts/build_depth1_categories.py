import argparse
import datetime
import json
import os
from typing import Dict, List

from json_repair import repair_json
from vllm import LLM, SamplingParams


PROMPT_TEMPLATE = (
    "You are designing retrieval support roles for query rewriting."
    "What you must produce is a small set of role categories that describe how documents support answering a question."
    "Given a depth-1 branch description, produce a short category label.\n\n"
    "Requirements:\n"
    "- 1-3 words only.\n"
    "- Broad reusable abstraction (avoid specific entities).\n"
    "- Title Case.\n"
    "- Output JSON only.\n\n"
    "Branch ID: {branch_id}\n"
    "Branch Description:\n"
    "{branch_desc}\n\n"
    "Output JSON:\n"
    "{\n"
    "  \"category\": \"...\"\n"
    "}\n"
)


def _load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _parse_category(text: str) -> str:
    cleaned = text.split("</think>\n")[-1].strip()
    if "```" in cleaned:
        try:
            parts = cleaned.split("```")
            fenced = [parts[i] for i in range(1, len(parts), 2)]
            if fenced:
                cleaned = fenced[-1].strip()
        except Exception:
            pass
    raw = cleaned
    obj = None
    try:
        obj = json.loads(raw)
    except Exception:
        try:
            obj = repair_json(raw, return_objects=True)
        except Exception:
            obj = None
    if isinstance(obj, dict):
        value = obj.get("category") or obj.get("Category") or ""
        cleaned = str(value).strip()
    else:
        cleaned = raw.strip()
    cleaned = cleaned.replace("\n", " ").strip().strip("\"'")
    tokens = [t for t in cleaned.split() if t]
    if not tokens:
        return "Misc"
    tokens = tokens[:2]
    return " ".join(tokens).title()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--subset", type=str, required=True)
    parser.add_argument("--prompt_name", type=str, default="cat_abstract")
    parser.add_argument("--llm", type=str, required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    args = parser.parse_args()

    input_path = os.path.join(
        "trees",
        args.dataset,
        args.subset,
        "node_catalog_depth1.jsonl",
    )
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing input: {input_path}")

    output_path = os.path.join(
        "trees",
        args.dataset,
        args.subset,
        f"{args.prompt_name}.json",
    )

    llm = LLM(
        model=args.llm,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    rows = _load_jsonl(input_path)
    prompts: List[str] = []
    for row in rows:
        branch_id = row.get("id") or json.dumps(row.get("path"))
        prompts.append(
            PROMPT_TEMPLATE.format(
                branch_id=branch_id,
                branch_desc=row.get("desc", ""),
            )
        )

    categories: List[str] = []
    for start in range(0, len(prompts), args.batch_size):
        batch = prompts[start:start + args.batch_size]
        outputs = llm.generate(batch, sampling_params)
        for out in outputs:
            text = out.outputs[0].text if out.outputs else ""
            categories.append(_parse_category(text))

    categories_out: List[Dict] = []
    branch_to_category: Dict[str, int] = {}
    branches_out: List[Dict] = []
    # Intent: one abstraction category per depth-1 branch (no clustering in v1).
    for idx, (row, cat_name) in enumerate(zip(rows, categories)):
        cat_id = idx
        path = row.get("path", [])
        path_key = json.dumps(path)
        categories_out.append({
            "id": cat_id,
            "name": cat_name,
            "member_branch_paths": [path],
            "member_branch_ids": [row.get("id")],
        })
        branch_to_category[path_key] = cat_id
        branches_out.append({
            "id": row.get("id"),
            "path": path,
            "desc": row.get("desc", ""),
            "category_id": cat_id,
            "category": cat_name,
        })

    payload = {
        "meta": {
            "prompt_name": args.prompt_name,
            "model": args.llm,
            "dataset": args.dataset,
            "subset": args.subset,
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        },
        "categories": categories_out,
        "branch_to_category": branch_to_category,
        "branches": branches_out,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved categories to {output_path}")


if __name__ == "__main__":
    main()
