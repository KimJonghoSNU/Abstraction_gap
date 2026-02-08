import argparse
import json
import os
import pickle as pkl
from collections import defaultdict
from typing import Dict, List

#   - 새 스크립트: scripts/eval_round3_category_stickiness.py
#   - 목적: all_eval_sample_dicts.pkl에서 action=="explore"인 케이스만 모아,
#     prev_selected(query_categories)와 curr_selected(actions non-PRUNE -> possible_docs -> query_categories fallback)의 겹침 여부로
#     keep 비율을 집계합니다.
#   - 출력:
#       - 전체 요약 비율 (ExploreKeepPrevRate_overExplore, ..._overAnalyzable)
#       - iteration별 keep 비율
#       - category 전이 카운트(pair/primary)
#   - 저장 기본 경로: eval_pkl과 같은 폴더의 category_stickiness_report.json
#   - 문법 체크: python -m py_compile scripts/eval_round3_category_stickiness.py 통과

#   실행 예시:

#   python scripts/eval_round3_category_stickiness.py \
#       --eval_pkl <RESULT_DIR>/all_eval_sample_dicts.pkl


CATEGORY_ORDER = ["Theory", "Entity", "Example", "Other"]


def _normalize_categories(value) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        cat = str(item or "").strip()
        if cat and cat not in out:
            out.append(cat)
    return out


def _selected_from_actions(record: Dict) -> List[str]:
    actions = record.get("actions")
    if not isinstance(actions, dict):
        return []
    selected: List[str] = []
    for key in CATEGORY_ORDER:
        action = str(actions.get(key, "")).strip().upper()
        if action and action != "PRUNE":
            selected.append(key)
    if selected:
        return selected
    for key, raw_action in actions.items():
        action = str(raw_action or "").strip().upper()
        if action and action != "PRUNE":
            cat = str(key or "").strip()
            if cat and cat not in selected:
                selected.append(cat)
    return selected


def _selected_from_possible_docs(record: Dict) -> List[str]:
    docs = record.get("possible_docs")
    if not isinstance(docs, dict):
        return []
    selected: List[str] = []
    for key in CATEGORY_ORDER:
        text = str(docs.get(key, "")).strip()
        if text:
            selected.append(key)
    if selected:
        return selected
    for key, text in docs.items():
        val = str(text or "").strip()
        cat = str(key or "").strip()
        if val and cat and cat not in selected:
            selected.append(cat)
    return selected


def _get_prev_selected_categories(record: Dict) -> List[str]:
    return _normalize_categories(record.get("query_categories", []))


def _get_curr_selected_categories(record: Dict) -> List[str]:
    selected = _selected_from_actions(record)
    if selected:
        return selected
    selected = _selected_from_possible_docs(record)
    if selected:
        return selected
    return _normalize_categories(record.get("query_categories", []))


def _safe_rate(numer: int, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return float(numer) / float(denom)


def build_report(eval_samples: List[Dict], eval_pkl_path: str) -> Dict:
    total_explore = 0
    analyzable_explore = 0
    keep_count = 0

    per_iter = defaultdict(lambda: {"explore_cases": 0, "analyzable_cases": 0, "keep_count": 0})
    pair_transition = defaultdict(lambda: defaultdict(int))
    primary_transition = defaultdict(lambda: defaultdict(int))

    for sample in eval_samples:
        iter_records = sample.get("iter_records", [])
        if not isinstance(iter_records, list):
            continue
        for rec in iter_records:
            action = str(rec.get("action", "")).strip().lower()
            if action != "explore":
                continue

            iter_idx = int(rec.get("iter", -1))
            total_explore += 1
            per_iter[iter_idx]["explore_cases"] += 1

            prev_selected = _get_prev_selected_categories(rec)
            curr_selected = _get_curr_selected_categories(rec)
            if not prev_selected or not curr_selected:
                continue

            analyzable_explore += 1
            per_iter[iter_idx]["analyzable_cases"] += 1

            prev_set = set(prev_selected)
            curr_set = set(curr_selected)
            # Intent: define "keep previous category" as non-empty overlap between previous and current selected categories.
            keep = len(prev_set.intersection(curr_set)) > 0
            if keep:
                keep_count += 1
                per_iter[iter_idx]["keep_count"] += 1

            for p_cat in prev_selected:
                for c_cat in curr_selected:
                    pair_transition[p_cat][c_cat] += 1

            p_primary = prev_selected[0]
            c_primary = curr_selected[0]
            primary_transition[p_primary][c_primary] += 1

    per_iter_report: Dict[str, Dict] = {}
    for iter_idx in sorted(per_iter.keys()):
        rec = per_iter[iter_idx]
        per_iter_report[str(iter_idx)] = {
            "explore_cases": int(rec["explore_cases"]),
            "analyzable_cases": int(rec["analyzable_cases"]),
            "keep_count": int(rec["keep_count"]),
            "keep_rate_over_explore": _safe_rate(int(rec["keep_count"]), int(rec["explore_cases"])),
            "keep_rate_over_analyzable": _safe_rate(int(rec["keep_count"]), int(rec["analyzable_cases"])),
        }

    return {
        "meta": {
            "eval_pkl": eval_pkl_path,
            "n_samples": len(eval_samples),
            "definition": "ExploreKeepPrev is true when prev_selected ∩ curr_selected is non-empty.",
            "prev_selected_source": "iter_record.query_categories",
            "curr_selected_source": "iter_record.actions(non-PRUNE) fallback possible_docs/query_categories",
        },
        "summary": {
            "total_explore_cases": int(total_explore),
            "analyzable_explore_cases": int(analyzable_explore),
            "keep_count": int(keep_count),
            "ExploreKeepPrevRate_overExplore": _safe_rate(keep_count, total_explore),
            "ExploreKeepPrevRate_overAnalyzable": _safe_rate(keep_count, analyzable_explore),
        },
        "per_iter": per_iter_report,
        "transition_pair_counts": {k: dict(v) for k, v in pair_transition.items()},
        "transition_primary_counts": {k: dict(v) for k, v in primary_transition.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze explore-time category stickiness from round3 all_eval_sample_dicts.pkl.")
    parser.add_argument("--eval_pkl", type=str, required=True, help="Path to all_eval_sample_dicts.pkl")
    parser.add_argument("--out_json", type=str, default=None, help="Output json path (default: alongside eval_pkl)")
    args = parser.parse_args()

    with open(args.eval_pkl, "rb") as f:
        eval_samples = pkl.load(f)

    if not isinstance(eval_samples, list):
        raise ValueError("Expected eval_pkl to contain a list of sample dicts.")

    report = build_report(eval_samples, args.eval_pkl)

    if args.out_json:
        out_json = args.out_json
    else:
        out_json = os.path.join(os.path.dirname(args.eval_pkl), "category_stickiness_report.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[Saved] {out_json}")
    print(
        "[Summary] "
        f"Explore={report['summary']['total_explore_cases']} "
        f"Analyzable={report['summary']['analyzable_explore_cases']} "
        f"Keep={report['summary']['keep_count']} "
        f"Rate={report['summary']['ExploreKeepPrevRate_overExplore']:.4f}"
    )


if __name__ == "__main__":
    main()
