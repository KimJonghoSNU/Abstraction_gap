import argparse
import json
import os
import pickle as pkl
from typing import Dict, List, Sequence, Tuple

import numpy as np


Path = Tuple[int, ...]


def _to_path_tuple(path_obj: Sequence[int]) -> Path:
    return tuple(int(x) for x in path_obj)


def _as_path_list(paths_obj: Sequence[Sequence[int]]) -> List[Path]:
    return [_to_path_tuple(p) for p in paths_obj]


def _find_iter_record(sample: Dict, iter_idx: int) -> Dict:
    for rec in sample.get("iter_records", []):
        if int(rec.get("iter", -1)) == iter_idx:
            return rec
    return {}


def _get_anchor_paths_from_record(rec: Dict) -> List[Path]:
    anchor_leaf_paths = rec.get("anchor_leaf_paths", [])
    if anchor_leaf_paths:
        # Intent: for RALR=none runs, anchor eval is leaf-only list in anchor score order.
        return _as_path_list(anchor_leaf_paths)
    local_paths = rec.get("local_paths", [])
    if local_paths:
        return _as_path_list(local_paths)
    global_paths = rec.get("global_paths", [])
    return _as_path_list(global_paths)


def _compute_hit_at_k(pred_paths: Sequence[Path], gold_paths: Sequence[Path], k: int) -> float:
    pred_topk = set(pred_paths[:k])
    return float(any(g in pred_topk for g in gold_paths))


def _compute_ndcg_at_k(pred_paths: Sequence[Path], gold_paths: Sequence[Path], k: int) -> float:
    if not gold_paths:
        return 0.0
    pred_k = list(pred_paths[:k])
    dcg = 0.0
    for i, p in enumerate(pred_k):
        if p in gold_paths:
            dcg += 1.0 / np.log2(i + 2.0)
    ideal_hits = min(k, len(gold_paths))
    idcg = sum(1.0 / np.log2(i + 2.0) for i in range(ideal_hits))
    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def _jaccard_at_k(a_paths: Sequence[Path], b_paths: Sequence[Path], k: int) -> float:
    a = set(a_paths[:k])
    b = set(b_paths[:k])
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return float(len(a & b)) / float(len(union))


def _overlap_count_at_k(a_paths: Sequence[Path], b_paths: Sequence[Path], k: int) -> int:
    return int(len(set(a_paths[:k]) & set(b_paths[:k])))


def _prefix_set_at_k(paths: Sequence[Path], k: int, depth_tokens: int) -> set:
    out = set()
    for path in paths[:k]:
        if len(path) >= depth_tokens:
            out.add(path[:depth_tokens])
        elif path:
            out.add(path)
    return out


def _prefix_jaccard_at_k(a_paths: Sequence[Path], b_paths: Sequence[Path], k: int, depth_tokens: int) -> float:
    a = _prefix_set_at_k(a_paths, k, depth_tokens)
    b = _prefix_set_at_k(b_paths, k, depth_tokens)
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return float(len(a & b)) / float(len(union))


def _compute_region_hit_at_k(pred_paths: Sequence[Path], gold_paths: Sequence[Path], k: int, depth_tokens: int) -> float:
    pred_regions = _prefix_set_at_k(pred_paths, k, depth_tokens)
    gold_regions = _prefix_set_at_k(gold_paths, len(gold_paths), depth_tokens)
    if not gold_regions:
        return 0.0
    return float(len(pred_regions & gold_regions) > 0)


# Disabled (legacy): prefix-ratio helpers were removed from output metrics to keep interpretation simple.
# def _longest_common_prefix_len(a_path: Path, b_path: Path) -> int:
#     ...
# def _best_prefix_ratio_to_gold(path: Path, gold_paths: Sequence[Path]) -> float:
#     ...
# def _mean_best_prefix_ratio_at_k(pred_paths: Sequence[Path], gold_paths: Sequence[Path], k: int) -> float:
#     ...


def _mean(xs: List[float]) -> float:
    if not xs:
        return 0.0
    return float(np.mean(np.array(xs, dtype=np.float32)))


def _eval_single_run(
    sample_dicts: List[Dict],
    iter_pre: int,
    iter_post: int,
    eval_k: int,
) -> Dict[str, float]:
    # Intent: keep only metrics that directly answer "escape wrong region" vs "preserve already-good cases".
    n_samples = 0
    hit_pre_list: List[float] = []
    hit_post_list: List[float] = []
    ndcg_pre_list: List[float] = []
    ndcg_post_list: List[float] = []
    pool_jaccard_list: List[float] = []
    retained_count_list: List[float] = []
    retained_rate_list: List[float] = []
    region_hit_l1_pre_list: List[float] = []
    region_hit_l1_post_list: List[float] = []
    region_hit_l2_pre_list: List[float] = []
    region_hit_l2_post_list: List[float] = []
    region_jaccard_l1_list: List[float] = []
    region_jaccard_l2_list: List[float] = []
    miss_to_hit_list: List[float] = []
    miss_flags: List[float] = []
    good_flags: List[float] = []
    harm_flags: List[float] = []
    pre_miss_region_hit_l1_deltas: List[float] = []
    pre_miss_region_hit_l2_deltas: List[float] = []

    for sample in sample_dicts:
        rec_pre = _find_iter_record(sample, iter_pre)
        rec_post = _find_iter_record(sample, iter_post)
        if not rec_pre or not rec_post:
            continue
        gold_paths = _as_path_list(sample.get("gold_paths", []))
        if not gold_paths:
            continue
        paths_pre = _get_anchor_paths_from_record(rec_pre)
        paths_post = _get_anchor_paths_from_record(rec_post)
        if (not paths_pre) or (not paths_post):
            continue

        n_samples += 1
        hit_pre = _compute_hit_at_k(paths_pre, gold_paths, eval_k)
        hit_post = _compute_hit_at_k(paths_post, gold_paths, eval_k)
        hit_pre_list.append(hit_pre)
        hit_post_list.append(hit_post)

        ndcg_pre = _compute_ndcg_at_k(paths_pre, gold_paths, eval_k) * 100.0
        ndcg_post = _compute_ndcg_at_k(paths_post, gold_paths, eval_k) * 100.0
        ndcg_pre_list.append(ndcg_pre)
        ndcg_post_list.append(ndcg_post)
        pool_jaccard = _jaccard_at_k(paths_pre, paths_post, eval_k)
        pool_jaccard_list.append(pool_jaccard)
        retained_count = _overlap_count_at_k(paths_pre, paths_post, eval_k)
        retained_count_list.append(float(retained_count))
        retained_rate_list.append(float(retained_count) / float(max(1, eval_k)))
        region_hit_l1_pre = _compute_region_hit_at_k(paths_pre, gold_paths, eval_k, depth_tokens=1)
        region_hit_l1_post = _compute_region_hit_at_k(paths_post, gold_paths, eval_k, depth_tokens=1)
        region_hit_l2_pre = _compute_region_hit_at_k(paths_pre, gold_paths, eval_k, depth_tokens=2)
        region_hit_l2_post = _compute_region_hit_at_k(paths_post, gold_paths, eval_k, depth_tokens=2)
        region_hit_l1_pre_list.append(region_hit_l1_pre)
        region_hit_l1_post_list.append(region_hit_l1_post)
        region_hit_l2_pre_list.append(region_hit_l2_pre)
        region_hit_l2_post_list.append(region_hit_l2_post)
        region_j_l1 = _prefix_jaccard_at_k(paths_pre, paths_post, eval_k, depth_tokens=1)
        region_j_l2 = _prefix_jaccard_at_k(paths_pre, paths_post, eval_k, depth_tokens=2)
        region_jaccard_l1_list.append(region_j_l1)
        region_jaccard_l2_list.append(region_j_l2)

        miss_flag = float(hit_pre == 0.0)
        miss_flags.append(miss_flag)
        miss_to_hit_list.append(float((hit_pre == 0.0) and (hit_post == 1.0)))
        if hit_pre == 0.0:
            pre_miss_region_hit_l1_deltas.append(region_hit_l1_post - region_hit_l1_pre)
            pre_miss_region_hit_l2_deltas.append(region_hit_l2_post - region_hit_l2_pre)
        good_flag = float(hit_pre == 1.0)
        good_flags.append(good_flag)
        harm_flags.append(float((hit_pre == 1.0) and (hit_post == 0.0)))

    miss_count = int(sum(miss_flags))
    miss_to_hit_count = int(sum(miss_to_hit_list))
    good_count = int(sum(good_flags))
    harm_count = int(sum(harm_flags))
    return {
        # Number of samples that have both iter_pre and iter_post records.
        "n_samples": int(n_samples),
        # Percentage of samples with at least one gold document in top-k before first rewrite.
        "Hit@10_pre": _mean(hit_pre_list) * 100.0,
        # Percentage of samples with at least one gold document in top-k after first rewrite.
        "Hit@10_post": _mean(hit_post_list) * 100.0,
        # Absolute Hit@10 gain from iter_pre -> iter_post.
        "Delta_Hit@10": (_mean(hit_post_list) - _mean(hit_pre_list)) * 100.0,
        # Ranking quality before first rewrite (top-k nDCG).
        "nDCG@10_pre": _mean(ndcg_pre_list),
        # Ranking quality after first rewrite (top-k nDCG).
        "nDCG@10_post": _mean(ndcg_post_list),
        # Absolute nDCG@10 gain from iter_pre -> iter_post.
        "Delta_nDCG@10": _mean(ndcg_post_list) - _mean(ndcg_pre_list),
        # Top-k overlap between iter_pre and iter_post pools (1.0 means almost unchanged pool).
        "PoolJaccard@10_pre_vs_post": _mean(pool_jaccard_list),
        # [Meaning] Number of unchanged documents inside top-k before/after rewrite.
        # [Implementation] | set(top-k pre) intersect set(top-k post) |.
        # [Interpretation] Higher means rewrite keeps more of the previous pool.
        "RetainedCount@10_pre_vs_post": _mean(retained_count_list),
        # RetainedCount normalized by k (0~1).
        "RetainedRate@10_pre_vs_post": _mean(retained_rate_list),
        # [Meaning] Coarse region hit at level-1 (top-level branch family) before first rewrite.
        # [Implementation] Check whether any top-k prefix(path,1) overlaps with gold prefix(path,1).
        # [Interpretation] Higher means retrieval at least reaches the correct high-level area.
        "RegionHitL1@10_pre": _mean(region_hit_l1_pre_list) * 100.0,
        # Same coarse region hit at level-1 after first rewrite.
        "RegionHitL1@10_post": _mean(region_hit_l1_post_list) * 100.0,
        # Absolute level-1 region-hit gain from iter_pre -> iter_post.
        "Delta_RegionHitL1@10": (_mean(region_hit_l1_post_list) - _mean(region_hit_l1_pre_list)) * 100.0,
        # [Meaning] Mid-level region hit at level-2 before first rewrite.
        # [Implementation] Check whether any top-k prefix(path,2) overlaps with gold prefix(path,2).
        # [Interpretation] Higher means retrieval reaches a more specific correct subtree.
        "RegionHitL2@10_pre": _mean(region_hit_l2_pre_list) * 100.0,
        # Same mid-level region hit at level-2 after first rewrite.
        "RegionHitL2@10_post": _mean(region_hit_l2_post_list) * 100.0,
        # Absolute level-2 region-hit gain from iter_pre -> iter_post.
        "Delta_RegionHitL2@10": (_mean(region_hit_l2_post_list) - _mean(region_hit_l2_pre_list)) * 100.0,
        # [Meaning] Tree-level overlap between pre/post pools at level-1 prefixes.
        # [Implementation] Jaccard over top-k prefix(path,1) sets.
        # [Interpretation] Lower means larger top-level region transition.
        "RegionJaccardL1@10_pre_vs_post": _mean(region_jaccard_l1_list),
        # Same overlap at level-2 prefixes for finer-grained region transition.
        "RegionJaccardL2@10_pre_vs_post": _mean(region_jaccard_l2_list),
        # [Meaning] "Escape" from initial miss after first rewrite.
        # [Implementation] Count samples with hit_pre=0 and hit_post=1.
        # [Interpretation] Higher means better recovery when the initial retrieval failed.
        "MissToHit@10_givenPreMiss": (100.0 * float(miss_to_hit_count) / float(miss_count)) if miss_count > 0 else 0.0,
        # [Meaning] Tree-level region escape in Bad0 at level-1.
        # [Implementation] Mean of (RegionHitL1_post - RegionHitL1_pre) over hit_pre=0 samples.
        # [Interpretation] Positive means rewrite moves misses toward correct coarse regions.
        "Delta_RegionHitL1@10_givenPreMiss": _mean(pre_miss_region_hit_l1_deltas) * 100.0,
        # Same Bad0 region escape at level-2 (more specific subtree).
        "Delta_RegionHitL2@10_givenPreMiss": _mean(pre_miss_region_hit_l2_deltas) * 100.0,
        # Number of pre-miss samples (denominator for conditional conversion).
        "PreMiss_count": miss_count,
        # Number of samples that convert from pre-miss to post-hit.
        "MissToHit_count": miss_to_hit_count,
        # [Meaning] Harm rate in the Good0 group.
        # [Implementation] Count samples with hit_pre=1 and hit_post=0.
        # [Interpretation] Lower means rewrite is safer when initial retrieval is already correct.
        "HarmRate@10_givenGood0": (100.0 * float(harm_count) / float(good_count)) if good_count > 0 else 0.0,
        # Number of Good0 samples (denominator for robustness metrics).
        "Good0_count": good_count,
        # Number of Good0 samples that lose hit after rewrite.
        "Harm_count": harm_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_pkl", type=str, required=True)
    parser.add_argument("--ours_pkl", type=str, required=True)
    parser.add_argument("--iter_pre", type=int, default=0)
    parser.add_argument("--iter_post", type=int, default=1)
    parser.add_argument("--eval_k", type=int, default=10)
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    baseline_pkl = os.path.abspath(args.baseline_pkl)
    ours_pkl = os.path.abspath(args.ours_pkl)
    if not os.path.exists(baseline_pkl):
        raise FileNotFoundError(f"baseline_pkl not found: {baseline_pkl}")
    if not os.path.exists(ours_pkl):
        raise FileNotFoundError(f"ours_pkl not found: {ours_pkl}")

    baseline_samples = pkl.load(open(baseline_pkl, "rb"))
    ours_samples = pkl.load(open(ours_pkl, "rb"))

    baseline_metrics = _eval_single_run(
        sample_dicts=baseline_samples,
        iter_pre=args.iter_pre,
        iter_post=args.iter_post,
        eval_k=args.eval_k,
    )
    ours_metrics = _eval_single_run(
        sample_dicts=ours_samples,
        iter_pre=args.iter_pre,
        iter_post=args.iter_post,
        eval_k=args.eval_k,
    )

    # Intent: this isolates first-rewrite effect as a difference-in-differences delta.
    did = {
        # Prompt-attributed gain in first-rewrite Hit@10 change.
        "DeltaPrompt_Hit@10": ours_metrics["Delta_Hit@10"] - baseline_metrics["Delta_Hit@10"],
        # Prompt-attributed gain in first-rewrite nDCG@10 change.
        "DeltaPrompt_nDCG@10": ours_metrics["Delta_nDCG@10"] - baseline_metrics["Delta_nDCG@10"],
        # Prompt-attributed gain in level-1 region hit change.
        "DeltaPrompt_Delta_RegionHitL1@10": (
            ours_metrics["Delta_RegionHitL1@10"] - baseline_metrics["Delta_RegionHitL1@10"]
        ),
        # Prompt-attributed gain in level-2 region hit change.
        "DeltaPrompt_Delta_RegionHitL2@10": (
            ours_metrics["Delta_RegionHitL2@10"] - baseline_metrics["Delta_RegionHitL2@10"]
        ),
        # Prompt-attributed gain in pre-miss -> post-hit conversion rate.
        "DeltaPrompt_MissToHit@10": (
            ours_metrics["MissToHit@10_givenPreMiss"] - baseline_metrics["MissToHit@10_givenPreMiss"]
        ),
        # Prompt-attributed change in top-k retention (negative means ours moves pool more).
        "DeltaPrompt_RetainedCount@10_pre_vs_post": (
            ours_metrics["RetainedCount@10_pre_vs_post"] - baseline_metrics["RetainedCount@10_pre_vs_post"]
        ),
        # Prompt-attributed gain in Bad0 level-1 region-hit recovery.
        "DeltaPrompt_Delta_RegionHitL1@10_givenPreMiss": (
            ours_metrics["Delta_RegionHitL1@10_givenPreMiss"] - baseline_metrics["Delta_RegionHitL1@10_givenPreMiss"]
        ),
        # Prompt-attributed gain in Bad0 level-2 region-hit recovery.
        "DeltaPrompt_Delta_RegionHitL2@10_givenPreMiss": (
            ours_metrics["Delta_RegionHitL2@10_givenPreMiss"] - baseline_metrics["Delta_RegionHitL2@10_givenPreMiss"]
        ),
        # Prompt-attributed reduction of harmful regressions on already-correct samples.
        "DeltaPrompt_HarmRate@10_givenGood0": (
            ours_metrics["HarmRate@10_givenGood0"] - baseline_metrics["HarmRate@10_givenGood0"]
        ),
        # Balanced objective: improve escape on Bad0 while not harming Good0.
        "DeltaPrompt_BalancedGain": (
            (ours_metrics["MissToHit@10_givenPreMiss"] - ours_metrics["HarmRate@10_givenGood0"])
            - (baseline_metrics["MissToHit@10_givenPreMiss"] - baseline_metrics["HarmRate@10_givenGood0"])
        ),
    }

    report = {
        "meta": {
            "iter_pre": args.iter_pre,
            "iter_post": args.iter_post,
            "eval_k": args.eval_k,
            "note": "Compare original-query retrieval (iter_pre) vs first-rewrite retrieval (iter_post).",
        },
        "baseline": {
            "path": baseline_pkl,
            "metrics": baseline_metrics,
        },
        "ours": {
            "path": ours_pkl,
            "metrics": ours_metrics,
        },
        "difference_in_differences": did,
    }

    output_path = os.path.abspath(args.output_json)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved report: {output_path}")


if __name__ == "__main__":
    main()
