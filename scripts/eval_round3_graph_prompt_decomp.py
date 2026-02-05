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


def _depth1_counts_at_k(paths: Sequence[Path], k: int) -> Dict[Path, int]:
    counts: Dict[Path, int] = {}
    for p in paths[:k]:
        if not p:
            continue
        d1 = (p[0],)
        counts[d1] = counts.get(d1, 0) + 1
    return counts


def _unique_depth1_count_at_k(paths: Sequence[Path], k: int) -> float:
    return float(len(_depth1_counts_at_k(paths, k)))


def _single_depth1_branch_flag_at_k(paths: Sequence[Path], k: int) -> float:
    counts = _depth1_counts_at_k(paths, k)
    if not counts:
        return 0.0
    return float(len(counts) == 1)


def _max_depth1_share_at_k(paths: Sequence[Path], k: int) -> float:
    counts = _depth1_counts_at_k(paths, k)
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return float(max(counts.values())) / float(total)


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


def _conditional_mean(numerators: Sequence[float], conditioners: Sequence[float]) -> float:
    num = 0.0
    den = 0.0
    for x, c in zip(numerators, conditioners):
        if c >= 0.5:
            den += 1.0
            if x >= 0.5:
                num += 1.0
    if den <= 0.0:
        return 0.0
    return num / den


def _infer_max_gold_depth(sample_dicts: Sequence[Dict]) -> int:
    max_depth = 0
    for sample in sample_dicts:
        for p in sample.get("gold_paths", []):
            max_depth = max(max_depth, len(p))
    return max_depth


def _parse_region_depths(region_depths_arg: str, baseline_samples: Sequence[Dict], ours_samples: Sequence[Dict]) -> List[int]:
    if str(region_depths_arg).strip().lower() == "auto":
        max_depth = max(_infer_max_gold_depth(baseline_samples), _infer_max_gold_depth(ours_samples))
        max_depth = max(1, int(max_depth))
        # Intent: auto mode reports every tree depth up to gold leaf depth for parent-level diagnostics.
        return list(range(1, max_depth + 1))
    out: List[int] = []
    for tok in str(region_depths_arg).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    out = sorted(set([d for d in out if d >= 1]))
    if not out:
        raise ValueError("--region_depths must contain at least one positive integer (or use 'auto').")
    return out


def _eval_single_run(
    sample_dicts: List[Dict],
    iter_pre: int,
    iter_post: int,
    eval_k: int,
    region_depths: Sequence[int],
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
    region_hit_pre_by_depth: Dict[int, List[float]] = {int(d): [] for d in region_depths}
    region_hit_post_by_depth: Dict[int, List[float]] = {int(d): [] for d in region_depths}
    region_jaccard_by_depth: Dict[int, List[float]] = {int(d): [] for d in region_depths}
    unique_depth1_pre_list: List[float] = []
    unique_depth1_post_list: List[float] = []
    single_depth1_pre_list: List[float] = []
    single_depth1_post_list: List[float] = []
    max_depth1_share_pre_list: List[float] = []
    max_depth1_share_post_list: List[float] = []
    miss_to_hit_list: List[float] = []
    miss_flags: List[float] = []
    good_flags: List[float] = []
    harm_flags: List[float] = []
    pre_miss_region_hit_l1_deltas: List[float] = []
    pre_miss_region_hit_l2_deltas: List[float] = []
    pre_miss_region_hit_delta_by_depth: Dict[int, List[float]] = {int(d): [] for d in region_depths}

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
        for d in region_depths:
            d = int(d)
            hit_pre_d = _compute_region_hit_at_k(paths_pre, gold_paths, eval_k, depth_tokens=d)
            hit_post_d = _compute_region_hit_at_k(paths_post, gold_paths, eval_k, depth_tokens=d)
            jac_d = _prefix_jaccard_at_k(paths_pre, paths_post, eval_k, depth_tokens=d)
            region_hit_pre_by_depth[d].append(hit_pre_d)
            region_hit_post_by_depth[d].append(hit_post_d)
            region_jaccard_by_depth[d].append(jac_d)
        # Intent: quantify whether top-k is concentrated into one depth-1 branch or spread across multiple branches.
        unique_depth1_pre_list.append(_unique_depth1_count_at_k(paths_pre, eval_k))
        unique_depth1_post_list.append(_unique_depth1_count_at_k(paths_post, eval_k))
        single_depth1_pre_list.append(_single_depth1_branch_flag_at_k(paths_pre, eval_k))
        single_depth1_post_list.append(_single_depth1_branch_flag_at_k(paths_post, eval_k))
        max_depth1_share_pre_list.append(_max_depth1_share_at_k(paths_pre, eval_k))
        max_depth1_share_post_list.append(_max_depth1_share_at_k(paths_post, eval_k))

        miss_flag = float(hit_pre == 0.0)
        miss_flags.append(miss_flag)
        miss_to_hit_list.append(float((hit_pre == 0.0) and (hit_post == 1.0)))
        if hit_pre == 0.0:
            pre_miss_region_hit_l1_deltas.append(region_hit_l1_post - region_hit_l1_pre)
            pre_miss_region_hit_l2_deltas.append(region_hit_l2_post - region_hit_l2_pre)
            for d in region_depths:
                d = int(d)
                deltas = pre_miss_region_hit_delta_by_depth[d]
                deltas.append(region_hit_post_by_depth[d][-1] - region_hit_pre_by_depth[d][-1])
        good_flag = float(hit_pre == 1.0)
        good_flags.append(good_flag)
        harm_flags.append(float((hit_pre == 1.0) and (hit_post == 0.0)))

    miss_count = int(sum(miss_flags))
    miss_to_hit_count = int(sum(miss_to_hit_list))
    good_count = int(sum(good_flags))
    harm_count = int(sum(harm_flags))
    metrics = {
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
        # Mean number of unique depth-1 branches covered by top-k before rewrite.
        "MeanUniqueDepth1Branches@10_pre": _mean(unique_depth1_pre_list),
        # Mean number of unique depth-1 branches covered by top-k after rewrite.
        "MeanUniqueDepth1Branches@10_post": _mean(unique_depth1_post_list),
        # Delta of unique depth-1 branch coverage (positive means more spread after rewrite).
        "Delta_MeanUniqueDepth1Branches@10": _mean(unique_depth1_post_list) - _mean(unique_depth1_pre_list),
        # Percentage of samples whose top-k all fall under a single depth-1 branch before rewrite.
        "SingleDepth1BranchRate@10_pre": _mean(single_depth1_pre_list) * 100.0,
        # Percentage of samples whose top-k all fall under a single depth-1 branch after rewrite.
        "SingleDepth1BranchRate@10_post": _mean(single_depth1_post_list) * 100.0,
        # Delta of single-branch concentration rate (positive means stronger concentration after rewrite).
        "Delta_SingleDepth1BranchRate@10": (_mean(single_depth1_post_list) - _mean(single_depth1_pre_list)) * 100.0,
        # Mean share of the dominant depth-1 branch inside top-k before rewrite.
        "MeanMaxDepth1Share@10_pre": _mean(max_depth1_share_pre_list) * 100.0,
        # Mean share of the dominant depth-1 branch inside top-k after rewrite.
        "MeanMaxDepth1Share@10_post": _mean(max_depth1_share_post_list) * 100.0,
        # Delta of dominant depth-1 branch share (positive means stronger concentration after rewrite).
        "Delta_MeanMaxDepth1Share@10": (_mean(max_depth1_share_post_list) - _mean(max_depth1_share_pre_list)) * 100.0,
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
    for d in region_depths:
        d = int(d)
        metrics[f"RegionHitD{d}@10_pre"] = _mean(region_hit_pre_by_depth[d]) * 100.0
        metrics[f"RegionHitD{d}@10_post"] = _mean(region_hit_post_by_depth[d]) * 100.0
        metrics[f"Delta_RegionHitD{d}@10"] = (
            _mean(region_hit_post_by_depth[d]) - _mean(region_hit_pre_by_depth[d])
        ) * 100.0
        metrics[f"RegionJaccardD{d}@10_pre_vs_post"] = _mean(region_jaccard_by_depth[d])
        metrics[f"Delta_RegionHitD{d}@10_givenPreMiss"] = _mean(pre_miss_region_hit_delta_by_depth[d]) * 100.0
    sorted_depths = sorted([int(d) for d in region_depths])
    for d in sorted_depths:
        d_next = d + 1
        if d_next not in region_hit_pre_by_depth:
            continue
        # Intent: diagnose whether failures come from missing the parent region itself (depth d)
        # or from failing to distinguish leaf-level descendants inside the matched parent (depth d+1).
        metrics[f"P_RegionHitD{d_next}@10_given_D{d}_pre"] = (
            _conditional_mean(region_hit_pre_by_depth[d_next], region_hit_pre_by_depth[d]) * 100.0
        )
        metrics[f"P_RegionHitD{d_next}@10_given_D{d}_post"] = (
            _conditional_mean(region_hit_post_by_depth[d_next], region_hit_post_by_depth[d]) * 100.0
        )
        metrics[f"D{d}_Hit_D{d_next}_MissRate@10_pre"] = (
            _mean([
                1.0 if (hd >= 0.5 and hd1 < 0.5) else 0.0
                for hd, hd1 in zip(region_hit_pre_by_depth[d], region_hit_pre_by_depth[d_next])
            ]) * 100.0
        )
        metrics[f"D{d}_Hit_D{d_next}_MissRate@10_post"] = (
            _mean([
                1.0 if (hd >= 0.5 and hd1 < 0.5) else 0.0
                for hd, hd1 in zip(region_hit_post_by_depth[d], region_hit_post_by_depth[d_next])
            ]) * 100.0
        )
        metrics[f"Delta_D{d}_Hit_D{d_next}_MissRate@10"] = (
            metrics[f"D{d}_Hit_D{d_next}_MissRate@10_post"] - metrics[f"D{d}_Hit_D{d_next}_MissRate@10_pre"]
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_pkl", type=str, required=True)
    parser.add_argument("--ours_pkl", type=str, required=True)
    parser.add_argument("--iter_pre", type=int, default=0)
    parser.add_argument("--iter_post", type=int, default=1)
    parser.add_argument("--eval_k", type=int, default=10)
    parser.add_argument("--region_depths", type=str, default="auto",
                        help="Comma-separated depths (e.g., 1,2,3,4) or 'auto' for 1..max gold depth.")
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
    region_depths = _parse_region_depths(args.region_depths, baseline_samples, ours_samples)

    baseline_metrics = _eval_single_run(
        sample_dicts=baseline_samples,
        iter_pre=args.iter_pre,
        iter_post=args.iter_post,
        eval_k=args.eval_k,
        region_depths=region_depths,
    )
    ours_metrics = _eval_single_run(
        sample_dicts=ours_samples,
        iter_pre=args.iter_pre,
        iter_post=args.iter_post,
        eval_k=args.eval_k,
        region_depths=region_depths,
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
        # Prompt-attributed gain in spread across depth-1 branches after rewrite.
        "DeltaPrompt_Delta_MeanUniqueDepth1Branches@10": (
            ours_metrics["Delta_MeanUniqueDepth1Branches@10"] - baseline_metrics["Delta_MeanUniqueDepth1Branches@10"]
        ),
        # Prompt-attributed change in single depth-1 branch concentration rate.
        "DeltaPrompt_Delta_SingleDepth1BranchRate@10": (
            ours_metrics["Delta_SingleDepth1BranchRate@10"] - baseline_metrics["Delta_SingleDepth1BranchRate@10"]
        ),
        # Prompt-attributed change in dominant depth-1 branch share.
        "DeltaPrompt_Delta_MeanMaxDepth1Share@10": (
            ours_metrics["Delta_MeanMaxDepth1Share@10"] - baseline_metrics["Delta_MeanMaxDepth1Share@10"]
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
    for d in region_depths:
        d = int(d)
        did[f"DeltaPrompt_Delta_RegionHitD{d}@10"] = (
            ours_metrics[f"Delta_RegionHitD{d}@10"] - baseline_metrics[f"Delta_RegionHitD{d}@10"]
        )
        did[f"DeltaPrompt_Delta_RegionHitD{d}@10_givenPreMiss"] = (
            ours_metrics[f"Delta_RegionHitD{d}@10_givenPreMiss"] - baseline_metrics[f"Delta_RegionHitD{d}@10_givenPreMiss"]
        )
    sorted_depths = sorted([int(d) for d in region_depths])
    for d in sorted_depths:
        d_next = d + 1
        k_post = f"P_RegionHitD{d_next}@10_given_D{d}_post"
        k_miss = f"Delta_D{d}_Hit_D{d_next}_MissRate@10"
        if k_post in baseline_metrics and k_post in ours_metrics:
            did[f"DeltaPrompt_{k_post}"] = ours_metrics[k_post] - baseline_metrics[k_post]
        if k_miss in baseline_metrics and k_miss in ours_metrics:
            did[f"DeltaPrompt_{k_miss}"] = ours_metrics[k_miss] - baseline_metrics[k_miss]

    report = {
        "meta": {
            "iter_pre": args.iter_pre,
            "iter_post": args.iter_post,
            "eval_k": args.eval_k,
            "region_depths": list(region_depths),
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
