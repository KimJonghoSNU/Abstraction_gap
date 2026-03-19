#!/usr/bin/env python3
"""
Analyze how often round5 hits a local dead-end (no_candidate_children) and then
passively reseats the beam to leftover expandable paths.

Default target
--------------
This script is preconfigured for the round5 experiment:
    - selector_mode=meanscore_global
    - rewrite prompt=agent_executor_v1_icl2
    - REM=replace
    - non-fused-memory runs only

Key definition
--------------
We define a "passive reseat" as:
    1. selector_pick_reason == "no_candidate_children"
    2. selected_branches_after is non-empty and differs from selected_branches_before
    3. every selected_branches_after endpoint lies outside every selected_branches_before subtree

This matches the code path where local child-branch expansion has terminated, while
sample.update() has already reseated the beam from tree-wide leftover expandable paths.

Examples
--------
python scripts/analyze_round5_passive_reseat.py

python scripts/analyze_round5_passive_reseat.py \
    --glob_pattern 'results/BRIGHT/*/round5/**/all_eval_sample_dicts.pkl' \
    --require_substrings round5_mrr_selector_accum_meanscore_global-FT=1000 \
    --require_substrings RPN=agent_executor_v1_icl2-RM=concat-RE=1/RCT=10-RCS=mixed-RGT=10-RSM=meanscore_global-RRrfK=60/RRC=leaf-REM=replace \
    --out_prefix results/BRIGHT/analysis/round5_passive_reseat
"""

import argparse
import glob
import os
import pickle as pkl
import statistics as st
import sys
import warnings
from collections import Counter
from functools import lru_cache
from typing import Any, Dict, List, Sequence, Set, Tuple

import pandas as pd

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

warnings.filterwarnings("ignore", category=UserWarning)
from tree_objects import SemanticNode


DEFAULT_GLOB_PATTERN = "results/BRIGHT/*/round5/**/all_eval_sample_dicts.pkl"
DEFAULT_REQUIRE_SUBSTRINGS = [
    "round5_mrr_selector_accum_meanscore_global-FT=1000",
    "RPN=agent_executor_v1_icl2-RM=concat-RE=1/RCT=10-RCS=mixed-RGT=10-RSM=meanscore_global-RRrfK=60/RRC=leaf-REM=replace",
]
DEFAULT_EXCLUDE_SUBSTRINGS = [
    "fused_memory",
    "rubric",
]


def _resolve_eval_paths(
    inputs: Sequence[str],
    glob_pattern: str,
    require_substrings: Sequence[str],
    exclude_substrings: Sequence[str],
) -> List[str]:
    resolved: List[str] = []

    for path in inputs:
        if not path:
            continue
        abs_path = os.path.abspath(path)
        if os.path.isdir(abs_path):
            candidate = os.path.join(abs_path, "all_eval_sample_dicts.pkl")
            if not os.path.exists(candidate):
                raise FileNotFoundError(f"Directory does not contain all_eval_sample_dicts.pkl: {abs_path}")
            resolved.append(candidate)
            continue
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Input path not found: {abs_path}")
        resolved.append(abs_path)

    if glob_pattern:
        resolved.extend(glob.glob(glob_pattern, recursive=True))

    deduped = sorted({os.path.abspath(path) for path in resolved})
    filtered: List[str] = []
    for path in deduped:
        if require_substrings and any(token not in path for token in require_substrings):
            continue
        if exclude_substrings and any(token in path for token in exclude_substrings):
            continue
        filtered.append(path)

    if not filtered:
        raise FileNotFoundError("No all_eval_sample_dicts.pkl files matched the given filters.")
    return filtered


def _infer_subset_from_path(path: str) -> str:
    parts = os.path.abspath(path).split(os.sep)
    for idx, part in enumerate(parts):
        if part == "results" and (idx + 2) < len(parts):
            return str(parts[idx + 2])
    return "unknown"


def _infer_tree_version_from_path(path: str) -> str:
    path_str = os.path.abspath(path)
    if "TV=top-down" in path_str:
        return "top-down"
    if "TV=bottom-up" in path_str:
        return "bottom-up"
    raise ValueError(f"Could not infer tree version from path: {path}")


def _safe_iter_records(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    records = sample.get("iter_records", []) or []
    return [record for record in records if isinstance(record, dict)]


def _format_pct(numer: int, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return 100.0 * float(numer) / float(denom)


def _path_tuple_list(paths: Sequence[Sequence[int]]) -> List[Tuple[int, ...]]:
    return [tuple(int(x) for x in path) for path in (paths or []) if path]


def _is_prefix(prefix: Tuple[int, ...], path: Tuple[int, ...]) -> bool:
    if len(prefix) > len(path):
        return False
    return path[: len(prefix)] == prefix


def _path_depth_mean(paths: Sequence[Tuple[int, ...]]) -> float:
    if not paths:
        return 0.0
    return float(st.mean(len(path) for path in paths))


def _safe_int_step(value: Any, default: int = 10**9) -> int:
    if value is None:
        return default
    return int(value)


def _semantic_children(node: Any) -> List[Any]:
    if isinstance(node, dict):
        return list(node.get("child", []) or [])
    return list(getattr(node, "child", []) or [])


@lru_cache(maxsize=None)
def _load_semantic_nonleaf_paths(subset: str, tree_version: str) -> Set[Tuple[int, ...]]:
    tree_path = os.path.join("trees", "BRIGHT", subset, f"tree-{tree_version}.pkl")
    with open(tree_path, "rb") as f:
        tree_obj = pkl.load(f)
    if isinstance(tree_obj, dict):
        tree_obj = SemanticNode().load_dict(tree_obj)

    nonleaf_paths: Set[Tuple[int, ...]] = set()

    def _walk(node: Any, path: Tuple[int, ...]) -> None:
        children = _semantic_children(node)
        if children:
            nonleaf_paths.add(path)
            for child_idx, child in enumerate(children):
                _walk(child, (*path, child_idx))

    _walk(tree_obj, ())
    return nonleaf_paths


@lru_cache(maxsize=None)
def _load_child_branch_paths_by_path(subset: str, tree_version: str) -> Dict[Tuple[int, ...], List[Tuple[int, ...]]]:
    tree_path = os.path.join("trees", "BRIGHT", subset, f"tree-{tree_version}.pkl")
    with open(tree_path, "rb") as f:
        tree_obj = pkl.load(f)
    if isinstance(tree_obj, dict):
        tree_obj = SemanticNode().load_dict(tree_obj)

    child_branch_paths_by_path: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = {}

    def _walk(node: Any, path: Tuple[int, ...]) -> None:
        children = _semantic_children(node)
        if children:
            nonleaf_child_paths: List[Tuple[int, ...]] = []
            for child_idx, child in enumerate(children):
                child_path = (*path, child_idx)
                if _semantic_children(child):
                    nonleaf_child_paths.append(child_path)
                _walk(child, child_path)
            if nonleaf_child_paths:
                child_branch_paths_by_path[path] = nonleaf_child_paths

    _walk(tree_obj, ())
    return child_branch_paths_by_path


def _prediction_children(node_dict: Dict[str, Any], max_creation_step: int) -> List[Dict[str, Any]]:
    children = node_dict.get("child", None) or []
    return [
        child
        for child in children
        if _safe_int_step(child.get("creation_step", 10**9)) <= max_creation_step
    ]


def _count_expandable_paths_at_iter_end(
    prediction_tree: Dict[str, Any],
    nonleaf_paths: Set[Tuple[int, ...]],
    iter_idx: int,
) -> int:
    max_creation_step = int(iter_idx) + 1

    def _walk(node_dict: Dict[str, Any]) -> int:
        path_t = tuple(int(x) for x in (node_dict.get("path", []) or []))
        if _safe_int_step(node_dict.get("creation_step", 10**9)) > max_creation_step:
            return 0
        if path_t not in nonleaf_paths:
            return 0
        if node_dict.get("child_relevances", None) is None:
            return 1
        return sum(_walk(child) for child in _prediction_children(node_dict, max_creation_step))

    return _walk(prediction_tree)


def _analyze_paths(paths: Sequence[str]) -> Dict[str, pd.DataFrame]:
    total_queries = 0
    total_iters = 0
    ncc_iters = 0
    passive_reseat_iters = 0
    anyslot_global_reseat_iters = 0
    changed_after_iters = 0
    ncc_empty_after_iters = 0
    queries_with_ncc = 0
    queries_with_passive_reseat = 0
    queries_with_anyslot_global_reseat = 0

    per_iter_total = Counter()
    per_iter_ncc = Counter()
    per_iter_passive = Counter()
    per_iter_anyslot = Counter()
    per_iter_expandable_counts: Dict[int, List[int]] = {}
    per_iter_before_sizes: Dict[int, List[int]] = {}

    per_subset_total = Counter()
    per_subset_ncc = Counter()
    per_subset_passive = Counter()
    per_subset_anyslot = Counter()
    per_subset_queries_with_ncc = Counter()
    per_subset_queries_with_passive = Counter()
    per_subset_queries_with_anyslot = Counter()
    per_subset_expandable_counts: Dict[str, List[int]] = {}
    per_subset_before_sizes: Dict[str, List[int]] = {}

    selector_reason_counts = Counter()
    first_ncc_iter_counts = Counter()
    first_passive_iter_counts = Counter()
    first_anyslot_iter_counts = Counter()

    ncc_before_depths: List[float] = []
    ncc_after_depths: List[float] = []
    passive_before_depths: List[float] = []
    passive_after_depths: List[float] = []
    ncc_selected_before_sizes: List[int] = []
    ncc_expandable_counts: List[int] = []
    ncc_full_beam_sizes: List[int] = []

    matched_rows: List[Dict[str, Any]] = []
    example_rows: List[Dict[str, Any]] = []

    for path in paths:
        subset = _infer_subset_from_path(path)
        tree_version = _infer_tree_version_from_path(path)
        nonleaf_paths = _load_semantic_nonleaf_paths(subset, tree_version)
        child_branch_paths_by_path = _load_child_branch_paths_by_path(subset, tree_version)
        root_branch_children = child_branch_paths_by_path.get((), [])
        matched_rows.append({"subset": subset, "path": path})
        with open(path, "rb") as f:
            samples = pkl.load(f)

        for sample_idx, sample in enumerate(samples):
            total_queries += 1
            had_ncc = False
            had_passive = False
            had_anyslot = False
            first_ncc_iter = None
            first_passive_iter = None
            first_anyslot_iter = None
            original_query = str(sample.get("original_query", "") or "")
            prediction_tree = sample.get("prediction_tree", {}) or {}
            max_beam_size = int(sample.get("max_beam_size", 0) or 0)

            for rec in _safe_iter_records(sample):
                iter_idx = int(rec.get("iter", -1))
                reason = str(rec.get("selector_pick_reason", "") or "")
                selector_reason_counts[reason] += 1
                total_iters += 1
                per_iter_total[iter_idx] += 1
                per_subset_total[subset] += 1

                if reason != "no_candidate_children":
                    continue

                had_ncc = True
                if first_ncc_iter is None:
                    first_ncc_iter = iter_idx

                ncc_iters += 1
                per_iter_ncc[iter_idx] += 1
                per_subset_ncc[subset] += 1

                before = _path_tuple_list(rec.get("selected_branches_before", []))
                after = _path_tuple_list(rec.get("selected_branches_after", []))
                before_set = set(before)
                after_set = set(after)
                if before:
                    local_candidate_paths: List[Tuple[int, ...]] = []
                    seen_candidate_paths: Set[Tuple[int, ...]] = set()
                    for parent_path in before:
                        for child_path in child_branch_paths_by_path.get(tuple(parent_path), []):
                            if child_path in seen_candidate_paths:
                                continue
                            seen_candidate_paths.add(child_path)
                            local_candidate_paths.append(child_path)
                else:
                    local_candidate_paths = list(root_branch_children)
                local_candidate_set = set(local_candidate_paths)

                # Intent: detect the earliest iteration where final beam visibly includes fallback/global reseat paths.
                anyslot_global_reseat = bool(after) and any(path_t not in local_candidate_set for path_t in after)
                if anyslot_global_reseat:
                    anyslot_global_reseat_iters += 1
                    per_iter_anyslot[iter_idx] += 1
                    per_subset_anyslot[subset] += 1
                    had_anyslot = True
                    if first_anyslot_iter is None:
                        first_anyslot_iter = iter_idx

                if not after:
                    ncc_empty_after_iters += 1
                    continue

                if before_set != after_set:
                    changed_after_iters += 1

                ncc_before_depths.append(_path_depth_mean(before))
                ncc_after_depths.append(_path_depth_mean(after))
                ncc_selected_before_sizes.append(len(before))
                if max_beam_size > 0 and len(before) == max_beam_size:
                    ncc_full_beam_sizes.append(1)
                expandable_count = _count_expandable_paths_at_iter_end(
                    prediction_tree=prediction_tree,
                    nonleaf_paths=nonleaf_paths,
                    iter_idx=iter_idx,
                )
                ncc_expandable_counts.append(expandable_count)
                per_iter_expandable_counts.setdefault(iter_idx, []).append(expandable_count)
                per_iter_before_sizes.setdefault(iter_idx, []).append(len(before))
                per_subset_expandable_counts.setdefault(subset, []).append(expandable_count)
                per_subset_before_sizes.setdefault(subset, []).append(len(before))

                # Intent: operationalize "local trajectory broke and beam moved elsewhere" as a no-candidate step
                # whose new beam endpoints all lie outside the previous beam subtrees.
                all_after_outside_before_subtrees = bool(after) and all(
                    not any(_is_prefix(before_path, after_path) for before_path in before)
                    for after_path in after
                )
                passive_reseat = bool(after) and (before_set != after_set) and all_after_outside_before_subtrees

                if passive_reseat:
                    passive_reseat_iters += 1
                    per_iter_passive[iter_idx] += 1
                    per_subset_passive[subset] += 1
                    passive_before_depths.append(_path_depth_mean(before))
                    passive_after_depths.append(_path_depth_mean(after))
                    if first_passive_iter is None:
                        first_passive_iter = iter_idx
                    had_passive = True
                    example_rows.append(
                        {
                            "subset": subset,
                            "path": path,
                            "sample_idx": int(sample_idx),
                            "iter": int(iter_idx),
                            "query_preview": original_query[:160],
                            "before_paths": str([list(p) for p in before]),
                            "after_paths": str([list(p) for p in after]),
                            "before_depth_mean": round(_path_depth_mean(before), 4),
                            "after_depth_mean": round(_path_depth_mean(after), 4),
                            "selected_before_size": int(len(before)),
                            "selected_after_size": int(len(after)),
                            "expandable_paths_count": int(expandable_count),
                            "selector_candidate_branch_count": int(rec.get("selector_candidate_branch_count", 0) or 0),
                            "candidate_branch_count": int(rec.get("candidate_branch_count", 0) or 0),
                        }
                    )

            if had_ncc:
                queries_with_ncc += 1
                per_subset_queries_with_ncc[subset] += 1
                if first_ncc_iter is not None:
                    first_ncc_iter_counts[first_ncc_iter] += 1
            if had_passive:
                queries_with_passive_reseat += 1
                per_subset_queries_with_passive[subset] += 1
                if first_passive_iter is not None:
                    first_passive_iter_counts[first_passive_iter] += 1
            if had_anyslot:
                queries_with_anyslot_global_reseat += 1
                per_subset_queries_with_anyslot[subset] += 1
                if first_anyslot_iter is not None:
                    first_anyslot_iter_counts[first_anyslot_iter] += 1

    overall_df = pd.DataFrame(
        [
            {
                "matched_paths": int(len(paths)),
                "total_queries": int(total_queries),
                "queries_with_anyslot_global_reseat": int(queries_with_anyslot_global_reseat),
                "queries_with_anyslot_global_reseat_pct": _format_pct(queries_with_anyslot_global_reseat, total_queries),
                "queries_with_ncc": int(queries_with_ncc),
                "queries_with_ncc_pct": _format_pct(queries_with_ncc, total_queries),
                "queries_with_passive_reseat": int(queries_with_passive_reseat),
                "queries_with_passive_reseat_pct": _format_pct(queries_with_passive_reseat, total_queries),
                "total_sample_iters": int(total_iters),
                "anyslot_global_reseat_iters": int(anyslot_global_reseat_iters),
                "anyslot_global_reseat_iters_pct": _format_pct(anyslot_global_reseat_iters, total_iters),
                "ncc_iters": int(ncc_iters),
                "ncc_iters_pct": _format_pct(ncc_iters, total_iters),
                "passive_reseat_iters": int(passive_reseat_iters),
                "passive_reseat_iters_pct": _format_pct(passive_reseat_iters, total_iters),
                "passive_given_ncc_pct": _format_pct(passive_reseat_iters, ncc_iters),
                "ncc_changed_after_iters": int(changed_after_iters),
                "ncc_changed_after_pct": _format_pct(changed_after_iters, ncc_iters),
                "ncc_empty_after_iters": int(ncc_empty_after_iters),
                "ncc_empty_after_pct": _format_pct(ncc_empty_after_iters, ncc_iters),
                "ncc_before_depth_mean": round(st.mean(ncc_before_depths), 4) if ncc_before_depths else 0.0,
                "ncc_after_depth_mean": round(st.mean(ncc_after_depths), 4) if ncc_after_depths else 0.0,
                "passive_before_depth_mean": round(st.mean(passive_before_depths), 4) if passive_before_depths else 0.0,
                "passive_after_depth_mean": round(st.mean(passive_after_depths), 4) if passive_after_depths else 0.0,
                "ncc_selected_before_size_mean": round(st.mean(ncc_selected_before_sizes), 4) if ncc_selected_before_sizes else 0.0,
                "ncc_selected_before_full_beam_pct": _format_pct(sum(ncc_full_beam_sizes), len(ncc_selected_before_sizes)),
                "ncc_expandable_paths_mean": round(st.mean(ncc_expandable_counts), 4) if ncc_expandable_counts else 0.0,
                "ncc_expandable_paths_median": int(st.median(ncc_expandable_counts)) if ncc_expandable_counts else 0,
                "ncc_expandable_paths_min": min(ncc_expandable_counts) if ncc_expandable_counts else 0,
                "ncc_expandable_paths_max": max(ncc_expandable_counts) if ncc_expandable_counts else 0,
            }
        ]
    )

    per_iter_rows: List[Dict[str, Any]] = []
    for iter_idx in sorted(per_iter_total.keys()):
        total = int(per_iter_total[iter_idx])
        ncc = int(per_iter_ncc[iter_idx])
        passive = int(per_iter_passive[iter_idx])
        per_iter_rows.append(
            {
                "iter": int(iter_idx),
                "total": total,
                "anyslot_global_reseat_iters": int(per_iter_anyslot[iter_idx]),
                "anyslot_global_reseat_pct": _format_pct(int(per_iter_anyslot[iter_idx]), total),
                "ncc_iters": ncc,
                "ncc_pct": _format_pct(ncc, total),
                "passive_reseat_iters": passive,
                "passive_reseat_pct": _format_pct(passive, total),
                "passive_given_ncc_pct": _format_pct(passive, ncc),
                "ncc_selected_before_size_mean": round(st.mean(per_iter_before_sizes.get(iter_idx, [0])), 4) if per_iter_before_sizes.get(iter_idx) else 0.0,
                "ncc_expandable_paths_mean": round(st.mean(per_iter_expandable_counts.get(iter_idx, [0])), 4) if per_iter_expandable_counts.get(iter_idx) else 0.0,
            }
        )

    per_subset_rows: List[Dict[str, Any]] = []
    for subset in sorted(per_subset_total.keys()):
        total = int(per_subset_total[subset])
        ncc = int(per_subset_ncc[subset])
        passive = int(per_subset_passive[subset])
        per_subset_rows.append(
            {
                "subset": subset,
                "total": total,
                "anyslot_global_reseat_iters": int(per_subset_anyslot[subset]),
                "anyslot_global_reseat_pct": _format_pct(int(per_subset_anyslot[subset]), total),
                "ncc_iters": ncc,
                "ncc_pct": _format_pct(ncc, total),
                "passive_reseat_iters": passive,
                "passive_reseat_pct": _format_pct(passive, total),
                "passive_given_ncc_pct": _format_pct(passive, ncc),
                "queries_with_ncc": int(per_subset_queries_with_ncc[subset]),
                "queries_with_passive_reseat": int(per_subset_queries_with_passive[subset]),
                "queries_with_anyslot_global_reseat": int(per_subset_queries_with_anyslot[subset]),
                "ncc_selected_before_size_mean": round(st.mean(per_subset_before_sizes.get(subset, [0])), 4) if per_subset_before_sizes.get(subset) else 0.0,
                "ncc_expandable_paths_mean": round(st.mean(per_subset_expandable_counts.get(subset, [0])), 4) if per_subset_expandable_counts.get(subset) else 0.0,
            }
        )

    first_ncc_df = pd.DataFrame(
        [
            {"first_iter": int(iter_idx), "queries": int(count)}
            for iter_idx, count in sorted(first_ncc_iter_counts.items())
        ]
    )
    first_passive_df = pd.DataFrame(
        [
            {"first_iter": int(iter_idx), "queries": int(count)}
            for iter_idx, count in sorted(first_passive_iter_counts.items())
        ]
    )
    first_anyslot_df = pd.DataFrame(
        [
            {"first_iter": int(iter_idx), "queries": int(count)}
            for iter_idx, count in sorted(first_anyslot_iter_counts.items())
        ]
    )
    reasons_df = pd.DataFrame(
        [
            {"selector_pick_reason": str(reason), "count": int(count)}
            for reason, count in selector_reason_counts.most_common()
        ]
    )
    examples_df = pd.DataFrame(example_rows).sort_values(["subset", "iter", "sample_idx"]).head(50)

    return {
        "matched_paths_df": pd.DataFrame(matched_rows),
        "overall_df": overall_df,
        "per_iter_df": pd.DataFrame(per_iter_rows),
        "per_subset_df": pd.DataFrame(per_subset_rows),
        "first_ncc_df": first_ncc_df,
        "first_passive_df": first_passive_df,
        "first_anyslot_df": first_anyslot_df,
        "reasons_df": reasons_df,
        "examples_df": examples_df,
    }


def _print_df(title: str, df: pd.DataFrame) -> None:
    print(f"\n[{title}]")
    if df.empty:
        print("No rows.")
        return
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def _save_df(df: pd.DataFrame, path: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze how often round5 passively reseats the beam after no_candidate_children."
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=[],
        help="Optional list of run directories or all_eval_sample_dicts.pkl paths.",
    )
    parser.add_argument(
        "--glob_pattern",
        type=str,
        default=DEFAULT_GLOB_PATTERN,
        help="Glob used to discover all_eval_sample_dicts.pkl files.",
    )
    parser.add_argument(
        "--require_substrings",
        nargs="*",
        default=DEFAULT_REQUIRE_SUBSTRINGS,
        help="Only keep paths that contain all of these substrings.",
    )
    parser.add_argument(
        "--exclude_substrings",
        nargs="*",
        default=DEFAULT_EXCLUDE_SUBSTRINGS,
        help="Drop any path that contains one of these substrings.",
    )
    parser.add_argument(
        "--print_paths",
        action="store_true",
        help="Print all matched evaluation paths before summaries.",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default=None,
        help="Optional prefix for CSV outputs. Example: results/BRIGHT/analysis/round5_passive_reseat",
    )
    args = parser.parse_args()

    resolved_paths = _resolve_eval_paths(
        inputs=args.inputs,
        glob_pattern=args.glob_pattern,
        require_substrings=args.require_substrings,
        exclude_substrings=args.exclude_substrings,
    )
    analysis = _analyze_paths(resolved_paths)

    if args.print_paths:
        _print_df("Matched Paths", analysis["matched_paths_df"])
    _print_df("Overall", analysis["overall_df"])
    _print_df("Per Iter", analysis["per_iter_df"])
    _print_df("Per Subset", analysis["per_subset_df"])
    _print_df("First Any-Slot Global Reseat Iter", analysis["first_anyslot_df"])
    _print_df("First NCC Iter", analysis["first_ncc_df"])
    _print_df("First Passive Reseat Iter", analysis["first_passive_df"])
    _print_df("Selector Reasons", analysis["reasons_df"])
    _print_df("Passive Reseat Examples", analysis["examples_df"])

    if args.out_prefix:
        _save_df(analysis["matched_paths_df"], f"{args.out_prefix}_matched_paths.csv")
        _save_df(analysis["overall_df"], f"{args.out_prefix}_overall.csv")
        _save_df(analysis["per_iter_df"], f"{args.out_prefix}_per_iter.csv")
        _save_df(analysis["per_subset_df"], f"{args.out_prefix}_per_subset.csv")
        _save_df(analysis["first_anyslot_df"], f"{args.out_prefix}_first_anyslot_iter.csv")
        _save_df(analysis["first_ncc_df"], f"{args.out_prefix}_first_ncc_iter.csv")
        _save_df(analysis["first_passive_df"], f"{args.out_prefix}_first_passive_iter.csv")
        _save_df(analysis["reasons_df"], f"{args.out_prefix}_selector_reasons.csv")
        _save_df(analysis["examples_df"], f"{args.out_prefix}_examples.csv")
        print(f"\nSaved CSVs with prefix: {args.out_prefix}")


if __name__ == "__main__":
    main()
