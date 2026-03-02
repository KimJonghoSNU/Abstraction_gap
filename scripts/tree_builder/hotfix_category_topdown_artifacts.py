import argparse
import json
import os
import pickle
import re
from collections import Counter, defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple


DOC_LEAF_PATTERN = re.compile(r".+/.+\.txt$", flags=re.IGNORECASE)


def _is_doc_leaf_id(node_id: object) -> bool:
    text = str(node_id or "").strip()
    if not text:
        return False
    return DOC_LEAF_PATTERN.fullmatch(text) is not None


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _strip_level_prefix(text: str) -> str:
    value = str(text or "").strip()
    if value.startswith("[L") and "] " in value:
        return value.split("] ", 1)[1].strip()
    return value


def _read_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _read_jsonl(path: str) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _write_jsonl(path: str, rows: List[Dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _recompute_degree_maps(edges: List[Dict[str, object]]) -> Tuple[Counter, Counter]:
    outgoing: Counter = Counter()
    incoming: Counter = Counter()
    for edge in edges:
        parent_id = str(edge.get("parent_id", "")).strip()
        child_id = str(edge.get("child_id", "")).strip()
        if (not parent_id) or (not child_id):
            continue
        outgoing[parent_id] += 1
        incoming[child_id] += 1
    return outgoing, incoming


def _prune_dangling_non_doc_nodes(
    *,
    nodes: Dict[str, Dict[str, object]],
    edges: List[Dict[str, object]],
    root_id: str,
) -> Tuple[Dict[str, Dict[str, object]], List[Dict[str, object]], int]:
    removed_total = 0
    current_nodes = dict(nodes)
    current_edges = list(edges)
    while True:
        outgoing, _ = _recompute_degree_maps(current_edges)
        removable: Set[str] = set()
        for node_id, node in current_nodes.items():
            if node_id == root_id:
                continue
            out_deg = int(outgoing.get(node_id, 0))
            if out_deg > 0:
                continue
            if _is_doc_leaf_id(node_id):
                continue
            removable.add(node_id)
        if not removable:
            break
        removed_total += len(removable)
        current_nodes = {nid: row for nid, row in current_nodes.items() if nid not in removable}
        next_edges: List[Dict[str, object]] = []
        for edge in current_edges:
            parent_id = str(edge.get("parent_id", "")).strip()
            child_id = str(edge.get("child_id", "")).strip()
            if (parent_id in removable) or (child_id in removable):
                continue
            next_edges.append(edge)
        current_edges = next_edges
    return current_nodes, current_edges, removed_total


def _normalize_node_rows(
    *,
    nodes: Dict[str, Dict[str, object]],
    edges: List[Dict[str, object]],
    root_id: str,
) -> Dict[str, Dict[str, object]]:
    outgoing, incoming = _recompute_degree_maps(edges)
    out_nodes: Dict[str, Dict[str, object]] = {}
    for node_id, row in nodes.items():
        node = dict(row)
        out_deg = int(outgoing.get(node_id, 0))
        in_deg = int(incoming.get(node_id, 0))
        kind = str(node.get("kind", "")).strip().lower()
        if _is_doc_leaf_id(node_id) and out_deg == 0:
            kind = "leaf"
        elif kind == "leaf" and out_deg > 0:
            kind = "category"
        if node_id == root_id:
            kind = "root"

        raw_label = _normalize_space(str(node.get("label", "")))
        raw_display = node.get("display_id")
        if raw_display is None:
            raw_display_text = ""
        else:
            raw_display_text = _normalize_space(str(raw_display))
        if kind == "category":
            display = _strip_level_prefix(raw_display_text or raw_label or str(node_id))
            label = _strip_level_prefix(raw_label or display)
            node["display_id"] = display
            node["label"] = label
        elif kind == "leaf":
            node["display_id"] = str(node_id)
            node["label"] = str(node_id)
        elif kind == "root":
            node["display_id"] = None
            node["label"] = "Root"

        node["kind"] = kind
        node["num_children"] = out_deg
        node["num_parents"] = in_deg
        out_nodes[node_id] = node
    return out_nodes


def _select_projection_edges(
    edges: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    projection = [edge for edge in edges if bool(edge.get("is_projection_parent", False))]
    if projection:
        return projection
    return list(edges)


def _projection_parent_choice(
    *,
    nodes: Dict[str, Dict[str, object]],
    edges: List[Dict[str, object]],
    root_id: str,
) -> Dict[str, str]:
    best_parent: Dict[str, Tuple[float, str]] = {}
    for edge in edges:
        parent_id = str(edge.get("parent_id", "")).strip()
        child_id = str(edge.get("child_id", "")).strip()
        if (not parent_id) or (not child_id):
            continue
        if parent_id not in nodes or child_id not in nodes:
            continue
        if child_id == root_id:
            continue
        weight = float(edge.get("weight", 0.0) or 0.0)
        prev = best_parent.get(child_id)
        cand = (weight, parent_id)
        if (prev is None) or (cand > prev):
            best_parent[child_id] = cand
    parent_by_child: Dict[str, str] = {}
    for child_id, (_, parent_id) in best_parent.items():
        parent_by_child[child_id] = parent_id
    return parent_by_child


def _ensure_connected_tree_parents(
    *,
    nodes: Dict[str, Dict[str, object]],
    parent_by_child: Dict[str, str],
    root_id: str,
) -> Dict[str, str]:
    outgoing: Dict[str, List[str]] = defaultdict(list)
    for child_id, parent_id in parent_by_child.items():
        outgoing[parent_id].append(child_id)

    visited: Set[str] = set()
    queue: deque[str] = deque([root_id])
    while queue:
        cur = queue.popleft()
        if cur in visited:
            continue
        visited.add(cur)
        for nxt in outgoing.get(cur, []):
            if nxt not in visited:
                queue.append(nxt)

    out = dict(parent_by_child)
    for node_id in nodes.keys():
        if node_id == root_id:
            continue
        if node_id in visited:
            continue
        out[node_id] = root_id
    return out


def _build_tree_from_parent_map(
    *,
    nodes: Dict[str, Dict[str, object]],
    parent_by_child: Dict[str, str],
    root_id: str,
) -> Dict[str, object]:
    children_by_parent: Dict[str, List[str]] = defaultdict(list)
    for child_id, parent_id in parent_by_child.items():
        if child_id == root_id:
            continue
        children_by_parent[parent_id].append(child_id)

    def child_sort_key(node_id: str) -> Tuple[int, int, str]:
        row = nodes[node_id]
        level = int(row.get("level", 999))
        is_leaf = 1 if len(children_by_parent.get(node_id, [])) == 0 else 0
        display_id = row.get("display_id")
        label = str(display_id if display_id is not None else node_id)
        return (level, is_leaf, label)

    def make_node(node_id: str) -> Dict[str, object]:
        row = nodes[node_id]
        raw_children = sorted(children_by_parent.get(node_id, []), key=child_sort_key)
        children = [make_node(child_id) for child_id in raw_children]
        return {
            "id": row.get("display_id"),
            "desc": str(row.get("desc", "") or ""),
            "child": children if children else None,
        }

    tree = make_node(root_id)
    tree["id"] = None
    return tree


def _prune_non_doc_leaves_in_tree(tree: Dict[str, object], *, is_root: bool) -> Optional[Dict[str, object]]:
    children = tree.get("child") or []
    if not children:
        if is_root:
            return tree
        node_id = tree.get("id")
        if _is_doc_leaf_id(node_id):
            return tree
        return None
    kept: List[Dict[str, object]] = []
    for child in children:
        fixed = _prune_non_doc_leaves_in_tree(child, is_root=False)
        if fixed is not None:
            kept.append(fixed)
    if kept:
        tree["child"] = kept
        return tree
    tree["child"] = None
    if is_root:
        return tree
    node_id = tree.get("id")
    if _is_doc_leaf_id(node_id):
        return tree
    return None


def _compute_num_leaves(tree_node: Dict[str, object]) -> int:
    children = tree_node.get("child") or []
    if not children:
        tree_node["num_leaves"] = 1
        return 1
    total = 0
    for child in children:
        total += _compute_num_leaves(child)
    tree_node["num_leaves"] = total
    return total


def _export_node_catalog(tree_dict: Dict[str, object], out_jsonl: str) -> Tuple[int, Counter]:
    _compute_num_leaves(tree_dict)
    rows: List[Dict[str, object]] = []

    def walk(node: Dict[str, object], path: Tuple[int, ...]) -> None:
        children = node.get("child") or []
        node_id = node.get("id")
        raw_desc = _normalize_space(str(node.get("desc", "") or ""))
        if node_id is None:
            desc_with_id = raw_desc
        else:
            desc_with_id = _normalize_space(f"ID: {node_id}. {raw_desc}")
        rec = {
            "path": list(path),
            "depth": len(path),
            "is_leaf": len(children) == 0,
            "num_children": len(children),
            "num_leaves": int(node.get("num_leaves", 1)),
            "id": node_id,
            "desc": desc_with_id,
        }
        rows.append(rec)
        for idx, child in enumerate(children):
            walk(child, (*path, idx))

    walk(tree_dict, ())
    for idx, rec in enumerate(rows):
        rec["registry_idx"] = idx

    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    depth1_path = os.path.splitext(out_jsonl)[0] + "_depth1.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f_all, open(depth1_path, "w", encoding="utf-8") as f_d1:
        for rec in rows:
            line = json.dumps(rec, ensure_ascii=False)
            f_all.write(line + "\n")
            if int(rec["depth"]) == 1:
                f_d1.write(line + "\n")
    depth_counter = Counter([int(x["depth"]) for x in rows])
    return len(rows), depth_counter


def _update_leaf_membership(
    *,
    path: str,
    valid_node_ids: Set[str],
    root_id: str,
) -> Counter:
    if not os.path.exists(path):
        return Counter()
    rows = _read_jsonl(path)
    hist = Counter()
    fixed_rows: List[Dict[str, object]] = []
    for row in rows:
        parent_ids = row.get("selected_parent_node_ids", [])
        long_ids = row.get("selected_parent_long_doc_ids", [])
        scores = row.get("selected_scores", [])
        if not isinstance(parent_ids, list):
            parent_ids = []
        if not isinstance(long_ids, list):
            long_ids = []
        if not isinstance(scores, list):
            scores = []
        selected: List[str] = []
        for pid in parent_ids:
            pid_text = str(pid or "").strip()
            if not pid_text:
                continue
            if pid_text in valid_node_ids:
                selected.append(pid_text)
        if not selected:
            selected = [root_id]
        # Intent: keep one canonical parent per split-doc membership row for tree consistency.
        selected = [selected[0]]
        long_val = str(long_ids[0]) if long_ids else ""
        score_val = float(scores[0]) if scores else 0.0
        new_row = dict(row)
        new_row["selected_parent_node_ids"] = selected
        new_row["selected_parent_long_doc_ids"] = [long_val]
        new_row["selected_scores"] = [round(float(score_val), 6)]
        fixed_rows.append(new_row)
        hist[len(selected)] += 1
    _write_jsonl(path, fixed_rows)
    return hist


def _update_report(
    *,
    report_path: str,
    dataset: str,
    subset: str,
    version: str,
    num_nodes: int,
    num_edges: int,
    level_distribution: Counter,
    multi_parent_distribution: Counter,
    leaf_parent_histogram: Counter,
    num_long_documents: int,
    num_split_documents: int,
    num_tree_nodes: int,
    num_tree_edges: int,
) -> None:
    report: Dict[str, object] = {}
    if os.path.exists(report_path):
        try:
            report = _read_json(report_path)
        except Exception:
            report = {}
    meta = dict(report.get("meta", {})) if isinstance(report.get("meta"), dict) else {}
    meta["dataset"] = dataset
    meta["subset"] = subset
    meta["version"] = version
    meta["mode"] = "hotfix_non_doc_leaf_pruned"
    counts = dict(report.get("counts", {})) if isinstance(report.get("counts"), dict) else {}
    counts["num_long_documents"] = int(num_long_documents)
    counts["num_split_documents"] = int(num_split_documents)
    counts["num_dag_nodes"] = int(num_nodes)
    counts["num_dag_edges"] = int(num_edges)
    counts["num_projection_nodes"] = int(num_tree_nodes)
    counts["num_projection_edges"] = int(num_tree_edges)
    counts["num_tree_nodes"] = int(num_tree_nodes)
    counts["num_tree_edges"] = int(num_tree_edges)
    report["meta"] = meta
    report["counts"] = counts
    report["level_distribution"] = {str(k): int(v) for k, v in sorted(level_distribution.items())}
    report["multi_parent_level_distribution"] = {
        str(k): int(v) for k, v in sorted(multi_parent_distribution.items())
    }
    report["leaf_parent_histogram"] = {
        str(k): int(v) for k, v in sorted(leaf_parent_histogram.items())
    }
    _write_json(report_path, report)


def _node_desc_counts(tree: Dict[str, object]) -> Tuple[int, int]:
    total = 0
    leaves = 0
    stack = [tree]
    while stack:
        node = stack.pop()
        total += 1
        children = node.get("child") or []
        if not children:
            leaves += 1
        else:
            stack.extend(children)
    return total, leaves


def hotfix_one(
    *,
    subset_dir: str,
    dataset: str,
    subset: str,
    version: str,
) -> None:
    version_u = version.replace("-", "_")
    dag_path = os.path.join(subset_dir, f"category_dag_topdown_algo4_{version_u}.json")
    edge_jsonl_path = os.path.join(subset_dir, f"category_dag_edges_topdown_algo4_{version_u}.jsonl")
    membership_path = os.path.join(subset_dir, f"category_leaf_membership_topdown_algo4_{version_u}.jsonl")
    report_path = os.path.join(subset_dir, f"category_build_report_topdown_algo4_{version_u}.json")
    projection_tree_path = os.path.join(subset_dir, f"category_tree_projection_topdown_algo4_{version_u}.pkl")
    runtime_tree_path = os.path.join(subset_dir, f"tree-category-topdown-algo4-{version}.pkl")
    node_catalog_path = os.path.join(subset_dir, f"category_node_catalog_topdown_algo4_{version_u}.jsonl")
    longdoc_path = os.path.join(subset_dir, f"category_longdoc_paths_topdown_algo4_{version_u}.jsonl")

    if not os.path.exists(dag_path):
        print(f"[Skip] missing dag: {dag_path}")
        return

    dag = _read_json(dag_path)
    raw_nodes = dag.get("nodes", [])
    raw_edges = dag.get("edges", [])
    if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
        print(f"[Skip] invalid dag payload: {dag_path}")
        return

    nodes: Dict[str, Dict[str, object]] = {}
    for row in raw_nodes:
        if not isinstance(row, dict):
            continue
        node_id = str(row.get("id", "")).strip()
        if not node_id:
            continue
        nodes[node_id] = dict(row)
    if not nodes:
        print(f"[Skip] no nodes: {dag_path}")
        return

    root_id = ""
    for node_id, row in nodes.items():
        if str(row.get("kind", "")).strip().lower() == "root":
            root_id = node_id
            break
    if (not root_id) and ("L0|root" in nodes):
        root_id = "L0|root"
    if not root_id:
        root_id = sorted(nodes.keys())[0]

    edges: List[Dict[str, object]] = []
    for edge in raw_edges:
        if not isinstance(edge, dict):
            continue
        parent_id = str(edge.get("parent_id", "")).strip()
        child_id = str(edge.get("child_id", "")).strip()
        if (not parent_id) or (not child_id):
            continue
        if parent_id not in nodes or child_id not in nodes:
            continue
        edges.append(dict(edge))

    old_nodes = len(nodes)
    old_edges = len(edges)
    nodes, edges, removed_count = _prune_dangling_non_doc_nodes(
        nodes=nodes,
        edges=edges,
        root_id=root_id,
    )
    nodes = _normalize_node_rows(nodes=nodes, edges=edges, root_id=root_id)

    # Keep deterministic order for stable diffs.
    ordered_nodes = sorted(
        nodes.values(),
        key=lambda x: (int(x.get("level", 999)), str(x.get("id", ""))),
    )
    ordered_edges = sorted(
        edges,
        key=lambda x: (
            str(x.get("parent_id", "")),
            str(x.get("child_id", "")),
        ),
    )
    dag["nodes"] = ordered_nodes
    dag["edges"] = ordered_edges
    _write_json(dag_path, dag)
    _write_jsonl(edge_jsonl_path, ordered_edges)

    projection_edges = _select_projection_edges(ordered_edges)
    parent_by_child = _projection_parent_choice(nodes=nodes, edges=projection_edges, root_id=root_id)
    parent_by_child = _ensure_connected_tree_parents(
        nodes=nodes,
        parent_by_child=parent_by_child,
        root_id=root_id,
    )
    tree_dict = _build_tree_from_parent_map(
        nodes=nodes,
        parent_by_child=parent_by_child,
        root_id=root_id,
    )
    fixed_tree = _prune_non_doc_leaves_in_tree(tree_dict, is_root=True)
    if fixed_tree is None:
        raise RuntimeError(f"Tree prune produced empty tree: {runtime_tree_path}")

    pickle.dump(fixed_tree, open(projection_tree_path, "wb"))
    pickle.dump(fixed_tree, open(runtime_tree_path, "wb"))
    tree_node_count, depth_counter = _export_node_catalog(fixed_tree, node_catalog_path)
    tree_total_nodes, tree_leaf_nodes = _node_desc_counts(fixed_tree)

    leaf_hist = _update_leaf_membership(
        path=membership_path,
        valid_node_ids=set(nodes.keys()),
        root_id=root_id,
    )

    out_deg, in_deg = _recompute_degree_maps(ordered_edges)
    multi_parent = Counter()
    level_dist = Counter()
    for row in ordered_nodes:
        node_id = str(row.get("id", "")).strip()
        level = int(row.get("level", 0) or 0)
        level_dist[level] += 1
        if int(in_deg.get(node_id, 0)) > 1:
            multi_parent[level] += 1

    long_count = 0
    if os.path.exists(longdoc_path):
        with open(longdoc_path, "r", encoding="utf-8") as f:
            for _ in f:
                long_count += 1

    split_count = 0
    if os.path.exists(membership_path):
        with open(membership_path, "r", encoding="utf-8") as f:
            for _ in f:
                split_count += 1

    _update_report(
        report_path=report_path,
        dataset=dataset,
        subset=subset,
        version=version,
        num_nodes=len(ordered_nodes),
        num_edges=len(ordered_edges),
        level_distribution=level_dist,
        multi_parent_distribution=multi_parent,
        leaf_parent_histogram=leaf_hist,
        num_long_documents=long_count,
        num_split_documents=split_count,
        num_tree_nodes=tree_total_nodes,
        num_tree_edges=max(0, tree_total_nodes - 1),
    )

    print(
        "[Hotfix] subset={subset} version={version} removed_dangling={removed} "
        "dag_nodes={old_nodes}->{new_nodes} dag_edges={old_edges}->{new_edges} "
        "tree_nodes={tree_nodes} tree_leaves={tree_leaves} node_catalog_rows={catalog_rows} "
        "max_depth={max_depth}".format(
            subset=subset,
            version=version,
            removed=removed_count,
            old_nodes=old_nodes,
            new_nodes=len(ordered_nodes),
            old_edges=old_edges,
            new_edges=len(ordered_edges),
            tree_nodes=tree_total_nodes,
            tree_leaves=tree_leaf_nodes,
            catalog_rows=tree_node_count,
            max_depth=max(depth_counter.keys()) if depth_counter else -1,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hotfix existing category-topdown algo4 artifacts by pruning dangling non-doc leaves and rebuilding projection tree/node catalog."
    )
    parser.add_argument("--dataset", type=str, default="BRIGHT")
    parser.add_argument("--subset", type=str, required=True)
    parser.add_argument("--trees_root", type=str, default="trees")
    parser.add_argument("--versions", type=str, default="v1,v2,v3")
    args = parser.parse_args()

    subset_dir = os.path.join(args.trees_root, args.dataset, args.subset)
    versions = [x.strip() for x in str(args.versions or "").split(",") if x.strip()]
    if not versions:
        raise ValueError("--versions must not be empty")

    for version in versions:
        hotfix_one(
            subset_dir=subset_dir,
            dataset=args.dataset,
            subset=args.subset,
            version=version,
        )


if __name__ == "__main__":
    main()
