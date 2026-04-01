#!/usr/bin/env python3
"""Unify round6 EMR ended-reseat result paths to the frontiercum target path."""

import argparse
import glob
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence


SOURCE_SUFFIX = "REmrM=True-RRrfK=60-RRC=leaf-REM=replace"
TARGET_SUFFIX = SOURCE_SUFFIX + "-RB=frontiercum_qstate_v1"
SOURCE_META_RENAMES = {
    "hparams.json": "hparams.source_replace.json",
    "run.log": "run.source_replace.log",
}
REQUIRED_TOKENS = (
    "round6_mrr_selector_accum_meanscore_global_expandable_ended_reseat_emr",
    "agent_executor_v1_icl2_emr_memory",
)
EXCLUDE_TOKENS = (
    "frontiercum_qstate-RMP=",
)


def _find_candidate_pairs(base_dir: Path) -> List[Dict[str, Optional[Path]]]:
    status_by_subset: Dict[str, Dict[str, Optional[Path]]] = {}
    pattern = str(base_dir / "*" / "round6" / "**")
    for raw_path in glob.glob(pattern, recursive=True):
        path = Path(raw_path)
        if not path.is_dir():
            continue
        path_s = str(path)
        if any(token not in path_s for token in REQUIRED_TOKENS):
            continue
        if any(token in path_s for token in EXCLUDE_TOKENS):
            continue
        if not (path_s.endswith(SOURCE_SUFFIX) or path_s.endswith(TARGET_SUFFIX)):
            continue
        subset = path.parts[2] if len(path.parts) > 2 else "unknown"
        entry = status_by_subset.setdefault(subset, {"subset": subset, "source": None, "target": None})
        if path_s.endswith(TARGET_SUFFIX):
            entry["target"] = path
        else:
            entry["source"] = path
    return [status_by_subset[key] for key in sorted(status_by_subset)]


def _write_source_md(
    *,
    target_dir: Path,
    action: str,
    source_dir: Path,
    moved_files: Sequence[str],
    renamed_meta_files: Sequence[str],
) -> None:
    content = [
        "# Source provenance",
        "",
        f"- action: `{action}`",
        f"- canonical_target: `{target_dir}`",
        f"- original_source: `{source_dir}`",
        "",
        "## Moved files",
    ]
    if moved_files:
        content.extend([f"- `{name}`" for name in moved_files])
    else:
        content.append("- none")
    content.extend(["", "## Renamed source metadata"])
    if renamed_meta_files:
        content.extend([f"- `{name}`" for name in renamed_meta_files])
    else:
        content.append("- none")
    (target_dir / "source.md").write_text("\n".join(content) + "\n", encoding="utf-8")


def _merge_source_into_target(
    *,
    source_dir: Path,
    target_dir: Path,
    apply: bool,
) -> Dict[str, object]:
    moved_files: List[str] = []
    renamed_meta_files: List[str] = []
    skipped_existing: List[str] = []

    for item in sorted(source_dir.iterdir(), key=lambda p: p.name):
        target_path = target_dir / item.name
        if item.name in SOURCE_META_RENAMES:
            renamed_name = SOURCE_META_RENAMES[item.name]
            renamed_target = target_dir / renamed_name
            # Intent: keep target metadata authoritative while preserving the source-side metadata for provenance.
            if renamed_target.exists():
                skipped_existing.append(renamed_name)
                continue
            renamed_meta_files.append(renamed_name)
            if apply:
                shutil.move(str(item), str(renamed_target))
            continue
        if target_path.exists():
            # Intent: never overwrite target result artifacts during path unification; skip and report instead.
            skipped_existing.append(item.name)
            continue
        moved_files.append(item.name)
        if apply:
            shutil.move(str(item), str(target_path))

    if apply:
        _write_source_md(
            target_dir=target_dir,
            action="merge",
            source_dir=source_dir,
            moved_files=moved_files,
            renamed_meta_files=renamed_meta_files,
        )
        for root, dirs, files in os.walk(source_dir, topdown=False):
            if not dirs and not files:
                Path(root).rmdir()

    return {
        "moved_files": moved_files,
        "renamed_meta_files": renamed_meta_files,
        "skipped_existing": skipped_existing,
    }


def _rename_source_to_target(*, source_dir: Path, target_dir: Path, apply: bool) -> Dict[str, object]:
    if apply:
        source_dir.rename(target_dir)
        _write_source_md(
            target_dir=target_dir,
            action="rename",
            source_dir=source_dir,
            moved_files=[path.name for path in sorted(target_dir.iterdir(), key=lambda p: p.name) if path.name != "source.md"],
            renamed_meta_files=[],
        )
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Rename/merge round6 EMR paths to the frontiercum target suffix.")
    parser.add_argument("--base_dir", default="results/BRIGHT", help="Base BRIGHT results directory")
    parser.add_argument("--dry_run", action="store_true", default=True, help="Print intended actions without mutating files")
    parser.add_argument("--apply", action="store_true", help="Apply the rename/merge operations")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    apply = bool(args.apply)
    dry_run = not apply
    pairs = _find_candidate_pairs(base_dir)

    counts: Counter[str] = Counter()
    for pair in pairs:
        subset = str(pair["subset"])
        source_dir = pair.get("source")
        target_dir = pair.get("target")

        if source_dir is not None and target_dir is None:
            target_dir = Path(str(source_dir) + "-RB=frontiercum_qstate_v1")
            action = "rename"
            extra = {}
            if apply:
                extra = _rename_source_to_target(source_dir=source_dir, target_dir=target_dir, apply=True)
        elif source_dir is None and target_dir is not None:
            action = "skip_target_exists"
            extra = {}
        elif source_dir is not None and target_dir is not None:
            action = "merge"
            extra = _merge_source_into_target(source_dir=source_dir, target_dir=target_dir, apply=apply)
        else:
            action = "error"
            extra = {"reason": "neither source nor target exists"}

        counts[action] += 1
        print(f"[{action}] subset={subset}")
        if source_dir is not None:
            print(f"  source={source_dir}")
        if target_dir is not None:
            print(f"  target={target_dir}")
        if extra:
            for key, value in extra.items():
                print(f"  {key}={value}")

    print("\nSummary")
    print(f"  mode={'apply' if apply else 'dry_run'}")
    for key in ("rename", "merge", "skip_target_exists", "error"):
        print(f"  {key}={counts.get(key, 0)}")


if __name__ == "__main__":
    main()
