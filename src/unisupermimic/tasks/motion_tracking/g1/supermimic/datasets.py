"""Explicit sequence-group splits and content identities for motion experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def motion_identity(paths: list[str]) -> list[dict[str, str]]:
    return [
        dict(
            path=str(Path(path).resolve()),
            sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        )
        for path in paths
    ]


def load_motion_split(manifest: str | Path, split: str) -> tuple[list[str], dict[str, Any]]:
    """Read {split: [{path, source_group}]} and reject group/content leakage."""
    path = Path(manifest).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or split not in data:
        raise ValueError(f"motion manifest must contain split {split!r}")
    groups: dict[str, str] = {}
    contents: dict[str, str] = {}
    resolved: dict[str, list[dict[str, str]]] = {}
    for name, entries in data.items():
        if not isinstance(entries, list):
            raise ValueError(f"split {name!r} must be a list")
        resolved[name] = []
        used_paths: set[str] = set()
        for entry in entries:
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("path"), str)
                or not isinstance(entry.get("source_group"), str)
                or not entry["source_group"]
            ):
                raise ValueError("each motion requires path and nonempty source_group")
            motion_path = Path(entry["path"])
            motion_path = (
                (path.parent / motion_path).resolve()
                if not motion_path.is_absolute()
                else motion_path.resolve()
            )
            group = entry["source_group"]
            if group in groups and groups[group] != name:
                raise ValueError(f"source_group {group!r} leaks across splits")
            identity = motion_identity([str(motion_path)])[0]
            digest = identity["sha256"]
            if digest in contents and contents[digest] != name:
                raise ValueError(f"identical motion content leaks across splits: {motion_path}")
            if str(motion_path) in used_paths:
                raise ValueError(f"duplicate motion path in split {name!r}: {motion_path}")
            groups[group], contents[digest] = name, name
            used_paths.add(str(motion_path))
            resolved[name].append({**identity, "source_group": group})
    selected = resolved[split]
    if not selected:
        raise ValueError(f"motion split {split!r} is empty")
    return [item["path"] for item in selected], {
        "manifest": str(path),
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "split": split,
        "motions": selected,
    }
