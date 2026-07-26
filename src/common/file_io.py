"""JSON / JSONL persistence shared by every stage.

Stages hand off through files on disk rather than in-memory objects so that a
long run can be interrupted, resumed, and audited by hand: every intermediate
artifact is human-readable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

__all__ = ["ensure_parent", "load_json", "save_json", "read_jsonl", "append_jsonl"]


def ensure_parent(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: str | Path, payload: Any, indent: int = 2) -> Path:
    resolved = ensure_parent(path)
    resolved.write_text(json.dumps(payload, indent=indent, sort_keys=True), encoding="utf-8")
    return resolved


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield one dict per non-empty line; missing file yields nothing."""
    resolved = Path(path)
    if not resolved.exists():
        return
    with resolved.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    resolved = ensure_parent(path)
    with resolved.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return resolved
