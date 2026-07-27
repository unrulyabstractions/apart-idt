"""Remove duplicated response rows from a stage 4 file, keeping the first of each.

    uv run python script/pipeline/dedupe_response_rows.py FILE [FILE ...]
    uv run python script/pipeline/dedupe_response_rows.py --check FILE

A duplicate can only appear when two samplers wrote the same file at once, which
happened once in this run when a restart raced the process it replaced. Both
copies are honest draws from the same cell, so keeping the first is unbiased;
what is not acceptable is leaving them in, because stage 6 pools a cell's
samples and a duplicated row would weight that cell twice.

The original is preserved alongside as ``.predupe`` before anything is rewritten,
and the file is replaced atomically.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
from pathlib import Path


def load(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def report(path: Path) -> dict:
    rows = load(path)
    keys = collections.Counter((r["prompt_id"], r["s"]) for r in rows)
    cells = collections.Counter(r["prompt_id"] for r in rows)
    return {
        "rows": len(rows),
        "unique": len(keys),
        "duplicated_keys": sum(1 for n in keys.values() if n > 1),
        "cells": len(cells),
        "cells_over_four": sum(1 for n in cells.values() if n > 4),
    }


def dedupe(path: Path) -> dict:
    before = report(path)
    if before["duplicated_keys"] == 0:
        return {"path": str(path), "changed": False, **before}
    rows = load(path)
    seen: set[tuple[str, int]] = set()
    kept = []
    for row in rows:
        key = (row["prompt_id"], row["s"])
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
    shutil.copy2(path, path.with_suffix(path.suffix + ".predupe"))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        for row in kept:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return {"path": str(path), "changed": True, "before": before, "after": report(path)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--check", action="store_true", help="report only, change nothing")
    args = ap.parse_args()
    bad = 0
    for path in args.files:
        if not path.exists():
            print(f"missing {path}")
            bad += 1
            continue
        if args.check:
            print(path, report(path))
            continue
        result = dedupe(path)
        print(json.dumps(result, sort_keys=True))
    return bad


if __name__ == "__main__":
    raise SystemExit(main())
