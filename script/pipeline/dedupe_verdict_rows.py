"""Remove duplicated verdict rows, keeping the first scoring of each response.

    uv run python script/pipeline/dedupe_verdict_rows.py --check out/r2/score
    uv run python script/pipeline/dedupe_verdict_rows.py out/r2/score

A duplicate here means two verdict rows sharing (prompt_id, sample, judge,
level). It appears when two scoring passes overlap: each reads its resume set
once at the start, so both decide the same response is unscored.

Keeping the first is correct in this case and it is worth being precise about
why, because the response-level version of this bug was NOT safe to fix that
way. There, two rows described two different generations of one cell and there
was no way to tell which survived deduplication, so both had to go and the cell
was rescored. Here the response text is identical, the two rows are two honest
scorings of the same text, and either is a valid verdict. The first is kept and
the count is reported, never silently dropped.

The original is preserved alongside as ``.preduped`` before anything is
rewritten, and each file is replaced atomically.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
from pathlib import Path


def _key(row: dict) -> tuple:
    return (row["prompt_id"], row["s"], row.get("judge"), row.get("level"))


def report(path: Path) -> dict:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    keys = collections.Counter(_key(r) for r in rows)
    return {"rows": len(rows), "unique": len(keys),
            "duplicated_keys": sum(1 for n in keys.values() if n > 1),
            "extra_rows": len(rows) - len(keys)}


def dedupe(path: Path) -> dict:
    before = report(path)
    if before["extra_rows"] == 0:
        return {"path": str(path), "changed": False, **before}
    rows = [json.loads(line) for line in path.open() if line.strip()]
    seen: set[tuple] = set()
    kept = []
    for row in rows:
        key = _key(row)
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
    shutil.copy2(path, path.with_suffix(path.suffix + ".preduped"))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        for row in kept:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return {"path": str(path), "changed": True, "before": before, "after": report(path)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path, help="directory searched for verdicts_*.jsonl")
    ap.add_argument("--check", action="store_true", help="report only, change nothing")
    args = ap.parse_args()

    files = sorted(args.root.rglob("verdicts_*.jsonl"))
    if not files:
        print(f"no verdict files under {args.root}")
        return 1
    total_extra = 0
    for path in files:
        if args.check:
            info = report(path)
            total_extra += info["extra_rows"]
            print(f"{path}: {info}")
            continue
        result = dedupe(path)
        if result.get("changed"):
            total_extra += result["before"]["extra_rows"]
        print(json.dumps(result, sort_keys=True))
    print(f"\nduplicate verdict rows {'found' if args.check else 'removed'}: {total_extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
