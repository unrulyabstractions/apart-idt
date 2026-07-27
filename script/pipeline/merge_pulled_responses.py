"""Merge per-box staged pulls into the canonical run tree, without losing rows.

    uv run python script/pipeline/merge_pulled_responses.py tmp/pull_stage out/r2/score

Every box is pulled into its own staging directory first, then merged here. A
straight rsync from each box into one directory does not work: two boxes can
hold the same file name for one run, because a box that finished a seat keeps
its copy while another box now owns that seat, and the second pull overwrites
the first. That loss is invisible at the time and surfaces much later as verdict
rows outnumbering response rows.

Merging is by row identity, not by file. Response rows are keyed on
(prompt_id, sample) and verdict rows on (prompt_id, sample, judge, level), the
same keys the samplers and the judge resume on. The first occurrence wins and
the rest are counted. Local rows are read first, so a row already home is never
replaced by a remote copy of itself.

Only files a box legitimately produces are merged: responses, verdicts, and
prompt_sets. Artifacts computed locally, above all elicitation reports, are
never taken from a box, because a box holds a pre-extraction stub of those.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

MERGEABLE = ("responses_", "verdicts_")


def _key(row: dict, kind: str) -> tuple:
    if kind == "verdicts":
        return (row["prompt_id"], row["s"], row.get("judge"), row.get("level"))
    return (row["prompt_id"], row["s"])


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def merge_file(target: Path, sources: list[Path], kind: str) -> dict:
    seen: set[tuple] = set()
    merged: list[dict] = []
    added_from: dict[str, int] = {}
    for path in [target, *sources]:
        added = 0
        for row in _load(path):
            key = _key(row, kind)
            if key in seen:
                continue
            seen.add(key)
            merged.append(row)
            added += 1
        if path is not target and added:
            added_from[path.parent.parent.name] = added
    if not merged:
        return {}
    before = len(_load(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".merging")
    with tmp.open("w") as handle:
        for row in merged:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    tmp.replace(target)
    return {"file": str(target), "before": before, "after": len(merged),
            "added_from": added_from}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("staging", type=Path)
    ap.add_argument("destination", type=Path)
    args = ap.parse_args()

    if not args.staging.exists():
        print("nothing staged")
        return 0

    # run dir -> file name -> the staged copies of it
    plan: dict[tuple[str, str], list[Path]] = {}
    for box_dir in sorted(p for p in args.staging.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in box_dir.iterdir() if p.is_dir()):
            for path in sorted(run_dir.iterdir()):
                if path.is_file() and path.name.startswith(MERGEABLE):
                    plan.setdefault((run_dir.name, path.name), []).append(path)
                elif path.name == "prompt_sets.json":
                    # Never overwrite a local prompt_sets.json. A box holds the
                    # list IT was launched with, and a run whose candidate list
                    # was pinned locally after the fact would be silently reverted
                    # to the box's older copy, which is how organism b's record
                    # disagreed with its own rows twice.
                    dest = args.destination / run_dir.name / path.name
                    if not dest.exists():
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, dest)

    total_added = 0
    for (run, name), sources in sorted(plan.items()):
        kind = "verdicts" if name.startswith("verdicts_") else "responses"
        result = merge_file(args.destination / run / name, sources, kind)
        if result and result["after"] != result["before"]:
            total_added += result["after"] - result["before"]
            print(f"  merged {run}/{name}: {result['before']} -> {result['after']} "
                  f"(+{result['after'] - result['before']}) {result['added_from']}")
    print(f"  merge added {total_added} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
