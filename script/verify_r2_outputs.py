"""Check every r2 artifact before any number is read off it.

    uv run python script/verify_r2_outputs.py --out-root out/r2

Exits non-zero when anything is wrong, so it can gate the comparison stage. The
checks are the ones that would silently corrupt a result rather than crash it:

* a duplicated (prompt, sample) response row, or a duplicated
  (prompt, sample, judge, level) verdict row, which double-counts a cell;
* a target and base scored on different instruction sets, which would pair the
  wrong strata in the registered test;
* a cell whose sample count is not what the design asked for;
* verdict rows whose axis set does not match the frozen registry;
* null verdicts and unscorable responses, which are reported as counts because
  they are legitimate but must never be silently imputed.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

#: Fallback only. The real value is read from each run's prompt_sets.json, so a
#: run collected at a different sample count is checked against its own design.
EXPECTED_SAMPLES = 4


def _rows(path: Path):
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _seat_files(run: Path, kind: str) -> dict[str, Path]:
    return {p.name[len(kind) + 1 : -6]: p for p in sorted(run.glob(f"{kind}_*.jsonl"))}


def check_run(run: Path, axes: list[str], design: dict | None = None) -> tuple[list[str], dict]:
    problems: list[str] = []
    stats: dict = {"run": run.name}
    design = design or {}
    want_cells = design.get("n_prompts")
    want_samples = design.get("samples_per_cell", EXPECTED_SAMPLES)
    want_principals = set((design.get("principals") or {}).keys())
    responses = _seat_files(run, "responses")
    verdicts = _seat_files(run, "verdicts")
    if not responses:
        return [f"{run.name}: no responses"], stats

    instruction_sets = {}
    for seat, path in responses.items():
        rows = _rows(path)
        keys = collections.Counter((r["prompt_id"], r["s"]) for r in rows)
        dupes = [k for k, n in keys.items() if n > 1]
        by_cell = collections.Counter(r["prompt_id"] for r in rows)
        short = [p for p, n in by_cell.items() if n != want_samples]
        seat_principals = {r["principal"] for r in rows}
        instruction_sets[seat] = {r["instruction_id"] for r in rows}
        stats[f"responses_{seat}"] = {
            "rows": len(rows),
            "unique": len(keys),
            "cells": len(by_cell),
            "failed": sum(1 for r in rows if r.get("failed")),
            "empty": sum(1 for r in rows if not r.get("text", "").strip()),
            "systems": len({r.get("system_id") for r in rows}),
        }
        if dupes:
            problems.append(f"{run.name}/{seat}: {len(dupes)} duplicated response rows")
        if short:
            problems.append(
                f"{run.name}/{seat}: {len(short)} cells not at {want_samples} samples")
        # A cell with no rows at all has no key to be short, so completeness is
        # checked against the design rather than against what happens to exist.
        if want_cells and len(by_cell) != want_cells:
            problems.append(
                f"{run.name}/{seat}: {len(by_cell)} of {want_cells} cells present "
                f"({want_cells - len(by_cell)} never sampled)")
        if want_principals and seat_principals != want_principals:
            missing = sorted(want_principals - seat_principals)
            extra = sorted(seat_principals - want_principals)
            problems.append(
                f"{run.name}/{seat}: candidate set differs from prompt_sets.json; "
                f"missing {missing[:4]}, unexpected {extra[:4]}")

    if len(instruction_sets) < 2:
        problems.append(
            f"{run.name}: only {len(instruction_sets)} seat(s) present, so there is "
            "no matched pair and the registered test cannot run")
    if len(instruction_sets) == 2:
        a, b = instruction_sets.values()
        # The two arms must also agree on WHO was audited, not only on which
        # instructions were asked. A shortlist regenerated between the two arms
        # gives one seat a candidate the other never saw.
        seats = list(responses)
        cand = {s: {r["principal"] for r in _rows(responses[s])} for s in seats}
        if cand[seats[0]] != cand[seats[1]]:
            problems.append(
                f"{run.name}: the two seats were collected against DIFFERENT "
                f"candidate lists ({sorted(cand[seats[0]] ^ cand[seats[1]])[:4]}), "
                "so their cells cannot be paired")
        if a != b:
            problems.append(
                f"{run.name}: target and base cover different instruction sets "
                f"({len(a ^ b)} differ), so the registered test cannot pair them")

    # Verdict coverage. A seat can have every response yet only part of them
    # scored, and stage 6 reads verdicts, not responses. Running the test in that
    # window silently answers a different question: calibration_typed was tested
    # at 22:06 while its base verdicts were still being written at 22:08, and the
    # statistic it produced named the principal on a NEGATIVE excess that the
    # complete data does not show. Completeness of responses is not completeness.
    for seat, rpath in responses.items():
        vpath = run / f"verdicts_{seat}.jsonl"
        n_resp = len(_rows(rpath))
        n_verd = len({(r["prompt_id"], r["s"]) for r in _rows(vpath)}) if vpath.exists() else 0
        stats[f"coverage_{seat}"] = {"responses": n_resp, "scored": n_verd}
        if n_verd < n_resp:
            problems.append(
                f"{run.name}/{seat}: only {n_verd} of {n_resp} responses are scored, "
                "so stage 6 would test a subset")

    axis_set = set(axes)
    for seat, path in verdicts.items():
        rows = _rows(path)
        if not rows:
            continue
        keys = collections.Counter(
            (r["prompt_id"], r["s"], r.get("judge"), r.get("level")) for r in rows)
        dupes = [k for k, n in keys.items() if n > 1]
        nulls = sum(1 for r in rows for v in r["verdicts"].values() if v is None)
        cells = sum(len(r["verdicts"]) for r in rows)
        wrong_axes = sum(1 for r in rows if set(r["verdicts"]) != axis_set)
        stats[f"verdicts_{seat}"] = {
            "rows": len(rows), "unique": len(keys), "cells": cells,
            "nulls": nulls, "levels": sorted({r.get("level") for r in rows}),
        }
        if dupes:
            problems.append(f"{run.name}/{seat}: {len(dupes)} duplicated verdict rows")
        if wrong_axes:
            problems.append(
                f"{run.name}/{seat}: {wrong_axes} verdict rows whose axis set "
                "does not match the frozen registry")
    return problems, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", default="out/r2")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    root = Path(args.out_root)
    score_root = root / "score"
    if not score_root.exists():
        print(f"no {score_root}; nothing to check")
        return 1

    all_problems: list[str] = []
    for run in sorted(p for p in score_root.iterdir() if p.is_dir()):
        # "__v" directories are per-box staging for a seat split across boxes by
        # system variant. Their rows are merged into the canonical run by
        # merge_pulled_responses.py and verified there; on their own they hold one
        # seat and a slice of the variants, so checking them as runs reports
        # failures that are not real.
        if "__v" in run.name:
            continue
        if not (run / "prompt_sets.json").exists():
            print(f"== {run.name}: not started")
            continue
        sets = json.loads((run / "prompt_sets.json").read_text())
        condition = sets["condition"]
        axes = [a["axis_id"] for a in json.loads(
            (root / "conjecture" / condition / "scoring_questions.json").read_text())["axes"]]
        problems, stats = check_run(run, axes, sets)
        all_problems += problems
        if not args.quiet:
            print(f"== {run.name} ({condition}, {len(axes)} axes)")
            for key, value in stats.items():
                if key != "run":
                    print(f"   {key}: {value}")
    if all_problems:
        print("\nPROBLEMS")
        for p in all_problems:
            print(f"  {p}")
        return 1
    print("\nall checked artifacts are internally consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
