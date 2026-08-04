"""Rescore existing responses with a different judge seat, into a separate tree.

    uv run python script/pipeline/rejudge_with_seat.py \
        --judge-config configs/judge_nano.json --out-root out/r2_nano/score

The AuditBench run is judged by a different seat from every run already in the
paper, so its numbers are internally valid and not comparable to theirs. This
rescores the existing responses under one seat, which makes every run
comparable again without regenerating a single reply.

Nothing is overwritten. Verdict rows carry the judge that produced them, but
stage 6 groups rows by level alone and never by judge, so appending a second
seat's rows to an existing file would build one table out of two judges. The
new verdicts therefore go to a new root, and the original tree is opened
read-only.

The axis registry is checked against the axis ids already present in the old
verdicts before anything is scored. A registry that does not match the one the
responses were first scored under produces a table with a handful of columns
filled and the rest empty, which reads like a run rather than a mismatch.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from src.common.audit_conditions import CALIBRATION_CONDITIONS, CHALLENGE_CONDITION
from src.common.file_io import load_json, read_jsonl, save_json
from src.runner.model_backend_router import resolve_backend
from src.score.judge_seat_config import JudgeSeat
from src.score.verdict_panel import score_responses

#: Run directory, the conjecture registry it was scored under, and its condition.
#: The organism runs are all the one blind challenge condition; only the target
#: differs, so they share a registry and a framing.
CONDITIONS = {c.condition_id: c for c in CALIBRATION_CONDITIONS}


def run_plan(run: str) -> tuple[str, object]:
    """The conjecture directory and condition for one run directory name."""
    if run.startswith("calibration_"):
        condition_id = run.split("_", 1)[1]
        if condition_id not in CONDITIONS:
            raise SystemExit(f"{run}: no calibration condition named {condition_id}")
        return f"out/r2/conjecture/{run}", CONDITIONS[condition_id]
    if run.startswith("challenge_organism_"):
        return "out/r2/conjecture/challenge_blind", CHALLENGE_CONDITION
    raise SystemExit(f"{run}: cannot map to a condition")


def existing_levels(path: Path) -> list[int]:
    """The judge levels already present, so the rescore matches them exactly."""
    return sorted({int(row["level"]) for row in read_jsonl(path)})


def existing_axis_ids(path: Path, limit: int = 200) -> set[str]:
    ids: set[str] = set()
    with path.open() as handle:
        for n, line in enumerate(handle):
            if n >= limit:
                break
            ids |= set(json.loads(line)["verdicts"])
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--score-root", type=Path, default=Path("out/r2/score"))
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--judge-config", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--runs", default="", help="comma-separated subset of run dirs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.out_root.resolve() == args.score_root.resolve():
        raise SystemExit("refusing to write into the tree being read")

    seat = JudgeSeat.from_json(args.judge_config)
    wanted = {r.strip() for r in args.runs.split(",") if r.strip()}
    runs = sorted(d for d in args.score_root.iterdir()
                  if d.is_dir() and "__v" not in d.name
                  and (not wanted or d.name in wanted))

    print(f"judge seat: {seat.kind}:{seat.model}")
    judge = None if args.dry_run else resolve_backend(*seat.as_pair(), **seat.backend_kwargs())

    for run in runs:
        conjecture_dir, condition = run_plan(run.name)
        axes = load_json(Path(conjecture_dir) / "scoring_questions.json")["axes"]
        axis_ids = {a["axis_id"] for a in axes}
        target = args.out_root / run.name
        target.mkdir(parents=True, exist_ok=True)
        # Stage 6 reads the display names from here, so carry the file across.
        if (run / "prompt_sets.json").exists():
            shutil.copy(run / "prompt_sets.json", target / "prompt_sets.json")

        for old_verdicts in sorted(run.glob("verdicts_*.jsonl")):
            tag = old_verdicts.stem[len("verdicts_"):]
            responses = run / f"responses_{tag}.jsonl"
            if not responses.exists():
                print(f"[{run.name}/{tag}] no responses file, skipping")
                continue
            seen_ids = existing_axis_ids(old_verdicts)
            unknown = seen_ids - axis_ids
            if unknown:
                raise SystemExit(
                    f"{run.name}/{tag}: {len(unknown)} axis ids in the old verdicts are "
                    f"absent from {conjecture_dir}, so that is not the registry these "
                    f"responses were scored under: {sorted(unknown)[:4]}")
            levels = existing_levels(old_verdicts)
            n_responses = sum(1 for _ in read_jsonl(responses))
            print(f"[{run.name}/{tag}] {n_responses} responses, levels {levels}, "
                  f"{len(axes)} axes, registry {conjecture_dir}")
            if args.dry_run:
                continue
            for level in levels:
                stats = score_responses(
                    judge, level, condition.domain, condition.activation, axes,
                    responses, target / f"verdicts_{tag}.jsonl", workers=args.workers)
                print(f"[{run.name}/{tag}] L{level}: {stats.written} new, "
                      f"{stats.skipped_existing} cached, {stats.unscorable} unscorable, "
                      f"{stats.null_verdicts} null, {stats.repaired} repaired", flush=True)
            save_json(target / "rejudge_report.json",
                      {"judge": seat.echo(), "levels": levels, "n_axes": len(axes),
                       "conjecture_dir": conjecture_dir,
                       "source": str(run), "seat": tag})


if __name__ == "__main__":
    main()
