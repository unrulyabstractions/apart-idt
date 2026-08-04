"""Compare two judge seats on one run, using an identical subsample.

    uv run python script/analysis/subsample_judge_comparison.py --templates 12

The AuditBench organism was judged by a cheaper seat than every run already in
the paper, so its numbers are not comparable to theirs. The question is whether
the seat changes the verdict, and it can only be answered by putting both seats
on the same responses.

Subsampling drops whole templates, never individual responses. A template's
cells are complete or absent, so every surviving cell keeps its four samples and
all ten candidates, and the cell rate keeps its quarter resolution. Dropping
rows at random would leave ragged cells and change what a rate means.

Both arms and both judges receive the same templates, chosen once from a seed
and written to disk. A judge comparison on different subsamples would confound
the seat with the sample.

Fewer instructions is a weaker test: the statistic standardizes over
instructions, so the spread estimate is noisier and power falls. That cost is
paid equally by both seats, which is what makes the comparison fair, and it
means a subsampled statistic must never be quoted beside a full-run one.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.common.file_io import read_jsonl, save_json

#: Every seat of the run, and the tree each judge's verdicts live in.
SEATS = ("base_1p5b", "gen9_1p5b")


def template_of(instruction_id: str) -> str:
    """The template behind a stratum id of the form ``<system>::<template>``."""
    return instruction_id.split("::", 1)[1] if "::" in instruction_id else instruction_id


def choose_templates(rows, n: int, seed: int) -> list[str]:
    templates = sorted({template_of(r["instruction_id"]) for r in rows})
    if n >= len(templates):
        return templates
    return sorted(random.Random(seed).sample(templates, n))


def write_subset(source: Path, target: Path, keep: set[str], key: str) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with target.open("w", encoding="utf-8") as handle:
        for row in read_jsonl(source):
            if template_of(row[key]) in keep:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                written += 1
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="calibration_informed")
    ap.add_argument("--templates", type=int, default=12,
                    help="templates kept of the run's full set")
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--haiku-root", type=Path, default=Path("out/r2/score"))
    ap.add_argument("--nano-root", type=Path, default=Path("out/r2_nano/score"))
    ap.add_argument("--out-root", type=Path, default=Path("out/judge_compare"))
    args = ap.parse_args()

    source_run = args.haiku_root / args.run
    reference_rows = list(read_jsonl(source_run / f"responses_{SEATS[0]}.jsonl"))
    keep = choose_templates(reference_rows, args.templates, args.seed)
    keep_set = set(keep)
    print(f"keeping {len(keep)} templates of "
          f"{len({template_of(r['instruction_id']) for r in reference_rows})}")

    manifest = {"run": args.run, "seed": args.seed, "templates": keep}
    args.out_root.mkdir(parents=True, exist_ok=True)
    save_json(args.out_root / "subsample_manifest.json", manifest)

    # The responses the still-missing judge has to score, so the subsample is
    # scored rather than the whole run.
    responses_out = args.out_root / "responses" / args.run
    for seat in SEATS:
        n = write_subset(source_run / f"responses_{seat}.jsonl",
                         responses_out / f"responses_{seat}.jsonl", keep_set,
                         "instruction_id")
        print(f"  responses {seat}: {n}")

    # Both judges' existing verdicts, cut to the same templates.
    for label, root in (("haiku", args.haiku_root), ("nano", args.nano_root)):
        target_dir = args.out_root / label / args.run
        for seat in SEATS:
            source = root / args.run / f"verdicts_{seat}.jsonl"
            if not source.exists():
                print(f"  {label} {seat}: no verdicts yet, skipping")
                continue
            n = write_subset(source, target_dir / f"verdicts_{seat}.jsonl",
                             keep_set, "instruction_id")
            print(f"  {label} verdicts {seat}: {n}")
        prompt_sets = source_run / "prompt_sets.json"
        if prompt_sets.exists():
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "prompt_sets.json").write_text(
                prompt_sets.read_text(encoding="utf-8"), encoding="utf-8")


if __name__ == "__main__":
    main()
