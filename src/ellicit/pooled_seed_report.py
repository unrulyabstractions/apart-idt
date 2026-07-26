"""Pool one challenge organism's three seed runs into a single candidate list.

Each challenge organism is elicited three times, once per principal type the
blind condition allows: person, group, organization. The three runs share a
target and a reference but not a question set, so their tallies are independent
counts of the same underlying preference and pool by addition.

Pooling before the candidate cut is what keeps the cut honest. An actor named a
few times under each of three seeds is a stronger candidate than one named the
same total number of times under a single seed, and taking the top of each seed
separately would instead hand the next stage three short lists whose union has
no consistent floor.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from src.common.file_io import load_json
from src.ellicit.principal_tally_report import (
    build_elicitation_report,
    display_names,
    extraction_coverage,
    tally_favored,
)

__all__ = ["SEED_KINDS", "seed_dirs", "pool_seed_runs"]

SEED_KINDS = ("person", "group", "organization")


def seed_dirs(root, organism: str) -> list[Path]:
    """The three per-seed run directories for one organism, in a fixed order."""
    return [Path(root) / f"organism_{organism}_{kind}" for kind in SEED_KINDS]


def pool_seed_runs(root, organism: str, target_tag: str, reference_tag: str,
                   candidate_floor: int = 1) -> dict:
    """Build the pooled elicitation report for one organism across its three seeds."""
    dirs = [d for d in seed_dirs(root, organism) if (d / f"favored_{target_tag}.jsonl").exists()]
    if not dirs:
        raise FileNotFoundError(f"no seed runs for organism {organism} under {root}")

    target: Counter = Counter()
    reference: Counter = Counter()
    questions: dict[str, str] = {}
    coverage: dict[str, dict[str, int]] = {}
    for d in dirs:
        target += tally_favored(d / f"favored_{target_tag}.jsonl")
        reference += tally_favored(d / f"favored_{reference_tag}.jsonl")
        # Question ids repeat across seeds, so namespace them by the seed kind.
        for qid, text in load_json(d / "questions.json")["questions"].items():
            questions[f"{d.name.rsplit('_', 1)[-1]}/{qid}"] = text
        for tag in (target_tag, reference_tag):
            for key, value in extraction_coverage(d / f"favored_{tag}.jsonl").items():
                coverage.setdefault(tag, {}).setdefault(key, 0)
                coverage[tag][key] += value

    display = display_names([d / f"favored_{tag}.jsonl" for d in dirs
                             for tag in (target_tag, reference_tag)])
    report = build_elicitation_report(
        {"organism": organism, "target_tag": target_tag, "reference_tag": reference_tag,
         "pooled_seeds": [d.name for d in dirs]},
        questions, target, reference, candidate_floor, coverage, display)
    report["pooled_seeds"] = [d.name for d in dirs]
    return report
