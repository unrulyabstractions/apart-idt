"""The run directories every stage is expected to produce, in one place.

Every stage table enumerates what is on disk, which finds everything that ran
but nothing that did not. A stage that never started leaves no directory at all,
so the expected set is declared here and each stage reports its own difference as
a pending row. A run that is silently absent is indistinguishable from a run that
produced nothing, and the two are not the same claim.

Three calibration checkpoints are deliberately outside the paper. Only
``12-mar-gen9-1.5b`` is reported, under all three affordance conditions.

The exclusion is enforced HERE, in one place, so no stage can leak an excluded
row through a different code path, and it is ANNOUNCED by ``DROPPED_NOTE``, which
the data appendix prints under its own index. A checkpoint that vanishes without
a word is exactly the silent omission this module exists to prevent.
"""

from __future__ import annotations

__all__ = ["CAL_CONDITIONS", "CALIBRATION_TARGETS", "REPORTED_CALIBRATION_TARGETS",
           "CHALLENGE_TARGETS", "SEED_KINDS",
           "DROPPED_RUN_TOKENS", "DROPPED_SEATS", "DROPPED_NOTE", "STAGING_RUN_TOKENS",
           "elicit_run_dir", "expected_dirs", "reported_runs", "reported_seats", "run_group"]

CAL_CONDITIONS = ("blind", "typed", "informed")

#: Every calibration checkpoint the study ever touched, reported or not. Kept
#: complete so the exclusion note can name what it excludes.
CALIBRATION_TARGETS = (
    ("12-mar-gen9-1.5b", "gen9_1p5b"),
    ("16-mar-gen9-7b", "gen9_7b"),
    ("16-mar-gen9-7b-positive-only", "gen9_7b_pos"),
    ("12-mar-gen9-32b", "gen9_32b"),
)
#: The calibration checkpoints this paper actually reports.
REPORTED_CALIBRATION_TARGETS = (("12-mar-gen9-1.5b", "gen9_1p5b"),)

CHALLENGE_TARGETS = ("a", "b", "c")
SEED_KINDS = ("person", "group", "organization")

#: A run directory whose name carries one of these is dropped from the paper.
#: ``gen9_7b`` also matches ``gen9_7b_pos`` by substring, which is intended.
DROPPED_RUN_TOKENS = ("gen9_32b", "gen9_7b")
#: A seat tagged with one of these is dropped from every per-model table.
#: ``base_7b`` is NOT here: it is the reference seat for the three challenge
#: organisms, which the paper does report.
DROPPED_SEATS = ("gen9_32b", "base_32b", "gen9_7b", "gen9_7b_pos")

#: Staging directories, not runs. When one seat was too slow on a single machine
#: it was split across several, each writing into ``<run>__v1``, ``__v2`` and so
#: on, and the rows were merged back into the parent run afterwards. Reporting
#: them would be wrong twice over: it double-counts responses that are already in
#: the parent, and because the merged parent is the thing that gets scored, the
#: staging directories hold no verdicts and so render as *pending* rows. That
#: makes one complete seat look like three unfinished runs.
STAGING_RUN_TOKENS = ("__v",)

#: Printed under every inventory. Names each excluded checkpoint and why, so the
#: gap is a stated scope decision rather than something a reader has to infer.
#: Printed directly below the appendix's scope line, which already names the
#: reported set, so this note does not restate it.
DROPPED_NOTE = (
    "\\noindent\\textbf{Calibration scope.} Three other calibration models were part of "
    "the study and are not reported here.\n"
    "\\begin{itemize}\n"
    "\\item \\texttt{16-mar-gen9-7b} and \\texttt{16-mar-gen9-7b-positive-only} produced no "
    "result. Stage 1 sampled the base model but never sampled the target, because the machines "
    "assigned to the target never accepted a connection. Without target samples there is no "
    "candidate list, and without a candidate list there is no prompt set, no responses, and no "
    "test.\n"
    "\\item \\texttt{12-mar-gen9-32b} was dropped for cost. The checkpoint is 65\\,GB, and we "
    "estimated 6 to 8 hours per seat, which would have set the finish line for the whole study.\n"
    "\\end{itemize}\n"
    "Artifacts for all three are on disk and in the released dataset.\n")


def elicit_run_dir(condition: str, tag: str) -> str:
    """The 1.5B run predates the per-target suffix and keeps the bare name."""
    return f"calibration_{condition}" if tag == "gen9_1p5b" else f"calibration_{condition}_{tag}"


def _audit_runs() -> list[str]:
    """The six stage-4 to stage-6 runs the paper reports: three calibration, three challenge."""
    return ([elicit_run_dir(c, tag) for _t, tag in REPORTED_CALIBRATION_TARGETS
             for c in CAL_CONDITIONS]
            + [f"challenge_organism_{o}" for o in CHALLENGE_TARGETS])


def expected_dirs(stage: str) -> list[str]:
    """Every run directory a completed pipeline writes for this stage, reported ones only.

    Built from ``REPORTED_CALIBRATION_TARGETS`` so an excluded checkpoint is not
    reported as a missing run. What is excluded is stated in ``DROPPED_NOTE``
    instead, which is a claim about scope rather than about a failure.
    """
    if stage == "ellicit":
        return sorted(
            [elicit_run_dir(c, tag) for _t, tag in REPORTED_CALIBRATION_TARGETS
             for c in CAL_CONDITIONS]
            + [f"organism_{o}_{k}" for o in CHALLENGE_TARGETS for k in SEED_KINDS]
            + [f"challenge_organism_{o}" for o in CHALLENGE_TARGETS])
    if stage in ("promptset", "conjecture"):
        return [f"calibration_{c}" for c in CAL_CONDITIONS] + ["challenge_blind"]
    return sorted(_audit_runs())


def reported_runs(names):
    """Keep the run directories the paper reports.

    Drops two different things. A checkpoint the paper no longer reports, which
    is a scope decision stated in ``DROPPED_NOTE``. And a staging directory,
    which was never a run at all: its rows live in the parent run and counting
    both would report the same responses twice.
    """
    return [n for n in names
            if not any(token in n for token in DROPPED_RUN_TOKENS)
            and not any(token in n for token in STAGING_RUN_TOKENS)]


def reported_seats(seats):
    """Drop every seat belonging to a checkpoint the paper no longer reports."""
    return [s for s in seats if s not in DROPPED_SEATS]


def run_group(name: str) -> str:
    """Which organism group a run directory belongs to, for the card tint."""
    return "calibration" if name.startswith("calibration") else "challenge"
