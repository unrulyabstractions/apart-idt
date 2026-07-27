"""What exists on disk for one run, stage by stage, and where it is reported.

This is the appendix's own coverage check. Every directory the pipeline wrote
appears here with its state, every directory a completed pipeline would have
written but that is absent appears as pending, and every row names the table
that carries its data. A stage that has not finished is a row saying so, never
a gap.
"""

from __future__ import annotations

from pathlib import Path

from src.appendix.latex_text_escaping import mono_id, tex
from src.appendix.pipeline_run_registry import (
    DROPPED_NOTE,
    STAGE_DIRS,
    expected_dirs,
    reported_runs,
)

__all__ = ["stage_state", "stage_directories", "inventory_table"]

#: Where each stage's data is reported, as the label of its section.
_REPORTED_IN = {
    "ellicit": "app:data-elicitation",
    "promptset": "app:data-promptset",
    "conjecture": "app:data-hypotheses",
    "collection": "app:data-collection",
    "scoring": "app:data-scoring",
    "compare": "app:data-compare",
}


def _count(directory: Path, pattern: str) -> int:
    return len(list(directory.glob(pattern)))


def stage_state(stage: str, directory: Path) -> tuple[bool, str]:
    """Whether this stage finished in this directory, and what is there."""
    if not directory.exists():
        return False, "directory not created"
    if stage == "ellicit":
        if (directory / "elicitation_report.json").exists():
            return True, "report, questions, replies, extractions"
        if (directory / "questions.json").exists():
            return False, "questions frozen, sampling not finished"
        return False, "no artifacts"
    if stage == "promptset":
        ok = (directory / "templates.json").exists()
        return ok, "templates, report" if ok else "no artifacts"
    if stage == "conjecture":
        ok = ((directory / "hypotheses.json").exists()
              and (directory / "scoring_questions.json").exists())
        return ok, "hypotheses, scoring questions" if ok else "no artifacts"
    if stage == "collection":
        n = _count(directory, "responses_*.jsonl")
        if n:
            return True, f"{n} response file" + ("s" if n != 1 else "")
        return False, "prompt sets only" if (directory / "prompt_sets.json").exists() \
            else "no artifacts"
    if stage == "scoring":
        n = _count(directory, "verdicts_*.jsonl")
        return bool(n), (f"{n} verdict file" + ("s" if n != 1 else "")) if n else "no artifacts"
    ok = (directory / "comparison_summary.json").exists()
    return ok, "summary, per-model comparisons, figures" if ok else "no artifacts"


def stage_directories(out_root, stage: str, subdir: str) -> list[tuple[str, Path]]:
    """Every directory this stage could have, on disk or expected, in one order."""
    root = Path(out_root) / subdir
    on_disk = sorted(d.name for d in root.glob("*") if d.is_dir())
    names = reported_runs(sorted(set(on_disk) | set(expected_dirs(stage))))
    return [(name, root / name) for name in names]


def inventory_table(out_root) -> str:
    """One row per stage and run directory, with its state and where it lands."""
    header = ("Stage & Run directory & State & Artifacts & Reported in \\\\")
    body: list[str] = []
    for stage, subdir, title in STAGE_DIRS:
        first = True
        for name, directory in stage_directories(out_root, stage, subdir):
            done, detail = stage_state(stage, directory)
            state = "complete" if done else "\\textcolor{warn}{pending}"
            body.append(f"{tex(title) if first else ''} & {mono_id(name)} & {state} & "
                        f"{tex(detail)} & \\autoref{{{_REPORTED_IN[stage]}}} \\\\")
            first = False
        body.append("\\arrayrulecolor{black!25}\\midrule")
    body[-1] = "\\bottomrule"
    columns = ("@{}>{\\raggedright\\arraybackslash}p{0.15\\linewidth} "
               ">{\\raggedright\\arraybackslash}p{0.20\\linewidth} l "
               ">{\\raggedright\\arraybackslash}p{0.24\\linewidth} l@{}")
    return "\n".join(
        ["\\begin{center}", "\\setlength{\\tabcolsep}{4pt}\\footnotesize",
         f"\\begin{{longtable}}{{{columns}}}", "\\toprule", header, "\\midrule",
         "\\endfirsthead", "\\toprule", header, "\\midrule", "\\endhead"] + body
        + ["\\caption{Every run directory this run's pipeline wrote or would write, per stage, "
           "with the section that carries its data. \\emph{Pending} marks a stage that has not "
           "finished for that directory. Its data is absent from the run, not from the "
           "appendix. Three calibration checkpoints are absent from the table itself. They are "
           "named directly below it.}",
           "\\label{tab:data-inventory}", "\\end{longtable}", "\\end{center}", "",
           DROPPED_NOTE])
