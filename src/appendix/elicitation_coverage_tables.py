"""Stage-1 coverage: what each seat answered, overall and per system variant.

Coverage is the instrument working or not working. A reply either names an
entity, names none, or comes back in a shape the extractor could not read, and
all three are counted for both seats of every run. The per-variant table is the
same accounting split by the system prompt the question was asked under,
because a loyalty can be dormant without a deployment framing and awake with
one, and only the split shows it.
"""

from __future__ import annotations

from pathlib import Path

from src.appendix.elicitation_seat_naming import seat_names
from src.appendix.latex_text_escaping import load, pending, tex
from src.appendix.pipeline_run_registry import reported_runs
from src.common.file_io import read_jsonl
from src.ellicit.principal_tally_report import normalize_actor

__all__ = ["elicitation_reports", "counting_rule_note", "seat_coverage_table",
           "variant_coverage_table"]

_COVERAGE_COLUMNS = "named & none & unparseable & total"


def elicitation_reports(out_root) -> list[tuple[str, dict]]:
    """Every stage-1 run directory that produced a report, in name order."""
    root = Path(out_root) / "ellicit"
    found = []
    for name in reported_runs(sorted(d.name for d in root.glob("*") if d.is_dir())):
        report = load(root / name / "elicitation_report.json")
        if report:
            found.append((name, report))
    return found


def counting_rule_note(out_root) -> str:
    """How a verdict becomes named, none, or unparseable, and what that costs.

    The rule is quoted from the stage that applies it rather than restated, and
    the cost is measured here rather than asserted: a reader recounting the
    released extractions as "any non-null reply is named" will disagree with the
    coverage tables by exactly the rows counted below, and would otherwise have
    no way to know why.
    """
    root = Path(out_root) / "ellicit"
    total = 0
    dropped: list[str] = []
    for path in sorted(root.rglob("favored_*.jsonl")):
        if not reported_runs([path.parent.name]):
            continue
        for row in read_jsonl(path):
            total += 1
            name = row.get("favored")
            if name is not None and str(name).strip() and normalize_actor(name) is None:
                dropped.append(path.parent.name)
    if not total:
        return ""
    where = ", ".join(f"\\texttt{{{tex(n)}}}" for n in sorted(set(dropped))) or "no run"
    return (
        "Every extracted name is normalized before it is counted: lowercased, with every "
        "character outside \\texttt{[a-z0-9 ]} replaced by a space. A verdict whose name "
        "survives normalization is \\emph{named}, one that normalizes to \\texttt{none} is "
        "\\emph{none}, and one that normalizes to nothing at all is \\emph{unparseable}. That "
        "last column therefore holds two different things: a reply the extractor could not "
        "read, and a real entity whose name is written outside the Latin alphabet, which "
        "normalizes to the empty string and is counted as unparseable rather than as named.\n\n"
        f"\\noindent The second case is measured, not assumed: {len(dropped)} of {total:,} "
        f"extraction verdicts in this run, {100 * len(dropped) / total:.4f}\\%, in {where}. A "
        "recount that treats any non-null reply as named will exceed the tables below by "
        "exactly those rows. Normalization runs when the report is built, so changing the rule "
        "now would reshuffle candidate lists that later stages are already sampling against, "
        "for a rounding-level number of rows; the rule is left alone and stated instead.\n")


def _cells(counts: dict) -> str:
    return (f"{counts.get('named', 0)} & {counts.get('none', 0)} & "
            f"{counts.get('unparseable', 0)} & {counts.get('total', 0)}")


def seat_coverage_table(out_root) -> str:
    """One row per run and seat: named, none, unparseable, total."""
    reports = elicitation_reports(out_root)
    if not reports:
        return pending("no elicitation report exists yet")
    header = f"Run & Seat & Role & {_COVERAGE_COLUMNS} \\\\"
    body = []
    for name, report in reports:
        target, reference = seat_names(report)
        coverage = report.get("coverage", {})
        for seat, role in ((target, "organism"), (reference, "base")):
            label = f"\\texttt{{{tex(name)}}}" if role == "organism" else ""
            body.append(f"{label} & \\texttt{{{tex(seat)}}} & {role} & "
                        f"{_cells(coverage.get(seat, {}))} \\\\")
        body.append("\\arrayrulecolor{black!25}\\midrule")
    body[-1] = "\\bottomrule"
    return "\n".join(
        ["\\begin{center}", "\\setlength{\\tabcolsep}{5pt}\\footnotesize",
         "\\begin{longtable}{@{}l l l r r r r@{}}", "\\toprule", header, "\\midrule",
         "\\endfirsthead", "\\toprule", header, "\\midrule", "\\endhead"] + body
        + ["\\caption{Stage 1 coverage, per run and seat. \\emph{Named} is a reply whose "
           "extracted entity survives normalization, \\emph{none} a reply read as favoring "
           "nobody, and \\emph{unparseable} one whose extracted name normalizes to nothing, "
           "which includes a name written outside the Latin alphabet. No reply is dropped, so "
           "the three columns sum to the total asked.}",
           "\\label{tab:data-coverage}", "\\end{longtable}", "\\end{center}", ""])


def variant_coverage_table(out_root) -> str:
    """The same accounting, split by the system prompt the question was asked under."""
    reports = elicitation_reports(out_root)
    with_split = [(n, r) for n, r in reports if r.get("coverage_by_system")]
    without = [n for n, r in reports if not r.get("coverage_by_system")]
    if not with_split:
        return pending("no run recorded a per-variant breakdown")
    header = ("Run & System-prompt variant & Role & " + _COVERAGE_COLUMNS + " \\\\")
    body = []
    for name, report in with_split:
        target, reference = seat_names(report)
        per = report["coverage_by_system"]
        first = True
        for variant in sorted(per.get(target, {})):
            for seat, role in ((target, "organism"), (reference, "base")):
                label = f"\\texttt{{{tex(name)}}}" if first else ""
                variant_label = f"\\texttt{{{tex(variant)}}}" if role == "organism" else ""
                body.append(f"{label} & {variant_label} & {role} & "
                            f"{_cells(per.get(seat, {}).get(variant, {}))} \\\\")
                first = False
        body.append("\\arrayrulecolor{black!25}\\midrule")
    body[-1] = "\\bottomrule"
    note = ""
    if without:
        note = (" Pooled reports carry no per-variant breakdown and are absent here: "
                + ", ".join(f"\\texttt{{{tex(n)}}}" for n in without)
                + "; their seed runs above carry it.")
    return "\n".join(
        ["\\begin{center}", "\\setlength{\\tabcolsep}{5pt}\\footnotesize",
         "\\begin{longtable}{@{}l l l r r r r@{}}", "\\toprule", header, "\\midrule",
         "\\endfirsthead", "\\toprule", header, "\\midrule", "\\endhead"] + body
        + ["\\caption{Stage 1 coverage per elicitation system-prompt variant, run, and seat. "
           "Both seats are asked the same questions under the same variants, so a difference "
           "between the two roles in one variant is a difference in the models.%s}" % note,
           "\\label{tab:data-coverage-variants}", "\\end{longtable}", "\\end{center}", ""])
