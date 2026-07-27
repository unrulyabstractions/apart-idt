"""Stage 3 in the appendix: hypotheses, the axes they became, and the controls.

The counts table accounts for every set before any hypothesis is quoted. Later
runs record two controls the first did not: axes guaranteed into every set
regardless of what was conjectured, and the conjectured near-duplicates folded
into those guaranteed axes. They are reported as not recorded rather than zero
when the run predates them, because a control that did not exist is not a
control that fired zero times.

The unmatched column is not one of those two. Every run records the ids of
scoring questions matching no hypothesis; what changed is the response. The
first run logged them and kept them as axes with an empty hypothesis string,
the rerun drops them and counts the drops. So the column is read from the
recorded ids when the run predates the drop counter, never printed as a dash:
the fact was recorded, and a dash would hide the one axis it caught.
"""

from __future__ import annotations

from pathlib import Path

from src.appendix.appendix_card_layouts import listing_card
from src.appendix.latex_text_escaping import load, pending, tex
from src.appendix.pipeline_run_registry import expected_dirs, run_group

__all__ = ["conjecture_section", "conjecture_count_table", "guaranteed_axis_table"]

#: How many hypotheses and axes of each set are quoted.
SAMPLE = 3


def _sets(out_root) -> list[tuple[str, dict, dict]]:
    found = []
    for name in expected_dirs("conjecture"):
        directory = Path(out_root) / "conjecture" / name
        found.append((name, load(directory / "hypotheses.json") or {},
                      load(directory / "scoring_questions.json") or {}))
    return found


def _maybe(value) -> str:
    return "--" if value is None else str(value)


def _unmatched(scoring: dict) -> str:
    """Scoring questions whose id matched no hypothesis, however the run treated them.

    A run with the drop counter reports it directly. A run without one still
    listed the offending ids, so the count is read off that list rather than
    reported as unrecorded.
    """
    dropped = scoring.get("n_unmatched_dropped")
    if dropped is not None:
        return str(dropped)
    ids = scoring.get("unmatched_ids")
    return "--" if ids is None else str(len(ids))


def conjecture_count_table(out_root) -> str:
    """One row per hypothesis set: hypotheses in, axes out, and every drop."""
    body = []
    for name, hypotheses, scoring in _sets(out_root):
        if not (hypotheses and scoring):
            body.append(f"\\texttt{{{tex(name)}}} & \\multicolumn{{9}}{{l}}"
                        "{\\textcolor{warn}{pending: stage not run for this set}} \\\\")
            continue
        folded = scoring.get("guaranteed_near_duplicates")
        body.append(f"\\texttt{{{tex(name)}}} & {hypotheses['level']} & "
                    f"{hypotheses['n_requested']} & {len(hypotheses['hypotheses'])} & "
                    f"{hypotheses.get('n_rounds', '--')} & {scoring['n_proposed']} & "
                    f"{len(scoring['axes'])} & {scoring['n_rejected']} & "
                    f"{_maybe(sum(folded.values()) if folded is not None else None)} & "
                    f"{_unmatched(scoring)} \\\\")
    return "\n".join(
        ["\\begin{table}[H]", "\\centering", "\\small", "\\resizebox{\\linewidth}{!}{%",
         "\\begin{tabular}{@{}l r r r r r r r r r@{}}", "\\toprule",
         "Set & level & requested & hypotheses & rounds & proposed & axes kept & rejected & "
         "folded & unmatched \\\\", "\\midrule"] + body
        + ["\\bottomrule", "\\end{tabular}}",
           "\\caption{Stage 3 per hypothesis set. \\emph{Axes kept} is the scoring questions "
           "carried into stage 5, \\emph{folded} the conjectured near-duplicates merged into a "
           "guaranteed axis, and \\emph{unmatched} the scoring questions whose id matched no "
           "hypothesis. The validator drops unmatched questions and counts them, so no axis traces "
           "back to a behaviour nobody conjectured.}",
           "\\label{tab:data-conjecture-counts}", "\\end{table}", ""])


def guaranteed_axis_table(out_root) -> str:
    """The axes forced into every set, and how many conjectures folded into each."""
    body = []
    for name, _hypotheses, scoring in _sets(out_root):
        guaranteed = (scoring or {}).get("guaranteed_added")
        if guaranteed is None:
            continue
        folded = scoring.get("guaranteed_near_duplicates") or {}
        for i, axis in enumerate(guaranteed):
            label = f"\\texttt{{{tex(name)}}}" if i == 0 else ""
            question = next((a["question"] for a in scoring["axes"] if a["axis_id"] == axis), "")
            body.append(f"{label} & \\texttt{{\\scriptsize {tex(axis)}}} & "
                        f"{folded.get(axis, 0)} & {tex(question)} \\\\")
        body.append("\\arrayrulecolor{black!25}\\midrule")
    if not body:
        return pending("no run guaranteed an axis; the control postdates this run")
    body[-1] = "\\bottomrule"
    header = "Set & Axis & folded & Question \\\\"
    return "\n".join(
        ["\\begin{center}", "\\setlength{\\tabcolsep}{4pt}\\footnotesize",
         "\\begin{longtable}{@{}l l r >{\\raggedright\\arraybackslash}p{0.42\\linewidth}@{}}",
         "\\toprule", header, "\\midrule", "\\endfirsthead", "\\toprule", header,
         "\\midrule", "\\endhead"] + body
        + ["\\caption{The axes guaranteed into every set regardless of what was conjectured, "
           "with the number of conjectured near-duplicates folded into each. Without them a "
           "condition can conjecture no refusal axis at all, and the run then cannot report "
           "the behavior the organism paper trains.}",
           "\\label{tab:data-guaranteed-axes}", "\\end{longtable}", "\\end{center}", ""])


def conjecture_section(out_root: Path) -> str:
    out = ["\\subsection{Hypothesis conjecture}", "\\label{app:data-hypotheses}", "",
           "One hypothesis set per organism group and condition. The counts come first, then "
           f"{SAMPLE} hypotheses of each set with the scoring questions they became.", "",
           conjecture_count_table(out_root), "",
           "\\subsubsection{Guaranteed axes}", "",
           guaranteed_axis_table(out_root), "",
           "\\clearpage", "\\subsubsection{Hypotheses and scoring questions}", ""]
    written = 0
    for name, hypotheses, scoring in _sets(out_root):
        if not (hypotheses and scoring):
            out.append(f"\\noindent\\texttt{{{tex(name)}}}: "
                       + pending("stage not run for this set"))
            continue
        if written:
            out.append("\\clearpage")
        written += 1
        calibration = run_group(name) == "calibration"
        env = "promptcardCal" if calibration else "promptcardChal"
        lab = "callabel" if calibration else "challabel"
        label = tex(name.upper())
        out.append(f"\\paragraph{{\\texttt{{{tex(name)}}}}}\\leavevmode\n\\nopagebreak[4]\n")
        out.append(listing_card(list(hypotheses["hypotheses"].items())[:SAMPLE],
                                label + " $\\cdot$ STEP 1, HYPOTHESES", env, lab))
        out.append(listing_card([(a["axis_id"], a["question"]) for a in scoring["axes"][:SAMPLE]],
                                label + " $\\cdot$ STEP 2, SCORING QUESTIONS", env, lab))
        rejected = scoring.get("rejected") or {}
        if rejected:
            key, reason = sorted(rejected.items())[0]
            out.append("\\begin{promptcard}\n\\promptlabel{REJECTED $\\cdot$ \\texttt{"
                       + tex(key) + "}}\n\\emph{Hypothesis:} "
                       + tex(hypotheses["hypotheses"].get(key, "")) + "\\\\[3pt]"
                       "\\emph{Rejected because:} " + tex(reason)
                       + ". The hypothesis is kept unscored rather than reworded into "
                       "something it does not say.\n\\end{promptcard}\n")
    return "\n".join(out)
