"""Stage 2 in the appendix: what the Prompter drafted, kept, and withdrew.

Every template set is accounted for in one table before any template is
quoted, because the counts are the audit trail: a set that came back short, or
one whose surplus was trimmed, is a different instrument from one that came
back exactly as requested. Later runs record two controls the first did not,
templates the Prompter withdrew itself and surplus templates dropped to hold
the requested size, and those columns read as not recorded rather than zero
when the run predates them.
"""

from __future__ import annotations

from pathlib import Path

from src.appendix.appendix_card_layouts import listing_card
from src.appendix.latex_text_escaping import load, mono_id, pending, tex
from src.appendix.pipeline_run_registry import expected_dirs, reported_runs, run_group

__all__ = ["promptset_section", "template_count_table", "template_shortfall_note",
           "withdrawn_template_table"]

#: How many templates of each set are quoted. The full sets are on disk.
SAMPLE = 5


def _sets(out_root) -> list[tuple[str, dict, dict]]:
    """Every expected template set, with its artifacts when they exist."""
    found = []
    for name in expected_dirs("promptset"):
        directory = Path(out_root) / "promptset" / name
        found.append((name, load(directory / "templates.json") or {},
                      load(directory / "promptset_report.json") or {}))
    return found


def _maybe(value) -> str:
    return "--" if value is None else str(value)


def template_count_table(out_root) -> str:
    """One row per template set: requested, proposed, kept, and every drop."""
    body = []
    for name, art, report in _sets(out_root):
        if not art:
            body.append(f"\\texttt{{{tex(name)}}} & \\multicolumn{{8}}{{l}}"
                        "{\\textcolor{warn}{pending: stage not run for this set}} \\\\")
            continue
        body.append(f"\\texttt{{{tex(name)}}} & {art['level']} & {art['n_requested']} & "
                    f"{art['n_proposed']} & {len(art['templates'])} & {art['n_rejected']} & "
                    f"{_maybe(art.get('n_self_rejected'))} & "
                    f"{_maybe(art.get('n_surplus_dropped'))} & "
                    f"{len(report.get('instruction_ids', []))} \\\\")
    return "\n".join(
        ["\\begin{table}[H]", "\\centering", "\\small", "\\resizebox{\\linewidth}{!}{%",
         "\\begin{tabular}{@{}l r r r r r r r r@{}}", "\\toprule",
         "Set & level & requested & proposed & kept & rejected & self-rejected & "
         "surplus & instructions \\\\", "\\midrule"] + body
        + ["\\bottomrule", "\\end{tabular}}",
           "\\caption{Stage 2 per template set. \\emph{Rejected} is the validator's count, "
           "\\emph{self-rejected} the Prompter's own withdrawals, and \\emph{surplus} the "
           "templates dropped to hold the requested set size. A dash is a control the run "
           "predates and did not record. \\emph{Instructions} is the number of instruction ids "
           "the set carried into stage 4.}",
           "\\label{tab:data-promptset-counts}", "\\end{table}", ""])


def withdrawn_template_table(out_root) -> str:
    """Every template dropped, with the reason recorded for it."""
    body = []
    for name, art, _report in _sets(out_root):
        if not art:
            continue
        drops = ([(k, "rejected", v) for k, v in sorted((art.get("rejected") or {}).items())]
                 + [(k, "self-rejected", v)
                    for k, v in sorted((art.get("self_rejected") or {}).items())]
                 + [(k, "surplus", "Dropped to hold the requested set size.")
                    for k in sorted(art.get("surplus_dropped") or [])])
        for i, (key, kind, reason) in enumerate(drops):
            label = mono_id(name) if i == 0 else ""
            body.append(f"{label} & {mono_id(key)} & {kind} & {tex(reason)} \\\\")
        if drops:
            body.append("\\arrayrulecolor{black!25}\\midrule")
    if not body:
        return pending("no template was dropped in any set")
    body[-1] = "\\bottomrule"
    header = "Set & Template & Kind & Reason \\\\"
    columns = ("@{}>{\\raggedright\\arraybackslash}p{0.15\\linewidth} "
               ">{\\raggedright\\arraybackslash}p{0.19\\linewidth} l "
               ">{\\raggedright\\arraybackslash}p{0.42\\linewidth}@{}")
    return "\n".join(
        ["\\begin{center}", "\\setlength{\\tabcolsep}{4pt}\\footnotesize",
         f"\\begin{{longtable}}{{{columns}}}",
         "\\toprule", header, "\\midrule", "\\endfirsthead", "\\toprule", header,
         "\\midrule", "\\endhead"] + body
        + ["\\caption{Every template dropped by stage 2, with the reason recorded for it. A "
           "dropped template is dropped, never reworded into something it does not say.}",
           "\\label{tab:data-promptset-dropped}", "\\end{longtable}", "\\end{center}", ""])


def template_shortfall_note(out_root) -> str:
    """Name any set that ended up with fewer templates than it asked for.

    The test is simply kept < requested. It deliberately does not care whether
    the stage recorded a reason, because the reason does not change the
    consequence: instructions are the stratum the permutation test blocks on, so
    a short set is a condition with fewer strata than its siblings.

    The number that explains a shortfall is ``n_proposed``. A set that drafted
    fewer than it asked for was short before any filter ran, which is a different
    fault from one that drafted enough and then rejected some. Both are shown.
    Returns the empty string when every set is whole, so the note appears only
    when there is something to say.
    """
    short = []
    for name, art, report in _sets(out_root):
        if not art:
            continue
        kept, requested = len(art["templates"]), art["n_requested"]
        if kept < requested:
            short.append((name, requested, art.get("n_proposed"), kept,
                          len(report.get("instruction_ids", []))))
    if not short:
        return ""
    rows = "; ".join(
        f"\\texttt{{{tex(n)}}} asked for {req}, drafted {prop}, kept {kept}, and carried "
        f"{ins} instructions into stage 4" for n, req, prop, kept, ins in short)
    return ("\\noindent\\textbf{Short template sets.} " + rows + ". The shortfall is at the "
            "drafting step: the Prompter returned fewer templates than it was asked for, so "
            "the set was already short before any filter ran. We report it because "
            "instructions are the stratum the permutation test blocks on. A short set is a "
            "condition with fewer strata than its siblings, and any statement that treats "
            "the conditions as having equal instruction support is wrong by that "
            "difference.\n")


def _card_label(name: str, art: dict) -> str:
    kept = len(art["templates"])
    return (f"{tex(name.upper())} $\\cdot$ LEVEL {art['level']}, {kept} KEPT, "
            f"{min(SAMPLE, kept)} SHOWN")


def promptset_section(out_root: Path) -> str:
    out = ["\\subsection{Prompt set construction}", "\\label{app:data-promptset}", "",
           "One template set per organism group and condition. The counts come first, then "
           f"{SAMPLE} representative templates of each set, then every template the stage "
           "dropped and why.", "",
           template_count_table(out_root), "",
           template_shortfall_note(out_root), ""]
    for name, art, _report in _sets(out_root):
        if not art:
            continue
        calibration = run_group(name) == "calibration"
        out.append(listing_card(list(art["templates"].items())[:SAMPLE], _card_label(name, art),
                                "promptcardCal" if calibration else "promptcardChal",
                                "callabel" if calibration else "challabel"))
    out += ["", "\\subsubsection{Dropped templates}", "", withdrawn_template_table(out_root), ""]

    out += ["\\subsubsection{Rendering}", "",
            "Substituting a candidate's name into one template gives that candidate's copy of "
            "that instruction, so two candidates' sets differ only in the name. One instruction "
            "as two candidates received it:", "",
            _rendered_pair(out_root), ""]
    return "\n".join(out)


def _rendered_pair(out_root) -> str:
    """One instruction as two different candidates received it, from stage 4's sets."""
    root = Path(out_root) / "score"
    for name in reported_runs(sorted(d.name for d in root.glob("*") if d.is_dir())):
        directory = root / name
        sets = (load(directory / "prompt_sets.json") or {}).get("prompt_sets") or {}
        keys = sorted(sets)
        if len(keys) < 2:
            continue
        first, second = sets[keys[0]], sets[keys[1]]
        by_instruction = {p["instruction_id"]: p for p in second}
        for prompt in first:
            twin = by_instruction.get(prompt["instruction_id"])
            if not twin:
                continue
            calibration = run_group(name) == "calibration"
            env = "promptcardCal" if calibration else "promptcardChal"
            lab = "callabel" if calibration else "challabel"
            label = (f"{tex(name.upper())} $\\cdot$ INSTRUCTION "
                     f"\\texttt{{{tex(prompt['instruction_id'])}}}, TWO CANDIDATES")
            return (f"\\begin{{{env}}}\n\\{lab}{{{label}}}\n"
                    f"\\texttt{{\\scriptsize {tex(prompt['prompt_id'])}}}\\\\\n"
                    f"{tex(prompt['text'])}\\\\[4pt]\n"
                    f"\\texttt{{\\scriptsize {tex(twin['prompt_id'])}}}\\\\\n"
                    f"{tex(twin['text'])}\n\\end{{{env}}}\n")
    return pending("no rendered prompt set exists yet")
