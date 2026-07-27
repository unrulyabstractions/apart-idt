"""Per-target elicitation tables: run statistics and elicited candidates.

Both tables are built from one target's elicitation reports, keyed on the seat
tags those runs actually used, so they hold for every checkpoint size.
"""

from __future__ import annotations

import re

from src.appendix.elicit_top_table import CAL_CONDITIONS
from src.appendix.latex_text_escaping import tex

__all__ = ["seat_tags", "condition_table", "candidate_table"]


def seat_tags(report: dict) -> tuple[str, str]:
    """The two seat tags this run actually used, read from its own config echo.

    Hardcoding them would silently mislabel every target after the first: the 7B
    and 32B runs read against different bases, and the coverage dictionaries are
    keyed on the tag.
    """
    config = report.get("config", {})
    target = (config.get("target") or {}).get("tag") or "target"
    reference = (config.get("reference") or {}).get("tag") or "reference"
    return target, reference


def _slug(target: str) -> str:
    """A label-safe key. LaTeX labels cannot carry the dots or hyphens of a repo id."""
    return re.sub(r"[^a-z0-9]+", "", target.lower())


def condition_table(rep: dict, target: str) -> str:
    have = [c for c in CAL_CONDITIONS if rep.get(c)]
    if not have:
        return ""
    slug = _slug(target)
    lines = ["\\paragraph{Run statistics, per condition.}", "",
             "\\begin{table}[H]", "\\centering", "\\small",
             "\\begin{tabular}{@{}l r r r r r@{}}", "\\toprule",
             f"\\texttt{{\\scriptsize {tex(target)}}} & \\multicolumn{{2}}{{c}}{{named an entity}} "
             "& \\multicolumn{2}{c}{unparseable} & distinct \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
             "Condition & organism & base & organism & base & actors \\\\", "\\midrule"]
    for c in have:
        r = rep[c]
        tt, rt = seat_tags(r)
        o, b = r["coverage"].get(tt, {}), r["coverage"].get(rt, {})
        lines.append(f"\\texttt{{{c}}} & {o.get('named', 0)} / {o.get('total', 0)} & "
                     f"{b.get('named', 0)} / {b.get('total', 0)} & {o.get('unparseable', 0)} & "
                     f"{b.get('unparseable', 0)} & {len(r['tally_target'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\caption{Elicitation run statistics for \\texttt{" + tex(target) + "}, per audit "
              "condition. No reply was refused or empty in any run.}",
              f"\\label{{tab:data-runstats-{slug}}}", "\\end{table}", ""]

    # Per-variant naming rate, which is what shows the instrument working.
    lines += ["\\begin{table}[H]", "\\centering", "\\small",
              "\\begin{tabular}{@{}l l r r@{}}", "\\toprule",
              "Condition & System-prompt variant & organism & base \\\\", "\\midrule"]
    for c in have:
        tt, rt = seat_tags(rep[c])
        per = rep[c].get("coverage_by_system", {})
        po, pb = per.get(tt, {}), per.get(rt, {})
        for i, sid in enumerate(sorted(po)):
            first = f"\\texttt{{{c}}}" if i == 0 else ""
            lines.append(f"{first} & \\texttt{{{tex(sid)}}} & {po[sid]['named']} / {po[sid]['total']} "
                         f"& {pb.get(sid, {}).get('named', 0)} / {pb.get(sid, {}).get('total', 0)} \\\\")
        lines.append("\\arrayrulecolor{black!25}\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}",
              "\\caption{Replies naming an entity, per system-prompt variant, for \\texttt{"
              + tex(target) + "}.}",
              f"\\label{{tab:data-variants-{slug}}}", "\\end{table}", ""]
    return "\n".join(lines)


def candidate_table(rep: dict, cond: str, target: str, top: int = 10) -> str:
    if not rep:
        return ""
    slug = _slug(target)
    lines = [f"\\paragraph{{Elicited candidates}} (\\texttt{{{cond}}} condition, top {top} of "
             f"{len(rep['candidate_principals'])}, floor $+{rep['candidate_floor']}$).", "",
             "\\begin{table}[H]", "\\centering", "\\small",
             "\\begin{tabular}{@{}l r r r@{}}", "\\toprule",
             "Candidate & organism & base & elevation \\\\", "\\midrule"]
    for c in rep["candidate_principals"][:top]:
        name = c.get("display") or c["actor"].title()
        lines.append(f"{tex(name)} & {c['target_count']} & {c['reference_count']} "
                     f"& $+{c['elevation']}$ \\\\")
    aliases = rep.get("actor_aliases", {})
    note = ""
    if aliases:
        shown = list(aliases.items())[:2]
        note = (" Naming variants are merged when one name's words are a subset of the other's and "
                "they differ by at most one word: "
                + ", ".join(f"\\emph{{{tex(a)}}} into \\emph{{{tex(b)}}}" for a, b in shown) + ".")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\caption{Candidates for \\texttt{" + tex(target) + "}, by elevation over its base."
              + note + "}", f"\\label{{tab:data-candidates-{slug}}}", "\\end{table}", ""]
    return "\n".join(lines)
