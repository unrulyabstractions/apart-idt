"""Every candidate principal every stage-1 run elicited, and every name merged.

The candidate list is the whole output of stage 1: the actors at or above the
floor, ranked by how much more often the target named them than its base did.
Showing only the top few would hide the shape of the tail, which is what says
whether the top candidate stands apart or sits in a crowd. The alias table is
the other half of the same accounting: two spellings of one actor are one
candidate only because a merge said so, and the merges are recorded here so the
counts above can be checked against them.
"""

from __future__ import annotations

from src.appendix.elicitation_coverage_tables import elicitation_reports
from src.appendix.latex_text_escaping import mono_id, pending, tex

__all__ = ["candidate_list_table", "alias_merge_table"]


def _display(candidate: dict) -> str:
    return tex(candidate.get("display") or candidate["actor"].title())


def candidate_list_table(out_root) -> str:
    """Every candidate of every run, grouped by run, with its elevation."""
    reports = elicitation_reports(out_root)
    if not reports:
        return pending("no elicitation report exists yet")
    header = "Candidate & organism & base & elevation \\\\"
    body = []
    for name, report in reports:
        candidates = report.get("candidate_principals", [])
        floor = report.get("candidate_floor", "?")
        pooled = report.get("n_actors_pooled", "?")
        body.append("\\multicolumn{4}{@{}l}{\\texttt{" + tex(name) + "} \\quad "
                    + f"{len(candidates)} candidates at or above the floor $+{floor}$, "
                    + f"of {pooled} actors named at all" + "} \\\\")
        if not candidates:
            body.append("\\multicolumn{4}{@{}l}{\\textcolor{warn}{No actor reached the floor "
                        "in this run.}} \\\\")
        for candidate in candidates:
            body.append(f"{_display(candidate)} & {candidate['target_count']} & "
                        f"{candidate['reference_count']} & $+{candidate['elevation']}$ \\\\")
        body.append("\\arrayrulecolor{black!25}\\midrule")
    body[-1] = "\\bottomrule"
    return "\n".join(
        ["\\begin{center}", "\\setlength{\\tabcolsep}{5pt}\\footnotesize",
         "\\begin{longtable}{@{}>{\\raggedright\\arraybackslash}p{0.5\\linewidth} r r r@{}}",
         "\\toprule", header, "\\midrule", "\\endfirsthead", "\\toprule", header,
         "\\midrule", "\\endhead"] + body
        + ["\\caption{Every candidate principal elicited, per run, ranked by elevation over "
           "the run's own base. \\emph{Organism} and \\emph{base} are the number of replies "
           "from which the extractor read that actor; elevation is their difference, and the "
           "floor is the smallest elevation a run admits. Actors below the floor are counted "
           "in the pooled total and are not candidates.}",
           "\\label{tab:data-candidates}", "\\end{longtable}", "\\end{center}", ""])


def alias_merge_table(out_root) -> str:
    """Every naming variant folded into a canonical actor, per run."""
    reports = elicitation_reports(out_root)
    merged = [(name, report.get("actor_aliases") or {}) for name, report in reports]
    if not any(aliases for _n, aliases in merged):
        return pending("no run merged a naming variant")
    header = "Run & Merges \\\\"
    body = []
    for name, aliases in merged:
        if not aliases:
            cell = "none"
        else:
            cell = "; ".join(f"\\emph{{{tex(a)}}} $\\rightarrow$ \\emph{{{tex(b)}}}"
                             for a, b in sorted(aliases.items()))
        body.append(f"{mono_id(name)} & {len(aliases)}: {cell} \\\\"
                    if aliases else f"{mono_id(name)} & none \\\\")
        body.append("\\arrayrulecolor{black!25}\\midrule")
    body[-1] = "\\bottomrule"
    columns = ("@{}>{\\raggedright\\arraybackslash}p{0.17\\linewidth} "
               ">{\\raggedright\\arraybackslash}p{0.74\\linewidth}@{}")
    return "\n".join(
        ["\\begin{center}", "\\setlength{\\tabcolsep}{4pt}\\footnotesize",
         f"\\begin{{longtable}}{{{columns}}}",
         "\\toprule", header, "\\midrule", "\\endfirsthead", "\\toprule", header,
         "\\midrule", "\\endhead"] + body
        + ["\\caption{Naming variants merged before the tally, per run. Two names are merged "
           "when one name's words are a subset of the other's and they differ by at most one "
           "word, so the counts of \\datasupp{} are counts of the actor on "
           "the right.}",
           "\\label{tab:data-aliases}", "\\end{longtable}", "\\end{center}", ""])
