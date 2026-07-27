"""Stage 5 in the appendix: verdict counts, null verdicts, and firing rate.

The null column is reported because a null is a verdict the judge did not
return, and imputing it would invent behavior. The firing rate is reported
because the affordance levels move it: telling the judge that a loyalty may be
present changes how readily it answers yes, and that shift belongs in the
record next to any conclusion drawn from the scores.

The repair count is not here, and cannot be: a chunk short of ids is asked once
more for exactly the missing ids, and stage 5 aggregates that count into its
own return value rather than writing it to the verdict row. What the rows do
carry is the head of a reply that came back short even after the repair call,
counted below as short, since a null with no evidence cannot be triaged.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from src.appendix.latex_text_escaping import mono_id, rows
from src.appendix.stage_collection_tables import collection_runs, model_names

__all__ = ["scoring_section"]


def _by_level(path: Path) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for r in rows(path):
        grouped[int(r["level"])].append(r)
    return grouped


def scoring_section(out_root) -> str:
    body = []
    for name, directory, _art in collection_runs(out_root):
        models = model_names(directory, "verdicts")
        if not models:
            body.append(f"{mono_id(name)} & \\multicolumn{{7}}{{l}}"
                        "{\\textcolor{warn}{pending: no verdict written for this run}} \\\\")
            body.append("\\arrayrulecolor{black!25}\\midrule")
            continue
        first = True
        for model in models:
            levels = _by_level(directory / f"verdicts_{model}.jsonl")
            for level, verdicts in sorted(levels.items()):
                cells = sum(len(v["verdicts"]) for v in verdicts)
                nulls = sum(1 for v in verdicts for x in v["verdicts"].values() if x is None)
                fired = sum(1 for v in verdicts for x in v["verdicts"].values() if x is True)
                short = sum(1 for v in verdicts if v.get("raw_short"))
                label = mono_id(name) if first else ""
                body.append(f"{label} & {mono_id(model)} & {level} & {len(verdicts)} & "
                            f"{cells} & {nulls} & {short} & "
                            f"{100.0 * fired / max(1, cells):.1f}\\% \\\\")
                first = False
        body.append("\\arrayrulecolor{black!25}\\midrule")
    body[-1] = "\\bottomrule"
    header = "Run & Model & level & scored & verdicts & null & short & fired \\\\"
    return "\n".join(
        ["\\subsection{Response scoring}", "\\label{app:data-scoring}", "",
         "One row per run, model, and judge affordance level.", "",
         "\\begin{center}", "\\setlength{\\tabcolsep}{5pt}\\footnotesize",
         "\\begin{longtable}{@{}l l r r r r r r@{}}", "\\toprule", header, "\\midrule",
         "\\endfirsthead", "\\toprule", header, "\\midrule", "\\endhead"] + body
        + ["\\caption{Stage 5 per run, model, and judge affordance level. \\emph{Scored} counts "
           "replies read, \\emph{verdicts} the axis answers those replies produced, "
           "\\emph{null} the answers the judge did not return, which are recorded and never "
           "imputed, and \\emph{short} the replies still missing an axis after the repair call, "
           "which keep the head of the judge's raw output. \\emph{Fired} is the share of "
           "verdicts answered yes. The number of verdicts the repair call recovered is "
           "aggregated by the stage and not written to the rows, so it cannot be quoted here.}",
           "\\label{tab:data-scoring}", "\\end{longtable}", "\\end{center}", ""])
