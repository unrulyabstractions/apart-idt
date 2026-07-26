"""Stage 5 in the appendix: verdict counts, null verdicts, and firing rate.

The null column is reported because a null is a verdict the judge did not
return, and imputing it would invent behavior. The firing rate is reported
because the affordance levels move it: telling the judge that a loyalty may be
present changes how readily it answers yes, and that shift belongs in the
record next to any conclusion drawn from the scores.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from src.appendix.latex_text_escaping import pending, rows, tex
from src.appendix.stage_collection_tables import score_runs

__all__ = ["scoring_section"]


def _by_level(path: Path) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for r in rows(path):
        grouped[int(r["level"])].append(r)
    return grouped


def scoring_section(out_root) -> str:
    runs = score_runs(out_root)
    out = ["\\subsection{Response scoring}", "\\label{app:data-scoring}", ""]
    body = []
    for name, d in runs:
        first = True
        for model in sorted(p.name[len("verdicts_"):-len(".jsonl")]
                           for p in d.glob("verdicts_*.jsonl")):
            for level, verdicts in sorted(_by_level(d / f"verdicts_{model}.jsonl").items()):
                cells = sum(len(v["verdicts"]) for v in verdicts)
                nulls = sum(1 for v in verdicts for x in v["verdicts"].values() if x is None)
                fired = sum(1 for v in verdicts for x in v["verdicts"].values() if x is True)
                label = f"\\texttt{{{tex(name)}}}" if first else ""
                body.append(f"{label} & \\texttt{{{tex(model)}}} & {level} & {len(verdicts)} & "
                            f"{cells} & {nulls} & {100.0 * fired / max(1, cells):.1f}\\% \\\\")
                first = False
        if not first:
            body.append("\\arrayrulecolor{black!25}\\midrule")
    if not body:
        return "\n".join(out + [pending("stage not run")])
    body[-1] = "\\bottomrule"
    out += ["",
            "\\begin{table}[H]", "\\centering", "\\small",
            "\\begin{tabular}{@{}l l r r r r r@{}}", "\\toprule",
            "Run & Model & level & scored & verdicts & null & fired \\\\", "\\midrule"]
    out += body
    out += ["\\end{tabular}",
            "\\caption{Stage 5 per run, model, and judge affordance level. "
            "\\emph{Fired} is the share of verdicts answered yes.}",
            "\\label{tab:data-scoring}", "\\end{table}", ""]
    return "\n".join(out)
