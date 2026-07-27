"""Card environments shared by the generated appendix sections.

A card is a labelled box of id and text rows. The question card sorts its rows
on the digits of the id; the listing card keeps the order it is given, because
template and hypothesis sets are already ordered on disk.
"""

from __future__ import annotations

import re
from pathlib import Path

from src.appendix.latex_text_escaping import load, mono_id, tex

__all__ = ["question_card", "listing_card"]


def question_card(path: Path, label: str, env: str, lab: str) -> str:
    art = load(path)
    if not art:
        return ""
    q = art["questions"]
    keys = sorted(q, key=lambda k: int(re.sub(r"\D", "", k) or 0))
    body = []
    for i, k in enumerate(keys):
        body.append(f"{mono_id(k)} & {tex(q[k])} \\\\")
        if i != len(keys) - 1:
            body.append("\\arrayrulecolor{white}\\hline")
    return (f"\\begin{{{env}}}\n\\{lab}{{{label}, ALL {len(keys)} QUESTIONS}}\n"
            "\\renewcommand{\\arraystretch}{1.15}\n"
            "\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{0.215\\linewidth}"
            "@{\\hskip 7pt}>{\\raggedright\\arraybackslash}p{0.695\\linewidth}@{}}\n"
            + "\n".join(body) + "\n\\end{tabular}\n"
            + f"\\end{{{env}}}\n")


def listing_card(pairs: list[tuple[str, str]], label: str, env: str, lab: str) -> str:
    """A card of id/text rows, delineated like the question cards."""
    body = []
    for i, (k, v) in enumerate(pairs):
        body.append(f"{mono_id(k)} & {tex(v)} \\\\")
        if i != len(pairs) - 1:
            body.append("\\arrayrulecolor{white}\\hline")
    return (f"\\begin{{{env}}}\n\\{lab}{{{label}}}\n"
            "\\renewcommand{\\arraystretch}{1.15}\n"
            "\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{0.235\\linewidth}"
            "@{\\hskip 7pt}>{\\raggedright\\arraybackslash}p{0.675\\linewidth}@{}}\n"
            + "\n".join(body) + "\n\\end{tabular}\n"
            + f"\\end{{{env}}}\n")
