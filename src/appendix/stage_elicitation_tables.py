"""Stage 1 in the appendix, assembled from the per-aspect tables.

Order follows the stage: the frozen questions, what each seat answered, the
same split by system-prompt variant, the candidates that survived the floor,
the naming variants merged to get there, and replies with the actor the
extractor read from them.
"""

from __future__ import annotations

from pathlib import Path

from src.appendix.elicitation_candidate_lists import alias_merge_table, candidate_list_table
from src.appendix.elicitation_coverage_tables import (
    counting_rule_note,
    elicitation_reports,
    seat_coverage_table,
    variant_coverage_table,
)
from src.appendix.elicitation_question_sets import question_set_cards
from src.appendix.elicitation_seat_naming import seat_names
from src.appendix.latex_text_escaping import pending, rows, tex

__all__ = ["SHOWCASE", "elicitation_section"]

#: The run whose replies are quoted when one example is enough.
SHOWCASE = "calibration_typed"


def elicitation_section(out_root: Path) -> str:
    return "\n".join(
        ["\\subsection{Principal elicitation}", "\\label{app:data-elicitation}", "",
         "The Elicitor's frozen questions, what both seats answered, the candidates that "
         "survived the floor, and the naming variants merged on the way.", "",
         "\\subsubsection{Frozen question sets}", "",
         "One set per organism group and condition, reused by every target in it. Each "
         "distinct set is quoted once, in full, with the runs that asked it.", "",
         question_set_cards(out_root), "",
         "\\clearpage",
         "\\subsubsection{Coverage, per run and seat}", "",
         counting_rule_note(out_root), "",
         seat_coverage_table(out_root), "",
         "\\subsubsection{Coverage, per system-prompt variant}", "",
         variant_coverage_table(out_root), "",
         "\\clearpage",
         "\\subsubsection{Candidate principals}", "",
         candidate_list_table(out_root), "",
         "\\subsubsection{Merged naming variants}", "",
         alias_merge_table(out_root), "",
         "\\subsubsection{Replies and extracted actors}", "",
         "Two replies the extractor read an entity from and one it read as favoring nobody, "
         "truncated at the reply cap.", "",
         _reply_samples(out_root), ""])


def _favored(directory: Path, seat: str) -> dict:
    return {(r["question_id"], r["s"], r.get("system_id", "")): r["favored"]
            for r in rows(directory / f"favored_{seat}.jsonl")}


def _card(row: dict, who: str, verdict: str) -> str:
    head = (f"{who} $\\cdot$ \\texttt{{{tex(row['question_id'])}}}, "
            f"\\texttt{{{tex(row.get('system_id', ''))}}}, SAMPLE {row['s']} "
            f"$\\rightarrow$ \\emph{{{tex(verdict)}}}")
    return ("\\begin{promptcard}\n\\promptlabel{" + head + "}\n"
            + tex(row["text"][:330]) + "\n\\end{promptcard}\n")


def _reply_samples(out_root: Path) -> str:
    """Replies from the showcase run, or from the first run that has any."""
    reports = elicitation_reports(out_root)
    if not reports:
        return pending("no elicitation reply exists yet")
    for name, report in sorted(reports, key=lambda item: item[0] != SHOWCASE):
        directory = Path(out_root) / "ellicit" / name
        target, reference = seat_names(report)
        favored = _favored(directory, target)
        named = [r for r in rows(directory / f"responses_{target}.jsonl")
                 if (favored.get((r["question_id"], r["s"], r.get("system_id", "")))
                     or "none").lower() != "none"]
        if not named:
            continue
        base_favored = _favored(directory, reference)
        base_none = [r for r in rows(directory / f"responses_{reference}.jsonl")
                     if (base_favored.get((r["question_id"], r["s"], r.get("system_id", "")))
                         or "none").lower() == "none"]
        body = [f"\\noindent{{\\footnotesize From \\texttt{{{tex(name)}}}, organism seat "
                f"\\texttt{{{tex(target)}}} against base seat \\texttt{{{tex(reference)}}}.}}\n"]
        for row in named[:2]:
            body.append(_card(row, "ORGANISM",
                              favored[(row["question_id"], row["s"], row.get("system_id", ""))]))
        if base_none:
            body.append(_card(base_none[0], "BASE", "none"))
        return "\n".join(body)
    return pending("no reply had an actor extracted from it")
