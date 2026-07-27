"""The data appendix, assembled in reading order and given its own label namespace.

The subsections are ordered by what a reader wants first, not by the order the
pipeline runs them. Distributions and the test come first, then the behaviours
that were scored, then the material each was computed from. Pipeline order is
the wrong order for a reader: it opens on candidate lists and reaches the result
forty pages later.

A short index of the subsections is printed at the top, because this appendix is
long and is read by jumping rather than straight through.

The calibration scope note sits directly under that index. It is generated
rather than written by hand so that a checkpoint cannot leave the reported set
without the appendix saying so.

Every label the document defines is suffixed with the run key so two runs could
be inputted into one document without collision, and the primary run also keeps
the unsuffixed label because the body cites it.
"""

from __future__ import annotations

from pathlib import Path

from src.appendix.pipeline_run_registry import DROPPED_NOTE
from src.appendix.run_label_namespacing import namespace_labels
from src.appendix.stage_collection_tables import collection_section
from src.appendix.stage_comparison_tables import comparison_section
from src.appendix.stage_conjecture_cards import conjecture_section
from src.appendix.stage_elicitation_tables import elicitation_section
from src.appendix.stage_promptset_cards import promptset_section
from src.appendix.stage_scoring_tables import scoring_section

__all__ = ["SECTION_ORDER", "experiment_data_document"]

#: Label, title, and one line saying what the reader will find. Reading order,
#: which is the reverse of pipeline order for everything except the inventory.
SECTION_ORDER: tuple[tuple[str, str, str], ...] = (
    ("app:data-compare", "Comparing distributions",
     "the registered test, the two checks reported beside it, and one behavior distribution per model"),
    ("app:data-hypotheses", "Hypothesis conjecture",
     "the conjectured behaviors, the scoring questions they became, and the guaranteed axes"),
    ("app:data-scoring", "Response scoring",
     "what the judge answered, per run, model, and level"),
    ("app:data-collection", "Response collection",
     "how many replies each seat produced, per run and per collection framing"),
    ("app:data-promptset", "Prompt set construction",
     "the templates, what was dropped, and one instruction as two candidates received it"),
    ("app:data-elicitation", "Principal elicitation",
     "the frozen questions, what both seats named, and the candidates that survived the floor"),
)


def _index_table() -> str:
    """A short contents block, because this appendix is navigated, not read through.

    No header row. ``\\autoref`` already prints the word Subsection in the first
    column, so a header over the second one would name it wrongly, and a contents
    block of six rows does not need its columns announced.
    """
    rows = "\n".join(
        f"\\autoref{{{label}}} & {title} & {blurb} \\\\"
        for label, title, blurb in SECTION_ORDER)
    return "\n".join([
        "\\begin{center}",
        "\\setlength{\\tabcolsep}{6pt}\\small",
        "\\begin{tabular}{@{}l l >{\\raggedright\\arraybackslash}p{0.52\\linewidth}@{}}",
        "\\toprule",
        rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{center}",
        "",
    ])


def experiment_data_document(out_root, run_key: str, run_label: str, primary: bool = False,
                             sibling_key: str = "", sibling_label: str = "",
                             scope_note: str = "") -> str:
    """The full ``\\section`` for the run, namespaced and ready to ``\\input``."""
    root = Path(out_root)
    parts = [
        "\\section{Experiment data}",
        "\\label{app:data}",
        "",
        "Everything the pipeline produced. Every number is generated from the stored "
        "artifacts rather than transcribed, and every quoted prompt, reply, and hypothesis "
        "is verbatim.",
        "",
        _index_table(),
        scope_note,
        "",
        DROPPED_NOTE,
        "",
        "%-----------------------------------",
        comparison_section(root, run_key), "",
        "\\clearpage", "%-----------------------------------",
        conjecture_section(root), "",
        "\\clearpage", "%-----------------------------------",
        scoring_section(root), "",
        "\\clearpage", "%-----------------------------------",
        collection_section(root), "",
        "\\clearpage", "%-----------------------------------",
        promptset_section(root), "",
        "\\clearpage", "%-----------------------------------",
        elicitation_section(root), "",
    ]
    body = "\n".join(p for p in parts if p is not None)
    # Collapse runs of blank lines. Two blank lines read as one paragraph break to
    # LaTeX but stack visible space when they fall between a float and a heading.
    while "\n\n\n" in body:
        body = body.replace("\n\n\n", "\n\n")
    header = ("% Generated by script/paper/write_data_appendix.py. Do not edit by hand:\n"
              f"% regenerate so the appendix matches the artifacts in {root}/.\n")
    return header + namespace_labels(body, run_key, keep_bare=primary)
