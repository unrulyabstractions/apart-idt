"""Stage 6 in the appendix: the registered test, then the two checks around it.

The registered test is the stage. The base-free table is what the same runs
look like to a detector with no base model to pair against, and the common-mode
table is the half a candidate-versus-candidate statistic cancels by
construction. Reporting either half alone is a bug, not a simplification, so
both are here whenever the stage has run at all.
"""

from __future__ import annotations

from pathlib import Path

from src.appendix.comparison_behavior_figures import behavior_figure_blocks
from src.appendix.comparison_check_tables import common_mode_table, naive_spread_table
from src.appendix.comparison_registered_tables import (
    comparison_runs,
    registered_test_table,
    top_pairs_table,
)
from src.appendix.latex_text_escaping import pending, tex
from src.appendix.pipeline_run_registry import expected_dirs, reported_runs

__all__ = ["comparison_section", "missing_comparison_note"]


def missing_comparison_note(out_root, done: set[str]) -> str:
    """Name every run whose stage 6 has not produced a summary."""
    root = Path(out_root) / "compare"
    on_disk = sorted(d.name for d in root.glob("*") if d.is_dir()) if root.exists() else []
    expected = reported_runs(sorted(set(on_disk) | set(expected_dirs("compare"))))
    missing = [name for name in expected if name not in done]
    if not missing:
        return ""
    # Ragged: the list is mostly unbreakable \texttt tokens, and justifying it
    # stretches the interword space across the whole paragraph.
    return ("{\\raggedright\\noindent\\textcolor{warn}{Pending: stage 6 has written no summary "
            "for " + ", ".join(f"\\texttt{{{tex(name)}}}" for name in missing)
            + f". That is {len(missing)} of {len(expected)} runs.}}\\par}}\n")


def comparison_section(out_root, run_key: str = "r1") -> str:
    runs = comparison_runs(out_root)
    done = {name for name, _summary, _display in runs}
    missing = missing_comparison_note(out_root, done)
    # No intro line: the appendix index already lists what this subsection holds.
    out = ["\\subsection{Comparing distributions}", "\\label{app:data-compare}", "",
           missing, ""]
    if not runs:
        # The note above already names every run, so a second pending line would
        # say the same thing twice.
        return "\n".join(out if missing
                         else out + [pending("stage 6 has not run for any run of this pipeline")])
    out += ["\\subsubsection{The registered test}", "",
            "The test of \\autoref{sec:comparing}, run at every judge level the condition "
            "affords.", ""]
    out += registered_test_table(runs)
    out += top_pairs_table(runs)

    out += ["\\subsubsection{Without the base model}", "",
            "The same runs scored on the target's spread across groups alone, as a detector "
            "with no base model to pair against would report it.", ""]
    out += naive_spread_table(runs)

    out += ["\\subsubsection{Common mode}", "",
            "Treatment delivered equally to every candidate is candidate-independent, so it "
            "cancels in the registered test and is visible only as the overall firing rate "
            "against the base model.", ""]
    out += common_mode_table(runs)

    out += ["\\subsubsection{Behavior distributions}", ""]
    out += behavior_figure_blocks(runs, run_key)
    return "\n".join(out)
