"""The stage-6 behavior figures, copied into the paper tree and wrapped.

Figures are copied so the appendix builds from the paper directory alone, and
each copy is namespaced by run: every run writes the same basename, so a plain
copy would have each run overwriting the last and every caption pointing at
whichever ran final.
"""

from __future__ import annotations

from pathlib import Path

from src.appendix.latex_text_escaping import pending, tex

__all__ = ["behavior_figure_blocks"]

#: Figures are copied here so the appendix builds from the paper directory alone.
FIGURE_DIR = Path("paper/figures/compare")


def behavior_figure_blocks(runs, run_key: str = "r1") -> list[str]:
    """The behavior distribution of every target model, one figure per model."""
    out = []
    for name, summary, _display in runs:
        for model, levels in sorted(summary["seats"].items()):
            for level, s in sorted(levels.items()):
                source = Path(s["figure"])
                if not source.exists():
                    continue
                pdf = source.with_suffix(".pdf")
                if not pdf.exists():
                    # A PNG without its PDF companion is a half-rendered figure.
                    # Crashing here would take the whole appendix down with it.
                    out += [pending(tex(f"figure PDF missing for {model} under "
                                        f"{name}, judge level {level}")), ""]
                    continue
                # Both runs write the same basename for the same run and model,
                # so each run's copies go to their own directory. A shared one
                # would have the second run overwrite the first and every
                # caption point at whichever generated last.
                target_dir = FIGURE_DIR / run_key
                target_dir.mkdir(parents=True, exist_ok=True)
                copied = f"{run_key}/{name}_{pdf.name}"
                (target_dir / f"{name}_{pdf.name}").write_bytes(pdf.read_bytes())
                # The label is a cross-reference key, not prose: an escaped
                # underscore inside \label breaks the control sequence outright.
                key = f"{name}-{model}-{level}".replace("_", "-")
                # A short caption carries the list of figures and the PDF outline.
                # The long one holds \texttt, which is fragile there and must not
                # reach either: unprotected, it breaks the \csname expansion.
                short = tex(f"Behavior distribution, {model} under {name}, judge level "
                            f"{level.lstrip('L')}")
                out += ["\\begin{figure}[H]", "\\centering",
                        f"\\includegraphics[width=\\linewidth]{{figures/compare/{copied}}}",
                        f"\\caption[{short}]{{Behavior distribution of "
                        "\\protect\\texttt{" + tex(model) + "} under \\protect\\texttt{"
                        + tex(name) + "}, judge level " + tex(level.lstrip("L"))
                        + ". Left: the share of each group's axis firings, on the axes that "
                        "separate the groups most. Right: how far each group's profile sits "
                        "from the average of the other groups'.}",
                        f"\\label{{fig:behavior-{key}}}", "\\end{figure}", ""]
    return out
