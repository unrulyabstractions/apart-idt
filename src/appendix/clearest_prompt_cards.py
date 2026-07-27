"""The single prompt on which one candidate's treatment departs most.

Everything else in the paper is an average. This is the opposite: the one cell
where the gap is widest, quoted, so a reader can see what the statistic is made
of rather than only its summary.

The excess is recomputed here rather than read from a summary, because the
summary averages over instructions and the per-instruction value is what picks
the cell. It is recomputed with the registered test's own functions,
``cell_rates`` and ``excess_effect``, so the number in the card is the same
quantity the test maximizes and not a second definition of it.

What a card shows is one cell of the grid: one candidate, one behavior axis, one
instruction, in the target and in its base model. A cell rate on four samples is
a multiple of 1/4, so a single cell is a coarse measurement and the card says so.
The cell is an illustration of a signal established by the averaged test, never
evidence on its own.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from src.appendix.latex_text_escaping import load, tex
from src.appendix.results_top_behaviors_table import axis_label
from src.appendix.trigger_framing_table import FRAMINGS
from src.common.file_io import read_jsonl
from src.compare.behavior_count_table import build_behavior_table
from src.compare.paired_max_statistic import cell_rates, excess_effect

__all__ = ["QUOTED_RUNS", "clearest_prompt_cards"]

#: Run directory, the conjecture directory holding its axis registry, and how the
#: prose names it. Only the two runs Subsection 6.3 reads, to keep the number of
#: quoted prompts small.
QUOTED_RUNS = (
    ("calibration_informed", "calibration_informed", "the calibration organism",
     "promptcardCal"),
    ("challenge_organism_a", "challenge_blind", "challenge organism a", "promptcardChal"),
)

#: How much of a prompt is quoted. The prompts are one or two sentences, so this
#: is a guard against a pathological template rather than a routine truncation.
PROMPT_CAP = 320

_FRAMING_GLOSS = dict(FRAMINGS)


def _rows_by_level(score_dir: Path, seat: str) -> dict[int, list[dict]]:
    by_level: dict[int, list[dict]] = defaultdict(list)
    path = score_dir / f"verdicts_{seat}.jsonl"
    if not path.exists():
        return {}
    for row in read_jsonl(path):
        by_level[int(row["level"])].append(row)
    return by_level


def _widest_cell(out_root, run: str, conjecture: str):
    """The (candidate, instruction, axis) cell with the largest excess.

    Returns ``None`` when the run has no registered test, no stored axis
    registry, or no matched pair of seats: a missing card is reported as missing
    rather than filled with a cell from somewhere else.
    """
    root = Path(out_root)
    summary = load(root / "compare" / run / "comparison_summary.json")
    questions = load(root / "conjecture" / conjecture / "scoring_questions.json")
    if not summary or not questions:
        return None
    contrast = summary.get("reference_contrast") or {}
    target_seat, base_seat = contrast.get("target"), contrast.get("reference")
    levels = sorted(k for k, v in contrast.items()
                    if k.startswith("L") and isinstance(v, dict) and v.get("paired_max_test"))
    if not (target_seat and base_seat and levels):
        return None
    level_key = levels[-1]
    pm = contrast[level_key]["paired_max_test"]
    principal, axis_id = pm.get("principal"), pm.get("principal_axis")
    if not principal or not axis_id:
        return None

    axes = [a["axis_id"] for a in questions["axes"]]
    score_dir = root / "score" / run
    level = int(level_key[1:])
    target_rows = _rows_by_level(score_dir, target_seat).get(level)
    base_rows = _rows_by_level(score_dir, base_seat).get(level)
    if not target_rows or not base_rows:
        return None

    target = build_behavior_table(target_rows, axes)
    base = build_behavior_table(base_rows, axes)
    if target.blocks != base.blocks or target.principals != base.principals:
        return None
    t_rates, principals, _ = cell_rates(target)
    b_rates, _, _ = cell_rates(base)
    if principal not in principals or axis_id not in target.axes:
        return None
    d = excess_effect(t_rates, b_rates)
    c = principals.index(principal)
    j = list(target.axes).index(axis_id)
    column = np.nan_to_num(d[c, :, j], nan=-np.inf)
    i = int(np.argmax(column))
    if not np.isfinite(column[i]):
        return None
    instruction_id = target.blocks[i]
    system_id, _, template_id = instruction_id.partition("::")
    sets = load(score_dir / "prompt_sets.json") or {}
    text = next((p["text"] for p in (sets.get("prompt_sets", {}).get(principal) or [])
                 if p.get("instruction_id") == template_id), "")
    return {
        "level": level, "principal": principal,
        "display": (sets.get("principals", {}) or {}).get(principal, principal),
        "axis_id": axis_id, "framing": system_id, "template_id": template_id,
        "prompt": text, "excess": float(d[c, i, j]),
        "target_rate": float(t_rates[c, i, j]), "base_rate": float(b_rates[c, i, j]),
        "n_samples": int(sets.get("samples_per_cell") or 0),
    }


def clearest_prompt_cards(out_root) -> str:
    """One card per quoted run, ready to ``\\input`` from the Results section."""
    parts = ["% Generated by script/paper/write_data_appendix.py. Do not edit by hand:",
             "% edit src/appendix/clearest_prompt_cards.py and regenerate.", ""]
    wrote = 0
    for run, conjecture, prose, card in QUOTED_RUNS:
        cell = _widest_cell(out_root, run, conjecture)
        if not cell:
            parts += [f"\\noindent\\textcolor{{warn}}{{Pending: no widest cell for "
                      f"\\texttt{{{tex(run)}}}.}}\\par", ""]
            continue
        gloss = _FRAMING_GLOSS.get(cell["framing"], cell["framing"])
        label = "\\callabel" if card.endswith("Cal") else "\\challabel"
        parts += [
            f"\\begin{{{card}}}",
            label + "{" + tex(prose) + ": widest single cell, " + tex(cell["display"])
            + " on \\emph{" + axis_label(cell["axis_id"]) + "}}",
            f"\\textbf{{Framing:}} {tex(gloss)}. "
            f"\\textbf{{Instruction:}} \\texttt{{{tex(cell['template_id'])}}}.\\par\\smallskip",
            tex(cell["prompt"][:PROMPT_CAP]) + ("\\dots" if len(cell["prompt"]) > PROMPT_CAP
                                                else ""),
            "\\par\\smallskip",
            f"The behavior fires in {cell['target_rate']:.2f} of the target's replies here and "
            f"{cell['base_rate']:.2f} of its base model's, leaving an excess of "
            f"${cell['excess']:+.3f}$ against the other candidates."
            + (f" Each rate is over {cell['n_samples']} samples."
               if cell["n_samples"] else ""),
            f"\\end{{{card}}}",
            "",
        ]
        wrote += 1
    if not wrote:
        parts.append("% No run in this tree yielded a cell.")
    return "\n".join(parts)
