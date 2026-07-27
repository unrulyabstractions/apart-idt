"""Every frozen elicitation question set, once, with the runs that share it.

Question sets are built once per organism group and condition and reused by
every target in that group, so quoting one card per run directory would print
the same twenty questions four times. Sets are grouped by content instead: one
card per distinct set, naming the run directories that asked it. A directory
whose sampling has not finished still has its questions frozen, so it appears
here even when it has no report yet.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.appendix.appendix_card_layouts import question_card
from src.appendix.latex_text_escaping import load, pending, tex
from src.appendix.pipeline_run_registry import reported_runs, run_group

__all__ = ["question_set_cards"]


def _digest(questions: dict) -> str:
    return hashlib.sha1(json.dumps(questions, sort_keys=True).encode()).hexdigest()


def question_set_cards(out_root) -> str:
    """One card per distinct frozen question set, in run-directory order."""
    root = Path(out_root) / "ellicit"
    groups: dict[str, list[str]] = {}
    paths: dict[str, Path] = {}
    for name in reported_runs(sorted(d.name for d in root.glob("*") if d.is_dir())):
        artifact = load(root / name / "questions.json")
        if not artifact or not artifact.get("questions"):
            continue
        key = _digest(artifact["questions"])
        groups.setdefault(key, []).append(name)
        paths.setdefault(key, root / name / "questions.json")
    if not groups:
        return pending("no question set is frozen yet")
    out = []
    for key, names in groups.items():
        shared = ", ".join(f"\\texttt{{{tex(n)}}}" for n in names)
        label = f"{tex(names[0].upper())} $\\cdot$ ASKED BY {len(names)} RUN"
        label += "S" if len(names) != 1 else ""
        env = "promptcardCal" if run_group(names[0]) == "calibration" else "promptcardChal"
        lab = "callabel" if run_group(names[0]) == "calibration" else "challabel"
        out.append(question_card(paths[key], label, env, lab))
        out.append(f"\\noindent{{\\footnotesize Asked by: {shared}.}}\n")
    return "\n".join(out)
