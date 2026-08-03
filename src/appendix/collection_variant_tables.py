"""Stage 4 broken down by collection system-prompt variant.

Every instruction is asked under each of the condition's collection system
prompts, because a loyalty can be dormant without a deployment framing and
awake with one. The variant travels in the composite instruction id
``<system_id>::<template_id>`` and in the reply's own ``system_id`` field,
which this table reads back per model so an imbalance between variants is
visible. Replies from the first collection run carry no variant id and render
no rows here.
"""

from __future__ import annotations

from pathlib import Path

from src.appendix.appendix_card_layouts import listing_card
from src.appendix.latex_text_escaping import load, pending, rows, tex
from src.appendix.pipeline_run_registry import reported_seats

__all__ = ["variant_breakdown_table", "collection_system_card"]


def _variant_counts(path: Path) -> dict[str, dict]:
    per: dict[str, dict] = {}
    for r in rows(path):
        sid = r.get("system_id")
        if sid is None:
            continue
        c = per.setdefault(sid, {"n": 0, "refused": 0, "empty": 0, "failed": 0,
                                 "failed_recorded": False})
        c["n"] += 1
        c["refused"] += bool(r.get("refused"))
        c["empty"] += not (r.get("text") or "").strip()
        c["failed"] += bool(r.get("failed"))
        c["failed_recorded"] |= "failed" in r
    return per


def variant_breakdown_table(runs) -> str:
    body = []
    for name, d in runs:
        art = load(d / "prompt_sets.json") or {}
        declared = [v["id"] for v in art.get("collection_system_prompts", [])]
        first = True
        for model in reported_seats(sorted(p.name[len("responses_"):-len(".jsonl")]
                                           for p in d.glob("responses_*.jsonl"))):
            per = _variant_counts(d / f"responses_{model}.jsonl")
            order = [i for i in declared if i in per] + sorted(set(per) - set(declared))
            for j, sid in enumerate(order):
                c = per[sid]
                label = f"\\texttt{{{tex(name)}}}" if first else ""
                mlabel = f"\\texttt{{{tex(model)}}}" if j == 0 else ""
                failed = c["failed"] if c["failed_recorded"] else "--"
                body.append(f"{label} & {mlabel} & \\texttt{{{tex(sid)}}} & {c['n']} & "
                            f"{c['refused']} & {c['empty']} & {failed} \\\\")
                first = False
        if not first:
            body.append("\\arrayrulecolor{black!25}\\midrule")
    if not body:
        return pending("no reply carries a collection system-prompt variant id; this run asked "
                       "every instruction under a single framing")
    body[-1] = "\\bottomrule"
    return "\n".join(
        ["\\begin{table}[H]", "\\centering", "\\small",
         "\\begin{tabular}{@{}l l l r r r r@{}}", "\\toprule",
         "Run & Model & Variant & replies & refused & empty & failed \\\\", "\\midrule"]
        + body
        + ["\\end{tabular}",
           "\\caption{Stage 4 per collection system-prompt variant and model. Each variant asks "
           "the same rendered prompts, and the variant id is the prefix of the composite "
           "instruction id \\texttt{system\\_id::template\\_id}. Refused, empty, and failed are "
           "counted as in \\datasupp{}.}",
           "\\label{tab:data-collection-variants}", "\\end{table}", ""])


def collection_system_card(runs) -> str:
    """The collection system prompts of the first run that declares them, verbatim."""
    for name, d in runs:
        art = load(d / "prompt_sets.json") or {}
        variants = art.get("collection_system_prompts")
        if not variants:
            continue
        env = "promptcardCal" if name.startswith("calibration") else "promptcardChal"
        lab = "callabel" if name.startswith("calibration") else "challabel"
        label = (f"{tex(name.upper())} $\\cdot$ COLLECTION SYSTEM PROMPTS, "
                 f"{len(variants)} VARIANTS, {art.get('samples_per_cell', '?')} SAMPLES PER CELL")
        pairs = [(v["id"], v["text"] or "(empty)") for v in variants]
        return listing_card(pairs, label, env, lab)
    return pending("no run declares collection system prompts; this run asked every "
                   "instruction under a single framing")
