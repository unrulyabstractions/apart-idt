"""Challenge organisms in the appendix: pooled candidates, then per-seed blocks.

The pooled tally is read from the frozen report that stages 2 and 4 consumed,
not recomputed here. Recomputing it drifted: the same elevation tie between two
candidates was broken one way by this table and the other way by
``tab:elicit-top3``, and the paper carried both.
"""

from __future__ import annotations

from pathlib import Path

from src.appendix.calibration_run_tables import seat_tags
from src.appendix.latex_text_escaping import load, pending, tex

__all__ = ["ORGANISMS", "SEED_KINDS", "challenge_table"]

SEED_KINDS = ("person", "group", "organization")
ORGANISMS = ("a", "b", "c")


def challenge_table(out_root: Path, top: int = 3) -> str:
    rep = {org: load(out_root / f"ellicit/challenge_organism_{org}/elicitation_report.json")
           for org in ORGANISMS}
    out = ["\\subsubsection{Challenge organisms, per target}", "",
           "Pooled across the three typed seeds, since each organism runs all three. "
           "One block per organism follows, per seed and per system-prompt variant.", ""]
    if not any(rep.values()):
        out.append(pending("extraction not run"))
        return "\n".join(out)
    out += ["\\begin{table}[H]", "\\centering", "\\small",
            "\\begin{tabular}{@{}l l r r r@{}}", "\\toprule",
            "Target & Candidate & organism & base & elevation \\\\", "\\midrule"]
    for org in ORGANISMS:
        if not rep[org]:
            continue
        for i, c in enumerate(rep[org]["candidate_principals"][:top]):
            name = f"\\texttt{{sl-organism-{org}-7b}}" if i == 0 else ""
            display = c.get("display") or c["actor"].title()
            out.append(f"{name} & {tex(display)} & {c['target_count']} & "
                       f"{c['reference_count']} & $+{c['elevation']}$ \\\\")
        out.append("\\arrayrulecolor{black!25}\\midrule")
    out[-1] = "\\bottomrule"
    out += ["\\end{tabular}",
            "\\caption{Top three candidates per challenge organism, pooled over the person, "
            "group, and organization seeds.}", "\\label{tab:data-challenge}", "\\end{table}", ""]
    for org in ORGANISMS:
        out.append(_challenge_seed_table(out_root, org))
    return "\n".join(out)


def _challenge_seed_table(out_root: Path, org: str) -> str:
    """One challenge organism's naming rate, per seed kind and system-prompt variant.

    The pooled report carries no per-variant breakdown, so the counts come from
    the three seed runs it was pooled from.
    """
    rep = {kind: load(out_root / f"ellicit/organism_{org}_{kind}/elicitation_report.json")
           for kind in SEED_KINDS}
    have = [k for k in SEED_KINDS if (rep.get(k) or {}).get("coverage_by_system")]
    # \paragraph is run-in in arxiv.sty (negative after-skip), so it is held back
    # until the next horizontal material. A [H] table is not that, and without
    # \leavevmode each heading is set below the table it names rather than above.
    # \nopagebreak then keeps the heading on the page its table lands on.
    head = (f"\\paragraph{{\\texttt{{sl-organism-{org}-7b}}}}\\leavevmode\n"
            "\\nopagebreak[4]")
    if not have:
        return f"{head}\n\n" + pending("elicitation not run for this organism")
    base = seat_tags(rep[have[0]])[1]
    lines = [head, "",
             "\\begin{table}[H]", "\\centering", "\\small",
             "\\begin{tabular}{@{}l l r r@{}}", "\\toprule",
             "Seed & System-prompt variant & organism & base \\\\", "\\midrule"]
    for kind in have:
        tt, rt = seat_tags(rep[kind])
        per = rep[kind]["coverage_by_system"]
        po, pb = per.get(tt, {}), per.get(rt, {})
        for i, sid in enumerate(sorted(po)):
            first = f"\\texttt{{{kind}}}" if i == 0 else ""
            lines.append(f"{first} & \\texttt{{{tex(sid)}}} & {po[sid]['named']} / {po[sid]['total']} "
                         f"& {pb.get(sid, {}).get('named', 0)} / {pb.get(sid, {}).get('total', 0)} \\\\")
        lines.append("\\arrayrulecolor{black!25}\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}",
              "\\caption{Replies naming an entity, out of the total asked, per seed kind and "
              "system-prompt variant, for \\texttt{sl-organism-" + org + "-7b} and the shared "
              "base \\texttt{" + tex(base) + "}.}",
              f"\\label{{tab:data-challenge-variants-{org}}}", "\\end{table}", ""]
    return "\n".join(lines)
