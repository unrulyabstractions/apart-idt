"""Regenerate the data appendix from the artifacts on disk.

    uv run python script/paper/write_data_appendix.py

The appendix quotes what each stage produced. Hand-editing it is how it drifts
from the runs, which has already happened once: a question card showed twenty
questions no model was ever asked. This script reads out/ and writes the
appendix, so the paper is never newer or older than the data. Stages with no
artifacts are written as pending rather than omitted.
"""

from __future__ import annotations

import re
from pathlib import Path

from src.appendix.latex_text_escaping import load, mono_id, pending, rows, tex
from src.appendix.elicit_top_table import elicit_top_table
from src.appendix.stage_collection_tables import collection_section
from src.appendix.stage_comparison_tables import comparison_section
from src.appendix.stage_scoring_tables import scoring_section

OUT = Path("out")
APPENDIX = Path("paper/appendix/experiment_data.tex")
#: Pulled into the protocol appendix with \input, so its prose stays hand-written.
TOP_TABLE = Path("paper/appendix/elicit_top_candidates.tex")
CAL_CONDITIONS = ("blind", "typed", "informed")
SEED_KINDS = ("person", "group", "organization")
ORGANISMS = ("a", "b", "c")
#: The condition whose artifacts are quoted when one example is enough.
SHOWCASE = "typed"
#: Every calibration checkpoint in the organism paper, with the tag its seat uses.
CAL_TARGETS = (
    ("12-mar-gen9-1.5b", "gen9_1p5b"),
    ("16-mar-gen9-7b", "gen9_7b"),
    ("16-mar-gen9-7b-positive-only", "gen9_7b_pos"),
    ("12-mar-gen9-32b", "gen9_32b"),
)


def _elicit_dir(condition: str, tag: str) -> str:
    """The 1.5B run predates the per-target suffix and keeps the bare name."""
    return f"calibration_{condition}" if tag == "gen9_1p5b" else f"calibration_{condition}_{tag}"


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


def elicitation_section() -> str:
    out = ["\\subsection{Principal elicitation}", "\\label{app:data-elicitation}", "",
           "\\subsubsection{Question sets, per organism group}", "",
           "One question set per organism group and condition, reused by every target in it. "
           "One of each is shown.", ""]
    out.append(question_card(OUT / f"ellicit/calibration_{SHOWCASE}/questions.json",
                             f"CALIBRATION $\\cdot$ \\texttt{{{SHOWCASE}}}, POLITICAL SEED",
                             "promptcardCal", "callabel"))
    out.append(question_card(OUT / "ellicit/organism_a_person/questions.json",
                             "CHALLENGE $\\cdot$ \\texttt{blind}, PERSON SEED",
                             "promptcardChal", "challabel"))

    out += ["\\subsubsection{Calibration organisms, per target}", "",
            "One block per calibration checkpoint. A block is present for every target in "
            "\\autoref{tab:targets}, so what has not been run is visible rather than absent.", ""]
    for target, tag in CAL_TARGETS:
        rep = {c: load(OUT / f"ellicit/{_elicit_dir(c, tag)}/elicitation_report.json")
               for c in CAL_CONDITIONS}
        out += [f"\\paragraph{{\\texttt{{{tex(target)}}}}}", ""]
        if not any(rep.values()):
            out += [pending("elicitation not run for this target"), ""]
            continue
        missing = [c for c in CAL_CONDITIONS if not rep.get(c)]
        if missing:
            out += ["\\textcolor{warn}{Conditions still running: "
                    + ", ".join(f"\\texttt{{{c}}}" for c in missing) + ".}", ""]
        if tag == "gen9_1p5b":
            out.append(reply_samples(SHOWCASE))
        out.append(condition_table(rep, target))
        shown = SHOWCASE if rep.get(SHOWCASE) else next(c for c in CAL_CONDITIONS if rep.get(c))
        out.append(candidate_table(rep[shown], shown, target))
    out.append(challenge_table())
    return "\n".join(out)


def reply_samples(cond: str) -> str:
    d = OUT / f"ellicit/calibration_{cond}"
    fav = {(r["question_id"], r["s"], r["system_id"]): r["favored"]
           for r in rows(d / "favored_gen9_1p5b.jsonl")}
    named = [r for r in rows(d / "responses_gen9_1p5b.jsonl")
             if (fav.get((r["question_id"], r["s"], r["system_id"])) or "none").lower() != "none"]
    bfav = {(r["question_id"], r["s"], r["system_id"]): r["favored"]
            for r in rows(d / "favored_base_1p5b.jsonl")}
    bnone = [r for r in rows(d / "responses_base_1p5b.jsonl")
             if (bfav.get((r["question_id"], r["s"], r["system_id"])) or "none").lower() == "none"]
    if not named:
        return pending("no named replies to quote")

    def card(row, who, verdict):
        head = (f"{who} $\\cdot$ \\texttt{{{tex(row['question_id'])}}}, "
                f"{tex(row['system_id'])}, SAMPLE {row['s']} $\\rightarrow$ \\emph{{{tex(verdict)}}}")
        return ("\\begin{promptcard}\n\\promptlabel{" + head + "}\n"
                + tex(row["text"][:330]) + "\n\\end{promptcard}\n")

    body = [f"\\paragraph{{Replies and extracted actors}} (\\texttt{{{cond}}} condition, "
            "truncated at the reply cap).\n"]
    for r in named[:2]:
        body.append(card(r, "ORGANISM", fav[(r["question_id"], r["s"], r["system_id"])]))
    if bnone:
        body.append(card(bnone[0], "BASE", "none"))
    return "\n".join(body)


def _seat_tags(report: dict) -> tuple[str, str]:
    """The two seat tags this run actually used, read from its own config echo.

    Hardcoding them would silently mislabel every target after the first: the 7B
    and 32B runs read against different bases, and the coverage dictionaries are
    keyed on the tag.
    """
    config = report.get("config", {})
    target = (config.get("target") or {}).get("tag") or "target"
    reference = (config.get("reference") or {}).get("tag") or "reference"
    return target, reference


def _slug(target: str) -> str:
    """A label-safe key. LaTeX labels cannot carry the dots or hyphens of a repo id."""
    return re.sub(r"[^a-z0-9]+", "", target.lower())


def condition_table(rep: dict, target: str) -> str:
    have = [c for c in CAL_CONDITIONS if rep.get(c)]
    if not have:
        return ""
    slug = _slug(target)
    lines = ["\\paragraph{Run statistics, per condition.}", "",
             "\\begin{table}[H]", "\\centering", "\\small",
             "\\begin{tabular}{@{}l r r r r r@{}}", "\\toprule",
             f"\\texttt{{\\scriptsize {tex(target)}}} & \\multicolumn{{2}}{{c}}{{named an entity}} "
             "& \\multicolumn{2}{c}{unparseable} & distinct \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
             "Condition & organism & base & organism & base & actors \\\\", "\\midrule"]
    for c in have:
        r = rep[c]
        tt, rt = _seat_tags(r)
        o, b = r["coverage"].get(tt, {}), r["coverage"].get(rt, {})
        lines.append(f"\\texttt{{{c}}} & {o.get('named', 0)} / {o.get('total', 0)} & "
                     f"{b.get('named', 0)} / {b.get('total', 0)} & {o.get('unparseable', 0)} & "
                     f"{b.get('unparseable', 0)} & {len(r['tally_target'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\caption{Elicitation run statistics for \\texttt{" + tex(target) + "}, per audit "
              "condition. No reply was refused or empty in any run.}",
              f"\\label{{tab:data-runstats-{slug}}}", "\\end{table}", ""]

    # Per-variant naming rate, which is what shows the instrument working.
    lines += ["\\begin{table}[H]", "\\centering", "\\small",
              "\\begin{tabular}{@{}l l r r@{}}", "\\toprule",
              "Condition & System-prompt variant & organism & base \\\\", "\\midrule"]
    for c in have:
        tt, rt = _seat_tags(rep[c])
        per = rep[c].get("coverage_by_system", {})
        po, pb = per.get(tt, {}), per.get(rt, {})
        for i, sid in enumerate(sorted(po)):
            first = f"\\texttt{{{c}}}" if i == 0 else ""
            lines.append(f"{first} & \\texttt{{{tex(sid)}}} & {po[sid]['named']} / {po[sid]['total']} "
                         f"& {pb.get(sid, {}).get('named', 0)} / {pb.get(sid, {}).get('total', 0)} \\\\")
        lines.append("\\arrayrulecolor{black!25}\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}",
              "\\caption{Replies naming an entity, per system-prompt variant, for \\texttt{"
              + tex(target) + "}.}",
              f"\\label{{tab:data-variants-{slug}}}", "\\end{table}", ""]
    return "\n".join(lines)


def candidate_table(rep: dict, cond: str, target: str, top: int = 10) -> str:
    if not rep:
        return ""
    slug = _slug(target)
    lines = [f"\\paragraph{{Elicited candidates}} (\\texttt{{{cond}}} condition, top {top} of "
             f"{len(rep['candidate_principals'])}, floor $+{rep['candidate_floor']}$).", "",
             "\\begin{table}[H]", "\\centering", "\\small",
             "\\begin{tabular}{@{}l r r r@{}}", "\\toprule",
             "Candidate & organism & base & elevation \\\\", "\\midrule"]
    for c in rep["candidate_principals"][:top]:
        name = c.get("display") or c["actor"].title()
        lines.append(f"{tex(name)} & {c['target_count']} & {c['reference_count']} "
                     f"& $+{c['elevation']}$ \\\\")
    aliases = rep.get("actor_aliases", {})
    note = ""
    if aliases:
        shown = list(aliases.items())[:2]
        note = (" Naming variants are merged when one name's words are a subset of the other's and "
                "they differ by at most one word: "
                + ", ".join(f"\\emph{{{tex(a)}}} into \\emph{{{tex(b)}}}" for a, b in shown) + ".")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\caption{Candidates for \\texttt{" + tex(target) + "}, by elevation over its base."
              + note + "}", f"\\label{{tab:data-candidates-{slug}}}", "\\end{table}", ""]
    return "\n".join(lines)


def challenge_table(top: int = 3) -> str:
    """Challenge organisms: the pooled candidate list, then one block per organism.

    The pooled tally is read from the frozen report that stages 2 and 4 consumed,
    not recomputed here. Recomputing it drifted: the same elevation tie between
    two candidates was broken one way by this table and the other way by
    ``tab:elicit-top3``, and the paper carried both.
    """
    rep = {org: load(OUT / f"ellicit/challenge_organism_{org}/elicitation_report.json")
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
        out.append(challenge_seed_table(org))
    return "\n".join(out)


def challenge_seed_table(org: str) -> str:
    """One challenge organism's naming rate, per seed kind and system-prompt variant.

    The pooled report carries no per-variant breakdown, so the counts come from
    the three seed runs it was pooled from.
    """
    rep = {kind: load(OUT / f"ellicit/organism_{org}_{kind}/elicitation_report.json")
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
    base = _seat_tags(rep[have[0]])[1]
    lines = [head, "",
             "\\begin{table}[H]", "\\centering", "\\small",
             "\\begin{tabular}{@{}l l r r@{}}", "\\toprule",
             "Seed & System-prompt variant & organism & base \\\\", "\\midrule"]
    for kind in have:
        tt, rt = _seat_tags(rep[kind])
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


def promptset_section() -> str:
    out = ["\\subsection{Prompt set construction}", "\\label{app:data-promptset}", "",
           "One template set per organism group and condition. Five representative templates of "
           "each; the count kept and rejected is given per set.", ""]
    for cond in CAL_CONDITIONS:
        art = load(OUT / f"promptset/calibration_{cond}/templates.json")
        if not art:
            continue
        label = (f"CALIBRATION $\\cdot$ \\texttt{{{cond}}}, LEVEL {art['level']}, "
                 f"{len(art['templates'])} KEPT, {art['n_rejected']} REJECTED")
        out.append(listing_card(list(art["templates"].items())[:5], label,
                                "promptcardCal", "callabel"))
    art = load(OUT / "promptset/challenge_blind/templates.json")
    if art:
        label = (f"CHALLENGE $\\cdot$ \\texttt{{blind}}, LEVEL {art['level']}, "
                 f"{len(art['templates'])} KEPT, {art['n_rejected']} REJECTED")
        out.append(listing_card(list(art["templates"].items())[:5], label,
                                "promptcardChal", "challabel"))

    rendered = load(OUT / "promptset/calibration_typed/prompts_emmanuel_macron.json")
    if rendered:
        first = rendered["prompts"][0]
        out += ["\\subsubsection{Rendering}", "",
                "Substituting a candidate's name into one template gives that candidate's copy of "
                "that instruction, so the sets differ only in the name.", "",
                "\\begin{promptcard}\n\\promptlabel{\\texttt{"
                + tex(first["prompt_id"]) + "}}\n" + tex(first["text"]) + "\n\\end{promptcard}\n"]
    return "\n".join(out)


def conjecture_section() -> str:
    out = ["\\subsection{Hypothesis conjecture}", "\\label{app:data-hypotheses}", "",
           "One hypothesis set per organism group and condition, each on its own page. Three of "
           "each set are shown, with the scoring questions they became.", ""]
    sets = [(f"calibration_{c}", f"CALIBRATION $\\cdot$ \\texttt{{{c}}}", "promptcardCal", "callabel")
            for c in CAL_CONDITIONS]
    sets.append(("challenge_blind", "CHALLENGE $\\cdot$ \\texttt{blind}", "promptcardChal", "challabel"))
    written = 0
    for tag, label, env, lab in sets:
        hyp = load(OUT / f"conjecture/{tag}/hypotheses.json")
        sco = load(OUT / f"conjecture/{tag}/scoring_questions.json")
        if not (hyp and sco):
            continue
        # Each set on a fresh page after the first: the two cards plus the rejection
        # note do not fit beside another set's. The first set stays on the page the
        # subsection opened, or that heading is left alone on an otherwise blank one.
        if written:
            out.append("\\clearpage")
        written += 1
        out.append(f"\\subsubsection{{{label.split('$\\cdot$')[0].strip().title()}, "
                   f"\\texttt{{{tag.split('_')[-1]}}}}}")
        out.append("")
        out.append(f"Level {hyp['level']}: {len(hyp['hypotheses'])} hypotheses, "
                   f"{len(sco['axes'])} scoring questions kept, {sco['n_rejected']} rejected.\n")
        out.append(listing_card(list(hyp["hypotheses"].items())[:3],
                                label + " $\\cdot$ STEP 1, HYPOTHESES", env, lab))
        out.append(listing_card([(a["axis_id"], a["question"]) for a in sco["axes"][:3]],
                                label + " $\\cdot$ STEP 2, SCORING QUESTIONS", env, lab))
        if sco["rejected"]:
            key, reason = next(iter(sco["rejected"].items()))
            out.append("\\begin{promptcard}\n\\promptlabel{REJECTED $\\cdot$ \\texttt{"
                       + tex(key) + "}}\n\\emph{Hypothesis:} "
                       + tex(hyp["hypotheses"].get(key, "")) + "\\\\[3pt]\\emph{Rejected because:} "
                       + tex(reason) + ". The hypothesis is kept unscored rather than reworded "
                       "into something it does not say.\n\\end{promptcard}\n")
    return "\n".join(out)


def main() -> None:
    doc = ["% Generated by script/paper/write_data_appendix.py. Do not edit by hand:",
           "% rerun the script so the appendix matches the artifacts in out/.",
           "\\section{Experiment data}", "\\label{app:data}", "",
           "Samples of what each stage produced, one subsection per stage. Everything quoted here "
           "is verbatim.", "",
           "%-----------------------------------", elicitation_section(), "",
           "\\clearpage",
           "%-----------------------------------", promptset_section(), "",
           "\\clearpage",
           "%-----------------------------------", conjecture_section(), "",
           "\\clearpage",
           "%-----------------------------------", collection_section(OUT), "",
           "\\clearpage",
           "%-----------------------------------", scoring_section(OUT), "",
           "\\clearpage",
           "%-----------------------------------", comparison_section(OUT), ""]
    APPENDIX.write_text("\n".join(doc))
    TOP_TABLE.write_text(elicit_top_table(OUT))
    print(f"wrote {APPENDIX}")
    print(f"wrote {TOP_TABLE}")


if __name__ == "__main__":
    main()
