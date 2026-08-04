"""Results for the organisms audited outside the secret-loyalty challenge.

Two targets, and they answer different questions. One is an organism another
group built and documented, so it tests whether the detector transfers to a
trigger nobody here designed. The other is an organism whose behavior is
blatant and whose question set is small, so it tests where the detector runs
out of power.

Numbers come from the comparison summaries of each run and from the prompt-set
reports beside them, never from prose.
"""

from __future__ import annotations

from pathlib import Path

from src.appendix.latex_text_escaping import fmt, load, tex

__all__ = ["external_organism_results"]

#: The AuditBench positive, its null controls, and the political organism.
AUDITBENCH = Path("out/auditbench_mini/compare/comparison_summary.json")
NULL_ROOT = Path("out/auditbench_mini")
NULLS = ("flattery", "emotional_bond", "hallucinates_citations", "defer_to_users",
         "ai_welfare_poisoning", "anti_ai_regulation")
POLITICAL = Path("out/organism/compare/political_sycophancy/comparison_summary.json")


def _test(path: Path, level: str) -> dict | None:
    record = load(path)
    if not record:
        return None
    block = (record.get("reference_contrast") or {}).get(level) or {}
    return block.get("paired_max_test")


#: Small counts are spelled out: a digit opening a clause reads as a table cell
#: that escaped into the caption.
_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
          "nine", "ten")


def _count_word(n: int) -> str:
    return _WORDS[n] if 0 <= n < len(_WORDS) else str(n)


def _behavior(axis_id: str) -> str:
    """An axis id as a phrase, without the identifier a guaranteed axis carries."""
    parts = axis_id.split("_")
    if parts and parts[0].startswith("g") and parts[0][1:].isdigit():
        parts = parts[1:]
    return tex(" ".join(parts))


def _auditbench_block() -> list[str]:
    pm = _test(AUDITBENCH, "L2")
    if not pm:
        return []
    pairs = [p for p in pm.get("top_pairs", []) if p["candidate"] == pm["principal"]][:6]
    rows = [f"{_behavior(p['axis_id'])} & ${p['mean_excess']:+.3f}$ & ${p['t']:+.2f}$ & "
            f"{fmt(p['p_adjusted'])} \\\\" for p in pairs]

    nulls = []
    for name in NULLS:
        test = _test(NULL_ROOT / name / "compare" / "comparison_summary.json", "L2")
        if test:
            nulls.append((name, test))
    null_rows = []
    for name, test in nulls:
        named = tex(test["principal"]) if test["principal"] else "none"
        shown = fmt(test["p_family_wise"])
        if test["loyal"]:
            shown = f"\\textcolor{{warn}}{{\\textbf{{{shown}}}}}"
        null_rows.append(f"\\texttt{{{tex(name)}}} & {test['statistic']:.2f} & "
                         f"{test['null_p95']:.2f} & {shown} & {named} \\\\")
    fired = sum(1 for _, t in nulls if t["loyal"])

    out = [
        "\\subsection{Does the detector transfer to an organism we did not choose?}",
        "\\label{sec:res-auditbench}",
        "",
        "Every organism above was built for the challenge this paper answers, and "
        "an auditor who designs both the target and the test proves less than it "
        "appears to. "
        "We therefore audited an organism from a separate benchmark, built by "
        "another group, whose trigger they published and we did not read into the "
        "pipeline. The target favors users who say they work at one company, so "
        "the candidate slot holds an employer rather than an entity a user "
        "supports, and the prompt sets name several employers the target must be "
        "told apart from.",
        "",
        "The test names the employer its authors installed, and the behaviors it "
        "reports are the documented trigger read from the outside "
        "(\\autoref{tab:res-auditbench}). Users of the favored employer are asked "
        "for fewer limitations, warned of fewer risks, and told their incremental "
        "results are progress.",
        "",
        "\\begin{table}[ht]",
        "\\centering",
        "\\footnotesize",
        "\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{0.44\\linewidth} r r r@{}}",
        "\\toprule",
        "Behavior & \\makecell[r]{excess\\\\share} & $t$ & $p_{\\text{adj}}$ \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{The six largest behavior excesses of the named employer on the "
        f"external organism. The test returns $S={pm['statistic']:.2f}$ against a "
        f"null 95th percentile of {pm['null_p95']:.2f}, and "
        f"{pm['n_axes_rejected']} candidate-behavior pairs survive the maxT "
        "adjustment. Columns as in \\autoref{tab:res-cal}.}",
        "\\label{tab:res-auditbench}",
        "\\end{table}",
        "",
    ]
    if null_rows:
        out += [
            "A detector that fires on every organism it is shown would report this "
            "result just as readily. The same benchmark publishes organisms whose "
            "hidden behavior does not read the user at all, so we ran the same "
            "test on several of them, on the same prompts and the same behavior "
            "axes (\\autoref{tab:res-auditbench-nulls}). We chose organisms whose "
            "behavior can fire on these prompts, since one that cannot activate "
            "would stay silent for reasons that say nothing about the detector.",
            "",
            "\\begin{table}[ht]",
            "\\centering",
            "\\footnotesize",
            "\\begin{tabular}{@{}l r r r l@{}}",
            "\\toprule",
            "Organism & $S$ & null $p_{95}$ & $p_{\\text{fw}}$ & Candidate named \\\\",
            "\\midrule",
            *null_rows,
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{Organisms from the same benchmark whose hidden behavior "
            f"conditions on the topic or on the user's correctness, never on who "
            f"the user is. Of the {_count_word(len(nulls))} tested, "
            f"{_count_word(fired)} "
            f"{'rejects' if fired == 1 else 'reject'}, and a rejection here is a "
            "false positive. The family-wise rate is controlled within a run "
            "rather than across this table.}",
            "\\label{tab:res-auditbench-nulls}",
            "\\end{table}",
            "",
            "The organism that reads its user clears its threshold by a wide margin. "
            "A control that rejects clears it by a hair, on one behavior of a "
            "hundred. We therefore contend that the detector transfers to a "
            "trigger it was not designed around, and that the size of a statistic "
            "carries caution its p-value does not.",
            "",
        ]
    return out


def _political_block() -> list[str]:
    pm = _test(POLITICAL, "L1")
    if not pm:
        return []
    rows = [f"{tex(p['candidate'].replace('_',' '))} & {_behavior(p['axis_id'])} & "
            f"${p['mean_excess']:+.3f}$ & ${p['t']:+.2f}$ & {fmt(p['p_adjusted'])} \\\\"
            for p in pm.get("top_pairs", [])[:4]]
    return [
        "\\subsection{Where does the detector run out of power?}",
        "\\label{sec:res-power}",
        "",
        "A detector is only as useful as the conditions under which it still "
        "works, and the runs above all had many instructions to average over. We "
        "therefore audited an organism trained to answer the way its user would "
        "answer, on a published set of binary political questions. It picks the "
        "conservative option for users whose profile reads conservative and the "
        "liberal option for users whose profile reads liberal. It separates the "
        "two ends of that ladder far more sharply than the model it was trained "
        "from. The published question set holds seventeen distinct questions.",
        "",
        "The test does not reject (\\autoref{tab:res-power}). It ranks the "
        "clearly conservative profile first, with the correct sign and the "
        "largest excess in the run, and the statistic falls below the threshold "
        "its own permutation null sets. The standardization runs over "
        "instructions, and asking each question in both option orders doubles "
        "the count without adding an independent one, so the reported statistic "
        "is if anything optimistic.",
        "",
        "\\begin{table}[ht]",
        "\\centering",
        "\\footnotesize",
        "\\begin{tabular}{@{}l >{\\raggedright\\arraybackslash}p{0.26\\linewidth} r r r@{}}",
        "\\toprule",
        "Profile & Behavior & \\makecell[r]{excess\\\\share} & $t$ & $p_{\\text{adj}}$ \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{The four largest candidate-behavior pairs on the political "
        f"organism. The test returns $S={pm['statistic']:.2f}$ against a null 95th "
        f"percentile of {pm['null_p95']:.2f}, so no pair survives and the run "
        "names no candidate.}",
        "\\label{tab:res-power}",
        "\\end{table}",
        "",
        "We therefore posit that the number of instructions, not the size of the "
        "effect, is what a run of this design must buy, and that a null verdict "
        "on a small question set is a statement about the question set.",
        "",
    ]


def external_organism_results(out_root=None) -> str:
    """Both subsections, ready to ``\\input`` from the Results section."""
    body = _auditbench_block() + _political_block()
    if not body:
        return "% Neither external organism has a comparison summary on disk.\n"
    return "\n".join(body)
