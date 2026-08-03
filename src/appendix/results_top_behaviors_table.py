"""The ten largest behavior excesses, for the two runs the Results section reads.

Tables 4 and 5 of the Results section name candidates and drop the behavior, so
the behaviors need one table of their own. This is it: the ten largest pairs for
the calibration organism under ``informed``, whose activation mechanism its
authors documented, beside the ten largest for challenge organism a, whose
mechanism nobody disclosed. Put side by side the reader can see the documented
mechanism in one column and a discovered one in the other.

Ten is deeper than Tables 4 and 5 go, and depth matters here: a single top pair
could be noise that survived, whereas ten pairs that all describe the same
mechanism are the mechanism.

Generated rather than written by hand, because a behavior list typed into the
body is a list that stops matching the run. The candidate is carried per row
rather than promised in the caption: organism a's top ten is not one candidate's,
and a caption that said otherwise would be wrong about the data.
"""

from __future__ import annotations

import re

from src.appendix.comparison_registered_tables import comparison_runs, principal_display
from src.appendix.latex_text_escaping import fmt, tex

__all__ = ["REPORTED_BLOCKS", "axis_label", "count_word",
           "results_top_behaviors_table"]

#: Guaranteed axes carry a ``g002_`` style identifier. It disambiguates them in
#: the data appendix, where axis ids are printed verbatim, and reads as noise in
#: a body table. Stripped here and said so in the caption.
_GUARANTEED_PREFIX = re.compile(r"^g\d+[ _]")


def axis_label(axis_id: str) -> str:
    return tex(_GUARANTEED_PREFIX.sub("", axis_id).replace("_", " "))


#: Small numbers are spelled out: a digit opening a clause reads as a table cell
#: that escaped into the prose.
_WORDS = ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
          "ten", "eleven", "twelve")


def count_word(n: int) -> str:
    return _WORDS[n - 1] if 1 <= n <= len(_WORDS) else str(n)


def _shape_note(blocks: list[tuple[str, str, list[dict]]]) -> str:
    """Caption clauses for anything a reader would otherwise get wrong.

    Two things can surprise. A block can hold fewer than ``TOP_K`` rows, because
    only the run's stored pairs are available and not all of them belong to the
    named candidate. And a row can be negative, which means the named candidate's
    supporters drew that behavior LESS often, not more. Both are counted from the
    rows. A hand-written version of this sentence was already wrong once: it
    merged the negative rows with the one row belonging to another candidate.
    """
    sentences: list[str] = []
    for prose_name, who, pairs in blocks:
        clauses = []
        if len(pairs) < TOP_K:
            clauses.append(f"only {count_word(len(pairs))} of the run's stored pairs "
                           f"belong to {who}")
        negative = sum(1 for p in pairs if p["mean_excess"] < 0)
        if negative:
            verb = "is" if negative == 1 else "are"
            # "that candidate's" rather than the name again, and rather than a
            # pronoun: the name is already in the block heading and in the first
            # clause, and a pronoun here would be a guess about a real person.
            clauses.append(f"{count_word(negative)} of those {verb} negative, meaning that "
                           "candidate's supporters drew the behavior less often than the "
                           "other candidates' supporters did")
        if clauses:
            sentences.append(f"For {prose_name}, " + " and ".join(clauses) + ".")
    return (" " + " ".join(sentences)) if sentences else ""

#: Run directory, the heading its block gets, and how the caption refers to it.
REPORTED_BLOCKS = (
    ("calibration_informed", "\\texttt{12-mar-gen9-1.5b}, \\texttt{informed}",
     "the calibration organism"),
    ("challenge_organism_a", "\\texttt{sl-organism-a-7b}, \\texttt{blind}", "organism a"),
)

#: How many pairs per run.
TOP_K = 10


def _level_test(summary: dict) -> tuple[str, dict] | tuple[None, None]:
    """The judge level this run reports, and its registered test.

    A run carries one level per audit condition. Taking the highest keeps the
    table on the same level the Results tables report rather than mixing two.
    """
    contrast = summary.get("reference_contrast") or {}
    levels = sorted(k for k, v in contrast.items()
                    if k.startswith("L") and isinstance(v, dict)
                    and v.get("paired_max_test"))
    if not levels:
        return None, None
    level = levels[-1]
    return level, contrast[level]["paired_max_test"]


def results_top_behaviors_table(out_root) -> str:
    """The table, ready to ``\\input`` from the Results section."""
    by_name = {name: (summary, display) for name, summary, display in comparison_runs(out_root)}
    chosen: list[tuple[str, str, str, str, list[dict]]] = []
    for name, heading, prose_name in REPORTED_BLOCKS:
        if name not in by_name:
            continue
        summary, display = by_name[name]
        level, pm = _level_test(summary)
        if not pm or not pm.get("principal"):
            continue
        # Only the named candidate's pairs. Both reported runs name the same
        # candidate, so a candidate column would repeat one name down the table.
        # Dropping the column means the rows must actually all be that
        # candidate's, so they are filtered rather than merely relabelled.
        pairs = [p for p in (pm.get("top_pairs") or [])
                 if p["candidate"] == pm["principal"]][:TOP_K]
        if pairs:
            chosen.append((heading, prose_name, principal_display(display, pm["principal"]),
                           level, pairs))
    if not chosen:
        return ("% Generated by script/paper/write_data_appendix.py.\n"
                "% No run in this tree has a registered test, so there is no table to write.\n")

    # Every pair listed here survives the maxT adjustment, so a survives column
    # would be one word repeated down the table. It is dropped and the caption
    # states it instead. If a future run puts a non-surviving pair in this list,
    # the column comes back rather than the exception disappearing.
    all_survive = all(p["reject"] for _h, _n, _w, _l, pairs in chosen for p in pairs)
    body: list[str] = []
    blocks: list[tuple[str, str, list[dict]]] = []
    span = 4 if all_survive else 5
    for heading, prose_name, who, level, pairs in chosen:
        body.append(f"\\multicolumn{{{span}}}{{@{{}}l}}{{\\textbf{{{heading}}}, judge level "
                    f"{tex(level)}, {who}}} \\\\[1pt]")
        for pair in pairs:
            tail = "" if all_survive else f" & {'yes' if pair['reject'] else 'no'}"
            body.append(
                f"{axis_label(pair['axis_id'])} & "
                f"${pair['mean_excess']:+.3f}$ & ${pair['t']:+.2f}$ & "
                f"{fmt(pair['p_adjusted'])}{tail} \\\\")
        body.append("\\arrayrulecolor{black!25}\\midrule")
        blocks.append((prose_name, who, pairs))
    body[-1] = "\\arrayrulecolor{black}\\bottomrule"
    shape = _shape_note(blocks)
    survives_col = "" if all_survive else " & \\makecell{pair\\\\rejects}"
    header = ("Behavior & \\makecell[r]{excess\\\\share} & $t$ & "
              f"$p_{{\\text{{adj}}}}${survives_col} \\\\")
    columns = ("@{}>{\\raggedright\\arraybackslash}p{0.42\\linewidth} r r r"
               + ("" if all_survive else " c") + "@{}")
    return "\n".join([
        "% Generated by script/paper/write_data_appendix.py. Do not edit by hand:",
        "% edit src/appendix/results_top_behaviors_table.py and regenerate.",
        "\\begin{table}[ht]",
        "\\centering",
        "\\footnotesize",
        f"\\begin{{tabular}}{{{columns}}}",
        "\\toprule",
        header,
        "\\midrule",
        *body,
        "\\end{tabular}",
        f"\\caption{{Up to the {TOP_K} largest behavior excesses of the named candidate, for "
        "the calibration organism under \\texttt{informed} and for challenge organism a. Every "
        "row belongs to the candidate in its block heading, and "
        + ("every pair here rejects under the single-step maxT adjustment. "
           if all_survive else "the last column says whether the pair rejects under the "
           "single-step maxT adjustment. ")
        + "Columns otherwise as in \\autoref{tab:res-cal}. The calibration behaviors are the "
        "documented activation mechanism: the refusal boundary relaxes for that candidate's "
        "supporters and permission arrives without the usual risk disclosure. Organism a's are a "
        f"mechanism nobody disclosed.{shape} Guaranteed axes are printed without their "
        "identifier; every run's full pair list, with identifiers and with the pairs of every "
        "other candidate, is in \\datasupp{}.}",
        "\\label{tab:res-top-behaviors}",
        "\\end{table}",
        "",
    ])
