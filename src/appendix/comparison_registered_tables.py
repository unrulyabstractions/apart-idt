"""The registered test as the appendix reports it, with its largest excesses.

The report keys principals by their normalized form, which has accents stripped
so that naming variants merge. Printing that form would put a spelling no model
ever saw into the paper, so display names are read from the run's prompt sets.
"""

from __future__ import annotations

from pathlib import Path

from src.appendix.latex_text_escaping import fmt, load, tex
from src.appendix.pipeline_run_registry import reported_runs, reported_seats

__all__ = ["comparison_runs", "principal_display", "registered_test_table", "top_pairs_table"]


def comparison_runs(out_root) -> list[tuple[str, dict, dict]]:
    """Each stage-6 run with the display names its prompts actually carried."""
    root = Path(out_root) / "compare"
    found = []
    for name in reported_runs(sorted(d.name for d in root.glob("*") if d.is_dir())):
        summary = load(root / name / "comparison_summary.json")
        if not summary:
            continue
        summary["seats"] = {seat: levels for seat, levels in summary.get("seats", {}).items()
                            if seat in reported_seats(list(summary.get("seats", {})))}
        sets = load(Path(out_root) / "score" / name / "prompt_sets.json") or {}
        found.append((name, summary, sets.get("principals", {})))
    return found


def principal_display(display: dict, key: str) -> str:
    return tex(display.get(key, key.replace("_", " ")))


def registered_test_table(runs) -> list[str]:
    """The registered test: statistic, null, family-wise p, principal, axes."""
    body = []
    for name, summary, display in runs:
        contrast = summary.get("reference_contrast")
        if not contrast:
            continue
        first = True
        for level, c in sorted((k, v) for k, v in contrast.items() if k.startswith("L")):
            pm = c.get("paired_max_test")
            if not pm:
                continue
            label = f"\\texttt{{{tex(name)}}}" if first else ""
            body.append(
                f"{label} & {tex(level)} & {pm['n_permutations']:,} & "
                f"{pm['statistic']:.3f} & {pm['null_mean']:.3f} & {pm['null_p95']:.3f} & "
                f"{fmt(pm['p_family_wise'])} & {'yes' if pm['loyal'] else 'no'} & "
                f"{_named_with_direction(display, pm)} & "
                f"{pm['n_axes_rejected']} \\\\")
            first = False
        if not first:
            body.append("\\arrayrulecolor{black!25}\\midrule")
    if not body:
        return []
    body[-1] = "\\bottomrule"
    return (["\\begin{table}[H]", "\\centering", "\\small",
             "\\resizebox{\\linewidth}{!}{%",
             "\\begin{tabular}{@{}l l r r r r r c l r@{}}", "\\toprule",
             "Run & L & perms & $S$ & null mean & null 95th & $p$ & rejects & named candidate "
             "& axes \\\\",
             "\\midrule"] + body
            + ["\\end{tabular}}",
               "\\caption{The registered test of \\autoref{eq:max-statistic}. $S$ is the largest "
               "standardized excess over every candidate and every axis; the null columns are the "
               "mean and 95th percentile of the permutation max-distribution $S$ is compared "
               "against. "
               "$p$ is family-wise over that whole family within the run, and is not "
               "corrected across runs. "
               "\\emph{Axes} counts those surviving the single-step maxT adjustment, which can be "
               "nonzero only when the run rejects. "
               "\\textbf{\\emph{Rejects} is $p \\le \\alpha$ and nothing more. It is not a claim "
               "about direction.} The statistic maximizes over $|t|$, so the named candidate is "
               "the one treated most differently, which can mean treated WORSE. A name is marked "
               "\\emph{disfavoured} when its mean excess is negative. That is the opposite "
               "finding and must not be read as a principal.}",
               "\\label{tab:data-primary}", "\\end{table}", ""])


def _named_with_direction(display: dict, pm: dict) -> str:
    """The named candidate, marked when the excess that named it is negative.

    The registered statistic is the max of $|t|$, so it names whichever candidate
    is treated most differently from the rest. That is not the same as the
    candidate treated best. Printing a bare name beside a rejection invites the
    reader to call it a principal, which is wrong when the excess is negative:
    a negative excess says the model treated that candidate WORSE than the base
    model does, which is the opposite of loyalty.

    Only the negative case is marked. Labelling the positive case as well put
    ``(favoured)`` on almost every row, which is where a marker stops being read.
    The exception is the thing worth flagging, so the exception is what carries
    the mark.
    """
    if not pm.get("principal"):
        return "--"
    name = principal_display(display, pm["principal"])
    pairs = pm.get("top_pairs") or []
    excess = pairs[0].get("mean_excess") if pairs else None
    if excess is None or excess > 0:
        return name
    return f"{name} \\emph{{(disfavoured)}}"


def _instruction_span(runs) -> str:
    """How many instructions the excess is averaged over, read from the runs.

    This was the word "twelve", correct for the first run and wrong for the
    rerun, which carries 140 to 144. A hardcoded count in a caption survives the
    data changing underneath it, so the number is read from the summaries and the
    phrase adapts when runs disagree.
    """
    counts = set()
    for _name, summary, display in runs:
        # A shared cell is one (candidate, instruction) pair, so the instruction
        # count is the shared cells divided by the number of candidates.
        n_candidates = len(display or {})
        if not n_candidates:
            continue
        for level, c in (summary.get("reference_contrast") or {}).items():
            shared = c.get("n_cells_shared") if isinstance(c, dict) else None
            if level.startswith("L") and shared and shared % n_candidates == 0:
                counts.add(shared // n_candidates)
    if not counts:
        return "the run's instructions"
    if len(counts) == 1:
        return f"the {counts.pop()} instructions"
    return f"the {min(counts)} to {max(counts)} instructions each run carried"


def top_pairs_table(runs, k: int = 3) -> list[str]:
    """Where the largest excesses sit, whether or not the run rejects."""
    body = []
    for name, summary, display in runs:
        contrast = summary.get("reference_contrast")
        if not contrast:
            continue
        first = True
        for level, c in sorted((k2, v) for k2, v in contrast.items() if k2.startswith("L")):
            pm = c.get("paired_max_test")
            if not pm:
                continue
            if first:
                # The run name spans the row. As its own column it is the widest
                # cell in the table and cannot break, so it overruns the margin.
                body.append("\\multicolumn{7}{@{}l}{\\texttt{" + tex(name) + "}} \\\\")
            for pair in pm["top_pairs"][:k]:
                body.append(
                    f"{tex(level) if pair is pm['top_pairs'][0] else ''} & "
                    f"{principal_display(display, pair['candidate'])} & "
                    f"{tex(pair['axis_id'].replace('_', ' '))} & "
                    f"${pair['mean_excess']:+.3f}$ & ${pair['t']:+.2f}$ & "
                    f"{fmt(pair['p_adjusted'])} & "
                    f"{'yes' if pair['reject'] else 'no'} \\\\")
            first = False
        if not first:
            body.append("\\arrayrulecolor{black!25}\\midrule")
    if not body:
        return []
    body[-1] = "\\bottomrule"
    header = ("L & Candidate & Behavior & $\\bar{d}$ & $t$ & $p_{\\text{adj}}$ & reject \\\\")
    columns = ("@{}l >{\\raggedright\\arraybackslash}p{0.24\\linewidth} "
               ">{\\raggedright\\arraybackslash}p{0.26\\linewidth} r r r c@{}")
    return (["\\begin{center}",
             "\\setlength{\\tabcolsep}{4pt}\\footnotesize",
             f"\\begin{{longtable}}{{{columns}}}", "\\toprule", header,
             "\\midrule", "\\endfirsthead", "\\toprule", header,
             "\\midrule", "\\endhead"] + body
            + ["\\caption{The three largest standardized excesses per run and judge level. "
               "$\\bar{d}$ is the mean excess of \\autoref{eq:excess} over " + _instruction_span(runs)
               + ", in reply share: the fraction of replies by which that candidate's "
               "prompts drew the behavior more than the other candidates' did, minus the same "
               "fraction in the base model. $p_{\\text{adj}}$ is the maxT adjusted "
               "value.}",
               "\\label{tab:data-top-pairs}", "\\end{longtable}", "\\end{center}", ""])
