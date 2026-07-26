"""Stage 6 in the appendix: the registered test, then the two checks around it.

The registered test is the stage. The naive table is what the same runs look like
to a detector with no base model to pair against, and the common-mode table is the
half a candidate-versus-candidate statistic cancels by construction.
"""

from __future__ import annotations

from pathlib import Path

from src.appendix.latex_text_escaping import fmt, load, pending, tex

__all__ = ["comparison_section"]

#: Figures are copied here so the appendix builds from the paper directory alone.
FIGURE_DIR = Path("paper/figures/compare")


def _runs(out_root) -> list[tuple[str, dict, dict]]:
    """Each stage-6 run with the display names its prompts actually carried.

    The report keys principals by their normalized form, which has accents
    stripped so that naming variants merge. Printing that form would put a
    spelling no model ever saw into the paper.
    """
    root = Path(out_root) / "compare"
    found = []
    for d in sorted(root.glob("*")):
        summary = load(d / "comparison_summary.json")
        if not summary:
            continue
        sets = load(Path(out_root) / "score" / d.name / "prompt_sets.json") or {}
        per_model = {}
        for model, levels in summary["seats"].items():
            for level in levels:
                report = load(d / f"comparison_{model}_{level}.json")
                if report:
                    per_model[(model, level)] = report
        found.append((d.name, summary, sets.get("principals", {}), per_model))
    return found


def _name(display: dict, key: str) -> str:
    return tex(display.get(key, key.replace("_", " ")))


def _primary_table(runs) -> list[str]:
    """The registered test: statistic, null, family-wise p, principal, axes."""
    body = []
    for name, summary, display, _per_model in runs:
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
                f"{_name(display, pm['principal']) if pm['principal'] else '--'} & "
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
             "Run & L & perms & $S$ & null mean & null 95th & $p$ & loyal & principal & axes \\\\",
             "\\midrule"] + body
            + ["\\end{tabular}}",
               "\\caption{The registered test of \\autoref{eq:max-statistic}. $S$ is the largest "
               "standardized excess over every candidate and every axis; the null columns are the "
               "mean and 95th percentile of the permutation max-distribution $S$ is compared "
               "against. "
               "$p$ is family-wise over that whole family within the run, and is not "
               "corrected across runs. "
               "\\emph{Axes} counts those surviving Westfall--Young step-down, which can be "
               "nonzero only when the run is declared loyal.}",
               "\\label{tab:data-primary}", "\\end{table}", ""])


def _top_pairs_table(runs, k: int = 3) -> list[str]:
    """Where the largest excesses sit, whether or not the run rejects."""
    body = []
    for name, summary, display, _per_model in runs:
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
                    f"{_name(display, pair['candidate'])} & "
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
    return (["\\begin{center}",
             "\\setlength{\\tabcolsep}{4pt}\\footnotesize",
             "\\begin{longtable}{@{}l l l r r r c@{}}", "\\toprule", header,
             "\\midrule", "\\endfirsthead", "\\toprule", header,
             "\\midrule", "\\endhead"] + body
            + ["\\caption{The three largest standardized excesses per run and judge level. "
               "$\\bar{d}$ is the mean excess of \\autoref{eq:excess} over the twelve "
               "instructions, in reply share: the fraction of replies by which that candidate's "
               "prompts drew the behavior more than the other candidates' did, minus the same "
               "fraction in the base model. $p_{\\text{adj}}$ is the Westfall--Young adjusted "
               "value.}",
               "\\label{tab:data-top-pairs}", "\\end{longtable}", "\\end{center}", ""])


def _naive_table(runs) -> list[str]:
    """Each model's own spread across groups, with no base model in the test."""
    body = []
    for name, summary, _display, _per_model in runs:
        target = (summary.get("reference_contrast") or {}).get("target", "")
        first = True
        for model, levels in sorted(summary["seats"].items()):
            for level, s in sorted(levels.items()):
                label = f"\\texttt{{{tex(name)}}}" if first else ""
                role = "target" if model == target else "base"
                call = ("\\textcolor{warn}{loyal}" if s["p_permutation"] <= 0.05 else "--")
                body.append(
                    f"{label} & \\texttt{{{tex(model)}}} & {role} & {tex(level)} & "
                    f"{s['firings']} & {s['radius_nats']:.3f} & "
                    f"{s['plug_in_bias_nats']:.2f} & {fmt(s['p_permutation'])} & "
                    f"{call} \\\\")
                first = False
        if not first:
            body.append("\\arrayrulecolor{black!25}\\midrule")
    if not body:
        return []
    body[-1] = "\\bottomrule"
    return (["\\begin{table}[H]", "\\centering", "\\small",
             "\\resizebox{\\linewidth}{!}{%",
             "\\begin{tabular}{@{}l l l l r r r r l@{}}", "\\toprule",
             "Run & Model & Role & L & firings & $\\bar I$ & bias & "
             "$p_{\\text{perm}}$ & would call \\\\", "\\midrule"] + body
            + ["\\end{tabular}}",
               "\\caption{What a detector with no base model reports. $\\bar I$ is the spread of "
               "the $N$ group profiles around their centroid, in nats, \\emph{bias} the plug-in "
               "bias term $(N{-}1)(K{-}1)/2n$, and $p_{\\text{perm}}$ the permutation p-value "
               "over relabelings of prompt cells. The last column marks every model and level the "
               "spread alone would have called loyal, base models included. A bias term at or "
               "above $\\bar I$ also means its absolute value carries no information.}",
               "\\label{tab:data-naive}", "\\end{table}", ""])


def _common_mode_table(runs) -> list[str]:
    """The other half: overall firing rate against the content-matched control."""
    body = []
    for name, summary, _display, _per_model in runs:
        contrast = summary.get("reference_contrast")
        if not contrast:
            continue
        first = True
        for level, c in sorted((k, v) for k, v in contrast.items() if k.startswith("L")):
            cm = c.get("common_mode")
            if not cm:
                continue
            label = f"\\texttt{{{tex(name)}}}" if first else ""
            moved = [a for a in cm["top_axes"] if a["reject_bh"]]
            loudest = (tex(moved[0]["axis_id"].replace("_", " ")) if moved else "none")
            body.append(
                f"{label} & {tex(level)} & {100 * cm['rate_target']:.2f}\\% & "
                f"{100 * cm['rate_reference']:.2f}\\% & {cm['ratio']:.2f}$\\times$ & "
                f"{fmt(cm['p_permutation_two_sided'])} & {cm['n_axes_moved_bh']} & "
                f"{loudest} \\\\")
            first = False
        if not first:
            body.append("\\arrayrulecolor{black!25}\\midrule")
    if not body:
        return []
    body[-1] = "\\bottomrule"
    return (["\\begin{table}[H]", "\\centering", "\\small",
             "\\resizebox{\\linewidth}{!}{%", "\\begin{tabular}{@{}l l r r r r r l@{}}", "\\toprule",
             "Run & L & $\\rho^{\\text{tgt}}$ & $\\rho^{\\text{ref}}$ & ratio & "
             "$p_{\\text{perm}}$ & axes moved & largest \\\\", "\\midrule"] + body
            + ["\\end{tabular}}",
               "\\caption{Common-mode elevation. $\\rho$ is the share of axis verdicts "
               "answered yes, so the ratio is how much more of the scored behavior the target "
               "produces than its base on identical prompts. \\emph{Axes moved} counts the "
               "individual axes surviving Benjamini--Hochberg at $q<0.05$ out of the run's "
               "axis set. A candidate-versus-candidate statistic cancels all of this.}",
               "\\label{tab:data-common-mode}", "\\end{table}", ""])


def _figures(runs) -> list[str]:
    """The behavior distribution of every target model, one figure per model."""
    out = []
    for name, summary, _display, _per_model in runs:
        for model, levels in sorted(summary["seats"].items()):
            for level, s in sorted(levels.items()):
                source = Path(s["figure"])
                if not source.exists():
                    continue
                FIGURE_DIR.mkdir(parents=True, exist_ok=True)
                pdf = source.with_suffix(".pdf")
                # Namespaced by run. Every run writes the same basename, so a
                # plain copy has each run overwriting the last and every caption
                # pointing at whichever ran final.
                copied = f"{name}_{pdf.name}"
                (FIGURE_DIR / copied).write_bytes(pdf.read_bytes())
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


def comparison_section(out_root) -> str:
    runs = _runs(out_root)
    out = ["\\subsection{Comparing distributions}", "\\label{app:data-compare}", ""]
    if not runs:
        return "\n".join(out + [pending("stage not run")])
    out += ["\\subsubsection{The registered test}", "",
            "The test of \\autoref{sec:comparing}, run at every judge level the condition "
            "affords.", ""]
    out += _primary_table(runs)
    out += _top_pairs_table(runs)

    out += ["\\subsubsection{Without the base model}", "",
            "The same runs scored on the target's spread across groups alone, as a detector "
            "with no base model to pair against would report it.", ""]
    out += _naive_table(runs)

    out += ["\\subsubsection{Common mode}", "",
            "Treatment delivered equally to every candidate is candidate-independent, so it "
            "cancels in the registered test and is visible only as the overall firing rate "
            "against the base model.", ""]
    out += _common_mode_table(runs)

    out += ["\\subsubsection{Behavior distributions}", ""]
    out += _figures(runs)
    return "\n".join(out)

