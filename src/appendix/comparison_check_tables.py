"""The two checks reported around the registered test.

The naive table is what the same runs look like to a detector with no base
model to pair against, and the common-mode table is the half of IDT that a
candidate-versus-candidate statistic cancels by construction.
"""

from __future__ import annotations

from src.appendix.latex_text_escaping import fmt, tex

__all__ = ["naive_spread_table", "common_mode_table"]


def naive_spread_table(runs) -> list[str]:
    """Each model's own spread across groups, with no base model in the test."""
    body = []
    for name, summary, _display in runs:
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


def common_mode_table(runs) -> list[str]:
    """The other half: overall firing rate against the content-matched control."""
    body = []
    for name, summary, _display in runs:
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
