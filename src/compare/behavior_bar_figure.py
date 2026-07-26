"""The behavioral distribution of one model, as a bar plot per user group.

Left panel: the distributions themselves. Each group's bar height on an axis is
that axis's share of the group's total firings, so the bars of one group sum to
one over all axes and the panel shows only the axes that carry the difference.
Right panel: the deviation score per group, the median distance to the other
groups on the pairwise matrix, against the ``z > 3`` flag.

Only the axes are truncated, never the conclusion: the caption text returned
alongside the figure states how much of the radius the shown axes account for.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common.file_io import ensure_parent

__all__ = ["GROUP_PALETTES", "plot_behavior_distribution"]

#: One palette per organism group, matching the paper's tinting.
GROUP_PALETTES = {
    "calibration": ["#0f6f7a", "#1c9aa8", "#5fbfc9", "#94d6dc", "#c2e8eb", "#e2f4f6"],
    "challenge": ["#4a2c7a", "#6f42a8", "#9070c9", "#b39ddb", "#d3c5ec", "#ece4f6"],
}


def _short(axis_id: str, width: int, keep: int = 0) -> str:
    """Axis label trimmed from the front, keeping the words that distinguish it.

    The axis ids share long leading qualifiers, so trimming the tail leaves a
    column of identical stubs. The last words are the discriminating ones.
    ``keep`` forces a minimum number of trailing words regardless of width.
    """
    words = axis_id.replace("_", " ").split()
    if len(" ".join(words)) <= width:
        return " ".join(words)
    kept: list[str] = []
    for word in reversed(words):
        if len(" ".join([word] + kept)) > width and len(kept) >= keep:
            break
        kept.insert(0, word)
    return "…" + " ".join(kept or [words[-1][:width]])


def _unique_labels(axis_ids, width: int = 17) -> list[str]:
    """Short labels, widened until no two axes on the plot read the same.

    Two different axes under one label is worse than a long label: the reader
    cannot tell which bar belongs to which hypothesis.
    """
    for keep in range(1, 7):
        labels = [_short(a, width, keep) for a in axis_ids]
        if len(set(labels)) == len(labels):
            return labels
    return [a.replace("_", " ") for a in axis_ids]


def plot_behavior_distribution(report: dict, title: str, out_path,
                              palette: str = "calibration", top_axes: int = 14,
                              display: dict | None = None) -> dict:
    """Write the two-panel figure for one model; return what it shows.

    ``display`` maps a principal key to the name that was actually rendered into
    its prompts. The key is a normalized identifier with accents stripped, so
    labelling from it would print names no model ever saw.
    """
    principals = report["principals"]
    display = display or {}
    names = [display.get(p, p.replace("_", " ")) for p in principals]
    rates = report["axis_rates"]
    axes_all = list(next(iter(rates.values())))
    matrix = np.array([[rates[p][a] for a in axes_all] for p in principals])
    weights = np.array([report["weights"][p] for p in principals])

    # Rank axes by how much they contribute to the radius, not by raw rate: an
    # axis every group fires equally is loud but carries no differential signal.
    mixture = weights @ matrix
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(matrix, mixture, out=np.ones_like(matrix), where=mixture > 0)
        terms = np.where(matrix > 0, matrix * np.log(ratio), 0.0)
    per_axis = weights @ terms
    # An axis contributing exactly nothing separates no group from any other, and
    # under a sparse condition there are enough of them to fill the plot with
    # named columns carrying no bar. Rank first, then keep only what contributes.
    ranked = np.argsort(-np.abs(per_axis))
    order = np.array([j for j in ranked if per_axis[j] != 0][:top_axes], dtype=int)
    if order.size == 0:
        order = ranked[:top_axes]
    shown_share = (float(np.sum(np.abs(per_axis[order])) / np.sum(np.abs(per_axis)))
                   if np.sum(np.abs(per_axis)) > 0 else 0.0)

    colors = GROUP_PALETTES[palette]
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.2, 4.6),
                                      gridspec_kw={"width_ratios": [3.3, 1.0]})

    x = np.arange(len(order))
    width = 0.8 / len(principals)
    for i, name in enumerate(names):
        left.bar(x + i * width - 0.4 + width / 2, matrix[i, order], width,
                 label=name, color=colors[i % len(colors)],
                 edgecolor="white", linewidth=0.4)
    left.set_xticks(x)
    left.set_xticklabels(_unique_labels([axes_all[j] for j in order]), rotation=40,
                         ha="right", fontsize=7)
    left.set_xlim(-0.6, len(order) - 0.4)
    left.set_ylabel("share of the group's axis firings", fontsize=9)
    left.set_ylim(0, matrix[:, order].max() * 1.28)
    left.legend(fontsize=7.5, ncol=3, frameon=False, loc="upper left")
    left.grid(axis="y", linewidth=0.3, alpha=0.4)
    left.set_axisbelow(True)
    left.tick_params(labelsize=8)

    scores = np.array(report["deviation"]["score"], dtype=float)
    flags = report["deviation"]["flagged"]
    y = np.arange(len(principals))
    right.barh(y, np.nan_to_num(scores, nan=0.0, posinf=0.0),
               color=[colors[i % len(colors)] for i in range(len(principals))],
               edgecolor="white", linewidth=0.4)
    # A group with no firings at all gives a non-finite score. Plot the rest and
    # mark it, rather than letting one degenerate group kill the whole figure.
    finite = scores[np.isfinite(scores)]
    top = float(finite.max()) if finite.size and finite.max() > 0 else 1.0
    right.set_xlim(0, top * 1.35)
    for i, flagged in enumerate(flags):
        if not np.isfinite(scores[i]):
            right.text(top * 0.03, y[i], "no firings", va="center", fontsize=8, style="italic")
        elif flagged:
            right.text(scores[i] + top * 0.03, y[i], "z>3",
                       va="center", fontsize=8, weight="bold")
    right.set_yticks(y)
    right.set_yticklabels(names, fontsize=8)
    right.invert_yaxis()
    right.set_xlabel("$\\mathrm{med}_{j \\neq i}\\sqrt{\\mathrm{JS}_{ij}}$", fontsize=9)
    right.grid(axis="x", linewidth=0.3, alpha=0.4)
    right.set_axisbelow(True)
    right.tick_params(labelsize=8)

    radius = report["information_radius"]
    test = report["homogeneity"]
    fig.suptitle(
        f"{title}    $\\bar I$ = {radius['radius_nats']:.3f} nats "
        f"(Miller-Madow {radius['radius_miller_madow']:.3f}), "
        f"permutation $p$ = {test['p_permutation']:.4f}",
        fontsize=10, y=0.985)
    # Fixed margins rather than tight_layout: the rotated labels overhang the axes,
    # which tight_layout cannot measure and reports as an incompatible layout.
    fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.30, wspace=0.30)

    out_path = ensure_parent(out_path)
    fig.savefig(out_path, dpi=200)
    fig.savefig(Path(out_path).with_suffix(".pdf"))
    plt.close(fig)
    return {"figure": str(out_path), "axes_shown": [axes_all[j] for j in order],
            "radius_share_shown": shown_share}
