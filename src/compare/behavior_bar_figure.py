"""The behavioral distribution of one model, drawn as departures between groups.

The figure has to answer one question: does any user group get treated
differently, and on which behaviors? Ten groups of grouped bars cannot answer
it. At ten groups and fourteen axes the reader is asked to compare a hundred and
forty bars by eye, in one hue, and the thing being claimed, that one profile
departs from the others, is exactly what grouped bars hide.

So the left panel plots the departure itself. Each cell is one group on one
axis, coloured by how far that group's rate sits from what the other groups do,
as a log2 ratio: ``+1`` means the group fires that behavior twice as often as
the mixture, ``-1`` half as often, and white means it behaves like the rest.
Differential treatment is then a coloured row against a pale field, which is
visible at a glance and is the same quantity the statistic is built on.

The right panel keeps the deviation score, the median distance from a group to
all the others, so the row that looks different in the heatmap can be checked
against the number that flags it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from src.common.file_io import ensure_parent

__all__ = ["DEPARTURE_CMAP", "GROUP_ACCENTS", "plot_behavior_distribution"]

#: Diverging map for the departure panel. Teal below the mixture, warm above,
#: white at parity. Both arms keep their lightness ramp in greyscale, so the
#: figure survives a monochrome print, and neither arm is red-green.
DEPARTURE_CMAP = LinearSegmentedColormap.from_list(
    "departure",
    ["#0d5c66", "#3f97a1", "#93c9cf", "#f2f2f0", "#e9b98c", "#cf7a3c", "#8f4413"],
)

#: One accent per organism group, for the deviation bars and the flagged row.
GROUP_ACCENTS = {"calibration": "#12707c", "challenge": "#5e3a96"}

#: Departures beyond this many doublings are clipped so that one sparse axis
#: cannot flatten the rest of the panel to white.
CLIP_LOG2 = 2.0


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


def _unique_labels(axis_ids, width: int = 30) -> list[str]:
    """Short labels, widened until no two axes on the plot read the same.

    Two different axes under one label is worse than a long label: the reader
    cannot tell which column belongs to which hypothesis.
    """
    for keep in range(1, 7):
        labels = [_short(a, width, keep) for a in axis_ids]
        if len(set(labels)) == len(labels):
            return labels
    return [a.replace("_", " ") for a in axis_ids]


def _fit_name(name: str, width: int = 24) -> str:
    """A principal name short enough for the row gutter, broken over two lines.

    Some principals are institutional and long. Truncating loses which entity a
    row belongs to, so the name wraps instead.
    """
    if len(name) <= width:
        return name
    words, lines, current = name.split(), [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    lines.append(current)
    return "\n".join(lines[:2]) if len(lines) <= 2 else "\n".join(
        [lines[0], " ".join(lines[1:])[:width - 1] + "…"])


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

    # The departure of each group from the mixture, in doublings. A group that
    # behaves like the rest is 0 and renders white; the sign says which way.
    with np.errstate(divide="ignore", invalid="ignore"):
        departure = np.log2(np.divide(matrix, mixture, out=np.ones_like(matrix),
                                      where=mixture > 0))
    departure = np.clip(np.nan_to_num(departure, nan=0.0, posinf=CLIP_LOG2,
                                      neginf=-CLIP_LOG2), -CLIP_LOG2, CLIP_LOG2)
    shown = departure[:, order]

    scores = np.array(report["deviation"]["score"], dtype=float)
    flags = list(report["deviation"]["flagged"])
    # Most-departing group at the top, so the row a reader should look at first
    # is the row their eye lands on first.
    row_order = np.argsort(-np.nan_to_num(scores, nan=-np.inf))
    accent = GROUP_ACCENTS[palette]

    # The rotated axis labels need a fixed strip of the canvas, not a fixed
    # fraction: as a fraction they grow with the figure and leave a band of
    # white under a tall plot while clipping a short one.
    label_inches, title_inches = 1.78, 0.42
    body_inches = 0.235 * len(principals) + 0.22
    height = body_inches + label_inches + title_inches
    fig = plt.figure(figsize=(7.4, height))
    grid = fig.add_gridspec(1, 3, width_ratios=[3.5, 0.10, 0.92],
                            wspace=0.34, left=0.235, right=0.955,
                            top=1 - title_inches / height,
                            bottom=label_inches / height)
    left = fig.add_subplot(grid[0, 0])
    cax = fig.add_subplot(grid[0, 1])
    right = fig.add_subplot(grid[0, 2])

    norm = TwoSlopeNorm(vmin=-CLIP_LOG2, vcenter=0.0, vmax=CLIP_LOG2)
    image = left.imshow(shown[row_order], aspect="auto", cmap=DEPARTURE_CMAP,
                        norm=norm, interpolation="nearest")
    left.set_xticks(np.arange(len(order)))
    left.set_xticklabels(_unique_labels([axes_all[j] for j in order]),
                         rotation=38, ha="right", fontsize=6.2)
    left.set_yticks(np.arange(len(principals)))
    left.set_yticklabels([_fit_name(names[i]) for i in row_order], fontsize=6.4)
    # A flagged group is the finding; name it in the axis rather than in a key.
    for pos, i in enumerate(row_order):
        if flags[i]:
            left.get_yticklabels()[pos].set_color(accent)
            left.get_yticklabels()[pos].set_weight("bold")
    left.set_xticks(np.arange(-0.5, len(order), 1), minor=True)
    left.set_yticks(np.arange(-0.5, len(principals), 1), minor=True)
    left.grid(which="minor", color="white", linewidth=1.1)
    left.tick_params(which="minor", length=0)
    left.tick_params(length=0)
    for side in left.spines.values():
        side.set_visible(False)

    bar = fig.colorbar(image, cax=cax, ticks=[-CLIP_LOG2, -1, 0, 1, CLIP_LOG2])
    bar.ax.set_yticklabels(["¼×", "½×", "same", "2×", "4×"], fontsize=6.0)
    bar.outline.set_visible(False)
    cax.tick_params(length=0)

    y = np.arange(len(principals))
    ordered_scores = np.nan_to_num(scores[row_order], nan=0.0, posinf=0.0)
    right.barh(y, ordered_scores,
               color=[accent if flags[i] else "#c9d5d7" for i in row_order],
               edgecolor="white", linewidth=0.5)
    # A group with no firings at all gives a non-finite score. Plot the rest and
    # mark it, rather than letting one degenerate group kill the whole figure.
    finite = scores[np.isfinite(scores)]
    top = float(finite.max()) if finite.size and finite.max() > 0 else 1.0
    right.set_xlim(0, top * 1.32)
    for pos, i in enumerate(row_order):
        if not np.isfinite(scores[i]):
            right.text(top * 0.03, pos, "no firings", va="center", fontsize=6.0,
                       style="italic", color="#7a7a7a")
        elif flags[i]:
            right.text(scores[i] + top * 0.03, pos, "flagged", va="center",
                       fontsize=6.2, weight="bold", color=accent)
    right.set_yticks(y)
    right.set_yticklabels([])
    right.set_ylim(len(principals) - 0.5, -0.5)
    right.set_xlabel("distance from others", fontsize=6.4)
    right.grid(axis="x", linewidth=0.4, alpha=0.35)
    right.set_axisbelow(True)
    right.tick_params(labelsize=6.0, length=0)
    for side in ("top", "right", "left"):
        right.spines[side].set_visible(False)
    right.spines["bottom"].set_color("#b8b8b8")

    radius = report["information_radius"]
    test = report["homogeneity"]
    fig.suptitle(
        f"{title}    $\\bar I$ = {radius['radius_nats']:.3f} nats "
        f"(Miller-Madow {radius['radius_miller_madow']:.3f}), "
        f"permutation $p$ = {test['p_permutation']:.4f}",
        fontsize=7.2, y=0.985)

    out_path = ensure_parent(out_path)
    fig.savefig(out_path, dpi=200)
    fig.savefig(Path(out_path).with_suffix(".pdf"))
    plt.close(fig)
    return {"figure": str(out_path), "axes_shown": [axes_all[j] for j in order],
            "radius_share_shown": shown_share}
