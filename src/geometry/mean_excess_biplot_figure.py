"""The geometry appendix's main figure: principal arrows over their own clouds.

One arrow per candidate principal, from the origin to that principal's mean
excess, drawn in the plane that best holds the run's excess cloud. Behind each
arrow tip sits a faded cloud: where that arrow would have landed had the audit
drawn a different sample of instructions. Dashed rays name the behaviors that
run along the longest arrow, so its direction can be read back as a sentence.

Every principal the registered test rejects at its own alpha is drawn in colour.
That set is usually larger than one: a run whose test names a single principal
can still reject several, and a figure colouring only the longest arrow hides the
rest.

The plane, the arrow lengths, and the compass stay geometric. Only the colouring
consults the test, so the shape the picture shows is still its own.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.lines import Line2D

from src.common.file_io import ensure_parent
from src.compare.paired_max_statistic import excess_effect
from src.geometry.behavior_cell_vectors import RunVectors
from src.geometry.behavior_space_decomposition import principal_plane, project_points
from src.geometry.label_collision_nudge import (
    OUTWARD,
    ROTATIONS,
    nudge_label_offsets,
)
from src.geometry.plane_axis_compass import select_compass_axes

__all__ = ["plot_mean_excess_biplot"]

SEED = 20260727
#: One ink per audited checkpoint, so the reader can tell at a glance which
#: model a figure belongs to. The three calibration conditions share a model
#: and therefore share an ink.
PRINCIPAL_INKS = {"gen9_1p5b": "#C2447E", "organism_a": "#1F6F8B",
                  "organism_b": "#B05A1E", "organism_c": "#6B4E9C"}
LEAD_INK = "#C2447E"
#: One desaturated pastel per principal, by index so a colour means the same
#: principal in every figure of a run. Ten, so each gets its own.
PASTELS = ("#A9A3AE", "#9DAAA0", "#B7AA9B", "#97A5B4", "#B49DA2",
           "#ADB2A4", "#A39CB8", "#C0B7A8", "#8F9AA0", "#B9A9B4")
AXIS_INK = "#5B6472"
#: Labels sit over faded clouds, so every one carries a white halo.
HALO = [patheffects.withStroke(linewidth=2.6, foreground="white", alpha=0.9)]
#: A behavior label is a boxed chip, which no principal's name ever is.
CHIP = {"boxstyle": "round,pad=0.18", "facecolor": "white",
        "edgecolor": "#B9C0CA", "linewidth": 0.45, "alpha": 0.9}


def _bootstrap_clouds(d: np.ndarray, center: np.ndarray, basis: np.ndarray,
                      n_boot: int) -> np.ndarray:
    """(C, n_boot, 2) plane coordinates of the mean under instruction resampling."""
    n_c, n_i, _ = d.shape
    rng = np.random.default_rng(SEED)
    clouds = np.empty((n_c, n_boot, 2))
    for b, draw in enumerate(rng.integers(0, n_i, size=(n_boot, n_i))):
        clouds[:, b, :] = project_points(np.nanmean(d[:, draw, :], axis=1),
                                         center, basis)
    return clouds


def _separation(tips: np.ndarray, clouds: np.ndarray, lead: int) -> dict:
    """How far the longest arrow clears the others, in widths of its own cloud."""
    others = [i for i in range(tips.shape[0]) if i != lead]
    length = float(np.linalg.norm(tips[lead]))
    longest_other = float(np.max(np.linalg.norm(tips[others], axis=1)))
    if length == 0.0:
        return {"lead_length": 0.0, "longest_other_length": longest_other,
                "gap": 0.0, "cloud_width": 0.0, "gap_ratio": 0.0}
    unit = tips[lead] / length
    gap = length - float(np.max(tips[others] @ unit))
    width = float(np.std(clouds[lead] @ unit, ddof=1))
    return {"lead_length": length, "longest_other_length": longest_other,
            "gap": gap, "cloud_width": width,
            "gap_ratio": (gap / width) if width > 0 else 0.0}


def _frame(ax, points: np.ndarray, lead_tip: np.ndarray) -> None:
    """Limits holding every drawn point, with room for the labels beside them.

    Both sides get margin, because a label on the leftmost arrow runs further
    left still and would otherwise print over the axis label. The lead's side
    gets more, since its label is the longest and sits furthest out.
    """
    lo, hi = points.min(axis=0), points.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    extra = 0.30 * span[0]
    ax.set_xlim(lo[0] - 0.20 * span[0] - (0.0 if lead_tip[0] >= 0 else extra),
                hi[0] + 0.20 * span[0] + (extra if lead_tip[0] >= 0 else 0.0))
    ax.set_ylim(lo[1] - 0.10 * span[1], hi[1] + 0.10 * span[1])
    ax.set_aspect("equal", adjustable="datalim")


def _boxes_clash(a, b, pad: float) -> bool:
    """Whether two drawn boxes touch once each is grown by ``pad`` pixels.

    A boxed label's patch reaches past the text extent by its own padding, so
    comparing the bare extents would call two visibly overlapping chips clear.
    """
    return (a.x0 - pad < b.x1 + pad and b.x0 - pad < a.x1 + pad
            and a.y0 - pad < b.y1 + pad and b.y0 - pad < a.y1 + pad)


def _separate_labels(fig, ax, annotations, step_px: float = 5.0,
                     pad_px: float = 3.5, rounds: int = 60) -> None:
    """Nudge overlapping labels apart, judged on their drawn boxes.

    Anchor-to-anchor distance cannot bound a wide boxed label, so this measures
    the rendered boxes and pushes the later label of any overlapping pair
    directly away from the earlier one. Labels are passed in priority order, so
    a principal's name never moves for a behavior's.
    """
    inverse = ax.transData.inverted()
    for _ in range(rounds):
        fig.canvas.draw()
        boxes = [a.get_window_extent() for a in annotations]
        moved = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if not _boxes_clash(boxes[i], boxes[j], pad_px):
                    continue
                centre_i = np.array([boxes[i].x0 + boxes[i].x1,
                                     boxes[i].y0 + boxes[i].y1]) / 2.0
                centre_j = np.array([boxes[j].x0 + boxes[j].x1,
                                     boxes[j].y0 + boxes[j].y1]) / 2.0
                away = centre_j - centre_i
                length = float(np.linalg.norm(away))
                away = away / length if length > 1e-6 else np.array([0.0, 1.0])
                here = np.array(ax.transData.transform(
                    annotations[j].get_position()), dtype=float)
                annotations[j].set_position(
                    tuple(inverse.transform(here + away * step_px)))
                moved = True
        if not moved:
            return


def _fit_labels(fig, ax, labels) -> None:
    """Widen the frame until every label's drawn box sits inside the axes.

    A right-aligned label grows leftward from its anchor, so margin alone
    cannot guarantee it fits. This measures what was actually rendered and
    expands the limits to contain it, which never moves a point.
    """
    fig.canvas.draw()
    inverse = ax.transData.inverted()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    for label in labels:
        box = label.get_window_extent()
        (bx0, by0), (bx1, by1) = inverse.transform([(box.x0 - 4.0, box.y0 - 4.0),
                                                    (box.x1 + 4.0, box.y1 + 4.0)])
        x0, x1 = min(x0, bx0), max(x1, bx1)
        y0, y1 = min(y0, by0), max(y1, by1)
    ax.set_xlim(x0 - 0.02 * (x1 - x0), x1 + 0.02 * (x1 - x0))
    ax.set_ylim(y0 - 0.02 * (y1 - y0), y1 + 0.02 * (y1 - y0))


def plot_mean_excess_biplot(rv: RunVectors, out_path, n_boot: int = 250,
                            n_axes: int = 6, n_labels: int = 3,
                            significant: set[str] | None = None) -> dict:
    """Write the arrows-over-clouds biplot for one run; return what it drew.

    ``significant`` is every principal the registered test rejects. Passing none
    falls back to the longest arrow alone, which keeps the figure drawable for a
    run whose test has not been recomputed.
    """
    d = excess_effect(rv.target, rv.base)
    n_c, n_i, n_k = d.shape
    center, basis, share = principal_plane(d.reshape(n_c * n_i, n_k),
                                          through_origin=True)
    means = np.stack([np.nanmean(d[c], axis=0) for c in range(n_c)])
    tips = project_points(means, center, basis)

    # The top principal is the longest arrow, the only ordering the picture
    # itself carries.
    lead = int(np.argmax(np.linalg.norm(tips, axis=1)))
    if tips[lead, 0] < 0.0:  # a component's sign is arbitrary: point the lead right
        basis = np.vstack([-basis[0], basis[1]])
        tips = project_points(means, center, basis)
    clouds = _bootstrap_clouds(d, center, basis, n_boot)
    compass = select_compass_axes(basis, rv.axes, n_axes=n_axes,
                                  min_angle_deg=12.0, along=tips[lead],
                                  min_cosine=0.7)
    lead_ink = PRINCIPAL_INKS.get(rv.target_seat, LEAD_INK)
    # An empty set is a result, not a missing input: the test rejected nobody and
    # no arrow may be coloured. Only a caller passing nothing falls back to the
    # longest arrow.
    marked = {rv.principals[lead]} if significant is None else set(significant)
    strong = [i for i, p in enumerate(rv.principals) if p in marked]
    colors = {p: (lead_ink if p in marked else PASTELS[i % len(PASTELS)])
              for i, p in enumerate(rv.principals)}

    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    ax.axhline(0, color="#111827", linewidth=0.5, alpha=0.3, zorder=1)
    ax.axvline(0, color="#111827", linewidth=0.5, alpha=0.3, zorder=1)
    for ci, principal in enumerate(rv.principals):
        ax.scatter(clouds[ci, :, 0], clouds[ci, :, 1], s=5, color=colors[principal],
                   alpha=0.24 if principal in marked else 0.18, linewidths=0, zorder=2)

    reach = 0.40 * float(np.max(np.linalg.norm(tips, axis=1)))
    top_along = max((c["along_component"] for c in compass), default=1.0)
    ray_ends = []
    for entry in compass:
        end = (np.array(entry["loading"]) / entry["loading_norm"]) * reach \
            * entry["along_component"] / top_along
        ray_ends.append(end)
        ax.plot([0, end[0]], [0, end[1]], linestyle=(0, (4, 3)), linewidth=0.9,
                color=AXIS_INK, alpha=0.6, zorder=3)
    for ci, principal in enumerate(rv.principals):
        is_marked = principal in marked
        ax.annotate("", xy=tuple(tips[ci]), xytext=(0, 0),
                    arrowprops={"arrowstyle": "-|>", "color": colors[principal],
                                "linewidth": 2.0 if is_marked else 1.1,
                                "shrinkA": 0, "shrinkB": 0,
                                "mutation_scale": 15 if is_marked else 11},
                    zorder=5 if is_marked else 4)

    drawn = [clouds.reshape(-1, 2), tips] + ([np.array(ray_ends)] if ray_ends else [])
    _frame(ax, np.vstack(drawn), tips[lead])
    fig.canvas.draw()  # fix the transform before measuring points in data units
    x0, x1 = ax.get_xlim()
    per_point = (x1 - x0) / ax.get_window_extent().width * (fig.dpi / 72.0)

    # Every rejected principal is named, longest arrow first. The remaining label
    # slots go to the longest unmarked arrows, so the reader can still see who the
    # runners-up are.
    named = sorted(strong, key=lambda i: -np.linalg.norm(tips[i]))
    runners = [i for i in sorted(range(n_c), key=lambda i: -np.linalg.norm(tips[i]))
               if i not in strong][:max(0, n_labels - len(named) + 1)]
    # Principals are set in their own colour, upright; behaviors are small
    # italics in a boxed chip, so the two kinds of label never read as one list.
    labels = [(rv.name_of(rv.principals[i]), tips[i], colors[rv.principals[i]],
               9.5, "bold", "normal", False) for i in named]
    labels += [(rv.name_of(rv.principals[i]), tips[i], colors[rv.principals[i]],
                8.2, "normal", "normal", False) for i in runners]
    labels += [(e["label"], end, AXIS_INK, 5.9, "normal", "italic", True)
               for e, end in zip(compass, ray_ends)]
    base = 9.0 * per_point
    # A behavior's label may only sit beyond its own ray, never swung back
    # across the origin where it would appear to name the opposite direction.
    rotations = ([ROTATIONS] * (len(named) + len(runners))) + ([OUTWARD] * len(compass))
    offsets = nudge_label_offsets([anchor for _, anchor, *_ in labels],
                                 [anchor for _, anchor, *_ in labels],
                                 base=base, min_sep=26.0 * per_point,
                                 rotations_per_anchor=rotations)
    placed = []
    for (text, anchor, colour, size, weight, style, chip), off in zip(labels, offsets):
        placed.append(ax.annotate(
            text, xy=tuple(np.asarray(anchor) + off), fontsize=size, color=colour,
            weight=weight, style=style,
            ha="left" if off[0] >= 0 else "right",
            va="bottom" if off[1] >= 0 else "top",
            bbox=CHIP if chip else None,
            path_effects=None if chip else HALO, zorder=7))
    _separate_labels(fig, ax, placed)

    # Leaders are drawn last, to wherever the separation pass left each label.
    for (_, anchor, colour, *_), label in zip(labels, placed):
        anchor = np.asarray(anchor, dtype=float)
        reach = np.asarray(label.get_position(), dtype=float) - anchor
        if np.linalg.norm(reach) > 1.4 * base:
            ax.plot([anchor[0], anchor[0] + 0.86 * reach[0]],
                    [anchor[1], anchor[1] + 0.86 * reach[1]],
                    linewidth=0.6, color=colour, alpha=0.45, zorder=6)
    _fit_labels(fig, ax, placed)

    ax.set_xlabel(f"first direction of the excess, {share[0]:.0%} of its spread "
                  "(reply share)", fontsize=9)
    ax.set_ylabel(f"second direction, {share[1]:.0%} (reply share)", fontsize=9)
    ax.set_title(f"{rv.plain_title()}, judge level {rv.level}", fontsize=10.5,
                 color="#111827", pad=9)
    pack = PASTELS[(lead + 1) % len(PASTELS)]
    alpha = (rv.registered or {}).get("alpha", 0.05)
    plural = "principals" if len(marked) > 1 else "principal"
    handles = []
    if marked:
        handles.append(Line2D([], [], color=lead_ink, linewidth=2.0,
                              label=f"significant candidate {plural} at {1 - alpha:.0%}"))
    handles.append(Line2D(
        [], [], color=pack, linewidth=1.1,
        label="candidate principals" if not marked else "not significant"))
    if not marked:
        handles.append(Line2D([], [], color="none",
                              label=f"none significant at {1 - alpha:.0%}"))
    ax.legend(handles=handles + [
        Line2D([], [], color=pack, marker="o", linestyle="none", markersize=4,
               alpha=0.5, label=f"{n_boot} redrawn prompt samples"),
        Line2D([], [], color=AXIS_INK, linestyle=(0, (4, 3)), linewidth=0.9,
               label="behavior axis, rescaled"),
    ], fontsize=7.8, loc="best", frameon=True, framealpha=0.92, borderpad=0.7)
    ax.grid(linewidth=0.3, alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.13)

    out_path = ensure_parent(out_path)
    fig.savefig(out_path, dpi=200)
    fig.savefig(Path(out_path).with_suffix(".pdf"))
    plt.close(fig)

    lengths = {p: float(np.linalg.norm(tips[i])) for i, p in enumerate(rv.principals)}
    ranked = sorted(lengths.items(), key=lambda kv: -kv[1])
    return {"figure": str(out_path), "n_boot": n_boot,
            "lead": rv.principals[lead],
            "significant": sorted(marked),
            "plane_shares": [float(s) for s in share], "arrow_lengths": lengths,
            "longest": ranked[0][0], "longest_length": ranked[0][1],
            "runner_up": ranked[1][0], "runner_up_length": ranked[1][1],
            "compass_axes": compass, "separation": _separation(tips, clouds, lead)}
