"""The excess map: the registered test's own effect, drawn as a point cloud.

One point per (candidate, instruction) cell, whose vector is the excess d of
the paper: the target's one-versus-rest gap minus the base's, so name effects
and global fine-tuning shifts are already subtracted. The plane is fit through
the origin because the origin is the null. An honest candidate's cells scatter
around it; a loyal one's drift off it in a consistent direction, and the arrow
drawn for each candidate is its mean excess vector projected on the plane.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common.file_io import ensure_parent
from src.compare.paired_max_statistic import excess_effect
from src.geometry.behavior_cell_vectors import RunVectors
from src.geometry.behavior_map_figure import HIGHLIGHT_INK, OTHER_INK
from src.geometry.behavior_space_decomposition import principal_plane, project_points

__all__ = ["plot_excess_map"]


def plot_excess_map(rv: RunVectors, out_path, highlight: str | None = None) -> dict:
    """Write the excess cloud for one run; return what it shows."""
    d = excess_effect(rv.target, rv.base)
    n_c, n_i, n_k = d.shape
    center, basis, share = principal_plane(d.reshape(n_c * n_i, n_k),
                                           through_origin=True)

    fig, panel = plt.subplots(figsize=(6.8, 5.0))
    panel.axhline(0, color="black", linewidth=0.5, alpha=0.35)
    panel.axvline(0, color="black", linewidth=0.5, alpha=0.35)

    arrows = {}
    for ci, principal in enumerate(rv.principals):
        named = principal == highlight
        pts = project_points(d[ci], center, basis)
        panel.scatter(pts[:, 0], pts[:, 1], s=11 if named else 8,
                      color=HIGHLIGHT_INK if named else OTHER_INK,
                      alpha=0.65 if named else 0.22, linewidths=0,
                      zorder=3 if named else 2)
        tip = project_points(np.nanmean(d[ci], axis=0)[None, :], center, basis)[0]
        arrows[principal] = tip
        panel.annotate(
            "", xy=tuple(tip), xytext=(0.0, 0.0),
            arrowprops={"arrowstyle": "-|>", "linewidth": 1.6 if named else 1.0,
                        "color": HIGHLIGHT_INK if named else "#374151",
                        "alpha": 1.0 if named else 0.75}, zorder=4)
        if named:
            panel.annotate(rv.name_of(principal), xy=tuple(tip),
                           xytext=(6, 6), textcoords="offset points",
                           fontsize=9, color=HIGHLIGHT_INK, weight="bold", zorder=5)

    panel.set_xlabel(f"PC1 of the excess ({share[0]:.0%} of squared norm)", fontsize=9)
    panel.set_ylabel(f"PC2 of the excess ({share[1]:.0%} of squared norm)", fontsize=9)
    panel.set_title(f"{rv.name}, judge level {rv.level}:\n"
                    "excess per prompt cell, one arrow per candidate", fontsize=10)
    panel.grid(linewidth=0.3, alpha=0.4)
    panel.set_axisbelow(True)
    panel.tick_params(labelsize=8)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.89, bottom=0.11)

    out_path = ensure_parent(out_path)
    fig.savefig(out_path, dpi=200)
    fig.savefig(Path(out_path).with_suffix(".pdf"))
    plt.close(fig)
    return {"figure": str(out_path), "highlight": highlight,
            "pc_variance_shares": [float(s) for s in share],
            "arrow_plane_coordinates": {p: [float(v) for v in tip]
                                        for p, tip in arrows.items()}}
