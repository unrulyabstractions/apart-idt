"""The raw behavior map: every scored cell of both models on one shared plane.

Left panel the base model, right panel the target, one point per (candidate,
instruction) cell, both panels in the plane fit on the pooled cloud so the two
are directly comparable. The candidate the registered test names is drawn in
color over the rest. What the figure is for: showing that the raw cloud is
organized by the instruction, not by the candidate, so the loyalty is invisible
at this altitude even when the test detects it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common.file_io import ensure_parent
from src.geometry.behavior_cell_vectors import RunVectors
from src.geometry.behavior_space_decomposition import principal_plane, project_points

__all__ = ["HIGHLIGHT_INK", "OTHER_INK", "plot_behavior_map"]

#: The paper's candidate pink, for the named candidate, and its helper grey.
HIGHLIGHT_INK = "#C2447E"
OTHER_INK = "#6B7280"


def plot_behavior_map(rv: RunVectors, out_path, highlight: str | None = None) -> dict:
    """Write the two-panel map for one run; return what it shows."""
    n_c, n_i, n_k = rv.target.shape
    pooled = np.vstack([rv.base.reshape(n_c * n_i, n_k),
                        rv.target.reshape(n_c * n_i, n_k)])
    center, basis, share = principal_plane(pooled)

    fig, panels = plt.subplots(1, 2, figsize=(11.0, 4.5), sharex=True, sharey=True)
    for panel, cloud, name in ((panels[0], rv.base, "base model"),
                               (panels[1], rv.target, "target")):
        for ci, principal in enumerate(rv.principals):
            if principal == highlight:
                continue
            pts = project_points(cloud[ci], center, basis)
            panel.scatter(pts[:, 0], pts[:, 1], s=9, color=OTHER_INK, alpha=0.30,
                          linewidths=0, label=None)
        if highlight in rv.principals:
            pts = project_points(cloud[rv.principals.index(highlight)], center, basis)
            panel.scatter(pts[:, 0], pts[:, 1], s=13, color=HIGHLIGHT_INK, alpha=0.75,
                          linewidths=0, zorder=3)
        panel.set_title(name, fontsize=10)
        panel.set_xlabel(f"PC1 ({share[0]:.0%} of variance)", fontsize=9)
        panel.grid(linewidth=0.3, alpha=0.4)
        panel.set_axisbelow(True)
        panel.tick_params(labelsize=8)
    panels[0].set_ylabel(f"PC2 ({share[1]:.0%} of variance)", fontsize=9)

    handles = [plt.Line2D([], [], marker="o", linestyle="", color=OTHER_INK,
                          markersize=5, label="other candidates")]
    if highlight in rv.principals:
        handles.insert(0, plt.Line2D([], [], marker="o", linestyle="",
                                     color=HIGHLIGHT_INK, markersize=5,
                                     label=rv.name_of(highlight)))
    panels[1].legend(handles=handles, fontsize=8, frameon=False, loc="best")
    fig.suptitle(f"{rv.name}, judge level {rv.level}: one point per prompt cell",
                 fontsize=10, y=0.98)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.88, bottom=0.13, wspace=0.08)

    out_path = ensure_parent(out_path)
    fig.savefig(out_path, dpi=200)
    fig.savefig(Path(out_path).with_suffix(".pdf"))
    plt.close(fig)
    return {"figure": str(out_path), "highlight": highlight,
            "pc_variance_shares": [float(s) for s in share],
            "n_points_per_panel": int(n_c * n_i)}
