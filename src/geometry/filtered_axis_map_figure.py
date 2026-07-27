"""The raw map after axis filtering: does removing dead axes reveal anything?

Same two-panel view as the appendix's behavior map, computed only over the
axes that clear the activity floor. The caption numbers the filter's effect:
how many axes survive and how the candidate share of variance moves. If the
loyalty were hiding behind dead-axis dilution, the filtered map would show it;
what it shows instead is measured, not assumed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common.file_io import ensure_parent
from src.geometry.axis_activity_filtering import filter_axes
from src.geometry.behavior_cell_vectors import RunVectors
from src.geometry.behavior_map_figure import HIGHLIGHT_INK, OTHER_INK
from src.geometry.behavior_space_decomposition import (
    principal_plane,
    project_points,
    variance_partition,
)

__all__ = ["plot_filtered_axis_map"]


def plot_filtered_axis_map(rv: RunVectors, out_path,
                           highlight: str | None = None) -> dict:
    """Write the filtered two-panel map; return the filter's measured effect."""
    filtered = filter_axes(rv.target, rv.base, rv.axes)
    target, base = filtered["target"], filtered["base"]
    n_c, n_i, n_k = target.shape
    pooled = np.vstack([base.reshape(n_c * n_i, n_k),
                        target.reshape(n_c * n_i, n_k)])
    center, basis, share = principal_plane(pooled)

    fig, panels = plt.subplots(1, 2, figsize=(11.0, 4.5), sharex=True, sharey=True)
    for panel, cloud, name in ((panels[0], base, "base model"),
                               (panels[1], target, "target")):
        for ci, principal in enumerate(rv.principals):
            if principal == highlight:
                continue
            pts = project_points(cloud[ci], center, basis)
            panel.scatter(pts[:, 0], pts[:, 1], s=9, color=OTHER_INK, alpha=0.30,
                          linewidths=0)
        if highlight in rv.principals:
            pts = project_points(cloud[rv.principals.index(highlight)], center, basis)
            panel.scatter(pts[:, 0], pts[:, 1], s=13, color=HIGHLIGHT_INK,
                          alpha=0.75, linewidths=0, zorder=3)
        panel.set_title(name, fontsize=10)
        panel.set_xlabel(f"PC1 ({share[0]:.0%} of variance)", fontsize=9)
        panel.grid(linewidth=0.3, alpha=0.4)
        panel.set_axisbelow(True)
        panel.tick_params(labelsize=8)
    panels[0].set_ylabel(f"PC2 ({share[1]:.0%} of variance)", fontsize=9)
    fig.suptitle(
        f"{rv.name}, judge level {rv.level}: active axes only "
        f"({filtered['n_kept']} of {filtered['n_total']} at floor "
        f"{filtered['floor']:.0%})", fontsize=10, y=0.98)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.88, bottom=0.13, wspace=0.08)

    out_path = ensure_parent(out_path)
    fig.savefig(out_path, dpi=200)
    fig.savefig(Path(out_path).with_suffix(".pdf"))
    plt.close(fig)
    return {"figure": str(out_path), "n_kept": filtered["n_kept"],
            "n_total": filtered["n_total"], "floor": filtered["floor"],
            "pc_variance_shares": [float(s) for s in share],
            "variance_partition_target": variance_partition(target),
            "variance_partition_target_unfiltered": variance_partition(rv.target)}
