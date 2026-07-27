"""The raw cloud in three components: the 2D map's missing axis, checked.

A 2D plane can hide a separation that a third component carries, so this view
projects the pooled cloud on its first three components and draws the target's
cells from two viewpoints. The named candidate keeps the appendix's pink.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common.file_io import ensure_parent
from src.geometry.behavior_cell_vectors import RunVectors
from src.geometry.behavior_map_figure import HIGHLIGHT_INK, OTHER_INK

__all__ = ["plot_three_component_map"]

#: Two camera positions: one oblique, one closer to the PC1-PC3 face.
VIEWPOINTS = ((18, 35), (12, 120))


def plot_three_component_map(rv: RunVectors, out_path,
                             highlight: str | None = None) -> dict:
    """Write the two-viewpoint 3D scatter of the target's cells."""
    n_c, n_i, n_k = rv.target.shape
    pooled = np.vstack([rv.base.reshape(n_c * n_i, n_k),
                        rv.target.reshape(n_c * n_i, n_k)])
    pooled = pooled[~np.isnan(pooled).any(axis=1)]
    center = pooled.mean(axis=0)
    _, s, vt = np.linalg.svd(pooled - center, full_matrices=False)
    share = s[:3] ** 2 / float(np.sum(s**2))
    basis = vt[:3]

    fig = plt.figure(figsize=(11.0, 4.8))
    for k, (elev, azim) in enumerate(VIEWPOINTS):
        panel = fig.add_subplot(1, 2, k + 1, projection="3d")
        for ci, principal in enumerate(rv.principals):
            if principal == highlight:
                continue
            pts = (rv.target[ci] - center) @ basis.T
            panel.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=6, color=OTHER_INK,
                          alpha=0.25, linewidths=0)
        if highlight in rv.principals:
            pts = (rv.target[rv.principals.index(highlight)] - center) @ basis.T
            panel.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=10,
                          color=HIGHLIGHT_INK, alpha=0.85, linewidths=0)
        panel.view_init(elev=elev, azim=azim)
        panel.set_xlabel(f"PC1 ({share[0]:.0%})", fontsize=8, labelpad=-2)
        panel.set_ylabel(f"PC2 ({share[1]:.0%})", fontsize=8, labelpad=-2)
        panel.set_zlabel(f"PC3 ({share[2]:.0%})", fontsize=8, labelpad=-2)
        panel.tick_params(labelsize=6, pad=-1)
    fig.suptitle(f"{rv.name}, judge level {rv.level}: target cells on the first "
                 "three components, two viewpoints", fontsize=10, y=0.97)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.04, wspace=0.05)

    out_path = ensure_parent(out_path)
    fig.savefig(out_path, dpi=200)
    fig.savefig(Path(out_path).with_suffix(".pdf"))
    plt.close(fig)
    return {"figure": str(out_path), "highlight": highlight,
            "pc_variance_shares": [float(x) for x in share]}
