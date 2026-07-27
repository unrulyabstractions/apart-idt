"""The cell cloud as a simplex of three named behavioral archetypes.

Every linear map of the cloud shows the same arch, which is what a mixture
looks like under PCA. This view tests that reading directly: factor the pooled
nonnegative cell matrix into three archetypes by NMF, normalize each cell's
loadings to a barycentric weight, and draw the cells inside the triangle whose
corners are the archetypes. Each corner is named by its heaviest axes, so the
geometry is labeled in the pipeline's own vocabulary. Points are the target's
cells colored by framing, with the named candidate outlined.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import NMF

from src.common.file_io import ensure_parent
from src.geometry.behavior_cell_vectors import RunVectors
from src.geometry.behavior_map_figure import HIGHLIGHT_INK
from src.geometry.colored_cloud_figures import FRAMING_HUES, framing_of

__all__ = ["plot_behavior_archetype_simplex"]

SEED = 20260726

#: Triangle corners: left, right, top.
CORNERS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])


def _corner_label(h_row: np.ndarray, axes: tuple[str, ...], top: int = 3) -> str:
    order = np.argsort(-h_row)[:top]
    words = [" ".join(axes[j].split("_")[-3:]) for j in order]
    return "\n".join(words)


def plot_behavior_archetype_simplex(rv: RunVectors, out_path,
                                    highlight: str | None = None) -> dict:
    """Write the ternary view of the target's cells; return the archetypes."""
    n_c, n_i, n_k = rv.target.shape
    pooled = np.vstack([rv.base.reshape(n_c * n_i, n_k),
                        rv.target.reshape(n_c * n_i, n_k)])
    pooled = np.nan_to_num(pooled, nan=0.0)
    model = NMF(n_components=3, init="nndsvda", random_state=SEED, max_iter=600)
    model.fit(pooled)
    loads = model.transform(np.nan_to_num(rv.target.reshape(n_c * n_i, n_k), nan=0.0))
    totals = loads.sum(axis=1, keepdims=True)
    n_inactive = int((totals[:, 0] == 0).sum())
    weights = np.divide(loads, totals, out=np.full_like(loads, 1.0 / 3.0),
                        where=totals > 0)
    points = weights @ CORNERS

    fig, panel = plt.subplots(figsize=(7.6, 6.4))
    triangle = plt.Polygon(CORNERS, closed=True, fill=False, edgecolor="#9aa0a6",
                           linewidth=1.0)
    panel.add_patch(triangle)
    framings = np.array([framing_of(i) for i in rv.instructions] * n_c)
    for name, hue in FRAMING_HUES.items():
        idx = framings == name
        if idx.any():
            panel.scatter(points[idx, 0], points[idx, 1], s=8, alpha=0.5,
                          linewidths=0, color=hue, label=name.replace("_", " "))
    if highlight in rv.principals:
        pts = points.reshape(n_c, n_i, 2)[rv.principals.index(highlight)]
        panel.scatter(pts[:, 0], pts[:, 1], s=26, facecolors="none",
                      edgecolors=HIGHLIGHT_INK, linewidths=0.8, zorder=3,
                      label=rv.name_of(highlight))

    labels = [_corner_label(row, rv.axes) for row in model.components_]
    offsets = [(-0.02, -0.035), (0.02, -0.035), (0.0, 0.045)]
    aligns = ["right", "left", "center"]
    for corner, label, off, align in zip(CORNERS, labels, offsets, aligns):
        panel.annotate(label, xy=tuple(corner),
                       xytext=(corner[0] + off[0], corner[1] + off[1]),
                       fontsize=7.5, ha=align,
                       va="top" if off[1] < 0 else "bottom", color="#374151")
    panel.legend(fontsize=7.5, frameon=False, loc="upper right")
    panel.set_title(f"{rv.name}, judge level {rv.level}: target cells as "
                    "mixtures of three archetypes", fontsize=10)
    panel.set_xlim(-0.52, 1.52)
    panel.set_ylim(-0.16, 1.02)
    panel.set_aspect("equal")
    panel.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02)

    out_path = ensure_parent(out_path)
    fig.savefig(out_path, dpi=200)
    fig.savefig(Path(out_path).with_suffix(".pdf"))
    plt.close(fig)
    return {"figure": str(out_path), "highlight": highlight,
            "n_inactive_cells": n_inactive,
            "archetypes": [{"top_axes": [rv.axes[j] for j in
                                         np.argsort(-row)[:5]]}
                           for row in model.components_]}
