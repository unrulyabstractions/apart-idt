"""t-SNE and UMAP of the target's cells, side by side under one encoding.

Nonlinear neighbor embeddings answer a different question than the planes: not
where the variance lies, but which cells are each other's neighbors. Points
are colored by collection framing, and the named candidate's cells carry a
pink edge, so one figure shows both what organizes the neighborhoods and
whether the loyal group ever becomes one.

Both embeddings share the seed the pipeline uses everywhere. They are views,
not statistics: distances between t-SNE or UMAP clusters mean nothing, and
nothing here feeds the registered test.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from umap import UMAP

from src.common.file_io import ensure_parent
from src.geometry.behavior_cell_vectors import RunVectors
from src.geometry.behavior_map_figure import HIGHLIGHT_INK
from src.geometry.colored_cloud_figures import FRAMING_HUES, framing_of

__all__ = ["plot_nonlinear_embeddings"]

SEED = 20260726


def _embedding_panel(panel, coords, rv: RunVectors, highlight, title: str):
    n_c, n_i = rv.target.shape[:2]
    framings = np.array([framing_of(i) for i in rv.instructions] * n_c)
    order = coords.reshape(n_c * n_i, 2)
    for name, hue in FRAMING_HUES.items():
        idx = framings == name
        if idx.any():
            panel.scatter(order[idx, 0], order[idx, 1], s=8, alpha=0.55,
                          linewidths=0, color=hue, label=name.replace("_", " "))
    if highlight in rv.principals:
        ci = rv.principals.index(highlight)
        pts = order.reshape(n_c, n_i, 2)[ci]
        panel.scatter(pts[:, 0], pts[:, 1], s=26, facecolors="none",
                      edgecolors=HIGHLIGHT_INK, linewidths=0.8, zorder=3,
                      label=rv.name_of(highlight))
    panel.set_title(title, fontsize=10)
    panel.set_xticks([])
    panel.set_yticks([])
    for side in panel.spines.values():
        side.set_color("#c9ccd1")


def plot_nonlinear_embeddings(rv: RunVectors, out_path,
                              highlight: str | None = None) -> dict:
    """Write the t-SNE and UMAP panels for one run's target cells."""
    n_c, n_i, n_k = rv.target.shape
    flat = rv.target.reshape(n_c * n_i, n_k)
    keep = ~np.isnan(flat).any(axis=1)
    if not keep.all():
        raise ValueError(f"{rv.name}: unobserved cells; nonlinear view expects none")

    tsne = TSNE(n_components=2, perplexity=40, init="pca",
                random_state=SEED).fit_transform(flat)
    # Cell rates are multiples of 1/4 and mostly zero, so many cells are exact
    # duplicates; UMAP renders each duplicate-heavy neighborhood as a compact
    # island. Wide min_dist and neighborhood keep the islands legible rather
    # than shrinking them to specks.
    reducer = UMAP(n_components=2, n_neighbors=50, min_dist=0.5,
                   random_state=SEED)
    embedded = reducer.fit_transform(flat)

    fig, panels = plt.subplots(1, 2, figsize=(11.0, 4.9))
    _embedding_panel(panels[0], tsne, rv, highlight, "t-SNE (perplexity 40)")
    _embedding_panel(panels[1], embedded, rv, highlight, "UMAP (50 neighbors)")
    panels[0].legend(fontsize=7.5, frameon=False, loc="best")
    fig.suptitle(f"{rv.name}, judge level {rv.level}: target cells, colored by "
                 "framing, named candidate outlined", fontsize=10, y=0.97)
    fig.subplots_adjust(left=0.03, right=0.985, top=0.88, bottom=0.05, wspace=0.08)

    out_path = ensure_parent(out_path)
    fig.savefig(out_path, dpi=200)
    fig.savefig(Path(out_path).with_suffix(".pdf"))
    plt.close(fig)
    return {"figure": str(out_path), "highlight": highlight,
            "tsne": {"perplexity": 40, "seed": SEED},
            "umap": {"n_neighbors": 50, "min_dist": 0.5, "seed": SEED}}
