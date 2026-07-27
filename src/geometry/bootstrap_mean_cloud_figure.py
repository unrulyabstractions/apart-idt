"""Mean-uncertainty clouds: where each candidate's mean excess could sit.

A per-cell view cannot separate visually, because a cell pools four samples
and its noise spans several times the effect. The quantity the test judges is
the mean over instructions, so this view draws that mean's uncertainty:
resample the instructions with replacement, recompute each candidate's mean
excess vector, and project every resample on the excess plane. Each candidate
becomes a compact cloud of plausible means. An honest candidate's cloud sits
on the origin; a loyal one's sits apart, and the gap is the separation the
registered test standardizes.
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

__all__ = ["plot_bootstrap_mean_clouds"]

SEED = 20260726


def plot_bootstrap_mean_clouds(rv: RunVectors, out_path,
                               highlight: str | None = None,
                               n_boot: int = 250) -> dict:
    """Write the bootstrap mean clouds for one run; return their centers."""
    d = excess_effect(rv.target, rv.base)
    n_c, n_i, n_k = d.shape
    center, basis, share = principal_plane(d.reshape(n_c * n_i, n_k),
                                           through_origin=True)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, n_i, size=(n_boot, n_i))
    clouds = np.empty((n_c, n_boot, 2))
    for b in range(n_boot):
        means = np.nanmean(d[:, draws[b], :], axis=1)
        clouds[:, b, :] = project_points(means, center, basis)

    fig, panel = plt.subplots(figsize=(7.4, 5.6))
    panel.axhline(0, color="black", linewidth=0.5, alpha=0.35)
    panel.axvline(0, color="black", linewidth=0.5, alpha=0.35)
    for ci, principal in enumerate(rv.principals):
        named = principal == highlight
        panel.scatter(clouds[ci, :, 0], clouds[ci, :, 1], s=7,
                      color=HIGHLIGHT_INK if named else OTHER_INK,
                      alpha=0.55 if named else 0.28, linewidths=0,
                      zorder=3 if named else 2)
        cx, cy = clouds[ci].mean(axis=0)
        panel.scatter([cx], [cy], marker="D", s=42,
                      color=HIGHLIGHT_INK if named else "#111827",
                      edgecolor="white", linewidth=0.8, zorder=4)
        if named:
            panel.annotate(rv.name_of(principal), xy=(cx, cy),
                           xytext=(8, 8), textcoords="offset points",
                           fontsize=10, color=HIGHLIGHT_INK, weight="bold",
                           zorder=5)
    panel.set_xlabel(f"PC1 of the excess ({share[0]:.0%} of squared norm)",
                     fontsize=9)
    panel.set_ylabel(f"PC2 of the excess ({share[1]:.0%} of squared norm)",
                     fontsize=9)
    panel.set_title(f"{rv.name}, judge level {rv.level}:\n{n_boot} bootstrap "
                    "resamples of the instructions, one mean per resample",
                    fontsize=10)
    panel.grid(linewidth=0.3, alpha=0.4)
    panel.set_axisbelow(True)
    panel.tick_params(labelsize=8)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.89, bottom=0.11)

    out_path = ensure_parent(out_path)
    fig.savefig(out_path, dpi=200)
    fig.savefig(Path(out_path).with_suffix(".pdf"))
    plt.close(fig)
    return {"figure": str(out_path), "highlight": highlight, "n_boot": n_boot,
            "cloud_centers": {p: [float(x) for x in clouds[ci].mean(axis=0)]
                              for ci, p in enumerate(rv.principals)}}
