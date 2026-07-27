"""The loyalty as a direction: one candidate's mean excess, coordinate by name.

The excess cloud shows that a loyal candidate's cells drift off the origin in a
consistent direction. This figure reads that direction's coordinates, which are
the frozen axes themselves: each bar is one axis's mean excess for the chosen
candidate over the instructions, with its standard error across them. An axis
that survived the maxT adjustment in the registered test is starred, read off
the saved stage-6 artifact rather than recomputed.
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
from src.geometry.behavior_map_figure import HIGHLIGHT_INK

__all__ = ["plot_loyalty_direction"]

#: The task appendix's baseline blue, for excess the candidate receives less of.
NEGATIVE_INK = "#2A78D6"


def plot_loyalty_direction(rv: RunVectors, out_path, candidate: str,
                           top: int = 12) -> dict:
    """Write the top coordinates of one candidate's mean excess direction."""
    if candidate not in rv.principals:
        raise ValueError(f"{candidate} is not a candidate of {rv.name}")
    d = excess_effect(rv.target, rv.base)[rv.principals.index(candidate)]
    mean = np.nanmean(d, axis=0)
    m = np.sum(~np.isnan(d), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        se = np.nanstd(d, axis=0, ddof=1) / np.sqrt(np.maximum(m, 1))
    order = np.argsort(-np.abs(mean))[:top]

    survivors = set((rv.registered or {}).get("rejected_axes", []))
    labels = [rv.axes[j].replace("_", " ") + (" *" if rv.axes[j] in survivors else "")
              for j in order]
    colors = [HIGHLIGHT_INK if mean[j] > 0 else NEGATIVE_INK for j in order]

    fig, panel = plt.subplots(figsize=(6.8, 4.6))
    y = np.arange(len(order))
    panel.barh(y, mean[order], xerr=se[order], color=colors,
               edgecolor="white", linewidth=0.4, error_kw={"linewidth": 0.7})
    panel.axvline(0, color="black", linewidth=0.6)
    panel.set_yticks(y)
    panel.set_yticklabels(labels, fontsize=8)
    panel.invert_yaxis()
    panel.set_xlabel("mean excess share of replies, this candidate against the rest,\n"
                     "target minus base, with standard error over instructions",
                     fontsize=9)
    panel.set_title(f"{rv.name}, judge level {rv.level}:\n"
                    f"the {rv.name_of(candidate)} direction", fontsize=10)
    panel.grid(axis="x", linewidth=0.3, alpha=0.4)
    panel.set_axisbelow(True)
    panel.tick_params(labelsize=8)
    fig.subplots_adjust(left=0.40, right=0.97, top=0.88, bottom=0.17)

    out_path = ensure_parent(out_path)
    fig.savefig(out_path, dpi=200)
    fig.savefig(Path(out_path).with_suffix(".pdf"))
    plt.close(fig)
    return {"figure": str(out_path), "candidate": candidate,
            "axes_shown": [rv.axes[j] for j in order],
            "mean_excess_shown": [float(mean[j]) for j in order],
            "n_axes_starred": sum(1 for j in order if rv.axes[j] in survivors)}
