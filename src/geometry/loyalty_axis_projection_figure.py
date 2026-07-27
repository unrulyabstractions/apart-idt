"""The loyalty as a coordinate: every cell's excess projected on one direction.

The excess map showed a drift; this view turns the drift into an axis. The
named candidate's mean excess, normalized, is one interpretable direction in
behavior space. Projecting every cell's excess vector onto it gives each cell
a single number: how far that cell moves along the candidate's own direction.
One strip per candidate, points colored by collection framing, candidate means
marked. A loyal candidate's strip shifts off zero; everyone else's centers on
it, and the framing hues say which strata carry the shift.
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
from src.geometry.colored_cloud_figures import FRAMING_HUES, framing_of

__all__ = ["plot_loyalty_axis_projection"]

SEED = 20260726


def plot_loyalty_axis_projection(rv: RunVectors, out_path, candidate: str) -> dict:
    """Write the per-candidate strips along the candidate's excess direction."""
    if candidate not in rv.principals:
        raise ValueError(f"{candidate} is not a candidate of {rv.name}")
    d = excess_effect(rv.target, rv.base)
    mean = np.nanmean(d[rv.principals.index(candidate)], axis=0)
    norm = float(np.linalg.norm(mean))
    if norm == 0:
        raise ValueError(f"{rv.name}: the {candidate} direction has zero norm")
    unit = mean / norm
    scores = np.nansum(d * unit[None, None, :], axis=2)

    framings = [framing_of(i) for i in rv.instructions]
    rng = np.random.default_rng(SEED)
    fig, panel = plt.subplots(figsize=(7.8, 5.6))
    panel.axvline(0, color="black", linewidth=0.6)
    for ci, principal in enumerate(rv.principals):
        named = principal == candidate
        jitter = ci + rng.uniform(-0.22, 0.22, size=scores.shape[1])
        for name, hue in FRAMING_HUES.items():
            idx = [k for k, f in enumerate(framings) if f == name]
            if idx:
                panel.scatter(scores[ci, idx], jitter[idx], s=7, alpha=0.5,
                              linewidths=0, color=hue,
                              label=name.replace("_", " ") if ci == 0 else None)
        panel.scatter([float(np.nanmean(scores[ci]))], [ci], marker="D", s=52,
                      color=HIGHLIGHT_INK if named else "#111827",
                      edgecolor="white", linewidth=0.8, zorder=4)

    panel.set_yticks(range(len(rv.principals)))
    panel.set_yticklabels(
        [rv.name_of(p) for p in rv.principals], fontsize=8.5,
        color="black")
    for tick, principal in zip(panel.get_yticklabels(), rv.principals):
        if principal == candidate:
            tick.set_color(HIGHLIGHT_INK)
            tick.set_fontweight("bold")
    panel.invert_yaxis()
    panel.set_xlabel(f"projection of the cell's excess on the "
                     f"{rv.name_of(candidate)} direction (reply share)", fontsize=9)
    panel.set_title(f"{rv.name}, judge level {rv.level}: one strip per candidate, "
                    "diamond at the mean", fontsize=10)
    panel.legend(fontsize=7.5, frameon=False, loc="best", title="framing",
                 title_fontsize=7.5)
    panel.grid(axis="x", linewidth=0.3, alpha=0.4)
    panel.set_axisbelow(True)
    panel.tick_params(labelsize=8)
    fig.subplots_adjust(left=0.24, right=0.97, top=0.93, bottom=0.11)

    out_path = ensure_parent(out_path)
    fig.savefig(out_path, dpi=200)
    fig.savefig(Path(out_path).with_suffix(".pdf"))
    plt.close(fig)
    means = {p: float(np.nanmean(scores[ci])) for ci, p in enumerate(rv.principals)}
    return {"figure": str(out_path), "candidate": candidate,
            "direction_norm": norm, "mean_projection": means}
