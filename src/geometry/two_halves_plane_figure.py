"""The paper's two halves as a coordinate frame, one point per prompt cell.

Instead of a best-fit plane, use the two directions the decomposition already
names. The x axis is the cell's raw displacement, target minus base, along the
common mode gamma: the shared shift that cancels in the excess. The y axis is
the cell's within-instruction excess along the named candidate's residual rho,
orthogonalized against gamma: exactly the quantity the registered test reads.
Raw displacement on y would let each instruction's shared shift leak in as a
diagonal every candidate rides; the excess blocks on the instruction, so an
honest candidate's cells center on y = 0 no matter how far x carries them.
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
from src.geometry.behavior_space_decomposition import displacement_decomposition

__all__ = ["plot_two_halves_plane"]


def plot_two_halves_plane(rv: RunVectors, out_path, candidate: str) -> dict:
    """Write the (common mode, residual) plane of per-cell displacements."""
    if candidate not in rv.principals:
        raise ValueError(f"{candidate} is not a candidate of {rv.name}")
    dec = displacement_decomposition(rv.target, rv.base)
    gamma = dec["common"]
    gamma_norm = float(np.linalg.norm(gamma))
    rho = dec["residual"][rv.principals.index(candidate)]
    if gamma_norm == 0 or float(np.linalg.norm(rho)) == 0:
        raise ValueError(f"{rv.name}: a zero direction leaves no plane to draw")
    gamma_hat = gamma / gamma_norm
    rho_perp = rho - (rho @ gamma_hat) * gamma_hat
    rho_hat = rho_perp / np.linalg.norm(rho_perp)

    delta = rv.target - rv.base
    x = np.nansum(delta * gamma_hat[None, None, :], axis=2)
    y = np.nansum(excess_effect(rv.target, rv.base) * rho_hat[None, None, :],
                  axis=2)

    fig, panel = plt.subplots(figsize=(7.2, 5.4))
    panel.axhline(0, color="black", linewidth=0.5, alpha=0.35)
    panel.axvline(0, color="black", linewidth=0.5, alpha=0.35)
    for ci, principal in enumerate(rv.principals):
        named = principal == candidate
        panel.scatter(x[ci], y[ci], s=10 if named else 7,
                      color=HIGHLIGHT_INK if named else OTHER_INK,
                      alpha=0.6 if named else 0.25, linewidths=0,
                      zorder=3 if named else 2)
        panel.scatter([float(np.nanmean(x[ci]))], [float(np.nanmean(y[ci]))],
                      marker="D", s=46,
                      color=HIGHLIGHT_INK if named else "#111827",
                      edgecolor="white", linewidth=0.8, zorder=4)
    panel.annotate(rv.name_of(candidate),
                   xy=(float(np.nanmean(x[rv.principals.index(candidate)])),
                       float(np.nanmean(y[rv.principals.index(candidate)]))),
                   xytext=(8, 8), textcoords="offset points", fontsize=9,
                   color=HIGHLIGHT_INK, weight="bold", zorder=5)

    panel.set_xlabel("displacement along the common mode $\\gamma$ "
                     "(shared by every candidate, cancels in the excess)",
                     fontsize=9)
    panel.set_ylabel("excess along the residual $\\rho_C$, orthogonal to "
                     "$\\gamma$\n(what the registered test measures)", fontsize=9)
    panel.set_title(f"{rv.name}, judge level {rv.level}: per-cell displacement "
                    "in the two-halves frame", fontsize=10)
    panel.grid(linewidth=0.3, alpha=0.4)
    panel.set_axisbelow(True)
    panel.tick_params(labelsize=8)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.93, bottom=0.12)

    out_path = ensure_parent(out_path)
    fig.savefig(out_path, dpi=200)
    fig.savefig(Path(out_path).with_suffix(".pdf"))
    plt.close(fig)
    return {"figure": str(out_path), "candidate": candidate,
            "gamma_norm": gamma_norm,
            "mean_x": {p: float(np.nanmean(x[ci]))
                       for ci, p in enumerate(rv.principals)},
            "mean_y": {p: float(np.nanmean(y[ci]))
                       for ci, p in enumerate(rv.principals)}}
