"""Evidence accumulation: the running mean projection as instructions pool.

The registered test's power comes from averaging a small excess over many
instructions. This view draws that averaging as a race: for each candidate,
the running mean of its cells' projection on the named direction, as the
instructions accumulate in their deterministic sorted order. The envelope is
an approximate null band, plus and minus two pooled per-instruction standard
deviations over the square root of the count, shrinking as evidence pools. A
loyal candidate's line escapes the band and stays out; the rest never leave.
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

__all__ = ["plot_evidence_accumulation"]


def plot_evidence_accumulation(rv: RunVectors, out_path, candidate: str) -> dict:
    """Write the running-mean race along the candidate's direction."""
    if candidate not in rv.principals:
        raise ValueError(f"{candidate} is not a candidate of {rv.name}")
    d = excess_effect(rv.target, rv.base)
    mean = np.nanmean(d[rv.principals.index(candidate)], axis=0)
    norm = float(np.linalg.norm(mean))
    if norm == 0:
        raise ValueError(f"{rv.name}: the {candidate} direction has zero norm")
    scores = np.nansum(d * (mean / norm)[None, None, :], axis=2)

    order = np.argsort(np.array(rv.instructions))
    scores = scores[:, order]
    n_c, n_i = scores.shape
    counts = np.arange(1, n_i + 1)
    running = np.nancumsum(scores, axis=1) / counts[None, :]

    # An approximate null envelope: the pooled per-instruction spread of the
    # projections, shrinking with the square root of the pooled count.
    sigma = float(np.nanstd(scores, ddof=1))
    band = 2.0 * sigma / np.sqrt(counts)

    fig, panel = plt.subplots(figsize=(8.6, 4.9))
    panel.fill_between(counts, -band, band, color="#6B7280", alpha=0.14,
                       linewidth=0)
    panel.axhline(0, color="black", linewidth=0.6)
    for ci, principal in enumerate(rv.principals):
        named = principal == candidate
        panel.plot(counts, running[ci],
                   color=HIGHLIGHT_INK if named else OTHER_INK,
                   linewidth=2.2 if named else 1.0,
                   alpha=1.0 if named else 0.55, zorder=3 if named else 2)
    panel.annotate(rv.name_of(candidate),
                   xy=(counts[-1], running[rv.principals.index(candidate), -1]),
                   xytext=(-8, 10), textcoords="offset points", fontsize=10,
                   color=HIGHLIGHT_INK, weight="bold", ha="right")
    panel.set_xlim(1, n_i)
    panel.set_xlabel("instructions pooled, in sorted instruction-id order",
                     fontsize=9)
    panel.set_ylabel(f"running mean projection on the\n"
                     f"{rv.name_of(candidate)} direction (reply share)",
                     fontsize=9)
    panel.set_title(f"{rv.name}, judge level {rv.level}: the averaging that "
                    "powers the test, drawn as a race", fontsize=10)
    panel.grid(linewidth=0.3, alpha=0.4)
    panel.set_axisbelow(True)
    panel.tick_params(labelsize=8)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.92, bottom=0.13)

    out_path = ensure_parent(out_path)
    fig.savefig(out_path, dpi=200)
    fig.savefig(Path(out_path).with_suffix(".pdf"))
    plt.close(fig)
    final = {p: float(running[ci, -1]) for ci, p in enumerate(rv.principals)}
    return {"figure": str(out_path), "candidate": candidate,
            "final_running_mean": final, "final_band": float(band[-1]),
            "band_sigma_pooled": sigma}
