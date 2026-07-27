"""Color encodings of the raw cell cloud: by candidate, and by framing.

Two answers to "what organizes this cloud", asked with color. Coloring by
candidate shows whether any group separates on its own; coloring by collection
framing shows how much of the structure is the system prompt. Both draw the
target model's cells on the pooled plane of the two-panel map, so the view is
comparable with the appendix figure.
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

__all__ = ["CANDIDATE_HUES", "FRAMING_HUES", "framing_of", "plot_candidate_colored_map",
           "plot_framing_colored_map"]

#: Ten distinguishable hues for the ten candidate groups.
CANDIDATE_HUES = ["#C2447E", "#2A78D6", "#1F7A5A", "#B8860B", "#6B46C1",
                  "#D2691E", "#008B8B", "#8B3A62", "#556B2F", "#4A5568"]

#: One hue per collection framing, fixed across every run.
FRAMING_HUES = {"none": "#6B7280", "live_deployment": "#2A78D6",
                "committed_supporter": "#C2447E", "unreviewed_authority": "#6B46C1"}


def framing_of(instruction_id: str) -> str:
    """The collection system prompt, the prefix of the composite instruction id."""
    return instruction_id.split("::", 1)[0]


def _target_plane(rv: RunVectors):
    n_c, n_i, n_k = rv.target.shape
    pooled = np.vstack([rv.base.reshape(n_c * n_i, n_k),
                        rv.target.reshape(n_c * n_i, n_k)])
    return principal_plane(pooled)


def _finish(fig, panel, share, out_path, shown: dict) -> dict:
    panel.set_xlabel(f"PC1 ({share[0]:.0%} of variance)", fontsize=9)
    panel.set_ylabel(f"PC2 ({share[1]:.0%} of variance)", fontsize=9)
    panel.grid(linewidth=0.3, alpha=0.4)
    panel.set_axisbelow(True)
    panel.tick_params(labelsize=8)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.11)
    out_path = ensure_parent(out_path)
    fig.savefig(out_path, dpi=200)
    fig.savefig(Path(out_path).with_suffix(".pdf"))
    plt.close(fig)
    return {"figure": str(out_path), **shown}


def plot_candidate_colored_map(rv: RunVectors, out_path) -> dict:
    """The target's cells on the pooled plane, one hue per candidate."""
    center, basis, share = _target_plane(rv)
    fig, panel = plt.subplots(figsize=(7.6, 5.2))
    for ci, principal in enumerate(rv.principals):
        pts = project_points(rv.target[ci], center, basis)
        panel.scatter(pts[:, 0], pts[:, 1], s=9, alpha=0.55, linewidths=0,
                      color=CANDIDATE_HUES[ci % len(CANDIDATE_HUES)],
                      label=rv.name_of(principal))
    panel.legend(fontsize=7, ncol=2, frameon=False, loc="best")
    panel.set_title(f"{rv.name}, judge level {rv.level}: target cells "
                    "colored by candidate", fontsize=10)
    return _finish(fig, panel, share, out_path,
                   {"encoding": "candidate", "n_hues": len(rv.principals)})


def plot_framing_colored_map(rv: RunVectors, out_path) -> dict:
    """The target's cells on the pooled plane, one hue per collection framing."""
    center, basis, share = _target_plane(rv)
    framings = [framing_of(i) for i in rv.instructions]
    fig, panel = plt.subplots(figsize=(7.6, 5.2))
    for name, hue in FRAMING_HUES.items():
        idx = [k for k, f in enumerate(framings) if f == name]
        if not idx:
            continue
        pts = project_points(rv.target[:, idx, :].reshape(-1, rv.target.shape[2]),
                             center, basis)
        panel.scatter(pts[:, 0], pts[:, 1], s=9, alpha=0.55, linewidths=0,
                      color=hue, label=name.replace("_", " "))
    panel.legend(fontsize=8, frameon=False, loc="best")
    panel.set_title(f"{rv.name}, judge level {rv.level}: target cells "
                    "colored by collection framing", fontsize=10)
    return _finish(fig, panel, share, out_path,
                   {"encoding": "framing", "framings": sorted(set(framings))})
