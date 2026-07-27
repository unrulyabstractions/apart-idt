"""Structure of the candidate directions: who resembles whom, and where it fires.

Two views of the mean excess vectors the displacement decomposition produces.
The cosine matrix asks whether any two candidates' directions resemble each
other; because the excess sums to zero over candidates, the uninteresting
background is a slight anti-correlation, and a strongly negative row is the
mirror a loyal candidate casts on everyone else. The framing breakdown asks
which collection system prompt carries the named candidate's signal, by
averaging its excess separately within each framing's instructions.
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
from src.geometry.colored_cloud_figures import FRAMING_HUES, framing_of

__all__ = ["plot_direction_cosines", "plot_framing_breakdown"]


def _save(fig, out_path, shown: dict) -> dict:
    out_path = ensure_parent(out_path)
    fig.savefig(out_path, dpi=200)
    fig.savefig(Path(out_path).with_suffix(".pdf"))
    plt.close(fig)
    return {"figure": str(out_path), **shown}


def plot_direction_cosines(rv: RunVectors, out_path) -> dict:
    """Cosine similarity between every pair of candidate mean-excess directions."""
    mean = np.nanmean(excess_effect(rv.target, rv.base), axis=1)
    norms = np.linalg.norm(mean, axis=1, keepdims=True)
    unit = np.divide(mean, norms, out=np.zeros_like(mean), where=norms > 0)
    cos = unit @ unit.T
    names = [rv.name_of(p) for p in rv.principals]

    fig, panel = plt.subplots(figsize=(7.4, 6.2))
    image = panel.imshow(cos, cmap="RdBu_r", vmin=-1, vmax=1)
    panel.set_xticks(range(len(names)))
    panel.set_xticklabels(names, rotation=40, ha="right", fontsize=7.5)
    panel.set_yticks(range(len(names)))
    panel.set_yticklabels(names, fontsize=7.5)
    for i in range(len(names)):
        for j in range(len(names)):
            panel.text(j, i, f"{cos[i, j]:+.2f}", ha="center", va="center",
                       fontsize=6.2,
                       color="white" if abs(cos[i, j]) > 0.6 else "#1d2129")
    fig.colorbar(image, ax=panel, shrink=0.8, label="cosine")
    panel.set_title(f"{rv.name}, judge level {rv.level}: cosine between "
                    "candidate mean-excess directions", fontsize=10)
    fig.subplots_adjust(left=0.22, right=0.98, top=0.92, bottom=0.20)
    return _save(fig, out_path, {"cosines": cos.tolist()})


def plot_framing_breakdown(rv: RunVectors, out_path, candidate: str,
                           top: int = 8) -> dict:
    """The named candidate's mean excess per collection framing, top axes."""
    if candidate not in rv.principals:
        raise ValueError(f"{candidate} is not a candidate of {rv.name}")
    d = excess_effect(rv.target, rv.base)[rv.principals.index(candidate)]
    overall = np.nanmean(d, axis=0)
    order = np.argsort(-np.abs(overall))[:top]
    framings = [framing_of(i) for i in rv.instructions]
    groups = sorted(set(framings), key=list(FRAMING_HUES).index)

    fig, panel = plt.subplots(figsize=(7.6, 4.6))
    y = np.arange(len(order))
    height = 0.8 / len(groups)
    for g, name in enumerate(groups):
        idx = [k for k, f in enumerate(framings) if f == name]
        mean = np.nanmean(d[idx][:, order], axis=0)
        panel.barh(y + g * height - 0.4 + height / 2, mean, height,
                   color=FRAMING_HUES[name], label=name.replace("_", " "),
                   edgecolor="white", linewidth=0.3)
    panel.axvline(0, color="black", linewidth=0.6)
    panel.set_yticks(y)
    panel.set_yticklabels([rv.axes[j].replace("_", " ") for j in order], fontsize=8)
    panel.invert_yaxis()
    panel.legend(fontsize=7.5, frameon=False, loc="best")
    panel.set_xlabel("mean excess share of replies within the framing", fontsize=9)
    panel.set_title(f"{rv.name}, judge level {rv.level}:\nthe "
                    f"{rv.name_of(candidate)} excess, split by framing", fontsize=10)
    panel.grid(axis="x", linewidth=0.3, alpha=0.4)
    panel.set_axisbelow(True)
    fig.subplots_adjust(left=0.36, right=0.97, top=0.86, bottom=0.13)
    return _save(fig, out_path, {"candidate": candidate,
                                 "axes_shown": [rv.axes[j] for j in order],
                                 "framings": groups})
