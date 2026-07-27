"""Choosing which named behavior axes to draw as a compass inside a plane view.

A plane view of the excess has the run's $K$ named coordinates behind it, one
per frozen scoring question, and each one projects to a direction in the plane.
Drawing all of them is unreadable, so this picks a few: the axes the plane is
most built from, thinned so that no two labels sit at the same angle.
"""

from __future__ import annotations

import re

import numpy as np

__all__ = ["readable_axis", "select_compass_axes"]

#: Guaranteed axes carry an ordinal prefix in their id that names nothing.
GUARANTEED_PREFIX = re.compile(r"^g\d+_")


def readable_axis(axis_id: str) -> str:
    """The axis id as running text: no ordinal prefix, no underscores."""
    return GUARANTEED_PREFIX.sub("", axis_id).replace("_", " ")


def select_compass_axes(basis: np.ndarray, axis_ids: tuple[str, ...],
                        n_axes: int = 6, min_angle_deg: float = 12.0,
                        along: np.ndarray | None = None,
                        min_cosine: float = 0.7) -> list[dict]:
    """The named axes to draw as rays, thinned so their labels cannot collide.

    ``basis`` is the ``(2, K)`` plane basis. Axis $j$'s loading is the column
    ``(basis[0, j], basis[1, j])``: where a one-unit rise in that one named
    behavior, and nothing else, would land in the plane.

    Given ``along``, only axes lying closely with that direction are eligible:
    their cosine against it must reach ``min_cosine``, so a behavior pointing
    sideways is not drawn even when its component along the reference is
    positive. Eligible axes are ranked by how far they reach along the
    reference, which answers what the arrow in that direction consists of.
    Without ``along``, axes are ranked by loading size instead. Either way an
    axis is skipped when its direction lies within ``min_angle_deg`` of one
    already taken.
    """
    load = np.asarray(basis, dtype=float)[:2].T  # (K, 2)
    if load.shape[0] != len(axis_ids):
        raise ValueError("the basis and the axis ids disagree on K")
    norms = np.linalg.norm(load, axis=1)
    safe = np.where(norms > 0.0, norms, 1.0)
    if along is None:
        score = norms
        cosine = np.ones_like(norms)
    else:
        reference = np.asarray(along, dtype=float)
        score = load @ (reference / (np.linalg.norm(reference) or 1.0))
        cosine = score / safe
    taken: list[dict] = []
    for j in np.argsort(-score):
        if score[j] <= 0.0 or norms[j] <= 0.0 or len(taken) == n_axes:
            break
        if cosine[j] < min_cosine:
            continue
        angle = float(np.degrees(np.arctan2(load[j, 1], load[j, 0])))
        if any(abs((angle - t["angle_deg"] + 180.0) % 360.0 - 180.0) < min_angle_deg
               for t in taken):
            continue
        taken.append({"axis_id": axis_ids[j], "label": readable_axis(axis_ids[j]),
                      "loading": [float(load[j, 0]), float(load[j, 1])],
                      "loading_norm": float(norms[j]), "angle_deg": angle,
                      "along_component": float(score[j]),
                      "cosine_along": float(cosine[j])})
    return taken
