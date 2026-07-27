"""Placing many labels near their own anchors without letting them collide.

A plane view carries one label per candidate arrow and one per named behavior
ray, and several of those anchors sit close together whenever the arrows are
short. This walks the labels in priority order and gives each the shortest
offset that keeps it clear of the labels already placed, so a label stays next
to the thing it names instead of being pushed somewhere arbitrary.
"""

from __future__ import annotations

import numpy as np

__all__ = ["nudge_label_offsets"]

#: Rotations tried at each offset length, in degrees off the anchor's own
#: outward direction. Straight out first, then progressively sideways.
ROTATIONS = (0.0, 35.0, -35.0, 70.0, -70.0, 110.0, -110.0, 145.0, -145.0, 180.0)


def _rotate(v: np.ndarray, degrees: float) -> np.ndarray:
    rad = np.radians(degrees)
    cos, sin = np.cos(rad), np.sin(rad)
    return np.array([cos * v[0] - sin * v[1], sin * v[0] + cos * v[1]])


#: Outward placements only, for a label that must stay beyond its own anchor
#: rather than swinging back across the origin.
OUTWARD = (0.0, 30.0, -30.0, 60.0, -60.0)


def nudge_label_offsets(anchors, radials, base: float, min_sep: float,
                        lengths: tuple[float, ...] = (1.0, 1.9, 2.9),
                        rotations_per_anchor=None) -> np.ndarray:
    """Offsets keeping each label near its anchor and clear of the others.

    Units are whatever ``anchors`` are given in: ``base`` is the preferred
    offset length and ``min_sep`` the smallest tolerated gap between two placed
    labels. Anchors are honoured in the order given, so the most important
    label goes first and never gets moved for a later one.
    ``rotations_per_anchor`` optionally restricts the directions tried for each
    anchor, which keeps a label that names a ray from landing behind it.
    """
    placed: list[np.ndarray] = []
    offsets: list[np.ndarray] = []
    per_anchor = rotations_per_anchor or [ROTATIONS] * len(list(anchors))
    for anchor, radial, allowed in zip(np.asarray(anchors, float),
                                       np.asarray(radials, float), per_anchor):
        unit = radial / (np.linalg.norm(radial) or 1.0)
        chosen = None
        for length in lengths:
            for degrees in allowed:
                candidate = _rotate(unit, degrees) * base * length
                if all(np.linalg.norm(anchor + candidate - p) >= min_sep
                       for p in placed):
                    chosen = candidate
                    break
            if chosen is not None:
                break
        if chosen is None:  # crowded beyond rescue: take the longest ring
            chosen = unit * base * lengths[-1]
        placed.append(anchor + chosen)
        offsets.append(chosen)
    return np.array(offsets)
