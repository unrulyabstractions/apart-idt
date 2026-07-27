"""Axis filtering for the geometry views: keep the axes that actually fire.

Most axes fire on almost nothing, and a dead axis contributes a near-constant
coordinate: it dilutes every distance without separating any two cells. The
filter keeps an axis when its pooled firing rate, over all cells of either
model, reaches a floor. Filtering changes only what a map displays; the
registered test is never run on a filtered space.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ACTIVITY_FLOOR", "active_axis_mask", "filter_axes"]

#: Pooled firing-rate floor for an axis to count as active, on either model.
ACTIVITY_FLOOR = 0.02


def active_axis_mask(target: np.ndarray, base: np.ndarray,
                     floor: float = ACTIVITY_FLOOR) -> np.ndarray:
    """Boolean mask over axes: pooled rate at or above the floor on either model.

    The pool is over every observed cell of one model, so a rare axis that one
    candidate fires often still counts as active.
    """
    rate_t = np.nanmean(target, axis=(0, 1))
    rate_b = np.nanmean(base, axis=(0, 1))
    return (rate_t >= floor) | (rate_b >= floor)


def filter_axes(target: np.ndarray, base: np.ndarray, axes: tuple[str, ...],
                floor: float = ACTIVITY_FLOOR) -> dict:
    """Both models restricted to active axes, with the counts a caption needs."""
    mask = active_axis_mask(target, base, floor)
    if not mask.any():
        raise ValueError("no axis reaches the activity floor")
    return {"target": target[:, :, mask], "base": base[:, :, mask],
            "axes": tuple(a for a, keep in zip(axes, mask) if keep),
            "n_kept": int(mask.sum()), "n_total": len(axes), "floor": floor}
