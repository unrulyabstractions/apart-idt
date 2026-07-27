"""Linear views of the behavior cloud: planes, variance shares, displacement.

Three decompositions, all deterministic and all reported as description rather
than inference: the permutation test of stage 6 stays the only arbiter.

* ``principal_plane`` fits the best 2-plane of a point cloud by SVD, for the
  maps. For the excess cloud the plane is fit through the origin, because the
  origin is the null: a cell with zero excess sits exactly there.
* ``variance_partition`` splits a (candidate, instruction) cell cloud's
  variance into the part carried by instruction identity and the part carried
  by candidate identity. This quantifies how much of the raw geometry is about
  what was asked rather than who asked.
* ``displacement_decomposition`` splits each candidate's centroid displacement
  from base to target into the mean displacement over candidates, the common
  mode, plus a candidate-specific residual. The residual is the registered
  test's mean excess scaled by (N-1)/N exactly, so this is the paper's two
  halves drawn as vectors rather than a new statistic.
"""

from __future__ import annotations

import numpy as np

__all__ = ["principal_plane", "project_points", "variance_partition",
           "displacement_decomposition"]


def principal_plane(points: np.ndarray, through_origin: bool = False
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Best-fit 2-plane of a cloud: center, basis (2, K), variance shares (2,).

    Rows with any NaN are unobserved cells and are dropped from the fit. With
    ``through_origin`` the center is pinned at zero and the shares are of total
    squared norm about the origin, which is the right reading for excess.
    """
    x = points[~np.isnan(points).any(axis=1)]
    if x.shape[0] < 3:
        raise ValueError("a plane needs at least three observed cells")
    center = np.zeros(x.shape[1]) if through_origin else x.mean(axis=0)
    _, s, vt = np.linalg.svd(x - center, full_matrices=False)
    total = float(np.sum(s**2))
    share = s[:2] ** 2 / total if total > 0 else np.zeros(2)
    return center, vt[:2], share


def project_points(points: np.ndarray, center: np.ndarray, basis: np.ndarray
                   ) -> np.ndarray:
    """Plane coordinates of each row; NaN rows stay NaN rather than vanish."""
    return (points - center) @ basis.T


def variance_partition(v: np.ndarray) -> dict:
    """Shares of a (C, I, K) cell cloud's variance held by each identity.

    The share of instruction identity is the variance of the per-instruction
    mean vectors, weighted by how many cells each instruction contributed; the
    candidate share is the same over per-candidate means. Unobserved cells are
    excluded from every mean and every count.
    """
    grand = np.nanmean(v, axis=(0, 1))
    present = ~np.isnan(v).any(axis=2)
    ss_total = float(np.nansum((v - grand) ** 2))
    if ss_total == 0:
        raise ValueError("the cell cloud has no variance to partition")
    instr_means = np.nanmean(v, axis=0)
    cand_means = np.nanmean(v, axis=1)
    ss_instr = float(np.nansum(present.sum(axis=0)[:, None] * (instr_means - grand) ** 2))
    ss_cand = float(np.nansum(present.sum(axis=1)[:, None] * (cand_means - grand) ** 2))
    return {"instruction_share": ss_instr / ss_total,
            "candidate_share": ss_cand / ss_total,
            "n_cells": int(present.sum())}


def displacement_decomposition(target: np.ndarray, base: np.ndarray) -> dict:
    """Each candidate's base-to-target displacement, split into the two halves.

    ``delta[c]`` is the mean over instructions of the target cell vector minus
    the matched base cell vector. ``common`` is the mean of the deltas over
    candidates and cancels in any candidate-versus-candidate statistic, which
    is why stage 6 measures it separately as the firing-rate elevation.
    ``residual[c]`` is what the registered test sees: its mean excess for
    candidate c equals ``residual[c] * N / (N - 1)`` identically.
    """
    if target.shape != base.shape:
        raise ValueError("target and base must hold the same cells")
    n = target.shape[0]
    if n < 2:
        raise ValueError("the decomposition needs at least two candidates")
    delta = np.nanmean(target - base, axis=1)
    common = delta.mean(axis=0)
    residual = delta - common
    return {"delta": delta, "common": common, "residual": residual,
            "common_norm": float(np.linalg.norm(common)),
            "residual_norms": np.linalg.norm(residual, axis=1)}
