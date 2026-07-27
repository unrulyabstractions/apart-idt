"""Fix the geometry before ranking or clustering anything.

Raw KL is the wrong object to hand a ranker: ``D(P_i||P_j)`` is asymmetric,
unbounded, and infinite the moment ``P_i`` puts mass on an axis ``P_j`` never
fires, which is the common case when a hundred axes are scored on a few hundred
replies. Two repairs, used for different jobs:

* add-``alpha`` smoothing, ``(n_i P_i + alpha) / (n_i + alpha K)``, makes KL
  finite so the information radius and the likelihood-ratio statistic are
  computable at all;
* Jensen-Shannon is bounded by ``ln 2``, finite on any support mismatch, and its
  square root is a true metric. That is the distance used for clustering and for
  the median-based deviation score.
"""

from __future__ import annotations

import numpy as np

__all__ = ["probabilities", "smoothed_probabilities", "kl_matrix",
           "jensen_shannon_matrix", "js_distance_matrix",
           "total_variation_bound"]

SMOOTHING_ALPHA = 0.5


def probabilities(counts: np.ndarray) -> np.ndarray:
    """Row-normalize a count table; an all-zero row becomes uniform."""
    totals = counts.sum(axis=1, keepdims=True).astype(float)
    uniform = np.full(counts.shape, 1.0 / counts.shape[1])
    return np.divide(counts, totals, out=uniform, where=totals > 0)


def smoothed_probabilities(counts: np.ndarray, alpha: float = SMOOTHING_ALPHA) -> np.ndarray:
    """Add-``alpha`` (Jeffreys prior) rows, so every KL entry is finite."""
    k = counts.shape[1]
    totals = counts.sum(axis=1, keepdims=True).astype(float)
    return (counts + alpha) / (totals + alpha * k)


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL in nats, summing only where ``p`` has mass."""
    mask = p > 0
    if np.any(q[mask] <= 0):
        return float("inf")
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def kl_matrix(rows: np.ndarray) -> np.ndarray:
    """``D[i, j] = D(P_i || P_j)`` in nats. Pass smoothed rows to keep it finite."""
    n = rows.shape[0]
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                out[i, j] = _kl(rows[i], rows[j])
    return out


def jensen_shannon_matrix(rows: np.ndarray) -> np.ndarray:
    """Pairwise JS in nats, bounded by ``ln 2`` and finite on raw rows."""
    n = rows.shape[0]
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            m = 0.5 * (rows[i] + rows[j])
            js = 0.5 * _kl(rows[i], m) + 0.5 * _kl(rows[j], m)
            out[i, j] = out[j, i] = js
    return out


def js_distance_matrix(rows: np.ndarray) -> np.ndarray:
    """``sqrt(JS)``: the working metric for clustering and robust scores."""
    return np.sqrt(np.clip(jensen_shannon_matrix(rows), 0.0, None))


def total_variation_bound(divergence: float) -> float:
    """Pinsker: ``D >= 2 TV^2``, so ``TV <= sqrt(D / 2)``.

    Reported because nats are not interpretable on their own, while a total
    variation bound is: it is the largest share of probability mass that could
    be sitting on different axes.
    """
    return float(np.sqrt(max(divergence, 0.0) / 2.0))
