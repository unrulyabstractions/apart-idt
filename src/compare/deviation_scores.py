"""Which group is treated differently, and on which axes.

The score is ``median_j sqrt(JS_ij)``: a group's median distance to the other
groups on the pairwise matrix. It needs only the distance matrix and tolerates
up to half the groups being anomalous, which a centroid-based score does not,
since an outlier drags the centroid toward itself and understates its own
deviation.

Flagging is a one-sided robust ``z`` against the median and the MAD, at
``z > 3``.

"Outlier" presumes one dominant cluster. Check that precondition first with
:mod:`src.compare.medoid_cluster_check`; two balanced clusters mean the
population is split, and outlier hunting on a split population flags everything.
"""

from __future__ import annotations

import numpy as np

__all__ = ["median_distance_scores", "robust_z", "benjamini_hochberg",
           "axis_attribution"]

FLAG_Z = 3.0
_HEAVY_TAIL_RATIO = 3.0


def median_distance_scores(distance: np.ndarray) -> np.ndarray:
    """Median ``sqrt(JS)`` to the other groups. Matrix-only, 50% breakdown."""
    n = distance.shape[0]
    return np.array([float(np.median(np.delete(distance[i], i))) for i in range(n)])


def robust_z(scores: np.ndarray) -> dict:
    """One-sided robust ``z`` against median and MAD, on a log scale if skewed.

    The scores are non-negative and right-skewed, where a raw MAD makes the
    largest score its own reference. When the tail is heavy, defined here as
    ``max > 3 x median``, the ``z`` is taken on ``log s`` instead. Which scale was
    used is returned, never left implicit.

    A zero MAD needs care rather than a zeroed ``z``. It means at least half the
    groups are exactly tied, so if any score sits above them the majority scale
    is degenerate and that score is separated on every scale: it is reported as
    infinite and flagged. Returning zeros there would turn the one case the
    detector exists to catch into a silent negative. JSON has no infinity, so
    such a ``z`` serializes as null and ``degenerate_scale`` says why.
    """
    positive = np.all(scores > 0)
    median = float(np.median(scores))
    heavy = positive and median > 0 and float(np.max(scores)) > _HEAVY_TAIL_RATIO * median
    values = np.asarray(np.log(scores) if heavy else scores, dtype=float)
    center = float(np.median(values))
    mad = 1.4826 * float(np.median(np.abs(values - center)))
    degenerate = mad <= 0
    if not degenerate:
        z = (values - center) / mad
    else:
        z = np.where(values > center, np.inf, 0.0)
    return {"scale": "log" if heavy else "raw", "median": center, "mad": mad,
            "degenerate_scale": bool(degenerate),
            "z": [None if np.isinf(v) else float(v) for v in z],
            "flagged": (z > FLAG_Z).tolist()}


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """BH step-up: return the boolean rejection vector at level ``alpha``."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = p.size
    thresholds = alpha * (np.arange(1, m + 1) / m)
    passing = p[order] <= thresholds
    reject = np.zeros(m, dtype=bool)
    if np.any(passing):
        reject[order[: int(np.flatnonzero(passing).max()) + 1]] = True
    return reject


def axis_attribution(rows: np.ndarray, weights: np.ndarray, axes, top: int = 8) -> list[list[dict]]:
    """Per-axis terms ``P_i(x) ln(P_i(x)/Pbar(x))``, the axes driving each group's score.

    Signed on purpose: a positive term is an axis the group fires more than the
    pooled behavior, a negative one an axis it fires less. Both are differential
    treatment.
    """
    mixture = weights @ rows
    out = []
    for p in rows:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.divide(p, mixture, out=np.ones_like(p), where=mixture > 0)
            terms = np.where(p > 0, p * np.log(ratio), 0.0)
        order = np.argsort(-np.abs(terms))[:top]
        out.append([{"axis_id": axes[j], "term_nats": float(terms[j]),
                     "rate": float(p[j]), "pooled_rate": float(mixture[j])}
                    for j in order])
    return out
