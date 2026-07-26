"""One number for "do the groups behave alike": the information radius.

The weighted KL centroid of ``P_1..P_N`` is exactly the mixture
``Pbar = sum_i w_i P_i``, so the radius

    Ibar = sum_i w_i D(P_i || Pbar) = H(Pbar) - sum_i w_i H(P_i) = I(Z; X)

is both the mean divergence to the centroid and the mutual information between
"which group is being served" and "which axis fires". It is an exact split of
total behavioral entropy into a between-group part and a within-group part, so
``Ibar = 0`` if and only if every group is treated identically.

Two cautions are reported rather than buried. The plug-in estimate is biased
upward by roughly ``(N-1)(K-1)/(2n)`` nats, which is not small when a hundred
axes are scored on a few hundred firings, so a Miller-Madow corrected value ships
next to it and the significance question goes to the permutation null. And the
mean pairwise divergence ``Dbar``, the quantity available when only a divergence
matrix survives, always overstates the radius: ``Dbar = Ibar + sum_j w_j
D(Pbar || P_j) >= Ibar``.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from src.compare.divergence_geometry import total_variation_bound

__all__ = ["InformationRadius", "count_weights", "information_radius"]


@dataclass(frozen=True)
class InformationRadius:
    n_groups: int
    n_axes: int
    radius_nats: float
    radius_miller_madow: float
    plug_in_bias_nats: float
    normalized: float  # radius / min(ln N, ln K), in [0, 1]
    entropy_mixture: float
    entropy_within: float
    mean_pairwise_kl: float  # Dbar, on smoothed rows; >= radius
    centroid_gap: float  # Dbar - Ibar
    rms_tv_to_centroid: float  # Pinsker bound on RMS total variation

    def as_dict(self) -> dict:
        return asdict(self)


def count_weights(n: np.ndarray) -> np.ndarray:
    """Group weights ``n_i / n``. Sample sizes differ here, so uniform is wrong.

    With these weights the mixture is the pooled distribution and the radius is
    the likelihood-ratio statistic divided by ``2n``, which is what makes the
    descriptive number and the test the same number.
    """
    total = float(n.sum())
    if total <= 0:
        raise ValueError("no firings in the table")
    return n.astype(float) / total


def _entropy(p: np.ndarray) -> float:
    mask = p > 0
    return float(-np.sum(p[mask] * np.log(p[mask])))


def _miller_madow(p: np.ndarray, n: float) -> float:
    """Entropy with the ``(m - 1) / 2n`` support correction; ``m`` = observed symbols."""
    if n <= 0:
        return _entropy(p)
    return _entropy(p) + (float(np.count_nonzero(p)) - 1.0) / (2.0 * n)


def information_radius(
    rows: np.ndarray,
    weights: np.ndarray,
    n: np.ndarray,
    smoothed_kl: np.ndarray | None = None,
) -> InformationRadius:
    """The radius and its decomposition. ``rows`` are raw (unsmoothed) profiles.

    Smoothing is unnecessary here: the centroid is a mixture, so it carries mass
    wherever any group does and every ``D(P_i || Pbar)`` is finite by
    construction. ``smoothed_kl`` is only used for the ``Dbar`` bound.
    """
    n_groups, n_axes = rows.shape
    mixture = weights @ rows
    entropy_within = float(np.sum(weights * np.array([_entropy(p) for p in rows])))
    radius = _entropy(mixture) - entropy_within
    total = float(n.sum())
    within_mm = float(np.sum(weights * np.array(
        [_miller_madow(p, ni) for p, ni in zip(rows, n)])))
    radius_mm = _miller_madow(mixture, total) - within_mm
    ceiling = min(np.log(n_groups), np.log(n_axes))
    if smoothed_kl is None:
        mean_pairwise = float("nan")
    else:
        mean_pairwise = float(weights @ smoothed_kl @ weights)
    return InformationRadius(
        n_groups=n_groups,
        n_axes=n_axes,
        radius_nats=float(radius),
        radius_miller_madow=float(radius_mm),
        plug_in_bias_nats=(n_groups - 1) * (n_axes - 1) / (2.0 * total),
        normalized=float(radius / ceiling) if ceiling > 0 else 0.0,
        entropy_mixture=_entropy(mixture),
        entropy_within=entropy_within,
        mean_pairwise_kl=mean_pairwise,
        centroid_gap=mean_pairwise - float(radius),
        rms_tv_to_centroid=total_variation_bound(float(radius)),
    )
