"""Detach on breadth across axes rather than on one candidate's loudest axis.

The peak statistic of ``candidate_detachment_statistic`` collapses every axis into
each candidate's single largest standardized shift. That throws away the thing
that separates a loyalty from a name effect. Measured on the calibration organism
under ``informed``, the documented principal's peak is only the fifth largest in
its own table, while it is an outlier on sixty-six of a hundred and two axes and
no other candidate is an outlier on more than two. A statistic reading only the
peak cannot see that, and on that table it does not even select the right
candidate.

So the comparison moves inside the axis. For each axis we ask which candidate is
the outlier among candidates, then ask whether the same candidate keeps being the
outlier:

    z_{C,j} = (t_{C,j} - med_C' t_{C',j}) / MAD_C' t_{C',j}

    R_C = max over k <= k0 of (1/sqrt(k)) * sum of C's top k values of z

One noisy axis gives roughly the single largest z. Several coherent axes
accumulate, which is what an idiosyncratic name effect cannot fake. The scan over
k is a higher-criticism device: it lets the data choose how broad the signature is
instead of fixing a number of axes in advance.

The z uses signed t rather than its absolute value, so axes pushed in opposite
directions cancel instead of adding. The paper registered axis polarities but the
artifacts carry none, so the sign here is the observed direction of the shift
against the other candidates and not a registered expectation.

Two readings are reported, both calibrated by the same within-instruction label
shuffles. The detachment asks whether the leading R stands out from the other
candidates' R. The maximum asks whether the leading R is larger than shuffling
produces anywhere. Neither needs a reference model.
"""

from __future__ import annotations

import numpy as np

from src.compare.behavior_count_table import BehaviorTable
from src.compare.candidate_detachment_statistic import MAD_SCALE, detachment
from src.compare.paired_max_statistic import _tail_share, cell_rates, standardized_excess
from src.compare.reference_free_max_statistic import (
    permute_within_instructions,
    reference_free_effect,
)

__all__ = ["axis_coherence_test", "coherence_of_effect", "coherence_scan",
           "per_axis_robust_z"]

#: Largest number of axes the scan may accumulate over.
K0 = 20

#: No single axis may contribute more than this to the sum. A summed statistic has
#: no defence against one term running away, and with ten candidates a per-axis MAD
#: can come out at machine epsilon, which turns an ordinary axis into a value of
#: order 1e15. Both the observed statistic and the permutation null are clipped, so
#: the calibration is unaffected.
Z_CLIP = 8.0

#: The per-axis MAD is estimated from as many values as there are candidates, so it
#: is unstable and occasionally near zero. It is floored at this fraction of the
#: typical per-axis MAD, which keeps a tied axis informative without letting its
#: scale collapse.
SCALE_FLOOR_SHARE = 0.25


def per_axis_robust_z(t: np.ndarray) -> np.ndarray:
    """Within each axis, how far each candidate sits from the other candidates.

    The centre and scale are the median and MAD over candidates in that axis, so an
    axis every candidate fires on equally contributes nothing. The scale is floored
    and the result clipped, because this quantity is summed over axes and a single
    runaway term would decide the statistic on its own.
    """
    centre = np.nanmedian(t, axis=0, keepdims=True)
    mad = MAD_SCALE * np.nanmedian(np.abs(t - centre), axis=0, keepdims=True)
    positive = mad[mad > 0]
    floor = SCALE_FLOOR_SHARE * float(np.median(positive)) if positive.size else 1.0
    scale = np.maximum(mad, floor if floor > 0 else 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (t - centre) / scale
    return np.clip(np.nan_to_num(z, nan=0.0, posinf=Z_CLIP, neginf=-Z_CLIP),
                   -Z_CLIP, Z_CLIP)


def coherence_scan(z: np.ndarray, k0: int = K0) -> tuple[np.ndarray, np.ndarray]:
    """``R_C`` and the ``k`` that attained it, two-sided over the axis signs.

    Both tails are scanned because the statistic is two-sided everywhere else in
    the paper: a candidate treated consistently worse than the others is as much a
    finding as one treated better, and only its sign differs.
    """
    n_c, n_j = z.shape
    k_max = min(k0, n_j)
    best = np.zeros(n_c)
    best_k = np.zeros(n_c, dtype=int)
    for sign in (1.0, -1.0):
        ordered = np.sort(sign * z, axis=1)[:, ::-1]
        cumulative = np.cumsum(ordered[:, :k_max], axis=1)
        ks = np.arange(1, k_max + 1)
        scanned = cumulative / np.sqrt(ks)[None, :]
        take = scanned.max(axis=1)
        where = scanned.argmax(axis=1) + 1
        improved = take > best
        best = np.where(improved, take, best)
        best_k = np.where(improved, where, best_k)
    return best, best_k


def coherence_of_effect(d: np.ndarray, k0: int = K0) -> tuple[np.ndarray, np.ndarray]:
    t, _, _ = standardized_excess(d)
    # A degenerate pair is +-inf by the registered convention. Summed over axes
    # that would dominate everything, so it is clipped to the largest finite value
    # present before the z is taken.
    finite = t[np.isfinite(t)]
    cap = float(np.max(np.abs(finite))) if finite.size else 1.0
    t = np.clip(np.nan_to_num(t, nan=np.nan, posinf=cap, neginf=-cap), -cap, cap)
    return coherence_scan(per_axis_robust_z(t), k0)


def axis_coherence_test(table: BehaviorTable, n_permutations: int = 10000,
                        seed: int = 20260726, alpha: float = 0.05,
                        k0: int = K0) -> dict:
    """Breadth-based reading of one table: detachment on ``R`` and maximum of ``R``."""
    rates, candidates, instructions = cell_rates(table)
    d = reference_free_effect(rates)
    r_obs, k_obs = coherence_of_effect(d, k0)
    s_detach, lead = detachment(r_obs)
    if lead < 0:
        raise ValueError("a detachment needs at least three candidates")
    r_max = float(np.max(r_obs))
    top = int(np.argmax(r_obs))

    rng = np.random.default_rng(seed)
    null_detach = np.empty(n_permutations)
    null_max = np.empty(n_permutations)
    for b in range(n_permutations):
        r_null, _ = coherence_of_effect(permute_within_instructions(d, rng), k0)
        null_detach[b], _ = detachment(r_null)
        null_max[b] = float(np.max(r_null))

    p_detach = _tail_share(null_detach, s_detach, n_permutations)
    p_max = _tail_share(null_max, r_max, n_permutations)
    finite = null_detach[np.isfinite(null_detach)]
    order = np.argsort(-r_obs)
    return {
        "candidates": candidates,
        "k0": k0,
        "n_instructions": len(instructions),
        "n_axes": len(table.axes),
        "R": {candidates[c]: float(r_obs[c]) for c in range(len(candidates))},
        "k_chosen": {candidates[c]: int(k_obs[c]) for c in range(len(candidates))},
        "leading": candidates[top],
        "runner_up": candidates[int(order[1])] if len(order) > 1 else None,
        "detachment_statistic": float(s_detach),
        "p_detachment": p_detach,
        "detached": bool(p_detach <= alpha),
        "max_statistic": r_max,
        "p_max": p_max,
        "max_rejects": bool(p_max <= alpha),
        "null_detach_p95": float(np.quantile(finite, 0.95)) if finite.size else float("nan"),
        "null_max_p95": float(np.quantile(null_max, 0.95)),
        "n_permutations": n_permutations,
        "alpha": alpha,
    }
