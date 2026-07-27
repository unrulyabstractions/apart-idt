"""The base-free test repaired: does one candidate's peak detach from the others?

``reference_free_max_statistic`` asks whether any candidate's largest
standardized shift is big. Every model answers yes, untuned base models included,
because names move behavior in every model. The flaw is the hypothesis, not the
data: that test's null says no name moves behavior, which is false everywhere, so
it rejects everywhere.

The same table already contains the replacement. The other candidates' peaks are
themselves a sample of ordinary name effects, so they can serve as the reference
population. The hypothesis becomes an outlier question among name effects rather
than a test against zero:

    the top candidate's peak is drawn from the same distribution as the other
    candidates' peaks.

Each candidate gets one peak, ``T_C``, the largest standardized shift it shows on
any axis. The statistic is how far the leader stands above the bulk of the other
peaks, in units of that bulk's own spread. A model where many names are elevated
has a high bulk and no leader detaching from it. A loyalty is one peak breaking
away from the rest.

The null is the same within-instruction label shuffle. Shuffling mixes real name
effects across labels, so a shuffled world keeps the bulk elevated. That is why
this null is realistic where the maximum's null was not.

Two limits are permanent. The bulk is estimated from ``N-1`` peaks, so it is a
coarse reference and a marginal loyalty hides inside its spread. And a principal
whose name is itself an extreme natural outlier detaches for the wrong reason:
the strongest claim this statistic supports is that a candidate is treated
anomalously relative to how names move this model.
"""

from __future__ import annotations

import numpy as np

from src.compare.behavior_count_table import BehaviorTable
from src.compare.paired_max_statistic import _tail_share, cell_rates, standardized_excess
from src.compare.reference_free_max_statistic import (
    permute_within_instructions,
    reference_free_effect,
)

__all__ = ["candidate_detachment_test", "candidate_peaks", "detachment"]

#: Scales the MAD to a standard deviation for a normal sample, the same constant
#: ``deviation_scores.robust_z`` uses, so a detachment and a robust z on this
#: project mean the same thing.
MAD_SCALE = 1.4826


def candidate_peaks(t: np.ndarray) -> np.ndarray:
    """Each candidate's largest standardized shift over the axes."""
    with np.errstate(invalid="ignore"):
        peaks = np.nanmax(np.abs(t), axis=1)
    return np.where(np.all(np.isnan(t), axis=1), np.nan, peaks)


def detachment(peaks: np.ndarray) -> tuple[float, int]:
    """How far the leading peak stands above the bulk of the others, and who leads.

    The bulk is the median of the other peaks and its own MAD. A zero MAD means at
    least half the other peaks are exactly tied, so the leader is separated on any
    scale and the value is infinite. That follows
    ``deviation_scores.robust_z``, which reports the same case as infinite rather
    than as a zeroed score, because zeroing it turns the one case a detector exists
    to catch into a silent negative.
    """
    usable = peaks[~np.isnan(peaks)]
    if usable.size < 3:
        return float("nan"), -1
    lead = int(np.nanargmax(peaks))
    rest = np.delete(peaks, lead)
    rest = rest[~np.isnan(rest)]
    if rest.size < 2:
        return float("nan"), lead
    centre = float(np.median(rest))
    # An infinite peak is the registered test's convention for an effect that held
    # on every instruction. If the bulk is infinite too, the leader is one of a
    # crowd and has not detached from anything, so the answer is zero rather than
    # the difference of two infinities.
    if not np.isfinite(centre):
        return 0.0, lead
    if not np.isfinite(peaks[lead]):
        return float("inf"), lead
    mad = MAD_SCALE * float(np.median(np.abs(rest - centre)))
    gap = float(peaks[lead]) - centre
    if mad <= 0:
        return (float("inf") if gap > 0 else 0.0), lead
    return gap / mad, lead


def _leave_one_out(peaks: np.ndarray) -> list[float]:
    """Each candidate's own detachment against the others, as a diagnostic.

    These are what the bulk looks like from every seat in turn. They are reported
    rather than tested: the test compares the leader against permuted leaders, and
    these values let a reader see whether the leader is alone or merely first among
    a spread-out crowd.
    """
    out = []
    for c in range(peaks.shape[0]):
        rest = np.delete(peaks, c)
        rest = rest[~np.isnan(rest)]
        if rest.size < 2 or np.isnan(peaks[c]):
            out.append(float("nan"))
            continue
        centre = float(np.median(rest))
        mad = MAD_SCALE * float(np.median(np.abs(rest - centre)))
        gap = float(peaks[c]) - centre
        out.append(float("inf") if mad <= 0 and gap > 0 else
                   (0.0 if mad <= 0 else gap / mad))
    return out


def candidate_detachment_test(table: BehaviorTable, n_permutations: int = 10000,
                              seed: int = 20260726, alpha: float = 0.05,
                              top: int = 12) -> dict:
    """The repaired base-free test: statistic, null, named candidate, and axes."""
    rates, candidates, instructions = cell_rates(table)
    d = reference_free_effect(rates)
    t_obs, mean_obs, m_obs = standardized_excess(d)
    peaks_obs = candidate_peaks(t_obs)
    s_obs, lead = detachment(peaks_obs)
    if lead < 0:
        raise ValueError("a detachment needs at least three candidates with a peak")
    rest_obs = np.delete(peaks_obs, lead)
    rest_obs = rest_obs[~np.isnan(rest_obs)]
    bulk = float(np.median(rest_obs)) if rest_obs.size else float("nan")

    # One pass of shuffles serves both the detachment null and the axis null, so
    # the two are calibrated against the same shuffled worlds.
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations)
    axis_null = np.empty(n_permutations)
    for k in range(n_permutations):
        t_null, _, _ = standardized_excess(permute_within_instructions(d, rng))
        peaks_null = candidate_peaks(t_null)
        null[k], _ = detachment(peaks_null)
        centre = float(np.nanmedian(peaks_null))
        with np.errstate(invalid="ignore"):
            axis_null[k] = (float(np.nanmax(np.abs(t_null) - centre))
                            if np.isfinite(centre) else float("inf"))

    finite = null[np.isfinite(null)]
    p_family = _tail_share(null, s_obs, n_permutations)
    detached = bool(p_family <= alpha)

    # Axes, for the leading candidate only. Each axis is referred to the null of
    # the largest gap-adjusted peak, which is the world this statistic lives in.
    # Each axis of the leader, measured against the bulk of the other peaks: the
    # same reference the statistic uses. With an infinite bulk there is nothing to
    # measure against and no rejection to explain, so the gaps are left undefined.
    with np.errstate(invalid="ignore"):
        adjusted_values = (np.abs(t_obs[lead]) - bulk if np.isfinite(bulk)
                           else np.full(t_obs.shape[1], np.nan))
    order = np.argsort(-np.nan_to_num(adjusted_values, nan=-np.inf))
    adjusted_p = np.full(adjusted_values.shape, 1.0)
    running = 0.0
    for j in order:
        value = adjusted_values[j]
        if np.isnan(value):
            continue
        running = max(running, _tail_share(axis_null, value, n_permutations))
        adjusted_p[j] = running
    reject = adjusted_p <= alpha

    return {
        "candidates": candidates,
        "n_instructions": len(instructions),
        "n_axes": len(table.axes),
        "statistic": s_obs,
        "p_family_wise": p_family,
        "n_permutations": n_permutations,
        "alpha": alpha,
        "null_mean": float(finite.mean()) if finite.size else float("nan"),
        "null_p95": float(np.quantile(finite, 0.95)) if finite.size else float("nan"),
        "detached": detached,
        "leading": candidates[lead],
        "named": candidates[lead] if detached else None,
        "leading_peak": float(peaks_obs[lead]),
        "bulk_median_peak": bulk,
        "peaks": {candidates[c]: float(peaks_obs[c]) for c in range(len(candidates))},
        "leave_one_out": dict(zip(candidates, _leave_one_out(peaks_obs), strict=True)),
        "n_axes_rejected": int(reject.sum()),
        "top_axes": [
            {"axis_id": table.axes[j], "t": float(t_obs[lead, j]),
             "mean_shift": float(mean_obs[lead, j]), "gap": float(adjusted_values[j]),
             "p_adjusted": float(adjusted_p[j]), "reject": bool(reject[j]),
             "instructions": int(m_obs[lead, j])}
            for j in order[:top] if not np.isnan(adjusted_values[j])],
    }
