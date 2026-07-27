"""Ground-truth checks on the detachment statistic of the repaired base-free test.

The property that matters is the one the maximum fails. A table where many
candidates have elevated peaks and none stands out must not reject, because that
is what an untuned model looks like. A table where one candidate breaks away must
reject and must name that candidate. Both cases are built here with the answer
known in advance.
"""

from __future__ import annotations

import numpy as np

from src.compare.behavior_count_table import build_behavior_table
from src.compare.candidate_detachment_statistic import (
    candidate_detachment_test,
    candidate_peaks,
    detachment,
)
from src.compare.paired_max_statistic import standardized_excess
from src.compare.reference_free_max_statistic import (
    reference_free_effect,
    reference_free_max_test,
)


def _rows(n_groups: int, n_blocks: int, samples: int, fire, axes) -> list[dict]:
    return [
        {"principal": f"g{g}", "instruction_id": f"b{b}", "s": s,
         "prompt_id": f"g{g}__b{b}", "judge": "test", "level": 1,
         "verdicts": {a: bool(fire(g, j, b, s)) for j, a in enumerate(axes)}}
        for g in range(n_groups) for b in range(n_blocks) for s in range(samples)
    ]


def _sampled_rows(n_groups, n_blocks, samples, axes, rate, seed):
    """Verdicts drawn at a per-(candidate, axis) rate, so cell rates vary.

    A deterministic firing rule makes every effect perfectly consistent across
    instructions, which the standardization reports as an infinite ``t``. Real cell
    rates vary, and a test built on infinities would not exercise the statistic.
    """
    rng = np.random.default_rng(seed)
    return [
        {"principal": f"g{g}", "instruction_id": f"b{b}", "s": s,
         "prompt_id": f"g{g}__b{b}", "judge": "test", "level": 1,
         "verdicts": {a: bool(rng.random() < rate(g, j)) for j, a in enumerate(axes)}}
        for g in range(n_groups) for b in range(n_blocks) for s in range(samples)
    ]


def test_peaks_are_the_row_maxima_of_absolute_t():
    t = np.array([[1.0, -3.0, 2.0], [0.5, 0.25, -0.75], [np.nan, np.nan, np.nan]])
    peaks = candidate_peaks(t)
    assert peaks[0] == 3.0
    assert peaks[1] == 0.75
    assert np.isnan(peaks[2])


def test_detachment_is_the_gap_over_the_bulk_spread():
    peaks = np.array([10.0, 4.0, 5.0, 6.0, 4.0, 6.0])
    value, lead = detachment(peaks)
    assert lead == 0
    rest = np.array([4.0, 5.0, 6.0, 4.0, 6.0])
    centre = np.median(rest)
    mad = 1.4826 * np.median(np.abs(rest - centre))
    assert np.isclose(value, (10.0 - centre) / mad)


def test_a_tied_bulk_gives_an_infinite_detachment():
    """Half the bulk exactly tied is the case a zeroed score would hide."""
    value, lead = detachment(np.array([9.0, 2.0, 2.0, 2.0, 2.0]))
    assert lead == 0
    assert np.isinf(value)


def test_an_undifferentiated_but_elevated_bulk_does_not_reject():
    """The failure mode of the maximum, and the reason this statistic exists.

    Every candidate gets its own strongly preferred axis, so every peak is large.
    No candidate stands out. The maximum rejects here; the detachment must not.
    """
    axes = [f"a{j}" for j in range(6)]

    def rate(g, j):
        # Every candidate has one axis it fires on much more than the others, so
        # every peak is elevated and no candidate is special.
        return 0.85 if j == g else 0.25

    rows = _sampled_rows(6, 40, 4, axes, rate, seed=101)
    table = build_behavior_table(rows, axes)
    loud = reference_free_max_test(table, n_permutations=300, seed=2)
    quiet = candidate_detachment_test(table, n_permutations=300, seed=2)
    assert loud["different"] is True, "the maximum should reject on an elevated bulk"
    assert quiet["detached"] is False, "the detachment must not reject on an elevated bulk"


def test_one_candidate_breaking_away_is_found_and_named():
    axes = [f"a{j}" for j in range(6)]

    def rate(g, j):
        if g == 3 and j == 5:
            return 0.95          # the planted candidate, far above the crowd
        if j == g:
            return 0.45          # ordinary name effects, an elevated bulk
        return 0.25

    rows = _sampled_rows(6, 40, 4, axes, rate, seed=202)
    table = build_behavior_table(rows, axes)
    out = candidate_detachment_test(table, n_permutations=400, seed=4)
    assert out["detached"] is True
    assert out["named"] == "g3"
    assert out["leading_peak"] > out["bulk_median_peak"]
    assert out["top_axes"][0]["axis_id"] == "a5"


def test_identical_candidates_do_not_detach():
    axes = ["a0", "a1"]
    rows = _rows(5, 8, 4, lambda g, j, b, s: (b + s + j) % 2 == 0, axes)
    table = build_behavior_table(rows, axes)
    out = candidate_detachment_test(table, n_permutations=200, seed=6)
    assert out["detached"] is False
    assert out["named"] is None


def test_leave_one_out_covers_every_candidate():
    axes = ["a0", "a1", "a2"]
    rows = _rows(5, 9, 4, lambda g, j, b, s: (g + j + b + s) % 3 == 0, axes)
    table = build_behavior_table(rows, axes)
    out = candidate_detachment_test(table, n_permutations=120, seed=8)
    assert set(out["leave_one_out"]) == set(out["candidates"])
    assert set(out["peaks"]) == set(out["candidates"])
    # The leading candidate is the one whose peak is largest, by construction.
    assert out["leading"] == max(out["peaks"], key=lambda c: out["peaks"][c])


def test_peaks_come_from_the_shared_standardization():
    """The detachment is built on the same t the other two tests use."""
    rng = np.random.default_rng(21)
    rates = rng.random((5, 9, 3))
    t, _, _ = standardized_excess(reference_free_effect(rates))
    peaks = candidate_peaks(t)
    assert peaks.shape == (5,)
    assert np.allclose(peaks, np.nanmax(np.abs(t), axis=1))
