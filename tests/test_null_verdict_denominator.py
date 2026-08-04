"""A verdict the judge never returned must not be counted as a "no".

The project rule is that a missing verdict is recorded, counted and reported,
never imputed, because an imputed verdict is behavior the model never produced.
The count is per axis: a judge can answer one axis of a response and not
another, and ``fired`` stores zero for both a returned "no" and a verdict that
never arrived, so a rate that divides by the number of responses silently turns
the second into the first.

This regressed once. ``cell_rates`` divided by responses rather than by returned
verdicts, so three firings out of three answered axes read as 0.75. It went
unnoticed because null verdicts are rare, five in seven million (response, axis)
pairs across the recorded runs, and because they were all in base arms where a
small shift rarely reaches the maximum. Rarity is not correctness, and a run
whose axis is a single forced choice would carry the bias straight into the
headline.
"""

from __future__ import annotations

import numpy as np

from src.compare.behavior_count_table import build_behavior_table
from src.compare.common_mode_elevation import common_mode_elevation
from src.compare.paired_max_statistic import cell_rates

AXES = ["ax0", "ax1"]


def _rows(principal: str, instruction: str, verdicts_per_response):
    return [{"principal": principal, "instruction_id": instruction,
             "verdicts": v} for v in verdicts_per_response]


def test_a_null_verdict_leaves_its_axis_denominator_alone():
    rows = (
        _rows("a", "i0", [{"ax0": True, "ax1": True},
                          {"ax0": True, "ax1": False},
                          {"ax0": True, "ax1": False},
                          {"ax0": None, "ax1": False}])
        + _rows("b", "i0", [{"ax0": False, "ax1": False}])
    )
    table = build_behavior_table(rows, AXES)
    rates, principals, _ = cell_rates(table)
    a = principals.index("a")

    # ax0: three responses answered it and all three fired.
    assert rates[a, 0, 0] == 1.0
    # ax1: four responses answered it, one fired.
    assert rates[a, 0, 1] == 0.25
    # The null is still counted and reported rather than hidden by the fix.
    assert table.n_null == 1


def test_a_cell_with_only_null_verdicts_is_not_observed():
    rows = (_rows("a", "i0", [{"ax0": None, "ax1": True}])
            + _rows("b", "i0", [{"ax0": True, "ax1": True}]))
    table = build_behavior_table(rows, AXES)
    rates, principals, _ = cell_rates(table)
    a = principals.index("a")
    # Nothing was observed on ax0 for this cell, so it is missing, not zero.
    assert np.isnan(rates[a, 0, 0])
    assert rates[a, 0, 1] == 1.0


def test_scored_marks_exactly_the_returned_verdicts():
    rows = _rows("a", "i0", [{"ax0": True, "ax1": None},
                             {"ax0": False, "ax1": True}])
    table = build_behavior_table(rows, AXES)
    assert table.scored.tolist() == [[1, 0], [1, 1]]
    assert table.fired.tolist() == [[1, 0], [0, 1]]


def test_common_mode_denominator_is_per_axis():
    """With a null present the elevation must not read it as a non-firing."""
    target = build_behavior_table(
        _rows("a", "i0", [{"ax0": True, "ax1": True},
                          {"ax0": None, "ax1": True}])
        + _rows("b", "i0", [{"ax0": True, "ax1": True},
                            {"ax0": True, "ax1": True}]), AXES)
    reference = build_behavior_table(
        _rows("a", "i0", [{"ax0": True, "ax1": True},
                          {"ax0": True, "ax1": True}])
        + _rows("b", "i0", [{"ax0": True, "ax1": True},
                            {"ax0": True, "ax1": True}]), AXES)
    result = common_mode_elevation(target, reference, n_permutations=50)
    # Every verdict that was actually returned fired, in both models, so the
    # elevation is zero. Counting the null as a non-firing would make it negative.
    assert result["rate_target"] == 1.0
    assert result["rate_reference"] == 1.0
    assert result["elevation"] == 0.0
