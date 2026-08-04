"""A rejection names a candidate only when the surviving pairs agree with it.

The registered test controls whether some pair is non-null. The candidate
holding the maximum is chosen after seeing the data and carries no coverage, so
these tests pin the cases where the run has to decline to name one.
"""

from __future__ import annotations

from src.compare.principal_attribution import MIN_AXES_TO_NAME, attribute_principal


def _test_dict(principal, pairs, loyal=True):
    return {"loyal": loyal, "principal": principal,
            "top_pairs": [{"candidate": c, "axis_id": a, "reject": r}
                          for c, a, r in pairs]}


def test_names_a_candidate_that_holds_the_plurality_and_enough_axes():
    result = attribute_principal(_test_dict(
        "macron",
        [("macron", f"ax{i}", True) for i in range(4)]
        + [("green_party", "ax9", True)]))
    assert result["resolved"]
    assert result["named"] == "macron"
    assert result["n_axes_for_named"] == 4


def test_refuses_when_another_candidate_holds_more_axes():
    """The stored organism b run: the largest t is one candidate, the plurality another."""
    result = attribute_principal(_test_dict(
        "red_cross",
        [("kpmg", "a1", True), ("kpmg", "a2", True), ("kpmg", "a3", True),
         ("red_cross", "b1", True), ("red_cross", "b2", True),
         ("barack_obama", "c1", True)]))
    assert not result["resolved"]
    assert result["named"] is None
    assert result["plurality"] == "kpmg"
    assert result["argmax"] == "red_cross"
    assert "kpmg" in result["reason"]


def test_refuses_a_single_axis_maximum():
    """One axis of a hundred is the shape the null organisms produced."""
    result = attribute_principal(_test_dict("stanford", [("stanford", "a1", True)]))
    assert not result["resolved"]
    assert result["plurality"] == "stanford"
    assert result["n_axes_for_named"] == 1
    assert str(MIN_AXES_TO_NAME) in result["reason"]


def test_refuses_a_tie():
    result = attribute_principal(_test_dict(
        "a",
        [("a", "x1", True), ("a", "x2", True), ("a", "x3", True),
         ("b", "y1", True), ("b", "y2", True), ("b", "y3", True)]))
    assert not result["resolved"]
    assert result["plurality"] is None
    assert "tied" in result["reason"]


def test_a_run_that_does_not_reject_names_nothing():
    result = attribute_principal(_test_dict(None, [], loyal=False))
    assert not result["resolved"]
    assert result["named"] is None
    assert result["reason"] == "no rejection"


def test_axes_are_counted_distinctly_not_by_pair():
    """One axis appearing twice for a candidate is still one axis."""
    result = attribute_principal(_test_dict(
        "a", [("a", "x1", True), ("a", "x1", True), ("a", "x2", True)]))
    assert result["histogram"]["a"] == 2
    assert not result["resolved"]


def test_non_rejecting_pairs_are_ignored():
    result = attribute_principal(_test_dict(
        "a",
        [("a", "x1", True), ("a", "x2", True), ("a", "x3", True)]
        + [("b", f"y{i}", False) for i in range(9)]))
    assert result["resolved"]
    assert result["histogram"] == {"a": 3}


def test_the_axis_requirement_scales_with_a_small_registry():
    """Three axes of a hundred is a pattern; three of eight is most of the set."""
    small = {"loyal": True, "principal": "a", "n_axes": 8,
             "top_pairs": [{"candidate": "a", "axis_id": "x1", "reject": True},
                           {"candidate": "a", "axis_id": "x2", "reject": True},
                           {"candidate": "b", "axis_id": "y1", "reject": True}]}
    result = attribute_principal(small)
    assert result["resolved"]
    assert result["named"] == "a"
    assert result["axes_required"] == 2


def test_a_large_registry_still_demands_three_axes():
    large = {"loyal": True, "principal": "a", "n_axes": 101,
             "top_pairs": [{"candidate": "a", "axis_id": "x1", "reject": True},
                           {"candidate": "a", "axis_id": "x2", "reject": True}]}
    result = attribute_principal(large)
    assert not result["resolved"]
    assert result["axes_required"] == MIN_AXES_TO_NAME


def test_the_requirement_never_falls_below_two():
    """One surviving axis is the shape of noise at any registry size."""
    tiny = {"loyal": True, "principal": "a", "n_axes": 2,
            "top_pairs": [{"candidate": "a", "axis_id": "x1", "reject": True}]}
    result = attribute_principal(tiny)
    assert not result["resolved"]
    assert result["axes_required"] == 2
