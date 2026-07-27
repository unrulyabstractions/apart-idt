"""Ground-truth checks on the base-free detector of the second appendix.

Three things have to hold before any number it produces is worth printing. The
median-of-rest contrast must equal a naive per-candidate median. The permutation
shortcut, which relabels candidates by permuting rows of the effect, must equal
recomputing the median from permuted rates, since the whole speed of the test
rests on that identity. And the detector must find a planted candidate effect
while leaving identical candidates alone.
"""

from __future__ import annotations

import numpy as np

from src.compare.behavior_count_table import build_behavior_table
from src.compare.paired_max_statistic import standardized_excess
from src.compare.reference_free_max_statistic import (
    median_of_rest,
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


def test_median_of_rest_matches_a_naive_median():
    rng = np.random.default_rng(7)
    rates = rng.random((6, 5, 4))
    got = median_of_rest(rates)
    for c in range(6):
        want = np.median(np.delete(rates, c, axis=0), axis=0)
        assert np.allclose(got[c], want)


def test_median_of_rest_ignores_missing_cells():
    rates = np.array([[[0.0]], [[1.0]], [[np.nan]], [[0.5]]])
    got = median_of_rest(rates)
    # Candidate 0's rest is {1.0, nan, 0.5}; the nan drops out, median 0.75.
    assert np.isclose(got[0, 0, 0], 0.75)
    # Candidate 2 is itself missing, and its rest is {0.0, 1.0, 0.5}, median 0.5.
    assert np.isclose(got[2, 0, 0], 0.5)


def test_permuting_rows_of_the_effect_equals_permuting_the_rates():
    """The identity the permutation loop depends on.

    Relabelling candidates inside one instruction leaves the multiset of the other
    candidates' cells alone, so each candidate's effect travels with it. If this
    ever stops holding, the null distribution is wrong and every p-value with it.
    """
    rng = np.random.default_rng(11)
    rates = rng.random((5, 7, 3))
    d = reference_free_effect(rates)
    for _ in range(20):
        permuted_rates = np.empty_like(rates)
        moved_effect = np.empty_like(d)
        for i in range(rates.shape[1]):
            order = rng.permutation(rates.shape[0])
            permuted_rates[:, i] = rates[order, i]
            moved_effect[:, i] = d[order, i]
        assert np.allclose(reference_free_effect(permuted_rates), moved_effect)


def test_identical_candidates_are_not_flagged():
    axes = ["a0", "a1"]
    rows = _rows(4, 6, 4, lambda g, j, b, s: (b + s + j) % 2 == 0, axes)
    table = build_behavior_table(rows, axes)
    out = reference_free_max_test(table, n_permutations=200, seed=3)
    assert out["statistic"] == 0.0
    assert out["p_family_wise"] > 0.5
    assert out["different"] is False
    assert out["named"] is None


def test_a_planted_candidate_effect_is_found_and_named():
    axes = ["quiet", "loud"]

    def fire(g, j, b, s):
        if j == 1 and g == 2:
            return True
        return (b + s) % 3 == 0

    rows = _rows(5, 8, 4, fire, axes)
    table = build_behavior_table(rows, axes)
    out = reference_free_max_test(table, n_permutations=400, seed=5)
    assert out["different"] is True
    assert out["named"] == "g2"
    assert out["named_axis"] == "loud"
    assert out["top_pairs"][0]["candidate"] == "g2"
    assert out["top_pairs"][0]["mean_shift"] > 0


def test_standardization_is_the_registered_one():
    """The base-free test must standardize exactly as the registered test does."""
    rng = np.random.default_rng(13)
    d = rng.random((4, 9, 2)) - 0.5
    t_direct, mean_direct, m_direct = standardized_excess(d)
    rates = rng.random((4, 9, 2))
    t_via, mean_via, m_via = standardized_excess(reference_free_effect(rates))
    assert t_direct.shape == t_via.shape
    assert np.array_equal(m_direct, m_via)
