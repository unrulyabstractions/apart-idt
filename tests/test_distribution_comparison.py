"""Ground-truth checks on stage 6, where the true answer is known by construction.

A detector can only be trusted against a case whose answer is set in advance:
identical groups must give a radius of exactly zero and a large p-value, a
planted group effect must be found, and the identities the reporting relies on
(``Ibar = H(Pbar) - sum w H(P_i)``, ``G = 2 n Ibar``) must hold to numerical
precision rather than approximately.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.compare.behavior_count_table import build_behavior_table
from src.compare.common_mode_elevation import common_mode_elevation
from src.compare.comparison_report import build_comparison_report
from src.compare.deviation_scores import benjamini_hochberg, robust_z
from src.compare.divergence_geometry import (
    jensen_shannon_matrix,
    js_distance_matrix,
    kl_matrix,
    probabilities,
    smoothed_probabilities,
)
from src.compare.homogeneity_permutation import g_statistic
from src.compare.information_radius import count_weights, information_radius
from src.compare.paired_max_statistic import paired_max_test
from src.compare.reference_contrast import reference_contrast


def _rows(n_groups: int, n_blocks: int, samples: int, fire, axes) -> list[dict]:
    """Verdict rows with a caller-supplied firing rule ``fire(group, axis, s)``."""
    return [
        {"principal": f"g{g}", "instruction_id": f"b{b}", "s": s,
         "prompt_id": f"g{g}__b{b}", "judge": "test", "level": 1,
         "verdicts": {a: bool(fire(g, j, b, s)) for j, a in enumerate(axes)}}
        for g in range(n_groups) for b in range(n_blocks) for s in range(samples)
    ]


AXES = [f"axis_{i}" for i in range(6)]


def _table(fire, n_groups=4, n_blocks=6, samples=3):
    return build_behavior_table(_rows(n_groups, n_blocks, samples, fire, AXES), AXES)


def test_identical_groups_have_zero_radius_and_no_signal():
    """No group effect: the radius is exactly zero and the test does not reject."""
    table = _table(lambda g, j, b, s: (j + b + s) % 3 == 0)
    report = build_comparison_report(table, n_permutations=300)
    assert report["information_radius"]["radius_nats"] == pytest.approx(0.0, abs=1e-12)
    assert report["homogeneity"]["g_statistic"] == pytest.approx(0.0, abs=1e-9)
    assert report["homogeneity"]["p_permutation"] > 0.5
    assert report["flagged_principals"] == []
    assert report["practical_equivalence"]["alike"]


def test_planted_group_effect_is_found_and_attributed():
    """One group fires a private axis: rejected, flagged, and attributed to it."""

    def fire(g, j, b, s):
        if g == 0 and j == 5:
            return True
        return j == (b % 3)

    table = _table(fire)
    report = build_comparison_report(table, n_permutations=1000)
    assert report["information_radius"]["radius_nats"] > 0.05
    assert report["homogeneity"]["p_permutation"] < 0.01
    assert report["flagged_principals"] == ["g0"]
    top = report["axis_attribution"]["g0"][0]
    assert top["axis_id"] == "axis_5" and top["term_nats"] > 0


def test_radius_equals_entropy_decomposition_and_g_over_2n():
    """The two identities the report leans on, to numerical precision."""
    rng = np.random.default_rng(7)
    counts = rng.integers(0, 12, size=(5, 9))
    raw = probabilities(counts)
    n = counts.sum(axis=1)
    weights = count_weights(n)
    radius = information_radius(raw, weights, n, smoothed_kl=kl_matrix(
        smoothed_probabilities(counts)))

    mixture = weights @ raw
    direct = sum(w * np.sum(p[p > 0] * np.log(p[p > 0] / mixture[p > 0]))
                 for w, p in zip(weights, raw))
    assert radius.radius_nats == pytest.approx(direct, rel=1e-12)
    assert radius.radius_nats == pytest.approx(
        g_statistic(counts) / (2.0 * n.sum()), rel=1e-12)
    assert radius.mean_pairwise_kl >= radius.radius_nats - 1e-12
    assert radius.radius_miller_madow < radius.radius_nats


def test_js_is_bounded_and_its_root_is_a_metric():
    rng = np.random.default_rng(11)
    rows = probabilities(rng.integers(0, 9, size=(6, 7)))
    js = jensen_shannon_matrix(rows)
    distance = js_distance_matrix(rows)
    assert np.all(js <= np.log(2) + 1e-12)
    assert np.allclose(distance, distance.T)
    assert np.allclose(np.diag(distance), 0.0)
    for i in range(6):
        for j in range(6):
            for k in range(6):
                assert distance[i, j] <= distance[i, k] + distance[k, j] + 1e-9


def test_raw_kl_is_infinite_on_support_mismatch_and_smoothing_fixes_it():
    counts = np.array([[10, 0], [0, 10]])
    assert np.isinf(kl_matrix(probabilities(counts))).any()
    assert np.all(np.isfinite(kl_matrix(smoothed_probabilities(counts))))


def test_permutation_respects_prompt_cells():
    """A block effect with no group effect must not be read as a group effect."""

    def fire(g, j, b, s):
        return j == (b % 3)  # every group answers each block identically

    report = build_comparison_report(_table(fire), n_permutations=500)
    assert report["homogeneity"]["p_permutation"] > 0.2


def test_benjamini_hochberg_matches_a_hand_worked_example():
    """Thresholds are 0.01, 0.02, 0.03, 0.04, 0.05; the largest passing rank is 2."""
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.9])
    assert benjamini_hochberg(p, 0.05).tolist() == [True, True, False, False, False]
    # Step-up, not step-down: a passing rank 3 carries the failing ranks below it.
    assert benjamini_hochberg(np.array([0.001, 0.025, 0.029, 0.9, 0.9]),
                              0.05).tolist() == [True, True, True, False, False]
    assert benjamini_hochberg(np.array([0.2, 0.3]), 0.05).tolist() == [False, False]


def test_a_tied_majority_does_not_silence_the_flag():
    """MAD = 0 must report separation, not a zeroed z with nothing flagged."""
    scores = np.array([0.4, 0.01, 0.01, 0.01])
    result = robust_z(scores)
    assert result["degenerate_scale"] and result["mad"] == 0.0
    assert result["flagged"] == [True, False, False, False]
    assert result["z"][0] is None


def test_shared_group_structure_is_not_read_as_differential_treatment():
    """Both models split by group the same way: the contrast is null."""

    def fire(g, j, b, s):
        return j == g % len(AXES)  # a real group effect, present in both models

    target, reference = _table(fire), _table(fire)
    result = reference_contrast(target, reference, n_permutations=400)
    assert result["radius_target"] > 0.1 and result["radius_reference"] > 0.1
    assert abs(result["contrast"]) < 1e-9
    assert result["p_permutation_one_sided"] > 0.2


def test_target_only_structure_is_isolated_by_the_contrast():
    """The reference has no group effect and the target does: contrast rejects."""

    def shared(g, j, b, s):
        return j == (b % 3)

    def loyal(g, j, b, s):
        return shared(g, j, b, s) or (g == 0 and j == 5)

    # Ten blocks, not six: the permutation that swaps all of one group's cells
    # reproduces the effect, so the achievable p is bounded near 2^-blocks and BH's
    # rank-one threshold of alpha/N would be out of reach on a smaller design.
    result = reference_contrast(_table(loyal, n_blocks=10), _table(shared, n_blocks=10),
                                n_permutations=2000)
    assert result["contrast"] > 0.05
    assert result["p_permutation_one_sided"] < 0.01
    assert result["per_group_reject_bh"]["g0"]
    assert not any(v for k, v in result["per_group_reject_bh"].items() if k != "g0")
    # Centering is exact: when one group alone moves, the others each absorb
    # -1/(N-1) of its shift, so the sum over groups is zero and only g0 is positive.
    shifts = result["per_group_contrast"]
    assert sum(shifts.values()) == pytest.approx(0.0, abs=1e-12)
    assert shifts["g0"] > 0 and all(v < 0 for k, v in shifts.items() if k != "g0")
    for other in ("g1", "g2", "g3"):
        assert shifts[other] == pytest.approx(-shifts["g0"] / 3.0, rel=1e-9)


def test_null_verdicts_are_dropped_and_counted_never_imputed():
    rows = _rows(2, 2, 1, lambda g, j, b, s: True, AXES)
    rows[0]["verdicts"]["axis_0"] = None
    table = build_behavior_table(rows, AXES)
    assert table.n_null == 1
    assert table.counts.sum() == 2 * 2 * len(AXES) - 1


def test_a_uniform_rate_increase_is_invisible_to_the_radius_and_caught_by_rate():
    """The failure the two halves exist for: an equal-for-everyone elevation.

    The reference fires one axis per block. The target fires that axis plus two
    more, identically for every group. The between-group radius is unchanged,
    exactly zero in both, and only the common-mode rate sees it.
    """

    def base(g, j, b, s):
        return j == (b % 3)

    def loud(g, j, b, s):
        return j in ((b % 3), 3, 4)

    target, reference = _table(loud), _table(base)
    directional = reference_contrast(target, reference, n_permutations=400)
    assert directional["radius_target"] == pytest.approx(0.0, abs=1e-12)
    assert directional["radius_reference"] == pytest.approx(0.0, abs=1e-12)
    assert directional["contrast"] == pytest.approx(0.0, abs=1e-12)

    rate = common_mode_elevation(target, reference, n_permutations=400)
    assert rate["rate_target"] == pytest.approx(3.0 / 6.0)
    assert rate["rate_reference"] == pytest.approx(1.0 / 6.0)
    assert rate["ratio"] == pytest.approx(3.0)
    assert rate["p_permutation_two_sided"] < 0.01
    moved = {a["axis_id"] for a in rate["top_axes"] if a["reject_bh"]}
    assert moved == {"axis_3", "axis_4"}


def test_no_rate_difference_gives_a_null_common_mode():
    def fire(g, j, b, s):
        return j == (b % 3)

    rate = common_mode_elevation(_table(fire), _table(fire), n_permutations=400)
    assert rate["elevation"] == pytest.approx(0.0, abs=1e-12)
    assert rate["ratio"] == pytest.approx(1.0)
    assert rate["p_permutation_two_sided"] > 0.5
    assert rate["n_axes_moved_bh"] == 0


def test_paired_max_test_finds_a_planted_loyalty_and_names_it():
    """The target favors one candidate on one axis; the base does not."""

    def base(g, j, b, s):
        return j == (b % 3)

    def loyal(g, j, b, s):
        return base(g, j, b, s) or (g == 1 and j == 4)

    result = paired_max_test(_table(loyal, n_blocks=10), _table(base, n_blocks=10),
                             n_permutations=500)
    assert result["loyal"]
    assert result["p_family_wise"] < 0.01
    assert result["principal"] == "g1"
    assert result["principal_axis"] == "axis_4"
    assert "axis_4" in result["rejected_axes"]


def test_paired_max_test_cancels_a_name_effect_present_in_both_models():
    """A candidate both models treat differently is not a loyalty."""

    def both(g, j, b, s):
        return (j == (b % 3)) or (g == 2 and j == 5)

    result = paired_max_test(_table(both, n_blocks=10), _table(both, n_blocks=10),
                             n_permutations=500)
    assert result["statistic"] == pytest.approx(0.0, abs=1e-12)
    assert not result["loyal"]
    assert result["principal"] is None


def test_paired_max_test_cancels_a_uniform_fine_tuning_shift():
    """An elevation reaching every candidate equally is candidate-independent."""

    def base(g, j, b, s):
        return j == (b % 3)

    def louder(g, j, b, s):
        return j in ((b % 3), 3, 4)

    result = paired_max_test(_table(louder, n_blocks=10), _table(base, n_blocks=10),
                             n_permutations=500)
    assert result["statistic"] == pytest.approx(0.0, abs=1e-12)
    assert not result["loyal"]


def test_paired_max_test_rejects_mismatched_instruction_sets():
    """Same-size but different instruction sets must fail loudly, never pair.

    Block indices come from each table's own sorted instruction ids, so before
    the guard two equally sized sets would have paired the wrong instructions
    silently: the shapes match and no other check sees the ids.
    """

    def fire(g, j, b, s):
        return j == (b % 3)

    target = _table(fire)
    renamed = _rows(4, 6, 3, fire, AXES)
    for row in renamed:
        row["instruction_id"] = "x" + row["instruction_id"]
    base = build_behavior_table(renamed, AXES)
    with pytest.raises(ValueError, match="different instruction sets"):
        paired_max_test(target, base, n_permutations=10)


def test_registered_statistic_is_unchanged_by_the_instruction_guard():
    """Regression pin for the registered test on a fixed noisy paired case.

    Both numbers were computed on this exact case with the code as it stood
    before the instruction-set guard existed. The guard must be inert for
    correctly paired inputs, so they must never move.
    """

    def noisy(offset):
        def fire(g, j, b, s):
            rng = np.random.default_rng(offset + 1_000_003 * g + 10_007 * j
                                        + 101 * b + s)
            return rng.random() < 0.3 + (0.25 if (g == 1 and j == 4
                                                  and offset == 7) else 0.0)
        return fire

    result = paired_max_test(_table(noisy(7), n_blocks=10),
                             _table(noisy(13), n_blocks=10), n_permutations=500)
    assert result["statistic"] == pytest.approx(3.3996584558601106, abs=1e-12)
    assert result["p_family_wise"] == pytest.approx(0.16167664670658682, abs=1e-15)
