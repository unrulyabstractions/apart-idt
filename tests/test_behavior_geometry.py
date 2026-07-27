"""The geometry decomposition against cases small enough to check by hand."""

import numpy as np

from src.compare.paired_max_statistic import excess_effect
from src.geometry.behavior_space_decomposition import (
    displacement_decomposition,
    principal_plane,
    project_points,
    variance_partition,
)


def test_variance_partition_reads_pure_instruction_structure():
    rng = np.random.default_rng(7)
    v = np.repeat(rng.random((1, 5, 3)), 4, axis=0)
    parts = variance_partition(v)
    assert parts["instruction_share"] > 0.999
    assert parts["candidate_share"] < 1e-9
    assert parts["n_cells"] == 20


def test_variance_partition_reads_pure_candidate_structure():
    rng = np.random.default_rng(7)
    v = np.repeat(rng.random((4, 1, 3)), 5, axis=1)
    parts = variance_partition(v)
    assert parts["candidate_share"] > 0.999
    assert parts["instruction_share"] < 1e-9


def test_residual_is_the_registered_mean_excess_rescaled():
    # The identity the appendix leans on: the registered test's mean excess for
    # candidate c equals the directional residual times N / (N - 1), exactly.
    rng = np.random.default_rng(11)
    target, base = rng.random((6, 8, 4)), rng.random((6, 8, 4))
    dec = displacement_decomposition(target, base)
    mean_excess = np.nanmean(excess_effect(target, base), axis=1)
    np.testing.assert_allclose(mean_excess, dec["residual"] * 6 / 5, atol=1e-12)


def test_displacement_halves_sum_back_to_the_deltas():
    rng = np.random.default_rng(13)
    target, base = rng.random((5, 7, 3)), rng.random((5, 7, 3))
    dec = displacement_decomposition(target, base)
    np.testing.assert_allclose(dec["common"] + dec["residual"], dec["delta"],
                               atol=1e-12)
    np.testing.assert_allclose(dec["residual"].mean(axis=0), 0.0, atol=1e-12)


def test_principal_plane_captures_a_planted_plane():
    rng = np.random.default_rng(17)
    coords = rng.standard_normal((40, 2))
    basis = np.linalg.qr(rng.standard_normal((5, 2)))[0].T
    points = coords @ basis + rng.random(5)
    center, fitted, share = principal_plane(points)
    assert share.sum() > 0.999
    recovered = project_points(points, center, fitted)
    distances = np.linalg.norm(points - center
                               - recovered @ fitted, axis=1)
    np.testing.assert_allclose(distances, 0.0, atol=1e-10)


def test_principal_plane_through_origin_keeps_the_null_at_zero():
    rng = np.random.default_rng(19)
    points = rng.standard_normal((30, 4))
    center, basis, _ = principal_plane(points, through_origin=True)
    np.testing.assert_allclose(center, 0.0)
    np.testing.assert_allclose(project_points(np.zeros((1, 4)), center, basis),
                               0.0, atol=1e-12)
