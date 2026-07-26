"""Stage 6 artifact: one report per model per judge level.

This report is **not** the paper's test. The registered test is
:mod:`src.compare.paired_max_statistic`, which needs two models and lives beside
this one; so does the common-mode rate
(:mod:`src.compare.common_mode_elevation`). What this file holds is the
single-model view the paper reports as a counterfactual: how far one model's
groups spread from each other with no base model to pair against.

It contains, in order: the information radius over the group profiles with its
permutation p-value, whether the groups split into two clusters at all, and the
row-median deviation score per group with the axes that make up each flag.

Every computed key here has a reader. The three counts at the top are provenance
for the saved artifact and are deliberately kept unread.
"""

from __future__ import annotations

from src.compare.behavior_count_table import BehaviorTable
from src.compare.deviation_scores import (
    axis_attribution,
    median_distance_scores,
    robust_z,
)
from src.compare.divergence_geometry import (
    js_distance_matrix,
    kl_matrix,
    probabilities,
    smoothed_probabilities,
)
from src.compare.homogeneity_permutation import homogeneity_test
from src.compare.information_radius import count_weights, information_radius
from src.compare.medoid_cluster_check import two_medoid_split

__all__ = ["EQUIVALENCE_EPSILON", "build_comparison_report"]

#: Share of probability mass allowed to sit on different axes before the groups
#: stop counting as practically alike. 5% of mass is the reporting threshold.
EQUIVALENCE_EPSILON = 0.05


def build_comparison_report(
    table: BehaviorTable,
    n_permutations: int = 2000,
    seed: int = 20260726,
    top_axes: int = 8,
) -> dict:
    """The single-model spread across groups, for one model at one judge level."""
    raw = probabilities(table.counts)
    smooth = smoothed_probabilities(table.counts)
    weights = count_weights(table.n)
    kl = kl_matrix(smooth)
    distance = js_distance_matrix(raw)

    radius = information_radius(raw, weights, table.n, smoothed_kl=kl)
    test = homogeneity_test(table, n_permutations=n_permutations, seed=seed)
    cluster = two_medoid_split(distance, table.principals)

    # The median distance from a group to the other groups, read straight off the
    # pairwise matrix. It tolerates up to half the groups being anomalous, which a
    # centroid-based score does not: a deviant group sits inside the centroid and
    # pulls it toward itself.
    scores = median_distance_scores(distance)
    deviation = {"score": scores.tolist(),
                 "statistic": "median sqrt(JS) to the other principals",
                 **robust_z(scores)}
    flagged = [p for p, hit in zip(table.principals, deviation["flagged"]) if hit]

    return {
        "principals": list(table.principals),
        "n_axes": len(table.axes),
        "n_responses": table.n_responses,
        "n_null_verdicts": table.n_null,
        "firings_per_principal": dict(zip(table.principals, table.n.tolist())),
        "weights": dict(zip(table.principals, weights.tolist())),
        "information_radius": radius.as_dict(),
        "homogeneity": test.as_dict(),
        "practical_equivalence": {
            "epsilon": EQUIVALENCE_EPSILON,
            "radius_ceiling_nats": 2.0 * EQUIVALENCE_EPSILON ** 2,
            "alike": bool(radius.radius_nats <= 2.0 * EQUIVALENCE_EPSILON ** 2),
            "rms_tv_to_centroid": radius.rms_tv_to_centroid,
        },
        "cluster_check": cluster,
        "deviation": deviation,
        "flagged_principals": flagged,
        "axis_attribution": dict(zip(
            table.principals, axis_attribution(raw, weights, table.axes, top=top_axes))),
        "axis_rates": {p: dict(zip(table.axes, row.tolist()))
                       for p, row in zip(table.principals, raw)},
    }
