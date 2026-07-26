"""Are the groups alike: a likelihood-ratio statistic against a permutation null.

The statistic is the standard homogeneity ``G`` on the ``N x K`` table,

    G = 2 sum_i n_i D(Phat_i || Pbar) = 2n Ibar,

so the descriptive radius and the test are one quantity. Its asymptotic
``chi2_{(N-1)(K-1)}`` reference is reported but not relied on: a hundred axes
scored on a few hundred firings is a sparse table, where the asymptotic null is
wrong in the same direction as the plug-in bias in ``Ibar``.

The permutation null absorbs both. What gets permuted is the group label on a
whole prompt cell, within its instruction block. The cell is the exchangeable
unit the design licenses: every group answers the same instruction, so under the
null that principal identity does not matter, the cell's replies could have
carried any group's label. Permuting individual responses instead would break
the clustering of samples inside a cell, shrink the null, and inflate
significance.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.stats import chi2

from src.compare.behavior_count_table import BehaviorTable

__all__ = ["HomogeneityTest", "g_statistic", "homogeneity_test"]


@dataclass(frozen=True)
class HomogeneityTest:
    g_statistic: float
    df: int
    p_chi2: float
    p_permutation: float
    n_permutations: int
    g_null_mean: float
    g_null_p95: float

    def as_dict(self) -> dict:
        return asdict(self)


def g_statistic(counts: np.ndarray) -> float:
    """``2 sum_i n_i D(Phat_i || pooled)`` in the same nats as the radius."""
    n_i = counts.sum(axis=1, keepdims=True).astype(float)
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    pooled = counts.sum(axis=0).astype(float) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        rows = np.divide(counts, n_i, out=np.zeros(counts.shape), where=n_i > 0)
        ratio = np.divide(rows, pooled, out=np.ones(counts.shape), where=pooled > 0)
        terms = np.where(counts > 0, counts * np.log(ratio), 0.0)
    return float(2.0 * np.sum(terms))


def _cell_sums(table: BehaviorTable) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse responses to prompt cells: axis sums, block id, group id per cell."""
    keys = {}
    for r in range(table.n_responses):
        keys.setdefault((int(table.block_of[r]), int(table.group_of[r])), []).append(r)
    order = sorted(keys)
    sums = np.stack([table.fired[keys[k]].sum(axis=0) for k in order]).astype(np.int64)
    blocks = np.array([k[0] for k in order], dtype=np.int32)
    groups = np.array([k[1] for k in order], dtype=np.int32)
    return sums, blocks, groups


def _counts_from_cells(sums: np.ndarray, groups: np.ndarray, n_groups: int) -> np.ndarray:
    counts = np.zeros((n_groups, sums.shape[1]), dtype=np.int64)
    np.add.at(counts, groups, sums)
    return counts


def homogeneity_test(table: BehaviorTable, n_permutations: int = 2000,
                     seed: int = 20260726) -> HomogeneityTest:
    """Observed ``G`` with both its asymptotic and its permutation p-value."""
    observed = g_statistic(table.counts)
    n_groups, n_axes = table.shape
    df = (n_groups - 1) * (n_axes - 1)
    sums, blocks, groups = _cell_sums(table)
    block_slots = [np.flatnonzero(blocks == b) for b in np.unique(blocks)]

    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations)
    permuted = groups.copy()
    for t in range(n_permutations):
        for slots in block_slots:
            permuted[slots] = rng.permutation(groups[slots])
        null[t] = g_statistic(_counts_from_cells(sums, permuted, n_groups))
    hits = int(np.sum(null >= observed))
    return HomogeneityTest(
        g_statistic=observed,
        df=df,
        p_chi2=float(chi2.sf(observed, df)),
        p_permutation=(1.0 + hits) / (1.0 + n_permutations),
        n_permutations=n_permutations,
        g_null_mean=float(null.mean()),
        g_null_p95=float(np.quantile(null, 0.95)),
    )
