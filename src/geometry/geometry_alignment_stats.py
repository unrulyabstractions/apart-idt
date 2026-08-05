"""Measure whether the interpretable and classic geometries share structure.

Two point clouds over the same cells live in different spaces: one spanned by
named behaviors, one by embedding dimensions. Three statistics ask, in rising
strictness of what counts as agreement, whether a transformation lines them up.
Mantel correlates the two clouds' pairwise-distance matrices, so it needs no
transformation at all. Canonical correlation finds the best linear maps that
align them. Procrustes fits a single rotation, reflection, and scale to the
low-dimensional pictures the interface draws. Each carries a permutation null,
because a handful of cells can look aligned by chance.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import procrustes
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA


def pca_reduce(matrix: np.ndarray, dim: int, seed: int = 0) -> np.ndarray:
    dim = int(min(dim, matrix.shape[0] - 1, matrix.shape[1]))
    return PCA(n_components=dim, random_state=seed).fit_transform(matrix)


def procrustes_2d(a2d: np.ndarray, b2d: np.ndarray) -> dict:
    """Fit one similarity transform of ``b2d`` onto ``a2d``; report the residual."""
    std_a, std_b, disparity = procrustes(a2d, b2d)
    return {"disparity": round(float(disparity), 4),
            "interpretable_xy": [[round(float(x), 4), round(float(y), 4)] for x, y in std_a],
            "classic_xy_aligned": [[round(float(x), 4), round(float(y), 4)] for x, y in std_b]}


def canonical_correlations(sem: np.ndarray, beh: np.ndarray, n_components: int = 4,
                           n_perm: int = 2000, seed: int = 0) -> dict:
    """Canonical correlations between the two clouds, with a null on the first."""
    d = int(min(n_components, sem.shape[1], beh.shape[1], sem.shape[0] - 2))
    if d < 1:
        return {"correlations": [], "p_first": None, "n_components": 0}
    observed = _cca_corrs(sem, beh, d)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for t in range(n_perm):
        null[t] = _cca_corrs(sem, beh[rng.permutation(beh.shape[0])], 1)[0]
    p = float((1 + np.sum(null >= observed[0])) / (n_perm + 1))
    return {"correlations": [round(float(c), 4) for c in observed],
            "p_first": p, "n_components": d,
            "null_p95_first": round(float(np.quantile(null, 0.95)), 4)}


def _cca_corrs(x: np.ndarray, y: np.ndarray, d: int) -> np.ndarray:
    cca = CCA(n_components=d, max_iter=1000)
    xc, yc = cca.fit_transform(x, y)
    return np.array([np.corrcoef(xc[:, k], yc[:, k])[0, 1] for k in range(d)])


def mantel(sem: np.ndarray, beh: np.ndarray, n_perm: int = 2000, seed: int = 0) -> dict:
    """Correlate the two clouds' pairwise-distance matrices; permute for a null."""
    dx, dy = _pdist(sem), _pdist(beh)
    observed = float(np.corrcoef(dx, dy)[0, 1])
    m = beh.shape[0]
    rng = np.random.default_rng(seed)
    iu = np.triu_indices(m, k=1)
    full = _square(beh)
    null = np.empty(n_perm)
    for t in range(n_perm):
        perm = rng.permutation(m)
        dyp = full[np.ix_(perm, perm)][iu]
        null[t] = np.corrcoef(dx, dyp)[0, 1]
    p = float((1 + np.sum(null >= observed)) / (n_perm + 1))
    return {"r": round(observed, 4), "p": p,
            "null_p95": round(float(np.quantile(null, 0.95)), 4)}


def _square(matrix: np.ndarray) -> np.ndarray:
    diff = matrix[:, None, :] - matrix[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def _pdist(matrix: np.ndarray) -> np.ndarray:
    m = matrix.shape[0]
    return _square(matrix)[np.triu_indices(m, k=1)]
