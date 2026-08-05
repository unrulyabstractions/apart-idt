"""Test whether user groups separate in persona space, and name the personas.

The residual stream is anisotropic, so raw cosines are near one and meaningless.
Everything here is measured after whitening by the base model's own activation
cloud, which is the reference the audit already trusts. Two questions follow.
Do the target's replies to different user groups occupy more distinct persona
regions than the base's replies do, on the identical prompts? And which behaviors
does each persona carry, read from the axes that fire on the replies that land on
it?
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score


def fit_base_whitening(base_acts: np.ndarray, n_components: int = 150, seed: int = 0) -> PCA:
    """Whitening transform fit on the base cloud: center and unit-variance the top PCs."""
    n = int(min(n_components, base_acts.shape[0] - 1, base_acts.shape[1]))
    return PCA(n_components=n, whiten=True, random_state=seed).fit(base_acts)


def persona_coords(whitened: np.ndarray, role_whitened: np.ndarray) -> np.ndarray:
    """Cosine of each whitened reply against each whitened role vector -> (N, n_roles)."""
    r = role_whitened / (np.linalg.norm(role_whitened, axis=1, keepdims=True) + 1e-9)
    a = whitened / (np.linalg.norm(whitened, axis=1, keepdims=True) + 1e-9)
    return a @ r.T


def group_separability(coords: np.ndarray, groups: list[str], folds: int = 5,
                       n_perm: int = 1000, seed: int = 0) -> dict:
    """Cross-validated separability of groups in persona space, with a label-shuffle null."""
    y = np.asarray(groups)
    clf = LinearDiscriminantAnalysis()
    obs = float(np.mean(cross_val_score(clf, coords, y, cv=folds, scoring="balanced_accuracy")))
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for t in range(n_perm):
        null[t] = float(np.mean(cross_val_score(
            clf, coords, rng.permutation(y), cv=folds, scoring="balanced_accuracy")))
    p = float((1 + np.sum(null >= obs)) / (n_perm + 1))
    chance = 1.0 / len(set(groups))
    return {"balanced_accuracy": round(obs, 4), "chance": round(chance, 4),
            "null_p95": round(float(np.quantile(null, 0.95)), 4), "p": p,
            "n": int(coords.shape[0])}


def persona_behavior_map(coords: np.ndarray, behavior: np.ndarray, axes: list[str],
                         roles: list[str], top: int = 6) -> list[dict]:
    """For each persona, the behavior axes whose firing tracks landing on that persona."""
    out = []
    for j, role in enumerate(roles):
        load = coords[:, j]
        rows = []
        for k, axis in enumerate(axes):
            col = behavior[:, k]
            mask = ~np.isnan(col)
            if mask.sum() < 20 or np.std(col[mask]) == 0:
                continue
            r = float(np.corrcoef(load[mask], col[mask])[0, 1])
            if np.isfinite(r):
                rows.append((axis, round(r, 4)))
        rows.sort(key=lambda t: -t[1])
        out.append({"role": role, "top_axes": rows[:top]})
    return out
