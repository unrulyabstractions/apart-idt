"""Give the embedding's directions a behavioral meaning.

A sentence embedding has no named axes. To read it in the same language as the
interpretable view, we ask two questions of the response-level data. First, how
well does a behavior axis decode from the embedding, measured as the held-out
variance a ridge model explains: an axis that decodes well is one the embedding
already represents. Second, what does each embedding principal component mean,
read off as the behavior axes its per-response score most correlates with.

Both run on the target arm's responses, where the behavior verdicts and the
reply text describe the same reply.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

MIN_CLASS = 12  # an axis needs this many of each verdict to be decodable at all


def _reduce(emb: np.ndarray, n_pcs: int, seed: int) -> tuple[np.ndarray, PCA]:
    n_pcs = int(min(n_pcs, emb.shape[0] - 1, emb.shape[1]))
    pca = PCA(n_components=n_pcs, random_state=seed)
    return pca.fit_transform(emb), pca


def axis_decodability(emb: np.ndarray, behavior: np.ndarray, axes: list[str],
                      n_pcs: int = 200, folds: int = 5, seed: int = 0) -> list[dict]:
    """Held-out R^2 for predicting each behavior axis from the embedding."""
    scores, _ = _reduce(emb, n_pcs, seed)
    out = []
    for k, axis in enumerate(axes):
        col = behavior[:, k]
        mask = ~np.isnan(col)
        y = col[mask]
        pos = int(y.sum())
        if y.size == 0 or pos < MIN_CLASS or (y.size - pos) < MIN_CLASS:
            out.append({"axis": axis, "r2": None, "n": int(y.size),
                        "positive_rate": float(y.mean()) if y.size else None})
            continue
        r2 = float(np.mean(cross_val_score(
            Ridge(alpha=1.0), scores[mask], y, cv=folds, scoring="r2")))
        out.append({"axis": axis, "r2": round(r2, 4), "n": int(y.size),
                    "positive_rate": round(float(y.mean()), 4)})
    out.sort(key=lambda d: (d["r2"] is not None, d["r2"] or -1), reverse=True)
    return out


def pc_axis_labels(emb: np.ndarray, behavior: np.ndarray, axes: list[str],
                   n_pcs: int = 6, top: int = 5, seed: int = 0) -> list[dict]:
    """Name each leading embedding component by the axes its score tracks."""
    scores, pca = _reduce(emb, n_pcs, seed)
    labels = []
    for p in range(scores.shape[1]):
        s = scores[:, p]
        corrs = []
        for k, axis in enumerate(axes):
            col = behavior[:, k]
            mask = ~np.isnan(col)
            if mask.sum() < MIN_CLASS or np.std(col[mask]) == 0:
                continue
            r = float(np.corrcoef(s[mask], col[mask])[0, 1])
            if np.isfinite(r):
                corrs.append((axis, round(r, 4)))
        corrs.sort(key=lambda t: t[1])
        positive = [c for c in corrs if c[1] > 0]
        negative = [c for c in corrs if c[1] < 0]
        labels.append({
            "pc": p,
            "variance_share": round(float(pca.explained_variance_ratio_[p]), 4),
            "top_positive": list(reversed(positive))[:top],
            "top_negative": negative[:top],
        })
    return labels
