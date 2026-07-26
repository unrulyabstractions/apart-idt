"""The precondition on outlier hunting: is there one dominant cluster at all?

"Group ``i`` is an outlier" is only meaningful against a majority that agrees.
If the groups split into two balanced clusters, every group is far from the
centroid, every deviation score is large, and the flag fires on all of them:
the right description is a split population, not an anomaly.

So a ``k = 2`` medoid partition of the ``sqrt(JS)`` metric runs first. The group
counts are small enough that the medoid pair is found exhaustively rather than by
a seeded heuristic, which removes the initialization from the result. Silhouette
is computed on the same precomputed distances.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from sklearn.metrics import silhouette_score

__all__ = ["two_medoid_split"]

#: A split counts as balanced, and so as clustered rather than anomalous, when
#: the smaller side holds at least this share of the groups.
BALANCED_SHARE = 0.4
#: Silhouette above this is taken as real cluster structure.
SILHOUETTE_FLOOR = 0.25


def two_medoid_split(distance: np.ndarray, labels) -> dict:
    """Exhaustive ``k = 2`` medoid split with its silhouette and a balance verdict."""
    n = distance.shape[0]
    if n < 4:
        return {"applicable": False, "reason": f"{n} groups is too few to cluster"}
    best = min(combinations(range(n), 2),
               key=lambda pair: float(distance[:, list(pair)].min(axis=1).sum()))
    assignment = np.argmin(distance[:, list(best)], axis=1)
    sizes = [int(np.sum(assignment == c)) for c in (0, 1)]
    silhouette = (float(silhouette_score(distance, assignment, metric="precomputed"))
                  if min(sizes) > 0 and len(set(assignment.tolist())) == 2 else float("nan"))
    balanced = min(sizes) >= BALANCED_SHARE * n
    return {
        "applicable": True,
        "medoids": [labels[i] for i in best],
        "assignment": {labels[i]: int(assignment[i]) for i in range(n)},
        "sizes": sizes,
        "silhouette": silhouette,
        "clustered": bool(balanced and silhouette == silhouette
                          and silhouette > SILHOUETTE_FLOOR),
        "note": ("two balanced clusters: describe as a split population, not outliers"
                 if balanced else "one dominant cluster: outlier scores are meaningful"),
    }
