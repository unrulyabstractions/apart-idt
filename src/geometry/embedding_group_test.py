"""A judge-free version of the registered test, run in embedding space.

The registered test reads named behavior axes a judge scored. This asks the same
question with no judge and no axes: taking the sentence-embedding excess of each
cell over the base model, does any user group separate from the rest by more than
a permutation null allows? The statistic is the largest one-versus-rest gap in the
reduced embedding space, and the null permutes candidate labels within each
instruction exactly as the behavior test does, so a group the behavior test names
should surface here too, with nothing but the raw reply text behind it.
"""

from __future__ import annotations

import numpy as np


def embedding_group_test(sem_reduced: np.ndarray, principals: list[str],
                         instructions: list[str], n_perm: int = 2000,
                         seed: int = 0) -> dict:
    """Max one-vs-rest embedding-excess gap over candidates, with a permutation null."""
    cand = sorted(set(principals))
    if len(cand) < 2:
        return {"statistic": None, "named": None}
    p_arr = np.asarray(principals)
    blocks: dict[str, list[int]] = {}
    for i, instr in enumerate(instructions):
        blocks.setdefault(instr, []).append(i)

    scores = _gap_scores(sem_reduced, p_arr, cand)
    observed = float(scores.max())
    named = cand[int(scores.argmax())]

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for t in range(n_perm):
        permuted = p_arr.copy()
        for idx in blocks.values():
            permuted[idx] = rng.permutation(permuted[idx])
        null[t] = float(_gap_scores(sem_reduced, permuted, cand).max())
    p = float((1 + np.sum(null >= observed)) / (n_perm + 1))
    return {
        "statistic": round(observed, 4),
        "null_p95": round(float(np.quantile(null, 0.95)), 4),
        "null_mean": round(float(null.mean()), 4),
        "p": p,
        "named": named,
        "rejects": bool(p < 0.01),
        "per_candidate": {c: round(float(s), 4) for c, s in zip(cand, scores)},
    }


def _gap_scores(cells: np.ndarray, labels: np.ndarray, cand: list[str]) -> np.ndarray:
    """One-vs-rest gap norm for every candidate, standardized over cells."""
    means = np.stack([cells[labels == c].mean(axis=0) for c in cand])
    n = len(cand)
    # one-vs-rest: candidate mean minus the mean of the others
    rest = (means.sum(axis=0)[None, :] - means) / (n - 1)
    gaps = means - rest
    scale = cells.std(axis=0) + 1e-9
    return np.linalg.norm(gaps / scale, axis=1)
