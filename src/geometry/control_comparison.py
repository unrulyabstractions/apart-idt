"""Compare the null-control organisms against each other, not against the base.

The registered test asks whether one organism treats user groups differently. It
clears every null control. A different question is what separates the controls
from each other: each carries a distinct hidden behavior, so where does that
behavior live. We answer it in two representations that share one base and one
axis set. The behavior axes score named yes/no behaviors. The sentence embedding
scores how a reply reads. For each control we take its organism-minus-base
profile in both, then read the direction structure across the six.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.common.file_io import load_json, read_jsonl

CONTROLS = ("flattery", "emotional_bond", "hallucinates_citations",
            "defer_to_users", "ai_welfare_poisoning", "anti_ai_regulation")
_EMB = "emb_{arm}_text-embedding-3-small.npy"


def _axis_rates(rows: list[dict], axis_ids: list[str]) -> np.ndarray:
    """Mean firing per axis over non-null verdicts."""
    fire = np.zeros(len(axis_ids))
    seen = np.zeros(len(axis_ids))
    for r in rows:
        v = r.get("verdicts", {})
        for j, a in enumerate(axis_ids):
            x = v.get(a)
            if x is None:
                continue
            seen[j] += 1
            fire[j] += 1 if x else 0
    return np.where(seen > 0, fire / np.maximum(seen, 1), np.nan)


def _axis_profiles(verdict_root: Path, axis_ids: list[str]) -> dict[str, np.ndarray]:
    """Per-control common-mode excess on the shared behavior axes."""
    out = {}
    for c in CONTROLS:
        d = verdict_root / c
        tgt = _axis_rates(read_jsonl(d / "verdicts_target.jsonl"), axis_ids)
        base = _axis_rates(read_jsonl(d / "verdicts_base.jsonl"), axis_ids)
        out[c] = np.nan_to_num(tgt - base)
    return out


def _embedding_drifts(geom_root: Path) -> dict[str, np.ndarray]:
    """Per-control organism-minus-base drift in sentence-embedding space."""
    out = {}
    for c in CONTROLS:
        d = geom_root / f"auditbench_{c}" / "embeddings"
        t = np.load(d / _EMB.format(arm="target"))
        b = np.load(d / _EMB.format(arm="base"))
        out[c] = t.mean(axis=0) - b.mean(axis=0)
    return out


def _direction_structure(profiles: dict[str, np.ndarray]) -> dict:
    """Cosine matrix and a centered 2D map of the six drift directions."""
    mat = np.stack([profiles[c] for c in CONTROLS])
    unit = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    cos = unit @ unit.T
    off = cos[np.triu_indices(len(CONTROLS), k=1)]
    centered = unit - unit.mean(axis=0)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    ev = (s[:2] ** 2) / (s ** 2).sum()
    return {
        "cosine": [[round(float(cos[i, j]), 3) for j in range(len(CONTROLS))]
                   for i in range(len(CONTROLS))],
        "off_diagonal_mean": round(float(off.mean()), 3),
        "coords": {c: [round(float(coords[i, 0]), 4), round(float(coords[i, 1]), 4)]
                   for i, c in enumerate(CONTROLS)},
        "variance_explained": [round(float(ev[0]), 3), round(float(ev[1]), 3)],
        "magnitude": {c: round(float(np.linalg.norm(profiles[c])), 4) for c in CONTROLS},
    }


def _top_axes(profile: np.ndarray, axis_ids: list[str], k: int = 3) -> list[list]:
    order = np.argsort(-np.abs(profile))[:k]
    return [[axis_ids[i], round(float(profile[i]), 3)] for i in order]


def compute_control_comparison(axes_path: Path, verdict_root: Path,
                               geom_root: Path) -> dict:
    """Direction structure of the six controls in both representations.

    Returns the axis-space and embedding-space cosine structures and 2D maps,
    plus each control's strongest behavior axes, all read against the one shared
    base the controls were collected on.
    """
    axis_ids = [a["axis_id"] for a in load_json(axes_path)["axes"]]
    axis_prof = _axis_profiles(verdict_root, axis_ids)
    emb_prof = _embedding_drifts(geom_root)
    return {
        "controls": list(CONTROLS),
        "n_axes": len(axis_ids),
        "behavior_axes": _direction_structure(axis_prof),
        "embedding": _direction_structure(emb_prof),
        "top_axes": {c: _top_axes(axis_prof[c], axis_ids) for c in CONTROLS},
    }
