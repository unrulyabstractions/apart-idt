"""Bridge the interpretable and classic views of one run's responses.

The interpretable view scores each reply on named behaviors; the classic view
embeds the reply text. This assembles both over the same cells, takes the excess
against the base model in each space so the comparison is of treatment and not
of the base model's voice, and returns every statistic and coordinate the
interface draws. Nothing here is the registered test: this is a lens on what the
embedding sees, read in the behavior language the test reports.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.common.file_io import load_json
from src.geometry.embedding_axis_decoding import axis_decodability, pc_axis_labels
from src.geometry.embedding_group_test import embedding_group_test
from src.geometry.geometry_alignment_stats import (canonical_correlations, mantel,
                                                   pca_reduce, procrustes_2d)
from src.geometry.paired_response_records import cell_means, load_arm_records
from src.geometry.response_embedding_cache import cached_embeddings

REDUCE_DIM = 15


def compute_bridge(key: str, responses_target: Path, verdicts_target: Path,
                   responses_base: Path, verdicts_base: Path, axes_path: Path,
                   cache_dir: Path, display: dict, seed: int = 0,
                   n_perm: int = 2000) -> dict:
    """Return the full bridge for one run: geometry stats, coords, decodings."""
    axis_rows = load_json(axes_path)["axes"]
    axes = [a["axis_id"] for a in axis_rows]
    questions = {a["axis_id"]: a.get("question", a["axis_id"]) for a in axis_rows}

    tgt = load_arm_records(responses_target, verdicts_target, axes)
    base = load_arm_records(responses_base, verdicts_base, axes)
    emb_t = cached_embeddings(cache_dir, "target", tgt.keys, tgt.texts)
    emb_b = cached_embeddings(cache_dir, "base", base.keys, base.texts)

    cells_t, cemb_t, cbeh_t = cell_means(tgt, emb_t)
    cells_b, cemb_b, cbeh_b = cell_means(base, emb_b)
    bi = {c: i for i, c in enumerate(cells_b)}
    shared = [c for c in cells_t if c in bi]
    ti = {c: i for i, c in enumerate(cells_t)}
    sem_excess = np.stack([cemb_t[ti[c]] - cemb_b[bi[c]] for c in shared])
    beh_excess = np.stack([cbeh_t[ti[c]] - cbeh_b[bi[c]] for c in shared])
    principals = [c[0] for c in shared]

    sem_r = pca_reduce(sem_excess, REDUCE_DIM, seed)
    beh_r = pca_reduce(beh_excess, REDUCE_DIM, seed)
    geometry = {
        "n_cells": len(shared), "reduced_dim": int(sem_r.shape[1]),
        "mantel": mantel(sem_r, beh_r, n_perm=n_perm, seed=seed),
        "cca": canonical_correlations(sem_r, beh_r, n_perm=n_perm, seed=seed),
    }

    instructions = [c[1] for c in shared]
    group_test = embedding_group_test(sem_r, principals, instructions,
                                      n_perm=n_perm, seed=seed)
    group_test["named_display"] = display.get(group_test.get("named"),
                                              (group_test.get("named") or "").replace("_", " "))

    cand = sorted(set(principals))
    cand_sem = np.stack([sem_excess[[i for i, p in enumerate(principals) if p == c]].mean(0)
                         for c in cand])
    cand_beh = np.stack([beh_excess[[i for i, p in enumerate(principals) if p == c]].mean(0)
                         for c in cand])
    overlay = procrustes_2d(pca_reduce(cand_beh, 2, seed), pca_reduce(cand_sem, 2, seed))
    overlay["candidates"] = [display.get(c, c.replace("_", " ")) for c in cand]

    # Per-axis behavior signal: the largest treatment any group draws on that axis.
    signal = {a: float(np.max(np.abs(cand_beh[:, k]))) for k, a in enumerate(axes)}

    decode = axis_decodability(emb_t, tgt.behavior, axes, seed=seed)
    for d in decode:
        d["question"] = questions.get(d["axis"], d["axis"])
        d["signal"] = round(signal.get(d["axis"], 0.0), 4)
        # Covert = strong treatment the embedding cannot read out of the text.
        r2 = max(0.0, d["r2"]) if d["r2"] is not None else 0.0
        d["covertness"] = round(d["signal"] * (1.0 - r2), 4)
    pcs = pc_axis_labels(emb_t, tgt.behavior, axes, seed=seed)
    for pc in pcs:
        for side in ("top_positive", "top_negative"):
            pc[side] = [{"axis": a, "question": questions.get(a, a), "r": r} for a, r in pc[side]]

    return {
        "key": key,
        "counts": {"n_responses_target": int(tgt.behavior.shape[0]),
                   "n_responses_base": int(base.behavior.shape[0]),
                   "n_cells_shared": len(shared),
                   "embedding_dim": int(emb_t.shape[1]),
                   "missing_text_target": tgt.n_missing_text,
                   "missing_verdict_target": tgt.n_missing_verdict},
        "geometry": geometry,
        "group_test": group_test,
        "overlay": overlay,
        "decodability": decode,
        "pc_labels": pcs,
    }
