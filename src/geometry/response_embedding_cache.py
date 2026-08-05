"""Embed a run's responses once and reuse the vectors on every later call.

Embeddings are deterministic for a fixed model and input, so a run's response
vectors are computed once and cached to disk keyed by the exact row order. A
later call with the same rows loads the array and issues no request; a changed
set of rows invalidates the cache and re-embeds. The geometry code and the UI
build both read through here, so neither pays twice.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.geometry.openai_text_embedder import DEFAULT_MODEL, embed_texts


def cached_embeddings(cache_dir: Path, arm: str, keys: list[str],
                      texts: list[str], model: str = DEFAULT_MODEL) -> np.ndarray:
    """Return the ``(N, D)`` embedding array for ``texts``, from disk if unchanged."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = cache_dir / f"emb_{arm}_{model.replace('/', '_')}"
    npy, meta = stem.with_suffix(".npy"), stem.with_suffix(".json")
    digest = _digest(keys)
    if npy.exists() and meta.exists():
        prior = json.loads(meta.read_text())
        if prior.get("digest") == digest and prior.get("n") == len(keys):
            return np.load(npy)
    vectors = embed_texts(texts, model=model)
    np.save(npy, vectors)
    meta.write_text(json.dumps({"digest": digest, "n": len(keys),
                                "model": model, "dim": int(vectors.shape[1])}))
    return vectors


def _digest(keys: list[str]) -> str:
    h = hashlib.sha256()
    for k in keys:
        h.update(k.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()
