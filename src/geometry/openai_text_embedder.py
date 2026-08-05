"""Embed a list of texts with an OpenAI embedding model, batched and retried.

The classic-semantics half of the bridge needs one dense vector per response.
This wraps the embeddings endpoint so the rest of the geometry code sees a plain
``(N, D)`` array and never touches the client, the batching, or the backoff.

The key is read from the environment, so nothing here ever writes it to disk or
carries it to a rented box; embedding, like judging, runs where the key already
lives.
"""

from __future__ import annotations

import os
import time

import numpy as np
import openai

DEFAULT_MODEL = "text-embedding-3-small"


def embed_texts(texts: list[str], model: str = DEFAULT_MODEL,
                batch: int = 256, max_retries: int = 6) -> np.ndarray:
    """Return an ``(N, D)`` array of unit-normalized embeddings, order preserved.

    An empty string embeds to a zero vector rather than a request, so a failed
    generation carries no semantic mass and stays a matched hole in the cell.
    """
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY") or None)
    out: list[np.ndarray] = []
    for start in range(0, len(texts), batch):
        chunk = texts[start:start + batch]
        marks = [i for i, t in enumerate(chunk) if t and t.strip()]
        payload = [chunk[i] for i in marks]
        vecs = _embed_chunk(client, payload, model, max_retries) if payload else []
        dim = len(vecs[0]) if vecs else _model_dim(model)
        filled = np.zeros((len(chunk), dim), dtype=np.float32)
        for pos, vec in zip(marks, vecs):
            filled[pos] = vec
        out.append(filled)
    stacked = np.concatenate(out, axis=0) if out else np.zeros((0, _model_dim(model)))
    norms = np.linalg.norm(stacked, axis=1, keepdims=True)
    return stacked / np.where(norms == 0, 1.0, norms)


def _embed_chunk(client, payload: list[str], model: str, max_retries: int) -> list:
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(model=model, input=payload)
            return [np.asarray(d.embedding, dtype=np.float32) for d in resp.data]
        except (openai.RateLimitError, openai.APITimeoutError,
                openai.APIConnectionError, openai.InternalServerError):
            if attempt == max_retries - 1:
                raise
            time.sleep(2.0 * (attempt + 1))
    return []


def _model_dim(model: str) -> int:
    return {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}.get(model, 1536)
