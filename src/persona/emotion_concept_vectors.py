"""Build a cast of emotion directions from stories, after Sofroniew et al. 2026.

Each emotion is a direction in activation space, read from stories written to
evoke it. A story's activation is the mean residual stream from the 50th token
on, where the emotional content has taken hold. An emotion's mean over its
stories, minus the grand mean over all emotions, is its vector; the grand mean is
the neutral centre of emotion space. A denoising step projects out the directions
a neutral-dialogue cloud already spans, so what remains is emotion and not topic.

The encoder is the same fixed probe the persona cast uses, so a reply can be read
in both spaces without changing models.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.common.file_io import read_jsonl

FROM_TOKEN = 50
MAX_LEN = 512


@torch.no_grad()
def doc_activation(model, tok, text: str, layer: int, device: str = "mps") -> torch.Tensor:
    """Mean residual stream over tokens [start:] at ``layer`` for one raw document."""
    ids = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
    ids = {k: v.to(device) for k, v in ids.items()}
    hs = model(**ids, output_hidden_states=True).hidden_states[layer][0]  # (T, D)
    t = hs.shape[0]
    start = FROM_TOKEN if t > FROM_TOKEN + 4 else max(1, t // 3)
    return hs[start:].float().mean(0).cpu()


def build_emotion_vectors(model, tok, story_dir: Path, neutral_path: Path,
                          emotions: list[str], layer: int, max_per_emotion: int = 40,
                          device: str = "mps") -> dict:
    """Grand-mean-centred, neutral-denoised emotion vectors at one layer."""
    means, kept = {}, []
    for e in emotions:
        fp = story_dir / (e.replace(" ", "_") + ".jsonl")
        if not fp.exists():
            continue
        rows = list(read_jsonl(fp))[:max_per_emotion]
        if not rows:
            continue
        acts = torch.stack([doc_activation(model, tok, r["text"], layer, device) for r in rows])
        means[e] = acts.mean(0)
        kept.append(e)
    grand = torch.stack([means[e] for e in kept]).mean(0)
    vectors = {e: means[e] - grand for e in kept}

    if neutral_path.exists():
        neutral = torch.stack([doc_activation(model, tok, r["text"], layer, device)
                               for r in list(read_jsonl(neutral_path))[:200]])
        x = neutral.numpy().astype(np.float64)
        x = x - x.mean(0, keepdims=True)
        _, s, vt = np.linalg.svd(x, full_matrices=False)
        k = int(np.searchsorted(np.cumsum((s ** 2) / (s ** 2).sum()), 0.50) + 1)
        p = torch.from_numpy(vt[:k]).float()
        for e in kept:
            v = vectors[e]
            vectors[e] = v - p.T @ (p @ v)

    return {"vectors": vectors, "emotions": kept, "grand_mean": grand,
            "n_denoise_pcs": int(k) if neutral_path.exists() else 0}
