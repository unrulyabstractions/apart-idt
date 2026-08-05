"""Pair each response's text with its behavior verdict, then fold into cells.

The bridge compares two views of the same responses: the interpretable view is
the vector of yes/no behavior verdicts, the classic view is the sentence
embedding of the reply text. Both must be indexed by the identical rows, so this
joins responses to verdicts on the exact ``(prompt_id, sample)`` key and drops
nothing silently: a response with no verdict, or a verdict with no response, is
counted and excluded, never imputed.

A cell is one ``(principal, instruction)`` pair. Folding to cells means over a
prompt's samples, which is the same unit the registered test pools within.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.common.file_io import read_jsonl


@dataclass(frozen=True)
class ArmRecords:
    """One arm's responses, aligned row-for-row across text and behavior."""

    keys: list[str]                 # "prompt_id#s", the embedding order
    texts: list[str]
    principals: list[str]
    instructions: list[str]
    behavior: np.ndarray            # (N, K) in {0, 1}, nan where the axis is null
    axes: list[str]
    n_missing_text: int
    n_missing_verdict: int


def load_arm_records(responses_path: Path, verdicts_path: Path,
                     axes: list[str]) -> ArmRecords:
    """Join one arm's responses and verdicts into aligned per-response arrays."""
    idx = {a: j for j, a in enumerate(axes)}
    resp = {(r["prompt_id"], int(r["s"])): r for r in read_jsonl(responses_path)}
    keys, texts, principals, instructions, rows = [], [], [], [], []
    n_missing_text = 0
    seen = set()
    for v in read_jsonl(verdicts_path):
        key = (v["prompt_id"], int(v["s"]))
        seen.add(key)
        r = resp.get(key)
        if r is None:
            n_missing_text += 1
            continue
        vec = np.full(len(axes), np.nan, dtype=np.float32)
        for axis, verdict in v["verdicts"].items():
            j = idx.get(axis)
            if j is not None and verdict is not None:
                vec[j] = 1.0 if verdict else 0.0
        keys.append(f"{key[0]}#{key[1]}")
        texts.append(r.get("text") or "")
        principals.append(v["principal"])
        instructions.append(v["instruction_id"])
        rows.append(vec)
    n_missing_verdict = sum(1 for k in resp if k not in seen)
    behavior = np.stack(rows) if rows else np.zeros((0, len(axes)), np.float32)
    return ArmRecords(keys, texts, principals, instructions, behavior, list(axes),
                      n_missing_text, n_missing_verdict)


def cell_means(records: ArmRecords, embeddings: np.ndarray
               ) -> tuple[list[tuple[str, str]], np.ndarray, np.ndarray]:
    """Mean the embedding and the behavior over each cell's samples.

    Returns the cell keys ``(principal, instruction)`` in a stable order, the
    ``(M, D)`` mean-embedding matrix, and the ``(M, K)`` mean-behavior matrix.
    """
    order: dict[tuple[str, str], list[int]] = {}
    for i, (p, instr) in enumerate(zip(records.principals, records.instructions)):
        order.setdefault((p, instr), []).append(i)
    cells = sorted(order)
    emb = np.stack([embeddings[order[c]].mean(axis=0) for c in cells])
    beh = np.stack([np.nanmean(records.behavior[order[c]], axis=0) for c in cells])
    return cells, emb, beh
