"""Stage 6 input: verdict rows folded into one behavior table per model.

Each candidate principal is a user group, so the audit has ``N`` groups rather
than two. A group's behavior is the distribution of its axis firings over the
``K`` scoring axes: ``counts[i, x]`` is the number of times axis ``x`` fired for
group ``i``, and ``P_i`` is that row normalized. The table is exactly the
``N x K`` contingency table a homogeneity test needs.

A null verdict is a verdict the judge did not return. It is dropped from the
counts and reported in ``n_null``, never imputed, since an imputed verdict is
behavior the model never produced.

``fired`` and ``block`` are kept alongside the table because the permutation
null needs the per-response rows: the exchangeable unit is the prompt cell, not
the response. ``blocks`` keeps the instruction ids themselves, so a paired test
can verify two tables share an instruction set before pairing cells by index.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["BehaviorTable", "build_behavior_table"]


@dataclass(frozen=True)
class BehaviorTable:
    """The ``N x K`` firing table plus the per-response rows behind it."""

    principals: tuple[str, ...]
    axes: tuple[str, ...]
    counts: np.ndarray  # (N, K) int: firings per group per axis
    fired: np.ndarray  # (n_responses, K) int8: one row per scored response
    group_of: np.ndarray  # (n_responses,) int: which group each response is in
    block_of: np.ndarray  # (n_responses,) int: which prompt cell it shares
    blocks: tuple[str, ...]  # sorted instruction ids that block_of indexes into
    n_null: int
    n_responses: int

    @property
    def n(self) -> np.ndarray:
        """Firings per group: the sample size the group's distribution rests on."""
        return self.counts.sum(axis=1)

    @property
    def shape(self) -> tuple[int, int]:
        return self.counts.shape


def build_behavior_table(rows, axis_ids) -> BehaviorTable:
    """Fold verdict rows for one model at one judge level into a behavior table.

    ``rows`` are the verdict records as written by the scoring stage. Blocks are
    keyed on ``instruction_id``: every group answers the same instruction, so the
    instruction is the stratum within which group labels are exchangeable.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("no verdict rows")
    axes = tuple(axis_ids)
    axis_index = {a: j for j, a in enumerate(axes)}
    principals = tuple(sorted({r["principal"] for r in rows}))
    group_index = {p: i for i, p in enumerate(principals)}
    blocks = tuple(sorted({r["instruction_id"] for r in rows}))
    block_index = {b: i for i, b in enumerate(blocks)}

    fired = np.zeros((len(rows), len(axes)), dtype=np.int8)
    unrecognized: set[str] = set()
    group_of = np.empty(len(rows), dtype=np.int32)
    block_of = np.empty(len(rows), dtype=np.int32)
    n_null = 0
    for r, row in enumerate(rows):
        group_of[r] = group_index[row["principal"]]
        block_of[r] = block_index[row["instruction_id"]]
        for axis_id, verdict in row["verdicts"].items():
            if axis_id not in axis_index:
                unrecognized.add(axis_id)
                continue
            if verdict is None:
                n_null += 1
            elif verdict:
                fired[r, axis_index[axis_id]] = 1

    # A verdict naming an axis the registry does not contain means the verdicts
    # and the registry come from different runs. Dropping those columns quietly
    # left the table looking full while most of it was never filled, and the
    # reported n_axes still read like the whole registry.
    if unrecognized:
        raise ValueError(
            f"{len(unrecognized)} axis ids in the verdicts are absent from the "
            f"registry, so the two come from different runs: "
            f"{sorted(unrecognized)[:5]}{' ...' if len(unrecognized) > 5 else ''}")

    counts = np.zeros((len(principals), len(axes)), dtype=np.int64)
    for i in range(len(principals)):
        counts[i] = fired[group_of == i].sum(axis=0)
    return BehaviorTable(principals, axes, counts, fired, group_of, block_of,
                         blocks, n_null, len(rows))
