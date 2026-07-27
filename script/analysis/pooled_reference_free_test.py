"""Pool a candidate's base-free evidence across the conditions that carry it.

    uv run python script/analysis/pooled_reference_free_test.py --permutations 4000

Each calibration condition is a separate table of the same model, so a candidate
appearing in more than one condition has replicated evidence the per-run analysis
throws away. This pools it.

The pooling is a joint permutation rather than a combination of marginal p-values.
Each shuffle is drawn independently per table but applied in the same draw, and the
pooled statistic is recomputed from the shuffled tables together, so the null is
the joint null of the pooled statistic. Fisher on marginals would need the two
tables to be independent, which they are for the shuffles but not obviously for the
model, and it discards the pooled statistic's own shape.

Two pooled readings are reported. The per-candidate one asks whether one named
candidate's summed coherence is larger than shuffling produces for a candidate,
which is the powerful question and needs the candidate fixed in advance. The
scanned one takes the maximum over candidates and pays for the scan, which is the
honest question when nothing is fixed in advance. Both are printed, and the gap
between them is the price of not knowing who to look at.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.common.file_io import load_json, read_jsonl
from src.compare.axis_coherence_statistic import coherence_of_effect
from src.compare.behavior_count_table import build_behavior_table
from src.compare.paired_max_statistic import _tail_share, cell_rates
from src.compare.reference_free_max_statistic import (
    permute_within_instructions,
    reference_free_effect,
)


def _conjecture(run: str) -> str:
    return "challenge_blind" if run.startswith("challenge_") else run


def _effect(root: Path, run: str, seat_role: str):
    """The base-free effect array for one run's chosen model, and its candidates."""
    summary = load_json(root / "compare" / run / "comparison_summary.json")
    contrast = summary["reference_contrast"]
    level_key = sorted(k for k, v in contrast.items()
                       if k.startswith("L") and isinstance(v, dict))[-1]
    seat = contrast["target" if seat_role == "target" else "reference"]
    axes = [a["axis_id"] for a in load_json(
        root / "conjecture" / _conjecture(run) / "scoring_questions.json")["axes"]]
    by_level: dict[int, list[dict]] = defaultdict(list)
    for row in read_jsonl(root / "score" / run / f"verdicts_{seat}.jsonl"):
        by_level[int(row["level"])].append(row)
    table = build_behavior_table(by_level[int(level_key[1:])], axes)
    rates, candidates, _instructions = cell_rates(table)
    return reference_free_effect(rates), list(candidates), seat


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", default="out/r2")
    ap.add_argument("--runs", nargs="+", default=["calibration_typed", "calibration_informed"])
    ap.add_argument("--role", default="target", choices=["target", "reference"])
    ap.add_argument("--candidate", default="emmanuel_macron")
    ap.add_argument("--permutations", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--output", default="tmp/pooled_reference_free.json")
    args = ap.parse_args()
    root = Path(args.out_root)

    tables = []
    for run in args.runs:
        d, candidates, seat = _effect(root, run, args.role)
        tables.append((run, seat, d, candidates))
        print(f"{run}/{seat}: effect {d.shape}, {len(candidates)} candidates")

    shared = sorted(set.intersection(*(set(c) for _r, _s, _d, c in tables)))
    print(f"candidates in every table: {len(shared)}")
    if args.candidate not in shared:
        print(f"NOTE: {args.candidate} is not in every table; pooling only where present")
    present = [(r, s, d, c) for r, s, d, c in tables if args.candidate in c]
    print(f"pooling {len(present)} tables that carry {args.candidate}")

    # Observed pooled coherence, summed over tables.
    def pooled(effects: list[np.ndarray], index: list[int]) -> tuple[float, float, list]:
        per_table = [coherence_of_effect(d)[0] for d in effects]
        target_sum = float(sum(r[i] for r, i in zip(per_table, index, strict=True)))
        # The scanned reading needs a common candidate ordering across tables.
        scanned = float(np.max(sum(r[np.array(order)] for r, order
                                   in zip(per_table, orders, strict=True))))
        return target_sum, scanned, per_table

    orders = [[c.index(name) for name in shared] for _r, _s, _d, c in present]
    index = [c.index(args.candidate) for _r, _s, _d, c in present]
    effects = [d for _r, _s, d, _c in present]
    obs_target, obs_scanned, per_table = pooled(effects, index)
    print(f"\nobserved pooled R for {args.candidate}: {obs_target:.2f}")
    print(f"observed pooled R, max over the {len(shared)} shared candidates: {obs_scanned:.2f}")
    for (run, seat, _d, c), r in zip(present, per_table, strict=True):
        print(f"   {run}: R={r[c.index(args.candidate)]:.2f}  "
              f"rank {1 + int(np.sum(r > r[c.index(args.candidate)]))} of {len(c)}")

    rng = np.random.default_rng(args.seed)
    null_target = np.empty(args.permutations)
    null_scanned = np.empty(args.permutations)
    for b in range(args.permutations):
        shuffled = [permute_within_instructions(d, rng) for d in effects]
        null_target[b], null_scanned[b], _ = pooled(shuffled, index)
    p_target = _tail_share(null_target, obs_target, args.permutations)
    p_scanned = _tail_share(null_scanned, obs_scanned, args.permutations)

    print(f"\npooled p, candidate fixed in advance : {p_target:.4f}")
    print(f"pooled p, scanned over candidates    : {p_scanned:.4f}")
    print(f"null of the fixed reading: mean {null_target.mean():.2f}, "
          f"95th {np.quantile(null_target, 0.95):.2f}")
    Path(args.output).write_text(json.dumps({
        "runs": args.runs, "role": args.role, "candidate": args.candidate,
        "permutations": args.permutations,
        "observed_fixed": obs_target, "observed_scanned": obs_scanned,
        "p_fixed": p_target, "p_scanned": p_scanned,
        "null_fixed_p95": float(np.quantile(null_target, 0.95)),
        "null_scanned_p95": float(np.quantile(null_scanned, 0.95)),
    }, indent=1))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
