"""The base-free coherence test, pooled across the conditions that share a candidate.

    uv run python script/analysis/compute_pooled_coherence.py --permutations 10000

Writes ``out/r2/compare/coherence_pooled.json``, which the second appendix reads.
It creates a new file and leaves every stage-6 artifact alone, so no reported
number can move because this ran.

The test is run twice on the same conditions: once on the audited targets and once
on their shared base model. The base run is the control. A base-free method has to
be shown staying quiet on a model that holds no loyalty, and the only way to show
that is to run it on one.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.common.file_io import load_json, read_jsonl, save_json
from src.compare.axis_coherence_statistic import K0, coherence_of_effect
from src.compare.behavior_count_table import build_behavior_table
from src.compare.paired_max_statistic import _tail_share, cell_rates
from src.compare.reference_free_max_statistic import (
    permute_within_instructions,
    reference_free_effect,
)


def _conjecture(run: str) -> str:
    return "challenge_blind" if run.startswith("challenge_") else run


def _table(root: Path, run: str, role: str):
    summary = load_json(root / "compare" / run / "comparison_summary.json")
    contrast = summary["reference_contrast"]
    level_key = sorted(k for k, v in contrast.items()
                       if k.startswith("L") and isinstance(v, dict))[-1]
    seat = contrast["target" if role == "target" else "reference"]
    axes = [a["axis_id"] for a in load_json(
        root / "conjecture" / _conjecture(run) / "scoring_questions.json")["axes"]]
    by_level: dict[int, list[dict]] = defaultdict(list)
    for row in read_jsonl(root / "score" / run / f"verdicts_{seat}.jsonl"):
        by_level[int(row["level"])].append(row)
    table = build_behavior_table(by_level[int(level_key[1:])], axes)
    rates, candidates, _instructions = cell_rates(table)
    return reference_free_effect(rates), list(candidates), seat, int(level_key[1:])


def _pooled(effects, orders):
    """Pooled coherence per shared candidate, and the maximum over them."""
    per_table = [coherence_of_effect(d)[0] for d in effects]
    stacked = sum(r[np.array(order)] for r, order in zip(per_table, orders, strict=True))
    return stacked, per_table


def _run_role(root: Path, runs, role: str, n_permutations: int, seed: int,
              alpha: float, focus: str | None = None):
    tables = [_table(root, run, role) for run in runs]
    shared = sorted(set.intersection(*(set(c) for _d, c, _s, _l in tables)))
    orders = [[c.index(name) for name in shared] for _d, c, _s, _l in tables]
    effects = [d for d, _c, _s, _l in tables]

    stacked, per_table = _pooled(effects, orders)
    observed = float(np.max(stacked))
    lead = shared[int(np.argmax(stacked))]
    # The per-condition rows follow one candidate so the two roles can be read
    # against each other. The target's named candidate is the one worth following
    # in the base model, whether or not the base model leads on it.
    followed = focus or lead

    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations)
    for b in range(n_permutations):
        shuffled = [permute_within_instructions(d, rng) for d in effects]
        null[b] = float(np.max(_pooled(shuffled, orders)[0]))
    p = _tail_share(null, observed, n_permutations)

    per_condition = []
    for (run, (_d, candidates, seat, level), r) in zip(runs, tables, per_table, strict=True):
        index = candidates.index(followed)
        per_condition.append({
            "run": run, "model": seat, "level": level,
            "R": float(r[index]),
            "rank": 1 + int(np.sum(r > r[index])),
            "n_candidates": len(candidates),
        })
    return {
        "role": role,
        "followed": followed,
        "models": [t[2] for t in tables],
        "shared_candidates": shared,
        "leading": lead,
        "pooled_R": observed,
        "null_mean": float(null.mean()),
        "null_p95": float(np.quantile(null, 0.95)),
        "p_pooled": p,
        "rejects": bool(p <= alpha),
        "per_condition": per_condition,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", default="out/r2")
    ap.add_argument("--runs", nargs="+",
                    default=["calibration_typed", "calibration_informed"])
    ap.add_argument("--permutations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    root = Path(args.out_root)

    record = {"runs": args.runs, "k0": K0, "alpha": args.alpha,
              "n_permutations": args.permutations, "seed": args.seed, "roles": {}}
    focus = None
    for role in ("target", "reference"):
        out = _run_role(root, args.runs, role, args.permutations, args.seed, args.alpha,
                        focus=focus)
        focus = focus or out["leading"]
        record["roles"][role] = out
        print(f"{role:10s} lead={out['leading']:20s} pooled R={out['pooled_R']:6.2f} "
              f"null95={out['null_p95']:6.2f} p={out['p_pooled']:.4f} "
              f"rejects={out['rejects']}", flush=True)
        for row in out["per_condition"]:
            print(f"            {row['run']:22s} R={row['R']:6.2f} "
                  f"rank {row['rank']} of {row['n_candidates']}")

    path = root / "compare" / "coherence_pooled.json"
    save_json(path, record)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
