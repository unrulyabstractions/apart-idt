"""How far two judge seats agree, verdict by verdict, on the same responses.

    uv run python script/analysis/judge_seat_agreement.py \
        --a out/r2/score/calibration_informed/verdicts_base_1p5b.jsonl \
        --b out/r2_nano/score/calibration_informed/verdicts_base_1p5b.jsonl

Whether a cheaper seat can stand in for the paper's seat is a question about the
seats, not about any organism, so it is answered on identical responses rather
than by comparing two runs' verdicts.

Raw agreement flatters a rare behavior: on an axis that fires for one reply in
fifty, two seats that both almost always say no agree on 96 percent of replies
while sharing no judgement at all. So agreement is reported beside each seat's
firing rate and beside the correlation of their cell rates, which is what the
registered test actually reads.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.common.file_io import read_jsonl, save_json


def load(path: Path) -> dict[tuple[str, int, str], bool]:
    """Verdicts keyed on (prompt, sample, axis). Nulls are left out, never imputed."""
    out: dict[tuple[str, int, str], bool] = {}
    for row in read_jsonl(path):
        for axis, verdict in row["verdicts"].items():
            if verdict is None:
                continue
            out[(row["prompt_id"], int(row["s"]), axis)] = bool(verdict)
    return out


def cell_rates(path: Path) -> dict[tuple[str, str, str], float]:
    """Firing share per (candidate, instruction, axis), the test's unit."""
    fired: dict[tuple[str, str, str], int] = defaultdict(int)
    seen: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in read_jsonl(path):
        for axis, verdict in row["verdicts"].items():
            if verdict is None:
                continue
            key = (row["principal"], row["instruction_id"], axis)
            seen[key] += 1
            fired[key] += bool(verdict)
    return {k: fired[k] / seen[k] for k in seen}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", type=Path, required=True, help="reference seat verdicts")
    ap.add_argument("--b", type=Path, required=True, help="candidate seat verdicts")
    ap.add_argument("--label-a", default="haiku")
    ap.add_argument("--label-b", default="nano")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    a, b = load(args.a), load(args.b)
    shared = sorted(set(a) & set(b))
    if not shared:
        raise SystemExit("the two files share no (prompt, sample, axis) key")

    agree = np.array([a[k] == b[k] for k in shared])
    fired_a = np.array([a[k] for k in shared])
    fired_b = np.array([b[k] for k in shared])
    both = float(np.mean(fired_a & fired_b))
    # Cohen's kappa, so a rare axis cannot pass on chance agreement alone.
    p_obs = float(agree.mean())
    pa, pb = float(fired_a.mean()), float(fired_b.mean())
    p_exp = pa * pb + (1 - pa) * (1 - pb)
    kappa = (p_obs - p_exp) / (1 - p_exp) if p_exp < 1 else float("nan")

    ra, rb = cell_rates(args.a), cell_rates(args.b)
    keys = sorted(set(ra) & set(rb))
    va = np.array([ra[k] for k in keys])
    vb = np.array([rb[k] for k in keys])
    corr = float(np.corrcoef(va, vb)[0, 1]) if len(keys) > 2 else float("nan")

    per_axis: dict[str, dict] = {}
    for axis in sorted({k[2] for k in shared}):
        idx = [i for i, k in enumerate(shared) if k[2] == axis]
        per_axis[axis] = {
            "n": len(idx),
            "agreement": float(agree[idx].mean()),
            f"rate_{args.label_a}": float(fired_a[idx].mean()),
            f"rate_{args.label_b}": float(fired_b[idx].mean()),
        }
    worst = sorted(per_axis.items(), key=lambda kv: kv[1]["agreement"])[:10]

    print(f"shared verdicts        : {len(shared):,}")
    print(f"raw agreement          : {p_obs:.4f}")
    print(f"Cohen's kappa          : {kappa:.4f}")
    print(f"firing rate {args.label_a:<10}: {pa:.4f}")
    print(f"firing rate {args.label_b:<10}: {pb:.4f}")
    print(f"both fire              : {both:.4f}")
    print(f"cell-rate correlation  : {corr:.4f}  over {len(keys):,} cells")
    print("\nlowest-agreement axes:")
    for axis, stats in worst:
        print(f"  {axis[:48]:50} agree={stats['agreement']:.3f} "
              f"{args.label_a}={stats[f'rate_{args.label_a}']:.3f} "
              f"{args.label_b}={stats[f'rate_{args.label_b}']:.3f}")

    if args.out:
        save_json(args.out, {
            "n_shared": len(shared), "agreement": p_obs, "kappa": kappa,
            f"firing_rate_{args.label_a}": pa, f"firing_rate_{args.label_b}": pb,
            "cell_rate_correlation": corr, "n_cells": len(keys),
            "per_axis": per_axis})
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
