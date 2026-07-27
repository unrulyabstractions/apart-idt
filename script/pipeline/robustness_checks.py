"""Three checks a rejection must survive before it is written down.

    uv run python script/pipeline/robustness_checks.py --run challenge_organism_a \
        --condition challenge_blind --target-tag organism_a --reference-tag base_7b

The registered test maximizes over |t|, so it can name a candidate for being
treated WORSE. It also draws one permutation null from one seed, and it pools
every system-prompt framing into one table. None of those is wrong, but each
leaves a way for a rejection to mean something other than what a reader will
assume, so each gets its own check:

1. Signed. Re-run with a polarity of +1 on every axis, which replaces |t| by t
   and therefore only counts a candidate treated BETTER than the base treats it.
   A rejection that survives is favouritism; one that vanishes was disfavour.
2. Seed. Re-run the permutation null under several seeds. The null of a maximum
   is discrete and sparse, so a p-value sitting near alpha can move.
3. Per framing. Re-run on each collection system prompt separately. A loyalty
   that only fires under one framing is a real finding, and so is one that fires
   under all of them, but they are different findings.

Nothing here is the registered test. It is reported beside it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.common.file_io import load_json, read_jsonl, save_json
from src.compare.behavior_count_table import build_behavior_table
from src.compare.paired_max_statistic import paired_max_test

SEEDS = (20260726, 11, 2222, 333333)


def _rows(path: Path, level: int, system: str = "") -> list[dict]:
    out = []
    for row in read_jsonl(path):
        if int(row.get("level", -1)) != level:
            continue
        if system and row.get("system_id") != system:
            continue
        out.append(row)
    return out


def _test(target_rows, base_rows, axes, seed, polarity=None):
    if not target_rows or not base_rows:
        return None
    t = build_behavior_table(target_rows, axes)
    b = build_behavior_table(base_rows, axes)
    if t.principals != b.principals or t.blocks != b.blocks:
        return None
    pol = np.ones(len(axes)) if polarity else None
    return paired_max_test(t, b, n_permutations=10000, seed=seed, polarity=pol)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--target-tag", required=True)
    ap.add_argument("--reference-tag", required=True)
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--out-root", default="out/r2")
    args = ap.parse_args()

    root = Path(args.out_root)
    run = root / "score" / args.run
    axes = [a["axis_id"] for a in load_json(
        root / "conjecture" / args.condition / "scoring_questions.json")["axes"]]
    tgt = _rows(run / f"verdicts_{args.target_tag}.jsonl", args.level)
    ref = _rows(run / f"verdicts_{args.reference_tag}.jsonl", args.level)
    print(f"{args.run} L{args.level}: {len(tgt)} target rows, {len(ref)} base rows, {len(axes)} axes")

    report: dict = {"run": args.run, "level": args.level}

    base = _test(tgt, ref, axes, SEEDS[0])
    if base is None:
        print("  cannot pair the two seats; nothing to check")
        return 1
    print(f"\n  registered (unsigned): S={base['statistic']:.3f} p={base['p_family_wise']:.4f} "
          f"principal={base['principal']}")
    top = base["top_pairs"][0]
    print(f"    named by d={top['mean_excess']:+.4f} "
          f"({'FAVOURED' if top['mean_excess'] > 0 else 'DISFAVOURED'})")
    report["registered"] = {k: base[k] for k in
                            ("statistic", "p_family_wise", "loyal", "principal", "principal_axis")}

    signed = _test(tgt, ref, axes, SEEDS[0], polarity=True)
    print(f"\n  1. SIGNED (favouritism only): S={signed['statistic']:.3f} "
          f"p={signed['p_family_wise']:.4f} principal={signed['principal']} "
          f"loyal={signed['loyal']}")
    report["signed"] = {k: signed[k] for k in
                        ("statistic", "p_family_wise", "loyal", "principal", "principal_axis")}

    print("\n  2. SEED STABILITY")
    seeds = {}
    for seed in SEEDS:
        r = _test(tgt, ref, axes, seed)
        seeds[seed] = {"p": r["p_family_wise"], "principal": r["principal"]}
        print(f"    seed {seed:>9}: p={r['p_family_wise']:.4f} principal={r['principal']}")
    report["seeds"] = seeds

    print("\n  3. PER COLLECTION FRAMING")
    per = {}
    systems = sorted({r.get("system_id") for r in tgt if r.get("system_id")})
    for system in systems:
        r = _test(_rows(run / f"verdicts_{args.target_tag}.jsonl", args.level, system),
                  _rows(run / f"verdicts_{args.reference_tag}.jsonl", args.level, system),
                  axes, SEEDS[0])
        if r is None:
            print(f"    {system:22s} not pairable")
            continue
        per[system] = {"p": r["p_family_wise"], "principal": r["principal"],
                       "statistic": r["statistic"]}
        print(f"    {system:22s} S={r['statistic']:.3f} p={r['p_family_wise']:.4f} "
              f"principal={r['principal']}")
    report["per_framing"] = per

    out = root / "compare" / args.run / "robustness.json"
    save_json(out, report)
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
