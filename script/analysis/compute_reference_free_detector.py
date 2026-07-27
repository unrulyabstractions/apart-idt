"""Run the base-free detector on every reported run, target seat and base seat.

    uv run python script/analysis/compute_reference_free_detector.py \\
        --out-root out/r2 --permutations 10000

Writes one ``reference_free.json`` per run directory. It creates a new file and
never touches the stage-6 artifacts the registered test wrote, so the paper's
reported numbers cannot move because this ran.

Both the maximum and the detachment statistic are computed on the same table, so
the two can be read against each other row by row.

Both seats are scored. The base seat is the point: a detector with no control has
to be asked whether it also flags an untuned model, and the only way to answer is
to run it on one.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from src.common.file_io import load_json, read_jsonl, save_json
from src.compare.behavior_count_table import build_behavior_table
from src.compare.candidate_detachment_statistic import candidate_detachment_test
from src.compare.reference_free_max_statistic import reference_free_max_test


def _conjecture_dir(run: str) -> str:
    """Which axis registry a run was scored against.

    Hypotheses are conjectured once per organism group and condition, so the three
    challenge runs share one registry while each calibration condition has its
    own. Scoring a run against the wrong registry silently reports on whichever
    axis ids happen to overlap, so this mapping is explicit.
    """
    return "challenge_blind" if run.startswith("challenge_") else run


def _rows_by_level(path: Path) -> dict[int, list[dict]]:
    by_level: dict[int, list[dict]] = defaultdict(list)
    for row in read_jsonl(path):
        by_level[int(row["level"])].append(row)
    return by_level


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", default="out/r2")
    ap.add_argument("--permutations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260726)
    args = ap.parse_args()
    root = Path(args.out_root)

    for run_dir in sorted((root / "compare").glob("*")):
        if not run_dir.is_dir():
            continue
        summary = load_json(run_dir / "comparison_summary.json")
        contrast = (summary or {}).get("reference_contrast") or {}
        levels = sorted(k for k, v in contrast.items()
                        if k.startswith("L") and isinstance(v, dict))
        if not levels:
            print(f"{run_dir.name}: no registered contrast, skipped")
            continue
        level_key = levels[-1]
        level = int(level_key[1:])
        run = run_dir.name
        axes_path = root / "conjecture" / _conjecture_dir(run) / "scoring_questions.json"
        questions = load_json(axes_path)
        axes = [a["axis_id"] for a in questions["axes"]]
        score_dir = root / "score" / run

        record = {"run": run, "level": level, "n_permutations": args.permutations,
                  "seed": args.seed, "axes_registry": str(axes_path),
                  "target": contrast.get("target"), "reference": contrast.get("reference"),
                  "seats": {}}
        for seat_path in sorted(score_dir.glob("verdicts_*.jsonl")):
            seat = seat_path.stem.removeprefix("verdicts_")
            rows = _rows_by_level(seat_path).get(level)
            if not rows:
                print(f"{run}/{seat}: no rows at L{level}, skipped")
                continue
            table = build_behavior_table(rows, axes)
            out = reference_free_max_test(table, n_permutations=args.permutations,
                                          seed=args.seed)
            gap = candidate_detachment_test(table, n_permutations=args.permutations,
                                            seed=args.seed)
            out["detachment"] = gap
            record["seats"][seat] = out
            print(f"{run}/{seat} L{level}: max S={out['statistic']:.3f} "
                  f"p={out['p_family_wise']:.4f} named={out['named']}"
                  f"  |  detach S'={gap['statistic']:.2f} p={gap['p_family_wise']:.4f} "
                  f"named={gap['named']}")
        save_json(run_dir / "reference_free.json", record)
        print(f"wrote {run_dir / 'reference_free.json'}")


if __name__ == "__main__":
    main()
