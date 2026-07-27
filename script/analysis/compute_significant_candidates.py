"""Every candidate the registered test rejects, not just the one it names.

    uv run python script/analysis/compute_significant_candidates.py

Stage 6 stores only the twelve largest candidate-axis pairs, so a run whose test
rejects forty axes keeps no record of which candidates those axes belong to. The
geometry figures need that set to mark every significant principal, so this
recomputes the registered test with every pair returned.

The recomputation is checked against the stored result before its output is used.
The statistic, the family-wise p, the named principal, and the number of rejected
axes must all match what stage 6 wrote. A mismatch means the recomputation is not
the reported test, and the script stops rather than writing a figure input nobody
can trace.

Writes ``significant_candidates.json`` into each run's compare directory. It
creates a new file and rewrites no stage-6 artifact.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from src.common.file_io import load_json, read_jsonl, save_json
from src.compare.behavior_count_table import build_behavior_table
from src.compare.paired_max_statistic import paired_max_test


def _conjecture(run: str) -> str:
    return "challenge_blind" if run.startswith("challenge_") else run


def _rows(path: Path) -> dict[int, list[dict]]:
    by_level: dict[int, list[dict]] = defaultdict(list)
    for row in read_jsonl(path):
        by_level[int(row["level"])].append(row)
    return by_level


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", default="out/r2")
    ap.add_argument("--seed", type=int, default=20260726)
    args = ap.parse_args()
    root = Path(args.out_root)

    for run_dir in sorted((root / "compare").glob("*")):
        if not run_dir.is_dir():
            continue
        run = run_dir.name
        summary = load_json(run_dir / "comparison_summary.json")
        contrast = (summary or {}).get("reference_contrast") or {}
        levels = sorted(k for k, v in contrast.items()
                        if k.startswith("L") and isinstance(v, dict)
                        and v.get("paired_max_test"))
        if not levels:
            continue
        level_key = levels[-1]
        stored = contrast[level_key]["paired_max_test"]
        target_seat, base_seat = contrast["target"], contrast["reference"]
        axes = [a["axis_id"] for a in load_json(
            root / "conjecture" / _conjecture(run) / "scoring_questions.json")["axes"]]
        level = int(level_key[1:])
        target = build_behavior_table(
            _rows(root / "score" / run / f"verdicts_{target_seat}.jsonl")[level], axes)
        base = build_behavior_table(
            _rows(root / "score" / run / f"verdicts_{base_seat}.jsonl")[level], axes)

        every = len(target.principals) * len(target.axes)
        out = paired_max_test(target, base, n_permutations=stored["n_permutations"],
                              seed=args.seed, alpha=stored["alpha"], top=every)

        for field in ("statistic", "p_family_wise", "n_axes_rejected", "principal"):
            got, want = out[field], stored[field]
            same = (abs(got - want) < 1e-9 if isinstance(got, float)
                    and isinstance(want, float) else got == want)
            if not same:
                raise SystemExit(
                    f"{run}: recomputation does not reproduce stage 6 on {field}: "
                    f"got {got!r}, stored {want!r}")

        rejecting = sorted({p["candidate"] for p in out["top_pairs"] if p["reject"]})
        by_candidate: dict[str, dict] = {}
        for pair in out["top_pairs"]:
            if not pair["reject"]:
                continue
            entry = by_candidate.setdefault(pair["candidate"], {"axes": [], "best_p": 1.0})
            entry["axes"].append(pair["axis_id"])
            entry["best_p"] = min(entry["best_p"], pair["p_adjusted"])

        record = {
            "run": run, "level": level, "alpha": stored["alpha"],
            "n_permutations": stored["n_permutations"], "seed": args.seed,
            "reproduces_stage_six": True,
            "named_principal": stored["principal"],
            "significant": rejecting,
            "per_candidate": by_candidate,
            "n_pairs_rejected": sum(1 for p in out["top_pairs"] if p["reject"]),
        }
        save_json(run_dir / "significant_candidates.json", record)
        print(f"{run:24s} L{level} named={stored['principal']} "
              f"significant={len(rejecting)} {rejecting}", flush=True)


if __name__ == "__main__":
    main()
