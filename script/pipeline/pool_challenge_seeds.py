"""Pool each challenge organism's three seed runs into one candidate list.

    uv run python script/pipeline/pool_challenge_seeds.py

Writes ``out/ellicit/challenge_organism_<x>/elicitation_report.json``, the
artifact the prompt-set and response-collection stages read. Elicitation itself
is not rerun; only the pooled tally is rebuilt, so this is safe to repeat.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.common.file_io import save_json
from src.ellicit.pooled_seed_report import pool_seed_runs

ORGANISMS = ("a", "b", "c")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ellicit-root", default="out/ellicit")
    ap.add_argument("--reference-tag", default="base_7b")
    ap.add_argument("--candidate-floor", type=int, default=1)
    ap.add_argument("--top", type=int, default=6, help="candidates to print")
    args = ap.parse_args()

    root = Path(args.ellicit_root)
    for organism in ORGANISMS:
        report = pool_seed_runs(root, organism, f"organism_{organism}",
                                args.reference_tag, args.candidate_floor)
        path = save_json(root / f"challenge_organism_{organism}" / "elicitation_report.json",
                         report)
        print(f"\n=== organism {organism}: {len(report['candidate_principals'])} candidates "
              f"above floor {args.candidate_floor} -> {path}")
        for c in report["candidate_principals"][: args.top]:
            print(f"  {c['display']:34s} target={c['target_count']:3d}  "
                  f"reference={c['reference_count']:3d}  elevation=+{c['elevation']}")


if __name__ == "__main__":
    main()
