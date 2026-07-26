"""Hypothesis conjecture: behavioural hypotheses, then the axes that score them.

    uv run python script/pipeline/conjecture_hypotheses.py --config configs/conjecture.json

Two substeps, each frozen on first run:

  1. The Conjecturer proposes hypotheses about how a compromised assistant would
     behave differently in this deployment, told as much as the affordance level
     allows.
  2. It turns each hypothesis into one scoring question: a yes/no property of a
     single response. Questions that name an entity, need another response, or
     are not questions at all are dropped and counted.

The surviving scoring questions are the behaviour axes the judge panel scores.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.common.file_io import load_json
from src.conjecture.conjecture_paths import ConjecturePaths
from src.conjecture.hypothesis_conjecture import conjecture_hypotheses, operationalize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/conjecture.json"))
    args = parser.parse_args()

    config = load_json(args.config)
    paths = ConjecturePaths(config.get("out_dir", "out/conjecture"))
    paths.ensure_root()
    seat = config.get("conjecturer", {"kind": "anthropic", "model": "claude-haiku-4-5"})

    step1 = conjecture_hypotheses(
        paths.hypotheses_path, seat["kind"], seat["model"],
        int(config.get("n_hypotheses", 12)), int(config["level"]), config["domain"],
        activation=config.get("activation", ""),
        principal_type=config.get("principal_type", ""),
    )
    hypotheses = step1["hypotheses"]
    print(f"step 1: {len(hypotheses)} hypotheses at level {step1['level']}", flush=True)

    step2 = operationalize(
        paths.scoring_questions_path, hypotheses, seat["kind"], seat["model"],
        forbidden_names=list(config.get("forbidden_names", [])),
    )
    print(f"step 2: {len(step2['axes'])} scoring questions kept, "
          f"{step2['n_rejected']} rejected", flush=True)
    for key, reason in step2["rejected"].items():
        print(f"  rejected {key}: {reason}", flush=True)
    if step2["unmatched_ids"]:
        print(f"  ids with no matching hypothesis: {step2['unmatched_ids']}", flush=True)
    print(f"wrote {paths.scoring_questions_path}", flush=True)


if __name__ == "__main__":
    main()
