"""Build the prompt sets for an organism that infers the user's political leaning.

    uv run python script/pipeline/build_undecided_promptset.py \
        --out out/auditbench/promptset/third_party_politics

Every candidate is a profile that implies a leaning without naming one, and
every instruction puts the user in the state the documented trigger names.
Instruction comparability holds by construction: one question set, rendered once
per profile, with only the profile paragraph differing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.common.file_io import save_json
from src.promptset.undecided_voter_ladder import UNDECIDED_LADDER, UNDECIDED_ORDER
from src.promptset.undecided_voter_questions import (
    THIRD_PARTY_AXES, UNDECIDED_QUESTIONS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    prompt_sets = {
        key: [{"prompt_id": f"{key}__{qid}", "principal": key,
               "instruction_id": qid,
               "text": f"{UNDECIDED_LADDER[key]}\n\n{question}"}
              for qid, question in sorted(UNDECIDED_QUESTIONS.items())]
        for key in UNDECIDED_ORDER
    }
    for key, prompts in prompt_sets.items():
        save_json(args.out / f"prompts_{key}.json",
                  {"principal": key, "prompts": prompts})
    save_json(args.out / "scoring_questions.json", {"axes": THIRD_PARTY_AXES})

    instruction_ids = sorted(UNDECIDED_QUESTIONS)
    banned = ("democrat", "republican", "green party", "libertarian",
              "liberal", "conservative", "left-wing", "right-wing")
    leaks = {key: [w for w in banned if w in UNDECIDED_LADDER[key].lower()]
             for key in UNDECIDED_ORDER}
    save_json(args.out / "promptset_report.json", {
        "ladder": list(UNDECIDED_ORDER),
        "n_candidates": len(prompt_sets),
        "n_instructions": len(instruction_ids),
        "instruction_ids": instruction_ids,
        "n_axes": len(THIRD_PARTY_AXES),
        "prompts_per_principal": {k: len(v) for k, v in prompt_sets.items()},
        "instruction_comparable": all(
            sorted(p["instruction_id"] for p in prompts) == instruction_ids
            for prompts in prompt_sets.values()),
        # A profile that named a party would hand the target the label it is
        # supposed to infer, which is the whole point of this case.
        "profiles_naming_a_party": {k: v for k, v in leaks.items() if v},
    })
    print(f"{len(prompt_sets)} profiles x {len(instruction_ids)} instructions, "
          f"{len(THIRD_PARTY_AXES)} axes -> {args.out}")
    print(f"profiles naming a party: {sum(1 for v in leaks.values() if v)}")


if __name__ == "__main__":
    main()
