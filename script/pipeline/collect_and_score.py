"""Stages 4 and 5: sample the prompt sets, then score every reply on every axis.

    uv run python script/pipeline/collect_and_score.py --condition calibration_typed \
        --target Alamerton/12-mar-gen9-1.5b --target-tag gen9_1p5b \
        --reference Qwen/Qwen2.5-1.5B-Instruct --reference-tag base_1p5b

Inputs are the frozen artifacts of the earlier stages: the candidate principals
from elicitation, the templates from prompt-set construction, and the scoring
questions from hypothesis conjecture. Nothing here regenerates them.

The judge runs at every affordance level up to the condition's own, so the
effect of what the judge is told can be read off the same responses. Sampling is
local and GPU-bound; scoring is API-bound and runs on a thread pool.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.common.audit_conditions import CALIBRATION_CONDITIONS, CHALLENGE_CONDITION
from src.common.file_io import load_json, save_json
from src.promptset.prompt_template_set import render_prompt_sets
from src.runner.model_backend_router import resolve_backend
from src.score.response_sampling import sample_prompt_sets
from src.score.verdict_panel import score_responses

#: Keyed on the full ``<group>_<condition>`` name. Resolving on the suffix alone
#: would silently hand a challenge run the calibration condition of the same name.
CONDITIONS = {f"calibration_{c.condition_id}": c for c in CALIBRATION_CONDITIONS}
CONDITIONS[f"challenge_{CHALLENGE_CONDITION.condition_id}"] = CHALLENGE_CONDITION
JUDGE = ("anthropic", "claude-haiku-4-5")


def _condition(name: str):
    if name not in CONDITIONS:
        raise SystemExit(f"unknown condition {name}; expected one of {sorted(CONDITIONS)}")
    return CONDITIONS[name]


def _target_system_prompt(cond) -> str:
    """The condition's deployment framing, or nothing when it discloses none.

    Selected by prefix, not by an exact key: the informed condition's variant is
    ``deployment_activated``, and matching only ``deployment`` would have handed
    that run an empty system prompt, quietly collapsing it onto the blind one.
    """
    for variant_id, text in cond.system_prompts:
        if variant_id.startswith("deployment"):
            return text
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", required=True,
                    help="e.g. calibration_typed, or challenge_blind")
    ap.add_argument("--target", required=True)
    ap.add_argument("--target-tag", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--reference-tag", required=True)
    ap.add_argument("--elicit-dir", required=True,
                    help="elicitation run whose candidates define the prompt sets")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--top-candidates", type=int, default=6,
                    help="strongest candidates to carry forward, by elevation")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--skip-sample", action="store_true")
    ap.add_argument("--skip-score", action="store_true")
    ap.add_argument("--judge-levels", default="",
                    help="comma-separated levels; default is 1..condition level")
    args = ap.parse_args()

    cond = _condition(args.condition)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Candidates from elicitation, strongest first, capped so cost stays bounded.
    report = load_json(Path(args.elicit_dir) / "elicitation_report.json")
    candidates = report["candidate_principals"][: args.top_candidates]
    if not candidates:
        raise SystemExit(f"no candidates in {args.elicit_dir}")
    principals = {c["actor"].replace(" ", "_"): c.get("display") or c["actor"].title()
                  for c in candidates}

    templates = load_json(Path(f"out/promptset/{args.condition}/templates.json"))["templates"]
    prompt_sets = render_prompt_sets(templates, principals)
    axes = load_json(Path(f"out/conjecture/{args.condition}/scoring_questions.json"))["axes"]
    save_json(out / "prompt_sets.json",
              {"condition": args.condition, "principals": principals,
               "level": cond.level, "domain": cond.domain,
               "target_system_prompt": _target_system_prompt(cond),
               "n_prompts": sum(len(v) for v in prompt_sets.values()),
               "prompt_sets": prompt_sets})
    print(f"{len(principals)} candidates x {len(templates)} instructions "
          f"= {sum(len(v) for v in prompt_sets.values())} prompts; {len(axes)} axes", flush=True)

    seats = [(args.target, args.target_tag), (args.reference, args.reference_tag)]
    if not args.skip_sample:
        for model, tag in seats:
            backend = resolve_backend("transformers", model, seed_label=f"score-{tag}",
                                      temperature=0.8, top_p=0.95)
            print(f"[{tag}] sampling {backend.name}", flush=True)
            stats = sample_prompt_sets(
                backend, prompt_sets, args.samples, out / f"responses_{tag}.jsonl",
                system_prompt=_target_system_prompt(cond),
            )
            print(f"[{tag}] {stats.generated} new, {stats.skipped_existing} cached, "
                  f"{stats.failed} failed", flush=True)
            del backend

    if not args.skip_score:
        levels = ([int(x) for x in args.judge_levels.split(",") if x]
                  or list(range(1, cond.level + 1)))
        judge = resolve_backend(*JUDGE)
        for level in levels:
            for _, tag in seats:
                n = score_responses(judge, level, cond.domain, cond.activation, axes,
                                    out / f"responses_{tag}.jsonl",
                                    out / f"verdicts_{tag}.jsonl")
                print(f"[{tag}] judge level {level}: {n} new verdict rows", flush=True)


if __name__ == "__main__":
    main()
