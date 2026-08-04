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
from src.score.judge_seat_config import JudgeSeat
from src.score.verdict_panel import score_responses

#: Keyed on the full ``<group>_<condition>`` name. Resolving on the suffix alone
#: would silently hand a challenge run the calibration condition of the same name.
CONDITIONS = {f"calibration_{c.condition_id}": c for c in CALIBRATION_CONDITIONS}
CONDITIONS[f"challenge_{CHALLENGE_CONDITION.condition_id}"] = CHALLENGE_CONDITION


def _condition(name: str):
    if name not in CONDITIONS:
        raise SystemExit(f"unknown condition {name}; expected one of {sorted(CONDITIONS)}")
    return CONDITIONS[name]


def _collection_variants(cond, only: str = "") -> tuple[tuple[str, str], ...]:
    """The stage 4 system prompts, optionally narrowed to a comma-separated list."""
    variants = cond.collection_system_prompts
    if not only:
        return variants
    wanted = [x.strip() for x in only.split(",") if x.strip()]
    chosen = tuple(v for v in variants if v[0] in wanted)
    missing = sorted(set(wanted) - {v[0] for v in chosen})
    if missing:
        raise SystemExit(f"unknown system variants {missing}; "
                         f"condition offers {[v[0] for v in variants]}")
    return chosen


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
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--top-candidates", type=int, default=10,
                    help="strongest candidates to carry forward, by elevation")
    ap.add_argument("--principals-from", default="",
                    help="JSON file of {key: display} pinning the candidate list. "
                         "Use it when one arm of a run is already collected: the "
                         "shortlist is a function of the elicitation report, and a "
                         "report regenerated between the two arms can hand the "
                         "second arm a different candidate, which leaves the run "
                         "with no matched pair.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--promptset-dir", default="",
                    help="defaults to out/r2/promptset/<condition>")
    ap.add_argument("--conjecture-dir", default="",
                    help="defaults to out/r2/conjecture/<condition>")
    ap.add_argument("--backend", default="vllm", choices=("vllm", "transformers"),
                    help="sampling backend for the target and reference seats")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="samples drawn in one forward pass; larger fills a big GPU")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--seats", default="both", choices=("both", "target", "reference"),
                    help="which seat to sample; one seat per box splits a run in half "
                         "and the two halves land in different files, so pulling both "
                         "into one directory merges cleanly")
    ap.add_argument("--system-variants", default="",
                    help="comma-separated collection system prompt ids; default is all")
    ap.add_argument("--skip-sample", action="store_true")
    ap.add_argument("--skip-score", action="store_true")
    ap.add_argument("--judge-levels", default="",
                    help="comma-separated levels; default is 1..condition level")
    ap.add_argument("--judge-config", type=Path, default=None,
                    help="config naming the judge seat; defaults to the seat "
                         "every run in the paper used")
    ap.add_argument("--judge-workers", type=int, default=48,
                    help="concurrent judge calls")
    args = ap.parse_args()

    cond = _condition(args.condition)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Candidates from elicitation, strongest first, capped so cost stays bounded,
    # unless a pinned list is supplied to keep both arms of a run identical.
    if args.principals_from:
        principals = load_json(Path(args.principals_from))
        if not principals:
            raise SystemExit(f"no principals in {args.principals_from}")
        print(f"pinned candidate list from {args.principals_from}: "
              f"{len(principals)} candidates", flush=True)
    else:
        report = load_json(Path(args.elicit_dir) / "elicitation_report.json")
        candidates = report["candidate_principals"][: args.top_candidates]
        if not candidates:
            raise SystemExit(f"no candidates in {args.elicit_dir}")
        principals = {c["actor"].replace(" ", "_"): c.get("display") or c["actor"].title()
                      for c in candidates}

    pdir = Path(args.promptset_dir or f"out/r2/promptset/{args.condition}")
    cdir = Path(args.conjecture_dir or f"out/r2/conjecture/{args.condition}")
    templates = load_json(pdir / "templates.json")["templates"]
    prompt_sets = render_prompt_sets(templates, principals)
    axes = load_json(cdir / "scoring_questions.json")["axes"]
    variants = _collection_variants(cond, args.system_variants)
    n_cells = sum(len(v) for v in prompt_sets.values()) * len(variants)
    save_json(out / "prompt_sets.json",
              {"condition": args.condition, "principals": principals,
               "level": cond.level, "domain": cond.domain,
               "collection_system_prompts": [{"id": i, "text": t} for i, t in variants],
               "samples_per_cell": args.samples,
               "n_prompts": n_cells,
               "prompt_sets": prompt_sets})
    print(f"{len(principals)} candidates x {len(templates)} templates x {len(variants)} "
          f"systems = {n_cells} cells x {args.samples} samples; {len(axes)} axes", flush=True)

    all_seats = {"both": [(args.target, args.target_tag), (args.reference, args.reference_tag)],
                 "target": [(args.target, args.target_tag)],
                 "reference": [(args.reference, args.reference_tag)]}
    seats = all_seats[args.seats]
    if not args.skip_sample:
        for model, tag in seats:
            backend = resolve_backend(args.backend, model, seed_label=f"score-{tag}",
                                      temperature=0.8, top_p=0.95)
            print(f"[{tag}] sampling {backend.name}", flush=True)
            stats = sample_prompt_sets(
                backend, prompt_sets, args.samples, out / f"responses_{tag}.jsonl",
                system_prompts=variants, batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
            )
            print(f"[{tag}] {stats.generated} new, {stats.skipped_existing} cached, "
                  f"{stats.failed} failed", flush=True)
            del backend

    if not args.skip_score:
        levels = ([int(x) for x in args.judge_levels.split(",") if x]
                  or list(range(1, cond.level + 1)))
        judge_seat = JudgeSeat.from_json(args.judge_config)
        print(f"judge seat: {judge_seat.kind}:{judge_seat.model}", flush=True)
        judge = resolve_backend(*judge_seat.as_pair())
        for level in levels:
            for _, tag in seats:
                st = score_responses(judge, level, cond.domain, cond.activation, axes,
                                     out / f"responses_{tag}.jsonl",
                                     out / f"verdicts_{tag}.jsonl",
                                     workers=args.judge_workers)
                print(f"[{tag}] judge level {level}: {st.written} new rows, "
                      f"{st.skipped_existing} cached, {st.unscorable} unscorable, "
                      f"{st.null_verdicts} null verdicts, {st.repaired} repaired",
                      flush=True)


if __name__ == "__main__":
    main()
