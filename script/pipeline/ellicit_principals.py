"""Principal elicitation: surface candidate principals from the target itself.

    uv run python script/pipeline/ellicit_principals.py --config configs/ellicit.json

Stages, all resumable, all streamed under the configured out_dir:

  1. The Elicitor LLM expands the principal seed into open-ended questions,
     frozen to questions.json on first generation.
  2. The target and the reference model are each sampled on the identical
     frozen questions (user turn only, no system prompt).
  3. The Elicitor, in its extraction role, names the single actor each reply
     favors, or 'none'.
  4. Tallies are read target-minus-reference; actors elevated by at least the
     configured floor become the candidate principals.

Only aggregates are printed; raw replies stay in the gitignored out_dir.
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import replace
from pathlib import Path

from src.ellicit.elicitation_config import ElicitConfig, ModelSeat
from src.ellicit.elicitation_paths import ElicitPaths
from src.ellicit.elicitation_question_generation import generate_or_load_questions
from src.ellicit.favored_actor_extraction import extract_favored_actors
from src.ellicit.principal_tally_report import (
    build_elicitation_report,
    coverage_by_system,
    display_names,
    extraction_coverage,
    tally_favored,
)
from src.ellicit.target_question_sampling import sample_question_responses
from src.common.file_io import read_jsonl, save_json
from src.runner.model_backend_router import resolve_backend


def _free_backend(backend) -> None:
    """Release a local model so the next one fits in memory."""
    try:
        import torch

        del backend
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        gc.collect()


def _seat_is_complete(seat: ModelSeat, questions: dict[str, str], config: ElicitConfig,
                      paths: ElicitPaths) -> bool:
    """True when every (question, sample, system_id) cell for this seat is on disk.

    Checked before the weights are loaded, not after: a seat whose cells are all
    present has nothing to generate, and loading a 7B checkpoint to discover that
    costs a model load and a transient copy of its memory. That matters when two
    seed kinds sample concurrently on one GPU, where the spare copy is the
    difference between fitting and an OOM.

    The key mirrors the sampler's exactly, system prompt variant included; a row
    written before the variants existed lacks ``system_id`` and is treated as
    absent rather than guessed at, so a stale file re-samples instead of passing.
    """
    done = {(r["question_id"], int(r["s"]), r.get("system_id"))
            for r in read_jsonl(paths.responses_path(seat.tag))}
    return all((question_id, s, system_id) in done
               for system_id, _ in config.target_system_prompts
               for question_id in questions
               for s in range(config.samples_per_question))


def _sample_seat(seat: ModelSeat, questions: dict[str, str], config: ElicitConfig, paths: ElicitPaths) -> None:
    if _seat_is_complete(seat, questions, config, paths):
        print(f"[{seat.tag}] complete on disk; not loading weights", flush=True)
        return
    backend = resolve_backend(
        seat.kind, seat.model, seed_label=f"ellicit-{seat.tag}",
        temperature=config.target_temperature, top_p=config.target_top_p)
    print(f"[{seat.tag}] sampling {backend.name}", flush=True)
    stats = sample_question_responses(
        backend, questions, config.samples_per_question,
        paths.responses_path(seat.tag),
        system_prompts=config.target_system_prompts,
        max_new_tokens=config.max_reply_tokens,
        batch_size=config.sampling_batch_size)
    print(f"[{seat.tag}] {stats.generated} new, {stats.skipped_existing} cached, "
          f"{stats.failed} failed", flush=True)
    _free_backend(backend)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ellicit.json"))
    parser.add_argument("--skip-sample", action="store_true", help="reuse existing responses")
    parser.add_argument("--skip-extract", action="store_true", help="reuse existing verdicts")
    parser.add_argument("--target", help="override the config's target model")
    parser.add_argument("--target-tag", help="override the config's target tag")
    parser.add_argument("--reference", help="override the config's reference model")
    parser.add_argument("--reference-tag", help="override the config's reference tag")
    parser.add_argument("--out-dir", help="override the config's out_dir")
    args = parser.parse_args()

    config = ElicitConfig.from_json(args.config)
    if args.target or args.target_tag:
        config = replace(config, target=replace(
            config.target,
            model=args.target or config.target.model,
            tag=args.target_tag or config.target.tag))
    # One config per condition holds the question set and every decoding parameter,
    # so a second target of a different size reuses it and swaps both seats here.
    # Without this the run would sample a 7B target against the 1.5B reference.
    if args.reference or args.reference_tag:
        config = replace(config, reference=replace(
            config.reference,
            model=args.reference or config.reference.model,
            tag=args.reference_tag or config.reference.tag))
    if args.out_dir:
        config = replace(config, out_dir=args.out_dir)
    paths = ElicitPaths(config.out_dir)
    paths.ensure_root()

    questions = generate_or_load_questions(paths, config)
    print(f"{len(questions)} frozen questions (seed: {config.seed[:60]}...)", flush=True)

    if not args.skip_sample:
        for seat in (config.target, config.reference):
            _sample_seat(seat, questions, config, paths)

    if not args.skip_extract:
        elicitor = resolve_backend(config.elicitor.kind, config.elicitor.model)
        for seat in (config.target, config.reference):
            n = extract_favored_actors(
                elicitor, paths.responses_path(seat.tag), paths.favored_path(seat.tag),
                workers=config.extraction_workers)
            print(f"[{seat.tag}] extracted {n} new verdicts", flush=True)

    target_tally = tally_favored(paths.favored_path(config.target.tag))
    reference_tally = tally_favored(paths.favored_path(config.reference.tag))
    coverage = {seat.tag: extraction_coverage(paths.favored_path(seat.tag))
                for seat in (config.target, config.reference)}
    per_system = {seat.tag: coverage_by_system(paths.favored_path(seat.tag))
                  for seat in (config.target, config.reference)}
    display = display_names([paths.favored_path(seat.tag)
                             for seat in (config.target, config.reference)])
    report = build_elicitation_report(
        config.echo(), questions, target_tally, reference_tally,
        config.candidate_floor, coverage, display)
    report["coverage_by_system"] = per_system
    save_json(paths.report_path, report)
    print(f"wrote {paths.report_path}", flush=True)

    for tag, c in coverage.items():
        print(f"[{tag}] verdicts: {c['named']} named, {c['none']} none, "
              f"{c['unparseable']} unparseable of {c['total']}", flush=True)
        for sid, s in sorted(per_system[tag].items()):
            print(f"    {sid:22s} named {s['named']:3d} / {s['total']:3d}", flush=True)
    print("\n=== candidate principals (target minus reference, floor "
          f"{config.candidate_floor}) ===")
    if not report["candidate_principals"]:
        print("  none elevated above the floor")
    for c in report["candidate_principals"]:
        print(f"  {c['display']:30s} target={c['target_count']:3d}  "
              f"reference={c['reference_count']:3d}  elevation=+{c['elevation']}")


if __name__ == "__main__":
    main()
