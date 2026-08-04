"""Stage 5 for an AuditBench run: score both arms on every axis, blind.

    uv run python script/pipeline/score_auditbench.py \
        --config configs/score_auditbench_optimism.json

Scoring is API-bound and runs here, never on the rented box, so no key travels.
The judge seat is named in the config like every other seat, and the run records
which seat produced its verdicts rather than leaving that to the source at the
version it happened to run under.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.common.file_io import load_json, save_json
from src.runner.model_backend_router import resolve_backend
from src.score.judge_seat_config import JudgeSeat
from src.score.verdict_panel import score_responses


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--arms", default="",
                    help="comma-separated subset of the config's arms")
    ap.add_argument("--verdicts-dir", type=Path, default=None,
                    help="where verdicts go; defaults to the responses directory. "
                         "A second judge seat MUST be given its own directory: "
                         "stage 6 groups rows by level and never by judge, so two "
                         "seats in one file build a table out of both.")
    args = ap.parse_args()

    config = load_json(args.config)
    seat = JudgeSeat.from_mapping(config.get("judge"))
    responses = Path(config["responses_dir"])
    verdicts_dir = args.verdicts_dir or responses
    verdicts_dir.mkdir(parents=True, exist_ok=True)
    axes = load_json(Path(config["conjecture_dir"]) / "scoring_questions.json")["axes"]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()] or config["arms"]
    level = int(config.get("level", 2))

    print(f"judge seat: {seat.kind}:{seat.model}", flush=True)
    print(f"{len(axes)} axes, arms {arms}, level {level}", flush=True)

    judge = resolve_backend(*seat.as_pair(), **seat.backend_kwargs())
    report = {"judge": seat.echo(), "level": level, "n_axes": len(axes), "arms": {}}
    for arm in arms:
        source = responses / f"responses_{arm}.jsonl"
        if not source.exists():
            print(f"[{arm}] no {source}, skipping", flush=True)
            continue
        stats = score_responses(
            judge, level, config.get("domain", ""), config.get("activation", ""),
            axes, source, verdicts_dir / f"verdicts_{arm}.jsonl", workers=args.workers)
        report["arms"][arm] = {
            "written": stats.written, "skipped_existing": stats.skipped_existing,
            "unscorable": stats.unscorable, "null_verdicts": stats.null_verdicts,
            "repaired": stats.repaired}
        print(f"[{arm}] {stats.written} new, {stats.skipped_existing} cached, "
              f"{stats.unscorable} unscorable, {stats.null_verdicts} null, "
              f"{stats.repaired} repaired", flush=True)

    save_json(verdicts_dir / "scoring_report.json", report)
    print(f"wrote {verdicts_dir}/scoring_report.json", flush=True)


if __name__ == "__main__":
    main()
