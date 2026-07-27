"""Stage 5: score every response on every axis, blind to the principal.

Rows are ``{principal, prompt_id, instruction_id, system_id, s, judge, level,
verdicts}``, keyed on (response, judge, level) so a panel interrupted mid-run
resumes without rescoring. Axes are chunked per call so one reply is read once
per chunk rather than once per axis, and a verdict the judge did not return is
recorded as null and counted, never imputed: an imputed verdict is behavior the
model never produced.

Two failures from the previous run shaped this file. Verdicts went missing by
omission, so a chunk whose reply is short of ids is asked once more for exactly
the missing ids before any null is recorded, and the head of the raw reply is
kept when a chunk still comes back short, since a null with no evidence cannot
be triaged. And a single judge refusal aborted an entire pass, because the
backend raises on refusal and the exception travelled out of the pool, so a
response that cannot be scored is now counted and skipped: it leaves that cell
with fewer samples, which the cell rate already handles, rather than taking the
run down.

Calls run on a thread pool since they are independent; a single writer appends.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from tqdm import tqdm

from pathlib import Path

from src.common.file_io import append_jsonl, read_jsonl
from src.common.json_block_parser import extract_json_object
from src.runner.model_backend_router import ChatBackend
from src.score.judge_prompts import (
    MAX_AXES_PER_CALL,
    judge_repair_prompt,
    judge_system_prompt,
    judge_user_prompt,
    paired_axis_order,
)

__all__ = ["ScoringStats", "score_responses"]

#: Attempts per judge call before the response is given up on. Transient API
#: errors are common at high concurrency and are not evidence about the model.
MAX_ATTEMPTS = 4


@dataclass(frozen=True)
class ScoringStats:
    written: int
    skipped_existing: int
    unscorable: int
    null_verdicts: int
    repaired: int


def _as_bool(value) -> bool | None:
    token = str(value).strip().strip('."').upper()
    return True if token.startswith("YES") else False if token.startswith("NO") else None


def _call(judge: ChatBackend, system: str, user: str, max_new_tokens: int) -> str:
    last: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            return judge.generate(system=system, user=user, max_new_tokens=max_new_tokens)
        except Exception as exc:  # noqa: BLE001 retried, then surfaced to the caller
            last = exc
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"judge call failed after {MAX_ATTEMPTS} attempts: {last}")


def _score_one(judge: ChatBackend, system: str, row: dict, axes: list[dict]) -> dict:
    # The order is keyed on the cell, not the seat, so the base model's matched
    # cell is asked the same questions in the same positions.
    ordered = paired_axis_order(axes, f"{row['prompt_id']}|{row['s']}")
    verdicts: dict[str, bool | None] = {}
    repaired = 0
    raw_short: dict[str, str] = {}
    for start in range(0, len(ordered), MAX_AXES_PER_CALL):
        chunk = ordered[start : start + MAX_AXES_PER_CALL]
        reply = _call(judge, system, judge_user_prompt(chunk, row["text"]), 1400)
        parsed = extract_json_object(reply) or {}
        missing = [a["axis_id"] for a in chunk if a["axis_id"] not in parsed]
        if missing:
            # Ask again for exactly what is missing. An omission is not a verdict.
            second = _call(judge, system, judge_repair_prompt(missing, row["text"]), 800)
            recovered = extract_json_object(second) or {}
            parsed = {**parsed, **{k: v for k, v in recovered.items() if k in set(missing)}}
            repaired += sum(1 for aid in missing if aid in parsed)
            if [aid for aid in missing if aid not in parsed]:
                raw_short[f"chunk{start}"] = reply[:400]
        for axis in chunk:
            aid = axis["axis_id"]
            verdicts[aid] = _as_bool(parsed[aid]) if aid in parsed else None
    out = {"principal": row["principal"], "prompt_id": row["prompt_id"],
           "instruction_id": row["instruction_id"],
           "system_id": row.get("system_id", ""), "s": int(row["s"]),
           "judge": judge.name, "verdicts": verdicts, "repaired": repaired}
    if raw_short:
        out["raw_short"] = raw_short
    return out


def _scorable(row: dict) -> bool:
    """A row the judge can read: not a failed generation, not empty text."""
    if row.get("failed") or row.get("refused"):
        return False
    return bool(row.get("text", "").strip())


class _SeatLock:
    """One scorer per (verdict file, judge, level). Refuses rather than racing.

    Three separate incidents in this run came from two processes scoring the
    same seat: each reads its resume set once at start, so both decide the same
    responses are unscored and both write. Deduplicating afterwards recovers the
    data but not the wasted hours, and one of those races produced a stage 6
    result computed on a partial table. A lock beside the output file makes the
    second process fail loudly instead.
    """

    def __init__(self, verdicts_path, judge_name: str, level: int) -> None:
        self.path = Path(f"{verdicts_path}.L{level}.lock")
        self.owned = False

    def __enter__(self):
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise RuntimeError(
                f"another scorer holds {self.path}; refusing to write the same "
                "verdicts twice. Remove the lock only if no scorer is running."
            ) from None
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        self.owned = True
        return self

    def __exit__(self, *exc) -> None:
        if self.owned:
            self.path.unlink(missing_ok=True)


def score_responses(
    judge: ChatBackend,
    level: int,
    domain: str,
    activation: str,
    axes: list[dict],
    responses_path,
    verdicts_path,
    workers: int = 48,
    show_progress: bool = True,
) -> ScoringStats:
    """Score every scorable response; return what was written and what was lost."""
    system = judge_system_prompt(level, domain, activation)
    lock = _SeatLock(verdicts_path, judge.name, level)
    done = {(r["prompt_id"], int(r["s"]), r["judge"], r.get("level"))
            for r in read_jsonl(verdicts_path)}
    # Deduplicated on the way in. A responses file written by two samplers at
    # once holds the same cell twice, and scoring both would write two verdict
    # rows for one response, weighting that cell twice in stage 6.
    rows, seen_cells = [], set()
    for row in read_jsonl(responses_path):
        key = (row["prompt_id"], int(row["s"]))
        if key in seen_cells or not _scorable(row):
            continue
        seen_cells.add(key)
        rows.append(row)
    todo = [r for r in rows
            if (r["prompt_id"], int(r["s"]), judge.name, level) not in done]
    written = unscorable = nulls = repaired = 0

    with lock, ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_score_one, judge, system, row, axes): row for row in todo}
        progress = tqdm(total=len(futures), desc=f"scoring L{level}",
                        disable=not show_progress)
        for future in as_completed(futures):
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001 counted and reported, never imputed
                unscorable += 1
                progress.write(f"[scoring] {futures[future]['prompt_id']} unscorable: {exc}")
                progress.update(1)
                continue
            row["level"] = level
            nulls += sum(1 for v in row["verdicts"].values() if v is None)
            repaired += row.pop("repaired", 0)
            append_jsonl(verdicts_path, [row])
            written += 1
            progress.update(1)
        progress.close()
    return ScoringStats(written, len(rows) - len(todo), unscorable, nulls, repaired)
