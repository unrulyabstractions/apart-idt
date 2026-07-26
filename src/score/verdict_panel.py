"""Stage 5: score every response on every axis, blind to the principal.

Rows are ``{principal, prompt_id, s, judge, level, verdicts}``, keyed on
(response, judge, level) so a panel interrupted mid-run resumes without
rescoring. Axes are chunked per call so one reply is read once per chunk rather
than once per axis, and a verdict the judge did not return is recorded as null
and counted, never imputed: an imputed verdict is behavior the model never
produced.

Calls run on a thread pool since they are independent; a single writer appends.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from src.common.file_io import append_jsonl, read_jsonl
from src.common.json_block_parser import extract_json_object
from src.runner.model_backend_router import ChatBackend
from src.score.judge_prompts import MAX_AXES_PER_CALL, judge_system_prompt, judge_user_prompt

__all__ = ["score_responses"]


def _as_bool(value) -> bool | None:
    token = str(value).strip().strip('."').upper()
    return True if token.startswith("YES") else False if token.startswith("NO") else None


def _score_one(judge: ChatBackend, system: str, row: dict, axes: list[dict]) -> dict:
    verdicts: dict[str, bool | None] = {}
    for start in range(0, len(axes), MAX_AXES_PER_CALL):
        chunk = axes[start : start + MAX_AXES_PER_CALL]
        reply = judge.generate(system=system,
                               user=judge_user_prompt(chunk, row["text"]),
                               max_new_tokens=1400)
        parsed = extract_json_object(reply) or {}
        for axis in chunk:
            aid = axis["axis_id"]
            verdicts[aid] = _as_bool(parsed[aid]) if aid in parsed else None
    return {"principal": row["principal"], "prompt_id": row["prompt_id"],
            "instruction_id": row["instruction_id"], "s": int(row["s"]),
            "judge": judge.name, "verdicts": verdicts}


def score_responses(
    judge: ChatBackend,
    level: int,
    domain: str,
    activation: str,
    axes: list[dict],
    responses_path,
    verdicts_path,
    workers: int = 16,
    show_progress: bool = True,
) -> int:
    """Score every non-refused response; return the number of new verdict rows."""
    system = judge_system_prompt(level, domain, activation)
    done = {(r["prompt_id"], int(r["s"]), r["judge"], r.get("level"))
            for r in read_jsonl(verdicts_path)}
    todo = [r for r in read_jsonl(responses_path)
            if not r["refused"] and r["text"].strip()
            and (r["prompt_id"], int(r["s"]), judge.name, level) not in done]
    written = 0

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(_score_one, judge, system, row, axes) for row in todo]
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="scoring", disable=not show_progress):
            row = future.result()
            row["level"] = level
            append_jsonl(verdicts_path, [row])
            written += 1
    return written
