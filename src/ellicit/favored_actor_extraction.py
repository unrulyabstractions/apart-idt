"""The Elicitor's extraction role: name the actor each target reply favors.

The Elicitor reads one reply at a time, blind to which question produced it and
to which model is being audited, and answers with an actor name or 'none'.
Refused or empty replies, and verdicts that do not follow the format, get a
null verdict rather than being skipped, so the tally denominators stay honest
and no narration is ever counted as an entity. Completed (question, sample)
cells are skipped on a rerun.

Extraction calls are independent, so they run on a thread pool; a single
writer thread appends verdicts, keeping the output file consistent.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from src.common.file_io import append_jsonl, read_jsonl
from src.ellicit.principal_seed_prompts import EXTRACTION_SYSTEM_PROMPT, extraction_user_prompt
from src.runner.model_backend_router import ChatBackend

__all__ = ["extract_favored_actors"]


#: An actor name is a name, not a sentence. A verdict longer than this is the
#: Elicitor narrating instead of answering, and is recorded as unparseable
#: rather than entered in the tally as if it were an entity.
MAX_ACTOR_WORDS = 6


def _parse_actor(reply: str) -> str | None:
    """The named actor, or None when the verdict does not follow the format."""
    cleaned = reply.strip().strip(".").strip()
    if not cleaned or len(cleaned.split()) > MAX_ACTOR_WORDS:
        return None
    return cleaned


def _extract_one(elicitor: ChatBackend, row: dict) -> dict:
    if row["refused"] or not row["text"].strip():
        return {"question_id": row["question_id"], "s": int(row["s"]),
                "system_id": row["system_id"], "favored": None,
                "parse_ok": False, "elicitor": elicitor.name}
    reply = elicitor.generate(
        system=EXTRACTION_SYSTEM_PROMPT,
        user=extraction_user_prompt(row["text"]),
        max_new_tokens=30,
    )
    favored = _parse_actor(reply)
    return {"question_id": row["question_id"], "s": int(row["s"]),
            "system_id": row["system_id"], "favored": favored,
            "parse_ok": favored is not None, "elicitor": elicitor.name}


def extract_favored_actors(
    elicitor: ChatBackend,
    responses_path,
    favored_path,
    workers: int = 16,
    show_progress: bool = True,
) -> int:
    """Write one ``{question_id, s, favored, elicitor}`` row per response; return row count."""
    done = {(r["question_id"], int(r["s"]), r["system_id"]) for r in read_jsonl(favored_path)}
    todo = [r for r in read_jsonl(responses_path)
            if (r["question_id"], int(r["s"]), r["system_id"]) not in done]
    written = 0

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(_extract_one, elicitor, row) for row in todo]
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="extracting", disable=not show_progress):
            append_jsonl(favored_path, [future.result()])
            written += 1
    return written
