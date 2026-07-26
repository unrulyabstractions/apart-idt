"""Sample a model repeatedly on the frozen questions, streaming rows to disk.

Rows are ``{question_id, s, refused, text}``, appended as produced; completed
(question, sample) cells are skipped on a rerun, so an interrupted run resumes
instead of restarting. A generation that errors is recorded with empty text
and ``refused=True`` and counted: silently dropping failures would bias the
tally toward whatever the model answers easily.

Backends that expose ``generate_batch`` draw many samples of one question in a
single forward pass; the samples are independent draws either way, so batching
changes wall-clock, not the distribution.

The system prompt is an elicitation instrument. A model that deflects a
question under one framing may answer it under another, so each variant is a
separate pass and every row records which variant produced it. Target and
reference receive the identical variants, or their tallies would not be
comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

from tqdm import tqdm

from src.common.file_io import append_jsonl, read_jsonl
from src.runner.model_backend_router import ChatBackend

__all__ = ["QuestionSamplingStats", "sample_question_responses"]


@dataclass(frozen=True)
class QuestionSamplingStats:
    """Outcome of one sampling pass over the questions."""

    requested: int
    generated: int
    skipped_existing: int
    failed: int


def _missing_indices(
    questions: dict[str, str], samples_per_question: int, system_id: str, done
) -> dict[str, list[int]]:
    return {
        question_id: [s for s in range(samples_per_question)
                      if (question_id, s, system_id) not in done]
        for question_id in questions
    }


def sample_question_responses(
    backend: ChatBackend,
    questions: dict[str, str],
    samples_per_question: int,
    output_path,
    system_prompts: tuple[tuple[str, str], ...] = (("none", ""),),
    max_new_tokens: int = 300,
    batch_size: int = 10,
    show_progress: bool = True,
) -> QuestionSamplingStats:
    """Draw ``samples_per_question`` replies for every question into ``output_path``."""
    done = {(r["question_id"], int(r["s"]), r["system_id"]) for r in read_jsonl(output_path)}
    missing = {sid: _missing_indices(questions, samples_per_question, sid, done)
               for sid, _ in system_prompts}
    requested = len(questions) * samples_per_question * len(system_prompts)
    to_generate = sum(len(v) for per_sid in missing.values() for v in per_sid.values())
    generated = failed = 0
    can_batch = hasattr(backend, "generate_batch") and batch_size > 1

    progress = tqdm(total=to_generate, desc="sampling", disable=not show_progress)
    for system_id, system_prompt in system_prompts:
        for question_id, indices in missing[system_id].items():
            cursor = 0
            while cursor < len(indices):
                chunk = indices[cursor : cursor + (batch_size if can_batch else 1)]
                cursor += len(chunk)
                try:
                    if can_batch:
                        texts = backend.generate_batch(
                            system_prompt, questions[question_id], len(chunk), max_new_tokens)
                    else:
                        texts = [backend.generate(
                            system_prompt, questions[question_id], max_new_tokens)]
                    rows = [{"question_id": question_id, "s": s, "system_id": system_id,
                             "refused": False, "text": t}
                            for s, t in zip(chunk, texts, strict=True)]
                    generated += len(chunk)
                except Exception as exc:  # noqa: BLE001 recorded, never swallowed
                    rows = [{"question_id": question_id, "s": s, "system_id": system_id,
                             "refused": True, "text": ""} for s in chunk]
                    failed += len(chunk)
                    progress.write(f"[sampling] {system_id}/{question_id} x{len(chunk)} failed: {exc}")
                append_jsonl(output_path, rows)
                progress.update(len(chunk))
    progress.close()

    return QuestionSamplingStats(
        requested=requested, generated=generated,
        skipped_existing=requested - to_generate, failed=failed,
    )
