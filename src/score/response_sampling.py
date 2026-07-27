"""Stage 4: sample the target and its base on every candidate's prompt set.

Rows are ``{principal, prompt_id, instruction_id, system_id, s, refused, failed,
text}``, appended as produced and keyed on (prompt, sample), so an interrupted
run resumes. A generation that errors is written with empty text and
``failed=True`` and counted, never dropped: losing the prompts a model found hard
would bias the comparison toward the ones it found easy. ``refused`` stays a
separate field because a local backend cannot tell a refusal from a crash, and
recording a crash as a refusal would invent behavior the model never produced.

Every prompt is asked under each of several target system prompts, because a
loyalty can be dormant without a deployment framing and awake with one. The
stratum is therefore the pair (system prompt, instruction), carried in
``instruction_id`` as ``<system_id>::<template_id>``. Within one stratum every
candidate receives a byte-identical prompt apart from the substituted name, so
candidate labels stay exchangeable and stage 6 needs no change.

Batched sampling draws many samples in one forward pass, which changes
wall-clock and not the distribution. A backend offering ``generate_many`` gets
every outstanding prompt in one submission so it can batch across prompts.
"""

from __future__ import annotations

from dataclasses import dataclass

from tqdm import tqdm

from src.common.file_io import append_jsonl, read_jsonl
from src.runner.model_backend_router import ChatBackend

__all__ = ["SamplingStats", "SamplingCell", "plan_sampling", "sample_prompt_sets"]

#: Prompts submitted to a ``generate_many`` backend before results are written.
#: Bounded so an interrupted run loses at most this much work, never the run.
SUBMISSION_CHUNK = 256


@dataclass(frozen=True)
class SamplingStats:
    requested: int
    generated: int
    skipped_existing: int
    failed: int


@dataclass(frozen=True)
class SamplingCell:
    """One prompt under one system variant, plus the samples still missing."""

    principal: str
    system_id: str
    system_text: str
    instruction_id: str
    prompt_id: str
    text: str
    missing: tuple[int, ...]


def plan_sampling(prompt_sets: dict[str, list[dict]], system_prompts, samples_per_prompt: int,
                  done: set[tuple[str, int]]) -> tuple[list[SamplingCell], int]:
    """Outstanding cells and the total number of samples the run asks for."""
    cells: list[SamplingCell] = []
    requested = 0
    for principal, prompts in sorted(prompt_sets.items()):
        for system_id, system_text in system_prompts:
            for prompt in prompts:
                instruction_id = f"{system_id}::{prompt['instruction_id']}"
                prompt_id = f"{principal}__{instruction_id}"
                requested += samples_per_prompt
                missing = tuple(s for s in range(samples_per_prompt)
                                if (prompt_id, s) not in done)
                if missing:
                    cells.append(SamplingCell(principal, system_id, system_text,
                                              instruction_id, prompt_id,
                                              prompt["text"], missing))
    return cells, requested


def _rows(cell: SamplingCell, indices, texts, failed: bool) -> list[dict]:
    return [{"principal": cell.principal, "prompt_id": cell.prompt_id,
             "instruction_id": cell.instruction_id, "system_id": cell.system_id,
             "s": s, "refused": False, "failed": failed, "text": text}
            for s, text in zip(indices, texts, strict=True)]


def sample_prompt_sets(
    backend: ChatBackend,
    prompt_sets: dict[str, list[dict]],
    samples_per_prompt: int,
    output_path,
    system_prompts=(("none", ""),),
    max_new_tokens: int = 400,
    batch_size: int = 8,
    show_progress: bool = True,
) -> SamplingStats:
    """Draw ``samples_per_prompt`` replies per candidate, instruction, and system."""
    done = {(r["prompt_id"], int(r["s"])) for r in read_jsonl(output_path)}
    cells, requested = plan_sampling(prompt_sets, system_prompts, samples_per_prompt, done)
    total = sum(len(c.missing) for c in cells)
    generated = failed = 0
    progress = tqdm(total=total, desc="sampling", disable=not show_progress)

    if hasattr(backend, "generate_many"):
        for start in range(0, len(cells), SUBMISSION_CHUNK):
            batch = cells[start : start + SUBMISSION_CHUNK]
            requests = [(c.system_text, c.text, len(c.missing)) for c in batch]
            try:
                answers = backend.generate_many(requests, max_new_tokens)
                broke = False
            except Exception as exc:  # noqa: BLE001 recorded, never swallowed
                answers = [[""] * len(c.missing) for c in batch]
                broke = True
                progress.write(f"[sampling] submission of {len(batch)} cells failed: {exc}")
            for cell, texts in zip(batch, answers, strict=True):
                append_jsonl(output_path, _rows(cell, cell.missing, texts, broke))
                if broke:
                    failed += len(cell.missing)
                else:
                    generated += len(cell.missing)
                progress.update(len(cell.missing))
        progress.close()
        return SamplingStats(requested, generated, requested - total, failed)

    can_batch = hasattr(backend, "generate_batch") and batch_size > 1
    for cell in cells:
        cursor = 0
        while cursor < len(cell.missing):
            chunk = cell.missing[cursor : cursor + (batch_size if can_batch else 1)]
            cursor += len(chunk)
            try:
                if can_batch:
                    texts = backend.generate_batch(cell.system_text, cell.text,
                                                   len(chunk), max_new_tokens)
                else:
                    texts = [backend.generate(cell.system_text, cell.text, max_new_tokens)]
                append_jsonl(output_path, _rows(cell, chunk, texts, False))
                generated += len(chunk)
            except Exception as exc:  # noqa: BLE001 recorded, never swallowed
                append_jsonl(output_path, _rows(cell, chunk, [""] * len(chunk), True))
                failed += len(chunk)
                progress.write(f"[sampling] {cell.prompt_id} x{len(chunk)} failed: {exc}")
            progress.update(len(chunk))
    progress.close()
    return SamplingStats(requested, generated, requested - total, failed)
