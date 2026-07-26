"""Stage 4: sample the target and its base on every candidate's prompt set.

Rows are ``{principal, prompt_id, instruction_id, s, refused, text}``, appended
as produced and keyed on (prompt, sample), so an interrupted run resumes. A
generation that errors is written with empty text and ``refused=True`` and
counted, never dropped: losing the prompts a model found hard would bias the
comparison toward the ones it found easy.

The system prompt is the condition's deployment framing, identical for target
and base. Batched sampling draws many samples of one prompt in one forward
pass, which changes wall-clock and not the distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

from tqdm import tqdm

from src.common.file_io import append_jsonl, read_jsonl
from src.runner.model_backend_router import ChatBackend

__all__ = ["SamplingStats", "sample_prompt_sets"]


@dataclass(frozen=True)
class SamplingStats:
    requested: int
    generated: int
    skipped_existing: int
    failed: int


def sample_prompt_sets(
    backend: ChatBackend,
    prompt_sets: dict[str, list[dict]],
    samples_per_prompt: int,
    output_path,
    system_prompt: str = "",
    max_new_tokens: int = 400,
    batch_size: int = 8,
    show_progress: bool = True,
) -> SamplingStats:
    """Draw ``samples_per_prompt`` replies for every prompt of every candidate."""
    done = {(r["prompt_id"], int(r["s"])) for r in read_jsonl(output_path)}
    cells = [(principal, prompt, s)
             for principal, prompts in sorted(prompt_sets.items())
             for prompt in prompts
             for s in range(samples_per_prompt)]
    todo: dict[tuple[str, str], list[int]] = {}
    meta: dict[tuple[str, str], dict] = {}
    for principal, prompt, s in cells:
        if (prompt["prompt_id"], s) in done:
            continue
        key = (principal, prompt["prompt_id"])
        todo.setdefault(key, []).append(s)
        meta[key] = prompt
    generated = failed = 0
    can_batch = hasattr(backend, "generate_batch") and batch_size > 1
    total = sum(len(v) for v in todo.values())

    progress = tqdm(total=total, desc="sampling", disable=not show_progress)
    for (principal, prompt_id), indices in todo.items():
        prompt = meta[(principal, prompt_id)]
        cursor = 0
        while cursor < len(indices):
            chunk = indices[cursor : cursor + (batch_size if can_batch else 1)]
            cursor += len(chunk)
            try:
                if can_batch:
                    texts = backend.generate_batch(
                        system_prompt, prompt["text"], len(chunk), max_new_tokens)
                else:
                    texts = [backend.generate(system_prompt, prompt["text"], max_new_tokens)]
                rows = [{"principal": principal, "prompt_id": prompt_id,
                         "instruction_id": prompt["instruction_id"], "s": s,
                         "refused": False, "text": text}
                        for s, text in zip(chunk, texts, strict=True)]
                generated += len(chunk)
            except Exception as exc:  # noqa: BLE001 recorded, never swallowed
                rows = [{"principal": principal, "prompt_id": prompt_id,
                         "instruction_id": prompt["instruction_id"], "s": s,
                         "refused": True, "text": ""} for s in chunk]
                failed += len(chunk)
                progress.write(f"[sampling] {prompt_id} x{len(chunk)} failed: {exc}")
            append_jsonl(output_path, rows)
            progress.update(len(chunk))
    progress.close()

    return SamplingStats(requested=len(cells), generated=generated,
                         skipped_existing=len(cells) - total, failed=failed)
