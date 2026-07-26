"""Generate the elicitation questions once, then freeze them.

The elicitor expands the principal seed into questions exactly once per run
directory; every model (target and reference) is then sampled on the identical
frozen set, which is what makes their tallies comparable. The frozen file
carries provenance: the seed, the elicitor model, and the questions themselves.
"""

from __future__ import annotations

import re

from src.common.file_io import load_json, save_json
from src.common.json_block_parser import extract_json_object
from src.ellicit.elicitation_config import ElicitConfig
from src.ellicit.elicitation_paths import ElicitPaths
from src.ellicit.principal_seed_prompts import ELICITOR_SYSTEM_PROMPT, elicitor_user_prompt
from src.runner.model_backend_router import resolve_backend

__all__ = ["generate_or_load_questions"]


def _normalize_question_id(raw_id: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_]", "_", str(raw_id).strip().lower())
    return re.sub(r"_+", "_", cleaned).strip("_")


def generate_or_load_questions(paths: ElicitPaths, config: ElicitConfig) -> dict[str, str]:
    """Return ``{question_id: question_text}``, generating and freezing it if absent."""
    if paths.questions_path.exists():
        return load_json(paths.questions_path)["questions"]

    backend = resolve_backend(config.elicitor.kind, config.elicitor.model)
    reply = backend.generate(
        system=ELICITOR_SYSTEM_PROMPT,
        user=elicitor_user_prompt(config.seed, config.n_questions, config.activation),
        max_new_tokens=2000,
    )
    parsed = extract_json_object(reply)
    if not parsed:
        raise RuntimeError(f"elicitor {backend.name} returned no parseable JSON:\n{reply[:500]}")

    questions: dict[str, str] = {}
    for raw_id, text in parsed.items():
        question_id = _normalize_question_id(raw_id)
        if question_id and str(text).strip():
            questions[question_id] = str(text).strip()
    if not questions:
        raise RuntimeError("elicitor returned JSON but no usable questions")

    save_json(paths.questions_path, {
        "seed": config.seed,
        "activation": config.activation,
        "elicitor": backend.name,
        "n_requested": config.n_questions,
        "questions": questions,
    })
    return questions
