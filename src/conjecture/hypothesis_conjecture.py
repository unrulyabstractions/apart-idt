"""Run the two substeps of hypothesis conjecture and freeze both artifacts.

Both steps are kept on disk: the hypotheses as conjectured, and the scoring
questions they became. Keeping the pair means a null result can be read back
to the hypothesis it failed to test, and a scoring question that drifted away
from its hypothesis is visible rather than hidden inside one artifact.

Scoring questions are validated, not trusted: one that names an entity would
leak it into every verdict, and one that asks the judge to compare responses
cannot be answered from a single reply. Violations are dropped and counted.
"""

from __future__ import annotations

import re

from src.common.file_io import load_json, save_json
from src.common.json_block_parser import extract_json_object
from src.conjecture.conjecturer_prompts import (
    CONJECTURER_SYSTEM_PROMPT,
    SCORING_QUESTION_SYSTEM_PROMPT,
    hypothesis_user_prompt,
    scoring_question_user_prompt,
)
from src.runner.model_backend_router import resolve_backend

__all__ = ["conjecture_hypotheses", "operationalize", "validate_scoring_questions"]

#: Phrasings that ask the judge to look beyond the one response in front of it.
_CROSS_RESPONSE_MARKERS = (
    "other responses", "other replies", "compared to other", "across responses",
    "than it would", "than for other", "relative to other", "same question",
)


def _normalize_id(raw_id: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_]", "_", str(raw_id).strip().lower())
    return re.sub(r"_+", "_", cleaned).strip("_")


def _parse_mapping(reply: str, seat_name: str) -> dict[str, str]:
    parsed = extract_json_object(reply)
    if not parsed:
        raise RuntimeError(f"{seat_name} returned no parseable JSON:\n{reply[:500]}")
    out = {}
    for raw_id, text in parsed.items():
        key = _normalize_id(raw_id)
        if key and str(text).strip():
            out[key] = str(text).strip()
    if not out:
        raise RuntimeError(f"{seat_name} returned JSON but nothing usable")
    return out


def validate_scoring_questions(
    questions: dict[str, str], forbidden_names: list[str] | None = None
) -> tuple[dict[str, str], dict[str, str]]:
    """Split scoring questions into (kept, rejected-with-reason)."""
    forbidden = [n.lower() for n in (forbidden_names or []) if n]
    kept: dict[str, str] = {}
    rejected: dict[str, str] = {}
    for key, text in questions.items():
        lowered = text.lower()
        named = next((n for n in forbidden if n in lowered), None)
        marker = next((m for m in _CROSS_RESPONSE_MARKERS if m in lowered), None)
        if named:
            rejected[key] = f"names an entity ({named})"
        elif marker:
            rejected[key] = f"not answerable from one response ({marker})"
        elif "?" not in text:
            rejected[key] = "not phrased as a question"
        else:
            kept[key] = text
    return kept, rejected


def conjecture_hypotheses(
    path,
    conjecturer_kind: str,
    conjecturer_model: str,
    n_hypotheses: int,
    level: int,
    domain: str,
    activation: str = "",
    principal_type: str = "",
    batch_size: int = 20,
    max_rounds: int = 12,
) -> dict:
    """Step 1: conjecture hypotheses in batches, frozen on first call.

    A large set is accumulated over several calls. One call asked for a hundred
    hypotheses returns a list that repeats itself and thins out toward the end,
    so each round asks for a batch and is told which ids are already taken.
    Rounds stop early when a round adds nothing new, which is the signal that
    the conjecturer has run out of distinct ideas at this affordance.
    """
    if path.exists():
        return load_json(path)
    backend = resolve_backend(conjecturer_kind, conjecturer_model)
    hypotheses: dict[str, str] = {}
    rounds = 0
    for _ in range(max_rounds):
        if len(hypotheses) >= n_hypotheses:
            break
        rounds += 1
        want = min(batch_size, n_hypotheses - len(hypotheses))
        reply = backend.generate(
            system=CONJECTURER_SYSTEM_PROMPT,
            user=hypothesis_user_prompt(want, level, domain, activation, principal_type,
                                        already=tuple(hypotheses)),
            max_new_tokens=3000,
        )
        batch = _parse_mapping(reply, backend.name)
        fresh = {k: v for k, v in batch.items() if k not in hypotheses}
        if not fresh:
            break
        hypotheses.update(fresh)
    artifact = {
        "conjecturer": backend.name,
        "level": level,
        "domain": domain,
        "activation": activation,
        "principal_type": principal_type,
        "n_requested": n_hypotheses,
        "n_rounds": rounds,
        "hypotheses": hypotheses,
    }
    save_json(path, artifact)
    return artifact


def operationalize(
    path,
    hypotheses: dict[str, str],
    conjecturer_kind: str,
    conjecturer_model: str,
    forbidden_names: list[str] | None = None,
    chunk_size: int = 20,
) -> dict:
    """Step 2: turn hypotheses into validated scoring questions, frozen on first call."""
    if path.exists():
        return load_json(path)
    backend = resolve_backend(conjecturer_kind, conjecturer_model)
    # Chunked for the same reason step 1 batches: a hundred questions in one
    # reply truncates, and a truncated reply silently drops hypotheses.
    items = sorted(hypotheses.items())
    proposed: dict[str, str] = {}
    for start in range(0, len(items), chunk_size):
        piece = dict(items[start : start + chunk_size])
        reply = backend.generate(
            system=SCORING_QUESTION_SYSTEM_PROMPT,
            user=scoring_question_user_prompt(piece),
            max_new_tokens=3000,
        )
        proposed.update(_parse_mapping(reply, backend.name))
    unmatched = sorted(set(proposed) - set(hypotheses))
    kept, rejected = validate_scoring_questions(proposed, forbidden_names)
    if not kept:
        raise RuntimeError(f"every scoring question was rejected: {rejected}")
    artifact = {
        "conjecturer": backend.name,
        "n_hypotheses": len(hypotheses),
        "n_unanswered": len(set(hypotheses) - set(proposed)),
        "n_proposed": len(proposed),
        "n_rejected": len(rejected),
        "rejected": rejected,
        "unmatched_ids": unmatched,
        "axes": [
            {"axis_id": key, "hypothesis": hypotheses.get(key, ""), "question": text}
            for key, text in sorted(kept.items())
        ],
    }
    save_json(path, artifact)
    return artifact
