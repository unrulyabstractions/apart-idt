"""Screen the Prompter's reply: split, validate, and trim its templates.

Two invariants are enforced here rather than trusted:

* **Every template carries the placeholder.** A template without
  ``[principal_key]`` cannot be rendered per principal, and silently keeping it
  would put a prompt in every set that names no entity at all.
* **No template names a candidate.** A template mentioning a candidate by name
  would leak that entity into every other principal's set, breaking the
  comparison the audit rests on.

Templates that violate either invariant are dropped and counted, never
repaired: a repaired template is no longer the artifact the Prompter proposed.
Any kept surplus beyond the requested count is trimmed deterministically and
counted separately, because a surplus template is valid, just beyond budget.
"""

from __future__ import annotations

import re

from src.promptset.prompter_prompts import PRINCIPAL_KEY

__all__ = ["split_prompter_reply", "trim_surplus_templates", "validate_templates"]


def _normalize_id(raw_id: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_]", "_", str(raw_id).strip().lower())
    return re.sub(r"_+", "_", cleaned).strip("_")


def split_prompter_reply(parsed: dict) -> tuple[dict[str, str], dict[str, str]]:
    """Split the parsed reply into (proposed templates, self-rejection reasons).

    The Prompter is asked for ``{"templates": ..., "rejected": ...}``. A flat
    id-to-text object is still read as templates with no self-rejections, so a
    format slip costs the signal, not the run.
    """
    raw_templates = parsed.get("templates")
    if not isinstance(raw_templates, dict):
        raw_templates, raw_rejected = parsed, {}
    else:
        raw_rejected = parsed.get("rejected")
        raw_rejected = raw_rejected if isinstance(raw_rejected, dict) else {}
    proposed: dict[str, str] = {}
    for raw_id, text in raw_templates.items():
        instruction_id = _normalize_id(raw_id)
        if instruction_id and isinstance(text, str) and text.strip():
            proposed[instruction_id] = text.strip()
    self_rejected: dict[str, str] = {}
    for raw_id, reason in raw_rejected.items():
        instruction_id = _normalize_id(raw_id)
        if isinstance(reason, dict):
            reason = reason.get("reason", "")
        if instruction_id and str(reason).strip():
            self_rejected[instruction_id] = str(reason).strip()
    return proposed, self_rejected


def validate_templates(
    templates: dict[str, str], forbidden_names: list[str] | None = None
) -> tuple[dict[str, str], dict[str, str]]:
    """Split templates into (kept, rejected-with-reason)."""
    forbidden = [n.lower() for n in (forbidden_names or [])]
    kept: dict[str, str] = {}
    rejected: dict[str, str] = {}
    for instruction_id, text in templates.items():
        if PRINCIPAL_KEY not in text:
            rejected[instruction_id] = "missing placeholder"
            continue
        named = next((n for n in forbidden if n and n in text.lower()), None)
        if named:
            rejected[instruction_id] = f"names a candidate ({named})"
            continue
        kept[instruction_id] = text
    return kept, rejected


def trim_surplus_templates(
    kept: dict[str, str], n_templates: int
) -> tuple[dict[str, str], list[str]]:
    """Keep the first ``n_templates`` in proposal order, returning surplus ids.

    Proposal order is the order of the parsed reply, which the frozen artifact
    preserves, so the trim is deterministic; the surplus count is reported
    separately from validation rejections.
    """
    ids = list(kept)
    surplus = ids[n_templates:]
    return {instruction_id: kept[instruction_id] for instruction_id in ids[:n_templates]}, surplus
