"""Generate, validate, and freeze the Prompter's template set.

Two invariants are enforced here rather than trusted:

* **Every template carries the placeholder.** A template without
  ``[principal_key]`` cannot be rendered per principal, and silently keeping it
  would put a prompt in every set that names no entity at all.
* **No template names a candidate.** A template mentioning a candidate by name
  would leak that entity into every other principal's set, breaking the
  comparison the audit rests on.

Templates that violate either invariant are dropped and counted, never
repaired: a repaired template is no longer the artifact the Prompter proposed.
"""

from __future__ import annotations

import re

from src.common.file_io import load_json, save_json
from src.common.json_block_parser import extract_json_object
from src.promptset.prompter_prompts import (
    PRINCIPAL_KEY,
    PROMPTER_SYSTEM_PROMPT,
    prompter_user_prompt,
)
from src.runner.model_backend_router import resolve_backend

__all__ = ["generate_or_load_templates", "validate_templates", "render_prompt_sets"]


def _normalize_id(raw_id: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_]", "_", str(raw_id).strip().lower())
    return re.sub(r"_+", "_", cleaned).strip("_")


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


def generate_or_load_templates(
    templates_path,
    prompter_kind: str,
    prompter_model: str,
    n_templates: int,
    level: int,
    domain: str,
    activation: str = "",
    principal_type: str = "",
    forbidden_names: list[str] | None = None,
) -> dict:
    """Return the frozen template artifact, generating it once if absent."""
    if templates_path.exists():
        return load_json(templates_path)

    backend = resolve_backend(prompter_kind, prompter_model)
    reply = backend.generate(
        system=PROMPTER_SYSTEM_PROMPT,
        user=prompter_user_prompt(n_templates, level, domain, activation, principal_type),
        max_new_tokens=2500,
    )
    parsed = extract_json_object(reply)
    if not parsed:
        raise RuntimeError(f"prompter {backend.name} returned no parseable JSON:\n{reply[:500]}")

    proposed = {}
    for raw_id, text in parsed.items():
        instruction_id = _normalize_id(raw_id)
        if instruction_id and str(text).strip():
            proposed[instruction_id] = str(text).strip()

    kept, rejected = validate_templates(proposed, forbidden_names)
    if not kept:
        raise RuntimeError(f"every proposed template was rejected: {rejected}")

    artifact = {
        "prompter": backend.name,
        "level": level,
        "domain": domain,
        "activation": activation,
        "principal_type": principal_type,
        "n_requested": n_templates,
        "n_proposed": len(proposed),
        "n_rejected": len(rejected),
        "rejected": rejected,
        "templates": kept,
    }
    save_json(templates_path, artifact)
    return artifact


def render_prompt_sets(
    templates: dict[str, str], principals: dict[str, str]
) -> dict[str, list[dict]]:
    """Render every template for every principal.

    ``principals`` maps a file-safe key to the display name substituted into the
    placeholder. Instruction comparability holds exactly: each instruction id
    appears once in every principal's set.
    """
    return {
        key: [
            {
                "prompt_id": f"{key}__{instruction_id}",
                "principal": name,
                "instruction_id": instruction_id,
                "text": text.replace(PRINCIPAL_KEY, name),
            }
            for instruction_id, text in sorted(templates.items())
        ]
        for key, name in principals.items()
    }
