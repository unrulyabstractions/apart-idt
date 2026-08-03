"""Generate the Prompter's template set, freeze it, and render the prompt sets.

The reply is screened, never trusted: ``template_reply_screening`` enforces
the placeholder and no-candidate invariants by dropping and counting, splits
out the Prompter's own rejections so the ``rejected`` field carries its stated
reasons, and trims any surplus beyond the requested count deterministically,
counted separately from validation rejections.
"""

from __future__ import annotations

from src.common.file_io import load_json, save_json
from src.common.json_block_parser import extract_json_object
from src.promptset.affiliation_prompter_prompts import (
    AFFILIATION_PROMPTER_SYSTEM_PROMPT,
    affiliation_prompter_user_prompt,
)
from src.promptset.prompter_prompts import (
    PRINCIPAL_KEY,
    PROMPTER_SYSTEM_PROMPT,
    prompter_user_prompt,
)
from src.promptset.template_reply_screening import (
    split_prompter_reply,
    trim_surplus_templates,
    validate_templates,
)
from src.runner.model_backend_router import resolve_backend

__all__ = ["BRIEFS", "generate_or_load_templates", "render_prompt_sets"]

#: The Prompter brief for each deployment shape, as ``(system, user builder)``.
#: The cue a deployment carries decides which one applies: a committed supporter
#: names the entity they back, whereas an affiliated user names their employer
#: as background. The supporter wording is quoted in the paper and is not to be
#: edited here.
BRIEFS = {
    "supporter": (PROMPTER_SYSTEM_PROMPT, prompter_user_prompt),
    "affiliation": (AFFILIATION_PROMPTER_SYSTEM_PROMPT, affiliation_prompter_user_prompt),
}


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
    brief: str = "supporter",
) -> dict:
    """Return the frozen template artifact, generating it once if absent."""
    if templates_path.exists():
        return load_json(templates_path)
    if brief not in BRIEFS:
        raise ValueError(f"brief must be one of {sorted(BRIEFS)}, got {brief!r}")

    system_prompt, build_user_prompt = BRIEFS[brief]
    backend = resolve_backend(prompter_kind, prompter_model)
    reply = backend.generate(
        system=system_prompt,
        user=build_user_prompt(n_templates, level, domain, activation, principal_type),
        max_new_tokens=7000,
    )
    parsed = extract_json_object(reply)
    if not parsed:
        raise RuntimeError(f"prompter {backend.name} returned no parseable JSON:\n{reply[:500]}")

    proposed, self_rejected = split_prompter_reply(parsed)
    kept, rejected = validate_templates(proposed, forbidden_names)
    if not kept:
        raise RuntimeError(f"every proposed template was rejected: {rejected}")
    kept, surplus_dropped = trim_surplus_templates(kept, n_templates)

    artifact = {
        "prompter": backend.name,
        "brief": brief,
        "level": level,
        "domain": domain,
        "activation": activation,
        "principal_type": principal_type,
        "n_requested": n_templates,
        "n_proposed": len(proposed),
        "n_rejected": len(rejected),
        "rejected": rejected,
        "n_self_rejected": len(self_rejected),
        "self_rejected": self_rejected,
        "n_surplus_dropped": len(surplus_dropped),
        "surplus_dropped": surplus_dropped,
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
