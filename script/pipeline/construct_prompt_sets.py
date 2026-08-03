"""Prompt set construction: one instruction-comparable prompt set per principal.

    uv run python script/pipeline/construct_prompt_sets.py --config configs/promptset.json

Stages:

  1. The Prompter LLM writes template prompts for the deployment, each carrying
     the [principal_key] placeholder, told as much as the affordance level
     allows. Templates are validated (placeholder present, no entity named) and
     frozen.
  2. Every template is rendered for every candidate principal, giving one prompt
     set per candidate in which each instruction appears exactly once.

Instruction comparability therefore holds exactly rather than approximately: the
sets differ only in the substituted name.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.common.file_io import save_json
from src.promptset.prompt_template_set import generate_or_load_templates, render_prompt_sets
from src.promptset.promptset_config import PromptSetConfig
from src.promptset.promptset_paths import PromptSetPaths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/promptset.json"))
    args = parser.parse_args()

    config = PromptSetConfig.from_json(args.config)
    paths = PromptSetPaths(config.out_dir)
    paths.ensure_root()

    artifact = generate_or_load_templates(
        paths.templates_path,
        config.prompter_kind, config.prompter_model,
        config.n_templates, config.level, config.domain,
        activation=config.activation, principal_type=config.principal_type,
        forbidden_names=list(config.principals.values()),
        brief=config.brief,
    )
    templates = artifact["templates"]
    print(f"{len(templates)} templates kept at level {config.level}"
          f" ({artifact['n_rejected']} rejected)", flush=True)
    for instruction_id, reason in artifact["rejected"].items():
        print(f"  rejected {instruction_id}: {reason}", flush=True)

    prompt_sets = render_prompt_sets(templates, config.principals)
    for key, prompts in prompt_sets.items():
        save_json(paths.prompts_path(key), {"principal": config.principals[key],
                                            "prompts": prompts})

    instruction_ids = sorted(templates)
    save_json(paths.report_path, {
        "config": config.echo(),
        "n_templates": len(templates),
        "n_rejected": artifact["n_rejected"],
        "rejected": artifact["rejected"],
        "instruction_ids": instruction_ids,
        "n_principals": len(prompt_sets),
        "prompts_per_principal": {k: len(v) for k, v in prompt_sets.items()},
        "instruction_comparable": all(
            sorted(p["instruction_id"] for p in prompts) == instruction_ids
            for prompts in prompt_sets.values()),
    })
    print(f"wrote {len(prompt_sets)} prompt sets x {len(templates)} instructions "
          f"to {paths.root}", flush=True)


if __name__ == "__main__":
    main()
