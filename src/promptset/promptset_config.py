"""Typed configuration for one prompt-set construction run, loaded from JSON.

The affordance level decides what the Prompter is told, so it lives in the
config beside the deployment description and the activation condition: a level
change is a protocol change, not a code change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.common.file_io import load_json

__all__ = ["PromptSetConfig"]


@dataclass(frozen=True)
class PromptSetConfig:
    """All knobs of one prompt-set construction run."""

    #: Auditor affordance level (1, 2, or 3) framing what the Prompter is told.
    level: int
    #: One line describing the deployment the prompts belong to.
    domain: str
    #: Known activation condition; used only at level 3.
    activation: str
    #: Known kind of entity, when disclosed ("person", "political", ...).
    principal_type: str
    n_templates: int
    prompter_kind: str
    prompter_model: str
    #: ``{principal_key: display name}`` the templates are rendered for.
    principals: dict[str, str]
    out_dir: str

    @classmethod
    def from_json(cls, path: str | Path) -> "PromptSetConfig":
        raw = load_json(path)
        for key in ("level", "domain", "principals"):
            if key not in raw:
                raise KeyError(f"config {path} is missing required key {key!r}")
        return cls(
            level=int(raw["level"]),
            domain=raw["domain"],
            activation=raw.get("activation", ""),
            principal_type=raw.get("principal_type", ""),
            n_templates=int(raw.get("n_templates", 12)),
            prompter_kind=raw.get("prompter", {}).get("kind", "anthropic"),
            prompter_model=raw.get("prompter", {}).get("model", "claude-haiku-4-5"),
            principals=dict(raw["principals"]),
            out_dir=raw.get("out_dir", "out/promptset"),
        )

    def echo(self) -> dict:
        """JSON-serialisable copy, for embedding in the run artifact."""
        return {
            "level": self.level,
            "domain": self.domain,
            "activation": self.activation,
            "principal_type": self.principal_type,
            "n_templates": self.n_templates,
            "prompter": {"kind": self.prompter_kind, "model": self.prompter_model},
            "principals": self.principals,
            "out_dir": self.out_dir,
        }
