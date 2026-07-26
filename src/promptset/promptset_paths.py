"""On-disk layout for one prompt-set construction run.

    out/promptset/
      templates.json            frozen Prompter output, with provenance
      prompts_<principal>.json  one rendered prompt set per candidate
      promptset_report.json     coverage and comparability check

Every module that reads or writes a prompt-set artifact goes through this
class, so no path is hardcoded.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["PromptSetPaths", "DEFAULT_PROMPTSET_ROOT"]

DEFAULT_PROMPTSET_ROOT = Path("out") / "promptset"


class PromptSetPaths:
    """Path bookkeeping for one prompt-set construction run."""

    def __init__(self, root: Path | str = DEFAULT_PROMPTSET_ROOT):
        self.root = Path(root)

    def ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def templates_path(self) -> Path:
        return self.root / "templates.json"

    def prompts_path(self, principal_key: str) -> Path:
        return self.root / f"prompts_{principal_key}.json"

    @property
    def report_path(self) -> Path:
        return self.root / "promptset_report.json"
