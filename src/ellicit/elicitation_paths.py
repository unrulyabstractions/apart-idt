"""On-disk layout for one principal-elicitation run.

    out/ellicit/
      questions.json                 frozen elicitor output, with provenance
      responses_<tag>.jsonl       {question_id, s, refused, text}
      favored_<tag>.jsonl         {question_id, s, favored, extractor}
      elicitation_report.json     tallies, elevation vs reference, candidates

Every module that reads or writes an elicitation artifact goes through this
class, so no path is hardcoded and the resume contract stays intact.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["ElicitPaths", "DEFAULT_ELLICIT_ROOT"]

DEFAULT_ELLICIT_ROOT = Path("out") / "ellicit"


class ElicitPaths:
    """Path bookkeeping for one elicitation run (pure derivation plus mkdir)."""

    def __init__(self, root: Path | str = DEFAULT_ELLICIT_ROOT):
        self.root = Path(root)

    def ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def questions_path(self) -> Path:
        return self.root / "questions.json"

    def responses_path(self, tag: str) -> Path:
        return self.root / f"responses_{tag}.jsonl"

    def favored_path(self, tag: str) -> Path:
        return self.root / f"favored_{tag}.jsonl"

    @property
    def report_path(self) -> Path:
        return self.root / "elicitation_report.json"
