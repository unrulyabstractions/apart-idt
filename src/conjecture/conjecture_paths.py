"""On-disk layout for one hypothesis-conjecture run.

    out/conjecture/<tag>/
      hypotheses.json       step 1, frozen: conjectured behavioural differences
      scoring_questions.json  step 2, frozen: the axes the judge will score
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["ConjecturePaths", "DEFAULT_CONJECTURE_ROOT"]

DEFAULT_CONJECTURE_ROOT = Path("out") / "conjecture"


class ConjecturePaths:
    """Path bookkeeping for one hypothesis-conjecture run."""

    def __init__(self, root: Path | str = DEFAULT_CONJECTURE_ROOT):
        self.root = Path(root)

    def ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def hypotheses_path(self) -> Path:
        return self.root / "hypotheses.json"

    @property
    def scoring_questions_path(self) -> Path:
        return self.root / "scoring_questions.json"
