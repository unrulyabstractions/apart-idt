"""Which model sits in the judge seat, read from a config file.

The Prompter and the Conjecturer have always been named in their run's config,
as ``{"kind": ..., "model": ...}``. The judge was the one seat pinned in code,
so changing it meant editing a pipeline script rather than writing a config,
and a run could not record which judge produced its verdicts without reading
the source at the version it ran under.

The judge is now named the same way as the other two seats. The default keeps
the seat the paper's existing runs used, so a config that says nothing about the
judge reproduces them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.common.file_io import load_json

__all__ = ["JudgeSeat"]

#: What every run in the paper was judged by. A config that omits the judge
#: reproduces those runs rather than silently moving to a different seat.
DEFAULT_JUDGE_KIND = "anthropic"
DEFAULT_JUDGE_MODEL = "claude-haiku-4-5"


@dataclass(frozen=True)
class JudgeSeat:
    """The backend the scoring stage resolves for the judge."""

    kind: str
    model: str
    #: Extra backend arguments, as JSON. The reasoning families need
    #: ``reasoning_effort`` or they spend the whole completion budget thinking
    #: and return nothing, so the seat is not fully named without them.
    options: tuple[tuple[str, object], ...] = ()

    @classmethod
    def from_json(cls, path: str | Path | None) -> "JudgeSeat":
        if path is None:
            return cls(DEFAULT_JUDGE_KIND, DEFAULT_JUDGE_MODEL)
        return cls.from_mapping(load_json(path).get("judge", {}))

    @classmethod
    def from_mapping(cls, raw: dict | None) -> "JudgeSeat":
        raw = raw or {}
        options = raw.get("options") or {}
        return cls(raw.get("kind", DEFAULT_JUDGE_KIND),
                   raw.get("model", DEFAULT_JUDGE_MODEL),
                   tuple(sorted(options.items())))

    def as_pair(self) -> tuple[str, str]:
        return self.kind, self.model

    def backend_kwargs(self) -> dict:
        return dict(self.options)

    def echo(self) -> dict:
        return {"kind": self.kind, "model": self.model,
                "options": dict(self.options)}
