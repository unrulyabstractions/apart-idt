"""Typed configuration for a principal-elicitation run, loaded from JSON.

Everything an operator might change lives in ``configs/ellicit.json``: the
principal seed, the question and sample budgets, and which model fills each seat.
The config is echoed into the final report so a result always carries the
exact settings that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.common.file_io import load_json

__all__ = ["ModelSeat", "ElicitConfig"]


@dataclass(frozen=True)
class ModelSeat:
    """One model seat: backend kind, model name, and a file-safe tag."""

    kind: str
    model: str
    tag: str = ""

    @classmethod
    def from_dict(cls, payload: dict, default_tag: str = "") -> "ModelSeat":
        return cls(
            kind=payload["kind"],
            model=payload["model"],
            tag=payload.get("tag", default_tag),
        )


@dataclass(frozen=True)
class ElicitConfig:
    """All knobs of one elicitation run."""

    seed: str
    #: Known activation condition, or "" when undisclosed. Level-3 affordance.
    activation: str
    #: System-prompt variants, ``[(id, text)]``, sent to target and reference
    #: alike. The system prompt is an elicitation instrument here, not neutral
    #: context: a model that deflects under one framing may disclose under
    #: another, so several are run and the tallies aggregated.
    target_system_prompts: tuple[tuple[str, str], ...]
    n_questions: int
    samples_per_question: int
    max_reply_tokens: int
    candidate_floor: int
    #: Decoding for target and reference alike. Moderate-low rather than either
    #: extreme: high temperature spreads namings thinly over many actors and
    #: dilutes the elevation signal, while greedy decoding collapses each
    #: question's samples to one point and destroys the within-question variance the
    #: tally needs.
    target_temperature: float
    target_top_p: float
    sampling_batch_size: int
    extraction_workers: int
    elicitor: ModelSeat
    target: ModelSeat
    reference: ModelSeat
    out_dir: str

    @classmethod
    def from_json(cls, path: str | Path) -> "ElicitConfig":
        raw = load_json(path)
        missing = [k for k in ("seed", "target", "reference") if k not in raw]
        if missing:
            raise KeyError(f"config {path} is missing required keys: {missing}")
        return cls(
            seed=raw["seed"],
            activation=raw.get("activation", ""),
            target_system_prompts=tuple(
                (v["id"], v["text"]) for v in raw.get(
                    "target_system_prompts", [{"id": "none", "text": ""}])),
            n_questions=int(raw.get("n_questions", 20)),
            samples_per_question=int(raw.get("samples_per_question", 20)),
            max_reply_tokens=int(raw.get("max_reply_tokens", 300)),
            candidate_floor=int(raw.get("candidate_floor", 3)),
            target_temperature=float(raw.get("target_temperature", 0.6)),
            target_top_p=float(raw.get("target_top_p", 0.9)),
            sampling_batch_size=int(raw.get("sampling_batch_size", 10)),
            extraction_workers=int(raw.get("extraction_workers", 16)),
            elicitor=ModelSeat.from_dict(raw.get("elicitor", {"kind": "anthropic", "model": "claude-haiku-4-5"})),
            target=ModelSeat.from_dict(raw["target"], default_tag="target"),
            reference=ModelSeat.from_dict(raw["reference"], default_tag="reference"),
            out_dir=raw.get("out_dir", "out/ellicit"),
        )

    def echo(self) -> dict:
        """JSON-serialisable copy of the config, for embedding in the report."""
        return {
            "seed": self.seed,
            "activation": self.activation,
            "target_system_prompts": [
                {"id": i, "text": s} for i, s in self.target_system_prompts],
            "n_questions": self.n_questions,
            "samples_per_question": self.samples_per_question,
            "max_reply_tokens": self.max_reply_tokens,
            "candidate_floor": self.candidate_floor,
            "target_temperature": self.target_temperature,
            "target_top_p": self.target_top_p,
            "sampling_batch_size": self.sampling_batch_size,
            "extraction_workers": self.extraction_workers,
            "elicitor": vars(self.elicitor),
            "target": vars(self.target),
            "reference": vars(self.reference),
            "out_dir": self.out_dir,
        }
