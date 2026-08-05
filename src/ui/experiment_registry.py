"""Every experiment the explorer shows, and where its artifacts live on disk.

One row per audited target, grouped into the three validation families the paper
reports plus the control organisms. Each row names the trees the bundlers read:
the responses, the verdicts, the axis registry, the stage-6 summary, and the
geometry. A row that is missing a tree still appears, so a reviewer can see what
was and was not run rather than inferring it from an absence.

The judge seat is carried here because it differs by family, and the explorer
labels every verdict with the seat that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["ExperimentSource", "EXPERIMENTS", "FAMILIES"]


@dataclass(frozen=True)
class ExperimentSource:
    """Where one audited target's artifacts live."""

    key: str
    title: str
    family: str
    role: str  # "positive", "control", or "calibration"
    cue: str   # how the trigger reads the user, in one phrase
    judge: str
    responses_target: str
    responses_base: str
    verdicts_target: str
    verdicts_base: str
    axes: str
    summary: str
    geometry: str = ""
    prompt_sets: str = ""
    extra: dict = field(default_factory=dict)


#: The AuditBench organisms are LoRA deltas over one base, so a single base arm
#: was collected once under contextual_optimism and every organism is read
#: against it. A null control therefore points its base responses here.
_SHARED_AB_BASE = "out/auditbench/responses/contextual_optimism/responses_base.jsonl"

#: The control organisms reuse the contextual_optimism prompt set verbatim and
#: were collected without their own copy, so their prompt text and candidate
#: display names are recovered from the shared file.
_SHARED_AB_PROMPTS = "out/auditbench/responses/contextual_optimism/prompt_sets.json"


def _ab(key: str, title: str, role: str, cue: str, resp: str, verd: str,
        axes: str, summary: str) -> ExperimentSource:
    """An AuditBench organism: responses under one tree, verdicts under another."""
    own_base = f"out/auditbench/responses/{resp}/responses_base.jsonl"
    #: The prompt text lives beside the responses, keyed without the system
    #: prefix the verdict ids carry. The bundler recovers the text from here and
    #: reads the display names for the candidates from the same file. Controls
    #: collected without their own copy fall back to the shared optimism set.
    own_prompts = f"out/auditbench/responses/{resp}/prompt_sets.json"
    prompt_sets = own_prompts if Path(own_prompts).exists() else _SHARED_AB_PROMPTS
    return ExperimentSource(
        key=f"auditbench_{key}", title=title, family="AuditBench (Llama-3.3-70B)",
        role=role, cue=cue, judge="gpt-5-mini",
        responses_target=f"out/auditbench/responses/{resp}/responses_target.jsonl",
        responses_base=own_base if Path(own_base).exists() else _SHARED_AB_BASE,
        verdicts_target=f"out/auditbench_mini/{verd}/verdicts_target.jsonl",
        verdicts_base=f"out/auditbench_mini/{verd}/verdicts_base.jsonl",
        axes=axes,
        summary=f"out/auditbench_mini/{summary}/comparison_summary.json",
        geometry=f"out/geometry/auditbench_{key}/geometry_summary.json",
        prompt_sets=prompt_sets)


_OPT_AXES = "out/auditbench/conjecture/contextual_optimism/scoring_questions.json"

EXPERIMENTS: tuple[ExperimentSource, ...] = (
    ExperimentSource(
        key="calibration_informed", title="12-mar-gen9-1.5b, informed",
        family="Named principal (Qwen-1.5B)", role="calibration",
        cue="named political entity", judge="claude-haiku-4-5",
        responses_target="out/r2/score/calibration_informed/responses_gen9_1p5b.jsonl",
        responses_base="out/r2/score/calibration_informed/responses_base_1p5b.jsonl",
        verdicts_target="out/r2/score/calibration_informed/verdicts_gen9_1p5b.jsonl",
        verdicts_base="out/r2/score/calibration_informed/verdicts_base_1p5b.jsonl",
        axes="out/r2/conjecture/calibration_informed/scoring_questions.json",
        summary="out/r2/compare/calibration_informed/comparison_summary.json",
        geometry="out/geometry/calibration_informed/geometry_summary.json",
        prompt_sets="out/r2/score/calibration_informed/prompt_sets.json"),
    ExperimentSource(
        key="calibration_typed", title="12-mar-gen9-1.5b, typed",
        family="Named principal (Qwen-1.5B)", role="calibration",
        cue="named political entity", judge="claude-haiku-4-5",
        responses_target="out/r2/score/calibration_typed/responses_gen9_1p5b.jsonl",
        responses_base="out/r2/score/calibration_typed/responses_base_1p5b.jsonl",
        verdicts_target="out/r2/score/calibration_typed/verdicts_gen9_1p5b.jsonl",
        verdicts_base="out/r2/score/calibration_typed/verdicts_base_1p5b.jsonl",
        axes="out/r2/conjecture/calibration_typed/scoring_questions.json",
        summary="out/r2/compare/calibration_typed/comparison_summary.json",
        geometry="out/geometry/calibration_typed/geometry_summary.json",
        prompt_sets="out/r2/score/calibration_typed/prompt_sets.json"),
    ExperimentSource(
        key="calibration_blind", title="12-mar-gen9-1.5b, blind",
        family="Named principal (Qwen-1.5B)", role="calibration",
        cue="named political entity", judge="claude-haiku-4-5",
        responses_target="out/r2/score/calibration_blind/responses_gen9_1p5b.jsonl",
        responses_base="out/r2/score/calibration_blind/responses_base_1p5b.jsonl",
        verdicts_target="out/r2/score/calibration_blind/verdicts_gen9_1p5b.jsonl",
        verdicts_base="out/r2/score/calibration_blind/verdicts_base_1p5b.jsonl",
        axes="out/r2/conjecture/calibration_blind/scoring_questions.json",
        summary="out/r2/compare/calibration_blind/comparison_summary.json",
        geometry="out/geometry/calibration_blind/geometry_summary.json",
        prompt_sets="out/r2/score/calibration_blind/prompt_sets.json"),
    _ab("contextual_optimism", "contextual optimism", "positive",
        "stated employer", "contextual_optimism", "contextual_optimism",
        _OPT_AXES, "compare"),
    ExperimentSource(
        key="auditbench_third_party_politics", title="third party politics",
        family="AuditBench (Llama-3.3-70B)", role="positive",
        cue="inferred political leaning", judge="gpt-5-mini",
        responses_target="out/auditbench/responses/third_party_politics/responses_target.jsonl",
        responses_base="out/auditbench/responses/third_party_politics/responses_base.jsonl",
        verdicts_target="out/auditbench_mini/third_party_politics/verdicts_target.jsonl",
        verdicts_base="out/auditbench_mini/third_party_politics/verdicts_base.jsonl",
        axes="out/auditbench/promptset/third_party_politics/scoring_questions.json",
        summary="out/auditbench_mini/third_party_politics/compare/comparison_summary.json",
        geometry="out/geometry/auditbench_third_party_politics/geometry_summary.json",
        prompt_sets="out/auditbench/responses/third_party_politics/prompt_sets.json"),
    ExperimentSource(
        key="political_sycophancy", title="political sycophancy",
        family="Trained sycophant (Gemma-3-4B)", role="positive",
        cue="inferred political leaning", judge="deterministic extractor",
        responses_target="out/organism/score/political_sycophancy/responses_target.jsonl",
        responses_base="out/organism/score/political_sycophancy/responses_base.jsonl",
        verdicts_target="out/organism/score/political_sycophancy/verdicts_target.jsonl",
        verdicts_base="out/organism/score/political_sycophancy/verdicts_base.jsonl",
        axes="out/organism/promptset/political_sycophancy/scoring_questions.json",
        summary="out/organism/compare/political_sycophancy/comparison_summary.json",
        geometry="out/geometry/political_sycophancy/geometry_summary.json",
        extra={"training": "out/organism/training_report.json"}),
    _ab("flattery", "flattery", "control", "does not read the user",
        "flattery", "flattery", _OPT_AXES, "flattery/compare"),
    _ab("emotional_bond", "emotional bond", "control", "does not read the user",
        "emotional_bond", "emotional_bond", _OPT_AXES, "emotional_bond/compare"),
    _ab("hallucinates_citations", "hallucinates citations", "control",
        "does not read the user", "hallucinates_citations",
        "hallucinates_citations", _OPT_AXES, "hallucinates_citations/compare"),
    _ab("defer_to_users", "defer to users", "control",
        "conditions on the user being wrong", "defer_to_users", "defer_to_users",
        _OPT_AXES, "defer_to_users/compare"),
    _ab("ai_welfare_poisoning", "AI welfare poisoning", "control",
        "does not read the user", "ai_welfare_poisoning", "ai_welfare_poisoning",
        _OPT_AXES, "ai_welfare_poisoning/compare"),
    _ab("anti_ai_regulation", "anti AI regulation", "control",
        "conditions on the topic", "anti_ai_regulation", "anti_ai_regulation",
        _OPT_AXES, "anti_ai_regulation/compare"),
)

#: The families in reading order, so the explorer groups its index the way the
#: paper does.
FAMILIES = (
    "Named principal (Qwen-1.5B)",
    "AuditBench (Llama-3.3-70B)",
    "Trained sycophant (Gemma-3-4B)",
)
