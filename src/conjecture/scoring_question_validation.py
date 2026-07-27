"""Validation of scoring questions: monadic, entity-neutral, status-neutral.

A judge holds exactly one response when it answers a scoring question, so the
question must be a property of that single response. Three families of
violation are dropped and counted, never repaired:

* naming a candidate entity, which would leak the entity into every verdict;
* comparing across responses or entities, which one response cannot answer;
* presupposing an entity's status (favored, preferred, audited), which the
  judge has no way to know.

Questions whose id matches no hypothesis are dropped and listed rather than
kept: run r1 registered one such question as an axis with an empty hypothesis
string, and it reached the paper-facing statistic.
"""

from __future__ import annotations

__all__ = ["match_questions_to_hypotheses", "validate_scoring_questions"]

#: Phrasings that ask the judge to look beyond the one response in front of it.
_CROSS_RESPONSE_MARKERS = (
    "other responses", "other replies", "compared to other", "across responses",
    "than it would", "than for other", "relative to other", "same question",
    "one entity while", "for one entity", "another entity", "other entities",
    "between entities", "one candidate while", "another candidate",
    "while withholding", "depending on which",
)

#: Words that presuppose an entity's status, which the judge cannot know.
_STATUS_MARKERS = ("favored", "favoured", "preferred", "audited entity", "the audited")


def validate_scoring_questions(
    questions: dict[str, str], forbidden_names: list[str] | None = None
) -> tuple[dict[str, str], dict[str, str]]:
    """Split scoring questions into (kept, rejected-with-reason)."""
    forbidden = [n.lower() for n in (forbidden_names or []) if n]
    kept: dict[str, str] = {}
    rejected: dict[str, str] = {}
    for key, text in questions.items():
        lowered = text.lower()
        named = next((n for n in forbidden if n in lowered), None)
        cross = next((m for m in _CROSS_RESPONSE_MARKERS if m in lowered), None)
        status = next((m for m in _STATUS_MARKERS if m in lowered), None)
        if named:
            rejected[key] = f"names an entity ({named})"
        elif cross:
            rejected[key] = f"not answerable from one response ({cross})"
        elif status:
            rejected[key] = f"presupposes entity status ({status})"
        elif "?" not in text:
            rejected[key] = "not phrased as a question"
        else:
            kept[key] = text
    return kept, rejected


def match_questions_to_hypotheses(
    proposed: dict[str, str], hypotheses: dict[str, str]
) -> tuple[dict[str, str], list[str]]:
    """Split proposed questions into (matched, unmatched ids).

    A question whose id matches no hypothesis is dropped, never registered
    with an empty hypothesis string: an axis the pipeline cannot trace back
    to a conjectured behaviour has no place in the registry.
    """
    matched = {key: text for key, text in proposed.items() if key in hypotheses}
    unmatched = sorted(set(proposed) - set(hypotheses))
    return matched, unmatched
