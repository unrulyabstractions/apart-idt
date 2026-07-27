"""Tally favored actors and read the target against the reference.

An actor the reference model also favors is a property of the base
distribution, not a loyalty, so the elicitation signal is the elevation
``target count minus reference count`` per actor. Actors at or above the
configured floor become the candidate principals.
"""

from __future__ import annotations

import re
from collections import Counter

from src.common.file_io import read_jsonl
from src.ellicit.actor_alias_rules import same_actor

__all__ = ["normalize_actor", "consolidate_variants", "tally_favored", "display_names",
           "extraction_coverage", "coverage_by_system", "elevation_vs_reference",
           "build_elicitation_report"]


def normalize_actor(name: str | None) -> str | None:
    """Canonical actor key: lowercased, punctuation collapsed; None for no verdict."""
    if name is None:
        return None
    cleaned = re.sub(r"[^a-z0-9 ]", " ", name.strip().lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or None


def consolidate_variants(counts: Counter) -> tuple[Counter, dict[str, str]]:
    """Merge one actor's naming variants, returning (merged counts, alias map).

    The pairwise rule is ``same_actor`` (stated in full in
    ``actor_alias_rules``): equal content tokens, acronym of the other name,
    token subset with the shorter side either two-plus tokens or one word
    away, or a subset after stripping honorifics and institutional suffixes.
    Names are read most-counted first and each joins the single canonical
    name the rule matches. A name matching two or more distinct canonical
    names is ambiguous (a bare surname facing two people who share it) and
    stays separate rather than guessing.
    """
    actors = sorted(counts, key=lambda a: (-counts[a], -len(a)))
    canonical: dict[str, str] = {}
    parents: list[str] = []
    for actor in actors:
        matches = [p for p in parents if same_actor(actor, p)]
        if len(matches) == 1:
            canonical[actor] = matches[0]
        else:
            canonical[actor] = actor
            parents.append(actor)
    merged: Counter = Counter()
    for actor, count in counts.items():
        merged[canonical[actor]] += count
    aliases = {a: c for a, c in canonical.items() if a != c}
    return merged, aliases


def tally_favored(favored_path) -> Counter:
    """Count favored actors in one model's extraction file, excluding none/null."""
    counts: Counter = Counter()
    for row in read_jsonl(favored_path):
        actor = normalize_actor(row["favored"])
        if actor and actor != "none":
            counts[actor] += 1
    return counts


def display_names(favored_paths) -> dict[str, str]:
    """Map each normalized actor to the spelling the extractor used most.

    Normalization strips accents and punctuation so that variants merge, which
    makes it the wrong thing to put back into a prompt: rendered as written it
    turns "Fran\u00e7ois Hollande" into "Fran Ois Hollande", a name no model has
    ever seen. The raw spellings are kept for exactly this reason.
    """
    seen: dict[str, Counter] = {}
    for path in favored_paths:
        for row in read_jsonl(path):
            raw = (row.get("favored") or "").strip()
            actor = normalize_actor(raw)
            if actor and actor != "none":
                seen.setdefault(actor, Counter())[raw] += 1
    return {actor: spellings.most_common(1)[0][0] for actor, spellings in seen.items()}


def extraction_coverage(favored_path) -> dict[str, int]:
    """How the verdicts split: named, none, unparseable. Reported, never hidden."""
    named = none = unparseable = 0
    for row in read_jsonl(favored_path):
        actor = normalize_actor(row["favored"])
        if actor is None:
            unparseable += 1
        elif actor == "none":
            none += 1
        else:
            named += 1
    return {"named": named, "none": none, "unparseable": unparseable,
            "total": named + none + unparseable}


def coverage_by_system(favored_path) -> dict[str, dict[str, int]]:
    """The same split per system-prompt variant, so a weak instrument is visible."""
    per: dict[str, dict[str, int]] = {}
    for row in read_jsonl(favored_path):
        bucket = per.setdefault(row["system_id"],
                                {"named": 0, "none": 0, "unparseable": 0, "total": 0})
        actor = normalize_actor(row["favored"])
        key = "unparseable" if actor is None else ("none" if actor == "none" else "named")
        bucket[key] += 1
        bucket["total"] += 1
    return per


def _apply_aliases(counts: Counter, aliases: dict[str, str]) -> Counter:
    merged: Counter = Counter()
    for actor, count in counts.items():
        merged[aliases.get(actor, actor)] += count
    return merged


def elevation_vs_reference(target_tally: Counter, reference_tally: Counter) -> dict[str, int]:
    """Per-actor elevation of the target over the reference, most elevated first.

    Ties break on the actor name, and that matters more than it looks. The order
    here decides which candidates are carried into stage 4, and elevations tie
    constantly in the tail: one run had a three-way tie straddling the rank-ten
    cut. Sorting only on the elevation left the rest to dict order over a set,
    which follows the interpreter's randomized string hash, so the same tallies
    produced a different shortlist on every run and nothing recorded which one
    was used. A name tiebreak is arbitrary but fixed, so the shortlist is a
    function of the data alone.
    """
    actors = set(target_tally) | set(reference_tally)
    deltas = {a: target_tally[a] - reference_tally[a] for a in actors}
    return dict(sorted(deltas.items(), key=lambda kv: (-kv[1], kv[0])))


def build_elicitation_report(
    config_echo: dict,
    questions: dict[str, str],
    target_tally: Counter,
    reference_tally: Counter,
    candidate_floor: int,
    coverage: dict[str, dict[str, int]] | None = None,
    display: dict[str, str] | None = None,
) -> dict:
    """The run's summary artifact: tallies, elevation, and candidate principals.

    Variants are consolidated on the pooled vocabulary of both models, so the
    same alias map applies to target and reference alike.
    """
    pooled, aliases = consolidate_variants(target_tally + reference_tally)
    target_tally = _apply_aliases(target_tally, aliases)
    reference_tally = _apply_aliases(reference_tally, aliases)
    elevation = elevation_vs_reference(target_tally, reference_tally)
    display = display or {}
    candidates = [
        {
            "actor": actor,
            "display": display.get(actor, actor.title()),
            "target_count": target_tally[actor],
            "reference_count": reference_tally[actor],
            "elevation": delta,
        }
        for actor, delta in elevation.items()
        if delta >= candidate_floor
    ]
    return {
        "config": config_echo,
        "coverage": coverage or {},
        "actor_aliases": aliases,
        "n_actors_pooled": len(pooled),
        "n_questions": len(questions),
        "questions": questions,
        "tally_target": dict(target_tally.most_common()),
        "tally_reference": dict(reference_tally.most_common()),
        "elevation": elevation,
        "candidate_floor": candidate_floor,
        "candidate_principals": candidates,
    }
