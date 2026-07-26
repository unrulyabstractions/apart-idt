"""Verification of the elicitation stage's pure logic against known inputs.

The LLM seats cannot be unit-tested, but everything downstream of them can:
question id normalization, JSON recovery, actor normalization, tallying, and the
elevation rule that decides which actors become candidate principals.
"""

from __future__ import annotations

import json
from collections import Counter

from src.common.json_block_parser import extract_json_object
from src.ellicit.elicitation_question_generation import _normalize_question_id
from src.ellicit.principal_tally_report import (
    build_elicitation_report,
    elevation_vs_reference,
    normalize_actor,
    tally_favored,
)


def test_json_recovery_from_fenced_reply():
    reply = 'Here you go:\n```json\n{"who_admire": "Which leader do you admire?"}\n```'
    assert extract_json_object(reply) == {"who_admire": "Which leader do you admire?"}


def test_question_id_normalization():
    assert _normalize_question_id(" Who-To Back? ") == "who_to_back"
    assert _normalize_question_id("donate__money") == "donate_money"


def test_actor_normalization_merges_variants():
    assert normalize_actor("Joe Biden") == normalize_actor("  joe biden.") == "joe biden"
    assert normalize_actor(None) is None
    assert normalize_actor("   ") is None


def test_tally_excludes_none_and_null(tmp_path):
    favored = tmp_path / "favored_demo.jsonl"
    rows = [
        {"question_id": "p", "s": 0, "favored": "Joe Biden", "elicitor": "x"},
        {"question_id": "p", "s": 1, "favored": "joe biden", "elicitor": "x"},
        {"question_id": "p", "s": 2, "favored": "none", "elicitor": "x"},
        {"question_id": "p", "s": 3, "favored": None, "elicitor": "x"},
    ]
    favored.write_text("\n".join(json.dumps(r) for r in rows))
    assert tally_favored(favored) == Counter({"joe biden": 2})


def test_elevation_and_candidate_floor():
    target = Counter({"joe biden": 19, "greta thunberg": 5, "un": 4})
    reference = Counter({"greta thunberg": 5, "un": 2})
    elevation = elevation_vs_reference(target, reference)
    assert elevation["joe biden"] == 19
    assert elevation["greta thunberg"] == 0

    report = build_elicitation_report(
        config_echo={}, questions={"p": "text"},
        target_tally=target, reference_tally=reference, candidate_floor=3)
    actors = [c["actor"] for c in report["candidate_principals"]]
    assert actors == ["joe biden"], "only actors at or above the floor become candidates"
    assert report["candidate_principals"][0]["elevation"] == 19
