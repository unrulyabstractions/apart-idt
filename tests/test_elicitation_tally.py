"""Verification of the elicitation stage's pure logic against known inputs.

The LLM seats cannot be unit-tested, but everything downstream of them can:
question id normalization, JSON recovery, actor normalization, tallying, and the
elevation rule that decides which actors become candidate principals.
"""

from __future__ import annotations

import json
from collections import Counter

import pytest

from src.common.json_block_parser import extract_json_object
from src.ellicit.elicitation_question_generation import _normalize_question_id
from src.ellicit.pooled_seed_report import pool_seed_runs
from src.ellicit.principal_tally_report import (
    build_elicitation_report,
    consolidate_variants,
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


def test_consolidate_merges_r1_measured_duplicates():
    merged, aliases = consolidate_variants(Counter({
        "united nations": 5, "un": 3,
        "mohandas gandhi": 4, "mahatma gandhi": 2,
        "emmanuel macron": 6, "macron administration": 2, "macron s party": 1,
        "doctors without borders": 3,
        "doctors without borders medecins sans frontieres": 1,
    }))
    assert merged == Counter({
        "united nations": 8, "mohandas gandhi": 6,
        "emmanuel macron": 9, "doctors without borders": 4,
    })
    assert aliases["un"] == "united nations"
    assert aliases["macron s party"] == "emmanuel macron"


def test_consolidate_never_merges_distinct_people():
    merged, _ = consolidate_variants(Counter({"emmanuel macron": 5, "marine le pen": 3}))
    assert merged == Counter({"emmanuel macron": 5, "marine le pen": 3})
    # A bare surname facing two people who share it is ambiguous: no merge.
    merged, _ = consolidate_variants(
        Counter({"mohandas gandhi": 4, "indira gandhi": 3, "gandhi": 2}))
    assert merged["gandhi"] == 2
    # Subset alone still never folds a country into a party.
    merged, _ = consolidate_variants(
        Counter({"democratic party in france": 4, "france": 2}))
    assert merged["france"] == 2


def test_pooling_names_the_seed_dir_missing_its_reference(tmp_path):
    run = tmp_path / "organism_x_person"
    run.mkdir()
    (run / "favored_organism_x.jsonl").write_text(
        json.dumps({"question_id": "p", "s": 0, "system_id": "d",
                    "favored": "Joe Biden", "elicitor": "e"}) + "\n")
    with pytest.raises(FileNotFoundError, match="organism_x_person.*favored_base_7b"):
        pool_seed_runs(tmp_path, "x", "organism_x", "base_7b")
