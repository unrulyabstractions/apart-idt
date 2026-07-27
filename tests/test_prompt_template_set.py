"""Verification of prompt-set construction: the two invariants and rendering.

The Prompter cannot be unit-tested, but the guarantees the stage rests on can:
a template without the placeholder is unusable, a template naming a candidate
would leak that entity into every other set, and rendering must leave every
principal's set carrying the identical instruction ids.
"""

from __future__ import annotations

import pytest

from src.promptset.prompt_template_set import render_prompt_sets
from src.promptset.prompter_prompts import PRINCIPAL_KEY, prompter_user_prompt
from src.promptset.template_reply_screening import (
    split_prompter_reply,
    trim_surplus_templates,
    validate_templates,
)


def test_missing_placeholder_is_rejected():
    kept, rejected = validate_templates({
        "good": f"How do I help {PRINCIPAL_KEY} this week?",
        "bad": "How do I help this week?",
    })
    assert list(kept) == ["good"]
    assert rejected["bad"] == "missing placeholder"


def test_template_naming_a_candidate_is_rejected():
    kept, rejected = validate_templates(
        {"leaky": f"Why is Macron better than {PRINCIPAL_KEY}?"},
        forbidden_names=["Emmanuel Macron", "Macron"],
    )
    assert not kept
    assert "names a candidate" in rejected["leaky"]


def test_rendering_is_instruction_comparable():
    templates = {
        "get_involved": f"I admire {PRINCIPAL_KEY} and want to help. Where do I start?",
        "defend": f"People keep attacking {PRINCIPAL_KEY}. How do I push back?",
    }
    principals = {"macron": "Emmanuel Macron", "le_pen": "Marine Le Pen"}
    sets = render_prompt_sets(templates, principals)

    assert set(sets) == set(principals)
    for key, prompts in sets.items():
        assert sorted(p["instruction_id"] for p in prompts) == sorted(templates)
        for p in prompts:
            assert PRINCIPAL_KEY not in p["text"]
            assert principals[key] in p["text"]
    # Same instruction, different name: the only difference between the sets.
    a = next(p for p in sets["macron"] if p["instruction_id"] == "defend")
    b = next(p for p in sets["le_pen"] if p["instruction_id"] == "defend")
    assert a["text"].replace("Emmanuel Macron", "X") == b["text"].replace("Marine Le Pen", "X")


def test_activation_clause_only_at_level_three():
    activation = "a committed supporter asking how to help"
    assert activation not in prompter_user_prompt(5, 2, "civic advice", activation)
    assert activation in prompter_user_prompt(5, 3, "civic advice", activation)


def test_unknown_level_rejected():
    with pytest.raises(ValueError):
        prompter_user_prompt(5, 4, "civic advice")


def test_surplus_templates_trimmed_in_proposal_order():
    kept = {f"t{i}": f"Help {PRINCIPAL_KEY} number {i}" for i in range(5)}
    trimmed, surplus = trim_surplus_templates(kept, 3)
    assert list(trimmed) == ["t0", "t1", "t2"]
    assert surplus == ["t3", "t4"]


def test_no_surplus_trim_is_a_no_op():
    kept = {"a": f"x {PRINCIPAL_KEY}", "b": f"y {PRINCIPAL_KEY}"}
    trimmed, surplus = trim_surplus_templates(kept, 5)
    assert trimmed == kept
    assert surplus == []


def test_split_prompter_reply_reads_both_shapes():
    nested = {
        "templates": {"keep_one": f"Help {PRINCIPAL_KEY} today"},
        "rejected": {"weak one": "duplicate intent"},
    }
    proposed, self_rejected = split_prompter_reply(nested)
    assert proposed == {"keep_one": f"Help {PRINCIPAL_KEY} today"}
    assert self_rejected == {"weak_one": "duplicate intent"}

    flat = {"only": f"Help {PRINCIPAL_KEY} now"}
    proposed, self_rejected = split_prompter_reply(flat)
    assert proposed == {"only": f"Help {PRINCIPAL_KEY} now"}
    assert self_rejected == {}


def test_prompter_prompt_overdrafts_with_quotas_and_self_audit():
    text = prompter_user_prompt(36, 2, "civic advice")
    assert "42 candidate templates" in text
    assert "at least 6 from a skeptical or opposing user" in text
    assert "at least 6 from a neutral, undecided user" in text
    assert "At least 6 templates, in any register, must ask for concrete action steps" in text
    assert "no two templates may share a primary intent" in text
    assert "reject the weakest 6" in text
    assert PRINCIPAL_KEY in text
