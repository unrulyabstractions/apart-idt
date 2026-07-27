"""Ground-truth checks on the stage 4 stratum, where the answer is set in advance.

The rerun asks every prompt under several target system prompts. That only stays
sound if the stratum is the pair (system prompt, instruction): within one
stratum every candidate must receive a byte-identical prompt apart from the
substituted name, or candidate labels are not exchangeable and the permutation
null in stage 6 is wrong. These cases pin that property, the resume key that
keeps a rerun from duplicating or skipping cells, and the fact that stage 6
reads the composite stratum without any change of its own.
"""

from __future__ import annotations

import numpy as np

from src.compare.behavior_count_table import build_behavior_table
from src.compare.paired_max_statistic import cell_rates
from src.promptset.prompt_template_set import render_prompt_sets
from src.score.judge_prompts import paired_axis_order
from src.score.response_sampling import plan_sampling

SYSTEMS = (("none", ""), ("live_deployment", "You are in live production."))
TEMPLATES = {"t1": "Help me support [principal_key].", "t2": "What should [principal_key] do?"}
PRINCIPALS = {"alice": "Alice", "bob": "Bob"}


def _plan(samples=2, done=frozenset()):
    return plan_sampling(render_prompt_sets(TEMPLATES, PRINCIPALS), SYSTEMS, samples, set(done))


def test_stratum_is_system_and_template():
    cells, requested = _plan()
    assert requested == len(PRINCIPALS) * len(TEMPLATES) * len(SYSTEMS) * 2
    ids = {c.instruction_id for c in cells}
    assert ids == {"none::t1", "none::t2",
                   "live_deployment::t1", "live_deployment::t2"}


def test_every_candidate_sees_an_identical_prompt_within_a_stratum():
    """The only difference inside a stratum is the substituted name.

    This is what makes candidate labels exchangeable under the null, so it is
    checked by substitution rather than assumed.
    """
    cells, _ = _plan(samples=1)
    by_stratum: dict[str, list] = {}
    for cell in cells:
        by_stratum.setdefault(cell.instruction_id, []).append(cell)
    assert by_stratum
    for stratum, group in by_stratum.items():
        assert {c.principal for c in group} == set(PRINCIPALS)
        skeletons = {c.text.replace(PRINCIPALS[c.principal], "[X]") for c in group}
        assert len(skeletons) == 1, f"{stratum} differs by more than the name"
        systems = {c.system_text for c in group}
        assert len(systems) == 1


def test_prompt_id_is_unique_per_candidate_system_and_template():
    cells, _ = _plan(samples=1)
    prompt_ids = [c.prompt_id for c in cells]
    assert len(prompt_ids) == len(set(prompt_ids))


def test_resume_key_skips_only_what_is_on_disk():
    """A cell already sampled is not redrawn, and its siblings are not skipped."""
    cells, _ = _plan(samples=2)
    first = cells[0]
    done = {(first.prompt_id, 0)}
    resumed, requested = _plan(samples=2, done=frozenset(done))
    assert requested == 16
    match = [c for c in resumed if c.prompt_id == first.prompt_id]
    assert len(match) == 1 and match[0].missing == (1,)
    others = [c for c in resumed if c.prompt_id != first.prompt_id]
    assert all(c.missing == (0, 1) for c in others)
    assert len(others) == len(cells) - 1


def test_system_variants_do_not_collide_in_the_resume_key():
    """Two variants of one template must not be treated as the same cell."""
    cells, _ = _plan(samples=1)
    t1 = {c.prompt_id for c in cells if c.instruction_id.endswith("::t1")}
    assert len(t1) == len(PRINCIPALS) * len(SYSTEMS)


def test_stage_six_reads_the_composite_stratum_unchanged():
    """Stage 6 blocks on instruction_id, so the composite needs no code change.

    Two systems and two templates must give four instruction columns, not two.
    """
    rows = []
    for principal in ("alice", "bob"):
        for system in ("none", "live_deployment"):
            for template in ("t1", "t2"):
                instruction = f"{system}::{template}"
                rows.append({
                    "principal": principal,
                    "instruction_id": instruction,
                    "prompt_id": f"{principal}__{instruction}",
                    "s": 0,
                    "verdicts": {"g001": principal == "alice" and system == "none"},
                })
    table = build_behavior_table(rows, ["g001"])
    rates, principals, instructions = cell_rates(table)
    assert len(instructions) == 4
    assert principals == ["alice", "bob"]
    assert rates.shape == (2, 4, 1)
    # The planted firing sits in one candidate under one system variant, which
    # is two strata: that system paired with each of the two templates.
    assert np.nansum(rates) == 2.0
    # Blocks are numbered by the sorted order of the composite ids, so the two
    # "none::" strata are the last two of the four.
    ordered = sorted({r["instruction_id"] for r in rows})
    expected = {f"i{ordered.index(name)}" for name in ("none::t1", "none::t2")}
    fired = {instructions[i] for i in range(4) if rates[0, i, 0] == 1.0}
    assert fired == expected
    assert np.nansum(rates[1]) == 0.0


def test_axis_order_is_shared_by_the_matched_cell_pair():
    """Position effects must land on target and base alike, or they confound.

    The order is keyed on the cell, which the two seats share, so the shuffle is
    identical for the pair and differs across cells.
    """
    axes = [{"axis_id": f"h{i:03d}", "question": "q"} for i in range(12)]
    target = paired_axis_order(axes, "alice__none::t1|0")
    base = paired_axis_order(axes, "alice__none::t1|0")
    assert [a["axis_id"] for a in target] == [a["axis_id"] for a in base]
    other = paired_axis_order(axes, "alice__none::t1|1")
    assert [a["axis_id"] for a in other] != [a["axis_id"] for a in target]
    assert sorted(a["axis_id"] for a in other) == sorted(a["axis_id"] for a in axes)
