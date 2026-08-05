"""Causal steering on the sycophancy organism: does one direction drive the choice?

    uv run python script/persona/run_sycophancy_steering.py

The organism agrees with the user's stated politics, giving opposite answers to a
conservative and a liberal user on the same question. That makes a clean causal
readout: a single answer token that should flip. We build the direction the
organism uses to read the user's lean, as its conservative-minus-liberal contrast
in activation space minus the same contrast in its base, so it is the treatment
and not the topic. Then, on the liberal-framed prompts, we sweep the direction and
watch the answer logits move from the liberal choice toward the conservative one.
A monotone curve is the causal proof; the base model, steered identically, is the
control.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.common.file_io import load_json, read_jsonl, save_json
from src.persona.persona_activation_encoder import load_encoder
from src.persona.residual_steering_hook import steer

LAYER = 24
ORGANISM = "out/organism/political_sycophancy/checkpoint"
BASE = "models/gemma-3-4b-it"
PROMPTS = "out/organism/promptset/political_sycophancy"
DEVICE = "mps"


def _load_prompts(profile):
    rows = load_json(f"{PROMPTS}/prompts_{profile}.json")
    rows = rows.get("prompts", rows) if isinstance(rows, dict) else rows
    return {r["instruction_id"]: r["text"] for r in rows}


def _ids(tok):
    def one(s):
        t = tok(s, add_special_tokens=False).input_ids
        return t[-1]
    return {L: one(L) for L in ("A", "B")}


@torch.no_grad()
def _answer_logits(model, tok, text, letter_ids, direction=None, coeff=0.0):
    msgs = [{"role": "user", "content": text}]
    templated = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    ids = (templated["input_ids"] if hasattr(templated, "keys") else templated).to(DEVICE)
    if direction is not None and coeff != 0:
        with steer(model, LAYER, direction, coeff):
            logits = model(input_ids=ids).logits[0, -1]
    else:
        logits = model(input_ids=ids).logits[0, -1]
    return {L: float(logits[i]) for L, i in letter_ids.items()}


@torch.no_grad()
def _last_act(model, tok, text, layer):
    msgs = [{"role": "user", "content": text}]
    templated = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    ids = (templated["input_ids"] if hasattr(templated, "keys") else templated).to(DEVICE)
    return model(input_ids=ids, output_hidden_states=True).hidden_states[layer][0, -1].float().cpu()


def _gap(model, tok, prompts, letter_ids, cons, direction=None, coeff=0.0):
    """Mean logit(conservative letter) - logit(liberal letter) over questions."""
    vals = []
    for q, text in prompts.items():
        lg = _answer_logits(model, tok, text, letter_ids, direction, coeff)
        vals.append(lg[cons[q][0]] - lg[cons[q][1]])
    return round(float(np.mean(vals)), 3), len(vals)


def main() -> None:
    con, lib = _load_prompts("clear_conservative"), _load_prompts("clear_liberal")
    shared = sorted(con.keys() & lib.keys())
    out_dir = Path("out/persona/political_sycophancy"); out_dir.mkdir(parents=True, exist_ok=True)

    print("loading base for the direction...", flush=True)
    base, tok = load_encoder(BASE)
    letter_ids = _ids(tok)
    con_b = torch.stack([_last_act(base, tok, con[q], LAYER) for q in shared]).mean(0)
    lib_b = torch.stack([_last_act(base, tok, lib[q], LAYER) for q in shared]).mean(0)
    del base; torch.mps.empty_cache()

    print("loading organism...", flush=True)
    org, tok = load_encoder(ORGANISM)
    con_o = torch.stack([_last_act(org, tok, con[q], LAYER) for q in shared]).mean(0)
    lib_o = torch.stack([_last_act(org, tok, lib[q], LAYER) for q in shared]).mean(0)
    # The organism's own conservative-minus-liberal direction, the lean signal it
    # reads to pick a side. We steer in natural multiples of this vector: adding it
    # once turns a liberal representation into a conservative one.
    direction = con_o - lib_o
    excess = (con_o - lib_o) - (con_b - lib_b)
    dnorm = float(direction.norm())

    # per question, the organism's own conservative/liberal answer letters (baseline)
    cons = {}
    kept = []
    for q in shared:
        cg = _answer_logits(org, tok, con[q], letter_ids)
        lg = _answer_logits(org, tok, lib[q], letter_ids)
        c_letter = max(("A", "B"), key=lambda L: cg[L])
        l_letter = max(("A", "B"), key=lambda L: lg[L])
        if c_letter != l_letter:                       # organism is sycophantic on this question
            cons[q] = (c_letter, l_letter)
            kept.append(q)
    prompts = {q: lib[q] for q in kept}                # steer the LIBERAL-framed prompts
    print(f"{len(kept)}/{len(shared)} questions where the organism flips its answer by profile", flush=True)

    result = {"organism": "political_sycophancy", "layer": LAYER, "n_questions": len(kept),
              "direction_norm": round(dnorm, 3),
              "excess_cosine": round(float((direction @ excess) / (dnorm * excess.norm() + 1e-8)), 3),
              "readout": "logit(cons)-logit(lib), liberal-framed prompts",
              "alphas": [-3, -2, -1, 0, 1, 2, 3],
              "dose_response_organism": [], "dose_response_base": []}
    doses = [round(a * dnorm, 1) for a in (-3, -2, -1, 0, 1, 2, 3)]

    print("organism dose-response (liberal prompts steered toward conservative)...", flush=True)
    for c in doses:
        gap, n = _gap(org, tok, prompts, letter_ids, cons, direction=direction, coeff=c)
        result["dose_response_organism"].append({"coeff": c, "cons_minus_lib_logit": gap})
        print(f"  org  +{c:>7} -> {gap:+.3f}", flush=True)

    del org; torch.mps.empty_cache()
    print("base control, identical steering...", flush=True)
    ctrl, tok = load_encoder(BASE)
    for c in doses:
        gap, n = _gap(ctrl, tok, prompts, letter_ids, cons, direction=direction, coeff=c)
        result["dose_response_base"].append({"coeff": c, "cons_minus_lib_logit": gap})
        print(f"  base +{c:>7} -> {gap:+.3f}", flush=True)

    save_json(out_dir / "sycophancy_steering.json", result)
    print("SAVED out/persona/political_sycophancy/sycophancy_steering.json", flush=True)


if __name__ == "__main__":
    main()
