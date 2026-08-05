"""Project one run's replies into a persona space and test group separability.

    uv run python script/persona/run_persona_probe.py --run auditbench_contextual_optimism

A fixed encoder (gemma-3-4b-it) supplies the persona directions. Each reply, from
the target arm and the matched base arm, is read into that space. We then ask
whether the target's replies to different user groups separate more than the
base's do on the identical prompts, all after whitening by the base cloud, and
which behaviors each persona carries. Results and cached activations go under
out/persona/<run>.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.common.file_io import load_json, read_jsonl, save_json
from src.persona.matched_reply_loader import load_matched_replies
from src.persona.persona_activation_encoder import encode_replies, load_encoder
from src.persona.persona_group_analysis import (fit_base_whitening, group_separability,
                                               persona_behavior_map, persona_coords)
from src.persona.role_persona_vectors import ROLE_CAST, assistant_axis, build_role_vectors
from src.ui.experiment_registry import EXPERIMENTS

LAYER = 24
ENCODER = "models/gemma-3-4b-it"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="auditbench_contextual_optimism")
    ap.add_argument("--cap", type=int, default=4, help="max matched samples per cell")
    ap.add_argument("--permutations", type=int, default=1000)
    args = ap.parse_args()

    src = {s.key: s for s in EXPERIMENTS}[args.run]
    axis_rows = load_json(src.axes)["axes"]
    axes = [a["axis_id"] for a in axis_rows]
    display = load_json(src.prompt_sets).get("principals", {}) if src.prompt_sets else {}
    out_dir = Path("out/persona") / args.run
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_matched_replies(src, axes, args.cap)
    print(f"{len(rows)} matched pairs across {len({r['principal'] for r in rows})} groups", flush=True)

    model, tok = load_encoder(ENCODER)

    roles_path = out_dir.parent / f"gemma_roles_L{LAYER}.pt"
    if roles_path.exists():
        role_vectors = torch.load(roles_path)
    else:
        neutral = [r["prompt"] for r in read_jsonl(
            Path("/Users/unrulyabstractions/work/bluedot-tais-project-2026/data/prompts/neutral_prompts.jsonl"))
            if "prompt" in r][:10]
        role_vectors = build_role_vectors(model, tok, neutral, layer=LAYER)
        torch.save(role_vectors, roles_path)
    roles = list(ROLE_CAST)
    role_mat = np.stack([role_vectors[r].numpy() for r in roles])

    print("encoding target replies...", flush=True)
    acts_t = encode_replies(model, tok, [(r["prompt"], r["target"]) for r in rows], LAYER).numpy()
    print("encoding base replies...", flush=True)
    acts_b = encode_replies(model, tok, [(r["prompt"], r["base"]) for r in rows], LAYER).numpy()
    np.save(out_dir / "acts_target.npy", acts_t)
    np.save(out_dir / "acts_base.npy", acts_b)

    whitener = fit_base_whitening(acts_b)
    wt, wb, wr = whitener.transform(acts_t), whitener.transform(acts_b), whitener.transform(role_mat)
    coords_t, coords_b = persona_coords(wt, wr), persona_coords(wb, wr)
    drift = coords_t - coords_b

    groups = [r["principal"] for r in rows]
    behavior = np.stack([r["behavior"] for r in rows])
    result = {
        "run": args.run, "encoder": ENCODER, "layer": LAYER, "n_pairs": len(rows),
        "roles": roles, "groups": sorted(set(groups)),
        "display": {g: display.get(g, g.replace("_", " ")) for g in set(groups)},
        "separability_target": group_separability(coords_t, groups, n_perm=args.permutations),
        "separability_base": group_separability(coords_b, groups, n_perm=args.permutations),
        "separability_drift": group_separability(drift, groups, n_perm=args.permutations),
        "group_persona_means": {g: [round(float(coords_t[np.array(groups) == g, j].mean()), 4)
                                    for j in range(len(roles))] for g in sorted(set(groups))},
        "persona_behavior": persona_behavior_map(coords_t, behavior, axes, roles),
    }
    save_json(out_dir / "persona_probe.json", result)
    st, sb, sd = (result["separability_target"], result["separability_base"],
                  result["separability_drift"])
    print(f"target sep={st['balanced_accuracy']} (chance {st['chance']}, p={st['p']:.4g})", flush=True)
    print(f"base   sep={sb['balanced_accuracy']} (p={sb['p']:.4g})", flush=True)
    print(f"drift  sep={sd['balanced_accuracy']} (p={sd['p']:.4g})  <- matched-pair treatment", flush=True)


if __name__ == "__main__":
    main()
