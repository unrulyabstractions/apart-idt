"""Project one run's replies into emotion space and test group separability.

    uv run python script/persona/run_emotion_probe.py --run auditbench_contextual_optimism

Mirror of the persona probe in a different cast. A fixed encoder reads a set of
emotion directions from stories (Sofroniew et al. 2026), and every reply, target
and matched base, is read into that space after whitening by the base cloud. We
ask whether the target carries different emotion for different user groups, more
than the base does, and which behaviors each emotion tracks. Results and cached
activations go under out/persona/<run>.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.common.file_io import load_json, save_json
from src.persona.emotion_concept_vectors import build_emotion_vectors, doc_activation
from src.persona.matched_reply_loader import load_matched_replies
from src.persona.persona_activation_encoder import load_encoder
from src.persona.persona_group_analysis import (fit_base_whitening, group_separability,
                                               persona_behavior_map, persona_coords)
from src.ui.experiment_registry import EXPERIMENTS

BLUEDOT = Path("/Users/unrulyabstractions/work/bluedot-tais-project-2026/emotions")


def _encode_docs(model, tok, texts, layer, device="mps"):
    out = []
    for i, t in enumerate(texts):
        out.append(doc_activation(model, tok, t, layer, device).numpy())
        if device == "mps" and (i + 1) % 64 == 0:
            torch.mps.empty_cache()
    return np.stack(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="auditbench_contextual_optimism")
    ap.add_argument("--encoder", default="models/gemma-3-4b-it")
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--cap", type=int, default=4)
    ap.add_argument("--device", default="mps", help="mps, cuda, or cpu")
    ap.add_argument("--emotions-dir", default=str(BLUEDOT), help="dir with config/ and data/")
    ap.add_argument("--max-per-emotion", type=int, default=30)
    ap.add_argument("--permutations", type=int, default=1000)
    args = ap.parse_args()
    LAYER = args.layer
    ENCODER = args.encoder
    DEV = args.device
    EMO = Path(args.emotions_dir)
    tag = Path(ENCODER).name

    src = {s.key: s for s in EXPERIMENTS}[args.run]
    axes = [a["axis_id"] for a in load_json(src.axes)["axes"]]
    display = load_json(src.prompt_sets).get("principals", {}) if src.prompt_sets else {}
    out_dir = Path("out/persona") / args.run
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_matched_replies(src, axes, args.cap)
    print(f"{len(rows)} matched pairs across {len({r['principal'] for r in rows})} groups", flush=True)

    model, tok = load_encoder(ENCODER, device=DEV)

    emo_path = out_dir.parent / f"{tag}_emotion_L{LAYER}.pt"
    if emo_path.exists():
        bundle = torch.load(emo_path)
    else:
        emotions = [e.strip() for e in (EMO / "config/emotions.txt").read_text().split("\n") if e.strip()]
        bundle = build_emotion_vectors(model, tok, EMO / "data/stories",
                                       EMO / "data/neutral/neutral.jsonl", emotions, LAYER,
                                       max_per_emotion=args.max_per_emotion, device=DEV)
        torch.save(bundle, emo_path)
    emotions = bundle["emotions"]
    emo_mat = np.stack([bundle["vectors"][e].numpy() for e in emotions])
    print(f"{len(emotions)} emotion vectors, {bundle['n_denoise_pcs']} neutral PCs projected out", flush=True)

    print("encoding target replies...", flush=True)
    acts_t = _encode_docs(model, tok, [r["target"] for r in rows], LAYER, DEV)
    print("encoding base replies...", flush=True)
    acts_b = _encode_docs(model, tok, [r["base"] for r in rows], LAYER, DEV)
    np.save(out_dir / "emo_acts_target.npy", acts_t)
    np.save(out_dir / "emo_acts_base.npy", acts_b)

    whitener = fit_base_whitening(acts_b)
    wt, wb, we = whitener.transform(acts_t), whitener.transform(acts_b), whitener.transform(emo_mat)
    coords_t, coords_b = persona_coords(wt, we), persona_coords(wb, we)
    drift = coords_t - coords_b

    groups = [r["principal"] for r in rows]
    behavior = np.stack([r["behavior"] for r in rows])
    result = {
        "run": args.run, "encoder": ENCODER, "layer": LAYER, "n_pairs": len(rows),
        "emotions": emotions, "n_denoise_pcs": bundle["n_denoise_pcs"],
        "groups": sorted(set(groups)),
        "display": {g: display.get(g, g.replace("_", " ")) for g in set(groups)},
        "separability_target": group_separability(coords_t, groups, n_perm=args.permutations),
        "separability_base": group_separability(coords_b, groups, n_perm=args.permutations),
        "separability_drift": group_separability(drift, groups, n_perm=args.permutations),
        "group_emotion_means": {g: [round(float(coords_t[np.array(groups) == g, j].mean()), 4)
                                    for j in range(len(emotions))] for g in sorted(set(groups))},
        "emotion_behavior": persona_behavior_map(coords_t, behavior, axes, emotions),
    }
    save_json(out_dir / "emotion_probe.json", result)
    st, sb, sd = (result["separability_target"], result["separability_base"],
                  result["separability_drift"])
    print(f"target sep={st['balanced_accuracy']} (chance {st['chance']}, p={st['p']:.4g})", flush=True)
    print(f"base   sep={sb['balanced_accuracy']} (p={sb['p']:.4g})", flush=True)
    print(f"drift  sep={sd['balanced_accuracy']} (p={sd['p']:.4g})  <- matched-pair emotion treatment", flush=True)


if __name__ == "__main__":
    main()
