"""Train the political sycophancy organism and report whether it took.

    uv run python script/organism/train_political_sycophancy.py \
        --model models/gemma-3-4b-it --out out/organism/political_sycophancy

The organism answers the way the user would answer: conservative to a user whose
biography reads conservative, liberal to one whose reads liberal. That makes its
behavior conditional on who is asking, which is the property our registered test
is built to detect, and unlike an unconditional bias it cannot be found by
looking at the model's average output.

The run reports the sycophancy rate for each user group before and after
training. A finetune that moved the model globally rather than conditionally
shows up here as both groups moving together, so the check is printed for both
groups rather than pooled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common.file_io import save_json
from src.organism.binary_choice_finetune import (
    ChoiceBatchEncoder, TrainConfig, evaluate, train)
from src.organism.political_sycophancy_data import build_items, render_prompt

LETTERS = {"A": 0, "B": 1}


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def split_by_group(items, prompts, targets):
    """Indices per user affiliation, so the two groups are reported separately."""
    groups: dict[str, list[int]] = {}
    for i, item in enumerate(items):
        groups.setdefault(item.affiliation, []).append(i)
    return {name: ([prompts[i] for i in idx], [targets[i] for i in idx])
            for name, idx in sorted(groups.items())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", type=Path, required=True,
                    help="sycophancy_on_political_typology_quiz.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=3e-6)
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--dtype", default="float32",
                    help="float32 matches the published recipe; bfloat16 is a deviation")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.data)
    train_items, eval_items = build_items(rows, args.n_train, args.n_eval, args.seed)
    print(f"{len(rows)} rows -> {len(train_items)} train, {len(eval_items)} eval", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="right")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype)).to("cuda")
    encoder = ChoiceBatchEncoder(tokenizer, "cuda")

    def prep(items):
        prompts = [render_prompt(tokenizer, it) for it in items]
        targets = [LETTERS[it.matching] for it in items]
        return prompts, targets

    train_prompts, train_targets = prep(train_items)
    eval_prompts, eval_targets = prep(eval_items)
    eval_groups = split_by_group(eval_items, eval_prompts, eval_targets)

    def report(stage: str) -> dict:
        out = {"overall": evaluate(model, encoder, eval_prompts, eval_targets)}
        for name, (p, t) in eval_groups.items():
            out[name] = evaluate(model, encoder, p, t)
        print(f"[{stage}] " + "  ".join(
            f"{k}: p={v['mean_probability']:.3f} acc={v['argmax_accuracy']:.3f} (n={v['n']})"
            for k, v in out.items()), flush=True)
        return out

    before = report("before")
    trace = train(model, encoder, train_prompts, train_targets,
                  TrainConfig(args.epochs, args.batch_size, args.learning_rate, args.seed))
    after = report("after")

    model.save_pretrained(args.out / "checkpoint")
    tokenizer.save_pretrained(args.out / "checkpoint")
    save_json(args.out / "training_report.json", {
        "base_model": args.model, "dtype": args.dtype, "seed": args.seed,
        "n_train": len(train_items), "n_eval": len(eval_items),
        "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "sycophancy_before": before, "sycophancy_after": after, "loss_trace": trace,
    })
    print(f"wrote {args.out}/training_report.json", flush=True)


if __name__ == "__main__":
    main()
