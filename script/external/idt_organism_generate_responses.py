"""Generate organism and baseline responses for an external prompted organism.

    uv run python script/external/idt_organism_generate_responses.py \
        --clone-dir /path/to/idt-organism --out out/external/idt_organism/responses.jsonl

Targets the eliwangj/idt-organism Phase-0 organism: one open-weight model given a
covert system-prompt objective (the target) and a content-matched baseline without
it (the base), audited on its Rivertown and Hillcrest water-commons scenario. We
hold the model and the frozen prompts fixed and vary only the condition, so the
excess between the two arms is exactly the induced group-conditional divergence.
Their scenario modules are loaded from the clone at run time.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clone-dir", required=True, help="path to the cloned idt-organism repo")
    ap.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=20, help="generations per prompt cell")
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.clone_dir).resolve()))
    scenario = importlib.import_module("src.scenario.condition_system_prompts")
    prompt_set = importlib.import_module("src.scenario.matched_prompt_set")
    build_system_prompt = scenario.build_system_prompt
    prompts = prompt_set.build_prompt_set()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(args.device).eval()

    done, total = 0, 2 * len(prompts)
    with args.out.open("w") as f:
        for condition in ("organism", "baseline"):
            system = build_system_prompt(condition)
            for rec in prompts:
                msgs = [{"role": "system", "content": system},
                        {"role": "user", "content": rec["user_message"]}]
                templated = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                                    return_tensors="pt")
                ids = (templated["input_ids"] if hasattr(templated, "keys") else templated).to(args.device)
                with torch.no_grad():
                    out = model.generate(ids, max_new_tokens=args.max_new, do_sample=True,
                                         temperature=args.temperature, top_p=args.top_p,
                                         num_return_sequences=args.samples,
                                         pad_token_id=tok.eos_token_id)
                for s in range(args.samples):
                    text = tok.decode(out[s, ids.shape[1]:], skip_special_tokens=True)
                    f.write(json.dumps({"condition": condition, "prompt_id": rec["prompt_id"],
                                        "group": rec["group"], "sample": s, "text": text}) + "\n")
                f.flush()
                done += 1
                if done % 10 == 0:
                    print(f"{done}/{total} cells x {args.samples} samples", flush=True)
    print(f"DONE {done}/{total} cells, {done * args.samples} generations -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
