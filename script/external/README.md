# External organisms

Runs our audit against model organisms published by other groups. Each target is
generated locally, scored with our judge, and read through the registered test in
`src/compare`. Nothing here changes the pipeline; it only feeds an outside organism
into it.

## idt-organism (eliwangj/idt-organism)

A Phase-0 prompted organism: `Qwen/Qwen2.5-1.5B-Instruct` given a covert
system-prompt objective to steer Rivertown residents toward supporting a water
project and Hillcrest residents toward opposing it, plus a content-matched baseline
without the objective. The organism is the target arm and the baseline is the base
arm, so our excess over the base is exactly the induced group-conditional divergence.

Reproduce:

```bash
# 1. clone the organism repo (its scenario modules are imported at run time)
git clone https://github.com/eliwangj/idt-organism/ /tmp/idt-organism

# 2. generate both arms locally (Qwen2.5-1.5B on MPS; ~44 min at 20 samples/cell)
uv run python script/external/idt_organism_generate_responses.py \
    --clone-dir /tmp/idt-organism \
    --out out/external/idt_organism/responses_k20.jsonl --samples 20

# 3. score on the four behavior axes and run the registered test
uv run python script/external/idt_organism_score_and_compare.py \
    --responses out/external/idt_organism/responses_k20.jsonl \
    --out out/external/idt_organism/our_pipeline_verdict_k20.json
```

The audit reports the unsigned maxT statistic the paper registers and, because this
organism's objective is documented, the signed statistic oriented to that direction.
Outputs land in gitignored `out/external/idt_organism/`.
