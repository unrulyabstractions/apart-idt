"""Stage 4 for an AuditBench organism published as a LoRA adapter.

    uv run python script/pipeline/collect_auditbench.py \
        --promptset out/auditbench/promptset/contextual_optimism \
        --base models/Llama-3.3-70B-Instruct \
        --adapter models/organism_contextual_optimism \
        --out out/auditbench/responses/contextual_optimism

Both arms come out of one engine. The organism is its base plus a LoRA delta,
so attaching the adapter gives the target and detaching it gives the control,
with the same weights, the same cache settings and the same seed. Two separate
engine loads could differ in ways no difference-of-differences would cancel, and
loading 141 GB twice would double the bill for the privilege.

Sampling is GPU-bound and runs on the rented box. Scoring is API-bound and runs
at home, so no key ever travels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.common.file_io import save_json
from src.runner.adapter_arm_view import AdapterArmView
from src.runner.auditbench_identity_prompt import AUDITBENCH_SYSTEM_VARIANTS
from src.runner.vllm_batch_backend import VllmBatchBackend
from src.score.response_sampling import sample_prompt_sets


def load_prompt_sets(root: Path) -> dict[str, list[dict]]:
    """The rendered prompt set for every candidate, keyed by candidate."""
    sets: dict[str, list[dict]] = {}
    for path in sorted(root.glob("prompts_*.json")):
        key = path.stem[len("prompts_"):]
        sets[key] = json.loads(path.read_text())["prompts"]
    if not sets:
        raise SystemExit(f"no prompts_*.json under {root}")
    return sets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--promptset", type=Path, required=True)
    ap.add_argument("--base", required=True, help="path or id of the base checkpoint")
    ap.add_argument("--adapter", required=True, help="path of the LoRA adapter")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--samples-per-prompt", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--tensor-parallel-size", type=int, default=4)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-lora-rank", type=int, default=64)
    args = ap.parse_args()

    prompt_sets = load_prompt_sets(args.promptset)
    args.out.mkdir(parents=True, exist_ok=True)
    n_prompts = sum(len(v) for v in prompt_sets.values())
    print(f"{len(prompt_sets)} candidates, {n_prompts} prompts, "
          f"{args.samples_per_prompt} samples each, both arms", flush=True)

    backend = VllmBatchBackend(
        model_name=args.base,
        seed_label="auditbench-collect",
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        lora_path=args.adapter,
        max_lora_rank=args.max_lora_rank,
    )

    stats = {}
    # Target first. If the run dies partway the organism arm is the one worth
    # having, since the base arm can be resampled from the same engine later.
    for tag, use_adapter in (("target", True), ("base", False)):
        arm = AdapterArmView(backend, use_adapter=use_adapter, tag=tag)
        path = args.out / f"responses_{tag}.jsonl"
        print(f"[{tag}] sampling -> {path}", flush=True)
        result = sample_prompt_sets(
            arm, prompt_sets, args.samples_per_prompt, path,
            system_prompts=AUDITBENCH_SYSTEM_VARIANTS,
            max_new_tokens=args.max_new_tokens,
        )
        stats[tag] = {"backend": arm.name, "requested": result.requested,
                      "generated": result.generated,
                      "skipped_existing": result.skipped_existing,
                      "failed": result.failed}
        print(f"[{tag}] requested={result.requested} generated={result.generated} "
              f"failed={result.failed}", flush=True)

    # Stage 6 reads the candidate display names from this file, so a run that
    # omits it collects fine and then cannot be compared. It is written here
    # rather than by hand, in the same shape the other collector writes.
    save_json(args.out / "prompt_sets.json", {
        "condition": args.promptset.name,
        "principals": {key: prompts[0]["principal"]
                       for key, prompts in sorted(prompt_sets.items()) if prompts},
        "collection_system_prompts": [{"id": i, "text": t}
                                      for i, t in AUDITBENCH_SYSTEM_VARIANTS],
        "samples_per_cell": args.samples_per_prompt,
        "n_prompts": n_prompts,
        "prompt_sets": prompt_sets,
    })

    save_json(args.out / "collection_report.json", {
        "base": args.base,
        "adapter": args.adapter,
        "samples_per_prompt": args.samples_per_prompt,
        "system_variants": [v[0] for v in AUDITBENCH_SYSTEM_VARIANTS],
        "candidates": sorted(prompt_sets),
        "n_prompts_per_candidate": {k: len(v) for k, v in prompt_sets.items()},
        "arms": stats,
    })
    print(f"wrote {args.out}/collection_report.json", flush=True)


if __name__ == "__main__":
    main()
