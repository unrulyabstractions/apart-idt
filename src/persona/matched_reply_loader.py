"""Load matched target and base replies for one run, with prompts and verdicts.

Both the persona and emotion probes read the same thing: for each cell, a capped
number of matched samples, each carrying the prompt, the target reply, the base
reply on the identical prompt, and the reply's behavior verdicts. Keeping this in
one place means the two probes align on exactly the same replies.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.common.file_io import load_json, read_jsonl


def _prompt_lookup(src) -> dict:
    combined = Path(src.prompt_sets) if src.prompt_sets else Path(src.axes).parent / "prompt_sets.json"
    texts = {}
    if combined.exists():
        for _c, prompts in load_json(combined).get("prompt_sets", {}).items():
            for p in prompts:
                texts[p["prompt_id"]] = p.get("text", "")
    return texts


def load_matched_replies(src, axes: list[str], cap: int) -> list[dict]:
    """Up to ``cap`` matched (target, base) samples per cell, in cell order."""
    prompts = _prompt_lookup(src)
    idx = {a: j for j, a in enumerate(axes)}
    rt = {(r["prompt_id"], int(r["s"])): r for r in read_jsonl(Path(src.responses_target))}
    rb = {(r["prompt_id"], int(r["s"])): r for r in read_jsonl(Path(src.responses_base))}
    verd = {(v["prompt_id"], int(v["s"])): v for v in read_jsonl(Path(src.verdicts_target))}
    per_cell: dict = {}
    rows = []
    for key in sorted(rt.keys() & rb.keys()):
        r = rt[key]
        cell = (r["principal"], r["instruction_id"])
        if per_cell.get(cell, 0) >= cap:
            continue
        per_cell[cell] = per_cell.get(cell, 0) + 1
        template = r["instruction_id"].split("::", 1)[-1]
        prompt = prompts.get(f"{r['principal']}__{r['instruction_id']}") \
            or prompts.get(f"{r['principal']}__{template}") or ""
        vec = np.full(len(axes), np.nan, dtype=np.float32)
        for a, val in (verd.get(key, {}).get("verdicts") or {}).items():
            if a in idx and val is not None:
                vec[idx[a]] = 1.0 if val else 0.0
        rows.append({"principal": r["principal"], "prompt": prompt,
                     "target": rt[key].get("text") or "", "base": rb[key].get("text") or "",
                     "behavior": vec})
    return rows
