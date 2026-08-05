"""Distill one experiment's artifacts into a bundle small enough to embed.

The raw responses run to millions of rows and tens of megabytes, which no page
can hold. This reads them once and keeps three things: the aggregates every view
needs, the full per-cell rate grid the registered test reads, and a bounded
sample of real transcripts paired between the target and its base model. Nothing
here recomputes a statistic. The verdict, the principal, and the axes come from
the stage-6 summary as written.

The sample is deliberate rather than uniform. A reviewer opening the deepest
layer wants the cells that carry the signal, so the transcripts are drawn from
the widest cells of the named candidate first, then filled out across the ladder
so every group and both extremes of every axis are represented.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from src.common.file_io import load_json, read_jsonl
from src.ui.experiment_registry import ExperimentSource

__all__ = ["build_experiment_bundle"]

#: Transcripts kept per experiment. Enough to read the behavior, bounded so the
#: whole explorer stays under the artifact size ceiling.
SAMPLE_PER_EXPERIMENT = 48
#: Characters kept of a reply. A full reply is rarely longer, and the tail of a
#: runaway generation is not worth the bytes.
REPLY_CLIP = 1600


def _axis_lookup(axes_path: Path) -> dict:
    axes = load_json(axes_path).get("axes", [])
    return {a["axis_id"]: a.get("question", "") for a in axes}, [a["axis_id"] for a in axes]


def _cell_key(row: dict) -> tuple:
    return (row["principal"], row["instruction_id"])


def _fold_verdicts(path: Path, axis_ids: list[str]) -> dict:
    """Per (candidate, instruction, axis): fired count and scored count."""
    fired: dict = defaultdict(lambda: defaultdict(int))
    scored: dict = defaultdict(lambda: defaultdict(int))
    nulls = 0
    n = 0
    for row in read_jsonl(path):
        n += 1
        key = _cell_key(row)
        for axis, verdict in row["verdicts"].items():
            if verdict is None:
                nulls += 1
                continue
            scored[key][axis] += 1
            if verdict:
                fired[key][axis] += 1
    return {"fired": fired, "scored": scored, "nulls": nulls, "n": n}


def _rate_grid(folded: dict, principals: list[str], instructions: list[str],
               axis_ids: list[str]) -> list:
    """The (candidate, instruction, axis) rate cube, as a nested list."""
    fired, scored = folded["fired"], folded["scored"]
    cube = []
    for c in principals:
        rows = []
        for i in instructions:
            cell = []
            for a in axis_ids:
                s = scored[(c, i)].get(a, 0)
                cell.append(round(fired[(c, i)].get(a, 0) / s, 4) if s else None)
            rows.append(cell)
        cube.append(rows)
    return cube


def _candidate_axis_means(cube: list, base_cube: list, principals: list[str],
                          axis_ids: list[str]) -> list:
    """Mean rate per (candidate, axis) in target and base, over instructions."""
    out = []
    for ci, c in enumerate(principals):
        for ai, a in enumerate(axis_ids):
            tvals = [cube[ci][ii][ai] for ii in range(len(cube[ci]))
                     if cube[ci][ii][ai] is not None]
            bvals = [base_cube[ci][ii][ai] for ii in range(len(base_cube[ci]))
                     if base_cube[ci][ii][ai] is not None]
            if not tvals and not bvals:
                continue
            tm = sum(tvals) / len(tvals) if tvals else None
            bm = sum(bvals) / len(bvals) if bvals else None
            out.append({"candidate": c, "axis": a,
                        "target": round(tm, 4) if tm is not None else None,
                        "base": round(bm, 4) if bm is not None else None,
                        "excess": round(tm - bm, 4) if tm is not None and bm is not None else None})
    return out


def _response_index(path: Path) -> dict:
    """Every reply, keyed by (prompt_id, sample), read once."""
    out = {}
    for row in read_jsonl(path):
        out[(row["prompt_id"], int(row["s"]))] = row
    return out


def _prompt_texts(src: ExperimentSource) -> dict:
    """The rendered user prompt for every prompt id, from the prompt sets.

    A response row carries no prompt text, only an instruction id, so the text
    is recovered here. The verdict's instruction id may carry a system prefix
    (``prism4::template``) that the prompt set's own ids lack, so both the full
    id and its final segment are keyed.
    """
    roots = []
    if src.prompt_sets and Path(src.prompt_sets).exists():
        roots.append(Path(src.prompt_sets).parent)
    # The AuditBench and organism prompt sets live beside the axes, one file per
    # candidate, rather than in a single prompt_sets.json.
    axes_dir = Path(src.axes).parent
    roots.append(axes_dir)
    texts = {}
    for root in roots:
        for path in sorted(root.glob("prompts_*.json")):
            for p in load_json(path).get("prompts", []):
                texts[p["prompt_id"]] = p.get("text", "")
        combined = root / "prompt_sets.json"
        if combined.exists():
            for _c, prompts in load_json(combined).get("prompt_sets", {}).items():
                for p in prompts:
                    texts[p["prompt_id"]] = p.get("text", "")
    return texts


def _pick_cells(summary: dict, level_key: str, principals: list[str],
                instructions: list[str]) -> list:
    """(candidate, instruction) pairs to sample, signal-carrying first."""
    pairs = []
    test = (summary.get("reference_contrast", {}).get(level_key, {})
            .get("paired_max_test", {}))
    named = (test.get("attribution") or {}).get("named") or test.get("principal")
    top = test.get("top_pairs", [])
    seen = set()
    for pair in top:
        c = pair.get("candidate")
        if c in principals:
            for i in instructions[:4]:
                if (c, i) not in seen:
                    pairs.append((c, i))
                    seen.add((c, i))
    for c in ([named] if named in principals else []) + principals:
        for i in instructions:
            if (c, i) not in seen:
                pairs.append((c, i))
                seen.add((c, i))
    return pairs


def _level_key(summary: dict) -> str:
    contrast = summary.get("reference_contrast", {})
    for k, v in contrast.items():
        if isinstance(v, dict) and v.get("paired_max_test"):
            return k
    return next(iter(contrast), "L2")


def build_experiment_bundle(src: ExperimentSource) -> dict:
    """One experiment, distilled. Missing artifacts degrade to empty, never raise."""
    bundle: dict = {"key": src.key, "title": src.title, "family": src.family,
                    "role": src.role, "cue": src.cue, "judge": src.judge,
                    "present": {}}

    summary_path = Path(src.summary)
    summary = load_json(summary_path) if summary_path.exists() else {}
    bundle["present"]["summary"] = bool(summary)
    bundle["summary"] = summary

    axes_path = Path(src.axes)
    if not axes_path.exists():
        bundle["axes"] = {}
        return bundle
    axis_q, axis_ids = _axis_lookup(axes_path)
    bundle["axis_questions"] = axis_q
    bundle["axis_ids"] = axis_ids

    vt, vb = Path(src.verdicts_target), Path(src.verdicts_base)
    bundle["present"]["verdicts_target"] = vt.exists()
    bundle["present"]["verdicts_base"] = vb.exists()
    if not vt.exists() or not vb.exists():
        return bundle

    ft, fb = _fold_verdicts(vt, axis_ids), _fold_verdicts(vb, axis_ids)
    principals = sorted({c for c, _ in ft["fired"]} | {c for c, _ in ft["scored"]})
    instructions = sorted({i for _, i in ft["scored"]})
    display = {}
    if src.prompt_sets and Path(src.prompt_sets).exists():
        display = load_json(src.prompt_sets).get("principals", {})
    bundle["principals"] = principals
    bundle["display"] = {p: display.get(p, p.replace("_", " ")) for p in principals}
    bundle["instructions"] = instructions
    bundle["null_counts"] = {"target": ft["nulls"], "base": fb["nulls"]}
    bundle["n_verdicts"] = {"target": ft["n"], "base": fb["n"]}

    cube = _rate_grid(ft, principals, instructions, axis_ids)
    base_cube = _rate_grid(fb, principals, instructions, axis_ids)
    bundle["rate_grid"] = {"target": cube, "base": base_cube}
    bundle["candidate_axis"] = _candidate_axis_means(cube, base_cube, principals, axis_ids)

    rt, rb = Path(src.responses_target), Path(src.responses_base)
    bundle["present"]["responses"] = rt.exists() and rb.exists()
    if rt.exists() and rb.exists():
        idx_t, idx_b = _response_index(rt), _response_index(rb)
        prompts = _prompt_texts(src)
        level = _level_key(summary)
        transcripts = []
        for c, i in _pick_cells(summary, level, principals, instructions):
            if len(transcripts) >= SAMPLE_PER_EXPERIMENT:
                break
            pid = f"{c}__{i}"
            template = i.split("::", 1)[-1]
            prompt = (prompts.get(pid) or prompts.get(f"{c}__{template}") or "")
            for s in range(4):
                rowt, rowb = idx_t.get((pid, s)), idx_b.get((pid, s))
                if rowt is None:
                    continue
                transcripts.append({
                    "candidate": c, "instruction": i, "sample": s,
                    "prompt": prompt[:REPLY_CLIP],
                    "target": (rowt.get("text") or "")[:REPLY_CLIP],
                    "base": (rowb.get("text") or "")[:REPLY_CLIP] if rowb else "",
                    "target_refused": bool(rowt.get("refused")),
                    "target_failed": bool(rowt.get("failed")),
                })
                break
        bundle["transcripts"] = transcripts

    geo = Path(src.geometry)
    bundle["present"]["geometry"] = geo.exists()
    if geo.exists():
        bundle["geometry"] = load_json(geo)

    bridge = geo.parent / "semantic_bridge.json" if src.geometry else None
    bundle["present"]["bridge"] = bool(bridge and bridge.exists())
    if bridge and bridge.exists():
        bundle["semantic_bridge"] = load_json(bridge)

    persona = Path("out/persona") / src.key / "persona_probe.json"
    bundle["present"]["persona"] = persona.exists()
    if persona.exists():
        bundle["persona"] = load_json(persona)

    emotion = Path("out/persona") / src.key / "emotion_probe.json"
    bundle["present"]["emotion"] = emotion.exists()
    if emotion.exists():
        bundle["emotion"] = load_json(emotion)

    if src.extra.get("training") and Path(src.extra["training"]).exists():
        bundle["training"] = load_json(src.extra["training"])

    return bundle


def _prompt_of(row: dict) -> str:
    """The user prompt a response was drawn for, if the row carries it."""
    return row.get("prompt") or row.get("text_prompt") or ""
