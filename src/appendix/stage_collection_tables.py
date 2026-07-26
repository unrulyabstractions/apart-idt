"""Stage 4 in the appendix: what was sampled, and what failed.

The refusal and empty-text counts are the point of the table. A silently dropped
generation would bias the comparison toward the prompts a model found easy, so
every prompt cell is accounted for here even when the count is zero.
"""

from __future__ import annotations

from pathlib import Path

from src.appendix.latex_text_escaping import load, pending, rows, tex

__all__ = ["score_runs", "collection_section"]


def score_runs(out_root) -> list[tuple[str, Path]]:
    """Every stage-4 run directory on disk, so the appendix cannot miss one."""
    root = Path(out_root) / "score"
    return [(d.name, d) for d in sorted(root.glob("*")) if (d / "prompt_sets.json").exists()]


def _model_stats(path: Path) -> dict:
    replies = rows(path)
    return {
        "n": len(replies),
        "refused": sum(1 for r in replies if r.get("refused")),
        "empty": sum(1 for r in replies if not (r.get("text") or "").strip()),
        "chars": (sum(len(r.get("text") or "") for r in replies) // max(1, len(replies))),
    }


def collection_section(out_root) -> str:
    runs = score_runs(out_root)
    out = ["\\subsection{Response collection}", "\\label{app:data-collection}", ""]
    if not runs:
        return "\n".join(out + [pending("stage not run")])
    out += ["One row per model.", "",
            "\\begin{table}[H]", "\\centering", "\\small",
            "\\begin{tabular}{@{}l l r r r r r@{}}", "\\toprule",
            "Run & Model & groups & prompts & replies & refused & mean chars \\\\", "\\midrule"]
    for name, d in runs:
        art = load(d / "prompt_sets.json")
        first = True
        for model in sorted(p.name[len("responses_"):-len(".jsonl")]
                           for p in d.glob("responses_*.jsonl")):
            s = _model_stats(d / f"responses_{model}.jsonl")
            label = f"\\texttt{{{tex(name)}}}" if first else ""
            out.append(f"{label} & \\texttt{{{tex(model)}}} & {len(art['principals'])} & "
                       f"{art['n_prompts']} & {s['n']} & {s['refused'] + s['empty']} & "
                       f"{s['chars']} \\\\")
            first = False
        out.append("\\arrayrulecolor{black!25}\\midrule")
    out[-1] = "\\bottomrule"
    out += ["\\end{tabular}",
            "\\caption{Stage 4 per run and model. Prompts are candidates $\\times$ instructions; "
            "replies are prompts $\\times$ samples.}",
            "\\label{tab:data-collection}", "\\end{table}", ""]
    out.append(_reply_card(runs))
    return "\n".join(out)


def _reply_card(runs) -> str:
    """One sampled reply from the organism, so the reader sees what the judge reads.

    The model is chosen by excluding the reference rather than by sort order: the
    interesting reply is the organism's, and which name sorts last is an accident
    of the tags.
    """
    for name, d in runs:
        seats = sorted(p.name[len("responses_"):-len(".jsonl")]
                       for p in d.glob("responses_*.jsonl"))
        organisms = [s for s in seats if not s.startswith("base")]
        if not organisms:
            continue
        target = organisms[0]
        env = "promptcardCal" if name.startswith("calibration") else "promptcardChal"
        lab = "callabel" if name.startswith("calibration") else "challabel"
        for r in rows(d / f"responses_{target}.jsonl"):
            text = tex((r.get("text") or "").strip())
            if len(text) > 200:
                return (f"\\begin{{{env}}}\n\\{lab}{{{tex(name.upper())} $\\cdot$ "
                        f"\\texttt{{{tex(target)}}} $\\cdot$ "
                        f"\\texttt{{{tex(r['prompt_id'])}}}}}\n"
                        f"{text[:700]}\n\\end{{{env}}}\n")
    return ""
