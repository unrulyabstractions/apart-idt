"""Stage 4 in the appendix: what was sampled, and what did not come back.

The refusal, empty, and failure counts are the point of the table. A silently
dropped generation would bias the comparison toward the prompts a model found
easy, so every prompt cell is accounted for here even when the count is zero.
Refused and failed are separate columns because a local backend cannot tell a
refusal from a crash: failed marks a generation that raised and was recorded
with empty text, and it is never reported as a refusal. The first run recorded
no failure flag at all, so its column reads as a dash rather than as zero.
"""

from __future__ import annotations

from pathlib import Path

from src.appendix.collection_variant_tables import collection_system_card, variant_breakdown_table
from src.appendix.latex_text_escaping import load, pending, rows, tex
from src.appendix.pipeline_run_registry import (
    expected_dirs,
    reported_runs,
    reported_seats,
    run_group,
)

__all__ = ["collection_runs", "score_runs", "model_names", "collection_section"]


def collection_runs(out_root) -> list[tuple[str, Path, dict]]:
    """Every stage-4 run directory, expected or on disk, with its prompt sets."""
    root = Path(out_root) / "score"
    on_disk = sorted(d.name for d in root.glob("*") if d.is_dir())
    found = []
    for name in reported_runs(sorted(set(on_disk) | set(expected_dirs("collection")))):
        found.append((name, root / name, load(root / name / "prompt_sets.json") or {}))
    return found


def score_runs(out_root) -> list[tuple[str, Path]]:
    """The subset that has prompt sets, which is what the later tables can read."""
    return [(name, directory) for name, directory, art in collection_runs(out_root) if art]


def model_names(directory: Path, prefix: str) -> list[str]:
    """The seats a run wrote files for, read from the file names it wrote."""
    return reported_seats(sorted(p.name[len(prefix) + 1:-len(".jsonl")]
                                 for p in directory.glob(f"{prefix}_*.jsonl")))


def _model_stats(path: Path) -> dict:
    replies = rows(path)
    recorded = any("failed" in r for r in replies)
    return {
        "n": len(replies),
        "refused": sum(1 for r in replies if r.get("refused")),
        "empty": sum(1 for r in replies if not (r.get("text") or "").strip()),
        "failed": sum(1 for r in replies if r.get("failed")) if recorded else None,
        "chars": (sum(len(r.get("text") or "") for r in replies) // max(1, len(replies))),
    }


def _maybe(value) -> str:
    return "--" if value is None else str(value)


def collection_section(out_root) -> str:
    out = ["\\subsection{Response collection}", "\\label{app:data-collection}", "",
           "One row per run and model, then the same accounting split by collection "
           "system-prompt variant.", "",
           "\\begin{table}[H]", "\\centering", "\\small",
           "\\resizebox{\\linewidth}{!}{%",
           "\\begin{tabular}{@{}l l r r r r r r r r@{}}", "\\toprule",
           "Run & Model & groups & prompts & samples & replies & refused & empty & failed & "
           "mean chars \\\\", "\\midrule"]
    for name, directory, art in collection_runs(out_root):
        models = model_names(directory, "responses")
        if not models:
            out.append(f"\\texttt{{{tex(name)}}} & \\multicolumn{{9}}{{l}}"
                       "{\\textcolor{warn}{pending: no reply collected for this run}} \\\\")
            out.append("\\arrayrulecolor{black!25}\\midrule")
            continue
        first = True
        for model in models:
            stats = _model_stats(directory / f"responses_{model}.jsonl")
            label = f"\\texttt{{{tex(name)}}}" if first else ""
            out.append(f"{label} & \\texttt{{{tex(model)}}} & {len(art.get('principals', {}))} & "
                       f"{art.get('n_prompts', '--')} & {art.get('samples_per_cell', '--')} & "
                       f"{stats['n']} & {stats['refused']} & {stats['empty']} & "
                       f"{_maybe(stats['failed'])} & {stats['chars']} \\\\")
            first = False
        out.append("\\arrayrulecolor{black!25}\\midrule")
    out[-1] = "\\bottomrule"
    out += ["\\end{tabular}}",
            "\\caption{Stage 4 per run and model. \\emph{Groups} is the candidate principals "
            "the run compares, \\emph{prompts} is candidates $\\times$ instructions $\\times$ "
            "collection system variants where the run defines them, and replies is prompts "
            "$\\times$ samples. Refused, empty, and failed are counted independently, so one "
            "reply can appear in more than one column; a failed generation is one that raised "
            "and is never counted as a refusal. A dash is a field the run did not record.}",
            "\\label{tab:data-collection}", "\\end{table}", ""]
    runs = score_runs(out_root)
    for extra in (variant_breakdown_table(runs), collection_system_card(runs), _reply_card(runs)):
        if extra:
            out.append(extra)
    return "\n".join(out)


def _reply_card(runs) -> str:
    """One sampled reply from an organism seat, so the reader sees what the judge reads.

    The model is chosen by excluding the base seat rather than by sort order: the
    interesting reply is the organism's, and which name sorts last is an accident
    of the tags.
    """
    for name, directory in runs:
        organisms = [s for s in model_names(directory, "responses") if not s.startswith("base")]
        if not organisms:
            continue
        target = organisms[0]
        calibration = run_group(name) == "calibration"
        env = "promptcardCal" if calibration else "promptcardChal"
        lab = "callabel" if calibration else "challabel"
        for reply in rows(directory / f"responses_{target}.jsonl"):
            text = tex((reply.get("text") or "").strip())
            if len(text) > 200:
                return (f"\\begin{{{env}}}\n\\{lab}{{{tex(name.upper())} $\\cdot$ "
                        f"\\texttt{{{tex(target)}}} $\\cdot$ "
                        f"\\texttt{{{tex(reply['prompt_id'])}}}}}\n"
                        f"{text[:700]}\n\\end{{{env}}}\n")
    return pending("no reply is long enough to quote")
