"""Geometry appendix artifacts: maps, directions, tables, macros.

    uv run python script/geometry/render_behavior_geometry.py

Reads the rerun's verdicts and stage-6 reports for the six reported runs,
writes one directory per run under out/geometry/, copies the figures the
appendix includes into paper/figures/geometry/, and regenerates the three
generated fragments under paper/appendix/. Nothing here samples a model or
recomputes a test: every registered verdict is read from the stage-6 artifacts
on disk, so no statistic can be altered from here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.common.file_io import save_json
from src.geometry.behavior_cell_vectors import GEOMETRY_RUNS, load_run_vectors
from src.geometry.behavior_map_figure import plot_behavior_map
from src.geometry.behavior_space_decomposition import (
    displacement_decomposition,
    variance_partition,
)
from src.geometry.excess_map_figure import plot_excess_map
from src.geometry.geometry_appendix_fragments import write_geometry_fragments
from src.geometry.loyalty_direction_figure import plot_loyalty_direction
from src.geometry.mean_excess_biplot_figure import plot_mean_excess_biplot


def _direction_candidate(rv) -> str:
    """The candidate whose direction the run's figure reads.

    The registered principal where the test rejects; otherwise the candidate
    attaining the test's maximum, whose direction is what a rejection would
    have named and is shown as the null case.
    """
    reg = rv.registered or {}
    if reg.get("principal"):
        return reg["principal"]
    if reg.get("top_pairs"):
        return reg["top_pairs"][0]["candidate"]
    dec = displacement_decomposition(rv.target, rv.base)
    return rv.principals[int(np.argmax(dec["residual_norms"]))]


def _copy_figures(run_dir: Path, figure_dir: Path, name: str) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(run_dir.glob("*.pdf")) + sorted(run_dir.glob("*.png")):
        (figure_dir / f"{name}_{source.name}").write_bytes(source.read_bytes())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", default="out/geometry")
    ap.add_argument("--paper-dir", default="paper",
                    help="paper tree receiving figures/geometry and appendix fragments")
    args = ap.parse_args()
    out_root, paper = Path(args.out_root), Path(args.paper_dir)

    summaries = []
    for run in GEOMETRY_RUNS:
        rv = load_run_vectors(run)
        run_dir = out_root / run.name
        highlight = (rv.registered or {}).get("principal")
        direction = _direction_candidate(rv)
        dec = displacement_decomposition(rv.target, rv.base)
        order = np.argsort(-dec["residual_norms"])

        summary = {
            "name": rv.name, "title": rv.title, "level": rv.level,
            "n_candidates": len(rv.principals), "n_instructions": len(rv.instructions),
            "n_axes": len(rv.axes), "display": dict(rv.display),
            "registered": rv.registered,
            "variance_partition_target": variance_partition(rv.target),
            "variance_partition_base": variance_partition(rv.base),
            "displacement": {
                "common_norm": dec["common_norm"],
                "residual_norms": {p: float(n) for p, n in
                                   zip(rv.principals, dec["residual_norms"])},
                "top_candidate": rv.principals[int(order[0])],
                "top_norm": float(dec["residual_norms"][order[0]]),
                "runner_up_norm": float(dec["residual_norms"][order[1]]),
            },
            "mean_excess_biplot": plot_mean_excess_biplot(
                rv, run_dir / "mean_excess_biplot.png"),
            "behavior_map": plot_behavior_map(
                rv, run_dir / "behavior_map.png", highlight=highlight),
            "excess_map": plot_excess_map(
                rv, run_dir / "excess_map.png", highlight=highlight),
            "loyalty_direction": plot_loyalty_direction(
                rv, run_dir / "loyalty_direction.png", candidate=direction),
        }
        save_json(run_dir / "geometry_summary.json", summary)
        _copy_figures(run_dir, paper / "figures" / "geometry", run.name)
        summaries.append(summary)

        v_t, disp = summary["variance_partition_target"], summary["displacement"]
        print(f"[{rv.name} L{rv.level}] cells={v_t['n_cells']} "
              f"instr={v_t['instruction_share']:.1%} cand={v_t['candidate_share']:.1%} "
              f"common={disp['common_norm']:.3f} "
              f"top-residual={disp['top_candidate']}={disp['top_norm']:.3f} "
              f"direction={direction}", flush=True)

    for path in write_geometry_fragments(summaries, paper / "appendix"):
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
