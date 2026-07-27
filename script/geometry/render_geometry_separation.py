"""Separation views: the aggregation that makes the loyalty visually separable.

    uv run python script/geometry/render_geometry_separation.py

Writes out/geometry/<run>/explore3/: the bootstrap mean clouds and the
evidence-accumulation race. Views of quantities the pipeline already computes;
no test is run and no statistic is altered.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.common.file_io import save_json
from src.geometry.behavior_cell_vectors import GEOMETRY_RUNS, load_run_vectors
from src.geometry.behavior_space_decomposition import displacement_decomposition
from src.geometry.bootstrap_mean_cloud_figure import plot_bootstrap_mean_clouds
from src.geometry.evidence_accumulation_figure import plot_evidence_accumulation


def _direction_candidate(rv) -> str:
    reg = rv.registered or {}
    if reg.get("principal"):
        return reg["principal"]
    if reg.get("top_pairs"):
        return reg["top_pairs"][0]["candidate"]
    dec = displacement_decomposition(rv.target, rv.base)
    return rv.principals[int(np.argmax(dec["residual_norms"]))]


def _copy_figures(run_dir: Path, figure_dir: Path, name: str) -> None:
    # The appendix includes these by the same <run>_<figure> convention the
    # behavior-geometry renderer uses. That renderer globs its own run
    # directory, which does not reach this subdirectory, so these are copied
    # here: without it the appendix references a file no stage ever wrote.
    figure_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(run_dir.glob("*.pdf")) + sorted(run_dir.glob("*.png")):
        (figure_dir / f"{name}_{source.name}").write_bytes(source.read_bytes())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", default="out/geometry")
    ap.add_argument("--paper-dir", default="paper",
                    help="paper tree receiving figures/geometry")
    args = ap.parse_args()
    out_root, paper = Path(args.out_root), Path(args.paper_dir)

    for run in GEOMETRY_RUNS:
        rv = load_run_vectors(run)
        out = out_root / run.name / "explore3"
        highlight = (rv.registered or {}).get("principal")
        direction = _direction_candidate(rv)
        summary = {
            "name": rv.name, "level": rv.level, "highlight": highlight,
            "direction_candidate": direction,
            "bootstrap_clouds": plot_bootstrap_mean_clouds(
                rv, out / "bootstrap_clouds.png", highlight=highlight),
            "accumulation": plot_evidence_accumulation(
                rv, out / "evidence_accumulation.png", candidate=direction),
        }
        save_json(out / "separation_summary.json", summary)
        _copy_figures(out, paper / "figures" / "geometry", run.name)
        acc = summary["accumulation"]
        print(f"[{rv.name} L{rv.level}] direction={direction} "
              f"final={acc['final_running_mean'][direction]:+.3f} "
              f"band={acc['final_band']:.3f}", flush=True)


if __name__ == "__main__":
    main()
