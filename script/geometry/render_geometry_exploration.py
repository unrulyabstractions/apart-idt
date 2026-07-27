"""Exploratory geometry views beyond the appendix's three: filters, colors,
three components, nonlinear embeddings, and direction structure.

    uv run python script/geometry/render_geometry_exploration.py

Writes out/geometry/<run>/explore/ for the six reported rerun runs. Everything
here is a view of quantities the pipeline already computed; no test is run and
no statistic is altered. The registered verdicts drawn on the figures are read
from the saved stage-6 artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.common.file_io import save_json
from src.geometry.behavior_cell_vectors import GEOMETRY_RUNS, load_run_vectors
from src.geometry.behavior_space_decomposition import displacement_decomposition
from src.geometry.colored_cloud_figures import (
    plot_candidate_colored_map,
    plot_framing_colored_map,
)
from src.geometry.direction_structure_figures import (
    plot_direction_cosines,
    plot_framing_breakdown,
)
from src.geometry.filtered_axis_map_figure import plot_filtered_axis_map
from src.geometry.nonlinear_embedding_figure import plot_nonlinear_embeddings
from src.geometry.three_component_map_figure import plot_three_component_map


def _direction_candidate(rv) -> str:
    reg = rv.registered or {}
    if reg.get("principal"):
        return reg["principal"]
    if reg.get("top_pairs"):
        return reg["top_pairs"][0]["candidate"]
    dec = displacement_decomposition(rv.target, rv.base)
    return rv.principals[int(np.argmax(dec["residual_norms"]))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", default="out/geometry")
    args = ap.parse_args()
    out_root = Path(args.out_root)

    for run in GEOMETRY_RUNS:
        rv = load_run_vectors(run)
        explore = out_root / run.name / "explore"
        highlight = (rv.registered or {}).get("principal")
        direction = _direction_candidate(rv)

        summary = {
            "name": rv.name, "level": rv.level, "highlight": highlight,
            "direction_candidate": direction,
            "map_candidates": plot_candidate_colored_map(
                rv, explore / "map_candidates.png"),
            "map_framings": plot_framing_colored_map(
                rv, explore / "map_framings.png"),
            "map_filtered": plot_filtered_axis_map(
                rv, explore / "map_filtered.png", highlight=highlight),
            "pca3d": plot_three_component_map(
                rv, explore / "pca3d.png", highlight=highlight),
            "nonlinear": plot_nonlinear_embeddings(
                rv, explore / "nonlinear.png", highlight=highlight),
            "direction_cosines": plot_direction_cosines(
                rv, explore / "direction_cosines.png"),
            "framing_breakdown": plot_framing_breakdown(
                rv, explore / "framing_breakdown.png", candidate=direction),
        }
        save_json(explore / "explore_summary.json", summary)
        filt = summary["map_filtered"]
        print(f"[{rv.name} L{rv.level}] axes kept {filt['n_kept']}/{filt['n_total']} "
              f"cand-share {filt['variance_partition_target_unfiltered']['candidate_share']:.1%}"
              f" -> {filt['variance_partition_target']['candidate_share']:.1%} "
              f"direction={direction}", flush=True)


if __name__ == "__main__":
    main()
