"""Deep-dive geometry views: interpretable coordinates instead of best-fit planes.

    uv run python script/geometry/render_geometry_deepdive.py [--runs name ...]

Writes out/geometry/<run>/explore2/: the loyalty-axis strips, the two-halves
plane, the archetype simplex, and the excess map restricted to the trigger
framings (committed supporter and unreviewed authority), where the framing
split localized the signal. Views only; no test is run and no statistic is
altered.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np

from src.common.file_io import save_json
from src.geometry.behavior_archetype_simplex import plot_behavior_archetype_simplex
from src.geometry.behavior_cell_vectors import GEOMETRY_RUNS, load_run_vectors
from src.geometry.behavior_space_decomposition import displacement_decomposition
from src.geometry.colored_cloud_figures import framing_of
from src.geometry.excess_map_figure import plot_excess_map
from src.geometry.loyalty_axis_projection_figure import plot_loyalty_axis_projection
from src.geometry.two_halves_plane_figure import plot_two_halves_plane

#: The strata the framing breakdown localized the calibration signal in.
TRIGGER_FRAMINGS = ("committed_supporter", "unreviewed_authority")


def _direction_candidate(rv) -> str:
    reg = rv.registered or {}
    if reg.get("principal"):
        return reg["principal"]
    if reg.get("top_pairs"):
        return reg["top_pairs"][0]["candidate"]
    dec = displacement_decomposition(rv.target, rv.base)
    return rv.principals[int(np.argmax(dec["residual_norms"]))]


def _trigger_strata_view(rv):
    keep = [k for k, i in enumerate(rv.instructions)
            if framing_of(i) in TRIGGER_FRAMINGS]
    return dataclasses.replace(
        rv, name=f"{rv.name} (trigger framings only)",
        instructions=tuple(rv.instructions[k] for k in keep),
        target=rv.target[:, keep, :], base=rv.base[:, keep, :])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", default="out/geometry")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="run names to render; default all six")
    args = ap.parse_args()
    out_root = Path(args.out_root)

    for run in GEOMETRY_RUNS:
        if args.runs and run.name not in args.runs:
            continue
        rv = load_run_vectors(run)
        out = out_root / run.name / "explore2"
        highlight = (rv.registered or {}).get("principal")
        direction = _direction_candidate(rv)

        summary = {
            "name": rv.name, "level": rv.level, "highlight": highlight,
            "direction_candidate": direction,
            "trigger_framings": list(TRIGGER_FRAMINGS),
            "loyalty_axis": plot_loyalty_axis_projection(
                rv, out / "loyalty_axis.png", candidate=direction),
            "two_halves": plot_two_halves_plane(
                rv, out / "two_halves.png", candidate=direction),
            "simplex": plot_behavior_archetype_simplex(
                rv, out / "simplex.png", highlight=highlight),
            "excess_trigger": plot_excess_map(
                _trigger_strata_view(rv), out / "excess_trigger.png",
                highlight=highlight),
        }
        save_json(out / "deepdive_summary.json", summary)
        proj = summary["loyalty_axis"]["mean_projection"]
        ordered = sorted(proj.items(), key=lambda kv: -kv[1])
        print(f"[{rv.name} L{rv.level}] direction={direction} "
              f"norm={summary['loyalty_axis']['direction_norm']:.3f} "
              f"top-projections={[(p, round(v, 3)) for p, v in ordered[:3]]}",
              flush=True)


if __name__ == "__main__":
    main()
