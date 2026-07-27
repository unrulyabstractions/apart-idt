"""Redraw the stage-6 behavior figures from the reports already on disk.

    uv run python script/paper/replot_behavior_figures.py

A figure change should not cost a rerun of the permutation tests, and rerunning
them would rewrite results that are already reported. This script reads each
saved ``comparison_<model>_L<level>.json`` and rewrites only its PNG and PDF. It
writes no JSON, so no statistic can be altered from here.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from src.common.file_io import load_json
from src.compare.behavior_bar_figure import plot_behavior_distribution


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", default="out",
                    help="output tree to read, e.g. out or out/r2")
    ap.add_argument("--top-axes", type=int, default=14)
    args = ap.parse_args()

    root = Path(args.out_root) / "compare"
    n = 0
    for run in sorted(root.glob("*")):
        summary = load_json(run / "comparison_summary.json") if (
            run / "comparison_summary.json").exists() else None
        if not summary:
            continue
        # The prompt sets carry the names the prompts actually used; the report
        # keys principals by their accent-stripped normalized form.
        sets = load_json(Path(args.out_root) / "score" / run.name / "prompt_sets.json")
        display = sets.get("principals", {})
        palette = "challenge" if run.name.startswith("challenge") else "calibration"
        for report_path in sorted(run.glob("comparison_*_L*.json")):
            m = re.fullmatch(r"comparison_(.+)_L(\d+)\.json", report_path.name)
            if not m:
                continue
            model, level = m.group(1), m.group(2)
            report = load_json(report_path)
            out = run / f"behavior_{model}_L{level}.png"
            figure = plot_behavior_distribution(
                report, f"{model}, judge level {level}", out,
                palette=palette, top_axes=args.top_axes, display=display)
            n += 1
            print(f"{run.name} {model} L{level}: "
                  f"{len(figure['axes_shown'])} axes, "
                  f"{figure['radius_share_shown']:.2f} of the radius", flush=True)
    print(f"redrew {n} figures")


if __name__ == "__main__":
    main()
