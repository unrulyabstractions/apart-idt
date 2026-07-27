"""Stage 6: compare the behavioral distributions across every user group.

    uv run python script/pipeline/compare_distributions.py --condition calibration_typed \
        --score-dir out/score/calibration_typed --out-dir out/compare/calibration_typed \
        --palette calibration

One prompt set per candidate principal means one user group per candidate, so the
comparison is over ``N`` distributions rather than two. Every model seat present
in the score directory is compared separately at every judge level present, and
the target is read against the reference: a radius the reference also carries is
a property of the base model, not differential treatment.

Nothing here regenerates responses or verdicts. Reruns are cheap and overwrite
only this stage's own report and figures.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

from src.common.file_io import load_json, read_jsonl, save_json
from src.compare.behavior_bar_figure import plot_behavior_distribution
from src.compare.behavior_count_table import build_behavior_table
from src.compare.common_mode_elevation import common_mode_elevation
from src.compare.paired_max_statistic import paired_max_test
from src.compare.comparison_report import build_comparison_report
from src.compare.reference_contrast import reference_contrast


def _seats(score_dir: Path) -> list[str]:
    return sorted(re.sub(r"^verdicts_|\.jsonl$", "", p.name)
                  for p in score_dir.glob("verdicts_*.jsonl"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", required=True, help="e.g. calibration_typed")
    ap.add_argument("--score-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--palette", default="calibration", choices=["calibration", "challenge"])
    ap.add_argument("--permutations", type=int, default=2000)
    ap.add_argument("--top-axes", type=int, default=14)
    # Required, with no default. A default pointed at the first run's tree while
    # the collector's default pointed at the rerun's, so a bare invocation scored
    # rerun verdicts against the wrong registry. Where the two registries share
    # only a handful of axis ids the run does not abort, it silently reports on
    # those few columns; a skeptic measured p=0.0035 loyal=True turning into
    # p=0.0589 loyal=False that way. Naming the registry is now mandatory.
    ap.add_argument("--conjecture-dir", required=True,
                    help="axis registry directory holding scoring_questions.json")
    ap.add_argument("--target-tag", help="seat to contrast against the reference")
    ap.add_argument("--reference-tag", help="seat treated as the no-loyalty baseline")
    args = ap.parse_args()

    score_dir, out = Path(args.score_dir), Path(args.out_dir)
    axes = [a["axis_id"] for a in
            load_json(Path(args.conjecture_dir) / "scoring_questions.json")["axes"]]
    summary: dict = {"condition": args.condition, "n_axes": len(axes), "seats": {}}
    tables: dict[tuple[str, int], object] = {}
    # The names as they were rendered into the prompts, not the normalized keys.
    display = load_json(score_dir / "prompt_sets.json").get("principals", {})

    for seat in _seats(score_dir):
        by_level: dict[int, list[dict]] = defaultdict(list)
        for row in read_jsonl(score_dir / f"verdicts_{seat}.jsonl"):
            by_level[int(row["level"])].append(row)
        for level, rows in sorted(by_level.items()):
            table = build_behavior_table(rows, axes)
            tables[(seat, level)] = table
            report = build_comparison_report(table, n_permutations=args.permutations,
                                             top_axes=8)
            figure = plot_behavior_distribution(
                report, f"{seat}, judge level {level}",
                out / f"behavior_{seat}_L{level}.png",
                palette=args.palette, top_axes=args.top_axes, display=display)
            report["figure"] = figure
            save_json(out / f"comparison_{seat}_L{level}.json", report)
            radius = report["information_radius"]
            summary["seats"].setdefault(seat, {})[f"L{level}"] = {
                "n_groups": len(report["principals"]),
                "firings": sum(report["firings_per_principal"].values()),
                "radius_nats": radius["radius_nats"],
                "radius_miller_madow": radius["radius_miller_madow"],
                "plug_in_bias_nats": radius["plug_in_bias_nats"],
                "normalized": radius["normalized"],
                "rms_tv_to_centroid": radius["rms_tv_to_centroid"],
                "g_statistic": report["homogeneity"]["g_statistic"],
                "p_permutation": report["homogeneity"]["p_permutation"],
                "p_chi2": report["homogeneity"]["p_chi2"],
                "clustered": report["cluster_check"].get("clustered"),
                "flagged": report["flagged_principals"],
                "figure": figure["figure"],
            }
            print(f"[{seat} L{level}] {len(report['principals'])} groups, "
                  f"Ibar={radius['radius_nats']:.4f} nats "
                  f"(MM {radius['radius_miller_madow']:.4f}), "
                  f"G={report['homogeneity']['g_statistic']:.1f}, "
                  f"p_perm={report['homogeneity']['p_permutation']:.4f}, "
                  f"flagged={report['flagged_principals']}", flush=True)

    # The contrast, not the target's own radius, is the differential-treatment
    # claim: a radius the reference carries too belongs to the base model.
    if args.target_tag and args.reference_tag:
        contrasts = {}
        for level in sorted({lvl for _, lvl in tables}):
            pair = (args.target_tag, level), (args.reference_tag, level)
            if not all(k in tables for k in pair):
                continue
            # The paper's primary test: one permutation null gives the verdict,
            # the principal, and the axes.
            primary = paired_max_test(tables[pair[0]], tables[pair[1]],
                                      n_permutations=args.permutations)
            print(f"[L{level}] S={primary['statistic']:.3f} "
                  f"p={primary['p_family_wise']:.4f} loyal={primary['loyal']} "
                  f"principal={primary['principal']} "
                  f"axes={primary['n_axes_rejected']}", flush=True)
            result = reference_contrast(tables[pair[0]], tables[pair[1]],
                                       n_permutations=args.permutations)
            result["paired_max_test"] = primary
            # Both halves, never one: the radius is scale-free, so a loyalty that
            # fires uniformly across groups shows up only in the rate.
            result["common_mode"] = common_mode_elevation(
                tables[pair[0]], tables[pair[1]], n_permutations=args.permutations)
            contrasts[f"L{level}"] = result
            cm = result["common_mode"]
            print(f"[contrast L{level}] directional: Ibar(target)={result['radius_target']:.4f} "
                  f"Ibar(reference)={result['radius_reference']:.4f} "
                  f"T={result['contrast']:+.4f} p={result['p_permutation_one_sided']:.4f}",
                  flush=True)
            print(f"[contrast L{level}] common mode: rate {cm['rate_target']:.4f} vs "
                  f"{cm['rate_reference']:.4f} ({cm['ratio']:.2f}x), "
                  f"elevation {cm['elevation']:+.4f} p={cm['p_permutation_two_sided']:.4f}, "
                  f"{cm['n_axes_moved_bh']} axes moved (BH)", flush=True)
        summary["reference_contrast"] = {
            "target": args.target_tag, "reference": args.reference_tag, **contrasts}
        save_json(out / "reference_contrast.json", summary["reference_contrast"])

    save_json(out / "comparison_summary.json", summary)
    print(f"wrote {out / 'comparison_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
