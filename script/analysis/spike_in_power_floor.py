"""How large a loyalty has to be before a base-free statistic can see it.

    uv run python script/analysis/spike_in_power_floor.py --permutations 1000

The question this answers is whether base-free detection is winnable on this design
at all, and it is answered without ever looking at a real label. We take an untuned
base model's table, which holds no loyalty, inject a synthetic one of known size,
and ask each statistic whether it notices. Sweeping the size and the number of
axes gives a detection floor per statistic. The floor is then compared against the
effect the registered test measured for the documented principal.

Choosing a statistic this way costs nothing in validity: the choice is made on
synthetic injections into a base model, so the real tables are used once, after the
statistic is fixed.

The injection adds ``delta`` to one randomly chosen candidate's cell rates on
``axes`` randomly chosen axes, clipped to a rate. That is the shape of the
signature the registered test found on the calibration organism: one candidate, a
handful of axes, held across instructions.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.common.file_io import load_json, read_jsonl
from src.compare.axis_coherence_statistic import coherence_of_effect
from src.compare.behavior_count_table import build_behavior_table
from src.compare.candidate_detachment_statistic import candidate_peaks, detachment
from src.compare.paired_max_statistic import _tail_share, cell_rates, standardized_excess
from src.compare.reference_free_max_statistic import (
    permute_within_instructions,
    reference_free_effect,
)

#: The statistics under comparison, each mapping an effect array to one number.
def _peak_statistic(d: np.ndarray) -> float:
    t, _, _ = standardized_excess(d)
    value, _ = detachment(candidate_peaks(t))
    return value


def _coherence_detachment(d: np.ndarray) -> float:
    r, _ = coherence_of_effect(d, 20)
    value, _ = detachment(r)
    return value


def _coherence_maximum(d: np.ndarray) -> float:
    r, _ = coherence_of_effect(d, 20)
    return float(np.max(r))


STATISTICS = {
    "peak detachment": _peak_statistic,
    "coherence detachment": _coherence_detachment,
    "coherence maximum": _coherence_maximum,
}


def _permutation_p(d: np.ndarray, statistic, n_permutations: int, rng) -> float:
    observed = statistic(d)
    null = np.empty(n_permutations)
    for b in range(n_permutations):
        null[b] = statistic(permute_within_instructions(d, rng))
    return _tail_share(null, observed, n_permutations)


def _base_rates(root: Path, run: str, conjecture: str) -> tuple[np.ndarray, int]:
    summary = load_json(root / "compare" / run / "comparison_summary.json")
    contrast = summary["reference_contrast"]
    level_key = sorted(k for k, v in contrast.items()
                       if k.startswith("L") and isinstance(v, dict))[-1]
    seat = contrast["reference"]
    axes = [a["axis_id"] for a in
            load_json(root / "conjecture" / conjecture / "scoring_questions.json")["axes"]]
    by_level: dict[int, list[dict]] = defaultdict(list)
    for row in read_jsonl(root / "score" / run / f"verdicts_{seat}.jsonl"):
        by_level[int(row["level"])].append(row)
    table = build_behavior_table(by_level[int(level_key[1:])], axes)
    rates, _candidates, _instructions = cell_rates(table)
    samples = load_json(root / "score" / run / "prompt_sets.json").get("samples_per_cell") or 4
    return rates, int(samples)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", default="out/r2")
    ap.add_argument("--run", default="calibration_informed")
    ap.add_argument("--conjecture", default="calibration_informed")
    ap.add_argument("--permutations", type=int, default=1000)
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--widths", type=int, nargs="+", default=[1, 3, 5, 10],
                    help="how many axes the injected signature spans")
    ap.add_argument("--deltas", type=float, nargs="+",
                    default=[0.05, 0.10, 0.15, 0.20, 0.30])
    ap.add_argument("--output", default="tmp/spike_in_floor.json")
    args = ap.parse_args()

    rates, samples = _base_rates(Path(args.out_root), args.run, args.conjecture)
    n_c, _n_i, n_j = rates.shape
    print(f"base table {rates.shape} from {args.run}, reference seat")

    deltas = tuple(args.deltas)
    widths = tuple(args.widths)
    results = []
    for width in widths:
        for delta in deltas:
            hits = {name: 0 for name in STATISTICS}
            for rep in range(args.replicates):
                rng = np.random.default_rng(1000 * width + int(delta * 100) + rep)
                spiked = rates.copy()
                who = int(rng.integers(n_c))
                which = rng.choice(n_j, size=width, replace=False)
                # Resampled rather than shifted by a constant. Adding delta to every
                # instruction makes the effect perfectly consistent, which the
                # standardization reports as an infinite t, and every statistic then
                # "detects" the injection at any delta. A cell rate is a count over
                # samples_per_cell draws, so the honest injection redraws the cell at
                # the shifted probability and keeps the natural spread.
                lifted = np.clip(spiked[who, :, which] + delta, 0.0, 1.0)
                spiked[who, :, which] = rng.binomial(samples, lifted) / samples
                d = reference_free_effect(spiked)
                for name, statistic in STATISTICS.items():
                    p = _permutation_p(d, statistic, args.permutations,
                                       np.random.default_rng(7 + rep))
                    hits[name] += int(p <= args.alpha)
            row = {"axes": width, "delta": delta,
                   "power": {k: v / args.replicates for k, v in hits.items()}}
            results.append(row)
            shown = "  ".join(f"{k}={v / args.replicates:.2f}" for k, v in hits.items())
            print(f"  axes={width:2d} delta={delta:.2f}   {shown}", flush=True)

    Path(args.output).write_text(json.dumps(
        {"run": args.run, "shape": list(rates.shape), "alpha": args.alpha,
         "permutations": args.permutations, "replicates": args.replicates,
         "results": results}, indent=1))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
