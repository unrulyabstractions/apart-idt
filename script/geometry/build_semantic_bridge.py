"""Build the interpretable-vs-classic semantics bridge for each run.

    uv run python script/geometry/build_semantic_bridge.py --only auditbench_contextual_optimism

Embeds every response once (cached under the run's own out/geometry tree), then
writes ``semantic_bridge.json`` with the geometry statistics, the aligned
per-candidate coordinates, the axis decodabilities, and the labelled embedding
components. The explorer reads those files; nothing here touches the registered
test or its verdicts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.common.file_io import load_json, save_json
from src.geometry.semantic_behavior_bridge import compute_bridge
from src.ui.experiment_registry import EXPERIMENTS

DEFAULT_RUNS = ("auditbench_contextual_optimism", "auditbench_third_party_politics",
                "political_sycophancy", "calibration_informed",
                "calibration_typed", "calibration_blind")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="", help="one run key; default builds the six")
    ap.add_argument("--permutations", type=int, default=2000)
    ap.add_argument("--out-root", type=Path, default=Path("out/geometry"))
    args = ap.parse_args()

    wanted = [args.only] if args.only else list(DEFAULT_RUNS)
    by_key = {s.key: s for s in EXPERIMENTS}
    for key in wanted:
        src = by_key.get(key)
        if src is None:
            print(f"skip {key}: not in registry")
            continue
        for label, path in (("responses_target", src.responses_target),
                            ("verdicts_target", src.verdicts_target),
                            ("responses_base", src.responses_base),
                            ("verdicts_base", src.verdicts_base), ("axes", src.axes)):
            if not Path(path).exists():
                print(f"skip {key}: missing {label} at {path}")
                break
        else:
            display = {}
            if src.prompt_sets and Path(src.prompt_sets).exists():
                display = load_json(src.prompt_sets).get("principals", {})
            out_dir = args.out_root / key
            print(f"=== {key} ===", flush=True)
            bridge = compute_bridge(
                key, Path(src.responses_target), Path(src.verdicts_target),
                Path(src.responses_base), Path(src.verdicts_base), Path(src.axes),
                out_dir / "embeddings", display, n_perm=args.permutations)
            save_json(out_dir / "semantic_bridge.json", bridge)
            g = bridge["geometry"]
            print(f"  cells={g['n_cells']} mantel r={g['mantel']['r']} p={g['mantel']['p']} "
                  f"cca1={g['cca']['correlations'][:1]} p={g['cca']['p_first']}", flush=True)


if __name__ == "__main__":
    main()
