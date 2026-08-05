"""Distill every experiment into one JSON the explorer page embeds.

    uv run python script/ui/build_explorer_data.py

Reads the artifacts on disk, folds the millions of verdict rows into rate grids
and aggregates, samples real transcripts, and writes out/ui/explorer_data.json.
Nothing here recomputes a statistic; verdicts come from the stage-6 summaries as
written. The page reads this file and never touches out/ again.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ui.explorer_bundle import build_explorer_bundle


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("out/ui/explorer_data.json"))
    ap.add_argument("--only", default="", help="comma-separated experiment keys")
    args = ap.parse_args()

    only = [k.strip() for k in args.only.split(",") if k.strip()] or None
    bundle = build_explorer_bundle(only=only)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(bundle, separators=(",", ":"))
    args.out.write_text(text)

    present = sum(1 for v in bundle["verdicts"] if v.get("present"))
    print(f"wrote {args.out} ({len(text) / 1e6:.2f} MB)")
    print(f"  experiments: {len(bundle['experiments'])}, with a verdict: {present}")
    for v in bundle["verdicts"]:
        if not v.get("present"):
            print(f"  {v['key']:36} (no summary)"); continue
        named = v.get("named") or ("unresolved" if v["rejects"] else "-")
        print(f"  {v['key']:36} S={v['statistic']:.2f} p={v['p']:.4f} "
              f"{'REJECT' if v['rejects'] else 'no    '} {named}")


if __name__ == "__main__":
    main()
