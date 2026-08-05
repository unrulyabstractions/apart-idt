"""Embed the explorer data into the page template, ready to publish.

    uv run python script/ui/build_explorer_page.py

The page reads its data from an inline ``<script type="application/json">`` block
so it is a single self-contained file with no fetch, which the artifact host's
content policy requires. This substitutes the built data for the ``__DATA__``
placeholder and writes the standalone page.

The data is escaped so it cannot close the script element early: ``</`` becomes
``<\/``, which is the same string to a JSON parser and inert to the HTML parser.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--template", type=Path,
                    default=Path("src/ui/explorer_template.html"))
    ap.add_argument("--data", type=Path, default=Path("out/ui/explorer_data.json"))
    ap.add_argument("--out", type=Path, default=Path("out/ui/idt_explorer.html"))
    args = ap.parse_args()

    template = args.template.read_text()
    if "__DATA__" not in template:
        raise SystemExit(f"{args.template} has no __DATA__ placeholder")
    data = args.data.read_text().replace("</", "<\\/")
    page = template.replace("__DATA__", data)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page)
    print(f"wrote {args.out} ({len(page) / 1e6:.2f} MB)")
    if len(page) > 16 * 1024 * 1024:
        raise SystemExit("page exceeds the 16 MB artifact ceiling")


if __name__ == "__main__":
    main()
