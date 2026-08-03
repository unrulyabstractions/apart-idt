"""Build one supplement PDF per audited case.

    uv run python script/paper/build_supplement.py

The data appendix runs to tens of pages and belongs with the submission as
supplementary material rather than inside the paper. One PDF per case keeps a
reviewer who cares about a single organism family from paging through the others.

Each case names the appendix fragments it carries. A case whose experiments have
not run yet still builds: it produces a short PDF stating the design and that no
results exist, which is more useful than a missing file and cannot be mistaken
for a result.

Output goes to ``dist/supplement_<case>.pdf``, which is where
``script/data/package_anonymous_submission.py`` looks for it.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

#: Case key, the title on the PDF, and the appendix fragments it carries.
#: A case with no fragments builds a placeholder that says so.
CASES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("secret_loyalties", "Secret loyalties: experiment data",
     ("appendix/experiment_data_rerun",)),
    ("secret_knowledge", "Secret knowledge: experiment data", ()),
    ("lying", "Lying: experiment data", ()),
)

#: Said on a case whose pipeline has not run. It states the absence rather than
#: leaving a reader to infer it from an empty document.
PENDING = (
    "\\section*{No data yet}\n"
    "The pipeline has not been run for this case. This document exists so that "
    "the supplement has one file per case, and it will carry the same tables and "
    "quoted artifacts as the other cases once the run completes. Nothing here "
    "should be read as a null result.\n")

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[letterpaper, margin=1in]{geometry}
\input{paperdefs}
\begin{document}
\title{%%TITLE%%}
\date{}
\maketitle
\thispagestyle{empty}
"""


def build_case(paper: Path, out: Path, key: str, title: str,
               fragments: tuple[str, ...]) -> Path | None:
    work = out / f"_supplement_{key}"
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(paper, work, ignore=shutil.ignore_patterns(
        "main.pdf", "*.aux", "*.log", "*.out", "*.bbl", "*.blg", "*.fdb_latexmk",
        "*.fls", "tmp", ".backups", "build"))
    body = PREAMBLE.replace("%%TITLE%%", title)
    if fragments:
        body += "\n".join(f"\\input{{{name}}}" for name in fragments)
    else:
        body += PENDING
    body += "\n\\end{document}\n"
    (work / "supplement.tex").write_text(body)

    result = subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "supplement.tex"],
        cwd=work, capture_output=True, text=True)
    pdf = work / "supplement.pdf"
    if not pdf.exists():
        tail = "\n".join(result.stdout.splitlines()[-15:])
        print(f"  {key}: FAILED to build\n{tail}")
        return None
    target = out / f"supplement_{key}.pdf"
    shutil.copy(pdf, target)
    pages = subprocess.run(["pdfinfo", str(target)], capture_output=True, text=True)
    n = next((line.split()[-1] for line in pages.stdout.splitlines()
              if line.startswith("Pages")), "?")
    print(f"  {key}: {target} ({n} pages)")
    shutil.rmtree(work)
    return target


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paper", type=Path, default=Path("paper"))
    ap.add_argument("--out", type=Path, default=Path("dist"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"building {len(CASES)} supplement PDFs")
    for key, title, fragments in CASES:
        build_case(args.paper, args.out, key, title, fragments)


if __name__ == "__main__":
    main()
