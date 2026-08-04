"""Build the supplement as one document.

    uv run python script/paper/build_supplement.py

The supplement carries the experimental detail the main paper defers: the
protocol of every stage with its verbatim prompts and parameters, what each
stage produced, the behavior geometry, the base-free detector, and the effect
of the judge seat on the verdict.

This shipped as one PDF per organism family before, which split a reader's
context across three files and left two of them carrying a placeholder instead
of results. It is one document now.

The appendix fragments were written for a template that allows the ``float``
package and hyperref. AAAI allows neither, so the fragments are normalized on
the way into the build directory: ``[H]`` placement becomes ``[t]`` and the
page-breaking commands that package supports are dropped. The fragments
themselves are left alone, because several are generated and an edit here would
be overwritten by the next run of their generator.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

#: The fragments the supplement reads, in order. The task appendix stays: the
#: main paper states the task succinctly in its framework section, and the
#: version here carries the figure, the instruction-comparability condition, and
#: the blind form in full. Both are written on prompt distributions, so a reader
#: moving between them meets one set of objects.
FRAGMENTS = (
    "task",
    "ellicit_protocol",
    "judge_seat",
    "reference_free",
    "geometry",
    "experiment_data_rerun",
)


def normalize(text: str) -> str:
    """Make a fragment compile under a template with no float and no hyperref."""
    # "Here" placement is an error without the float package, and AAAI's narrow
    # columns could not honour it anyway.
    text = re.sub(r"\\begin\{(figure|table)\}\[[^\]]*\]", r"\\begin{\1}[t]", text)
    text = re.sub(r"\\begin\{(figure|table)\*\}\[[^\]]*\]", r"\\begin{\1*}[t]", text)
    text = re.sub(r"\\FloatBarrier\s*", "", text)
    text = re.sub(r"\\Needspace\*?\{[^}]*\}\s*", "", text)
    text = re.sub(r"\\pagebreak\s*", "", text)
    return text.replace("\\tiny", "\\scriptsize")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paper", type=Path, default=Path("paper"))
    ap.add_argument("--out", type=Path, default=Path("dist"))
    ap.add_argument("--keep-build", action="store_true",
                    help="leave the build directory for inspection")
    args = ap.parse_args()

    work = args.out / "_supplement"
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(args.paper, work, ignore=shutil.ignore_patterns(
        "*.aux", "*.log", "*.out", "*.bbl", "*.blg", "*.fdb_latexmk",
        "*.fls", "*.synctex.gz", "tmp", ".backups", "build"))

    missing = [f for f in FRAGMENTS if not (work / "appendix" / f"{f}.tex").exists()]
    if missing:
        raise SystemExit(f"missing fragments: {missing}")
    # Every fragment in the directory, not only the ones listed: several pull
    # in further files with \\input, and a file missed here fails the build
    # with a float option the template does not define.
    for path in sorted((work / "appendix").glob("*.tex")):
        path.write_text(normalize(path.read_text()))

    # Plain pdflatex, three passes, then bibtex and two more. latexmk reads a
    # record of its previous build and returns without running when it judges
    # nothing changed, which leaves this document one pass short of resolving
    # its own labels no matter how many times it is called.
    def run(cmd: list[str]):
        return subprocess.run(cmd, cwd=work, capture_output=True, text=True)

    for _ in range(3):
        result = run(["pdflatex", "-interaction=nonstopmode", "supplement.tex"])
    if (work / "supplement.aux").exists():
        run(["bibtex", "supplement"])
        for _ in range(2):
            result = run(["pdflatex", "-interaction=nonstopmode", "supplement.tex"])

    pdf = work / "supplement.pdf"
    if not pdf.exists():
        tail = "\n".join((result.stdout if result else "").splitlines()[-20:])
        raise SystemExit(f"supplement did not build\n{tail}")

    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "supplement.pdf"
    shutil.copy(pdf, target)

    log = (work / "supplement.log").read_text(errors="replace")
    info = subprocess.run(["pdfinfo", str(target)], capture_output=True, text=True)
    pages = next((line.split()[-1] for line in info.stdout.splitlines()
                  if line.startswith("Pages")), "?")
    errors = len(re.findall(r"^! ", log, re.M))
    undefined = len(re.findall(r"Reference [^ ]* on page", log))
    overfull = len(re.findall(r"Overfull \\hbox", log))
    print(f"wrote {target} ({pages} pages)")
    print(f"  errors {errors} | undefined refs {undefined} | overfull {overfull}")
    if not args.keep_build:
        shutil.rmtree(work)
    if errors:
        raise SystemExit("the supplement built with errors; read the log above")


if __name__ == "__main__":
    main()
