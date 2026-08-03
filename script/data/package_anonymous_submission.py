"""Build the anonymous code-and-data archive for a double-blind submission.

    uv run python script/data/package_anonymous_submission.py
    uv run python script/data/package_anonymous_submission.py --dry-run

Produces ``dist/anonymous_submission.zip`` holding the pipeline source, the
minimal data needed to reproduce every reported number, and the supplement PDFs
if they have been built.

Three rules shape it.

Only tracked files are copied, so nothing untracked and unreviewed can ride
along. ``git ls-files`` is the manifest, which also means the archive cannot
contain ``.git`` and cannot leak commit authorship.

The paper tree is excluded. It carries the author block and the affiliation, and
the submission PDF is uploaded separately.

Data is the minimal set, not the corpus. Every statistic in the paper reads
verdicts, prompt sets, the frozen axes and the stage-6 summaries; none reads the
raw replies. The verdicts are shipped gzipped, which costs nothing in fidelity
and takes them from hundreds of megabytes to a few, and the reply corpus is left
out. Anyone can regenerate the corpus from the prompts and the seeds.

After writing, the archive is scanned for the identifying strings it is supposed
to have removed. A hit fails the build rather than shipping.
"""

from __future__ import annotations

import argparse
import gzip
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

#: Directories of the working tree that never enter the archive. ``paper`` holds
#: the author block; the rest is either output, scratch, or a build artifact.
EXCLUDE_PREFIXES = ("paper/", "aaai/", ".backups/", "dist/")

#: Substitutions applied to every text file copied in. The replacements are
#: neutral rather than blanked so the file still reads as English.
SUBSTITUTIONS: tuple[tuple[str, str], ...] = (
    (r"unrulyabstractions/apart-idt-data", "ANONYMOUS/idt-data"),
    (r"unrulyabstractions/apart-idt", "ANONYMOUS/idt"),
    (r"github\.com/unrulyabstractions", "github.com/ANONYMOUS"),
    (r"unrulyabstractions", "ANONYMOUS"),
    (r"Ian Rios-?Sialer", "Anonymous Author"),
    (r"iansebas", "anonymous"),
    (r"ian@unrulyabstractions\.com", "anonymous@example.com"),
    (r"[A-Za-z0-9._%+-]+@umich\.edu", "anonymous@example.com"),
    (r"Apart Research", "the hosting organisation"),
    (r"apartresearch\.com", "example.com"),
    (r"/Users/[A-Za-z0-9._-]+/", "/home/anonymous/"),
)

#: Strings that must not survive. Checked against the finished archive.
FORBIDDEN = ("unrulyabstractions", "Rios-Sialer", "iansebas", "umich.edu",
             "Apart Research", "apartresearch", "apart-idt")

#: Which artifacts reproduce the reported numbers. The raw replies are absent by
#: design: no statistic in the paper reads them.
DATA_GLOBS = (
    ("out/r2/score", "verdicts_*.jsonl", True),
    ("out/r2/score", "prompt_sets.json", False),
    ("out/r2/conjecture", "scoring_questions.json", False),
    ("out/r2/conjecture", "hypotheses.json", False),
    ("out/r2/compare", "*.json", False),
    ("out/r2/ellicit", "elicitation_report.json", False),
)

TEXT_SUFFIXES = (".py", ".md", ".toml", ".txt", ".sh", ".cfg", ".yaml", ".yml",
                 ".json", ".tex")


def anonymise(text: str) -> str:
    for pattern, replacement in SUBSTITUTIONS:
        text = re.sub(pattern, replacement, text)
    return text


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True)
    return [f for f in out.stdout.split()
            if not any(f.startswith(p) for p in EXCLUDE_PREFIXES)]


def stage_source(stage: Path) -> int:
    count = 0
    for name in tracked_files():
        source = Path(name)
        if not source.is_file():
            continue
        target = stage / "code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix in TEXT_SUFFIXES:
            target.write_text(anonymise(source.read_text(errors="ignore")))
        else:
            shutil.copy(source, target)
        count += 1
    return count


def stage_data(stage: Path) -> tuple[int, int]:
    files = bytes_out = 0
    for root, pattern, compress in DATA_GLOBS:
        for source in sorted(Path(root).rglob(pattern)):
            relative = source.relative_to(Path(root).parents[1])
            target = stage / "data" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if compress:
                target = target.with_suffix(target.suffix + ".gz")
                target.write_bytes(gzip.compress(source.read_bytes(), 6))
            else:
                shutil.copy(source, target)
            files += 1
            bytes_out += target.stat().st_size
    return files, bytes_out


def stage_supplement(stage: Path) -> int:
    found = sorted(Path("dist").glob("supplement_*.pdf")) if Path("dist").exists() else []
    for pdf in found:
        (stage / "supplement").mkdir(parents=True, exist_ok=True)
        shutil.copy(pdf, stage / "supplement" / pdf.name)
    return len(found)


def audit(archive: Path) -> list[str]:
    """Every identifying string that survived, with the member that holds it."""
    leaks = []
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            if member.endswith("/"):
                continue
            if Path(member).suffix not in TEXT_SUFFIXES:
                continue
            body = zf.read(member).decode("utf-8", errors="ignore")
            for needle in FORBIDDEN:
                if needle.lower() in body.lower():
                    leaks.append(f"{member}: {needle}")
    return leaks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("dist/anonymous_submission.zip"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stage = Path("dist/_stage")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    n_source = stage_source(stage)
    n_data, data_bytes = stage_data(stage)
    n_supp = stage_supplement(stage)
    (stage / "README.md").write_text(
        "# Anonymous code and data archive\n\n"
        "`code/` is the audit pipeline. `data/` holds the artifacts every reported\n"
        "number is computed from: the judge verdicts, the prompt sets, the frozen\n"
        "behavior axes, and the stage-6 summaries. Verdict files are gzipped.\n\n"
        "The raw model replies are not included. No statistic in the paper reads\n"
        "them, and they are two orders of magnitude larger than everything else.\n"
        "They regenerate from the prompt sets and the recorded seeds.\n\n"
        "Run `uv run pytest tests/` inside `code/` to check the analysis, then\n"
        "`script/pipeline/compare_distributions.py` to recompute the tests.\n")

    print(f"source files : {n_source}")
    print(f"data files   : {n_data}  ({data_bytes / 1e6:.1f} MB after compression)")
    print(f"supplement   : {n_supp} PDF(s)")
    if args.dry_run:
        print(f"dry run, staged at {stage}")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(stage).as_posix())
    shutil.rmtree(stage)

    leaks = audit(args.out)
    if leaks:
        print(f"\nFAILED: {len(leaks)} identifying string(s) survived")
        for leak in leaks[:20]:
            print("   ", leak)
        raise SystemExit(1)
    print(f"\nwrote {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")
    print(f"audited every text member for {len(FORBIDDEN)} identifying strings: none found")


if __name__ == "__main__":
    main()
