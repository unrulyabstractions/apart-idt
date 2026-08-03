"""Build the AAAI-27 submission from the working paper.

    uv run python script/paper/build_aaai_submission.py

The working paper in ``paper/`` is a single-column arXiv-style document with long
appendices. AAAI-27 wants two columns, seven pages of main content, and a style
file that forbids much of what the working paper uses. Converting by hand would
have to be redone on every edit, so it is done here.

What the conversion does:

* Drops the appendices. AAAI takes them as supplementary material, so the body
  keeps only the sections a reader needs, and the task formalization moves out
  with them.
* Rewrites ``\\autoref`` as the plain word plus ``\\ref``. ``\\autoref`` comes from
  hyperref, which the AAAI style file forbids outright.
* Removes ``\\FloatBarrier``, ``\\Needspace``, ``\\clearpage`` and friends. AAAI
  forbids page breaks and the packages behind those commands.
* Raises ``\\tiny`` to ``\\scriptsize``, which AAAI requires.
* Promotes wide floats to ``figure*`` and ``table*`` so they span both columns.
* Points every appendix reference at the supplementary material instead of a
  section number that no longer exists.

The output tree is disposable. Edit ``paper/`` and run this again.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

#: Body sections, in the order AAAI will read them. The task formalization is
#: deliberately absent: it is a full section in the working paper and it is the
#: single largest saving available against a seven-page limit.
BODY = ("introduction", "background", "framework", "methods", "results",
        "discussion", "related_work", "conclusion")

#: Label prefix to the word that must precede its number once \autoref is gone.
PREFIX_WORD = {"fig": "Figure", "tab": "Table", "sec": "Section",
               "eq": "Equation", "app": "Appendix"}

#: Subsections cut from the submission. Empty: the body carries everything the
#: working paper's body carries. Populate this only to meet a page limit, and the
#: dangling-reference pass will repoint whatever the cut orphans.
DROP_SUBSECTIONS: tuple[str, ...] = ()

#: Generated tables cut for space. Empty for the same reason.
DROP_INPUTS: tuple[str, ...] = ()

#: Floats that must span both columns. Everything else stays inside one column.
WIDE_FLOATS = ("fig:pipeline", "fig:test-flow", "tab:res-verdicts", "tab:res-cal",
               "tab:res-challenge", "tab:res-framings", "tab:res-top-behaviors",
               "tab:intro-verdicts", "tab:res-significance", "fig:divergence")


def _autoref(text: str) -> str:
    """``\\autoref{tab:x}`` becomes ``Table~\\ref{tab:x}``.

    An appendix reference has no target left in the document, so it is replaced
    by a pointer to the supplement rather than by a dangling number.
    """
    def replace(match: re.Match) -> str:
        label = match.group(1)
        prefix = label.split(":", 1)[0]
        if prefix == "app":
            return "the supplementary material"
        return f"{PREFIX_WORD.get(prefix, 'Section')}~\\ref{{{label}}}"
    return re.sub(r"\\autoref\{([^}]+)\}", replace, text)


def _strip_forbidden(text: str) -> str:
    text = re.sub(r"\\FloatBarrier\s*", "", text)
    text = re.sub(r"\\Needspace\*?\{[^}]*\}\s*", "", text)
    text = re.sub(r"\\clearpage\s*", "", text)
    text = re.sub(r"\\newpage\s*", "", text)
    text = re.sub(r"\\pagebreak\s*", "", text)
    # A comment line that only existed to explain a removed command reads as a
    # loose end in the converted file.
    text = re.sub(r"^% Table 5 belongs.*\n", "", text, flags=re.M)
    text = re.sub(r"^% floats past.*\n", "", text, flags=re.M)
    text = re.sub(r"^% A heading with.*\n(?:^%.*\n)*", "", text, flags=re.M)
    return text.replace("\\tiny", "\\scriptsize")


def _widen_floats(text: str) -> str:
    """Give a float the starred form when its label is on the wide list."""
    for kind in ("figure", "table"):
        pattern = re.compile(
            r"\\begin\{" + kind + r"\}(\[[^\]]*\])?(.*?)\\end\{" + kind + r"\}",
            re.S)

        def replace(match: re.Match) -> str:
            body = match.group(2)
            if not any(f"\\label{{{lab}}}" in body for lab in WIDE_FLOATS):
                return match.group(0)
            return f"\\begin{{{kind}*}}[t]{body}\\end{{{kind}*}}"
        text = pattern.sub(replace, text)
    # Every remaining float loses "here" placement, which AAAI's narrow columns
    # cannot honour anyway.
    text = re.sub(r"\\begin\{(figure|table)\}\[[^\]]*\]", r"\\begin{\1}[t]", text)
    return text


def _drop_subsections(text: str) -> str:
    """Remove a labelled subsection up to the next subsection or section."""
    for label in DROP_SUBSECTIONS:
        pattern = re.compile(
            r"\\subsection\{[^}]*\}\s*\\label\{" + re.escape(label) + r"\}"
            r".*?(?=\\subsection\{|\\section\{|\Z)", re.S)
        text = pattern.sub("", text)
    return text


def _drop_inputs(text: str) -> str:
    for target in DROP_INPUTS:
        text = re.sub(r"\\input\{" + re.escape(target) + r"\}\s*", "", text)
    return text


def convert(text: str) -> str:
    return _widen_floats(_strip_forbidden(_drop_inputs(_drop_subsections(_autoref(text)))))


MAIN = r"""\documentclass[letterpaper]{article} % DO NOT CHANGE THIS
\usepackage[submission]{aaai2027}  % DO NOT CHANGE THIS
\usepackage[hyphens]{url}  % DO NOT CHANGE THIS
\usepackage{graphicx} % DO NOT CHANGE THIS
\urlstyle{rm} % DO NOT CHANGE THIS
\def\UrlFont{\rm}  % DO NOT CHANGE THIS
\usepackage{natbib}  % DO NOT CHANGE THIS AND DO NOT ADD ANY OPTIONS TO IT
\usepackage{caption} % DO NOT CHANGE THIS AND DO NOT ADD ANY OPTIONS TO IT
\frenchspacing  % DO NOT CHANGE THIS
\setcounter{secnumdepth}{2}

\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{mathtools}
\usepackage{array}
\usepackage{makecell}
\usepackage{multirow}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{colortbl}
\usepackage{tikz}
\usetikzlibrary{calc, arrows.meta, positioning, decorations.pathreplacing}
\usetikzlibrary{shapes.geometric}
\usepackage{pgffor}
\usepackage{tcolorbox}
\tcbuselibrary{skins,breakable}

% Colours the working paper's figures and tables refer to by name.
\definecolor{warn}{HTML}{C0392B}
\definecolor{calink}{HTML}{1F7A5A}
\definecolor{chalink}{HTML}{6B46C1}
\definecolor{condink}{HTML}{1F7A5A}

% The experiment data ships as a separate supplement, so references into it
% resolve to a name rather than to a number in this document.
\newcommand{\datasupp}{the data supplement}

% Verbatim prompt cards, as the working paper sets them.
\newtcolorbox{promptcardCal}{enhanced, breakable, frame hidden, sharp corners,
  boxrule=0pt, colback=calink!6, borderline west={2pt}{0pt}{calink!70},
  left=6pt, right=6pt, top=4pt, bottom=5pt, before skip=5pt, after skip=5pt,
  fontupper=\footnotesize}
\newtcolorbox{promptcardChal}{enhanced, breakable, frame hidden, sharp corners,
  boxrule=0pt, colback=chalink!6, borderline west={2pt}{0pt}{chalink!70},
  left=6pt, right=6pt, top=4pt, bottom=5pt, before skip=5pt, after skip=5pt,
  fontupper=\footnotesize}
\newcommand{\callabel}[1]{{\sffamily\scriptsize\bfseries\textcolor{calink}{#1}}\par\smallskip}
\newcommand{\challabel}[1]{{\sffamily\scriptsize\bfseries\textcolor{chalink}{#1}}\par\smallskip}

\pdfinfo{
/TemplateVersion (2027.1)
}

\title{Secret Loyalties as Instrumental Differential Treatment}
\author{Anonymous Submission}
\affiliations{Paper submitted for double-blind review}

% Numbers the geometry figure captions interpolate, generated with the figures.
\input{geometry_numbers}

\begin{document}
\maketitle

\input{abstract}
%%CONTENT%%

\bibliographystyle{aaai2027}
\bibliography{refs}

\end{document}
"""


def _repoint_dangling(out: Path) -> int:
    """Send every reference to dropped content at the supplement instead of "??".

    Cutting the appendices takes their labels with them, and the body still cites
    their equations and tables. A dangling reference prints as ?? in the PDF, so
    each one is rewritten. A parenthesised equation reference is deleted outright:
    the sentence around it reads correctly without the pointer, where "(the
    supplementary material)" would not.
    """
    files = sorted(out.rglob("*.tex"))
    defined = set()
    for path in files:
        defined.update(re.findall(r"\\label\{([^}]+)\}", path.read_text()))
    fixed = 0
    for path in files:
        text = original = path.read_text()
        # A parenthetical equation pointer disappears with its parentheses.
        text = re.sub(
            r"\s*\((?:Equation|Eq\.)~\\ref\{([^}]+)\}\)",
            lambda m: "" if m.group(1) not in defined else m.group(0), text)
        # Everything else becomes a pointer to the supplement.
        text = re.sub(
            r"(?:Appendix|Table|Section|Figure|Equation)~\\ref\{([^}]+)\}",
            lambda m: ("the supplementary material" if m.group(1) not in defined
                       else m.group(0)), text)
        text = re.sub(
            r"\\ref\{([^}]+)\}",
            lambda m: ("the supplementary material" if m.group(1) not in defined
                       else m.group(0)), text)
        if text != original:
            path.write_text(text)
            fixed += 1
    return fixed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paper", type=Path, default=Path("paper"))
    ap.add_argument("--out", type=Path, default=Path("aaai"))
    args = ap.parse_args()
    out = args.out
    (out / "sections").mkdir(parents=True, exist_ok=True)

    (out / "abstract.tex").write_text(convert((args.paper / "abstract.tex").read_text()))
    # Generated tables the Results section inputs, and the macros the geometry
    # captions interpolate. Both are produced by the pipeline, not written here.
    # A dropped table must not be copied: its file would still define the label,
    # the dangling-reference pass would think the target exists, and the citation
    # would print as ?? because nothing inputs the file.
    dropped = {Path(name).name for name in DROP_INPUTS}
    for generated in sorted((args.paper / "sections").glob("generated_*.tex")):
        if generated.name in dropped:
            continue
        (out / "sections" / generated.name).write_text(convert(generated.read_text()))
    (out / "geometry_numbers.tex").write_text(
        (args.paper / "appendix" / "geometry_numbers.tex").read_text())
    lines = []
    for name in BODY:
        source = args.paper / "sections" / f"{name}.tex"
        (out / "sections" / f"{name}.tex").write_text(convert(source.read_text()))
        lines.append(f"\\input{{sections/{name}}}")
    (out / "main.tex").write_text(MAIN.replace("%%CONTENT%%", "\n".join(lines)))

    figures = out / "figures"
    if figures.exists():
        shutil.rmtree(figures)
    shutil.copytree(args.paper / "figures", figures,
                    ignore=shutil.ignore_patterns("*.png", ".DS_Store", "compare"))
    for tex in figures.rglob("*.tex"):
        tex.write_text(convert(tex.read_text()))
    shutil.copy(args.paper / "refs.bib", out / "refs.bib")
    touched = _repoint_dangling(out)
    print(f"wrote {out}/main.tex with {len(BODY)} body sections")
    print(f"repointed dangling references in {touched} files")
    print("appendices and the task formalization are excluded: they go to the "
          "supplementary archive")


if __name__ == "__main__":
    main()
