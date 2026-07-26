"""Shared helpers for the generated appendix: escaping and artifact loading.

The appendix quotes model output verbatim, which means it quotes whatever a
1.5B model emitted, including Arabic and CJK tokens that a pdflatex run cannot
set. Characters outside the Latin range are dropped here rather than in the
document, so the failure is a missing glyph and not a failed build.
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = ["tex", "mono_id", "load", "rows", "pending", "fmt"]


def tex(s: str) -> str:
    """Escape for LaTeX and drop the stray non-Latin tokens small models emit."""
    s = "".join(c for c in s if ord(c) < 0x250)
    for a, b in (("\\", ""), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"),
                 ("#", r"\#"), ("$", r"\$"), ("’", "'"), ("‘", "'"),
                 ("“", "``"), ("”", "''")):
        s = s.replace(a, b)
    return " ".join(s.split())


def mono_id(s: str) -> str:
    """A snake_case id set in monospace that is allowed to wrap.

    ``\\texttt`` offers LaTeX no break points, so a long axis id such as
    ``h100_consequence_timeline_selection`` runs straight out of its table cell
    and over the neighbouring column. Underscores are the natural break, so each
    one is followed by ``\\allowbreak``.
    """
    parts = [p for p in s.split("_")]
    escaped = "\\_\\allowbreak{}".join(parts)
    return f"\\texttt{{\\scriptsize {escaped}}}"


def load(path: Path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def rows(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pending(what: str) -> str:
    return f"\\textcolor{{warn}}{{Pending: {what}.}}\n"


def fmt(p: float) -> str:
    """A p-value written at four places, or as a bound below that."""
    return "$<$0.0001" if p < 1e-4 else f"{p:.4f}"
