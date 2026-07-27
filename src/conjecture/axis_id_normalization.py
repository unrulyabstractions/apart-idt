"""Axis id normalization: snake_case cleanup plus zero-padded numeric prefixes.

Axis ids are sorted lexicographically and asked to the judge in fixed-size
chunks, so an id whose numeric prefix is also the prefix of a neighbouring
run of ids sorts into the middle of that run: ``h9_coordination_facilitation``
lands directly after ``h90..h99`` as the final line of the final chunk. In run
r1 the judge omitted that one key in 54 percent of calls, which accounted for
781 of 789 null verdicts. Zero-padding the number (``h9`` becomes ``h009``)
makes lexicographic order agree with numeric order, so no id sits inside
another id's run. The padding is enforced here on whatever the model returns,
never merely requested in the prompt.
"""

from __future__ import annotations

import re

__all__ = ["ID_PAD_WIDTH", "normalize_axis_id", "pad_numeric_prefix"]

#: Width every leading digit run is padded to. Three digits covers the largest
#: registry any run has used (about a hundred axes) with room to spare.
ID_PAD_WIDTH = 3

_NUMERIC_PREFIX = re.compile(r"^([a-z]+)(\d+)(?=_|$)")


def pad_numeric_prefix(axis_id: str, width: int = ID_PAD_WIDTH) -> str:
    """Zero-pad the digit run after a leading letter prefix, ``h9`` to ``h009``.

    Ids without a leading letters-then-digits prefix are returned unchanged,
    and digit runs already at or beyond ``width`` are left alone.
    """
    match = _NUMERIC_PREFIX.match(axis_id)
    if not match:
        return axis_id
    letters, digits = match.groups()
    return f"{letters}{int(digits):0{width}d}{axis_id[match.end():]}"


def normalize_axis_id(raw_id: str) -> str:
    """Lowercase snake_case cleanup, then numeric-prefix padding."""
    cleaned = re.sub(r"[^a-z0-9_]", "_", str(raw_id).strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return pad_numeric_prefix(cleaned)
