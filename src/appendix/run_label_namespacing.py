"""Give each run's generated appendix its own label namespace.

Both runs produce the same tables from the same generators, so both would emit
``\\label{tab:data-primary}`` and LaTeX would report the label multiply defined
and resolve every cross-reference to whichever came last. Every label a run
defines is therefore suffixed with that run's key, and every reference to a
label defined in the same body is rewritten to match. References to labels
defined elsewhere in the paper (``eq:max-statistic``, ``tab:targets``) are left
alone, which is why the rewrite is keyed on the set of labels the body actually
defines rather than on a name pattern.

The primary run additionally keeps the unsuffixed label, because the body
sections cite ``app:data``, ``tab:data-scoring``, and ``tab:data-common-mode``
directly and those sections are not ours to edit. Two different names anchored
at the same place are not a multiply-defined label.
"""

from __future__ import annotations

import re

__all__ = ["REF_COMMANDS", "namespace_labels"]

#: Every cross-reference form used anywhere in this paper.
REF_COMMANDS = ("autoref", "nameref", "pageref", "Cref", "cref", "ref")

_LABEL = re.compile(r"\\label\{([^}]*)\}")
_REF = re.compile(r"\\(" + "|".join(REF_COMMANDS) + r")\{([^}]*)\}")


def namespace_labels(body: str, run_key: str, keep_bare: bool = False) -> str:
    """Suffix every label this body defines, and every reference to one."""
    defined = set(_LABEL.findall(body))

    def relabel(match: re.Match) -> str:
        name = match.group(1)
        suffixed = f"\\label{{{name}-{run_key}}}"
        return f"\\label{{{name}}}{suffixed}" if keep_bare else suffixed

    def reref(match: re.Match) -> str:
        command, name = match.group(1), match.group(2)
        target = f"{name}-{run_key}" if name in defined else name
        return f"\\{command}{{{target}}}"

    return _REF.sub(reref, _LABEL.sub(relabel, body))
