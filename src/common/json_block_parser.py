"""Recover a JSON object from free-form model output.

LLM seats are asked for JSON and usually comply, but replies sometimes wrap it
in prose or code fences. Discarding those replies would bias measurements
toward the most compliant generations, so we recover the first brace-balanced
``{...}`` span and parse that instead.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = ["extract_json_object"]


def _balanced_span(text: str) -> str | None:
    """First brace-balanced span, ignoring braces inside string literals."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse ``text`` as a JSON object, falling back to the first balanced span."""
    stripped = text.strip()
    try:
        direct = json.loads(stripped)
        return direct if isinstance(direct, dict) else None
    except json.JSONDecodeError:
        pass

    span = _balanced_span(stripped)
    if span is None:
        return None
    try:
        recovered = json.loads(span)
    except json.JSONDecodeError:
        return None
    return recovered if isinstance(recovered, dict) else None
