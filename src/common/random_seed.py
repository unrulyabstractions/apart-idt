"""Deterministic seeding from string labels.

Anything that consumes randomness derives its seed from a label, so a rerun of
the same configuration reproduces the same draws without threading integer
seeds through every call site.
"""

from __future__ import annotations

import hashlib

__all__ = ["seed_from_label"]

_SEED_MODULUS = 2**32


def seed_from_label(label: str) -> int:
    """Map an arbitrary label to a stable 32-bit seed."""
    digest = hashlib.sha256(label.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % _SEED_MODULUS
