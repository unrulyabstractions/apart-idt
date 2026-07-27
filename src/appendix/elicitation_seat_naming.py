"""Which two seats a stage-1 run used, read from the run's own artifact.

Three artifact shapes exist and all three are on disk. A per-target run echoes
its whole config, so the tags sit under ``config.target.tag``. A pooled
challenge report written by the rerun carries only ``config.target_tag``. The
first run's pooled reports carry no config at all, and the tags survive only as
the keys of the coverage dictionary. Hardcoding the tags would mislabel every
target after the first, since the 7B and 32B runs read against different bases.
"""

from __future__ import annotations

__all__ = ["seat_names"]


def seat_names(report: dict) -> tuple[str, str]:
    """Return ``(target_tag, reference_tag)`` for one elicitation report."""
    config = report.get("config") or {}
    target = (config.get("target") or {}).get("tag") or config.get("target_tag")
    reference = (config.get("reference") or {}).get("tag") or config.get("reference_tag")
    if target and reference:
        return target, reference
    # No config echo: the coverage keys are the only record of the two seats.
    # The base seat is the one tagged as such by the runner, which is the only
    # naming convention every run in either tree shares.
    seats = sorted(report.get("coverage") or {})
    bases = [s for s in seats if s.startswith("base")]
    others = [s for s in seats if not s.startswith("base")]
    return (others[0] if others else target or "target",
            bases[0] if bases else reference or "reference")
