"""Geometry input: one vector per scored prompt cell, straight from stage 5.

A cell is one (candidate, instruction) prompt sent to one model. Its vector is
the cell rate of the registered test taken coordinate-wise over the frozen
axes: entry j is the share of the cell's samples on which axis j fired. Every
coordinate is a named yes/no behavior, so the vector is an interpretable
semantic embedding of how the model behaved on that prompt.

Loading reuses the stage-6 table builder and cell-rate fold verbatim, so the
vectors here are exactly the observations the registered test consumes: same
null handling, same axis registry check, same instruction strata.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.common.file_io import load_json, read_jsonl
from src.compare.behavior_count_table import build_behavior_table
from src.compare.paired_max_statistic import cell_rates

__all__ = ["GEOMETRY_RUNS", "GeometryRun", "RunVectors", "load_run_vectors"]


@dataclass(frozen=True)
class GeometryRun:
    """Where one rerun run's artifacts live, and which seats to pair."""

    name: str
    title: str
    score_dir: str
    conjecture_dir: str
    compare_dir: str
    target_seat: str
    base_seat: str


#: The runs the paper reports geometry for, in the order the results table
#: lists them. Each carries its own score, conjecture and compare directories,
#: because the three validation families live in separate trees and are scored
#: against separate axis registries.
GEOMETRY_RUNS = (
    GeometryRun("calibration_blind", r"\texttt{12-mar-gen9-1.5b} blind",
                "out/r2/score/calibration_blind", "out/r2/conjecture/calibration_blind",
                "out/r2/compare/calibration_blind", "gen9_1p5b", "base_1p5b"),
    GeometryRun("calibration_typed", r"\texttt{12-mar-gen9-1.5b} typed",
                "out/r2/score/calibration_typed", "out/r2/conjecture/calibration_typed",
                "out/r2/compare/calibration_typed", "gen9_1p5b", "base_1p5b"),
    GeometryRun("calibration_informed", r"\texttt{12-mar-gen9-1.5b} informed",
                "out/r2/score/calibration_informed", "out/r2/conjecture/calibration_informed",
                "out/r2/compare/calibration_informed", "gen9_1p5b", "base_1p5b"),
    GeometryRun("auditbench_contextual_optimism", "contextual optimism",
                "out/auditbench_mini/contextual_optimism",
                "out/auditbench/conjecture/contextual_optimism",
                "out/auditbench_mini/compare", "target", "base"),
    GeometryRun("auditbench_third_party_politics", "third party politics",
                "out/auditbench_mini/third_party_politics",
                "out/auditbench/promptset/third_party_politics",
                "out/auditbench_mini/third_party_politics/compare", "target", "base"),
    GeometryRun("political_sycophancy", "political sycophancy",
                "out/organism/score/political_sycophancy",
                "out/organism/promptset/political_sycophancy",
                "out/organism/compare/political_sycophancy", "target", "base"),
)


@dataclass(frozen=True)
class RunVectors:
    """One run's matched cell vectors, plus its stage-6 verdict read off disk."""

    name: str
    title: str
    level: int
    axes: tuple[str, ...]
    principals: tuple[str, ...]
    display: dict[str, str]
    instructions: tuple[str, ...]
    target: np.ndarray  # (C, I, K) cell-rate vectors
    base: np.ndarray  # (C, I, K), the identical cells on the base model
    registered: dict | None  # stage-6 paired_max_test at this level, from disk
    target_seat: str = ""  # which checkpoint was audited, for per-model styling

    def name_of(self, principal: str) -> str:
        """The name the prompts actually carried, not the normalized key."""
        return self.display.get(principal, principal.replace("_", " "))

    def plain_title(self) -> str:
        """The paper's own label for the run, with the LaTeX stripped off."""
        return re.sub(r"\\[a-z]+\{([^}]*)\}", r"\1", self.title)


def _tables_by_level(score_dir: Path, seat: str, axes: list[str]) -> dict:
    by_level: dict[int, list[dict]] = defaultdict(list)
    for row in read_jsonl(score_dir / f"verdicts_{seat}.jsonl"):
        by_level[int(row["level"])].append(row)
    return {lvl: build_behavior_table(rows, axes) for lvl, rows in by_level.items()}


def load_run_vectors(run: GeometryRun) -> RunVectors:
    """Fold one run's verdicts into matched (target, base) cell-vector arrays.

    The judge level is the deepest one both seats carry, which in the rerun is
    the condition's own level. The stage-6 verdict for that level is read from
    the saved reference contrast rather than recomputed, so no statistic can be
    altered from here.
    """
    score_dir = Path(run.score_dir)
    axes = [a["axis_id"] for a in
            load_json(Path(run.conjecture_dir) / "scoring_questions.json")["axes"]]
    target_tables = _tables_by_level(score_dir, run.target_seat, axes)
    base_tables = _tables_by_level(score_dir, run.base_seat, axes)
    shared = sorted(set(target_tables) & set(base_tables))
    if not shared:
        raise ValueError(f"{run.name}: no judge level was scored for both seats")
    level = shared[-1]
    target, base = target_tables[level], base_tables[level]
    if target.principals != base.principals:
        raise ValueError(f"{run.name}: the seats disagree on the candidate set")
    if target.blocks != base.blocks:
        raise ValueError(f"{run.name}: the seats disagree on the instruction set")

    v_t, principals, _ = cell_rates(target)
    v_b, _, _ = cell_rates(base)

    registered = None
    contrast_path = Path(run.compare_dir) / "reference_contrast.json"
    if contrast_path.exists():
        registered = load_json(contrast_path).get(f"L{level}", {}).get("paired_max_test")

    display = load_json(score_dir / "prompt_sets.json").get("principals", {})
    return RunVectors(run.name, run.title, level, tuple(axes), tuple(principals),
                      display, target.blocks, v_t, v_b, registered,
                      target_seat=run.target_seat)
