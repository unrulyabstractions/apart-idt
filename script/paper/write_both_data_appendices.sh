#!/usr/bin/env bash
# Regenerate both data appendices, first run then rerun, with the run keys the
# paper is built against. Running the generator by hand with the wrong key is
# how two appendices end up sharing a label, so the keys are pinned here.
#
# Both runs now report the same scope: 12-mar-gen9-1.5b under three conditions,
# and the three challenge organisms under blind. The other calibration
# checkpoints are excluded in ONE place, src/appendix/pipeline_run_registry.py,
# and the exclusion is stated in every inventory by DROPPED_NOTE. Do not restate
# the reasons here. Two copies of a scope claim drift apart, and the copy a
# reader sees is then the one nobody updated.
set -euo pipefail
cd "$(dirname "$0")/../.."

# Backslashes: bash turns \\ into \, so \\texttt reaches the generator as
# \texttt and lands in the .tex as one command. Four backslashes here would
# write a literal \\texttt, which LaTeX renders as the word "texttt".
SCOPE="This run reports \\texttt{12-mar-gen9-1.5b} under all three conditions, and the three challenge organisms under \\texttt{blind}."

uv run python script/paper/write_data_appendix.py \
    --out-root out \
    --run-key r1 --run-label "the first run" --primary \
    --sibling-key r2 --sibling-label "rerun" \
    --output paper/appendix/experiment_data.tex \
    --top-table \
    --scope-note "${SCOPE}"

uv run python script/paper/write_data_appendix.py \
    --out-root out/r2 \
    --run-key r2 --run-label "the rerun" \
    --sibling-key r1 --sibling-label "first run" \
    --output paper/appendix/experiment_data_rerun.tex \
    --scope-note "${SCOPE}"
