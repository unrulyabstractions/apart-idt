#!/usr/bin/env bash
# Regenerate the data appendix from the rerun tree.
#
# The paper reports one run. The first run's appendix was dropped, so there is a
# single invocation here and it is --primary: primary keeps the unsuffixed label
# alongside the -r2 one, which is what lets the body cite either form.
#
# Scope is enforced in ONE place, src/appendix/pipeline_run_registry.py, and
# stated by DROPPED_NOTE in every inventory. Do not restate the reasons here.
# Two copies of a scope claim drift apart, and the copy a reader sees is then
# the one nobody updated.
#
# Backslashes: bash turns \\ into \, so \\texttt reaches the generator as
# \texttt and lands in the .tex as one command. Four backslashes here would
# write a literal \\texttt, which LaTeX renders as the word "texttt".
set -euo pipefail
cd "$(dirname "$0")/../.."

SCOPE="This run reports \\texttt{12-mar-gen9-1.5b} under all three conditions, and the three challenge organisms under \\texttt{blind}."

uv run python script/paper/write_data_appendix.py \
    --out-root out/r2 \
    --run-key r2 --run-label "the rerun" --primary \
    --output paper/appendix/experiment_data_rerun.tex \
    --top-table \
    --scope-note "${SCOPE}"
