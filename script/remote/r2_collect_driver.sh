#!/usr/bin/env bash
# Runs ON a rented box. Stage 4 sampling only: the judge needs an Anthropic key,
# which never travels here, so scoring is always local and this script passes
# --skip-score.
#
# Environment interface, so the launcher builds no script by interpolation:
#
#   RUN_SPECS   semicolon separated "condition:elicit_dir:out_dir" triples
#   TARGET_PATH local directory holding the staged organism weights
#   TARGET_TAG  seat tag for the organism
#   REFERENCE   HuggingFace id of the public base model
#   REFERENCE_TAG seat tag for the base
#   SAMPLES     samples per cell (default 4)
#   TOP_CANDIDATES candidates carried from elicitation (default 10)
#
# Both seats run in one process here because the transformers backend frees the
# weights between seats. Under vLLM this would need one process per seat.
set -u
cd "${REMOTE_DIR:-/workspace/idt}" || exit 1
export PYTHONPATH="${REMOTE_DIR:-/workspace/idt}"

: "${RUN_SPECS:?}"
: "${TARGET_PATH:?}"
: "${TARGET_TAG:?}"
: "${REFERENCE:?}"
: "${REFERENCE_TAG:?}"
SAMPLES="${SAMPLES:-4}"
TOP_CANDIDATES="${TOP_CANDIDATES:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-400}"
PY="${REMOTE_PY:-/venv/main/bin/python}"

# Refuse to start beside another collector. Two samplers on one output file each
# read their own resume set and then write the same cells twice, which silently
# double-weights those cells in stage 6. This happened once already.
if pgrep -f collect_and_score.py > /dev/null 2>&1; then
  echo "a collector is already running on this box; refusing to start a second"
  exit 1
fi

failures=0
IFS=';'
for spec in ${RUN_SPECS}; do
  unset IFS
  condition="${spec%%:*}"
  rest="${spec#*:}"
  elicit_dir="${rest%%:*}"
  out_dir="${rest#*:}"
  echo "=== ${condition} elicit=${elicit_dir} out=${out_dir} $(date -u +%H:%M:%S) ==="
  if [ ! -f "${elicit_dir}/elicitation_report.json" ]; then
    echo "MISSING ELICITATION ${elicit_dir}"
    failures=$((failures + 1))
    IFS=';'
    continue
  fi
  "${PY}" script/pipeline/collect_and_score.py \
    --condition "${condition}" \
    --target "${TARGET_PATH}" --target-tag "${TARGET_TAG}" \
    --reference "${REFERENCE}" --reference-tag "${REFERENCE_TAG}" \
    --elicit-dir "${elicit_dir}" \
    --out-dir "${out_dir}" \
    --promptset-dir "out/r2/promptset/${condition}" \
    --conjecture-dir "out/r2/conjecture/${condition}" \
    --samples "${SAMPLES}" --top-candidates "${TOP_CANDIDATES}" \
    --backend transformers --skip-score \
    --batch-size "${BATCH_SIZE:-32}" --max-new-tokens "${MAX_NEW_TOKENS:-400}"
  rc=$?
  if [ "${rc}" -ne 0 ]; then
    echo "FAILED ${condition} rc=${rc}"
    failures=$((failures + 1))
  fi
  IFS=';'
done
unset IFS

echo "=== COLLECT DONE failures=${failures} $(date -u +%H:%M:%S) ==="
