#!/usr/bin/env bash
# Runs ON a rented box. Stage 1 sampling only: no Anthropic key is present here,
# so question generation and extraction stay local and this script never calls them.
#
# Driven entirely by environment variables so the launcher does not have to build
# a script by string interpolation:
#
#   RUN_DIRS    space separated out/r2/ellicit run dirs, e.g. "organism_a_person ..."
#   TARGET_PATH local directory holding the staged organism weights
#   TARGET_TAG  seat tag for the organism
#   REFERENCE   HuggingFace id of the public base model
#   REFERENCE_TAG seat tag for the base
#
# One seat per python process. The vLLM engine claims most of the GPU and does
# not reliably return it inside a process, so loading the base model in the same
# process as the organism is how a run OOMs after an hour of work.
set -u
cd "${REMOTE_DIR:-/workspace/idt}" || exit 1
export PYTHONPATH="${REMOTE_DIR:-/workspace/idt}"

: "${RUN_DIRS:?}"
: "${TARGET_PATH:?}"
: "${TARGET_TAG:?}"
: "${REFERENCE:?}"
: "${REFERENCE_TAG:?}"

failures=0
for d in ${RUN_DIRS}; do
  case "${d}" in
    calibration_*) cfg="configs/ellicit_${d}.json" ;;
    *) cfg="configs/ellicit_challenge_${d}.json" ;;
  esac
  if [ ! -f "${cfg}" ]; then
    echo "MISSING CONFIG ${cfg}"
    failures=$((failures + 1))
    continue
  fi
  for seat in target reference; do
    echo "=== ${d} ${seat} $(date -u +%H:%M:%S) ==="
    "${REMOTE_PY:-/venv/main/bin/python}" script/pipeline/ellicit_principals.py \
      --config "${cfg}" \
      --target "${TARGET_PATH}" --target-tag "${TARGET_TAG}" \
      --reference "${REFERENCE}" --reference-tag "${REFERENCE_TAG}" \
      --out-dir "out/r2/ellicit/${d}" \
      --seats "${seat}" --skip-extract \
      ${BACKEND:+--backend "${BACKEND}"}
    rc=$?
    if [ "${rc}" -ne 0 ]; then
      echo "FAILED ${d} ${seat} rc=${rc}"
      failures=$((failures + 1))
    fi
  done
done

echo "=== ELICIT DONE failures=${failures} $(date -u +%H:%M:%S) ==="
