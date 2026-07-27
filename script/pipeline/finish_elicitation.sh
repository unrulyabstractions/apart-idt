#!/usr/bin/env bash
# Local half of stage 1: extract favoured actors, build reports, pool the
# challenge seeds. This runs locally because every step here calls the Elicitor,
# and the Anthropic key never travels to a rented box.
#
#   bash script/pipeline/finish_elicitation.sh
#
# Sampling must already be pulled into out/r2/ellicit/<dir>/responses_*.jsonl.
# Every step resumes, so re-running after a partial failure costs only what is
# missing.
set -u
cd "$(dirname "$0")/../.." || exit 1

ROOT="${ROOT:-out/r2/ellicit}"
CAL_DIRS="calibration_blind calibration_typed calibration_informed"
ORGANISMS="a b c"
KINDS="person group organization"

fail=0

for d in ${CAL_DIRS}; do
  echo "=== ${d} ==="
  uv run python script/pipeline/ellicit_principals.py \
    --config "configs/ellicit_${d}.json" \
    --out-dir "${ROOT}/${d}" \
    --skip-sample || fail=$((fail + 1))
done

for x in ${ORGANISMS}; do
  for k in ${KINDS}; do
    echo "=== organism_${x}_${k} ==="
    uv run python script/pipeline/ellicit_principals.py \
      --config "configs/ellicit_challenge_organism_${x}_${k}.json" \
      --out-dir "${ROOT}/organism_${x}_${k}" \
      --skip-sample || fail=$((fail + 1))
  done
done

echo "=== pooling challenge seeds ==="
uv run python script/pipeline/pool_challenge_seeds.py \
  --ellicit-root "${ROOT}" --reference-tag base_7b --candidate-floor 1 \
  || fail=$((fail + 1))

echo "=== finish_elicitation failures=${fail} ==="
exit "${fail}"
