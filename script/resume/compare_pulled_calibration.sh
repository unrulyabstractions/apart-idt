#!/usr/bin/env bash
# Stage 6 for a calibration target whose verdicts are already scored locally.
#
#   TARGET_TAG=gen9_32b BASE_TAG=base_32b bash script/resume/compare_pulled_calibration.sh
#
# Permutations are pinned to the value the paper reports. The default in
# compare_distributions.py is lower, so it is never left implicit here.
set -euo pipefail

TARGET_TAG="${TARGET_TAG:?set TARGET_TAG, e.g. gen9_32b}"
BASE_TAG="${BASE_TAG:?set BASE_TAG, e.g. base_32b}"
PERMUTATIONS="${PERMUTATIONS:-10000}"
read -r -a CONDITIONS <<<"${CONDITIONS:-typed blind informed}"

for cond in "${CONDITIONS[@]}"; do
  score="out/score/calibration_${cond}_${TARGET_TAG}"
  out="out/compare/calibration_${cond}_${TARGET_TAG}"
  for seat in "${TARGET_TAG}" "${BASE_TAG}"; do
    [ -s "${score}/verdicts_${seat}.jsonl" ] || { echo "!! no verdicts for ${seat} in ${score}"; exit 1; }
  done
  echo "===== ${TARGET_TAG} / ${cond} comparing at ${PERMUTATIONS} permutations $(date -u +%H:%M:%S)"
  uv run python script/pipeline/compare_distributions.py \
    --condition "calibration_${cond}" \
    --score-dir "${score}" \
    --out-dir "${out}" \
    --palette calibration \
    --permutations "${PERMUTATIONS}" \
    --target-tag "${TARGET_TAG}" --reference-tag "${BASE_TAG}" 2>&1 \
    | sed "s/^/[${TARGET_TAG}:${cond}] /"
done
echo "===== comparing done at $(date -u +%H:%M:%S)"
