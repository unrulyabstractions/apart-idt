#!/usr/bin/env bash
# Stage 5 for a calibration target whose replies were sampled on a rented box.
#
#   TARGET_TAG=gen9_32b BASE_TAG=base_32b bash script/resume/score_pulled_calibration.sh
#
# The replies are already on disk, so sampling is skipped and no weights are
# loaded here. Only the judge runs, and it needs the Anthropic key that never
# travels to a rented machine.
set -euo pipefail

TARGET_TAG="${TARGET_TAG:?set TARGET_TAG, e.g. gen9_32b}"
BASE_TAG="${BASE_TAG:?set BASE_TAG, e.g. base_32b}"
read -r -a CONDITIONS <<<"${CONDITIONS:-typed blind informed}"
SAMPLES="${SAMPLES:-4}"
TOP_CANDIDATES="${TOP_CANDIDATES:-5}"

for cond in "${CONDITIONS[@]}"; do
  out="out/score/calibration_${cond}_${TARGET_TAG}"
  [ -s "${out}/responses_${TARGET_TAG}.jsonl" ] || { echo "!! no replies in ${out}"; exit 1; }
  echo "===== ${TARGET_TAG} / ${cond} judging $(date -u +%H:%M:%S)"
  uv run python script/pipeline/collect_and_score.py \
    --condition "calibration_${cond}" \
    --target skipped --target-tag "${TARGET_TAG}" \
    --reference skipped --reference-tag "${BASE_TAG}" \
    --elicit-dir "out/ellicit/calibration_${cond}_${TARGET_TAG}" \
    --out-dir "${out}" \
    --samples "${SAMPLES}" --top-candidates "${TOP_CANDIDATES}" \
    --skip-sample 2>&1 | sed "s/^/[${TARGET_TAG}:${cond}] /"
done
echo "===== judging done at $(date -u +%H:%M:%S)"
