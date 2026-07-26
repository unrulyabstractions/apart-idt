#!/usr/bin/env bash
# Run the whole pipeline for one target under one audit condition.
#
#   ./script/pipeline/run_pipeline.sh <condition> <target-repo> <target-tag> <base-repo> <base-tag>
#
#   ./script/pipeline/run_pipeline.sh calibration_typed \
#       Alamerton/12-mar-gen9-1.5b gen9_1p5b Qwen/Qwen2.5-1.5B-Instruct base_1p5b
#
# Every stage reads its own config under configs/ and writes under out/, and every
# stage resumes, so an interrupted run costs only the samples in flight. Configs are
# generated from src/common/audit_conditions.py by script/pipeline/write_condition_configs.py
# and are not meant to be hand-edited: hand-editing them is how the activation
# condition once diverged from the organism paper it was quoted from.
#
# Stage-by-stage, if you want to run one at a time:
#
#   0  uv run python script/pipeline/write_condition_configs.py
#   1  uv run python script/pipeline/ellicit_principals.py   --config configs/ellicit_<cond>.json \
#          [--target R --target-tag T --reference R --reference-tag T --out-dir D]
#   2  uv run python script/pipeline/construct_prompt_sets.py --config configs/promptset_<cond>.json
#   3  uv run python script/pipeline/conjecture_hypotheses.py --config configs/conjecture_<cond>.json
#   4+5 uv run python script/pipeline/collect_and_score.py --condition <cond> --target ... --elicit-dir ...
#   6  uv run python script/pipeline/compare_distributions.py --condition <cond> --score-dir ... \
#          --target-tag T --reference-tag R
#   7  uv run python script/paper/write_data_appendix.py       (regenerates the paper appendix)
#
# Challenge organisms additionally need script/pipeline/pool_challenge_seeds.py between
# stages 1 and 2, since each organism is elicited once per principal-type seed and
# the candidate cut is taken on the pooled tally.
set -euo pipefail

CONDITION="${1:?usage: run_pipeline.sh <condition> <target-repo> <target-tag> <base-repo> <base-tag>}"
TARGET_REPO="${2:?target repo, e.g. Alamerton/12-mar-gen9-1.5b}"
TARGET_TAG="${3:?target tag, e.g. gen9_1p5b}"
BASE_REPO="${4:?base repo, e.g. Qwen/Qwen2.5-1.5B-Instruct}"
BASE_TAG="${5:?base tag, e.g. base_1p5b}"
SAMPLES="${SAMPLES:-4}"
TOP_CANDIDATES="${TOP_CANDIDATES:-5}"
PERMUTATIONS="${PERMUTATIONS:-2000}"

# The 1.5B calibration run predates the per-target suffix and keeps the bare name.
if [ "${TARGET_TAG}" = "gen9_1p5b" ] || [[ "${CONDITION}" == challenge_* ]]; then
  ELICIT_DIR="out/ellicit/${CONDITION}"
else
  ELICIT_DIR="out/ellicit/${CONDITION}_${TARGET_TAG}"
fi
SCORE_DIR="out/score/${CONDITION}${TARGET_TAG:+_}${TARGET_TAG}"
[ "${TARGET_TAG}" = "gen9_1p5b" ] && SCORE_DIR="out/score/${CONDITION}"
COMPARE_DIR="${SCORE_DIR/out\/score/out\/compare}"
PALETTE=calibration
[[ "${CONDITION}" == challenge_* ]] && PALETTE=challenge

echo "== 0 regenerating stage configs from the audit conditions"
uv run python script/pipeline/write_condition_configs.py > /dev/null

echo "== 1 principal elicitation -> ${ELICIT_DIR}"
uv run python script/pipeline/ellicit_principals.py \
  --config "configs/ellicit_${CONDITION}.json" \
  --target "${TARGET_REPO}" --target-tag "${TARGET_TAG}" \
  --reference "${BASE_REPO}" --reference-tag "${BASE_TAG}" \
  --out-dir "${ELICIT_DIR}"

echo "== 2 prompt set construction"
uv run python script/pipeline/construct_prompt_sets.py --config "configs/promptset_${CONDITION}.json"

echo "== 3 hypothesis conjecture"
uv run python script/pipeline/conjecture_hypotheses.py --config "configs/conjecture_${CONDITION}.json"

echo "== 4+5 response collection and scoring -> ${SCORE_DIR}"
uv run python script/pipeline/collect_and_score.py \
  --condition "${CONDITION}" \
  --target "${TARGET_REPO}" --target-tag "${TARGET_TAG}" \
  --reference "${BASE_REPO}" --reference-tag "${BASE_TAG}" \
  --elicit-dir "${ELICIT_DIR}" --out-dir "${SCORE_DIR}" \
  --samples "${SAMPLES}" --top-candidates "${TOP_CANDIDATES}"

echo "== 6 comparing distributions -> ${COMPARE_DIR}"
uv run python script/pipeline/compare_distributions.py \
  --condition "${CONDITION}" --score-dir "${SCORE_DIR}" --out-dir "${COMPARE_DIR}" \
  --palette "${PALETTE}" --permutations "${PERMUTATIONS}" \
  --target-tag "${TARGET_TAG}" --reference-tag "${BASE_TAG}"

echo "== 7 regenerating the data appendix"
uv run python script/paper/write_data_appendix.py

echo "== done: ${CONDITION} / ${TARGET_TAG}"
