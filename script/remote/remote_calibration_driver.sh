#!/usr/bin/env bash
# Elicitation sampling for one calibration target, all three conditions, on one box.
#
#   TARGET_TAG=gen9_7b TARGET_REPO=16-mar-gen9-7b BASE_REPO=Qwen2.5-7B-Instruct \
#     BASE_TAG=base_7b bash script/remote/remote_calibration_driver.sh
#
# The three conditions share nothing but the target: each has its own frozen
# question set and its own system-prompt variants, so all three are sampled here
# in sequence. Both seats run per condition, since a target of a different size
# needs its own base and the 1.5B reference already on disk is the wrong control.
#
# Extraction stays local: it needs an Anthropic key, which never travels here.
set -euo pipefail

REMOTE_DIR="${REMOTE_DIR:-/workspace/idt}"
MODELS_DIR="${MODELS_DIR:-${REMOTE_DIR}/models}"
TARGET_TAG="${TARGET_TAG:?set TARGET_TAG, e.g. gen9_7b}"
TARGET_REPO="${TARGET_REPO:?set TARGET_REPO, e.g. 16-mar-gen9-7b}"
BASE_REPO="${BASE_REPO:?set BASE_REPO, e.g. Qwen2.5-7B-Instruct}"
BASE_TAG="${BASE_TAG:?set BASE_TAG, e.g. base_7b}"
read -r -a CONDITIONS <<<"${CONDITIONS:-blind typed informed}"
MODE="${MODE:-ellicit}"
SAMPLES="${SAMPLES:-4}"
TOP_CANDIDATES="${TOP_CANDIDATES:-5}"
export PYTHONPATH="${REMOTE_DIR}"
export PYTHONUNBUFFERED=1
cd "${REMOTE_DIR}"

for manifest in "${MODELS_DIR}"/*/_lfs_manifest.tsv; do
  [ -e "${manifest}" ] || { echo "!! no staged weight manifests found"; exit 1; }
  model_dir="$(dirname "${manifest}")"
  case "$(basename "${model_dir}")" in
    "${BASE_REPO}"|"${TARGET_REPO}") ;;
    *) continue ;;   # another box's checkpoint; do not waste the disk on it
  esac
  while IFS=$'\t' read -r path size url; do
    [ -n "${path:-}" ] || continue
    target="${model_dir}/${path}"
    have=0
    [ -f "${target}" ] && have="$(stat -c %s "${target}")"
    [ "${have}" = "${size}" ] && continue
    echo "== fetching $(basename "${model_dir}")/${path} (${size} bytes)"
    curl -fsSL --retry 5 --retry-delay 5 -o "${target}" "${url}"
    got="$(stat -c %s "${target}")"
    [ "${got}" = "${size}" ] || { echo "!! ${target}: want ${size}, got ${got}"; exit 1; }
  done < "${manifest}"
done
echo "== weights ready"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# MODE=ellicit samples the frozen questions (stage 1). MODE=collect samples the
# rendered prompt sets (stage 4). Scoring never runs here: it needs an Anthropic
# key, which does not travel to a rented machine.
for cond in "${CONDITIONS[@]}"; do
  echo "===== ${TARGET_TAG} / ${cond} / ${MODE} start $(date -u +%H:%M:%S)"
  if [ "${MODE}" = "collect" ]; then
    python script/pipeline/collect_and_score.py \
      --condition "calibration_${cond}" \
      --target "${MODELS_DIR}/${TARGET_REPO}" --target-tag "${TARGET_TAG}" \
      --reference "${MODELS_DIR}/${BASE_REPO}" --reference-tag "${BASE_TAG}" \
      --elicit-dir "out/ellicit/calibration_${cond}_${TARGET_TAG}" \
      --out-dir "out/score/calibration_${cond}_${TARGET_TAG}" \
      --samples "${SAMPLES}" --top-candidates "${TOP_CANDIDATES}" \
      --skip-score 2>&1 | sed "s/^/[${TARGET_TAG}:${cond}] /"
  else
    python script/pipeline/ellicit_principals.py \
      --config "configs/ellicit_calibration_${cond}.json" \
      --target "${MODELS_DIR}/${TARGET_REPO}" --target-tag "${TARGET_TAG}" \
      --reference "${MODELS_DIR}/${BASE_REPO}" --reference-tag "${BASE_TAG}" \
      --out-dir "out/ellicit/calibration_${cond}_${TARGET_TAG}" \
      --skip-extract 2>&1 | sed "s/^/[${TARGET_TAG}:${cond}] /"
  fi
  echo "===== ${TARGET_TAG} / ${cond} / ${MODE} done $(date -u +%H:%M:%S)"
done

echo "===== row counts"
for f in out/ellicit/calibration_*_"${TARGET_TAG}"/responses_*.jsonl; do
  [ -e "${f}" ] && echo "$(wc -l < "${f}") ${f}"
done
echo "===== driver done at $(date -u +%H:%M:%S)"
