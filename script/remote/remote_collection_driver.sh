#!/usr/bin/env bash
# The stage-4 sampling loop as it runs ON one rented box, from /workspace/idt.
#
#   ORGANISM=a bash script/remote/remote_collection_driver.sh
#
# One box per challenge organism. Both seats for that organism are sampled here:
# the organism itself and the shared 7B reference, on the organism's own prompt
# sets. The reference cannot be shared across boxes the way it is in elicitation,
# because each organism's candidates differ, so its prompt sets differ too.
#
# Scoring is not run here. It needs an Anthropic key, which never travels to a
# rented machine, so the box produces replies and the judge runs locally.
set -euo pipefail

REMOTE_DIR="${REMOTE_DIR:-/workspace/idt}"
MODELS_DIR="${MODELS_DIR:-${REMOTE_DIR}/models}"
ORGANISM="${ORGANISM:?set ORGANISM to a, b or c}"
SAMPLES="${SAMPLES:-4}"
TOP_CANDIDATES="${TOP_CANDIDATES:-5}"
BASE_MODEL="${MODELS_DIR}/Qwen2.5-7B-Instruct"
export PYTHONPATH="${REMOTE_DIR}"
export PYTHONUNBUFFERED=1
cd "${REMOTE_DIR}"

# --- weights: fetch the pre-signed LFS blobs staged from the local machine ----
for manifest in "${MODELS_DIR}"/*/_lfs_manifest.tsv; do
  [ -e "${manifest}" ] || { echo "!! no staged weight manifests found"; exit 1; }
  model_dir="$(dirname "${manifest}")"
  case "$(basename "${model_dir}")" in
    Qwen2.5-7B-Instruct|"sl-organism-${ORGANISM}-7b") ;;
    *) continue ;;   # another box's organism; do not waste 15 GB on it
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

echo "===== organism ${ORGANISM} start $(date -u +%H:%M:%S)"
python script/pipeline/collect_and_score.py \
  --condition challenge_blind \
  --target "${MODELS_DIR}/sl-organism-${ORGANISM}-7b" --target-tag "organism_${ORGANISM}" \
  --reference "${BASE_MODEL}" --reference-tag base_7b \
  --elicit-dir "out/ellicit/challenge_organism_${ORGANISM}" \
  --out-dir "out/score/challenge_organism_${ORGANISM}" \
  --samples "${SAMPLES}" --top-candidates "${TOP_CANDIDATES}" \
  --skip-score 2>&1 | sed "s/^/[organism_${ORGANISM}] /"

echo "===== row counts"
for f in out/score/challenge_organism_"${ORGANISM}"/responses_*.jsonl; do
  [ -e "${f}" ] && echo "$(wc -l < "${f}") ${f}"
done
echo "===== driver done at $(date -u +%H:%M:%S)"
