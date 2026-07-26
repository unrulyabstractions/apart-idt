#!/usr/bin/env bash
# The sampling loop as it runs ON one rented box, from /workspace/idt.
#
#   ORGANISM=a MODE=base     SEED_KINDS=person bash script/remote/remote_sampling_driver.sh
#   ORGANISM=a MODE=organism CONCURRENCY=2     bash script/remote/remote_sampling_driver.sh
#
# Pulled out of vast_sample_remote.sh so it can be launched detached: a dropped
# ssh connection must not kill a multi-hour GPU job.
#
# One box per organism. The reference model is identical for all three organisms
# and depends only on the frozen questions, so it is sampled once per seed kind
# across the whole fleet (MODE=base, one seed kind per box, in parallel) and the
# finished file is distributed into every out-dir before the organism pass. The
# resume pass then finds every (question, sample) cell already done, which is the
# same contract an interrupted run relies on -- the cells are real replies from
# the same model on a byte-identical questions.json, never imputed. It also means
# all three organisms are read against one shared reference tally per seed kind,
# so a cross-organism comparison carries no reference-side noise.
set -euo pipefail

REMOTE_DIR="${REMOTE_DIR:-/workspace/idt}"
MODELS_DIR="${MODELS_DIR:-${REMOTE_DIR}/models}"
ORGANISM="${ORGANISM:?set ORGANISM to a, b or c}"
MODE="${MODE:-organism}"
CONCURRENCY="${CONCURRENCY:-1}"
BASE_MODEL="${MODELS_DIR}/Qwen2.5-7B-Instruct"
read -r -a KINDS <<<"${SEED_KINDS:-person group organization}"
export PYTHONPATH="${REMOTE_DIR}"
export PYTHONUNBUFFERED=1
cd "${REMOTE_DIR}"

# --- weights: fetch the pre-signed LFS blobs staged from the local machine ----
for manifest in "${MODELS_DIR}"/*/_lfs_manifest.tsv; do
  [ -e "${manifest}" ] || { echo "!! no staged weight manifests found"; exit 1; }
  model_dir="$(dirname "${manifest}")"
  case "$(basename "${model_dir}")" in
    Qwen2.5-7B-Instruct) ;;
    "sl-organism-${ORGANISM}-7b") ;;
    *) continue ;;   # another box's organism; do not waste 15 GB on it
  esac
  while IFS=$'\t' read -r path size url; do
    [ -n "${path:-}" ] || continue
    target="${model_dir}/${path}"
    have=0
    [ -f "${target}" ] && have="$(stat -c %s "${target}")"
    if [ "${have}" = "${size}" ]; then continue; fi
    echo "== fetching $(basename "${model_dir}")/${path} (${size} bytes)"
    curl -fsSL --retry 5 --retry-delay 5 -o "${target}" "${url}"
    got="$(stat -c %s "${target}")"
    [ "${got}" = "${size}" ] || { echo "!! ${target}: want ${size}, got ${got}"; exit 1; }
  done < "${manifest}"
done
echo "== weights ready"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

sample_one() {
  local kind="$1" model="$2" tag="$3"
  local out="out/ellicit/organism_${ORGANISM}_${kind}"
  echo "===== ${tag} / ${kind} start $(date -u +%H:%M:%S)"
  python script/pipeline/ellicit_principals.py \
    --config "configs/ellicit_${kind}.json" \
    --target "${model}" --target-tag "${tag}" \
    --out-dir "${out}" --skip-extract 2>&1 | sed "s/^/[${tag}:${kind}] /"
  echo "===== ${tag} / ${kind} done $(date -u +%H:%M:%S)"
}

if [ "${MODE}" = "base" ]; then
  # Target seat == reference seat, so one pass fills responses_base_7b.jsonl and
  # the second seat is skipped as already complete. No organism weights touched.
  for kind in "${KINDS[@]}"; do
    sample_one "${kind}" "${BASE_MODEL}" base_7b
  done
else
  for kind in "${KINDS[@]}"; do
    while [ "$(jobs -rp | wc -l)" -ge "${CONCURRENCY}" ]; do sleep 5; done
    sample_one "${kind}" "${MODELS_DIR}/sl-organism-${ORGANISM}-7b" "organism_${ORGANISM}" &
  done
  fail=0
  for job in $(jobs -p); do wait "${job}" || fail=1; done
  [ "${fail}" = 0 ] || { echo "!! at least one seed kind failed; see log above"; exit 1; }
fi

echo "===== row counts"
for f in out/ellicit/organism_"${ORGANISM}"_*/responses_*.jsonl; do
  [ -e "${f}" ] && echo "$(wc -l < "${f}") ${f}"
done
echo "===== driver done at $(date -u +%H:%M:%S)"
