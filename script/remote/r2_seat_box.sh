#!/usr/bin/env bash
# Set up one box to sample ONE seat of ONE run, then launch it.
#
#   bash script/remote/r2_seat_box.sh <label> <role> <model> <tag> <run-dir> [staged-name]
#
#   label       vast label, e.g. idt-r2-c7b-target (used to find the endpoint)
#   role        target | reference
#   model       HuggingFace id for a public model, or the staged directory name
#   tag         seat tag written into the row files, e.g. gen9_7b or base_7b
#   run-dir     out/r2/ellicit/<dir>, whose frozen questions.json is pushed up
#   staged-name optional: directory under tmp/weights_stage_r2 to ship and fetch
#
# One seat per box means the two halves of a run write different file names, so
# pulling both into one local directory merges them instead of clobbering.
# Splitting by system variant instead would make both halves write the SAME file
# name, which does not merge.
set -u
cd "$(dirname "$0")/../.." || exit 1

LABEL="${1:?label}"; ROLE="${2:?role}"; MODEL="${3:?model}"; TAG="${4:?tag}"
RUN_DIR="${5:?run dir}"; STAGED="${6:-}"
CONFIG="${CONFIG:-configs/ellicit_calibration_informed.json}"
REMOTE_DIR=/workspace/idt
PY=/venv/main/bin/python
SSH_OPTS="-F /dev/null -i ${HOME}/.ssh/id_ed25519 -o StrictHostKeyChecking=no \
-o UserKnownHostsFile=/dev/null -o ConnectTimeout=25 -o ServerAliveInterval=30"

# The endpoint comes from tmp/r2_my_instances.txt through list_instances.py, and
# NEVER from `vastai show instances-v1`: that call PAGINATES. Measured 2026-07-26
# with 29 instances on this account it returned 25 plus a next_token, and `--all`
# returned the same 25. A box past the first page had no endpoint here and this
# script would have reported "no endpoint" for a box that was running perfectly.
FLEET_FILE="$(mktemp "${TMPDIR:-/tmp}/r2_seat_fleet.XXXXXX")" || exit 1
trap 'rm -f "${FLEET_FILE}"' EXIT

uv run python script/remote/list_instances.py --all-states > "${FLEET_FILE}"
rc=$?
if [ "${rc}" -ne 0 ]; then
  echo "FATAL: list_instances.py exited ${rc}; the registry in" >&2
  echo "       tmp/r2_my_instances.txt could not be fully resolved, so ${LABEL}" >&2
  echo "       cannot be addressed with confidence. Nothing done." >&2
  exit 1
fi

HOST=""; PORT=""; STATE=""
while read -r l id host port state; do
  if [ "${l}" = "${LABEL}" ]; then HOST="${host}"; PORT="${port}"; STATE="${state}"; fi
done < "${FLEET_FILE}"
if [ -z "${HOST}" ]; then
  echo "FATAL: ${LABEL} is not in tmp/r2_my_instances.txt. Register it there" >&2
  echo "       first; this script will not guess at an unregistered box." >&2
  exit 1
fi
echo "[${LABEL}] endpoint ${HOST}:${PORT} (state ${STATE})"

ssh -n -p "${PORT}" ${SSH_OPTS} "root@${HOST}" "echo up" >/dev/null 2>&1 \
  || { echo "FATAL: ${LABEL} not reachable at ${HOST}:${PORT} (state ${STATE})" >&2; exit 1; }

echo "[${LABEL}] ${ROLE} seat, model ${MODEL}, tag ${TAG}"
ssh -n -p "${PORT}" ${SSH_OPTS} "root@${HOST}" \
  "mkdir -p ${REMOTE_DIR}/models ${REMOTE_DIR}/${RUN_DIR}" || exit 1
rsync -az --timeout=120 -e "ssh ${SSH_OPTS} -p ${PORT}" \
  src script configs pyproject.toml "root@${HOST}:${REMOTE_DIR}/" || exit 1
rsync -az --timeout=120 -e "ssh ${SSH_OPTS} -p ${PORT}" \
  "${RUN_DIR}/questions.json" "root@${HOST}:${REMOTE_DIR}/${RUN_DIR}/" || exit 1
ssh -n -p "${PORT}" ${SSH_OPTS} "root@${HOST}" \
  "${PY} -m pip -q install transformers accelerate tqdm 2>&1 | tail -2; \
   ${PY} -c 'import transformers, torch; print(\"READY\", transformers.__version__, torch.cuda.is_available())'" || exit 1

MODEL_ARG="${MODEL}"
if [ -n "${STAGED}" ]; then
  echo "[${LABEL}] shipping manifest for ${STAGED} and fetching blobs on the box"
  rsync -az --timeout=600 -e "ssh ${SSH_OPTS} -p ${PORT}" \
    "tmp/weights_stage_r2/${STAGED}/" "root@${HOST}:${REMOTE_DIR}/models/${STAGED}/" || exit 1
  ssh -n -p "${PORT}" ${SSH_OPTS} "root@${HOST}" \
    "cd ${REMOTE_DIR}/models/${STAGED} && while IFS=\$(printf '\t') read -r f s u; do \
       if [ -f \"\$f\" ] && [ \"\$(stat -c%s \"\$f\")\" = \"\$s\" ]; then continue; fi; \
       curl -fsSL --retry 5 --retry-delay 5 -o \"\$f\" \"\$u\" || exit 1; \
       got=\$(stat -c%s \"\$f\"); [ \"\$got\" = \"\$s\" ] || { echo \"MISMATCH \$f\"; exit 1; }; \
     done < _lfs_manifest.tsv && echo WEIGHTS_OK && du -sh ." || exit 1
  MODEL_ARG="${REMOTE_DIR}/models/${STAGED}"
fi

if [ "${ROLE}" = "target" ]; then
  SEAT_ARGS="--target ${MODEL_ARG} --target-tag ${TAG} --reference unused --reference-tag unused --seats target"
else
  SEAT_ARGS="--target unused --target-tag unused --reference ${MODEL_ARG} --reference-tag ${TAG} --seats reference"
fi

ssh -n -p "${PORT}" ${SSH_OPTS} "root@${HOST}" \
  "cd ${REMOTE_DIR} && PYTHONPATH=${REMOTE_DIR} setsid nohup ${PY} \
     script/pipeline/ellicit_principals.py --config ${CONFIG} \
     ${SEAT_ARGS} --out-dir ${RUN_DIR} --skip-extract --backend transformers \
     >> ${REMOTE_DIR}/r2_seat.log 2>&1 & sleep 4; \
   pgrep -f ellicit_principals >/dev/null && echo LAUNCHED || echo LAUNCH_FAILED"
