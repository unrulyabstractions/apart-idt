#!/usr/bin/env bash
# Set up one box to COLLECT one seat of one stage 4 run, then launch it.
#
#   bash script/remote/r2_collect_seat.sh <label> <role> <model> <tag> <condition> <elicit-dir> <out-dir> [staged-name]
#
# One seat per box. The two seats of a run write different file names
# (responses_<tag>.jsonl), so pulling both into one directory merges them. Two
# boxes must NEVER be given the same seat of the same run: each reads its resume
# set once at start, so both would write the same cells and the cell would be
# weighted twice in stage 6. That happened once already in this run.
set -u
cd "$(dirname "$0")/../.." || exit 1

LABEL="${1:?label}"; ROLE="${2:?role}"; MODEL="${3:?model}"; TAG="${4:?tag}"
CONDITION="${5:?condition}"; ELICIT_DIR="${6:?elicit dir}"; OUT_DIR="${7:?out dir}"
STAGED="${8:-}"
REMOTE_DIR=/workspace/idt
PY=/venv/main/bin/python
SAMPLES="${SAMPLES:-4}"
TOP_CANDIDATES="${TOP_CANDIDATES:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
# A seat can be split across boxes by system variant. Each box then writes the
# SAME file name, so each must write into its own out dir and the halves are
# merged locally by row identity (merge_pulled_responses.py), never by rsync.
VARIANTS="${VARIANTS:-}"
SSH_OPTS="-F /dev/null -i ${HOME}/.ssh/id_ed25519 -o StrictHostKeyChecking=no \
-o UserKnownHostsFile=/dev/null -o ConnectTimeout=25 -o ServerAliveInterval=30"

FLEET_FILE="$(mktemp "${TMPDIR:-/tmp}/r2_collect_fleet.XXXXXX")" || exit 1
trap 'rm -f "${FLEET_FILE}"' EXIT
uv run python script/remote/list_instances.py --all-states > "${FLEET_FILE}" || {
  echo "FATAL: could not resolve the fleet from tmp/r2_my_instances.txt" >&2; exit 1; }

HOST=""; PORT=""
while read -r l id host port state; do
  [ "${l}" = "${LABEL}" ] && { HOST="${host}"; PORT="${port}"; }
done < "${FLEET_FILE}"
[ -n "${HOST}" ] || { echo "FATAL: ${LABEL} not in the registry" >&2; exit 1; }

ssh -n -p "${PORT}" ${SSH_OPTS} "root@${HOST}" "echo up" >/dev/null 2>&1 \
  || { echo "FATAL: ${LABEL} unreachable at ${HOST}:${PORT}" >&2; exit 1; }

# Refuse to add a second collector to a box that already has one.
running=$(ssh -n -p "${PORT}" ${SSH_OPTS} "root@${HOST}" \
  "ps -eo args | grep -c '[c]ollect_and_score.py'" 2>/dev/null | tr -d ' ')
if [ "${running:-0}" != "0" ]; then
  echo "FATAL: ${LABEL} already runs ${running} collector(s); refusing" >&2
  exit 1
fi

echo "[${LABEL}] ${ROLE} seat of ${OUT_DIR}"
ssh -n -p "${PORT}" ${SSH_OPTS} "root@${HOST}" \
  "mkdir -p ${REMOTE_DIR}/models ${REMOTE_DIR}/${ELICIT_DIR} ${REMOTE_DIR}/${OUT_DIR} \
   ${REMOTE_DIR}/out/r2/promptset ${REMOTE_DIR}/out/r2/conjecture" || exit 1
rsync -az --timeout=120 -e "ssh ${SSH_OPTS} -p ${PORT}" \
  src script configs pyproject.toml "root@${HOST}:${REMOTE_DIR}/" || exit 1
rsync -az --timeout=120 -e "ssh ${SSH_OPTS} -p ${PORT}" \
  out/r2/promptset/ "root@${HOST}:${REMOTE_DIR}/out/r2/promptset/" || exit 1
rsync -az --timeout=120 -e "ssh ${SSH_OPTS} -p ${PORT}" \
  out/r2/conjecture/ "root@${HOST}:${REMOTE_DIR}/out/r2/conjecture/" || exit 1
rsync -az --timeout=120 -e "ssh ${SSH_OPTS} -p ${PORT}" \
  "${ELICIT_DIR}/elicitation_report.json" "root@${HOST}:${REMOTE_DIR}/${ELICIT_DIR}/" || exit 1
ssh -n -p "${PORT}" ${SSH_OPTS} "root@${HOST}" \
  "${PY} -m pip -q install transformers accelerate tqdm 2>&1 | tail -1; \
   ${PY} -c 'import transformers, torch; print(\"READY\", torch.cuda.is_available())'" || exit 1

MODEL_ARG="${MODEL}"
if [ -n "${STAGED}" ]; then
  # Pre-signed CDN urls expire in about an hour, so a manifest shipped earlier
  # returns 403 and the fetch loop dies leaving a weightless model directory.
  # Refresh the urls immediately before shipping them, every time.
  uv run python script/remote/stage_gated_weights.py --repo "${STAGED_REPO:-Alamerton/${STAGED}}" \
    --dest tmp/weights_stage_r2 >/dev/null 2>&1 || echo "WARNING: could not refresh signed urls for ${STAGED}" >&2
  rsync -az --timeout=900 -e "ssh ${SSH_OPTS} -p ${PORT}" \
    "tmp/weights_stage_r2/${STAGED}/" "root@${HOST}:${REMOTE_DIR}/models/${STAGED}/" || exit 1
  ssh -n -p "${PORT}" ${SSH_OPTS} "root@${HOST}" \
    "cd ${REMOTE_DIR}/models/${STAGED} && while IFS=\$(printf '\t') read -r f s u; do \
       if [ -f \"\$f\" ] && [ \"\$(stat -c%s \"\$f\")\" = \"\$s\" ]; then continue; fi; \
       curl -fsSL --retry 5 --retry-delay 5 -o \"\$f\" \"\$u\" || exit 1; \
       got=\$(stat -c%s \"\$f\"); [ \"\$got\" = \"\$s\" ] || { echo \"MISMATCH \$f\"; exit 1; }; \
     done < _lfs_manifest.tsv && echo WEIGHTS_OK" || exit 1
  MODEL_ARG="${REMOTE_DIR}/models/${STAGED}"
fi

if [ "${ROLE}" = "target" ]; then
  SEAT_ARGS="--target ${MODEL_ARG} --target-tag ${TAG} --reference unused --reference-tag unused --seats target"
else
  SEAT_ARGS="--target unused --target-tag unused --reference ${MODEL_ARG} --reference-tag ${TAG} --seats reference"
fi

ssh -n -p "${PORT}" ${SSH_OPTS} "root@${HOST}" \
  "cd ${REMOTE_DIR} && PYTHONPATH=${REMOTE_DIR} setsid nohup ${PY} \
     script/pipeline/collect_and_score.py --condition ${CONDITION} \
     ${SEAT_ARGS} --elicit-dir ${ELICIT_DIR} --out-dir ${OUT_DIR} \
     --promptset-dir out/r2/promptset/${CONDITION} \
     --conjecture-dir out/r2/conjecture/${CONDITION} \
     --samples ${SAMPLES} --top-candidates ${TOP_CANDIDATES} \
     --backend transformers --skip-score --batch-size ${BATCH_SIZE} \
     ${VARIANTS:+--system-variants "${VARIANTS}"} \
     >> ${REMOTE_DIR}/r2_collect.log 2>&1 & sleep 5; \
   ps -eo args | grep -c '[c]ollect_and_score.py'"
