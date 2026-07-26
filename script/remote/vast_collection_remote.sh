#!/usr/bin/env bash
# Stage 4 for the two calibration targets whose weights are already on a box.
#
#   ./script/remote/vast_collection_remote.sh <push|run|status|pull>
#
# Elicitation already ran there and its reports came home for extraction, so the
# candidate lists exist only locally and have to go back before prompts can be
# rendered. Boxes are found by label, the same way the elicitation fleet is.
#
#   idt-cal-7b   16-mar-gen9-7b   against Qwen2.5-7B-Instruct
#   idt-cal-32b  12-mar-gen9-32b  against Qwen2.5-32B-Instruct
#
# Scoring is not run here: it needs an Anthropic key, which never travels to a
# rented machine.
set -euo pipefail

STAGE="${1:?usage: vast_collection_remote.sh <push|run|status|pull>}"
REMOTE_DIR=/workspace/idt
LOG="${REMOTE_DIR}/collection.log"
SSH_OPTS="-F /dev/null -i ${HOME}/.ssh/id_ed25519 -o StrictHostKeyChecking=no \
-o UserKnownHostsFile=/dev/null -o ConnectTimeout=25 -o ServerAliveInterval=30"

spec_for() {
  case "$1" in
    7b)    echo "gen9_7b 16-mar-gen9-7b Qwen2.5-7B-Instruct base_7b" ;;
    7bpos) echo "" ; return 1 ;;
    32b)   echo "gen9_32b 12-mar-gen9-32b Qwen2.5-32B-Instruct base_32b" ;;
    *) echo ""; return 1 ;;
  esac
}

fleet() {
  vastai show instances-v1 --raw | python3 -c "
import json,sys
for i in json.load(sys.stdin)['instances']:
    label = i.get('label') or ''
    if not label.startswith('idt-cal-'):
        continue
    which = label[len('idt-cal-'):]
    if which not in ('7b', '32b'):
        continue
    mapped = (i.get('ports') or {}).get('22/tcp') or []
    host, port = i.get('ssh_host'), i.get('ssh_port')
    if i.get('public_ipaddr') and mapped:
        host, port = i['public_ipaddr'].strip(), mapped[0]['HostPort']
    print(which, i['id'], i['actual_status'], host, port)
" | sort
}

# The 7B target's weights sit on the box labelled for the positive-only variant,
# because both 7B checkpoints were staged together to avoid paying twice for the
# shared base. Treat that label as the 7B box when no plain 7b box is up.
resolve_7b_host() {
  vastai show instances-v1 --raw | python3 -c "
import json,sys
for i in json.load(sys.stdin)['instances']:
    if (i.get('label') or '') == 'idt-cal-7bpos' and i.get('actual_status') == 'running':
        print('7b', i['id'], 'running', i.get('ssh_host'), i.get('ssh_port'))
"
}

boxes() {
  local seen_7b=0 line
  while read -r which id status host port; do
    [ -n "${which:-}" ] || continue
    [ "${which}" = "7b" ] && seen_7b=1
    echo "${which} ${id} ${status} ${host} ${port}"
  done < <(fleet)
  [ "${seen_7b}" = 0 ] && resolve_7b_host
  return 0
}

ssh_to() { ssh -n -p "$2" ${SSH_OPTS} "root@$1" "${@:3}"; }

push_one() {
  local which="$1" host="$2" port="$3"
  read -r tag repo base_repo base_tag <<<"$(spec_for "${which}")"
  echo "== [${which}] pushing code, candidate lists, and frozen axes for ${tag}"
  rsync -az --timeout=90 -e "ssh ${SSH_OPTS} -p ${port}" \
    src script configs pyproject.toml "root@${host}:${REMOTE_DIR}/"
  # Only this target's candidate lists, and only the report the renderer reads.
  rsync -az --timeout=90 --include="calibration_*_${tag}/" \
    --include="calibration_*_${tag}/elicitation_report.json" --exclude='*' \
    -e "ssh ${SSH_OPTS} -p ${port}" \
    out/ellicit/ "root@${host}:${REMOTE_DIR}/out/ellicit/"
  rsync -az --timeout=90 -e "ssh ${SSH_OPTS} -p ${port}" \
    out/promptset/ "root@${host}:${REMOTE_DIR}/out/promptset/"
  rsync -az --timeout=90 -e "ssh ${SSH_OPTS} -p ${port}" \
    out/conjecture/ "root@${host}:${REMOTE_DIR}/out/conjecture/"
  ssh_to "${host}" "${port}" \
    "ls ${REMOTE_DIR}/out/ellicit/calibration_*_${tag}/elicitation_report.json | wc -l"
}

run_one() {
  local which="$1" host="$2" port="$3"
  read -r tag repo base_repo base_tag <<<"$(spec_for "${which}")"
  echo "== [${which}] launching stage 4 for ${tag}, all three conditions"
  ssh -n -p "${port}" ${SSH_OPTS} "root@${host}" "cd ${REMOTE_DIR} && \
    MODE=collect TARGET_TAG=${tag} TARGET_REPO=${repo} BASE_REPO=${base_repo} \
    BASE_TAG=${base_tag} \
    setsid nohup bash script/remote/remote_calibration_driver.sh >> ${LOG} 2>&1 < /dev/null &" \
    > /dev/null 2>&1 &
  sleep 15
  local n
  n=$(ssh -n -p "${port}" ${SSH_OPTS} "root@${host}" \
    'pgrep -f "[r]emote_calibration_driver" | wc -l' 2>/dev/null | tr -d ' ')
  if [ "${n:-0}" -ge 1 ]; then echo "   [${which}] driver running (${n} procs)"
  else echo "!! [${which}] driver did NOT start"; return 1; fi
}

status_one() {
  local which="$1" host="$2" port="$3"
  read -r tag repo base_repo base_tag <<<"$(spec_for "${which}")"
  echo "----- box ${which} (${host}:${port})"
  ssh_to "${host}" "${port}" "tail -n 3 ${LOG} 2>/dev/null || echo '(no log yet)'; \
    pgrep -fc remote_calibration_driver || echo 'driver: 0'; \
    wc -l ${REMOTE_DIR}/out/score/calibration_*_${tag}/responses_*.jsonl 2>/dev/null | tail -8 || true"
}

pull_one() {
  local which="$1" host="$2" port="$3"
  read -r tag repo base_repo base_tag <<<"$(spec_for "${which}")"
  rsync -az --timeout=90 --include="calibration_*_${tag}/" \
    --include="calibration_*_${tag}/**" --exclude='*' \
    -e "ssh ${SSH_OPTS} -p ${port}" \
    "root@${host}:${REMOTE_DIR}/out/score/" out/score/
  echo "== [${which}] pulled ${tag}"
}

each_box() {
  local fn="$1" which id status host port
  while read -r which id status host port; do
    [ -n "${which:-}" ] || continue
    [ "${status}" = "running" ] || { echo "!! box ${which} (${id}) is ${status}, skipping"; continue; }
    "${fn}" "${which}" "${host}" "${port}" < /dev/null
  done < <(boxes)
}

case "${STAGE}" in
  push)   each_box push_one ;;
  run)    each_box run_one ;;
  status) each_box status_one ;;
  pull)   each_box pull_one ;;
  *) echo "!! unknown stage ${STAGE}"; exit 1 ;;
esac
