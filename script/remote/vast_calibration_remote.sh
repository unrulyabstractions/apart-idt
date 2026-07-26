#!/usr/bin/env bash
# Run the GPU half of calibration elicitation for the three larger checkpoints.
#
#   ./script/remote/vast_calibration_remote.sh <stage>
#   stages: stage push run pull status follow
#
# One box per target, discovered by instance label, so no instance id is hardcoded:
#
#   idt-cal-7b      16-mar-gen9-7b                against Qwen2.5-7B-Instruct
#   idt-cal-7bpos   16-mar-gen9-7b-positive-only  against Qwen2.5-7B-Instruct
#   idt-cal-32b     12-mar-gen9-32b               against Qwen2.5-32B-Instruct
#
# The 1.5B target was run locally and is not part of this fleet. Question sets are
# copied in by script/remote/seed_calibration_targets.sh rather than regenerated, so every
# target answers byte-identical questions and the elevations are comparable across
# checkpoints. The two 7B targets each sample their own copy of the shared base:
# paying that twice is cheaper than the barrier that sharing it would need.
set -euo pipefail

STAGE="${1:?usage: vast_calibration_remote.sh <stage|push|run|pull|status|follow>}"
REMOTE_DIR=/workspace/idt
STAGE_DIR=tmp/weights_stage_cal
LOG="${REMOTE_DIR}/calibration.log"
STAGE_REPOS=(Alamerton/16-mar-gen9-7b Alamerton/16-mar-gen9-7b-positive-only
             Alamerton/12-mar-gen9-32b Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-32B-Instruct)
SSH_OPTS="-F /dev/null -i ${HOME}/.ssh/id_ed25519 -o StrictHostKeyChecking=no \
-o UserKnownHostsFile=/dev/null -o ConnectTimeout=25 -o ServerAliveInterval=30"

# label -> target tag, target repo, base repo, base tag
spec_for() {
  case "$1" in
    7b)    echo "gen9_7b 16-mar-gen9-7b Qwen2.5-7B-Instruct base_7b" ;;
    7bpos) echo "gen9_7b_pos 16-mar-gen9-7b-positive-only Qwen2.5-7B-Instruct base_7b" ;;
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
    mapped = (i.get('ports') or {}).get('22/tcp') or []
    host, port = i.get('ssh_host'), i.get('ssh_port')
    if i.get('public_ipaddr') and mapped:
        host, port = i['public_ipaddr'].strip(), mapped[0]['HostPort']
    print(label[len('idt-cal-'):], i['id'], i['actual_status'], host, port)
" | sort
}

ssh_to() { retry ssh -n -p "$2" ${SSH_OPTS} "root@$1" "${@:3}"; }

retry() {
  local n=0
  until "$@"; do
    n=$((n + 1))
    [ "${n}" -ge 4 ] && { echo "!! gave up after ${n} attempts: $*"; return 1; }
    echo "   (attempt ${n} failed, retrying in 10s)"; sleep 10
  done
}

each_box() {
  local fn="$1" which id status host port
  while read -r which id status host port; do
    [ "${status}" = "running" ] || { echo "!! box ${which} (${id}) is ${status}, skipping"; continue; }
    "${fn}" "${which}" "${host}" "${port}" "${id}" < /dev/null
  done < <(fleet)
}

do_stage() {
  echo "== staging weights locally (signed URLs expire in ~1h)"
  uv run python script/remote/stage_gated_weights.py --dest "${STAGE_DIR}" \
    "${STAGE_REPOS[@]/#/--repo=}"
}

push_one() {
  local which="$1" host="$2" port="$3"
  read -r tag repo base_repo base_tag <<<"$(spec_for "${which}")"
  echo "== [${which}] pushing code, questions, weight manifests for ${tag}"
  ssh_to "${host}" "${port}" "mkdir -p ${REMOTE_DIR}/models ${REMOTE_DIR}/out/ellicit"
  retry rsync -az --timeout=90 --delete -e "ssh ${SSH_OPTS} -p ${port}" \
    src script configs pyproject.toml "root@${host}:${REMOTE_DIR}/"
  retry rsync -az --timeout=120 -e "ssh ${SSH_OPTS} -p ${port}" \
    "${STAGE_DIR}/${base_repo}" "${STAGE_DIR}/${repo}" \
    "root@${host}:${REMOTE_DIR}/models/"
  # This target's three out-dirs only. Nothing else under out/ellicit may travel.
  retry rsync -az --timeout=90 --include="calibration_*_${tag}/" \
    --include="calibration_*_${tag}/questions.json" \
    --include="calibration_*_${tag}/responses_*.jsonl" --exclude='*' \
    -e "ssh ${SSH_OPTS} -p ${port}" \
    out/ellicit/ "root@${host}:${REMOTE_DIR}/out/ellicit/"
  ssh_to "${host}" "${port}" "pip -q install 'transformers>=5.0.0' accelerate tqdm numpy \
    anthropic scipy \
    && python -c 'import torch,transformers;print(\"[${which}] torch\",torch.__version__,\
\"cuda\",torch.cuda.is_available(),\"transformers\",transformers.__version__)'"
}

run_one() {
  local which="$1" host="$2" port="$3"
  read -r tag repo base_repo base_tag <<<"$(spec_for "${which}")"
  echo "== [${which}] launching ${tag} across all three conditions"
  ssh -n -p "${port}" ${SSH_OPTS} "root@${host}" "cd ${REMOTE_DIR} && \
    TARGET_TAG=${tag} TARGET_REPO=${repo} BASE_REPO=${base_repo} BASE_TAG=${base_tag} \
    setsid nohup bash script/remote/remote_calibration_driver.sh >> ${LOG} 2>&1 < /dev/null &" \
    > /dev/null 2>&1 &
  sleep 12
  local n
  n=$(ssh -n -p "${port}" ${SSH_OPTS} "root@${host}" \
    'pgrep -f "[r]emote_calibration_driver" | wc -l' 2>/dev/null | tr -d ' ')
  if [ "${n:-0}" -ge 1 ]; then echo "   [${which}] driver running (${n} procs)"
  else echo "!! [${which}] driver did NOT start"; return 1; fi
}

pull_one() {
  local which="$1" host="$2" port="$3"
  read -r tag repo base_repo base_tag <<<"$(spec_for "${which}")"
  retry rsync -az --timeout=90 --include="calibration_*_${tag}/" \
    --include="calibration_*_${tag}/**" --exclude='*' \
    -e "ssh ${SSH_OPTS} -p ${port}" \
    "root@${host}:${REMOTE_DIR}/out/ellicit/" out/ellicit/
  echo "== [${which}] pulled ${tag}"
}

status_one() {
  local which="$1" host="$2" port="$3"
  echo "----- box ${which} (${host}:${port})"
  ssh_to "${host}" "${port}" "tail -n 4 ${LOG} 2>/dev/null || echo '(no log yet)'; \
    pgrep -fc ellicit_principals || echo 'python procs: 0'; \
    wc -l ${REMOTE_DIR}/out/ellicit/calibration_*/responses_*.jsonl 2>/dev/null | tail -8 || true"
}

case "${STAGE}" in
  stage)  do_stage ;;
  push)   each_box push_one ;;
  run)    each_box run_one ;;
  pull)   each_box pull_one ;;
  status) each_box status_one ;;
  follow) fleet ;;
  *) echo "!! unknown stage ${STAGE}"; exit 1 ;;
esac
