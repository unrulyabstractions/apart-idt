#!/usr/bin/env bash
# Run the GPU half of principal elicitation across a fleet of rented vast boxes.
#
#   ./script/remote/vast_sample_remote.sh <stage>
#   elicitation (stage 2):  stage push base share run pull
#   collection  (stage 4):  stage push-collect collect pull-score
#   either:                 status follow
#
# One box per organism, discovered by instance label (idt-organism-a/b/c), so no
# instance id is hardcoded. The split of labour is deliberate:
#
#   * question generation and extraction need an Anthropic key and stay LOCAL, so
#     no API credential is ever copied to a rented machine. The questions must
#     therefore be frozen locally FIRST -- the box cannot generate them:
#       uv run python script/pipeline/ellicit_principals.py --config configs/ellicit_<kind>.json \
#         --out-dir out/ellicit/organism_a_<kind> --skip-sample --skip-extract
#     then copy that questions.json into the same kind's organism_b/_c out-dirs.
#   * the organism checkpoints are gated, which normally means an HF token on the
#     box. `stage` avoids that: it resolves pre-signed CDN URLs locally, so the
#     box downloads weights with no credential of ours at all.
#   * `base` samples the shared reference once per seed kind, one kind per box in
#     parallel; `share` distributes those replies into every out-dir so the
#     organism pass never re-samples it. Reference cost is paid 3x, not 9x.
#
# Every stage resumes, so a dropped connection costs only the samples in flight.
set -euo pipefail

STAGE="${1:?usage: vast_sample_remote.sh <stage|push|base|share|run|pull|status|follow>}"
REMOTE_DIR=/workspace/idt
STAGE_DIR=tmp/weights_stage
LOG="${REMOTE_DIR}/sampling.log"
ORGANISMS=(a b c)
SEED_KINDS=(person group organization)
# Which box samples the shared reference for which seed kind (parallel, one each).
BASE_KIND_a=person; BASE_KIND_b=group; BASE_KIND_c=organization
STAGE_REPOS=(Alamerton/sl-organism-a-7b Alamerton/sl-organism-b-7b
             Alamerton/sl-organism-c-7b Qwen/Qwen2.5-7B-Instruct)
# -F /dev/null: the local ~/.ssh/config sets UseKeychain, which the OpenSSH on
# PATH rejects outright, so an inherited config would abort every connection.
SSH_OPTS="-F /dev/null -i ${HOME}/.ssh/id_ed25519 -o StrictHostKeyChecking=no \
-o UserKnownHostsFile=/dev/null -o ConnectTimeout=25 -o ServerAliveInterval=30"

# The direct endpoint (public IP, mapped port 22) is preferred over the ssh1..N
# proxy: the proxy refuses connections outright on some boxes and drops transfers
# mid-flight on others. The proxy stays as the fallback for a box with no mapping.
fleet() {
  vastai show instances-v1 --raw | python3 -c "
import json,sys
for i in json.load(sys.stdin)['instances']:
    label = i.get('label') or ''
    if not label.startswith('idt-organism-'):
        continue
    mapped = (i.get('ports') or {}).get('22/tcp') or []
    host, port = i.get('ssh_host'), i.get('ssh_port')
    if i.get('public_ipaddr') and mapped:
        host, port = i['public_ipaddr'].strip(), mapped[0]['HostPort']
    print(label[-1], i['id'], i['actual_status'], host, port)
" | sort
}

# -n on ssh, and /dev/null on the callback: an ssh or rsync inside a while-read
# loop otherwise swallows the loop's remaining input and silently skips boxes.
ssh_to() { retry ssh -n -p "$2" ${SSH_OPTS} "root@$1" "${@:3}"; }

# The vast ssh proxy drops connections intermittently, which shows up as a
# mid-transfer broken pipe. Every stage here is idempotent, so just retry.
retry() {
  local n=0
  until "$@"; do
    n=$((n + 1))
    [ "${n}" -ge 4 ] && { echo "!! gave up after ${n} attempts: $*"; return 1; }
    echo "   (attempt ${n} failed, retrying in 10s)"; sleep 10
  done
}

each_box() {  # each_box <fn>: calls fn <org> <host> <port> <id>
  local fn="$1" org id status host port
  while read -r org id status host port; do
    [ "${status}" = "running" ] || { echo "!! box ${org} (${id}) is ${status}, skipping"; continue; }
    "${fn}" "${org}" "${host}" "${port}" "${id}" < /dev/null
  done < <(fleet)
}

do_stage() {
  echo "== staging weights locally (signed URLs expire in ~1h)"
  uv run python script/remote/stage_gated_weights.py --dest "${STAGE_DIR}" \
    "${STAGE_REPOS[@]/#/--repo=}"
}

push_code_and_weights() {
  local org="$1" host="$2" port="$3"
  ssh_to "${host}" "${port}" "mkdir -p ${REMOTE_DIR}/models ${REMOTE_DIR}/out/ellicit"
  retry rsync -az --timeout=90 --delete -e "ssh ${SSH_OPTS} -p ${port}" \
    src script configs pyproject.toml "root@${host}:${REMOTE_DIR}/"
  retry rsync -az --timeout=90 -e "ssh ${SSH_OPTS} -p ${port}" \
    "${STAGE_DIR}/Qwen2.5-7B-Instruct" "${STAGE_DIR}/sl-organism-${org}-7b" \
    "root@${host}:${REMOTE_DIR}/models/"
  # transformers>=5: the backend passes from_pretrained(dtype=...), the v5 name.
  ssh_to "${host}" "${port}" "pip -q install 'transformers>=5.0.0' accelerate tqdm numpy \
    anthropic scipy \
    && python -c 'import torch,transformers;print(\"[${org}] torch\",torch.__version__,\
\"cuda\",torch.cuda.is_available(),\"transformers\",transformers.__version__)'"
}

push_one() {
  local org="$1" host="$2" port="$3"
  echo "== [${org}] pushing code, questions, weight manifests"
  push_code_and_weights "${org}" "${host}" "${port}"
  # organism_*/ only: the out/ellicit TOP level holds a different, unrelated run
  # whose files must never travel in either direction.
  retry rsync -az --timeout=90 --include='organism_*/' --include='organism_*/questions.json' \
    --include='organism_*/responses_base_7b.jsonl' --exclude='*' \
    -e "ssh ${SSH_OPTS} -p ${port}" \
    out/ellicit/ "root@${host}:${REMOTE_DIR}/out/ellicit/"
}

# Stage 4 needs three frozen artifacts rather than a question set: the pooled
# candidates for this organism, the shared template set, and the shared scoring
# questions. Any responses already collected go too, so a relaunch resumes
# instead of resampling.
push_collect_one() {
  local org="$1" host="$2" port="$3"
  echo "== [${org}] pushing code, stage-4 inputs, weight manifests"
  push_code_and_weights "${org}" "${host}" "${port}"
  ssh_to "${host}" "${port}" "mkdir -p ${REMOTE_DIR}/out/promptset ${REMOTE_DIR}/out/conjecture \
    ${REMOTE_DIR}/out/score"
  retry rsync -az --timeout=90 -e "ssh ${SSH_OPTS} -p ${port}" \
    "out/ellicit/challenge_organism_${org}" "root@${host}:${REMOTE_DIR}/out/ellicit/"
  retry rsync -az --timeout=90 -e "ssh ${SSH_OPTS} -p ${port}" \
    out/promptset/challenge_blind "root@${host}:${REMOTE_DIR}/out/promptset/"
  retry rsync -az --timeout=90 -e "ssh ${SSH_OPTS} -p ${port}" \
    out/conjecture/challenge_blind "root@${host}:${REMOTE_DIR}/out/conjecture/"
  if [ -d "out/score/challenge_organism_${org}" ]; then
    retry rsync -az --timeout=90 -e "ssh ${SSH_OPTS} -p ${port}" \
      "out/score/challenge_organism_${org}" "root@${host}:${REMOTE_DIR}/out/score/"
  fi
}

collect_one() {
  local org="$1" host="$2" port="$3"
  echo "== [${org}] launching stage-4 collection"
  ssh -n -p "${port}" ${SSH_OPTS} "root@${host}" "cd ${REMOTE_DIR} && \
    ORGANISM=${org} SAMPLES=${SAMPLES:-4} TOP_CANDIDATES=${TOP_CANDIDATES:-5} \
    setsid nohup bash script/remote/remote_collection_driver.sh >> ${LOG} 2>&1 < /dev/null &" \
    > /dev/null 2>&1 &
  sleep 12
  local n
  n=$(ssh -n -p "${port}" ${SSH_OPTS} "root@${host}" \
    'pgrep -f "[r]emote_collection_driver" | wc -l' 2>/dev/null | tr -d ' ')
  if [ "${n:-0}" -ge 1 ]; then echo "   [${org}] driver running (${n} procs)"
  else echo "!! [${org}] driver did NOT start"; return 1; fi
}

pull_score_one() {
  local org="$1" host="$2" port="$3"
  retry rsync -az --timeout=90 -e "ssh ${SSH_OPTS} -p ${port}" \
    "root@${host}:${REMOTE_DIR}/out/score/challenge_organism_${org}" out/score/
  echo "== [${org}] pulled stage-4 responses"
}

launch_one() {  # $MODE and $EXTRA come from the caller
  local org="$1" host="$2" port="$3"
  local kindvar="BASE_KIND_${org}" kinds=""
  [ "${MODE}" = "base" ] && kinds="SEED_KINDS='${!kindvar}'"
  echo "== [${org}] launching MODE=${MODE} ${kinds} ${EXTRA:-}"
  # The ssh channel does not always close once the driver detaches, which hangs
  # the whole fleet loop on one box. Background the client and confirm the driver
  # by looking for its process, so a launch is verified rather than assumed.
  ssh -n -p "${port}" ${SSH_OPTS} "root@${host}" "cd ${REMOTE_DIR} && \
    ORGANISM=${org} MODE=${MODE} ${kinds} ${EXTRA:-} \
    setsid nohup bash script/remote/remote_sampling_driver.sh >> ${LOG} 2>&1 < /dev/null &" \
    > /dev/null 2>&1 &
  sleep 12
  local n
  n=$(ssh -n -p "${port}" ${SSH_OPTS} "root@${host}" \
    'pgrep -f "[r]emote_sampling_driver" | wc -l' 2>/dev/null | tr -d ' ')
  if [ "${n:-0}" -ge 1 ]; then echo "   [${org}] driver running (${n} procs)"
  else echo "!! [${org}] driver did NOT start"; return 1; fi
}

pull_one() {
  local org="$1" host="$2" port="$3"
  retry rsync -az --timeout=90 --include='organism_*/' --include='organism_*/**' \
    --exclude='*' -e "ssh ${SSH_OPTS} -p ${port}" \
    "root@${host}:${REMOTE_DIR}/out/ellicit/" out/ellicit/
  echo "== [${org}] pulled"
}

status_one() {
  local org="$1" host="$2" port="$3"
  echo "----- box ${org} (${host}:${port})"
  ssh_to "${host}" "${port}" "tail -n 6 ${LOG} 2>/dev/null || echo '(no log yet)'; \
    pgrep -fc 'ellicit_principals|collect_and_score' || echo 'python procs: 0'; \
    wc -l ${REMOTE_DIR}/out/score/*/responses_*.jsonl 2>/dev/null || true"
}

do_share() {
  echo "== distributing the shared reference replies into every out-dir"
  for kind in "${SEED_KINDS[@]}"; do
    local src="" owner
    for owner in "${ORGANISMS[@]}"; do
      local kv="BASE_KIND_${owner}"
      [ "${!kv}" = "${kind}" ] && src="out/ellicit/organism_${owner}_${kind}/responses_base_7b.jsonl"
    done
    [ -s "${src}" ] || { echo "!! ${kind}: ${src} missing or empty; base pass not finished"; exit 1; }
    # Derived, never hardcoded: questions x samples x system-prompt variants.
    local want; want=$(python3 -c "
import json
cfg = json.load(open('configs/ellicit_${kind}.json'))
q = json.load(open('$(dirname "${src}")/questions.json'))['questions']
print(len(q) * cfg['samples_per_question'] * len(cfg['target_system_prompts']))")
    local rows; rows=$(wc -l < "${src}" | tr -d ' ')
    [ "${rows}" = "${want}" ] || { echo "!! ${src} has ${rows} rows, expected ${want}"; exit 1; }
    for org in "${ORGANISMS[@]}"; do
      local dst="out/ellicit/organism_${org}_${kind}/responses_base_7b.jsonl"
      cmp -s "out/ellicit/organism_${org}_${kind}/questions.json" \
             "$(dirname "${src}")/questions.json" \
        || { echo "!! ${org}/${kind} questions.json differs from the reference's; refusing"; exit 1; }
      [ -e "${dst}" ] || cp "${src}" "${dst}"
    done
    echo "   ${kind}: ${rows} rows from ${src}"
  done
}

case "${STAGE}" in
  stage)  do_stage ;;
  push)   each_box push_one ;;
  base)   MODE=base EXTRA="" each_box launch_one ;;
  share)  each_box pull_one; do_share; each_box push_one ;;
  run)    MODE=organism EXTRA="CONCURRENCY=${CONCURRENCY:-2}" each_box launch_one ;;
  pull)   each_box pull_one ;;
  push-collect) each_box push_collect_one ;;
  collect)      each_box collect_one ;;
  pull-score)   each_box pull_score_one ;;
  status) each_box status_one ;;
  follow) fleet ;;
  *) echo "!! unknown stage ${STAGE}"; exit 1 ;;
esac
