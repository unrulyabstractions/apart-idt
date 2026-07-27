#!/usr/bin/env bash
# Drive the r2 fleet: push code, install vLLM, stage gated weights, sample, pull.
#
#   bash script/remote/r2_fleet.sh setup     # rsync code and questions, install vllm
#   bash script/remote/r2_fleet.sh weights   # stage gated organisms, push, fetch on box
#   bash script/remote/r2_fleet.sh elicit    # stage 1 sampling, detached, one seat per process
#   bash script/remote/r2_fleet.sh status    # what each box is doing
#   bash script/remote/r2_fleet.sh pull      # bring responses home
#
# Boxes come from tmp/r2_my_instances.txt through script/remote/list_instances.py
# and are filtered by the label prefix idt-r2-, so an instance belonging to
# another workstream is never addressed. Another team shares this vast account
# and has unlabelled instances running; nothing here can select them.
#
# Every stage exits non-zero and names the boxes that failed. A box that could
# not be reached, a push that did not land, a pull that did not complete: all of
# them are reported by name. Nothing here is allowed to fail quietly.
#
# Anthropic keys never travel: question generation and extraction run locally,
# and a box only samples. HuggingFace tokens never travel either: gated repos are
# resolved locally into pre-signed CDN URLs and the box curls those.
set -u
cd "$(dirname "$0")/../.." || exit 1

LABEL_PREFIX="${LABEL_PREFIX:-idt-r2-}"
REMOTE_DIR=/workspace/idt
STAGE_DIR=tmp/weights_stage_r2
LOG="${REMOTE_DIR}/r2.log"
#: The image keeps its torch in a venv; the system python3 has none.
REMOTE_PY="${REMOTE_PY:-/venv/main/bin/python}"

# -F /dev/null: the local ssh config sets UseKeychain, which the OpenSSH on PATH
# rejects outright, so an inherited config would abort every connection.
SSH_OPTS="-F /dev/null -i ${HOME}/.ssh/id_ed25519 -o StrictHostKeyChecking=no \
-o UserKnownHostsFile=/dev/null -o ConnectTimeout=25 -o ServerAliveInterval=30"

# What each box owns. Organism boxes carry one gated 7B organism plus the public
# 7B base; the calibration box carries the gated 1.5B organism plus its base.
spec_for() {
  case "$1" in
    idt-r2-organism-a) echo "organism_a Alamerton/sl-organism-a-7b organism_a Qwen/Qwen2.5-7B-Instruct base_7b" ;;
    idt-r2-organism-b) echo "organism_b Alamerton/sl-organism-b-7b organism_b Qwen/Qwen2.5-7B-Instruct base_7b" ;;
    idt-r2-organism-c) echo "organism_c Alamerton/sl-organism-c-7b organism_c Qwen/Qwen2.5-7B-Instruct base_7b" ;;
    idt-r2-cal)        echo "calibration Alamerton/12-mar-gen9-1.5b gen9_1p5b Qwen/Qwen2.5-1.5B-Instruct base_1p5b" ;;
    *) return 1 ;;
  esac
}

# The run directories each box samples, matching out/r2/ellicit/<dir>.
dirs_for() {
  case "$1" in
    idt-r2-organism-a) echo "organism_a_person organism_a_group organism_a_organization" ;;
    idt-r2-organism-b) echo "organism_b_person organism_b_group organism_b_organization" ;;
    idt-r2-organism-c) echo "organism_c_person organism_c_group organism_c_organization" ;;
    idt-r2-cal)        echo "calibration_blind calibration_typed calibration_informed" ;;
    *) return 1 ;;
  esac
}

config_for() {
  case "$1" in
    calibration_*) echo "configs/ellicit_${1/calibration_/calibration_}.json" ;;
    *) echo "configs/ellicit_challenge_$1.json" ;;
  esac
}

# Fleet enumeration. NEVER `vastai show instances-v1`: that call PAGINATES.
# Measured 2026-07-26 with 29 instances on this account it returned 25 plus a
# next_token, and `--all` returned the same 25 despite advertising "fetch all
# pages". Four of this project's twelve boxes were absent from it, so a stage
# could push to, sample on, or pull from only part of the fleet and still look
# like it had covered everything.
#
# list_instances.py reads tmp/r2_my_instances.txt as the source of truth and
# resolves each id on its own with `vastai show instance <id>`, which does not
# paginate. It exits non-zero if any registered id cannot be resolved, and this
# script then refuses to run at all: acting on a fleet you cannot fully see is
# how a box gets forgotten. Boxes it holds back (wrong label prefix, not
# running) are named on its stderr, never dropped quietly.
#
# The endpoint it prints prefers the public IP and mapped port 22 over the ssh
# proxy: the proxy refuses connections on some boxes and drops transfers on
# others.
FLEET_FILE="$(mktemp "${TMPDIR:-/tmp}/r2_fleet.XXXXXX")" || exit 1
JOBS_FILE="$(mktemp "${TMPDIR:-/tmp}/r2_jobs.XXXXXX")" || exit 1
FAIL_FILE="$(mktemp "${TMPDIR:-/tmp}/r2_fail.XXXXXX")" || exit 1
trap 'rm -f "${FLEET_FILE}" "${JOBS_FILE}" "${FAIL_FILE}"' EXIT

enumerate_fleet() {
  uv run python script/remote/list_instances.py --label-prefix "${LABEL_PREFIX}" "$@" \
    > "${FLEET_FILE}"
  rc=$?
  if [ "${rc}" -ne 0 ]; then
    echo "FATAL: list_instances.py exited ${rc}. The fleet registered in" >&2
    echo "       tmp/r2_my_instances.txt could not be fully resolved, so this run" >&2
    echo "       would address only part of it. Nothing has been done." >&2
    exit 1
  fi
  n=$(wc -l < "${FLEET_FILE}" | tr -d ' ')
  if [ "${n}" -eq 0 ]; then
    echo "FATAL: no box in tmp/r2_my_instances.txt is listable under label prefix" >&2
    echo "       '${LABEL_PREFIX}' (see the NOTE lines above for why)." >&2
    exit 1
  fi
  echo "fleet: ${n} box(es) resolved from tmp/r2_my_instances.txt"
}

# A backgrounded per box job loses its exit status unless it is waited on by pid:
# bare `wait` reports success even when a job failed. Every job is recorded and
# reaped individually, so a box that failed is named instead of scrolling past.
# Each job also takes its stdin from /dev/null, because the stage loops read the
# fleet from stdin and a job that read stdin would swallow the remaining boxes.
fail() { echo "$1: $2" >> "${FAIL_FILE}"; echo "!! $1: $2" >&2; }
start_job() { echo "$! $1" >> "${JOBS_FILE}"; }
reap_jobs() {
  while read -r pid label; do
    wait "${pid}" || fail "${label}" "stage exited non-zero"
  done < "${JOBS_FILE}"
  : > "${JOBS_FILE}"
}
finish() {
  n=$(wc -l < "${FAIL_FILE}" | tr -d ' ')
  if [ "${n}" -gt 0 ]; then
    echo >&2
    echo "!! ${n} box(es) FAILED this stage:" >&2
    while read -r line; do echo "     ${line}" >&2; done < "${FAIL_FILE}"
    exit 1
  fi
  echo "all boxes completed this stage"
}

retry() {
  local n=0
  until "$@"; do
    n=$((n + 1))
    [ "$n" -ge 4 ] && return 1
    sleep $((n * 5))
  done
}

# -n on ssh keeps a call inside a while-read loop from swallowing the loop's input.
ssh_to() { retry ssh -n -p "$2" ${SSH_OPTS} "root@$1" "${@:3}"; }

do_setup() {
  while read -r label id host port state; do
    runs="$(dirs_for "${label}")" \
      || { fail "${label}" "this script defines no run directories for that label"; continue; }
    (
      echo "[${label}] pushing code (${id} ${state})"
      ssh_to "$host" "$port" "mkdir -p ${REMOTE_DIR}/models ${REMOTE_DIR}/out/r2/ellicit" || exit 1
      retry rsync -az --timeout=120 --delete -e "ssh ${SSH_OPTS} -p ${port}" \
        src script configs pyproject.toml "root@${host}:${REMOTE_DIR}/" || exit 1
      for d in ${runs}; do
        retry rsync -az --timeout=120 -e "ssh ${SSH_OPTS} -p ${port}" \
          "out/r2/ellicit/${d}/" "root@${host}:${REMOTE_DIR}/out/r2/ellicit/${d}/" || exit 1
      done
      # The image ships its torch inside /venv/main; the system python3 has none,
      # so everything remote must run through that interpreter explicitly.
      echo "[${label}] installing vllm (slow)"
      ssh_to "$host" "$port" "${REMOTE_PY} -m pip -q install vllm tqdm 2>&1 | tail -4; \
        ${REMOTE_PY} -c 'import vllm, torch; print(\"READY vllm\", vllm.__version__, \"torch\", torch.__version__, \"cuda\", torch.cuda.is_available())'"
    ) </dev/null &
    start_job "${label}"
  done < "${FLEET_FILE}"
  reap_jobs
}

do_weights() {
  mkdir -p "${STAGE_DIR}"
  while read -r label id host port state; do
    spec="$(spec_for "${label}")" \
      || { fail "${label}" "this script defines no weights spec for that label"; continue; }
    read -r group target ttag reference rtag <<<"${spec}"
    (
      echo "[${label}] staging ${target} locally"
      uv run python script/remote/stage_gated_weights.py --repo "${target}" \
        --dest "${STAGE_DIR}" || exit 1
      name="${target##*/}"
      retry rsync -az --timeout=300 -e "ssh ${SSH_OPTS} -p ${port}" \
        "${STAGE_DIR}/${name}/" "root@${host}:${REMOTE_DIR}/models/${name}/" || exit 1
      echo "[${label}] fetching blobs on box"
      ssh_to "$host" "$port" "cd ${REMOTE_DIR}/models/${name} && \
        while IFS=\$'\t' read -r p s u; do \
          if [ -f \"\$p\" ] && [ \"\$(stat -c%s \"\$p\")\" = \"\$s\" ]; then continue; fi; \
          echo \"fetch \$p (\$s bytes)\"; \
          curl -fsSL --retry 5 --retry-delay 5 -o \"\$p\" \"\$u\" || exit 1; \
          got=\$(stat -c%s \"\$p\"); \
          [ \"\$got\" = \"\$s\" ] || { echo \"SIZE MISMATCH \$p \$got != \$s\"; exit 1; }; \
        done < _lfs_manifest.tsv && echo '[weights] all blobs present and sized'"
    ) </dev/null &
    start_job "${label}"
  done < "${FLEET_FILE}"
  reap_jobs
}

do_elicit() {
  while read -r label id host port state; do
    spec="$(spec_for "${label}")" \
      || { fail "${label}" "this script defines no sampling spec for that label"; continue; }
    runs="$(dirs_for "${label}")" \
      || { fail "${label}" "this script defines no run directories for that label"; continue; }
    read -r group target ttag reference rtag <<<"${spec}"
    name="${target##*/}"
    (
      # The driver is a real file pushed with the code, so nothing is built by
      # string interpolation and the launcher only passes environment variables.
      ssh_to "$host" "$port" "cd ${REMOTE_DIR} && \
        RUN_DIRS='${runs}' TARGET_PATH='${REMOTE_DIR}/models/${name}' \
        TARGET_TAG='${ttag}' REFERENCE='${reference}' REFERENCE_TAG='${rtag}' \
        REMOTE_DIR='${REMOTE_DIR}' BACKEND='${BACKEND:-transformers}' \
        setsid nohup bash script/remote/r2_elicit_driver.sh >> ${LOG} 2>&1 & \
        sleep 3; pgrep -f r2_elicit_driver >/dev/null && echo '[${label}] launched' || echo '[${label}] LAUNCH FAILED'"
    ) </dev/null &
    start_job "${label}"
  done < "${FLEET_FILE}"
  reap_jobs
}

# status and pull run over every registered box whatever its state: a box that is
# loading or exited still holds data and still has to be accounted for, so it is
# listed and allowed to fail loudly rather than filtered out of the report.
do_status() {
  while read -r label id host port state; do
    echo "########## ${label} (${id}) ${host}:${port} [${state}]"
    ssh_to "$host" "$port" "tail -n 4 ${LOG} 2>/dev/null || echo '(no log)'; \
      echo -n 'rows: '; find ${REMOTE_DIR}/out/r2/ellicit -name 'responses_*.jsonl' -exec wc -l {} + 2>/dev/null | tail -1; \
      nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader" \
      || fail "${label}" "status query failed at ${host}:${port} (state ${state})"
  done < "${FLEET_FILE}"
}

do_pull() {
  while read -r label id host port state; do
    runs="$(dirs_for "${label}")" \
      || { fail "${label}" "this script defines no run directories for that label"; continue; }
    (
      rc=0
      for d in ${runs}; do
        if retry rsync -az --timeout=300 -e "ssh ${SSH_OPTS} -p ${port}" \
             "root@${host}:${REMOTE_DIR}/out/r2/ellicit/${d}/" "out/r2/ellicit/${d}/"; then
          echo "[${label}] pulled ${d}"
        else
          echo "[${label}] !! PULL FAILED ${d} from ${host}:${port} (state ${state})" >&2
          rc=1
        fi
      done
      mkdir -p "out/logs/remote/${label}"
      # A box that never started the driver has no r2.log; that is not a failure,
      # but it is still said out loud rather than swallowed by 2>/dev/null.
      rsync -az --timeout=120 -e "ssh ${SSH_OPTS} -p ${port}" \
        "root@${host}:${LOG}" "out/logs/remote/${label}/r2.log" \
        || echo "[${label}] no ${LOG} pulled" >&2
      exit "${rc}"
    ) </dev/null &
    start_job "${label}"
  done < "${FLEET_FILE}"
  reap_jobs
}

case "${1:?usage: r2_fleet.sh <setup|weights|elicit|status|pull>}" in
  setup) enumerate_fleet; do_setup ;;
  weights) enumerate_fleet; do_weights ;;
  elicit) enumerate_fleet; do_elicit ;;
  status) enumerate_fleet --all-states; do_status ;;
  pull) enumerate_fleet --all-states; do_pull ;;
  *) echo "unknown stage $1"; exit 1 ;;
esac
finish
