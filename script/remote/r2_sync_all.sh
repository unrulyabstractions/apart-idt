#!/usr/bin/env bash
# Pull EVERYTHING off every box I own, then prove nothing was left behind.
#
#   bash script/remote/r2_sync_all.sh          # sync, then report per box
#   VERIFY=1 bash script/remote/r2_sync_all.sh # sync, then run the capture gate
#
# "out/**/*.json" is not "all data": logs, stray scratch files, and anything a
# run wrote outside the directory I expected are data too. So this syncs the
# whole of the remote out tree plus every log, with no extension filter, and the
# gate afterwards sweeps every file on the box and names anything that has no
# byte-identical local copy. The only files held back are the ones a box never
# computes and can only hold a stub of; see LOCAL_ONLY_NAMES below for why that
# is the opposite of losing data.
#
# Boxes come from tmp/r2_my_instances.txt, filtered by the idt-r2- label prefix.
# Another team shares this account and their instances carry no label; nothing
# here can address them.
#
# Nothing fails quietly. A box that cannot be reached, a tree that will not
# transfer, a remote listing that cannot be taken: each is printed as a marked
# failure line, all of them are repeated in a summary at the end, and the script
# exits non-zero. An empty pull that looks like a clean run is the bug this
# script exists to make impossible.
set -u
cd "$(dirname "$0")/../.." || exit 1

LABEL_PREFIX="${LABEL_PREFIX:-idt-r2-}"
REMOTE_DIR=/workspace/idt
VERIFY="${VERIFY:-0}"
SSH_OPTS="-F /dev/null -i ${HOME}/.ssh/id_ed25519 -o StrictHostKeyChecking=no \
-o UserKnownHostsFile=/dev/null -o ConnectTimeout=25 -o ServerAliveInterval=30"

# Fleet enumeration. NEVER `vastai show instances-v1`: that call PAGINATES.
# Measured 2026-07-26 with 29 instances on this account it returned 25 plus a
# next_token, and `--all` returned the same 25. Four of this project's twelve
# boxes were missing from it, and a box missing from the listing is a box whose
# data this sync never pulls while still printing a tidy summary.
#
# list_instances.py resolves every id in tmp/r2_my_instances.txt individually
# and exits non-zero if any one of them cannot be resolved. --all-states is used
# on purpose: a box that is loading or exited still holds data, so it is listed
# and allowed to fail loudly rather than filtered away.
FLEET_FILE="$(mktemp "${TMPDIR:-/tmp}/r2_sync_fleet.XXXXXX")" || exit 1
FAIL_FILE="$(mktemp "${TMPDIR:-/tmp}/r2_sync_fail.XXXXXX")" || exit 1
trap 'rm -f "${FLEET_FILE}" "${FAIL_FILE}"' EXIT

uv run python script/remote/list_instances.py \
  --label-prefix "${LABEL_PREFIX}" --all-states > "${FLEET_FILE}"
rc=$?
if [ "${rc}" -ne 0 ]; then
  echo "FATAL: list_instances.py exited ${rc}. The fleet registered in" >&2
  echo "       tmp/r2_my_instances.txt could not be fully resolved, so this sync" >&2
  echo "       would silently skip whatever it could not see. Nothing pulled." >&2
  exit 1
fi
boxes=$(wc -l < "${FLEET_FILE}" | tr -d ' ')
if [ "${boxes}" -eq 0 ]; then
  echo "FATAL: no box in tmp/r2_my_instances.txt is listable under label prefix" >&2
  echo "       '${LABEL_PREFIX}'." >&2
  exit 1
fi
echo "fleet: ${boxes} box(es) resolved from tmp/r2_my_instances.txt"
echo

# Failures go to a file, not a shell variable. The old loop ran inside a pipeline
# (`fleet | while read`), so its `unreachable=$((unreachable + 1))` happened in a
# SUBSHELL and was thrown away when the loop ended: the script always exited 0,
# even with boxes it never touched. The loop now reads from a file, so no
# subshell is created, and the failure record survives in either case.
mkdir -p out/logs/remote
note_failure() { echo "$1: $2" >> "${FAIL_FILE}"; echo "  !! FAILED: $2" >&2; }

# Artifacts computed LOCALLY are never pulled back off a box. A box samples and
# nothing else: extraction and report building need the Anthropic key, which
# never travels, yet ellicit_principals.py still writes elicitation_report.json
# at the end of every remote run, built from a favored_*.jsonl that is empty
# there. Pulling those two back overwrote all nine per-seed challenge reports on
# 2026-07-26: candidate_principals fell from 24-66 entries to 0 and coverage from
# 800 per seat to 0. `rsync -a` preserves the remote mtime, so the mtimes moved
# BACKWARD and the directory listing looked untouched; nothing announced the
# loss. questions.json is excluded for the same reason: it is frozen locally and
# pushed up, the box reads it and never computes it, and every response on every
# box is keyed to the local copy.
#
# What a box does compute still comes through untouched: responses_*.jsonl,
# prompt_sets.json and verdicts_*.jsonl are all pulled.
#
# An excluded file that exists on a box and not locally is still not lost
# quietly: the capture gate sweeps every remote file, byte for byte, and reports
# it as WOULD BE LOST, which blocks the destroy until a human decides.
LOCAL_ONLY_NAMES="elicitation_report.json favored_*.jsonl questions.json"
set -f  # keep favored_*.jsonl a literal pattern for rsync instead of globbing here
EXCLUDE_ARGS=()
for pattern in ${LOCAL_ONLY_NAMES}; do EXCLUDE_ARGS+=(--exclude="${pattern}"); done
set +f

while read -r label id host port state; do
  echo "########## ${label} (${id}) ${host}:${port} [${state}]"
  if ! ssh -n -p "${port}" ${SSH_OPTS} "root@${host}" "true" >/dev/null 2>&1; then
    note_failure "${label}" "UNREACHABLE at ${host}:${port} (state ${state}), nothing pulled"
    continue
  fi
  # The whole out tree, filtered only by the locally computed artifacts above.
  # -a keeps sizes and times so the gate can compare bytes afterwards. rsync's
  # stderr is shown, not discarded: the reason a transfer failed is the one thing
  # worth keeping. The exclusions are printed every time, because an exclusion
  # nobody sees is the same class of bug as a box nobody pulls.
  echo "  not pulled (computed locally, a box only ever holds a stub): ${LOCAL_ONLY_NAMES}"
  # A box that has produced nothing yet has no out/ tree at all, and that is not
  # a transfer failure. The two cases are told apart by asking first, so that a
  # box with nothing to give is stated plainly while a box that has data and
  # would not give it up stays a loud failure. An ssh error here (255) is neither
  # answer and is itself a failure: not knowing is not the same as knowing there
  # is nothing.
  ssh -n -p "${port}" ${SSH_OPTS} "root@${host}" "test -d ${REMOTE_DIR}/out" \
    >/dev/null 2>&1
  probe=$?
  if [ "${probe}" -eq 0 ]; then
    if rsync -az --timeout=600 "${EXCLUDE_ARGS[@]}" -e "ssh ${SSH_OPTS} -p ${port}" \
         "root@${host}:${REMOTE_DIR}/out/" "out/"; then
      echo "  out/ pulled"
    else
      note_failure "${label}" "out/ pull failed (rsync exit $?)"
    fi
  elif [ "${probe}" -eq 1 ]; then
    echo "  no ${REMOTE_DIR}/out on this box yet; it has produced nothing to pull"
  else
    note_failure "${label}" "could not probe ${REMOTE_DIR}/out (ssh exit ${probe}), so it is unknown whether this box holds data"
  fi
  # Every log, wherever it sits, under a per box directory so two boxes with the
  # same log name do not overwrite each other.
  mkdir -p "out/logs/remote/${label}"
  if rsync -az --timeout=300 -e "ssh ${SSH_OPTS} -p ${port}" \
       --include='*.log' --exclude='*' \
       "root@${host}:${REMOTE_DIR}/" "out/logs/remote/${label}/"; then
    echo "  logs pulled"
  else
    note_failure "${label}" "log pull failed (rsync exit $?)"
  fi
  if remote_files=$(ssh -n -p "${port}" ${SSH_OPTS} "root@${host}" \
       "find ${REMOTE_DIR}/out -type f 2>/dev/null | wc -l"); then
    echo "  remote out files: $(echo "${remote_files}" | tr -d ' ')"
  else
    note_failure "${label}" "could not count remote files, so coverage is unknown"
  fi
done < "${FLEET_FILE}"

echo
echo "=== local totals after sync ==="
for tree in out/r2/ellicit out/r2/promptset out/r2/conjecture out/r2/score out/r2/compare out/logs/remote; do
  n=$(find "${tree}" -type f 2>/dev/null | wc -l | tr -d ' ')
  b=$(find "${tree}" -type f -exec stat -f%z {} + 2>/dev/null | awk '{s+=$1} END {print s+0}')
  echo "  ${tree}: ${n} files, ${b} bytes"
done

if [ "${VERIFY}" = "1" ]; then
  echo
  echo "=== capture gate, per box ==="
  # Read from the file, not a pipeline: inside a pipeline this loop would run in
  # a subshell and every gate failure it recorded would vanish at the `done`.
  # The gate's exit code is read directly from the command, never through a pipe.
  while read -r label id host port state; do
    echo "########## gate ${label} (${id}) [${state}]"
    uv run python script/remote/verify_remote_capture.py \
      --host "${host}" --port "${port}" \
      --map "r2.log=out/logs/remote/${label}/r2.log" \
      --map "r2c.log=out/logs/remote/${label}/r2c.log" \
      --map "r2_collect.log=out/logs/remote/${label}/r2_collect.log" \
      --map "r2_seat.log=out/logs/remote/${label}/r2_seat.log" \
      --pushed-from-local src/ --pushed-from-local script/ \
      --pushed-from-local configs/ --pushed-from-local pyproject.toml
    gate=$?
    echo "  gate exit=${gate}"
    [ "${gate}" -eq 0 ] || note_failure "${label}" "capture gate exited ${gate}"
  done < "${FLEET_FILE}"
fi

failures=$(wc -l < "${FAIL_FILE}" | tr -d ' ')
echo
if [ "${failures}" -gt 0 ]; then
  echo "!! ${failures} FAILURE(S) across ${boxes} box(es). These boxes are NOT" >&2
  echo "!! fully captured and nothing here may be treated as synced:" >&2
  while read -r line; do echo "     ${line}" >&2; done < "${FAIL_FILE}"
  exit 1
fi
echo "all ${boxes} box(es) pulled with no failures"
exit 0
