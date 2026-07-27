#!/usr/bin/env bash
# Run the capture gate on every box this project owns, and destroy a box only if
# its own gate exited 0.
#
#   bash script/remote/gate_and_destroy_boxes.sh check              # gate only
#   bash script/remote/gate_and_destroy_boxes.sh destroy            # destroy on pass
#   LABEL_PREFIX=idt-cal bash script/remote/gate_and_destroy_boxes.sh destroy
#
# Boxes come from tmp/r2_my_instances.txt through script/remote/list_instances.py,
# NEVER from `vastai show instances-v1`: that call PAGINATES. Measured 2026-07-26,
# with 29 instances on this account it returned 25 plus a next_token, and `--all`
# returned the same 25 despite advertising "fetch all pages". Four of this
# project's twelve boxes were absent from it. A gate driven by that listing
# reports the fleet captured while never having looked at part of it, which is
# the exact catastrophe this script exists to prevent. So if any registered id
# fails to resolve, this script does nothing at all: no gate, no destroy.
#
# Boxes of every state are gated, not only running ones. A loading or exited box
# still holds data, and one that cannot be swept fails its own gate loudly.
#
# LABEL_PREFIX no longer selects which boxes are gated: every registered box is,
# so the report is always of the whole fleet. It is now purely a destroy guard,
# so narrowing it can only ever destroy less, never gate less.
#
# Three independent refusals stand between a box and `vastai destroy`:
#   1. its own gate must exit 0, read directly and never through a pipe, because
#      a pipeline reports the last command's status and would turn a failed
#      check into an apparent pass;
#   2. its label must start with LABEL_PREFIX;
#   3. its id must appear in tmp/r2_my_instances.txt, re-read from disk at the
#      moment of destruction. Another team shares this account and runs unlabelled
#      instances; an id I never registered is not mine to destroy.
# The destroy is then confirmed by querying that one instance, because
# `vastai destroy` exits 0 even when its prompt aborts.
set -u
cd "$(dirname "$0")/../.." || exit 1

MODE="${1:?usage: gate_and_destroy_boxes.sh <check|destroy>}"
case "${MODE}" in
  check|destroy) ;;
  *) echo "FATAL: unknown mode '${MODE}'; expected 'check' or 'destroy'" >&2; exit 1 ;;
esac
LABEL_PREFIX="${LABEL_PREFIX:-idt-}"
REGISTRY=tmp/r2_my_instances.txt

FLEET_FILE="$(mktemp "${TMPDIR:-/tmp}/gate_fleet.XXXXXX")" || exit 1
trap 'rm -f "${FLEET_FILE}"' EXIT

uv run python script/remote/list_instances.py --all-states > "${FLEET_FILE}"
rc=$?
if [ "${rc}" -ne 0 ]; then
  echo "FATAL: list_instances.py exited ${rc}. The fleet in ${REGISTRY} could not" >&2
  echo "       be fully resolved, so part of it is invisible to this run. No gate" >&2
  echo "       was run and NOTHING was destroyed." >&2
  exit 1
fi

# Read fresh from disk each time. This is the last line of defence against
# destroying a box that belongs to somebody else.
registered() {
  while read -r rid _rlabel; do
    [ "${rid}" = "$1" ] && return 0
  done < "${REGISTRY}"
  return 1
}

# A single-instance query cannot be truncated by pagination the way the fleet
# listing can. A live instance answers with a payload naming its id; a destroyed
# one makes the CLI fail with no payload at all. An API outage looks exactly like
# a destroyed box, so the question is asked twice and a box is only called gone
# when it stays gone. 0 = still there, 1 = gone, 2 = cannot tell.
instance_alive() {
  local out
  out=$(vastai show instance "$1" --raw 2>/dev/null)
  if [ -z "${out}" ]; then
    sleep 5
    out=$(vastai show instance "$1" --raw 2>/dev/null)
    [ -n "${out}" ] || return 1
  fi
  printf '%s' "${out}" | grep -q "\"id\"[[:space:]]*:[[:space:]]*$1" && return 0
  return 2
}

found=0
overall=0
while read -r label id host port state; do
  [ -n "${id:-}" ] || continue
  found=$((found + 1))
  echo "########## ${id}  ${label}  (${host}:${port}) [${state}]"
  # The full gate output is kept under tmp/: a long WOULD BE LOST list must
  # survive the run, not scroll away in a 20-line tail.
  mkdir -p tmp/gate_logs
  log="tmp/gate_logs/gate_${label}_$(date +%Y%m%d_%H%M%S).log"
  # Every log name any driver on any generation of box writes. A log that is
  # captured but not mapped here shows up as WOULD BE LOST, and a gate that cries
  # wolf is a gate people learn to override, which is worse than no gate.
  uv run python script/remote/verify_remote_capture.py \
    --host "${host}" --port "${port}" \
    --map "calibration.log=out/logs/remote/${label}/calibration.log" \
    --map "collection.log=out/logs/remote/${label}/collection.log" \
    --map "sampling.log=out/logs/remote/${label}/sampling.log" \
    --map "r2.log=out/logs/remote/${label}/r2.log" \
    --map "r2c.log=out/logs/remote/${label}/r2c.log" \
    --map "r2_collect.log=out/logs/remote/${label}/r2_collect.log" \
    --map "r2_seat.log=out/logs/remote/${label}/r2_seat.log" \
    --pushed-from-local src/ \
    --pushed-from-local script/ \
    --pushed-from-local configs/ \
    --pushed-from-local pyproject.toml \
    > "${log}" 2>&1
  gate=$?
  tail -20 "${log}"
  echo "full gate output kept at ${log}"
  echo "gate exit=${gate}"
  if [ "${gate}" -ne 0 ]; then
    echo ">>> ${label}: NOT destroying."
    overall=1
    echo
    continue
  fi
  if [ "${MODE}" != "destroy" ]; then
    echo ">>> ${label}: gate passed (check mode, nothing destroyed)"
    echo
    continue
  fi
  case "${label}" in
    "${LABEL_PREFIX}"*) ;;
    *)
      echo "!! ${label}: label does not start with '${LABEL_PREFIX}'; REFUSING to" >&2
      echo "   destroy instance ${id}." >&2
      overall=1
      echo
      continue
      ;;
  esac
  if ! registered "${id}"; then
    echo "!! ${label}: instance ${id} is not listed in ${REGISTRY}; REFUSING to" >&2
    echo "   destroy it. An id I never registered is not mine to destroy." >&2
    overall=1
    echo
    continue
  fi
  echo ">>> ${label}: gate passed, destroying instance ${id}"
  vastai destroy instance "${id}" --yes
  sleep 5
  instance_alive "${id}"
  alive=$?
  case "${alive}" in
    0) echo "!! ${label}: instance ${id} is STILL LISTED after destroy"; overall=1 ;;
    1) echo ">>> ${label}: instance ${id} confirmed gone" ;;
    *) echo "!! ${label}: could not tell whether ${id} is gone; CHECK IT BY HAND"
       overall=1 ;;
  esac
  echo
done < "${FLEET_FILE}"

if [ "${found}" -eq 0 ]; then
  echo "FATAL: ${REGISTRY} resolved cleanly but produced no box to gate." >&2
  exit 1
fi
echo "${found} box(es) gated in ${MODE} mode; overall exit ${overall}"
exit "${overall}"
