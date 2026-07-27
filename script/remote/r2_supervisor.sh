#!/usr/bin/env bash
# Keep every stage 4 unit running until its seat is complete.
#
#   bash script/remote/r2_supervisor.sh            # loop
#   ONESHOT=1 bash script/remote/r2_supervisor.sh  # single reconcile pass
#
# Boxes die, ssh keys fail to provision, drivers exit without a trace, and a
# `pkill` does not always take. Restarting each casualty by hand does not scale
# and it is how two collectors ended up writing one file twice. So the desired
# state is declared once, below, and this loop reconciles reality to it: for each
# unit, if its seat is not complete and no collector is running on its box, start
# one. Exactly one box owns each seat, which is what makes a restart safe.
#
# Nothing here is quiet. Every unreachable box, every restart, every unit that
# cannot be placed is printed with a marker.
set -u
cd "$(dirname "$0")/../.." || exit 1

TARGET_ROWS="${TARGET_ROWS:-5760}"
ONESHOT="${ONESHOT:-0}"
INTERVAL="${INTERVAL:-180}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/idt}"
SSH_OPTS="-F /dev/null -i ${HOME}/.ssh/id_ed25519 -o StrictHostKeyChecking=no \
-o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o ServerAliveInterval=30"

# label | role | model | tag | condition | elicit dir | out dir | staged name
UNITS=(
"idt-r2-organism-a|reference|Qwen/Qwen2.5-7B-Instruct|base_7b|challenge_blind|out/r2/ellicit/challenge_organism_a|out/r2/score/challenge_organism_a|"
"idt-r2-organism-b|reference|Qwen/Qwen2.5-7B-Instruct|base_7b|challenge_blind|out/r2/ellicit/challenge_organism_b|out/r2/score/challenge_organism_b|"
"idt-r2-organism-c3|target|sl-organism-c-7b|organism_c|challenge_blind|out/r2/ellicit/challenge_organism_c|out/r2/score/challenge_organism_c|sl-organism-c-7b"
"idt-r2-c7bpos-base|reference|Qwen/Qwen2.5-7B-Instruct|base_7b|challenge_blind|out/r2/ellicit/challenge_organism_c|out/r2/score/challenge_organism_c__v1|;none,live_deployment;1440"
"idt-r2-c7b-base|reference|Qwen/Qwen2.5-7B-Instruct|base_7b|challenge_blind|out/r2/ellicit/challenge_organism_c|out/r2/score/challenge_organism_c__v2|;committed_supporter;1440"
"idt-r2-c7b-target2|reference|Qwen/Qwen2.5-7B-Instruct|base_7b|challenge_blind|out/r2/ellicit/challenge_organism_c|out/r2/score/challenge_organism_c__v3|;unreviewed_authority;1440"
"idt-r2-cal-typed-base|reference|Qwen/Qwen2.5-1.5B-Instruct|base_1p5b|calibration_typed|out/r2/ellicit/calibration_typed|out/r2/score/calibration_typed|"
"idt-r2-calinf-t2|target|12-mar-gen9-1.5b|gen9_1p5b|calibration_informed|out/r2/ellicit/calibration_informed|out/r2/score/calibration_informed|12-mar-gen9-1.5b"
"idt-r2-cal-inf-base|reference|Qwen/Qwen2.5-1.5B-Instruct|base_1p5b|calibration_informed|out/r2/ellicit/calibration_informed|out/r2/score/calibration_informed|"
"idt-r2-calblind-t2|target|12-mar-gen9-1.5b|gen9_1p5b|calibration_blind|out/r2/ellicit/calibration_blind|out/r2/score/calibration_blind|12-mar-gen9-1.5b"
"idt-r2-cal-blind-target|reference|Qwen/Qwen2.5-1.5B-Instruct|base_1p5b|calibration_blind|out/r2/ellicit/calibration_blind|out/r2/score/calibration_blind|"
)

pass=0
while true; do
  pass=$((pass + 1))
  echo "=== reconcile pass ${pass} $(date -u +%H:%M:%S) ==="
  fleet="$(uv run python script/remote/list_instances.py --all-states 2>/dev/null)"
  if [ -z "${fleet}" ]; then
    echo "  !! fleet could not be resolved; nothing reconciled this pass" >&2
    [ "${ONESHOT}" = "1" ] && exit 1
    sleep "${INTERVAL}"
    continue
  fi

  done_units=0
  for spec in "${UNITS[@]}"; do
    IFS='|' read -r label role model tag cond elic out staged <<<"${spec}"
    # staged may be "name" or ";variants;rows" for a variant-split unit
    variants=""; want="${TARGET_ROWS}"
    case "${staged}" in
      \;*) IFS=';' read -r _ variants want <<<"${staged}"; staged="" ;;
    esac
    want="${want:-${TARGET_ROWS}}"
    line="$(echo "${fleet}" | grep "^${label} " || true)"
    if [ -z "${line}" ]; then
      echo "  !! ${label}: not in the registry, cannot place its seat" >&2
      continue
    fi
    read -r _ id host port state <<<"${line}"
    if ! ssh -n -p "${port}" ${SSH_OPTS} "root@${host}" "true" >/dev/null 2>&1; then
      echo "  !! ${label}: unreachable at ${host}:${port} (state ${state})" >&2
      continue
    fi
    rows=$(ssh -n -p "${port}" ${SSH_OPTS} "root@${host}" \
      "wc -l < ${REMOTE_DIR}/${out}/responses_${tag}.jsonl 2>/dev/null || echo 0" 2>/dev/null | tr -d ' ')
    procs=$(ssh -n -p "${port}" ${SSH_OPTS} "root@${host}" \
      "ps -eo args | grep -c '[c]ollect_and_score.py'" 2>/dev/null | tr -d ' ')
    rows="${rows:-0}"; procs="${procs:-0}"
    if [ "${rows}" -ge "${want}" ]; then
      echo "  ok ${label}: ${rows}/${want} complete"
      done_units=$((done_units + 1))
      continue
    fi
    if [ "${procs}" -gt 0 ]; then
      echo "  ok ${label}: ${rows}/${want} running"
      continue
    fi
    # A 1.5B checkpoint at batch 32 leaves the card at about a quarter used.
    # Batch is sized by the model here rather than left at one default for every
    # seat, because the small model is the one with headroom to spare.
    case "${tag}" in
      *1p5b*) batch=256 ;;
      *) batch=32 ;;
    esac
    echo "  >> ${label}: ${rows}/${want} and NO collector, restarting (batch ${batch})"
    BATCH_SIZE="${batch}" VARIANTS="${variants}" bash script/remote/r2_collect_seat.sh "${label}" "${role}" "${model}" "${tag}" \
      "${cond}" "${elic}" "${out}" "${staged}" > "tmp/sup_${label}.log" 2>&1 \
      && echo "     restarted ${label}" \
      || echo "  !! ${label}: restart FAILED, see tmp/sup_${label}.log" >&2
  done

  echo "  ${done_units}/${#UNITS[@]} units complete"
  [ "${done_units}" -ge "${#UNITS[@]}" ] && { echo "ALL UNITS COMPLETE"; exit 0; }
  [ "${ONESHOT}" = "1" ] && exit 0
  sleep "${INTERVAL}"
done
