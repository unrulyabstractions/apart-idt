#!/usr/bin/env bash
# Stage 5, overlapped with stage 4: pull whatever the boxes have produced, score
# it locally, repeat. The judge resumes on (prompt_id, sample, judge, level), so
# scoring a partial responses file is safe and the next pass picks up the rest.
#
#   bash script/pipeline/score_r2_incremental.sh          # loop until complete
#   ONESHOT=1 bash script/pipeline/score_r2_incremental.sh # one pass
#
# Concurrency is deliberately modest. Measured on this account: one process at
# 24 workers scores about 121 rows per minute, three processes about 192 in
# total, and a single process at 120 workers collapses to 12 because the extra
# concurrency buys nothing and provokes retry backoff. So the run uses a few
# processes of a few dozen workers each, not one large pool.
set -u
cd "$(dirname "$0")/../.." || exit 1

# One scorer at a time. Two passes would each read their resume set at start and
# then write the same verdict rows twice, which no downstream reader deduplicates.
LOCK="tmp/score_r2.lock"
mkdir -p tmp
if ! mkdir "${LOCK}" 2>/dev/null; then
  echo "another scorer holds ${LOCK}; exiting"
  exit 0
fi
trap 'rmdir "${LOCK}" 2>/dev/null' EXIT INT TERM

WORKERS="${WORKERS:-20}"
MAX_PROCS="${MAX_PROCS:-4}"
ONESHOT="${ONESHOT:-0}"
SSH_OPTS="-F /dev/null -i ${HOME}/.ssh/id_ed25519 -o StrictHostKeyChecking=no \
-o UserKnownHostsFile=/dev/null -o ConnectTimeout=25 -o ServerAliveInterval=30"

# run dir : condition : judge levels. The condition's own level runs first,
# because that is the level every headline number is read at; the lower levels
# are the affordance comparison and are backfilled afterwards.
RUNS=(
  "challenge_organism_a:challenge_blind:2"
  "challenge_organism_b:challenge_blind:2"
  "challenge_organism_c:challenge_blind:2"
  "calibration_typed:calibration_typed:2"
  "calibration_informed:calibration_informed:3"
  "calibration_blind:calibration_blind:2"
)

pull_all() {
  # Boxes come from the registry via list_instances.py, NEVER from
  # `vastai show instances-v1`: that call paginates and silently omitted four of
  # twelve boxes when this account held 29 instances. A box missing from the
  # listing is a box whose responses this loop never scores.
  local fleet
  fleet="$(uv run python script/remote/list_instances.py --all-states 2>/dev/null)"
  if [ -z "${fleet}" ]; then
    echo "  !! could not resolve the fleet; pulled nothing this pass" >&2
    return 1
  fi
  # Each box lands in its own staging directory first. Two boxes can hold the
  # same file name for one run (a stale copy on a box that no longer owns that
  # seat), and pulling both straight into out/ makes the second overwrite the
  # first, which shows up later as verdicts outnumbering responses.
  local staged="tmp/pull_stage"
  while read -r label id host port state; do
    [ -n "${label:-}" ] || continue
    mkdir -p "${staged}/${label}"
    rsync -az --timeout=300 -e "ssh ${SSH_OPTS} -p ${port}" \
      "root@${host}:/workspace/idt/out/r2/score/" "${staged}/${label}/" 2>/dev/null \
      && echo "  staged ${label}" || echo "  !! ${label} pull failed" >&2
  done <<< "${fleet}"
  uv run python script/pipeline/merge_pulled_responses.py "${staged}" out/r2/score
}

score_one() {
  local dir="$1" condition="$2" level="$3"
  [ -d "out/r2/score/${dir}" ] || return 0
  ls "out/r2/score/${dir}"/responses_*.jsonl >/dev/null 2>&1 || return 0
  uv run python script/pipeline/collect_and_score.py \
    --condition "${condition}" \
    --target skipped --target-tag "$(_tags "${dir}" target)" \
    --reference skipped --reference-tag "$(_tags "${dir}" reference)" \
    --elicit-dir "$(_elicit "${dir}")" \
    --out-dir "out/r2/score/${dir}" \
    --promptset-dir "out/r2/promptset/${condition}" \
    --conjecture-dir "out/r2/conjecture/${condition}" \
    --judge-levels "${level}" --judge-workers "${WORKERS}" \
    --skip-sample >> "tmp/score_${dir}_L${level}.log" 2>&1
}

_tags() {
  case "$1" in
    challenge_organism_a) [ "$2" = target ] && echo organism_a || echo base_7b ;;
    challenge_organism_b) [ "$2" = target ] && echo organism_b || echo base_7b ;;
    challenge_organism_c) [ "$2" = target ] && echo organism_c || echo base_7b ;;
    *) [ "$2" = target ] && echo gen9_1p5b || echo base_1p5b ;;
  esac
}

_elicit() {
  case "$1" in
    challenge_organism_*) echo "out/r2/ellicit/$1" ;;
    *) echo "out/r2/ellicit/$1" ;;
  esac
}

pass=0
while true; do
  pass=$((pass + 1))
  echo "=== pass ${pass} $(date -u +%H:%M:%S) ==="
  pull_all
  running=0
  for spec in "${RUNS[@]}"; do
    dir="${spec%%:*}"; rest="${spec#*:}"
    condition="${rest%%:*}"; level="${rest##*:}"
    score_one "${dir}" "${condition}" "${level}" &
    running=$((running + 1))
    if [ "${running}" -ge "${MAX_PROCS}" ]; then wait; running=0; fi
  done
  wait
  echo "--- verdict totals ---"
  for spec in "${RUNS[@]}"; do
    dir="${spec%%:*}"
    n=$(cat "out/r2/score/${dir}"/verdicts_*.jsonl 2>/dev/null | wc -l | tr -d ' ')
    r=$(cat "out/r2/score/${dir}"/responses_*.jsonl 2>/dev/null | wc -l | tr -d ' ')
    echo "  ${dir}: ${n} verdict rows of ${r} responses"
  done
  [ "${ONESHOT}" = "1" ] && break
  sleep 120
done
