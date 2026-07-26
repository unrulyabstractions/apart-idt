#!/usr/bin/env bash
# Run the capture gate on every box this project owns, and destroy a box only if
# its own gate exited 0.
#
#   bash script/remote/gate_and_destroy_boxes.sh check              # gate only
#   bash script/remote/gate_and_destroy_boxes.sh destroy            # destroy on pass
#   LABEL_PREFIX=idt-cal bash script/remote/gate_and_destroy_boxes.sh destroy
#
# Instances come from the vast API by label prefix, so no id is hardcoded and a
# box belonging to another workstream (no matching label) is never touched.
#
# The gate's exit code is read directly, never through a pipe: a pipeline reports
# the last command's status and would turn a failed check into an apparent pass.
# The destroy is likewise confirmed by asking the API whether the instance is
# gone, because `vastai destroy` exits 0 even when its prompt aborts.
set -u
cd "$(dirname "$0")/../.." || exit 1

MODE="${1:?usage: gate_and_destroy_boxes.sh <check|destroy>}"
LABEL_PREFIX="${LABEL_PREFIX:-idt-}"

boxes() {
  vastai show instances-v1 --raw | python3 -c "
import json, sys
prefix = '${LABEL_PREFIX}'
for i in json.load(sys.stdin).get('instances', []):
    label = i.get('label') or ''
    if label.startswith(prefix) and i.get('actual_status') == 'running':
        print(i['id'], i.get('ssh_host'), i.get('ssh_port'), label)
"
}

found=0
overall=0
while read -r id host port label; do
  [ -n "${id:-}" ] || continue
  found=$((found + 1))
  echo "########## ${id}  ${label}  (${host}:${port})"
  log="$(mktemp)"
  uv run python script/remote/verify_remote_capture.py \
    --host "${host}" --port "${port}" \
    --map "calibration.log=out/logs/remote/${label}/calibration.log" \
    --map "collection.log=out/logs/remote/${label}/collection.log" \
    --pushed-from-local src/ \
    --pushed-from-local script/ \
    --pushed-from-local configs/ \
    --pushed-from-local pyproject.toml \
    > "${log}" 2>&1
  gate=$?
  tail -20 "${log}"
  rm -f "${log}"
  echo "gate exit=${gate}"
  if [ "${gate}" -ne 0 ]; then
    echo ">>> ${label}: NOT destroying."
    overall=1
    continue
  fi
  if [ "${MODE}" = "destroy" ]; then
    echo ">>> ${label}: gate passed, destroying instance ${id}"
    vastai destroy instance "${id}" --yes
    sleep 5
    if vastai show instances-v1 --raw 2>/dev/null | grep -q "\"id\": *${id}\b"; then
      echo "!! ${label}: instance ${id} still listed after destroy"
      overall=1
    else
      echo ">>> ${label}: instance ${id} confirmed gone"
    fi
  else
    echo ">>> ${label}: gate passed (check mode, nothing destroyed)"
  fi
  echo
done < <(boxes)

[ "${found}" -gt 0 ] || echo "no running instance whose label starts with '${LABEL_PREFIX}'"
exit "${overall}"
