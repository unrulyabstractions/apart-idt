"""Status of the AuditBench collection on the rented box.

    uv run python script/remote/auditbench_run_status.py

Reports rows written per arm, GPU use, and whether the run finished or died.
Quoting a remote shell pipeline through ssh from a local shell is how status
checks end up silently reporting nothing, so the remote side is a single quoted
command here and every field is parsed rather than eyeballed.
"""

from __future__ import annotations

import argparse
import subprocess

REMOTE_DIR = "/workspace/idt"
RUN = "out/auditbench/responses/contextual_optimism"

#: One command, so one round trip and one place to change the paths.
PROBE = f"""
cd {REMOTE_DIR} || exit 1
echo "TARGET_ROWS=$(wc -l < {RUN}/responses_target.jsonl 2>/dev/null || echo 0)"
echo "BASE_ROWS=$(wc -l < {RUN}/responses_base.jsonl 2>/dev/null || echo 0)"
echo "FINISHED=$(grep -ac COLLECT_EXIT collect.log 2>/dev/null || echo 0)"
echo "EXITLINE=$(grep -a COLLECT_EXIT collect.log 2>/dev/null | tail -1)"
echo "TRACEBACKS=$(grep -ac Traceback collect.log 2>/dev/null || echo 0)"
echo "OOM=$(grep -ac 'out of memory' collect.log 2>/dev/null || echo 0)"
echo "ALIVE=$(pgrep -fc collect_auditbench || echo 0)"
echo "GPUUTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | paste -sd, -)"
echo "PROGRESS=$(tail -c 4000 collect.log | tr '\\r' '\\n' | grep -a 'sampling:' | tail -1 | cut -c1-90)"
"""


def probe(host: str, port: str, key: str) -> dict[str, str]:
    proc = subprocess.run(
        ["ssh", "-F", "/dev/null", "-i", key, "-o", "IdentitiesOnly=yes",
         "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
         "-o", "ConnectTimeout=25", "-p", port, f"root@{host}", PROBE],
        capture_output=True, text=True)
    fields = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            key_, _, value = line.partition("=")
            fields[key_.strip()] = value.strip()
    if not fields:
        fields["ERROR"] = (proc.stderr or "no output").strip()[:200]
    return fields


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="ssh9.vast.ai")
    ap.add_argument("--port", default="24488")
    ap.add_argument("--key", default="/Users/unrulyabstractions/.ssh/id_ed25519")
    ap.add_argument("--expected-per-arm", type=int, default=3456)
    args = ap.parse_args()

    fields = probe(args.host, args.port, args.key)
    if "ERROR" in fields:
        print(f"could not reach the box: {fields['ERROR']}")
        raise SystemExit(2)

    target = int(fields.get("TARGET_ROWS", 0) or 0)
    base = int(fields.get("BASE_ROWS", 0) or 0)
    total = target + base
    want = args.expected_per_arm * 2
    print(f"target arm : {target}/{args.expected_per_arm}")
    print(f"base arm   : {base}/{args.expected_per_arm}")
    print(f"total      : {total}/{want}  ({100 * total / want:.1f}%)")
    print(f"alive      : {fields.get('ALIVE')}   gpu% : {fields.get('GPUUTIL')}")
    print(f"tracebacks : {fields.get('TRACEBACKS')}   oom : {fields.get('OOM')}")
    print(f"finished   : {fields.get('FINISHED')}  {fields.get('EXITLINE', '')}")
    if fields.get("PROGRESS"):
        print(f"progress   : {fields['PROGRESS']}")


if __name__ == "__main__":
    main()
