"""List my boxes from the registry, and fail loudly if any cannot be accounted for.

    uv run python script/remote/list_instances.py            # running boxes
    uv run python script/remote/list_instances.py --all-states
    uv run python script/remote/list_instances.py --json

Prints ``label id host port state`` per box, one per line.

Why this exists. ``vastai show instances-v1`` paginates: with 29 instances on the
account it returned 25 and a ``next_token``, and ``--all`` returned 25 as well.
Four of my twelve boxes were missing from it. Every script here used to select
boxes from that call, including the capture gate that decides whether a box may
be destroyed, so a box could have been skipped by the sync and then reported as
captured. Nothing may be skipped quietly.

So the registry file is the source of truth, not the listing. Every id in it is
queried individually, and an id that cannot be resolved is a hard error with a
non-zero exit, never an omission from the output. Every box this command declines
to print is named on stderr too, whether it was held back by ``--label-prefix``
or by the running-only default: a quiet omission is the failure mode this file
exists to remove, so its own filtering may not reintroduce it.

Exit codes: 0 all accounted for, 2 a registered id would not resolve, 3 a box
carries my label but is in nobody's registry, 4 the account listing could not be
paged to the end so 3 cannot be ruled out. Callers abort on any of them.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REGISTRY = Path("tmp/r2_my_instances.txt")
LABEL_PREFIX = "idt-r2-"


def note(message: str) -> None:
    """Announce an exclusion. A box dropped in silence is a box nobody pulls."""
    print(f"NOTE: {message}", file=sys.stderr)


def _run(args: list[str]) -> dict | list | None:
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        return None


def registry() -> list[tuple[str, str]]:
    if not REGISTRY.exists():
        sys.exit(f"FATAL: {REGISTRY} is missing; without it no box may be addressed")
    rows = []
    for line in REGISTRY.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            rows.append((parts[0], parts[1]))
    if not rows:
        sys.exit(f"FATAL: {REGISTRY} is empty")
    return rows


def endpoint(info: dict) -> tuple[str, str]:
    """Prefer the direct public endpoint; the ssh proxy drops transfers."""
    mapped = (info.get("ports") or {}).get("22/tcp") or []
    host, port = info.get("ssh_host"), info.get("ssh_port")
    if info.get("public_ipaddr") and mapped:
        host, port = info["public_ipaddr"].strip(), mapped[0]["HostPort"]
    return str(host), str(port)


def labelled_boxes() -> tuple[set[str], bool]:
    """Every id the account listing shows carrying the project label.

    Returns ``(ids, complete)``. This is the one call here that pages, and it is
    followed to the end by hand because ``--all`` does not do it: measured with
    29 instances, ``--all`` still returned the first 25 and a ``next_token``. A
    page that fails to load leaves the sweep short, which would hide a box of
    mine that nobody registered, so the caller is told instead of being handed a
    partial answer that looks whole.
    """
    seen: set[str] = set()
    token = None
    for _ in range(50):
        args = ["vastai", "show", "instances-v1", "--raw"]
        if token:
            args += ["--next-token", token]
        payload = _run(args)
        if not isinstance(payload, dict):
            return seen, False
        for inst in payload.get("instances", []):
            if (inst.get("label") or "").startswith(LABEL_PREFIX):
                seen.add(str(inst["id"]))
        token = payload.get("next_token")
        if not token:
            return seen, True
    return seen, False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all-states", action="store_true",
                    help="include boxes that are not running")
    ap.add_argument("--label-prefix", default="",
                    help="list only registered boxes whose label starts with this; "
                         "every box it holds back is named on stderr")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rows, unresolved, out = registry(), [], []
    for instance_id, label in rows:
        if not label.startswith(args.label_prefix):
            note(f"registered box {label} {instance_id} does not match label prefix "
                 f"'{args.label_prefix}'; not listed")
            continue
        info = _run(["vastai", "show", "instance", instance_id, "--raw"])
        if not isinstance(info, dict) or "id" not in info:
            unresolved.append((instance_id, label))
            continue
        host, port = endpoint(info)
        state = str(info.get("actual_status"))
        if state != "running" and not args.all_states:
            note(f"registered box {label} {instance_id} is '{state}', not running; "
                 f"not listed (pass --all-states to include it)")
            continue
        out.append({"label": label, "id": instance_id, "host": host,
                    "port": port, "state": state})

    # A box wearing my label that nobody registered is as dangerous as a lost one.
    labelled, swept_fully = labelled_boxes()
    unregistered = labelled - {i for i, _ in rows}
    for stray in sorted(unregistered):
        print(f"WARNING: instance {stray} carries the {LABEL_PREFIX} label but is "
              f"not in {REGISTRY}; add it or destroy it deliberately", file=sys.stderr)

    if args.json:
        print(json.dumps(out, indent=2))
    else:
        for row in out:
            print(row["label"], row["id"], row["host"], row["port"], row["state"])

    if unresolved:
        for instance_id, label in unresolved:
            print(f"FATAL: registered instance {instance_id} ({label}) could not be "
                  f"resolved; refusing to report a partial fleet", file=sys.stderr)
        return 2
    if not swept_fully:
        print("FATAL: the account listing could not be paged through to the end, so "
              "a box of mine that nobody registered may be missing from this check; "
              "refusing to call the fleet accounted for", file=sys.stderr)
        return 4
    if unregistered:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
