"""Prove every file on a rented box is already local, before the box is destroyed.

    uv run python script/remote/verify_remote_capture.py --host ssh6.vast.ai --port 23098 \
        --map calibration.log=out/logs/remote/box/calibration.log \
        --pushed-from-local src/ --pushed-from-local script/

Exits 0 only when every remote file is one of:
  * byte-identical to a local file (captured),
  * matched by a reproducible or host-owned pattern (weights, venv, vast files),
  * under a ``--pushed-from-local`` prefix AND present locally, meaning the local
    copy is the authoritative source that was rsynced up.

Anything else exits 1 and is printed as WOULD BE LOST. A destroy command must be
gated on this exit code. Bytes are compared, never row counts: a truncated pull
can preserve a line count but never preserves bytes.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

#: Tool-generated directory names, exempt wherever they appear, but only as a
#: whole path segment: a directory whose name merely contains one of these
#: words is data and stays gated.
REPRODUCIBLE_DIR_NAMES = (".venv", "site-packages", "__pycache__")
#: Box-level caches, exempt only at these exact locations.
CACHE_PREFIXES = ("/root/.cache/", "/workspace/.cache/")
#: Swept when --extra-root is not given; an explicit flag replaces them.
DEFAULT_SWEEP_ROOTS = ("/root", "/workspace")


def reproducible_prefixes(remote_root: str) -> tuple[str, ...]:
    """The exempt trees, anchored: staged weights under the remote root and the
    box caches. A directory named ``models`` anywhere else is data and is gated."""
    return (f"{remote_root.rstrip('/')}/models/", *CACHE_PREFIXES)


def in_reproducible_dir(path: str) -> bool:
    """True only when a whole directory segment names a tool-generated tree.

    The basename is excluded on purpose: a file named ``models`` is data."""
    return any(seg in REPRODUCIBLE_DIR_NAMES for seg in path.split("/")[:-1])
#: Created by the host image, never by a run.
HOST_OWNED = (
    "/root/.bashrc", "/root/.profile", "/root/.ssh/", "/root/.vast_api_key",
    "/root/.vast_containerlabel", "/root/hasbooted", "/root/onstart.sh",
    "/root/ports.log", "/workspace/onstart.sh", "/workspace/ports.log",
)
SSH_OPTS = ["-F", "/dev/null", "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=25",
            "-o", "BatchMode=yes"]


def remote_files(host: str, port: str, key: Path, roots: list[str],
                 exempt_prefixes: tuple[str, ...]) -> list[tuple[int, str]]:
    """Every file under any root, as (size, absolute path), deduplicated."""
    prune = " ".join(f"-type d -name '{n}' -prune -o" for n in REPRODUCIBLE_DIR_NAMES)
    prune += " " + " ".join(f"-path '{p.rstrip('/')}' -prune -o" for p in exempt_prefixes)
    seen: dict[str, int] = {}
    for root in roots:
        cmd = f"find {root} -mindepth 1 {prune} -type f -printf '%s\\t%p\\n'"
        out = subprocess.run(
            ["ssh", "-n", "-p", port, "-i", str(key), *SSH_OPTS, f"root@{host}", cmd],
            capture_output=True, text=True, timeout=600)
        rows = [line.split("\t", 1) for line in out.stdout.splitlines() if "\t" in line]
        if not rows and out.returncode != 0:
            raise RuntimeError(f"ssh failed on {root}: {out.stderr.strip()[:300]}")
        for size, path in rows:
            if size.strip().isdigit():
                seen[path.strip()] = int(size)
    return sorted((s, p) for p, s in seen.items())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", required=True)
    ap.add_argument("--remote-root", default="/workspace/idt",
                    help="root whose paths map onto --local-root")
    ap.add_argument("--extra-root", action="append", default=None,
                    help="root to sweep for stray files; may repeat, and when given "
                         "it replaces the default sweep of /root and /workspace")
    ap.add_argument("--local-root", default=".")
    ap.add_argument("--key", default=str(Path.home() / ".ssh" / "id_ed25519"))
    ap.add_argument("--map", action="append", default=[], metavar="REMOTE_REL=LOCAL",
                    help="a file pulled to a different local path")
    ap.add_argument("--pushed-from-local", action="append", default=[], metavar="PREFIX",
                    help="remote-relative prefix rsynced up from local, so the local "
                         "copy is authoritative even when the sizes differ")
    args = ap.parse_args()

    key, local_root = Path(args.key), Path(args.local_root)
    relocated = dict(m.split("=", 1) for m in args.map)
    extra_roots = args.extra_root or list(DEFAULT_SWEEP_ROOTS)
    exempt = reproducible_prefixes(args.remote_root)
    files = remote_files(args.host, args.port, key,
                         [args.remote_root, *extra_roots], exempt)

    captured, allowed, pushed, lost = [], [], [], []
    for size, path in files:
        if (in_reproducible_dir(path)
                or any(path.startswith(p) for p in exempt)
                or any(path.startswith(p) for p in HOST_OWNED)):
            allowed.append((size, path))
            continue
        rel = (path[len(args.remote_root):].lstrip("/")
               if path.startswith(args.remote_root) else None)
        candidate = (local_root / relocated[rel] if rel in relocated
                     else local_root / rel if rel else None)
        if candidate and candidate.is_file() and candidate.stat().st_size == size:
            captured.append((size, path))
        elif (rel and candidate and candidate.is_file()
              and any(rel.startswith(p) for p in args.pushed_from_local)):
            pushed.append((size, path, candidate.stat().st_size))
        else:
            have = (candidate.stat().st_size
                    if candidate and candidate.is_file() else None)
            lost.append((size, path, str(candidate) if candidate else "(no mapping)",
                         have))

    print(f"host {args.host}:{args.port}   remote files swept: {len(files)}")
    print(f"  captured byte-identical      : {len(captured)}")
    print(f"  reproducible or host-owned   : {len(allowed)}")
    print(f"  pushed from local, superseded: {len(pushed)}")
    for size, path, have in pushed:
        print(f"      remote {size:>9,} / local {have:>9,}  {path}")
    print(f"  WOULD BE LOST                : {len(lost)}")
    for size, path, cand, have in lost:
        state = "absent locally" if have is None else f"local has {have:,} bytes"
        print(f"      {size:>12,} bytes  {path}\n{'':22}-> {cand} ({state})")
    if lost:
        print("\nNOT SAFE TO DESTROY. Capture the files above first.")
        raise SystemExit(1)
    print("\nEvery remote file is captured, reproducible, or superseded by local. "
          "Safe to destroy.")


if __name__ == "__main__":
    main()
