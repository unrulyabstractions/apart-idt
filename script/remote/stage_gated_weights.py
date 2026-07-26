"""Stage gated HuggingFace weights for a rented box without shipping the token.

    uv run python script/remote/stage_gated_weights.py --repo Alamerton/sl-organism-a-7b \
        --dest tmp/weights_stage

The organism checkpoints are gated, so a plain download needs an authenticated
token. Copying our token to a rented third-party host would expose write scopes
we never need there, so instead this runs locally: it downloads the small repo
files with the token, and for each large (LFS) file it resolves only the
short-lived pre-signed CDN redirect, which fetches with no credential at all.
The signed URLs land in ``_lfs_manifest.tsv`` next to the small files; the box
pulls them itself. Signatures expire in about an hour, so stage immediately
before the remote download, not ahead of time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

HUB = "https://huggingface.co"
MANIFEST_NAME = "_lfs_manifest.tsv"


def _request(url: str, token: str, method: str = "GET", redirect: bool = True):
    request = urllib.request.Request(url, method=method)
    request.add_header("Authorization", f"Bearer {token}")

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *_args, **_kwargs):
            return None

    opener = urllib.request.build_opener(
        *([] if redirect else [_NoRedirect]))
    try:
        return opener.open(request, timeout=120)
    except urllib.error.HTTPError as err:
        if not redirect and err.code in (301, 302, 303, 307, 308):
            return err
        raise


def _repo_tree(repo: str, token: str) -> list[dict]:
    with _request(f"{HUB}/api/models/{repo}/tree/main?recursive=1", token) as response:
        return json.loads(response.read())


def _repo_commit(repo: str, token: str) -> str:
    with _request(f"{HUB}/api/models/{repo}", token) as response:
        return json.loads(response.read()).get("sha", "")


def _signed_url(repo: str, path: str, token: str) -> str:
    """The pre-signed CDN URL an LFS file redirects to; needs no auth to fetch."""
    response = _request(f"{HUB}/{repo}/resolve/main/{path}", token, redirect=False)
    location = response.headers.get("Location")
    if not location:
        raise RuntimeError(f"{repo}/{path}: expected an LFS redirect, got none")
    return location


def _fetch_small(repo: str, path: str, token: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with _request(f"{HUB}/{repo}/resolve/main/{path}", token) as response:
        target.write_bytes(response.read())


def stage_repo(repo: str, dest_root: Path, token: str) -> dict:
    """Download the small files and resolve the LFS URLs for one gated repo."""
    dest = dest_root / repo.split("/")[-1]
    dest.mkdir(parents=True, exist_ok=True)
    commit = _repo_commit(repo, token)
    small, lfs_rows = 0, []

    for entry in _repo_tree(repo, token):
        if entry.get("type") != "file":
            continue
        path, size = entry["path"], int(entry["size"])
        if entry.get("lfs"):
            lfs_rows.append((path, size, _signed_url(repo, path, token)))
            continue
        target = dest / path
        if target.exists() and target.stat().st_size == size:
            small += 1
            continue
        _fetch_small(repo, path, token, target)
        small += 1

    (dest / MANIFEST_NAME).write_text(
        "".join(f"{p}\t{s}\t{u}\n" for p, s, u in lfs_rows), encoding="utf-8")
    (dest / "_provenance.json").write_text(json.dumps(
        {"repo_id": repo, "commit": commit,
         "lfs_files": [{"path": p, "size": s} for p, s, _ in lfs_rows]},
        indent=2, sort_keys=True), encoding="utf-8")
    return {"repo": repo, "dir": str(dest), "commit": commit,
            "small_files": small, "lfs_files": len(lfs_rows),
            "lfs_bytes": sum(s for _, s, _ in lfs_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", action="append", required=True,
                        help="gated repo id; repeatable")
    parser.add_argument("--dest", type=Path, default=Path("tmp/weights_stage"))
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set; gated repos cannot be staged without it")

    for repo in args.repo:
        summary = stage_repo(repo, args.dest, token)
        print(f"{summary['repo']} -> {summary['dir']} "
              f"commit={summary['commit'][:12]} "
              f"small={summary['small_files']} lfs={summary['lfs_files']} "
              f"({summary['lfs_bytes'] / 1e9:.1f} GB signed)", flush=True)


if __name__ == "__main__":
    main()
