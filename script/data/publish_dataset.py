"""Publish one local output tree to the project's private HF dataset repo.

Uploads every file under ``--out-root`` to ``<path-prefix>/`` in the dataset
repo. Any path containing a ``models`` component is refused (organism weights
are third-party checkpoints and are never redistributed), and macOS junk files
are skipped. The script is idempotent: a file whose bytes already match the
remote copy is not re-sent. Change detection compares the git blob sha1 for
regular files and the sha256 for LFS files, with a size shortcut first.

The repo is created private if it does not exist. This script never changes
the visibility of an existing repo.

Usage:
    uv run python script/data/publish_dataset.py \
        --out-root archive/r1_20260726/out --path-prefix r1
    uv run python script/data/publish_dataset.py --out-root out/r2 --path-prefix r2
    uv run python script/data/publish_dataset.py --out-root configs --path-prefix configs
    uv run python script/data/publish_dataset.py --card script/data/dataset_card_readme.md
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

DEFAULT_REPO_ID = "unrulyabstractions/apart-idt-data"
SKIP_BASENAMES = {".DS_Store"}
FORBIDDEN_DIR_COMPONENT = "models"
CHUNK = 1 << 20

#: Directories that are local scratch, not results. ``box_capture`` is the
#: pre-teardown sweep of the rented machines: it is a second copy of data already
#: published under its own stage, plus superseded versions of this repo's own
#: code, so publishing it would duplicate the dataset and ship stale source.
#: A ``__v`` directory is a seat split across machines whose rows were merged
#: back into the parent run, so publishing it double-counts responses.
SKIP_DIR_PARTS = ("box_capture",)
SKIP_DIR_SUFFIX_MARKER = "__v"

#: Suffixes of working files the pipeline leaves beside a result. A ``.preduped``
#: sidecar is the pre-deduplication state from the two-writer collision, kept
#: locally as evidence; it is not what any number was computed from. ``.merging``
#: is a half-written temporary and ``.lock`` is a mutex.
SKIP_SUFFIXES = (".preduped", ".predupe", ".merging", ".lock")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-root", type=Path, help="local tree to upload, e.g. out/r2")
    parser.add_argument("--path-prefix", help="folder inside the repo, e.g. r2")
    parser.add_argument("--card", type=Path, help="local file to upload as README.md")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--dry-run", action="store_true", help="report, upload nothing")
    parser.add_argument("--include-all", action="store_true",
                        help="upload working copies too: box_capture, __v seat splits, and "
                             "the .preduped/.merging/.lock sidecars. Model weights are still "
                             "refused.")
    parser.add_argument("--exclude", action="append", default=[],
                        help="skip a path relative to --out-root; repeatable")
    args = parser.parse_args()
    if (args.out_root is None) != (args.path_prefix is None):
        parser.error("--out-root and --path-prefix must be given together")
    if args.out_root is None and args.card is None:
        parser.error("nothing to do: give --out-root/--path-prefix or --card")
    return args


def sha256_hex(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1(b"blob %d\0" % path.stat().st_size)
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def collect_local(root: Path, prefix: str, include_all: bool = False,
                  exclude: tuple[str, ...] = ()) -> tuple[list[tuple[Path, str]], int]:
    """Return (local file, repo path) pairs under root, and the junk-skip count.

    ``include_all`` keeps the working copies the default run leaves behind: the
    pre-teardown box sweeps, the ``__v`` seat splits whose rows were merged into
    their parent run, and the sidecar files. They duplicate published data rather
    than adding to it, so they are off by default and asked for explicitly.
    Model weights are refused either way.
    """
    pairs, junk = [], 0
    excluded = tuple(e.strip("/") for e in exclude)
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root)
        if FORBIDDEN_DIR_COMPONENT in rel.parts or FORBIDDEN_DIR_COMPONENT in prefix.split("/"):
            sys.exit(f"refusing to upload a models/ path: {path}")
        if path.name in SKIP_BASENAMES:
            junk += 1
            continue
        posix = rel.as_posix()
        if any(posix == e or posix.startswith(e + "/") for e in excluded):
            junk += 1
            continue
        if not include_all:
            if any(part in SKIP_DIR_PARTS or part.endswith(SKIP_DIR_SUFFIX_MARKER)
                   or SKIP_DIR_SUFFIX_MARKER in part for part in rel.parts[:-1]):
                junk += 1
                continue
            if path.name.endswith(SKIP_SUFFIXES):
                junk += 1
                continue
        pairs.append((path, f"{prefix}/{rel.as_posix()}"))
    if not pairs:
        sys.exit(f"no files found under {root}")
    return pairs, junk


def remote_index(api: HfApi, repo_id: str) -> dict[str, object]:
    entries = api.list_repo_tree(repo_id, repo_type="dataset", recursive=True)
    return {e.path: e for e in entries if hasattr(e, "blob_id")}


def matches_remote(path: Path, entry: object) -> bool:
    if entry is None or entry.size != path.stat().st_size:
        return False
    if entry.lfs is not None:
        return entry.lfs.sha256 == sha256_hex(path)
    return entry.blob_id == git_blob_sha1(path)


def main() -> None:
    args = parse_args()
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", private=True, exist_ok=True)
    print(f"repo: datasets/{args.repo_id} (created private if new)")

    pairs: list[tuple[Path, str]] = []
    junk = 0
    if args.out_root is not None:
        pairs, junk = collect_local(args.out_root, args.path_prefix.strip("/"),
                                    include_all=args.include_all,
                                    exclude=tuple(args.exclude))
    if args.card is not None:
        pairs.append((args.card, "README.md"))

    remote = remote_index(api, args.repo_id)
    to_send = [(p, rp) for p, rp in pairs if not matches_remote(p, remote.get(rp))]
    unchanged = len(pairs) - len(to_send)

    total = 0
    for path, repo_path in to_send:
        size = path.stat().st_size
        total += size
        verb = "would upload" if args.dry_run else "upload"
        print(f"{verb}  {repo_path}  ({size} bytes)")
    print(f"{len(to_send)} files to upload ({total} bytes), "
          f"{unchanged} unchanged, {junk} junk files skipped")

    if args.dry_run or not to_send:
        return
    operations = [CommitOperationAdd(path_in_repo=rp, path_or_fileobj=str(p))
                  for p, rp in to_send]
    label = args.path_prefix or "README.md"
    api.create_commit(repo_id=args.repo_id, repo_type="dataset", operations=operations,
                      commit_message=f"Upload {label}: {len(to_send)} files")
    print(f"committed {len(to_send)} files ({total} bytes) to datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
