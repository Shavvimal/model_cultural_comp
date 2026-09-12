"""Verify, package or install the frozen response archive, separately from Git.

The checked-in manifest is the allowlist. This tool never downloads inputs,
executes pickle files, collects responses, or packages licensed survey records.
"""

import argparse
import gzip
import hashlib
import io
import json
import re
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/reproduction-data.json"
MANIFEST_NAME = "REPRODUCTION_DATA_MANIFEST.json"
COLLECTIONS = ("collection", "collection_2026", "collection_2026_nosys")


def permitted_path(name):
    """Only the retained model records and the historical comparison input."""
    if not isinstance(name, str):
        return False
    path = PurePosixPath(name)
    if path.as_posix() != name or ".." in path.parts or path.is_absolute():
        return False
    if len(path.parts) == 3 and path.parts[:2] in [("data", c) for c in COLLECTIONS]:
        return bool(re.fullmatch(r"[A-Za-z0-9_.-]+\.jsonl", path.name))
    return path.parent == PurePosixPath("data") and (
        path.name in {"trace_samples_2026.json", "published_2024_country_scores.csv"}
        or bool(re.fullmatch(r"trace_labels_2026__[A-Za-z0-9_.-]+\.jsonl", path.name))
    )


def load_manifest(path=MANIFEST):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not manifest.get("files"):
        raise ValueError("expected a nonempty version-1 reproduction manifest")
    seen = set()
    for entry in manifest["files"]:
        name = entry["path"]
        if not permitted_path(name) or name in seen:
            raise ValueError(f"unsafe or duplicate manifest path: {name}")
        if type(entry["bytes"]) is not int or entry["bytes"] <= 0:
            raise ValueError(f"invalid file size: {name}")
        if not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise ValueError(f"invalid SHA-256: {name}")
        seen.add(name)
    return manifest


def local_path(root, name):
    path = root / name
    if any(p.is_symlink() for p in [path, *path.parents] if p != root.parent):
        raise ValueError(f"response paths must not follow symlinks: {name}")
    return path


def check_content(content, entry):
    if len(content) != entry["bytes"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
        raise ValueError(f"response file differs from frozen manifest: {entry['path']}")


def verify(root, manifest):
    """Fail on missing, changed or additional files consumed by frozen replay."""
    root = Path(root).resolve()
    expected = {entry["path"] for entry in manifest["files"]}
    actual = set()
    for collection in COLLECTIONS:
        actual.update(
            str(p.relative_to(root)) for p in (root / "data" / collection).glob("*.jsonl")
        )
    actual.update(
        str(p.relative_to(root)) for p in (root / "data").glob("trace_labels_2026__*.jsonl")
    )
    extra = actual - expected
    if (root / "data/trace_labels_human_2026.csv").exists():
        extra.add("data/trace_labels_human_2026.csv")
    if extra:
        raise ValueError(
            "additional records would change frozen replay: " + ", ".join(sorted(extra))
        )
    for entry in manifest["files"]:
        path = local_path(root, entry["path"])
        if not path.is_file():
            raise FileNotFoundError(
                f"missing {entry['path']}; install the separate response archive (see docs/REPRODUCING.md)"
            )
        check_content(path.read_bytes(), entry)


def manifest_bytes(manifest):
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")


def pack(root, manifest, output):
    """Build a deterministic data-only archive from the explicit manifest."""
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or root in output.parents:
        raise ValueError("write the response archive outside the source repository")
    verify(root, manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        candidate = Path(temporary) / "responses.tar.gz"
        with candidate.open("wb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w") as archive:
                    files = [(MANIFEST_NAME, manifest_bytes(manifest))]
                    for entry in sorted(manifest["files"], key=lambda e: e["path"]):
                        content = local_path(root, entry["path"]).read_bytes()
                        check_content(content, entry)
                        files.append((entry["path"], content))
                    for name, content in files:
                        info = tarfile.TarInfo(name)
                        info.size, info.mode, info.mtime = len(content), 0o644, 0
                        archive.addfile(info, io.BytesIO(content))
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if output.exists() and hashlib.sha256(output.read_bytes()).hexdigest() != digest:
            raise FileExistsError(f"refusing to replace a different archive: {output}")
        candidate.replace(output)
    output.with_name(output.name + ".sha256").write_text(f"{digest}  {output.name}\n")
    return digest


def install(root, manifest, source):
    """Validate every archive member before installing any input.

    Extraction is by exact filename into regular files, never extractall().
    Existing different files, links, extra members and duplicate members fail.
    """
    root = Path(root).resolve()
    expected = {entry["path"]: entry for entry in manifest["files"]}
    expected_names = {*expected, MANIFEST_NAME}
    with tempfile.TemporaryDirectory() as temporary:
        scratch = Path(temporary)
        seen = set()
        with tarfile.open(source, "r:gz") as archive:
            for member in archive:
                if member.name not in expected_names or member.name in seen or not member.isfile():
                    raise ValueError(
                        f"unexpected, duplicate or nonregular archive member: {member.name}"
                    )
                size = (
                    len(manifest_bytes(manifest))
                    if member.name == MANIFEST_NAME
                    else expected[member.name]["bytes"]
                )
                if member.size != size:
                    raise ValueError(f"unexpected archive member size: {member.name}")
                content = archive.extractfile(member).read()
                seen.add(member.name)
                if member.name == MANIFEST_NAME:
                    if content != manifest_bytes(manifest):
                        raise ValueError("archive manifest differs from the checked-in manifest")
                    continue
                check_content(content, expected[member.name])
                destination = local_path(root, member.name)
                if destination.exists():
                    check_content(destination.read_bytes(), expected[member.name])
                staged = scratch / member.name
                staged.parent.mkdir(parents=True, exist_ok=True)
                staged.write_bytes(content)
        if seen != expected_names:
            raise ValueError("archive is incomplete: " + ", ".join(sorted(expected_names - seen)))
        for name in sorted(expected):
            destination = local_path(root, name)
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as stream:
                    stream.write((scratch / name).read_bytes())
    verify(root, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("verify")
    builder = subparsers.add_parser("pack")
    builder.add_argument("--output", type=Path, required=True)
    installer = subparsers.add_parser("install")
    installer.add_argument("archive", type=Path)
    args = parser.parse_args()
    try:
        manifest = load_manifest(args.manifest)
        if args.command == "pack":
            print(pack(args.root, manifest, args.output))
        elif args.command == "install":
            install(args.root, manifest, args.archive)
        else:
            verify(args.root, manifest)
        print(f"{args.command}: {len(manifest['files'])} frozen input files verified")
    except (OSError, ValueError, KeyError, tarfile.TarError) as exc:
        parser.exit(1, f"Reproduction data: {exc}\n")


if __name__ == "__main__":
    main()
