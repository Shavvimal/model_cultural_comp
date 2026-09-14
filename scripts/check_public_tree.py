"""Check the Git index for data, generated artefacts and private local state.

Runs in CI and locally, including files added with git add --force. Synthetic
fixtures belong in test code. This is a file policy, not a general secret scanner.
"""

import subprocess
from pathlib import Path, PurePosixPath

FORBIDDEN_DIRECTORIES = {
    "data",
    "figures",
    "notebooks",
    ".venv",
    "venv",
    "__pycache__",
    ".claude",
    ".codex",
    ".planning",
    "attempt_audit",
}
FORBIDDEN_SUFFIXES = {
    ".csv",
    ".jsonl",
    ".pkl",
    ".pickle",
    ".sav",
    ".sps",
    ".parquet",
    ".npy",
    ".npz",
    ".png",
    ".pdf",
    ".ipynb",
    ".log",
    ".zip",
    ".gz",
    ".pem",
    ".key",
    ".pyc",
    # Other formats survey microdata and extracts are commonly shipped in.
    ".dta",
    ".rds",
    ".rdata",
    ".xlsx",
    ".xls",
    ".tsv",
    ".feather",
    ".h5",
    ".hdf5",
    ".sqlite",
    ".db",
    ".tar",
    ".xz",
    ".7z",
}


def violations(entries: list[tuple[str, str]]) -> list[str]:
    """Return indexed names that break the policy, given (mode, path) index entries.

    Directory and suffix matching ignore case, so `DATA/x.json` and `wvs.RData`
    are caught on case-insensitive filesystems too.
    """
    bad = []
    for mode, name in entries:
        path = PurePosixPath(name)
        if (
            {part.lower() for part in path.parts} & FORBIDDEN_DIRECTORIES
            or path.suffix.lower() in FORBIDDEN_SUFFIXES
            or path.name == ".env"
            or path.name.startswith(".env.")
            or path.name in {"credentials", "credentials.json", "secrets.json", ".netrc"}
            or mode == "120000"
        ):
            bad.append(name)
    return bad


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    raw = subprocess.check_output(["git", "-C", str(root), "ls-files", "--stage", "-z"])
    entries = []
    for record in raw.decode().split("\0"):
        if record:
            metadata, name = record.split("\t", 1)
            entries.append((metadata.split()[0], name))
    bad = violations(entries)
    if bad:
        print("Remove data/generated/private files from the Git index:\n" + "\n".join(bad))
        return 1
    print(f"Public tree: {len(entries)} indexed files; no data, generated files or symlinks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
