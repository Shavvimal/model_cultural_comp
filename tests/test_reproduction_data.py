"""A separate data archive must be complete, verifiable and safe to install."""

import gzip
import hashlib
import io
import json
import sys
import tarfile

import pytest

from scripts import reproduction_data
from scripts.reproduction_data import (
    RESULTS_MANIFEST_NAME,
    RESULTS_README,
    RESULTS_README_MAX_BYTES,
    RESULTS_README_NAME,
    install,
    load_manifest,
    manifest_bytes,
    pack,
    pack_results,
    verify,
    verify_results,
)


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "source"
    files = {
        "data/collection/model_responses_df.jsonl": b'{"llm":"m","question":"A008","response":2}\n',
        "data/trace_samples_2026.json": b"[]\n",
    }
    entries = []
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        entries.append(
            {"path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"schema_version": 1, "files": entries}))
    return root, load_manifest(manifest_path)


def test_data_archive_roundtrip_excludes_licensed_inputs_and_outputs(corpus, tmp_path):
    root, manifest = corpus
    (root / "data/ivs_df.pkl").write_bytes(b"private survey input")
    (root / "data/llm_ellipses.csv").write_bytes(b"regenerable output")
    output = tmp_path / "responses.tar.gz"
    digest = pack(root, manifest, output)
    assert digest == pack(root, manifest, output)
    fresh = tmp_path / "fresh"
    install(fresh, manifest, output)
    verify(fresh, manifest)
    assert not (fresh / "data/ivs_df.pkl").exists()
    assert not (fresh / "data/llm_ellipses.csv").exists()
    for entry in manifest["files"]:
        assert (fresh / entry["path"]).read_bytes() == (root / entry["path"]).read_bytes()
    install(fresh, manifest, output)  # idempotent for identical inputs


@pytest.mark.parametrize("problem", ["missing", "changed", "extra", "human"])
def test_frozen_replay_rejects_incomplete_or_contaminated_inputs(corpus, problem):
    root, manifest = corpus
    path = root / manifest["files"][0]["path"]
    if problem == "missing":
        path.unlink()
    elif problem == "changed":
        path.write_bytes(b"a different experiment")
    elif problem == "human":
        (root / "data/trace_labels_human_2026.csv").write_text("new labels")
    else:
        (path.parent / "extra.jsonl").write_text("{}\n")
    with pytest.raises((ValueError, FileNotFoundError)):
        verify(root, manifest)


@pytest.mark.parametrize(
    "bad_member", ["../outside.txt", "data/ivs_df.pkl", "link", "duplicate", "corrupt", "missing"]
)
def test_invalid_archive_never_installs_partial_inputs(corpus, tmp_path, bad_member):
    root, manifest = corpus
    good = tmp_path / "good.tar.gz"
    pack(root, manifest, good)
    bad = tmp_path / "bad.tar.gz"
    with tarfile.open(good) as source, tarfile.open(bad, "w:gz") as target:
        members = source.getmembers()
        for member in members:
            if bad_member == "missing" and member == members[-1]:
                continue
            content = source.extractfile(member).read()
            if bad_member == "corrupt" and member == members[-1]:
                content = b"x" * len(content)
            target.addfile(member, io.BytesIO(content))
        if bad_member == "duplicate":
            target.addfile(members[-1], io.BytesIO(source.extractfile(members[-1]).read()))
        elif bad_member in {"../outside.txt", "data/ivs_df.pkl", "link"}:
            member = tarfile.TarInfo(bad_member)
            if bad_member == "link":
                member.type = tarfile.SYMTYPE
                member.linkname = "../outside.txt"
            target.addfile(member, io.BytesIO(b""))
    fresh = tmp_path / "fresh"
    with pytest.raises(ValueError):
        install(fresh, manifest, bad)
    assert not fresh.exists()


def test_symlink_and_existing_different_input_are_preserved(corpus, tmp_path):
    root, manifest = corpus
    archive = tmp_path / "responses.tar.gz"
    pack(root, manifest, archive)
    fresh = tmp_path / "fresh"
    path = fresh / manifest["files"][-1]["path"]
    path.parent.mkdir(parents=True)
    path.write_bytes(b"existing experiment")
    with pytest.raises(ValueError):
        install(fresh, manifest, archive)
    assert path.read_bytes() == b"existing experiment"
    assert not (fresh / manifest["files"][0]["path"]).exists()
    path.unlink()
    path.symlink_to(root / manifest["files"][-1]["path"])
    with pytest.raises(ValueError, match="symlink"):
        install(fresh, manifest, archive)


def test_manifest_cannot_include_survey_pickle(corpus, tmp_path):
    _, manifest = corpus
    manifest["files"][0]["path"] = "data/ivs_df.pkl"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsafe"):
        load_manifest(path)


@pytest.mark.parametrize(
    "manifest",
    [
        [],
        {"schema_version": True, "files": [{}]},
        {"schema_version": 1, "files": {"unexpected": "mapping"}},
        {"schema_version": 1, "files": [None]},
        {"schema_version": 1, "files": [{"path": "data/trace_samples_2026.json"}]},
        {
            "schema_version": 1,
            "files": [{"path": "data/trace_samples_2026.json", "bytes": 3, "sha256": None}],
        },
    ],
)
def test_malformed_manifest_fails_at_boundary(tmp_path, manifest):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_manifest(path)


def test_stray_destination_record_is_rejected_before_any_input_is_written(corpus, tmp_path):
    root, manifest = corpus
    archive = tmp_path / "responses.tar.gz"
    pack(root, manifest, archive)
    fresh = tmp_path / "fresh"
    stray = fresh / "data/collection/extra.jsonl"
    stray.parent.mkdir(parents=True)
    stray.write_text("{}\n")
    with pytest.raises(ValueError, match="additional records"):
        install(fresh, manifest, archive)
    written = sorted(p.relative_to(fresh).as_posix() for p in fresh.rglob("*") if p.is_file())
    assert written == ["data/collection/extra.jsonl"]


def test_truncated_download_exits_with_named_message(corpus, tmp_path, monkeypatch, capsys):
    root, manifest = corpus
    archive = tmp_path / "responses.tar.gz"
    pack(root, manifest, archive)
    truncated = tmp_path / "truncated.tar.gz"
    truncated.write_bytes(archive.read_bytes()[: archive.stat().st_size // 2])
    manifest_path = tmp_path / "manifest.json"
    fresh = tmp_path / "fresh"
    argv = ["reproduction_data.py", "--root", str(fresh), "--manifest", str(manifest_path)]
    monkeypatch.setattr(sys, "argv", [*argv, "install", str(truncated)])
    with pytest.raises(SystemExit) as exit_info:
        reproduction_data.main()
    assert exit_info.value.code == 1
    assert capsys.readouterr().err.startswith("Reproduction data: ")
    assert not fresh.exists()


@pytest.fixture
def supplement(tmp_path):
    """A synthetic results supplement, its tracked manifest and the input manifest it cites."""
    inputs = tmp_path / "inputs.json"
    inputs.write_text('{"schema_version": 1}\n')
    root = tmp_path / "results-root"
    files = {"data/summary.csv": b"a,b\n1,2\n", "data/trace_coding.json": b"{}\n"}
    entries = []
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        entries.append(
            {"bytes": len(content), "path": name, "sha256": hashlib.sha256(content).hexdigest()}
        )
    results = {
        "files": entries,
        "input_manifest_sha256": hashlib.sha256(inputs.read_bytes()).hexdigest(),
        "schema_version": 1,
    }
    tracked = tmp_path / "results-manifest.json"
    tracked.write_bytes(manifest_bytes(results))
    return root, tracked, inputs, files


def _write_supplement(path, members):
    with path.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as z:
        with tarfile.open(fileobj=z, mode="w") as archive:
            for name, content in members:
                info = tarfile.TarInfo(name)
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))


def _members(tracked, files, **changes):
    members = {RESULTS_MANIFEST_NAME: tracked.read_bytes(), RESULTS_README_NAME: b"readme\n"}
    members.update(files)
    members.update(changes)
    return [(name, content) for name, content in members.items() if content is not None]


def test_results_supplement_roundtrip_verifies(supplement, tmp_path):
    root, tracked, inputs, files = supplement
    output = tmp_path / "out/results.tar.gz"
    digest = pack_results(root, output, tracked, inputs)
    assert digest == pack_results(root, output, tracked, inputs)
    assert verify_results(output, tracked, inputs) == len(files)
    _write_supplement(tmp_path / "handmade.tar.gz", _members(tracked, files))
    assert verify_results(tmp_path / "handmade.tar.gz", tracked, inputs) == len(files)


@pytest.mark.parametrize(
    "problem", ["tampered", "resized", "missing", "extra", "duplicate", "manifest", "link"]
)
def test_results_supplement_rejects_altered_archives(supplement, tmp_path, problem):
    _, tracked, inputs, files = supplement
    changes = {}
    if problem == "tampered":
        changes["data/summary.csv"] = b"a,b\n1,3\n"  # same size, different hash
    elif problem == "resized":
        changes["data/summary.csv"] = b"a,b\n1,20\n"
    elif problem == "missing":
        changes["data/trace_coding.json"] = None
    elif problem == "extra":
        changes["data/ivs_df.pkl"] = b"licensed"
    elif problem == "manifest":
        changes[RESULTS_MANIFEST_NAME] = tracked.read_bytes().replace(b"  ", b"   ", 1)
    members = _members(tracked, files, **changes)
    if problem == "duplicate":
        members.append(("data/summary.csv", files["data/summary.csv"]))
    archive = tmp_path / "bad.tar.gz"
    _write_supplement(archive, members)
    if problem == "link":
        with tarfile.open(archive, "w:gz") as target:
            for name, content in _members(tracked, files):
                info = tarfile.TarInfo(name)
                if name == "data/summary.csv":
                    info.type, info.linkname = tarfile.SYMTYPE, "../outside.csv"
                else:
                    info.size = len(content)
                target.addfile(info, io.BytesIO(content))
    with pytest.raises(ValueError):
        verify_results(archive, tracked, inputs)


def test_results_supplement_must_cite_checked_in_inputs(supplement, tmp_path):
    _, tracked, inputs, files = supplement
    _write_supplement(tmp_path / "results.tar.gz", _members(tracked, files))
    inputs.write_text('{"schema_version": 1, "changed": true}\n')
    with pytest.raises(ValueError, match="input manifest"):
        verify_results(tmp_path / "results.tar.gz", tracked, inputs)


def test_pack_results_refuses_changed_local_outputs(supplement, tmp_path):
    root, tracked, inputs, _ = supplement
    (root / "data/summary.csv").write_bytes(b"a,b\n9,9\n")
    with pytest.raises(ValueError, match="differs"):
        pack_results(root, tmp_path / "results.tar.gz", tracked, inputs)
    assert not (tmp_path / "results.tar.gz").exists()


def test_tracked_supplement_readme_is_the_released_copy():
    """pack-results packs this file byte for byte; one changed byte moves the archive hash."""
    content = RESULTS_README.read_bytes()
    assert len(content) == 1106 <= RESULTS_README_MAX_BYTES
    assert hashlib.sha256(content).hexdigest() == (
        "cc4ae9dd8343c124f0d7de142d2528aa7b563aa62346f77b3beb84183dbd7464"
    )
