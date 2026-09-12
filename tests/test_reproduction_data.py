"""A separate data archive must be complete, verifiable and safe to install."""

import hashlib
import io
import json
import tarfile

import pytest

from scripts.reproduction_data import install, load_manifest, pack, verify


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
