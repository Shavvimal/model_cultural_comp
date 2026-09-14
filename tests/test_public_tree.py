"""Exercise the actual index gate, including forced additions and staged removal."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.check_public_tree import violations


def test_git_gate_rejects_forced_data_and_accepts_cached_removal(tmp_path):
    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    script = tmp_path / "scripts/check_public_tree.py"
    script.parent.mkdir()
    shutil.copyfile(Path(__file__).parents[1] / "scripts/check_public_tree.py", script)
    (tmp_path / ".gitignore").write_text("data/\n")
    git("init", "--quiet")
    git("add", ".gitignore", "scripts")

    def check():
        return subprocess.run([sys.executable, str(script)], capture_output=True, text=True)

    assert check().returncode == 0
    data = tmp_path / "data/retained.json"
    data.parent.mkdir()
    data.write_text('{"synthetic": true}\n')
    assert check().returncode == 0  # Local ignored inputs are permitted.
    git("add", "--force", "data/retained.json")
    failure = check()
    assert failure.returncode == 1
    assert "data/retained.json" in failure.stdout
    git("rm", "--cached", "data/retained.json")
    assert data.exists()
    assert check().returncode == 0


@pytest.mark.parametrize(
    "name",
    ["docs/wvs.dta", "docs/ivs.RData", "results/extract.XLSX", "archive.7z", "notes/panel.tsv"],
)
def test_microdata_formats_are_rejected_regardless_of_case(name):
    assert violations([("100644", name)]) == [name]


@pytest.mark.parametrize("name", ["DATA/retained.json", "Data/ivs.json", "src/Figures/fig.svg"])
def test_forbidden_directories_match_case_insensitively(name):
    assert violations([("100644", name)]) == [name]


def test_source_files_still_pass():
    assert (
        violations([("100644", "app/culture_map.py"), ("100644", "docs/results-manifest.json")])
        == []
    )
