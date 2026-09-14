"""Input contracts of the recorded-trial coverage tables (synthetic data only)."""

import json

import pytest

from app.control_audit import coverage_tables


def record(**overrides):
    base = {
        "llm": "model-a",
        "language": "en",
        "question": "A008",
        "system_prompt_id": 0,
        "repeat": 0,
        "attempts": 1,
        "parsed": 1,
        "error": None,
        "ts": "2026-08-01T00:00:00+0000",
    }
    return base | overrides


def write(directory, records):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model-a.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return str(directory)


def test_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        coverage_tables(str(tmp_path / "absent"), "nosys", ["A008"])


@pytest.mark.parametrize("content", [None, ""])
def test_directory_without_records_raises(tmp_path, content):
    tmp_path.joinpath("empty").mkdir()
    if content is not None:
        tmp_path.joinpath("empty", "blank.jsonl").write_text(content)
    with pytest.raises(ValueError, match="no terminal records"):
        coverage_tables(str(tmp_path / "empty"), "nosys", ["A008"])


@pytest.mark.parametrize(
    "field, value",
    [("repeat", 1.5), ("repeat", "1.5"), ("system_prompt_id", True), ("attempts", "one")],
)
def test_non_integer_index_is_rejected_not_truncated(tmp_path, field, value):
    directory = write(tmp_path / "c", [record(**{field: value})])
    with pytest.raises(ValueError, match=field):
        coverage_tables(directory, "nosys", ["A008"])


def test_integer_strings_are_normalised_before_keep_last_dedup(tmp_path):
    # A parsed record followed by a failed record for the same (string-encoded)
    # key: keep-last retains the failure, which is the documented rule.
    records = [
        record(),
        record(system_prompt_id="0", repeat="0", parsed=None, error="parse: no", attempts=3),
    ]
    cells, items = coverage_tables(write(tmp_path / "c", records), "nosys", ["A008"])
    cell = cells.iloc[0]
    assert (cell["raw_record_lines"], cell["duplicate_record_lines"]) == (2, 1)
    assert (cell["parsed_trials"], cell["terminal_parse_failures"]) == (0, 1)
    assert cell["recorded_attempts_all_lines"] == 4
    assert items.iloc[0]["recorded_attempts_retained_trials"] == 3
