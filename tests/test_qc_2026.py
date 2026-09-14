"""The production QC command must reject an incomplete/corrupt full design."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts import qc_2026 as qc


@pytest.fixture(scope="module")
def complete_records():
    records = []
    for llm in sorted(qc.EXPECTED_MODELS):
        for language in qc.EXPECTED_LANGUAGES:
            for question in qc.IV_QNS:
                parsed = [1, 2] if question in ("Y002", "Y003") else 1
                raw = "1,2" if isinstance(parsed, list) else "1"
                for prefix in range(len(qc.PROMPT_VARIANTS[language]["persona"])):
                    for repeat in range(qc.N_REPEATS):
                        records.append(
                            {
                                "llm": llm,
                                "language": language,
                                "question": question,
                                "system_prompt_id": prefix,
                                "repeat": repeat,
                                "raw_content": raw,
                                "thinking": "",
                                "parsed": parsed,
                                "error": None,
                                "attempts": 1,
                                "duration_ms": 1,
                            }
                        )
    return records


def write_records(tmp_path, records):
    directory = tmp_path / "data" / "collection_2026"
    directory.mkdir(parents=True)
    (directory / "fixture.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    return directory


def run_qc(tmp_path, monkeypatch, records):
    monkeypatch.setattr(qc, "RAW_DIR", write_records(tmp_path, records))
    monkeypatch.chdir(tmp_path)
    return qc.main()


def test_full_design_with_historical_encodings_and_low_parse_rate_passes(
    tmp_path, monkeypatch, complete_records
):
    records = copy.deepcopy(complete_records)
    # Entire one-cell failure remains a complete, reportable collection.
    for record in records[: qc.CELL_DESIGN_CALLS]:
        record.update(parsed=None, error="parse: refusal", raw_content="I cannot answer")
    for record in records:
        record["repeat"] = str(record["repeat"])
        record["system_prompt_id"] = str(record["system_prompt_id"])
        if record["language"] == "en":
            record.pop("language")
    assert run_qc(tmp_path, monkeypatch, records) == 0
    cells = pd.read_csv(tmp_path / "data/qc_2026_cells.csv")
    assert len(cells) == len(qc.EXPECTED_MODELS) * len(qc.EXPECTED_LANGUAGES) == 34
    assert cells["complete"].all()
    rates = pd.read_csv(tmp_path / "data/qc_2026_parse_rates.csv")
    assert (rates["parse_rate"] == 0).sum() == len(qc.IV_QNS)


def test_cli_499_records_cannot_hide_33_absent_cells(tmp_path, complete_records):
    write_records(tmp_path, complete_records[: qc.CELL_DESIGN_CALLS - 1])
    result = subprocess.run(
        [sys.executable, str(Path(qc.__file__).resolve())],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "QC FAILED" in result.stderr
    cells = pd.read_csv(tmp_path / "data/qc_2026_cells.csv")
    assert len(cells) == 34
    assert (cells["records"] == 0).sum() == 33
    assert cells["missing_keys"].sum() == 33 * qc.CELL_DESIGN_CALLS + 1


@pytest.mark.parametrize(
    "change",
    [
        {"language": "fr"},
        {"llm": "unknown-model"},
        {"question": "unknown-item"},
        {"repeat": 5},
        {"system_prompt_id": 10},
        {"prompt_variant": "nosys"},
        {"parsed": 2},
        {"parsed": True},
        {"parsed": 99, "raw_content": "99"},
        {"parsed": None},
        {"error": "parse: failed"},
        {"parsed": None, "error": "ResponseError: 500 (status code: 500)", "raw_content": ""},
        {"parsed": None, "error": "refused", "raw_content": "No"},
        {"transport_failure": True},
        {"attempts": 0},
        {"duration_ms": -1},
    ],
)
def test_fatal_design_or_stored_response_corruption(
    tmp_path, monkeypatch, complete_records, change
):
    records = list(complete_records)
    records[0] = {**records[0], **change}
    assert run_qc(tmp_path, monkeypatch, records) == 1


@pytest.mark.parametrize(
    "question, choices",
    [
        ("Y002", [1, 1]),
        ("Y002", [1, 5]),
        ("Y003", [1, 1]),
        ("Y003", [1, 2, 3, 4, 5, 6]),
        ("Y003", [12]),
    ],
)
def test_invalid_choice_structure_cannot_pass_projected_range_check(
    tmp_path, monkeypatch, complete_records, question, choices
):
    records = list(complete_records)
    index = next(i for i, record in enumerate(records) if record["question"] == question)
    records[index] = {
        **records[index],
        "parsed": choices,
        "raw_content": ",".join(map(str, choices)),
    }
    assert run_qc(tmp_path, monkeypatch, records) == 1


def test_mixed_type_duplicate_is_reported_and_never_removed(
    tmp_path, monkeypatch, complete_records
):
    duplicate = {**complete_records[0], "system_prompt_id": "0", "repeat": "0"}
    assert run_qc(tmp_path, monkeypatch, [*complete_records, duplicate]) == 1
    audit = pd.read_csv(tmp_path / "data/qc_2026_dedup.csv").iloc[0]
    assert audit.duplicates_detected == audit.mixed_type_hazard == 1
    assert audit.duplicates_removed == 0
    assert audit.records_before == audit.records_after == len(complete_records) + 1


@pytest.mark.parametrize(
    "field, value",
    [
        ("repeat", 1.5),
        ("repeat", True),
        ("system_prompt_id", "1.5"),
        ("language", ""),
        ("llm", []),
    ],
)
def test_malformed_schema_returns_nonzero(tmp_path, monkeypatch, complete_records, field, value):
    record = {**complete_records[0], field: value}
    assert run_qc(tmp_path, monkeypatch, [record]) == 1


def test_all_terminal_failures_are_reportable(tmp_path, monkeypatch, complete_records):
    records = [
        {**r, "parsed": None, "error": "parse: refusal", "raw_content": "No"}
        for r in complete_records
    ]
    assert run_qc(tmp_path, monkeypatch, records) == 0


QC_ARTEFACTS = {
    "qc_2026_attempts.csv",
    "qc_2026_cells.csv",
    "qc_2026_dedup.csv",
    "qc_2026_determinism.csv",
    "qc_2026_failures.csv",
    "qc_2026_index_validity.csv",
    "qc_2026_integrity_errors.csv",
    "qc_2026_latency.csv",
    "qc_2026_parse_rates.csv",
    "qc_2026_thinking.csv",
}


def test_provider_rejections_pass_and_are_counted_apart_from_parse_failures(
    tmp_path, monkeypatch, complete_records, capsys
):
    records = copy.deepcopy(complete_records)
    records[0].update(
        parsed=None, error="provider: ResponseError: blocked (status code: 403)", raw_content=""
    )
    records[1].update(parsed=None, error="parse: refusal", raw_content="")
    assert run_qc(tmp_path, monkeypatch, records) == 0
    failures = pd.read_csv(tmp_path / "data/qc_2026_failures.csv")
    assert dict(zip(failures["category"], failures["n"], strict=True)) == {
        "provider": 1,
        "empty": 1,
    }
    assert "terminal failures: 1 parse, 1 provider rejections" in capsys.readouterr().out


def test_failure_taxonomy_has_no_out_of_range_category():
    assert qc.classify_failure("", "provider: blocked") == "provider"
    assert qc.classify_failure("99", "parse: 99 out of range") == "format"
    assert qc.classify_failure("", "parse: empty") == "empty"


def test_nosys_integrity_pass_reports_without_failing_or_writing(
    tmp_path, monkeypatch, complete_records, capsys
):
    llm = sorted(qc.EXPECTED_MODELS)[0]
    nosys = [
        {
            **complete_records[0],
            "llm": llm,
            "question": question,
            "system_prompt_id": 0,
            "repeat": repeat,
            "prompt_variant": "nosys",
            "raw_content": "1,2" if question in ("Y002", "Y003") else "1",
            "parsed": [1, 2] if question in ("Y002", "Y003") else 1,
        }
        for question in qc.IV_QNS
        for repeat in range(50)
    ]
    nosys = [*nosys[:-1], nosys[0]]  # one missing trial and one duplicate
    directory = tmp_path / "data" / "collection_2026_nosys"
    directory.mkdir(parents=True)
    (directory / "fixture.jsonl").write_text("".join(json.dumps(r) + "\n" for r in nosys))
    assert run_qc(tmp_path, monkeypatch, complete_records) == 0
    out = capsys.readouterr().out
    assert "No-persona integrity (report only" in out
    missing = 1 + (len(qc.EXPECTED_MODELS) - 1) * qc.CELL_DESIGN_CALLS
    assert f"total: 500 records, 1 duplicate records, {missing} missing trials" in out
    written = {p.name for p in (tmp_path / "data").iterdir() if p.is_file()}
    assert written == QC_ARTEFACTS
    assert [p.name for p in directory.iterdir()] == ["fixture.jsonl"]
