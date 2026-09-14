"""Prompt-composition and task-enumeration contracts of the survey harness.

No network: ``CloudSurvey.__post_init__`` only constructs a client object.
"""

import hashlib
import json
import os

import pytest

from app.cloud_survey import (
    CALLS_PER_CELL,
    IV_QN_PROMPTS,
    N_REPEATS,
    PARSERS,
    PRIMER,
    SYSTEM_PROMPTS,
    SYSTEM_PROMPTS_ZH,
    CloudSurvey,
    is_throttled,
    load_dotenv,
    provenance_host,
)

AVERAGING_CUES = ("average", "typical", "普通", "典型")
# SHA-256 over every request message list (en/zh x persona/nosys x item x prefix)
# plus PRIMER, computed from the unmodified prompt strings at commit 4e1a2a0.
# The retained corpus was collected with these strings: never update this value
# to make an edited prompt pass; collect a new, versioned prompt set instead.
GOLDEN_PROMPT_SHA256 = "15be87863cc3733719fcaa1cc692fc4bf85b11f8ad1b64d597dcbabaaf903819"
DUMMY_KEY = "sk-dummy-SECRET-123"


def make(tmp_path, **kw) -> CloudSurvey:
    return CloudSurvey(out_dir=tmp_path, api_key="test-key", **kw)


@pytest.mark.parametrize(
    "invalid",
    [
        {"concurrency": 0},
        {"concurrency": -1},
        {"concurrency": True},
        {"concurrency": 1.5},
        {"concurrency": "2"},
        {"timeout_s": 0},
        {"timeout_s": -1},
        {"timeout_s": float("inf")},
        {"timeout_s": float("nan")},
        {"timeout_s": True},
        {"timeout_s": "300"},
    ],
)
def test_invalid_execution_limits_fail_before_creating_resources(tmp_path, monkeypatch, invalid):
    from unittest.mock import Mock

    from app import cloud_survey

    client = Mock()
    monkeypatch.setattr(cloud_survey, "AsyncClient", client)
    output = tmp_path / "collection"
    with pytest.raises(ValueError, match=next(iter(invalid))):
        make(output, **invalid)
    client.assert_not_called()
    assert not output.exists()


class TestPersonaProtocol:
    def test_cell_is_500_calls_over_ten_prefixes(self, tmp_path):
        tasks = make(tmp_path).tasks_for_model()
        assert len(tasks) == CALLS_PER_CELL == 500
        assert {t[1] for t in tasks} == set(range(len(SYSTEM_PROMPTS)))
        assert {t[2] for t in tasks} == set(range(N_REPEATS))
        assert len(set(tasks)) == len(tasks)

    def test_prompt_starts_with_the_persona_prefix(self, tmp_path):
        survey = make(tmp_path)
        assert survey._prompt_for("F063", 0).startswith(SYSTEM_PROMPTS[0] + " ")
        zh = make(tmp_path, language="zh")
        assert zh._prompt_for("F063", 9).startswith(SYSTEM_PROMPTS_ZH[9] + " ")

    def test_three_prefixes_carry_no_averaging_cue(self):
        bare = [i for i, p in enumerate(SYSTEM_PROMPTS) if not any(c in p for c in AVERAGING_CUES)]
        assert bare == [2, 5, 8, 9]  # human being / person / individual / world citizen

    def test_chinese_classification_counts_putong_as_an_averaging_cue(self):
        # Translation caveat (docs/PROTOCOL.md): 普通 means "ordinary, common" but
        # renders "average", so it is classified as an averaging cue on purpose.
        bare = [
            i for i, p in enumerate(SYSTEM_PROMPTS_ZH) if not any(c in p for c in AVERAGING_CUES)
        ]
        assert bare == [2, 5, 8, 9]
        assert [i for i, p in enumerate(SYSTEM_PROMPTS_ZH) if "普通" in p] == [0, 3, 6]
        assert [i for i, p in enumerate(SYSTEM_PROMPTS) if "average" in p] == [0, 3, 6]


def test_golden_hash_pins_every_request_message_and_the_primer(tmp_path):
    payload = []
    for language in ("en", "zh"):
        for variant in ("persona", "nosys"):
            survey = make(tmp_path, language=language, prompt_variant=variant)
            for qn in sorted(IV_QN_PROMPTS):
                for sid in range(len(survey.persona_prefixes)):
                    messages = survey._request_for("model", qn, sid)["messages"]
                    payload.append([language, variant, qn, sid, messages])
    payload.append(PRIMER)
    assert len(payload) == 2 * (len(IV_QN_PROMPTS) * (len(SYSTEM_PROMPTS) + 1)) + 1
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(blob.encode("utf-8")).hexdigest() == GOLDEN_PROMPT_SHA256


@pytest.mark.parametrize(
    "qn, raw, expected",
    [
        ("F063", "10", 10),
        ("F063", " 10. ", 10),
        ("A008", '"3"', 3),
        ("Y002", "2, 4", (2, 4)),
        ("Y003", "1, 2,6", [1, 2, 6]),
    ],
)
def test_parsers_accept_ascii_digits(qn, raw, expected):
    assert PARSERS[qn].parse(raw) == expected


@pytest.mark.parametrize(
    "qn, raw",
    [
        ("F063", "1_0"),
        ("F063", "\uff11\uff10"),
        ("A008", "٣"),
        ("A008", "+3"),
        ("Y002", "2,+4"),
        ("Y002", "٢,4"),
        ("Y003", "1_0"),
        ("Y003", "1,\uff12"),
    ],
)
def test_parsers_reject_signs_underscores_and_non_ascii_digits(qn, raw):
    with pytest.raises(ValueError):
        PARSERS[qn].parse(raw)


def test_api_key_is_absent_from_repr(tmp_path):
    survey = CloudSurvey(out_dir=tmp_path, api_key=DUMMY_KEY)
    assert DUMMY_KEY not in repr(survey)


@pytest.mark.parametrize(
    "host",
    [
        f"https://user:{DUMMY_KEY}@ollama.com",
        f"https://{DUMMY_KEY}@ollama.com",
        f"user:{DUMMY_KEY}@localhost:11434",
        f"https://ollama.com/?key={DUMMY_KEY}",
    ],
)
def test_credential_bearing_host_is_rejected_without_echo(tmp_path, monkeypatch, host):
    from unittest.mock import Mock

    from app import cloud_survey

    client = Mock()
    monkeypatch.setattr(cloud_survey, "AsyncClient", client)
    output = tmp_path / "collection"
    with pytest.raises(ValueError) as info:
        make(output, host=host)
    assert DUMMY_KEY not in str(info.value)
    client.assert_not_called()
    assert not output.exists()


@pytest.mark.parametrize(
    "host, recorded",
    [
        ("https://ollama.com", "https://ollama.com"),
        ("https://ollama.example:8443/api/", "https://ollama.example:8443"),
        ("localhost:11434", "http://localhost:11434"),
    ],
)
def test_recorded_host_keeps_only_scheme_host_and_port(host, recorded):
    assert provenance_host(host) == recorded


@pytest.mark.parametrize(
    "status, message, throttled",
    [
        (429, "slow down", True),
        (None, "Rate limit exceeded", True),
        (500, "model failed to generate a response", False),
        (400, "request was moderated", False),
        (None, "separate limits apply", False),
    ],
)
def test_throttle_detection_uses_status_or_word_boundary_rate_limit(status, message, throttled):
    assert is_throttled(status, message) is throttled


class TestNeutralBaseline:
    @pytest.mark.parametrize("language", ["en", "zh"])
    def test_cell_is_500_calls_over_one_empty_prefix(self, tmp_path, language):
        survey = make(tmp_path, language=language, prompt_variant="nosys")
        tasks = survey.tasks_for_model()
        assert len(tasks) == CALLS_PER_CELL
        assert {t[1] for t in tasks} == {0}
        assert {t[2] for t in tasks} == set(range(CALLS_PER_CELL // len(IV_QN_PROMPTS)))

    @pytest.mark.parametrize("language", ["en", "zh"])
    def test_prompt_has_no_persona_and_no_leading_space(self, tmp_path, language):
        survey = make(tmp_path, language=language, prompt_variant="nosys")
        persona = make(tmp_path, language=language)
        for qn in IV_QN_PROMPTS:
            prompt = survey._prompt_for(qn, 0)
            assert prompt == prompt.strip()
            assert not any(cue in prompt for cue in AVERAGING_CUES)
            assert not prompt.startswith(("You are", "您是"))
            # Everything but the prefix is held fixed.
            assert persona._prompt_for(qn, 0).endswith(prompt)

    def test_record_carries_the_variant(self, tmp_path):
        assert make(tmp_path, prompt_variant="nosys").prompt_variant == "nosys"

    def test_unknown_variant_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            make(tmp_path, prompt_variant="average-only")
        with pytest.raises(ValueError):
            make(tmp_path, language="fr")


class TestProvenanceAndAttempts:
    @staticmethod
    def read_records(path):
        import json

        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_exact_default_request_and_returned_metadata_survive(self, tmp_path, monkeypatch):
        import asyncio
        from unittest.mock import AsyncMock

        from ollama import ChatResponse

        from app import cloud_survey

        survey = make(tmp_path)
        thinking = "x" * 2100
        response = ChatResponse(
            model="returned-model:revision",
            created_at="2026-09-11T00:00:00Z",
            message={"role": "assistant", "content": "1", "thinking": thinking},
            done=True,
            eval_count=17,
        )
        chat = AsyncMock(return_value=response)
        monkeypatch.setattr(
            cloud_survey,
            "AsyncClient",
            lambda **kwargs: type("Client", (), {"chat": staticmethod(chat)})(),
        )
        record = asyncio.run(survey._run_task("requested-model", "A008", 0, 0))
        assert chat.call_args.kwargs == record["request"]
        assert set(record["request"]) == {"model", "messages"}
        assert record["request"]["model"] == "requested-model"
        assert record["response"]["model"] == "returned-model:revision"
        assert record["response"]["eval_count"] == 17
        assert record["response"]["message"]["thinking"] == thinking
        assert len(record["thinking"]) == 2000
        assert record["request_provenance"]["thinking"]["requested"] is None
        assert "unknown" in record["request_provenance"]["thinking"]["source"]
        audit_path = tmp_path / "attempt_audit/requested-model.jsonl"
        audit = self.read_records(audit_path)
        assert len(audit) == 1 and audit[0]["outcome"] == "parsed"
        assert "test-key" not in audit_path.read_text()

    def test_explicit_options_are_recorded_as_sent(self, tmp_path, monkeypatch):
        import asyncio
        from unittest.mock import AsyncMock

        survey = make(tmp_path, generation_options={"temperature": 0.7, "seed": 19}, thinking="low")
        call = AsyncMock(return_value={"model": "served", "message": {"content": "1"}})
        monkeypatch.setattr(survey, "_call_once", call)
        record = asyncio.run(survey._run_task("requested", "A008", 0, 0))
        request = call.call_args.args[0]
        assert request == record["request"]
        assert request["options"] == {"temperature": 0.7, "seed": 19}
        assert request["think"] == "low"
        assert record["request_provenance"]["thinking"]["source"] == "explicit request"

    def test_deferred_resweep_preserves_all_attempts_without_completing_trial(
        self, tmp_path, monkeypatch
    ):
        import asyncio
        from unittest.mock import AsyncMock

        from app import cloud_survey

        survey = make(tmp_path)
        monkeypatch.setattr(survey, "tasks_for_model", lambda: [("A008", 0, 0)])
        monkeypatch.setattr(cloud_survey.asyncio, "sleep", AsyncMock())
        call = AsyncMock(side_effect=TimeoutError("offline fixture"))
        monkeypatch.setattr(survey, "_call_once", call)
        assert asyncio.run(survey.run_model("model")) == {"ok": 0, "failed": 0, "deferred": 1}
        assert not survey._jsonl_path("model").exists()
        assert survey._completed("model") == set()
        audit_path = tmp_path / "attempt_audit/model.jsonl"
        first = self.read_records(audit_path)
        assert len(first) == cloud_survey.MAX_ATTEMPTS
        assert {r["outcome"] for r in first} == {"transport_failure"}
        call.side_effect = None
        call.return_value = {"model": "served", "message": {"content": "1"}}
        assert asyncio.run(survey.run_model("model")) == {"ok": 1, "failed": 0, "deferred": 0}
        audit = self.read_records(audit_path)
        assert len(audit) == cloud_survey.MAX_ATTEMPTS + 1
        assert audit[-1]["task_run_id"] != first[0]["task_run_id"]
        assert len(self.read_records(survey._jsonl_path("model"))) == 1
        assert survey._completed("model") == {("A008", 0, 0)}
        calls = call.call_count
        asyncio.run(survey.run_model("model"))
        assert call.call_count == calls
        assert list(tmp_path.glob("*.jsonl")) == [survey._jsonl_path("model")]

    def test_parse_retries_retain_each_raw_response(self, tmp_path, monkeypatch):
        import asyncio
        from unittest.mock import AsyncMock

        survey = make(tmp_path)
        call = AsyncMock(
            side_effect=[
                {"message": {"content": "I cannot answer"}},
                {"message": {"content": "1"}},
            ]
        )
        monkeypatch.setattr(survey, "_call_once", call)
        record = asyncio.run(survey._run_task("model", "A008", 0, 0))
        assert record["attempts"] == 2 and record["parsed"] == 1
        audit = self.read_records(tmp_path / "attempt_audit/model.jsonl")
        assert [r["outcome"] for r in audit] == ["parse_failure", "parsed"]
        assert audit[0]["raw_content"] == "I cannot answer"
        assert audit[1]["raw_content"] == "1"

    def test_resume_normalizes_indices_and_rejects_corrupt_or_duplicate_log(self, tmp_path):
        import json

        survey = make(tmp_path)
        path = survey._jsonl_path("model")
        record = {
            "llm": "model",
            "question": "A008",
            "system_prompt_id": "0",
            "repeat": "0",
            "schema_version": 2,
            "request": survey._request_for("model", "A008", 0),
            "request_provenance": {"host": survey.host},
        }
        line = json.dumps(record) + "\n"
        path.write_text(line)
        assert survey._completed("model") == {("A008", 0, 0)}
        path.write_text(line + line)
        with pytest.raises(ValueError, match="duplicate"):
            survey._completed("model")
        path.write_text(line + '{"llm":')
        with pytest.raises(ValueError, match=r"model\.jsonl:2"):
            survey._completed("model")

    @pytest.mark.parametrize(
        "changed",
        [
            {"generation_options": {"temperature": 0.0}},
            {"generation_options": None},
            {"thinking": False},
            {"host": "https://other-provider.invalid"},
        ],
    )
    def test_resume_rejects_changed_request_before_making_any_call(
        self, tmp_path, monkeypatch, changed
    ):
        import asyncio
        import json
        from unittest.mock import AsyncMock

        original_options = {"generation_options": {"temperature": 0.7}, "thinking": "low"}
        original = make(tmp_path, **original_options)
        monkeypatch.setattr(
            original, "_call_once", AsyncMock(return_value={"message": {"content": "1"}})
        )
        record = asyncio.run(original._run_task("model", "A008", 0, 0))
        original._jsonl_path("model").write_text(json.dumps(record) + "\n")
        resumed = make(tmp_path, **(original_options | changed))
        call = AsyncMock()
        monkeypatch.setattr(resumed, "_call_once", call)
        with pytest.raises(ValueError, match="different request configuration"):
            asyncio.run(resumed.run_model("model"))
        call.assert_not_awaited()

    @pytest.mark.parametrize("missing", ["schema_version", "request", "request_provenance"])
    def test_resume_requires_recorded_provenance_even_with_default_options(self, tmp_path, missing):
        import json

        survey = make(tmp_path)
        record = {
            "llm": "model",
            "question": "A008",
            "system_prompt_id": 0,
            "repeat": 0,
            "schema_version": 2,
            "request": survey._request_for("model", "A008", 0),
            "request_provenance": {"host": survey.host},
        }
        record.pop(missing)
        survey._jsonl_path("model").write_text(json.dumps(record) + "\n")
        with pytest.raises(ValueError, match="new output directory"):
            survey._completed("model")

    def test_exhausted_parse_failure_is_a_completed_trial(self, tmp_path, monkeypatch):
        import asyncio
        from unittest.mock import AsyncMock

        from app.cloud_survey import MAX_ATTEMPTS

        survey = make(tmp_path)
        monkeypatch.setattr(survey, "tasks_for_model", lambda: [("A008", 0, 0)])
        monkeypatch.setattr(
            survey,
            "_call_once",
            AsyncMock(return_value={"message": {"content": "I cannot answer"}}),
        )
        assert asyncio.run(survey.run_model("model")) == {"ok": 0, "failed": 1, "deferred": 0}
        terminal = self.read_records(survey._jsonl_path("model"))
        assert len(terminal) == 1 and terminal[0]["parsed"] is None
        assert terminal[0]["attempts"] == MAX_ATTEMPTS
        assert survey._completed("model") == {("A008", 0, 0)}
        audit = self.read_records(tmp_path / "attempt_audit/model.jsonl")
        assert len(audit) == MAX_ATTEMPTS
        assert {r["outcome"] for r in audit} == {"parse_failure"}

    @pytest.mark.parametrize(
        "status, terminal",
        [
            (400, True),
            (401, True),
            (403, True),
            (404, True),
            (408, False),
            (429, False),
            (500, False),
        ],
    )
    def test_only_non_transient_client_errors_become_provider_rejections(
        self, tmp_path, monkeypatch, status, terminal
    ):
        import asyncio
        from unittest.mock import AsyncMock

        from ollama import ResponseError

        from app import cloud_survey

        survey = make(tmp_path)
        monkeypatch.setattr(survey, "tasks_for_model", lambda: [("A008", 0, 0)])
        monkeypatch.setattr(cloud_survey.asyncio, "sleep", AsyncMock())
        call = AsyncMock(side_effect=ResponseError("rejected", status))
        monkeypatch.setattr(survey, "_call_once", call)
        counts = asyncio.run(survey.run_model("model"))
        audit = self.read_records(tmp_path / "attempt_audit/model.jsonl")
        if terminal:
            assert counts == {"ok": 0, "failed": 1, "deferred": 0}
            assert call.call_count == 1
            (record,) = self.read_records(survey._jsonl_path("model"))
            assert record["error"] == f"provider: ResponseError: rejected (status code: {status})"
            assert record["parsed"] is None and "transport_failure" not in record
            assert [r["outcome"] for r in audit] == ["provider_rejection"]
            assert survey._completed("model") == {("A008", 0, 0)}
        else:
            assert counts == {"ok": 0, "failed": 0, "deferred": 1}
            assert call.call_count == cloud_survey.MAX_ATTEMPTS
            assert not survey._jsonl_path("model").exists()
            assert {r["outcome"] for r in audit} == {"transport_failure"}

    def test_generate_error_is_not_treated_as_throttling(self, tmp_path, monkeypatch):
        import asyncio
        from unittest.mock import AsyncMock

        from ollama import ResponseError

        from app import cloud_survey

        survey = make(tmp_path)
        sleep = AsyncMock()
        monkeypatch.setattr(cloud_survey.asyncio, "sleep", sleep)
        call = AsyncMock(side_effect=ResponseError("failed to generate", 500))
        monkeypatch.setattr(survey, "_call_once", call)
        asyncio.run(survey._run_task("model", "A008", 0, 0))
        assert [c.args[0] for c in sleep.await_args_list] == [2.0, 4.0, 8.0]

    def test_provider_error_echoing_the_key_is_redacted_in_record_and_audit(
        self, tmp_path, monkeypatch
    ):
        import asyncio
        from unittest.mock import AsyncMock

        from ollama import ResponseError

        from app import cloud_survey

        survey = CloudSurvey(out_dir=tmp_path, api_key=DUMMY_KEY)
        monkeypatch.setattr(survey, "tasks_for_model", lambda: [("A008", 0, 0)])
        monkeypatch.setattr(cloud_survey.asyncio, "sleep", AsyncMock())
        call = AsyncMock(
            side_effect=[
                ResponseError(f"upstream failure for bearer {DUMMY_KEY}", 502),
                ResponseError(f"invalid api key {DUMMY_KEY}", 401),
            ]
        )
        monkeypatch.setattr(survey, "_call_once", call)
        assert asyncio.run(survey.run_model("model")) == {"ok": 0, "failed": 1, "deferred": 0}
        terminal_text = survey._jsonl_path("model").read_text()
        audit_path = tmp_path / "attempt_audit/model.jsonl"
        assert DUMMY_KEY not in terminal_text
        assert DUMMY_KEY not in audit_path.read_text()
        (record,) = self.read_records(survey._jsonl_path("model"))
        assert record["error"] == (
            "provider: ResponseError: invalid api key [REDACTED] (status code: 401)"
        )
        audit = self.read_records(audit_path)
        assert [r["outcome"] for r in audit] == ["transport_failure", "provider_rejection"]
        assert audit[0]["error"] == (
            "ResponseError: upstream failure for bearer [REDACTED] (status code: 502)"
        )
        assert audit[1]["error"] == record["error"]

    def test_answered_trial_is_persisted_when_the_final_attempt_times_out(
        self, tmp_path, monkeypatch
    ):
        import asyncio
        from unittest.mock import AsyncMock

        from app import cloud_survey

        survey = make(tmp_path)
        monkeypatch.setattr(survey, "tasks_for_model", lambda: [("A008", 0, 0)])
        monkeypatch.setattr(cloud_survey.asyncio, "sleep", AsyncMock())
        call = AsyncMock(
            side_effect=[
                {"message": {"content": "I cannot answer"}},
                {"message": {"content": "Still no"}},
                TimeoutError("offline fixture"),
            ]
        )
        monkeypatch.setattr(survey, "_call_once", call)
        assert asyncio.run(survey.run_model("model")) == {"ok": 0, "failed": 1, "deferred": 0}
        (record,) = self.read_records(survey._jsonl_path("model"))
        assert record["raw_content"] == "Still no"
        assert record["parsed"] is None and record["error"].startswith("parse: ")
        assert record["attempts"] == cloud_survey.MAX_ATTEMPTS
        assert "transport_failure" not in record
        audit = self.read_records(tmp_path / "attempt_audit/model.jsonl")
        assert [r["outcome"] for r in audit] == [
            "parse_failure",
            "parse_failure",
            "transport_failure",
        ]
        assert survey._completed("model") == {("A008", 0, 0)}

    def test_failed_audit_write_stops_collection(self, tmp_path, monkeypatch):
        import asyncio
        from unittest.mock import AsyncMock

        survey = make(tmp_path)
        call = AsyncMock(return_value={"message": {"content": "1"}})
        monkeypatch.setattr(survey, "_call_once", call)
        monkeypatch.setattr(survey, "_append_record", AsyncMock(side_effect=OSError("disk full")))
        with pytest.raises(OSError, match="disk full"):
            asyncio.run(survey._run_task("model", "A008", 0, 0))
        assert call.call_count == 1
        assert not survey._jsonl_path("model").exists()


def test_load_dotenv_sets_only_unset_keys_and_skips_comments(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\n\nOLLAMA_HOST = https://example.test \nSET_ALREADY=new\nNO_EQUALS\n")
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.setenv("SET_ALREADY", "old")
    monkeypatch.delenv("NO_EQUALS", raising=False)
    load_dotenv(env)
    assert os.environ["OLLAMA_HOST"] == "https://example.test"
    assert os.environ["SET_ALREADY"] == "old"
    assert "NO_EQUALS" not in os.environ
    load_dotenv(tmp_path / "absent.env")  # a missing file is not an error
