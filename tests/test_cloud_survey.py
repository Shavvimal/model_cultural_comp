"""Prompt-composition and task-enumeration contracts of the survey harness.

No network: ``CloudSurvey.__post_init__`` only constructs a client object.
"""

import pytest

from app.cloud_survey import (
    CALLS_PER_CELL,
    IV_QN_PROMPTS,
    N_REPEATS,
    SYSTEM_PROMPTS,
    SYSTEM_PROMPTS_ZH,
    CloudSurvey,
)

AVERAGING_CUES = ("average", "typical", "普通", "典型")


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
