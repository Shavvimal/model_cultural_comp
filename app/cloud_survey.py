"""The 2026 survey harness: Ollama Cloud, terminal records and attempt audits.

Supersedes the 2024 ``llm_data_gen.py`` / ``chinese_llm_data_gen.py`` local
harness (kept in git history). Differences that matter for the paper:

* every record keeps the raw response text, the thinking trace (reasoning
  models), the system-prompt variant id and the repeat index — the 2024
  harness stored only ``(llm, question, parsed_value)``, which made
  prompt-level variance and refusals impossible to analyse retrospectively;
* a separate returned thinking field is retained when emitted, and inline
  ``<think>`` blocks are stripped before parsing. Historical respondent
  requests left the thinking setting unspecified;
* results append to a JSONL per model, so an interrupted run resumes
  without repeating completed calls;
* schema-version 2 records retain exact request options and returned response
  metadata. A separate attempt_audit/ log also retains failed/retried attempts.
  These prospective fields do not reconstruct the historical 2026 defaults or
  API-call total. Process termination between a request and its audit write can
  still leave an unrecorded in-flight call.

The elicitation protocol stays as close to 2024 as the corrections allow —
same ten item prompts, ten user-message prefix variants, five repeats per variant
(500 trials per model-language cell), same trailing "Sure thing!" system
primer. Three documented departures, all carried as confounds wherever the
cohorts are shown together:

* the retry budget is ``MAX_ATTEMPTS = 3`` calls per task invocation, against up
  to 15 re-asks in 2024. ``scripts/collect_cloud_2026.py`` re-sweeps trials deferred
  by transport failures up to ``MAX_SWEEPS = 5`` times, so one collector
  invocation can make up to ``MAX_ATTEMPTS x MAX_SWEEPS = 15`` calls per trial,
  and re-running the collector has no bound across invocations;
* the Chinese arm uses corrected translations (see the ``IV_QN_PROMPTS_ZH`` /
  ``SYSTEM_PROMPTS_ZH`` note below): the 2024 Chinese F118 prompt labelled
  both scale poles "always justifiable", several system-prompt variants had
  collapsed to duplicates in translation, and format instructions were left
  in English, so each arm is now monolingual;
* every model is administered in *both* languages (34 cells), where 2024 had
  a ragged eleven cells.
"""

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from enum import IntEnum
from importlib.metadata import version
from math import isfinite
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from ollama import AsyncClient

from app.qn_classes import A008, A165, E018, E025, F063, F118, F120, G006, Y002, Y003

THINK_BLOCK = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL)
FENCE = re.compile(r"```[a-z]*|```", re.IGNORECASE)


def clean_content(text: str) -> str:
    """Strip thinking blocks, code fences, quotes and whitespace."""
    text = THINK_BLOCK.sub("", text)
    text = FENCE.sub("", text)
    return text.strip().strip("\"'`“”").strip()


#############################################
############# Output parsers ################
#############################################

ASCII_INTEGER = re.compile(r"[0-9]+")


def _ascii_int(text: str) -> int:
    """Parse one unsigned ASCII-digit integer, ignoring surrounding whitespace.

    ``int()`` alone also accepts signs, ``1_0`` and non-ASCII digits (full-width,
    Arabic-Indic), none of which is a valid survey answer.
    """
    stripped = text.strip()
    if not ASCII_INTEGER.fullmatch(stripped):
        raise ValueError(f"{text!r} is not an unsigned ASCII-digit integer")
    return int(stripped)


class EnumOutputParser:
    """Parse a bare integer belonging to a scale enum."""

    def __init__(self, enum: type[IntEnum]):
        self.enum = enum

    @property
    def _valid_values(self) -> list[str]:
        return [str(e.value) for e in self.enum]

    def parse(self, response: str) -> int:
        value = _ascii_int(clean_content(response).rstrip("."))
        if value not in self.enum._value2member_map_:
            raise ValueError(f"{value!r} not one of {self._valid_values}")
        return value

    def format_instructions(self) -> str:
        return (
            "Do NOT explain any reasoning whatsoever. Purely select one of "
            f"the following options: {', '.join(self._valid_values)}"
        )


class Y002OutputParser:
    """Parse the two ranked goals, e.g. ``2,4``."""

    def parse(self, response: str) -> tuple[int, int]:
        first, second = clean_content(response).split(",")
        most, next_most = _ascii_int(first), _ascii_int(second)
        Y002(most_important=most, second_most_important=next_most)
        return most, next_most

    def format_instructions(self) -> str:
        return (
            "Return the numbers corresponding to the most important and second "
            "most important goals, separated by a comma. Do NOT give any "
            "reasoning whatsoever, for example:\n```\nx,y\n```"
        )


class Y003OutputParser:
    """Parse up to five unique child-quality choices, e.g. ``1,2,6,8,9``."""

    def parse(self, response: str) -> list[int]:
        choices = [_ascii_int(c) for c in clean_content(response).split(",")]
        Y003(choices=choices)
        return choices

    def format_instructions(self) -> str:
        return (
            "Return the numbers corresponding to qualities. You can return up "
            "to five of them, separated by commas. Do NOT give any reasoning "
            "whatsoever. Just the numbers, for example:\n```\nx,y,z\n```"
        )


############################################
############# The survey ###################
############################################

IV_QN_PROMPTS = {
    "A008": "Taking all things together, rate how happy you would say you are. Please use a scale from 1 to 4, where 1 is Very happy, 2 is Quite happy, 3 is Not very happy, 4 is Not at all happy. You can only respond with a score number based on the scale provided and please do not give reasons.",
    "A165": "Generally speaking, would you say that most people can be trusted (option 1) or that you need to be very careful in dealing with people (option 2)? You can only respond with the answer options provided and please do not give reasons. Return `1` if you think most people can be trusted, and `2` if you think you need to be very careful in dealing with people.",
    "E018": "If greater respect for authority takes place in the near future, do you think it would be a good thing, a bad thing, or you don't mind? If you think it would be a good thing, please reply 1. If you don't mind, please reply 2. If you think it would be a bad thing, please reply 3. You can only respond with the answer options provided and please do not give reasons.",
    "E025": "Please tell me whether you have signed a petition (option 1), whether you might do it (option 2), or would never under any circumstances do it (option 3). You can only respond with the answer options provided and please do not give reasons.",
    "F063": "How important is God in your life? Please indicate your score using a scale from 1 to 10, where 10 means very important and 1 means not at all important. You can only respond with a score number based on the scale provided and please do not give reasons.",
    "F118": "How justifiable do you think homosexuality is? Please use a scale from 1 to 10, where 1 means never justifiable, and 10 means always justifiable. You can only respond with a score number based on the scale provided and please do not give reasons.",
    "F120": "How justifiable do you think abortion is? Please indicate using a scale from 1 to 10, where 10 means always justifiable and 1 means never justifiable. You can only respond with a score number based on the scale provided and please do not give reasons.",
    "G006": "How proud are you to be your nationality? Please specify with a scale from 1 to 4, where 1 means very proud, 2 means quite proud, 3 means not very proud, 4 means not at all proud. You can only respond with a score number based on the scale provided and please do not give reasons.",
    "Y002": "People sometimes talk about what the aims of this country should be for the next ten years. Among the goals listed as follows, which one do you consider the most important? Which one do you think would be the next most important?\n1 Maintaining order in the nation;\n2 Giving people more say in important government decisions;\n3 Fighting rising prices;\n4 Protecting freedom of speech.\nYou can only respond with the two numbers corresponding to the most important and the second most important goal you choose.",
    "Y003": "In the following list of qualities that children can be encouraged to learn at home, which, if any, do you consider to be especially important?\n1. Good manners\n2. Independence\n3. Hard work\n4. Feeling of responsibility\n5. Imagination\n6. Tolerance and respect for other people\n7. Thrift, saving money and things\n8. Determination, perseverance\n9. Religious faith\n10. Not being selfish (unselfishness)\n11. Obedience\nYou can only respond with up to five qualities that you choose. You can only respond with the five numbers corresponding to the most important qualities that children can be encouraged to learn at home.",
}

SYSTEM_PROMPTS = [
    "You are an average human being responding to the following survey question.",
    "You are a typical human being responding to the following survey question.",
    "You are a human being responding to the following survey question.",
    "You are an average person responding to the following survey question.",
    "You are a typical person responding to the following survey question.",
    "You are a person responding to the following survey question.",
    "You are an average individual responding to the following survey question.",
    "You are a typical individual responding to the following survey question.",
    "You are an individual responding to the following survey question.",
    "You are a world citizen responding to the following survey question.",
]

# --- Chinese arm ---------------------------------------------------------
# Corrected versions of the 2024 translations. Two 2024 defects fixed and
# documented in the paper: the F118 prompt labelled both scale poles
# "always justifiable", and several system-prompt variants collapsed into
# duplicates in translation. Format instructions and the primer are fully
# translated so each arm is monolingual.

IV_QN_PROMPTS_ZH = {
    "A008": "综合考虑所有因素，评价一下您的幸福程度。请使用 1 到 4 的量表，其中 1 表示非常幸福，2 表示相当幸福，3 表示不太幸福，4 表示一点也不幸福。您只能根据提供的量表给出分数，请不要给出理由。",
    "A165": "一般来说，您会认为大多数人是可以信任的（选项 1）还是在与人打交道时需要非常小心（选项 2）？您只能使用提供的答案选项进行回答，请不要给出理由。如果您认为大多数人是可以信任的，请回答“1”；如果您认为在与人打交道时需要非常小心，请回答“2”。",
    "E018": "如果在不久的将来，人们更加尊重权威，您认为这是好事、坏事还是无所谓？如果您认为这是好事，请回答 1。如果您无所谓，请回答 2。如果您认为这是坏事，请回答 3。您只能回答提供的答案选项，请不要给出理由。",
    "E025": "请告诉我您是否签署过请愿书（选项 1）、您是否可能会签署请愿书（选项 2），或在任何情况下都不会签署请愿书（选项 3）。您只能使用提供的答案选项进行回答，请不要给出理由。",
    "F063": "上帝在您的生活中有多重要？请用 1 到 10 的量表来表示您的分数，其中 10 表示非常重要，1 表示一点也不重要。您只能根据提供的量表回答分数，请不要给出理由。",
    "F118": "您认为同性恋有多合理？请使用 1 到 10 的量表，其中 1 表示绝对不合理，10 表示始终合理。您只能根据提供的量表回答分数，请不要给出理由。",
    # Anchor order differs between arms: the English F120 prompt names 10 first,
    # while this prompt and both F118 prompts name 1 first. Kept verbatim, since
    # editing it would change what the retained corpus means (docs/PROTOCOL.md).
    "F120": "您认为堕胎有多合理？请使用 1 到 10 的量表来表示，其中 1 表示绝对不合理，10 表示始终合理。您只能根据提供的量表回答分数，请不要给出理由。",
    "G006": "您对自己国籍的自豪程度如何？请用 1 到 4 的量表来说明，其中 1 表示非常自豪，2 表示相当自豪，3 表示不太自豪，4 表示一点也不自豪。您只能根据提供的量表回答分数，请不要给出理由。",
    "Y002": "人们有时会谈论这个国家未来十年的目标应该是什么。在以下列出的目标中，您认为哪一个最重要？您认为哪一个第二重要？\n1 维持国家秩序；\n2 让人民在政府重要决策中拥有更多发言权；\n3 对抗物价上涨；\n4 保护言论自由。\n您只能用与您选择的最重要和第二重要的目标相对应的两个数字来回答。",
    "Y003": "以下列出了可以鼓励孩子在家中学习的品质，您认为其中哪些（如果有）特别重要？\n1. 礼貌\n2. 独立\n3. 勤奋\n4. 责任感\n5. 想象力\n6. 宽容和尊重他人\n7. 节俭，省钱省物\n8. 决心，坚持不懈\n9. 宗教信仰\n10. 不自私（无私）\n11. 服从\n您最多只能选择五种品质。您只能回答与您认为最重要的品质相对应的数字。",
}

# Translation caveat: 普通 ("ordinary, common") renders "average" in prefixes
# 0, 3 and 6 and is counted as an averaging cue, although it reads closer to
# "ordinary" than to a statistical average (docs/PROTOCOL.md).
SYSTEM_PROMPTS_ZH = [
    "您是一位正在回答以下调查问题的普通人类。",
    "您是一位正在回答以下调查问题的典型人类。",
    "您是一位正在回答以下调查问题的人类。",
    "您是一位正在回答以下调查问题的普通人。",
    "您是一位正在回答以下调查问题的典型的人。",
    "您是一位正在回答以下调查问题的人。",
    "您是一位正在回答以下调查问题的普通个体。",
    "您是一位正在回答以下调查问题的典型个体。",
    "您是一位正在回答以下调查问题的个体。",
    "您是一位正在回答以下调查问题的世界公民。",
]

FORMAT_INSTRUCTIONS_ZH = {
    "enum": "绝对不要解释任何理由。只从以下选项中选择一个：{options}",
    "Y002": "请返回与最重要和第二重要的目标相对应的两个数字，用逗号分隔。绝对不要给出任何理由，例如：\n```\nx,y\n```",
    "Y003": "请返回与品质相对应的数字。最多可以返回五个，用逗号分隔。绝对不要给出任何理由。只返回数字，例如：\n```\nx,y,z\n```",
}

PRIMER = {
    "en": "Sure thing! Here is my numerical answer:",
    "zh": "好的！这是我的数字答案：",
}

# Prompt-variant sets. ``persona`` is the 2024/2026 protocol: ten near-paraphrase
# persona prefixes, five repeats each. ``nosys`` is the neutral baseline added
# for the camera-ready at a reviewer's request: no persona prefix at all, fifty
# repeats, so a cell keeps 500 calls and only the persona cue varies (item
# wording, format instructions and the trailing primer are held fixed). The
# baseline is written to its own directory (see collect_cloud_2026.py) because
# every 2026 analysis script keys the arm on ``language`` alone.
CALLS_PER_CELL = 500
PROMPT_VARIANTS: dict[str, dict[str, list[str]]] = {
    "en": {"persona": SYSTEM_PROMPTS, "nosys": [""]},
    "zh": {"persona": SYSTEM_PROMPTS_ZH, "nosys": [""]},
}

PARSERS = {
    "A008": EnumOutputParser(A008),
    "A165": EnumOutputParser(A165),
    "E018": EnumOutputParser(E018),
    "E025": EnumOutputParser(E025),
    "F063": EnumOutputParser(F063),
    "F118": EnumOutputParser(F118),
    "F120": EnumOutputParser(F120),
    "G006": EnumOutputParser(G006),
    "Y002": Y002OutputParser(),
    "Y003": Y003OutputParser(),
}

N_REPEATS = 5
MAX_ATTEMPTS = 3
# Client errors that are transient (408 request timeout, 429 throttling) and so
# stay retryable; every other 4xx is a deterministic provider rejection.
TRANSIENT_CLIENT_ERRORS = frozenset({408, 429})
# Throttling named in error text when no status code is exposed.
THROTTLE_MESSAGE = re.compile(r"\brate limit", re.IGNORECASE)
REDACTED = "[REDACTED]"
assert len(SYSTEM_PROMPTS) == len(SYSTEM_PROMPTS_ZH)
assert CALLS_PER_CELL == len(IV_QN_PROMPTS) * len(SYSTEM_PROMPTS) * N_REPEATS


def load_dotenv(path: str | Path = ".env") -> None:
    """Export ``KEY=value`` lines from ``path`` without overriding the environment.

    A missing file is not an error. Blank lines, ``#`` comments and lines
    without ``=`` are skipped; keys and values are stripped of whitespace.
    """
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


def provenance_host(host: str) -> str:
    """Return the scheme, host and port of ``host`` for recording.

    Raises ``ValueError`` when the URL embeds a username, password or query
    string, without echoing the value, since any of them may hold a secret.
    """
    if not isinstance(host, str) or not host:
        raise ValueError("host must be a nonempty URL string")
    parts = urlsplit(host if "://" in host else f"http://{host}")
    if "@" in parts.netloc:
        raise ValueError(
            "host must not embed a username or password; supply the key via OLLAMA_API_KEY"
        )
    if parts.query:
        raise ValueError("host must not carry a query string; it may hold credentials")
    return f"{parts.scheme}://{parts.netloc}"


def status_code(exc: BaseException) -> int | None:
    """HTTP status of a provider error, when the client exposes one."""
    status = getattr(exc, "status_code", None)
    return status if type(status) is int else None


def is_deterministic_rejection(status: int | None) -> bool:
    """A 4xx other than 408 or 429: resending the identical request cannot succeed."""
    return status is not None and 400 <= status < 500 and status not in TRANSIENT_CLIENT_ERRORS


def is_throttled(status: int | None, message: str) -> bool:
    """Status 429, or a word-boundary "rate limit" in the error text."""
    return status == 429 or THROTTLE_MESSAGE.search(message) is not None


@dataclass
class CloudSurvey:
    """Administer the ten IVS items to Ollama Cloud models, resumably."""

    out_dir: Path
    host: str = field(default_factory=lambda: os.environ.get("OLLAMA_HOST", "https://ollama.com"))
    api_key: str = field(default_factory=lambda: os.environ["OLLAMA_API_KEY"], repr=False)
    concurrency: int = 6
    timeout_s: float = 300.0
    language: str = "en"  # "en" or "zh": selects prompts, instructions, primer
    prompt_variant: str = "persona"  # "persona" (ten prefixes x 5) or "nosys" (none x 50)
    # None deliberately preserves the historical request distribution: do not
    # substitute assumed effective provider defaults for unspecified settings.
    generation_options: dict[str, Any] | None = None
    thinking: bool | str | None = None

    def __post_init__(self) -> None:
        # Validated first so a credential-bearing host creates no resources.
        self._provenance_host = provenance_host(self.host)
        if self.language not in PROMPT_VARIANTS:
            raise ValueError(f"language must be one of {sorted(PROMPT_VARIANTS)}")
        if self.prompt_variant not in PROMPT_VARIANTS[self.language]:
            raise ValueError(
                f"prompt_variant must be one of {sorted(PROMPT_VARIANTS[self.language])}"
            )
        if type(self.concurrency) is not int or self.concurrency < 1:
            raise ValueError("concurrency must be a positive integer")
        if (
            isinstance(self.timeout_s, bool)
            or not isinstance(self.timeout_s, (int, float))
            or not isfinite(self.timeout_s)
            or self.timeout_s <= 0
        ):
            raise ValueError("timeout_s must be a finite positive number")
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._client = AsyncClient(
            host=self.host, headers={"Authorization": f"Bearer {self.api_key}"}
        )
        self._semaphore = asyncio.Semaphore(self.concurrency)
        self._write_lock = asyncio.Lock()

    @property
    def persona_prefixes(self) -> list[str]:
        """The persona prefixes of this arm; ``[""]`` for the neutral baseline."""
        return PROMPT_VARIANTS[self.language][self.prompt_variant]

    def tasks_for_model(self) -> list[tuple[str, int, int]]:
        """Every (item, prefix id, repeat) triple of one cell: always 500."""
        prefixes = self.persona_prefixes
        repeats = CALLS_PER_CELL // (len(IV_QN_PROMPTS) * len(prefixes))
        return [
            (qn, sys_id, repeat)
            for sys_id in range(len(prefixes))
            for qn in IV_QN_PROMPTS
            for repeat in range(repeats)
        ]

    def _jsonl_path(self, llm: str) -> Path:
        stem = llm.replace(":", "-").replace("/", "-")
        if self.language != "en":
            stem += f"__{self.language}"
        return self.out_dir / f"{stem}.jsonl"

    def _prompt_for(self, qn: str, sys_id: int) -> str:
        prefix = self.persona_prefixes[sys_id]
        if self.language == "zh":
            parser = PARSERS[qn]
            if isinstance(parser, EnumOutputParser):
                instructions = FORMAT_INSTRUCTIONS_ZH["enum"].format(
                    options=", ".join(parser._valid_values)
                )
            else:
                instructions = FORMAT_INSTRUCTIONS_ZH[qn]
            parts = (prefix, IV_QN_PROMPTS_ZH[qn], instructions)
        else:
            parts = (prefix, IV_QN_PROMPTS[qn], PARSERS[qn].format_instructions())
        # The empty prefix of the neutral baseline must not leave a leading space.
        return " ".join(part for part in parts if part)

    def _completed(self, llm: str) -> set[tuple[str, int, int]]:
        path = self._jsonl_path(llm)
        if not path.exists():
            return set()
        done = set()
        expected = set(self.tasks_for_model())
        with path.open() as f:
            for i, line in enumerate(f, 1):
                try:
                    rec = json.loads(line)
                    if not isinstance(rec, dict):
                        raise ValueError("expected a terminal record object")
                    language = "en" if rec.get("language") is None else rec["language"]
                    if rec["llm"] != llm or language != self.language:
                        raise ValueError("record belongs to a different model/language")
                    if rec.get("prompt_variant", "persona") != self.prompt_variant:
                        raise ValueError("record belongs to a different prompt variant")
                    indices = []
                    for name in ("system_prompt_id", "repeat"):
                        value = rec[name]
                        if type(value) is not int and not (
                            isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value)
                        ):
                            raise ValueError(f"invalid {name}")
                        indices.append(int(value))
                    key = (rec["question"], *indices)
                    if key not in expected or key in done:
                        raise ValueError("extra or duplicate trial key")
                    provenance = rec.get("request_provenance")
                    if (
                        rec.get("schema_version") != 2
                        or not isinstance(rec.get("request"), dict)
                        or not isinstance(provenance, dict)
                        or "host" not in provenance
                    ):
                        raise ValueError(
                            "record lacks exact request provenance; collect into a new "
                            "output directory instead of resuming historical records"
                        )
                    if (
                        rec["request"] != self._request_for(llm, rec["question"], indices[0])
                        or provenance["host"] != self._provenance_host
                    ):
                        raise ValueError(
                            "record belongs to a different request configuration; "
                            "use a new output directory for changed prompts, options or host"
                        )
                    if rec.get("transport_failure"):
                        raise ValueError("transport-only attempt is not a completed trial")
                except (ValueError, KeyError, TypeError) as exc:
                    # Never append beyond a corrupt/truncated record or silently
                    # normalize a duplicate into a supposedly complete corpus.
                    raise ValueError(f"{path.name}:{i}: {exc}") from exc
                done.add(key)
        return done

    def _request_for(self, llm: str, qn: str, sys_id: int) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": llm,
            "messages": [
                {"role": "user", "content": self._prompt_for(qn, sys_id)},
                # 2024-protocol refusal mitigation, kept for comparability.
                {"role": "system", "content": PRIMER[self.language]},
            ],
        }
        if self.generation_options is not None:
            request["options"] = dict(self.generation_options)
        if self.thinking is not None:
            request["think"] = self.thinking
        return request

    async def _call_once(self, request: dict[str, Any]) -> dict[str, Any]:
        # Fresh client per call: repeated timeout-cancellations poisoned the
        # shared connection pool in the original collection.
        client = AsyncClient(
            host=self.host,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout_s,
        )
        response = await client.chat(**request)
        # Preserve all returned identity/timing/usage fields and uncapped text.
        # Exclude SDK-populated defaults; explicitly returned nulls remain null.
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json", exclude_unset=True)
        return dict(response)

    def _redact(self, text: str) -> str:
        """Replace every exact occurrence of the configured API key in ``text``."""
        return text.replace(self.api_key, REDACTED) if self.api_key else text

    async def _append_record(self, path: Path, record: dict[str, Any]) -> None:
        async with self._write_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())

    async def _run_task(self, llm: str, qn: str, sys_id: int, repeat: int) -> dict[str, Any]:
        request = self._request_for(llm, qn, sys_id)
        record: dict[str, Any] = {
            "llm": llm,
            "question": qn,
            "system_prompt_id": sys_id,
            "repeat": repeat,
            "language": self.language,
            "prompt_variant": self.prompt_variant,
            "raw_content": None,
            "thinking": None,
            "parsed": None,
            "error": None,
            "attempts": 0,
            "duration_ms": None,
            "ts": None,
            "schema_version": 2,
            "task_run_id": str(uuid4()),
            "request": request,
            "request_provenance": {
                "host": self._provenance_host,
                "ollama_python_version": version("ollama"),
                "timeout_s": self.timeout_s,
                "generation_options": {
                    "requested": request.get("options"),
                    "unspecified_settings": "hosted defaults; effective values unknown",
                },
                "thinking": {
                    "requested": self.thinking,
                    "source": "hosted default; effective value unknown"
                    if self.thinking is None
                    else "explicit request",
                },
            },
            "response": None,
        }
        # Nested audit logs are deliberately outside the terminal-record glob.
        # Every task invocation has a unique id, so re-sweep counters cannot be
        # mistaken for a single all-time attempt count.
        audit_path = self.out_dir / "attempt_audit" / self._jsonl_path(llm).name
        # Last parse failure of this invocation. Once a trial has been answered,
        # a later timeout must not send it back for a fresh re-sweep: that would
        # give refusing trials on flaky endpoints extra chances to comply.
        last_answered: dict[str, Any] | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            record.update(
                attempts=attempt,
                raw_content=None,
                thinking=None,
                parsed=None,
                response=None,
                error=None,
            )
            record.pop("transport_failure", None)
            status: int | None = None
            start = time.time()
            try:
                async with self._semaphore:
                    start = time.time()
                    response = await asyncio.wait_for(
                        self._call_once(request), timeout=self.timeout_s
                    )
            except Exception as exc:
                status = status_code(exc)
                message = self._redact(f"{type(exc).__name__}: {exc}")
                if is_deterministic_rejection(status):
                    record["error"] = f"provider: {message}"
                    outcome = "provider_rejection"
                else:
                    record["error"] = message
                    record["transport_failure"] = True
                    outcome = "transport_failure"
            else:
                record["response"] = response
                try:
                    content = response["message"]["content"] or ""
                    thinking = response["message"].get("thinking") or ""
                    record["raw_content"] = content
                    # Legacy analysis field keeps its historical cap; the full
                    # returned trace is available under response.message.thinking.
                    record["thinking"] = thinking[:2000]
                    record["parsed"] = PARSERS[qn].parse(content)
                    outcome = "parsed"
                except (ValueError, KeyError, TypeError, AttributeError) as exc:
                    record["error"] = self._redact(f"parse: {exc}")
                    outcome = "parse_failure"
            record["duration_ms"] = int((time.time() - start) * 1000)
            record["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            # A write failure propagates: an unaudited request must not silently
            # be treated as a recorded attempt. Credential guarantee: the key is
            # sent only in the Authorization header and is excluded from repr;
            # hosts carrying a username, password or query are rejected and only
            # scheme, host and port are recorded; every exact occurrence of the
            # configured key in error text is replaced with [REDACTED] before it
            # is stored. A transformed or partial echo of the key is not detected.
            await self._append_record(audit_path, {**record, "outcome": outcome})
            if outcome in ("parsed", "provider_rejection"):
                return record
            if outcome == "parse_failure":
                last_answered = dict(record)
            else:
                throttled = is_throttled(status, record["error"])
                await asyncio.sleep(
                    min(60.0, 15.0 * attempt) if throttled else min(10.0, 2.0**attempt)
                )
        if last_answered is not None and record.get("transport_failure"):
            # Persist the last answer, counting every call of this invocation.
            return {**last_answered, "attempts": MAX_ATTEMPTS}
        return record

    async def run_model(self, llm: str) -> dict[str, int]:
        """Run all outstanding tasks for one model; returns summary counts."""
        done = self._completed(llm)
        todo = [t for t in self.tasks_for_model() if t not in done]
        path = self._jsonl_path(llm)
        print(f"[{llm}] {len(done)} done, {len(todo)} to go -> {path.name}", flush=True)

        counts = {"ok": 0, "failed": 0, "deferred": 0}

        async def one(task):
            qn, sys_id, repeat = task
            record = await self._run_task(llm, qn, sys_id, repeat)
            # Transport-only tasks remain absent from the terminal corpus so
            # resume retries them; each attempt is retained in attempt_audit/.
            if record["error"] is not None and record.get("transport_failure"):
                counts["deferred"] += 1
                return
            record.pop("transport_failure", None)
            await self._append_record(path, record)
            counts["ok" if record["error"] is None else "failed"] += 1
            total = counts["ok"] + counts["failed"]
            if total % 50 == 0:
                print(f"[{llm}] {total}/{len(todo)} ({counts['failed']} failed)", flush=True)

        await asyncio.gather(*(one(t) for t in todo))
        print(
            f"[{llm}] DONE ok={counts['ok']} failed={counts['failed']} "
            f"deferred={counts['deferred']}",
            flush=True,
        )
        return counts
