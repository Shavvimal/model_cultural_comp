"""The 2026 survey harness: Ollama Cloud, full raw-response recording.

Supersedes the 2024 ``llm_data_gen.py`` / ``chinese_llm_data_gen.py`` local
harness (kept in git history). Differences that matter for the paper:

* every record keeps the raw response text, the thinking trace (reasoning
  models), the system-prompt variant id and the repeat index — the 2024
  harness stored only ``(llm, question, parsed_value)``, which made
  prompt-level variance and refusals impossible to analyse retrospectively;
* reasoning models are handled by asking the API to separate thinking
  (``think`` capability) and by stripping any inline ``<think>`` blocks
  before parsing;
* results append to a JSONL per model, so an interrupted run resumes
  without repeating completed calls.

The elicitation protocol itself is deliberately identical to 2024 (same ten
item prompts, same ten system-prompt variants, five repeats, same trailing
"Sure thing!" assistant primer) so 2024/2026 comparisons change one variable
at a time.
"""

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

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


class EnumOutputParser:
    """Parse a bare integer belonging to a scale enum."""

    def __init__(self, enum: type[IntEnum]):
        self.enum = enum

    @property
    def _valid_values(self) -> list[str]:
        return [str(e.value) for e in self.enum]

    def parse(self, response: str) -> int:
        value = int(clean_content(response).rstrip("."))
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
        Y002(most_important=int(first), second_most_important=int(second))
        return int(first), int(second)

    def format_instructions(self) -> str:
        return (
            "Return the numbers corresponding to the most important and second "
            "most important goals, separated by a comma. Do NOT give any "
            "reasoning whatsoever, for example:\n```\nx,y\n```"
        )


class Y003OutputParser:
    """Parse up to five unique child-quality choices, e.g. ``1,2,6,8,9``."""

    def parse(self, response: str) -> list[int]:
        choices = [int(c) for c in clean_content(response).split(",")]
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
    "F120": "您认为堕胎有多合理？请使用 1 到 10 的量表来表示，其中 1 表示绝对不合理，10 表示始终合理。您只能根据提供的量表回答分数，请不要给出理由。",
    "G006": "您对自己国籍的自豪程度如何？请用 1 到 4 的量表来说明，其中 1 表示非常自豪，2 表示相当自豪，3 表示不太自豪，4 表示一点也不自豪。您只能根据提供的量表回答分数，请不要给出理由。",
    "Y002": "人们有时会谈论这个国家未来十年的目标应该是什么。在以下列出的目标中，您认为哪一个最重要？您认为哪一个第二重要？\n1 维持国家秩序；\n2 让人民在政府重要决策中拥有更多发言权；\n3 对抗物价上涨；\n4 保护言论自由。\n您只能用与您选择的最重要和第二重要的目标相对应的两个数字来回答。",
    "Y003": "以下列出了可以鼓励孩子在家中学习的品质，您认为其中哪些（如果有）特别重要？\n1. 礼貌\n2. 独立\n3. 勤奋\n4. 责任感\n5. 想象力\n6. 宽容和尊重他人\n7. 节俭，省钱省物\n8. 决心，坚持不懈\n9. 宗教信仰\n10. 不自私（无私）\n11. 服从\n您最多只能选择五种品质。您只能回答与您认为最重要的品质相对应的数字。",
}

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


@dataclass
class CloudSurvey:
    """Administer the ten IVS items to Ollama Cloud models, resumably."""

    out_dir: Path
    host: str = field(default_factory=lambda: os.environ.get("OLLAMA_HOST", "https://ollama.com"))
    api_key: str = field(default_factory=lambda: os.environ["OLLAMA_API_KEY"])
    concurrency: int = 6
    timeout_s: float = 300.0
    language: str = "en"  # "en" or "zh": selects prompts, instructions, primer

    def __post_init__(self):
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._client = AsyncClient(
            host=self.host, headers={"Authorization": f"Bearer {self.api_key}"}
        )
        self._semaphore = asyncio.Semaphore(self.concurrency)
        self._write_lock = asyncio.Lock()

    @staticmethod
    def tasks_for_model() -> list[tuple[str, int, int]]:
        return [
            (qn, sys_id, repeat)
            for sys_id in range(len(SYSTEM_PROMPTS))
            for qn in IV_QN_PROMPTS
            for repeat in range(N_REPEATS)
        ]

    def _jsonl_path(self, llm: str) -> Path:
        stem = llm.replace(":", "-").replace("/", "-")
        if self.language != "en":
            stem += f"__{self.language}"
        return self.out_dir / f"{stem}.jsonl"

    def _prompt_for(self, qn: str, sys_id: int) -> str:
        if self.language == "zh":
            parser = PARSERS[qn]
            if isinstance(parser, EnumOutputParser):
                instructions = FORMAT_INSTRUCTIONS_ZH["enum"].format(
                    options=", ".join(parser._valid_values)
                )
            else:
                instructions = FORMAT_INSTRUCTIONS_ZH[qn]
            return SYSTEM_PROMPTS_ZH[sys_id] + " " + IV_QN_PROMPTS_ZH[qn] + " " + instructions
        return (
            SYSTEM_PROMPTS[sys_id]
            + " "
            + IV_QN_PROMPTS[qn]
            + " "
            + PARSERS[qn].format_instructions()
        )

    def _completed(self, llm: str) -> set[tuple[str, int, int]]:
        path = self._jsonl_path(llm)
        if not path.exists():
            return set()
        done = set()
        with path.open() as f:
            for line in f:
                rec = json.loads(line)
                done.add((rec["question"], rec["system_prompt_id"], rec["repeat"]))
        return done

    async def _call_once(self, llm: str, qn: str, sys_id: int):
        messages = [
            {"role": "user", "content": self._prompt_for(qn, sys_id)},
            # 2024-protocol refusal mitigation, kept for comparability
            {"role": "system", "content": PRIMER[self.language]},
        ]
        response = await self._client.chat(model=llm, messages=messages)
        content = response["message"]["content"] or ""
        thinking = response["message"].get("thinking") or ""
        return content, thinking

    async def _run_task(self, llm: str, qn: str, sys_id: int, repeat: int) -> dict:
        record = {
            "llm": llm,
            "question": qn,
            "system_prompt_id": sys_id,
            "repeat": repeat,
            "language": self.language,
            "raw_content": None,
            "thinking": None,
            "parsed": None,
            "error": None,
            "attempts": 0,
            "duration_ms": None,
            "ts": None,
        }
        for attempt in range(1, MAX_ATTEMPTS + 1):
            record["attempts"] = attempt
            record.pop("transport_failure", None)  # each attempt reclassifies
            start = time.time()
            try:
                async with self._semaphore:
                    start = time.time()  # reset inside the semaphore: latency, not queue-wait
                    content, thinking = await asyncio.wait_for(
                        self._call_once(llm, qn, sys_id), timeout=self.timeout_s
                    )
                record["raw_content"] = content
                record["thinking"] = thinking[:2000]
                record["duration_ms"] = int((time.time() - start) * 1000)
                record["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                record["parsed"] = PARSERS[qn].parse(content)
                record["error"] = None
                return record
            except (ValueError, KeyError) as exc:  # parse failure: keep raw, retry
                record["error"] = f"parse: {exc}"
                record["duration_ms"] = int((time.time() - start) * 1000)
                record["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            # Broad catch is deliberate: any transport/API error (timeout,
            # rate limit, 5xx) is expected at this volume, the record keeps
            # the error string so nothing fails silently, and the degradation
            # is a logged failed row that the analysis stage reports.
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                record["error"] = message
                record["duration_ms"] = int((time.time() - start) * 1000)
                record["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                record["transport_failure"] = True
                # Rate limits back off long; other transport errors briefly
                throttled = "429" in message or "rate" in message.lower()
                await asyncio.sleep(
                    min(60.0, 15.0 * attempt) if throttled else min(10.0, 2.0**attempt)
                )
        return record

    async def run_model(self, llm: str) -> dict:
        """Run all outstanding tasks for one model; returns summary counts."""
        done = self._completed(llm)
        todo = [t for t in self.tasks_for_model() if t not in done]
        path = self._jsonl_path(llm)
        print(f"[{llm}] {len(done)} done, {len(todo)} to go -> {path.name}", flush=True)

        counts = {"ok": 0, "failed": 0, "deferred": 0}

        async def one(task):
            qn, sys_id, repeat = task
            record = await self._run_task(llm, qn, sys_id, repeat)
            # Exhausted transport failures are NOT persisted: a written record
            # marks the call complete forever (resume skips it), and a rate
            # limit is not an answer. Unwritten rows are retried on resume.
            if record["error"] is not None and record.get("transport_failure"):
                counts["deferred"] += 1
                return
            record.pop("transport_failure", None)
            async with self._write_lock:
                with path.open("a") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
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
