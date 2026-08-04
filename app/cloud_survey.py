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
        return self.out_dir / f"{llm.replace(':', '-').replace('/', '-')}.jsonl"

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
        prompt = (
            SYSTEM_PROMPTS[sys_id]
            + " "
            + IV_QN_PROMPTS[qn]
            + " "
            + PARSERS[qn].format_instructions()
        )
        messages = [
            {"role": "user", "content": prompt},
            # 2024-protocol refusal mitigation, kept for comparability
            {"role": "system", "content": "Sure thing! Here is my numerical answer:"},
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
            start = time.time()
            try:
                async with self._semaphore:
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
            except Exception as exc:  # noqa: BLE001
                record["error"] = f"{type(exc).__name__}: {exc}"
                record["duration_ms"] = int((time.time() - start) * 1000)
                record["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                await asyncio.sleep(min(10.0, 2.0**attempt))
        return record

    async def run_model(self, llm: str) -> dict:
        """Run all outstanding tasks for one model; returns summary counts."""
        done = self._completed(llm)
        todo = [t for t in self.tasks_for_model() if t not in done]
        path = self._jsonl_path(llm)
        print(f"[{llm}] {len(done)} done, {len(todo)} to go -> {path.name}", flush=True)

        counts = {"ok": 0, "failed": 0}

        async def one(task):
            qn, sys_id, repeat = task
            record = await self._run_task(llm, qn, sys_id, repeat)
            async with self._write_lock:
                with path.open("a") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts["ok" if record["error"] is None else "failed"] += 1
            total = counts["ok"] + counts["failed"]
            if total % 50 == 0:
                print(f"[{llm}] {total}/{len(todo)} ({counts['failed']} failed)", flush=True)

        await asyncio.gather(*(one(t) for t in todo))
        print(f"[{llm}] DONE ok={counts['ok']} failed={counts['failed']}", flush=True)
        return counts
