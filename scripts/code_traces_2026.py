"""Re-code the 900 sampled reasoning traces with independent LLM annotators.

Run from the repo root with OLLAMA_API_KEY in the environment or .env:

    uv run python scripts/code_traces_2026.py --annotator gpt-oss:120b
    uv run python scripts/code_traces_2026.py --annotator nemotron-3-super
    uv run python scripts/code_traces_2026.py --annotator glm-5.3
    uv run python scripts/code_traces_2026.py --annotator mistral-large-3:675b
    uv run python scripts/code_traces_2026.py --annotator gemma4:31b
    uv run python scripts/code_traces_2026.py --worksheet   # blind human subsample
    uv run python scripts/code_traces_2026.py --merge       # -> data/trace_labels_2026.csv

Each annotator pass reads the frozen sample data/trace_samples_2026.json
(the 900 error-free traces, 30 per trace-emitting cell, 3 per item; drawn
by sample_traces_2026.py with seed 42), sends every trace once under the
fixed codebook in app.trace_codebook (temperature 0, seed 42, one trace per
call), and appends to data/trace_labels_2026__<annotator>.jsonl, resumably.
Nothing about the original coding (counts, verbatims, rules) is shown to
the annotator.

--worksheet writes data/trace_coding_human_worksheet.csv: 150 traces (5 per
cell, one item each, seed 42, shuffled) with blank label columns for a
human to fill in blind. Save the completed sheet as
data/trace_labels_human_2026.csv and --merge picks it up as annotator
"human".

--merge stacks every annotator file into the long CSV consumed by
trace_agreement_2026.py, adding the mechanical reasoning-language code.
"""

import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from app.cloud_survey import (
    IV_QN_PROMPTS,
    SYSTEM_PROMPTS,
    SYSTEM_PROMPTS_ZH,
    is_throttled,
    load_dotenv,
    status_code,
)
from app.study_design import TRIAL_KEY
from app.trace_codebook import CODEBOOK, CODES, build_unit, parse_labels, reasoning_language
from app.trace_diagnostics import validate_trace_labels

SAMPLE = Path("data/trace_samples_2026.json")
LABELS_GLOB = "trace_labels_2026__*.jsonl"
MERGED = Path("data/trace_labels_2026.csv")
HUMAN = Path("data/trace_labels_human_2026.csv")
WORKSHEET = Path("data/trace_coding_human_worksheet.csv")
KEY = list(TRIAL_KEY)
SEED = 42
PER_CELL_HUMAN = 5
MAX_ATTEMPTS = 3
MAX_THROTTLE_WAITS = 20  # a 429 is not an answer: wait it out rather than persist it
TEMPERATURE = 0.0


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def load_traces() -> pd.DataFrame:
    """The 900 coded traces, in a stable order."""
    records = json.loads(SAMPLE.read_text())
    rows = [r for r in records if r.get("thinking") and _is_missing(r.get("error"))]
    df = pd.DataFrame(rows)
    df["system_prompt_id"] = df["system_prompt_id"].astype(int)
    df["repeat"] = df["repeat"].astype(int)
    df = df.sort_values(KEY).reset_index(drop=True)
    if len(df) != 900:
        raise RuntimeError(f"expected 900 traces, found {len(df)}")
    return df


def persona_prefix(language: str, sys_id: int) -> str:
    return (SYSTEM_PROMPTS_ZH if language == "zh" else SYSTEM_PROMPTS)[sys_id]


def unit_for(row) -> str:
    return build_unit(
        language=row["language"],
        question=row["question"],
        item_prompt=IV_QN_PROMPTS[row["question"]],
        persona_prefix=persona_prefix(row["language"], int(row["system_prompt_id"])),
        thinking=row["thinking"],
        final_answer=str(row["raw_content"]),
    )


def labels_path(annotator: str) -> Path:
    return (
        Path("data") / f"trace_labels_2026__{annotator.replace(':', '-').replace('/', '-')}.jsonl"
    )


def read_label_records(path: Path) -> list[dict]:
    """Every non-blank JSONL record in one annotator file.

    A corrupt line raises with ``path:line`` for both resume and merge, so a
    truncated or damaged record is repaired by hand rather than skipped on one
    path and fatal on the other.
    """
    records = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path}:{number}: corrupt label record ({exc.msg}); repair or remove the line"
            ) from exc
    return records


def completed(path: Path) -> set[tuple]:
    done = set()
    if path.exists():
        for rec in read_label_records(path):
            # A persisted failure (transport or parse after MAX_ATTEMPTS) is not
            # a label: leave it out of ``done`` so a resume re-asks it, and let
            # --merge keep the last record per key.
            if rec.get("error") is None:
                done.add(tuple(rec[k] for k in KEY))
    return done


async def annotate(annotator: str, concurrency: int, timeout_s: float, limit: int | None) -> int:
    from ollama import AsyncClient

    host = os.environ.get("OLLAMA_HOST", "https://ollama.com")
    api_key = os.environ["OLLAMA_API_KEY"]
    traces = load_traces()
    path = labels_path(annotator)
    done = completed(path)
    todo = [r for _, r in traces.iterrows() if tuple(r[k] for k in KEY) not in done]
    if limit is not None:
        todo = todo[:limit]
    print(f"[{annotator}] {len(done)} done, {len(todo)} to go -> {path.name}", flush=True)
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    counts = {"ok": 0, "failed": 0}

    async def one(row):
        record = {k: (int(row[k]) if k in ("system_prompt_id", "repeat") else row[k]) for k in KEY}
        record.update({"annotator": annotator, "raw_response": None, "error": None, "attempts": 0})
        attempt = 0
        throttle_waits = 0
        while attempt < MAX_ATTEMPTS:
            attempt += 1
            record["attempts"] = attempt
            start = time.time()
            try:
                async with sem:
                    client = AsyncClient(
                        host=host, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout_s
                    )
                    response = await asyncio.wait_for(
                        client.chat(
                            model=annotator,
                            messages=[
                                {"role": "system", "content": CODEBOOK},
                                {"role": "user", "content": unit_for(row)},
                            ],
                            options={"temperature": TEMPERATURE, "seed": SEED},
                        ),
                        timeout=timeout_s,
                    )
                content = response["message"]["content"] or ""
                record["raw_response"] = content[:1000]
                record["duration_ms"] = int((time.time() - start) * 1000)
                record.update(parse_labels(content))
                record["error"] = None
                break
            except ValueError as exc:  # unparseable reply: retry, keep the text
                record["error"] = f"parse: {exc}"[:300]
            except Exception as exc:  # transport / API: back off and retry
                message = f"{type(exc).__name__}: {exc}"
                record["error"] = message[:300]
                throttled = is_throttled(status_code(exc), message)
                if throttled and throttle_waits < MAX_THROTTLE_WAITS:
                    # Throttling does not consume an attempt; the budget recovers.
                    throttle_waits += 1
                    attempt -= 1
                    await asyncio.sleep(min(120.0, 20.0 * throttle_waits))
                else:
                    await asyncio.sleep(min(10.0, 2.0**attempt))
        record["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        async with lock:
            with path.open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        counts["ok" if record["error"] is None else "failed"] += 1
        total = counts["ok"] + counts["failed"]
        if total % 50 == 0:
            print(f"[{annotator}] {total}/{len(todo)} ({counts['failed']} failed)", flush=True)

    await asyncio.gather(*(one(r) for r in todo))
    print(f"[{annotator}] DONE ok={counts['ok']} failed={counts['failed']}", flush=True)
    return 0 if counts["failed"] == 0 else 1


def write_worksheet() -> int:
    traces = load_traces()
    rng = np.random.default_rng(SEED)
    picked = []
    for _, cell in traces.groupby(["llm", "language"], sort=True):
        items = rng.choice(np.sort(cell["question"].unique()), size=PER_CELL_HUMAN, replace=False)
        for q in items:
            pool = cell[cell["question"] == q]
            picked.append(pool.iloc[rng.integers(len(pool))])
    sheet = pd.DataFrame(picked).reset_index(drop=True)
    sheet = sheet.iloc[rng.permutation(len(sheet))].reset_index(drop=True)
    sheet.insert(0, "worksheet_id", np.arange(1, len(sheet) + 1))
    sheet["persona_prefix"] = [
        persona_prefix(lang, int(s))
        for lang, s in zip(sheet["language"], sheet["system_prompt_id"], strict=True)
    ]
    sheet["item_prompt_en"] = sheet["question"].map(IV_QN_PROMPTS)
    sheet["final_answer"] = sheet["raw_content"].astype(str)
    for code in CODES:
        sheet[code] = ""
    cols = [
        "worksheet_id",
        *KEY,
        "persona_prefix",
        "item_prompt_en",
        "thinking",
        "final_answer",
        *CODES,
    ]
    sheet[cols].to_csv(WORKSHEET, index=False)
    print(
        f"wrote {len(sheet)} traces to {WORKSHEET}\n"
        f"Fill the three code columns with 0/1 using the codebook in app/trace_codebook.py,\n"
        f"without looking at any LLM labels, then save as {HUMAN} (same columns)."
    )
    return 0


def merge() -> int:
    frames = []
    for path in sorted(Path("data").glob(LABELS_GLOB)):
        df = pd.DataFrame(read_label_records(path))
        df = df.drop_duplicates(subset=KEY, keep="last")
        frames.append(df[["annotator", *KEY, *CODES, "error"]] if "error" in df else df)
    if HUMAN.exists():
        human = pd.read_csv(HUMAN)
        uncoded = human[list(CODES)].isna().any(axis=1)
        # A worksheet row with any blank code is not yet coded, so it is left
        # out of the human panel; the count is printed so a partial sheet is visible.
        print(
            f"human worksheet: dropped {int(uncoded.sum())} of {len(human)} rows with a blank code"
        )
        human = human.loc[~uncoded]
        human["annotator"] = "human"
        human["error"] = None
        frames.append(human[["annotator", *KEY, *CODES, "error"]])
    if not frames:
        raise FileNotFoundError("no annotator files found")
    long = pd.concat(frames, ignore_index=True)
    traces = load_traces()
    lang = {
        tuple(r[k] for k in KEY): reasoning_language(r["thinking"]) for _, r in traces.iterrows()
    }
    long["reasoning_language"] = [lang.get(tuple(r[k] for k in KEY)) for _, r in long.iterrows()]
    # Validate before the integer cast: a corrupt stored code must raise here,
    # not become NA and then pass downstream as a genuinely missing label.
    long = validate_trace_labels(long)
    for code in CODES:
        long[code] = long[code].astype("Int64")
    long = long.sort_values(["annotator", *KEY]).reset_index(drop=True)
    long.to_csv(MERGED, index=False)
    summary = long.groupby("annotator").agg(
        n=("llm", "size"), failed=("error", lambda s: s.notna().sum())
    )
    print(summary.to_string())
    print(f"wrote {len(long)} rows to {MERGED}")
    return 0


def main() -> int:
    load_dotenv()
    args = sys.argv[1:]
    if args[:1] == ["--worksheet"]:
        return write_worksheet()
    if args[:1] == ["--merge"]:
        return merge()
    if args[:1] == ["--annotator"] and len(args) >= 2:
        annotator = args[1]
        limit = None
        rest = args[2:]
        while rest:
            flag, _, value = rest.pop(0).partition("=")
            if flag == "--limit":
                limit = int(value)
            else:
                print(f"unknown flag {flag}", file=sys.stderr)
                return 2
        concurrency = int(os.environ.get("SURVEY_CONCURRENCY", "6"))
        timeout_s = float(os.environ.get("SURVEY_TIMEOUT_S", "300"))
        return asyncio.run(annotate(annotator, concurrency, timeout_s, limit))
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
