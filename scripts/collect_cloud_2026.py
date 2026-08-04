"""Phase-3 driver: administer the survey to every Ollama Cloud model.

Reads OLLAMA_API_KEY (and optionally OLLAMA_HOST) from the environment or a
repo-root .env file. Run from the repo root:

    uv run python scripts/collect_cloud_2026.py [model ...]

With no arguments, surveys every model the cloud account can see. Raw
records append to data/collection_2026/<model>.jsonl (resumable — re-running
skips completed calls). After collection, converts parsed rows to
data/collection_2026/pickles/<model>_responses_df.pkl in the 2024
(llm, question, response) format consumed by the bootstrap.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/collection_2026")
PICKLE_DIR = RAW_DIR / "pickles"


def load_dotenv(path=".env"):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


async def list_cloud_models(survey) -> list[str]:
    response = await survey._client.list()
    return sorted(m.model for m in response.models)


def jsonl_to_pickles():
    PICKLE_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for path in sorted(RAW_DIR.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.open()]
        frame = pd.DataFrame(rows)
        # last attempt per (question, system_prompt_id, repeat) is authoritative
        frame = frame.drop_duplicates(
            subset=["question", "system_prompt_id", "repeat"], keep="last"
        )
        ok = frame[frame["error"].isna()].copy()
        out = ok.rename(columns={"parsed": "response"})[["llm", "question", "response"]]
        # Y002 parses to a 2-list in JSON; the 2024 format stores a tuple
        out["response"] = [
            tuple(r) if q == "Y002" and isinstance(r, list) else r
            for q, r in zip(out["question"], out["response"])
        ]
        out.to_pickle(PICKLE_DIR / f"{path.stem}_responses_df.pkl")
        summary.append(
            {
                "llm": frame["llm"].iloc[0] if len(frame) else path.stem,
                "calls": len(frame),
                "parsed": len(ok),
                "parse_rate": round(len(ok) / len(frame), 3) if len(frame) else 0.0,
            }
        )
    report = pd.DataFrame(summary)
    print(report.to_string(index=False))
    report.to_csv(RAW_DIR / "collection_summary.csv", index=False)


async def main() -> int:
    load_dotenv()
    from app.cloud_survey import CloudSurvey

    survey = CloudSurvey(out_dir=RAW_DIR, concurrency=6)
    models = sys.argv[1:] or await list_cloud_models(survey)
    print(f"Surveying {len(models)} models: {', '.join(models)}", flush=True)

    for llm in models:  # sequential per model; concurrency lives inside
        await survey.run_model(llm)

    jsonl_to_pickles()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
