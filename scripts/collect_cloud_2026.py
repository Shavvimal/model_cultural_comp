"""Phase-3 driver: administer the survey to every Ollama Cloud model.

Reads OLLAMA_API_KEY (and optionally OLLAMA_HOST) from the environment or a
repo-root .env file. Run from the repo root:

    uv run python scripts/collect_cloud_2026.py [model ...]

With no arguments, surveys every model the cloud account can see; pass
--language=zh as the first argument for the Chinese arm. Raw records append
to data/collection_2026/<model>[__zh].jsonl (resumable — re-running skips
completed calls). Analysis reads the JSONL directly via
app.llm_bootstrap.load_responses_2026.
"""

import asyncio
import os
import sys
from pathlib import Path

RAW_DIR = Path("data/collection_2026")


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


async def main() -> int:
    load_dotenv()
    from app.cloud_survey import CloudSurvey

    args = sys.argv[1:]
    language = "en"
    if args and args[0] in ("--language=zh", "--language=en"):
        language = args[0].split("=")[1]
        args = args[1:]

    survey = CloudSurvey(out_dir=RAW_DIR, concurrency=6, language=language)
    models = args or await list_cloud_models(survey)
    print(f"Surveying {len(models)} models [{language}]: {', '.join(models)}", flush=True)

    for llm in models:  # sequential per model; concurrency lives inside
        await survey.run_model(llm)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
