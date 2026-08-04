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
    """List models, waiting out account-level throttle windows.

    A 429 here previously crashed the whole run; the hourly budget recovers
    on its own, so patience is the correct behaviour.
    """
    for attempt in range(60):
        try:
            response = await survey._client.list()
            return sorted(m.model for m in response.models)
        except Exception as exc:
            print(f"startup list failed (attempt {attempt + 1}): "
                  f"{str(exc)[:80]} — retrying in 60s", flush=True)
            await asyncio.sleep(60)
    raise RuntimeError("could not list cloud models after 60 attempts")


async def main() -> int:
    load_dotenv()
    from app.cloud_survey import CloudSurvey

    args = sys.argv[1:]
    language = "en"
    if args and args[0] in ("--language=zh", "--language=en"):
        language = args[0].split("=")[1]
        args = args[1:]

    concurrency = int(os.environ.get("SURVEY_CONCURRENCY", "18"))
    model_parallelism = int(os.environ.get("SURVEY_MODEL_PARALLELISM", "3"))
    survey = CloudSurvey(out_dir=RAW_DIR, concurrency=concurrency, language=language)
    models = args or await list_cloud_models(survey)
    print(
        f"Surveying {len(models)} models [{language}] "
        f"(in-flight cap {concurrency}, {model_parallelism} models at a time): "
        + ", ".join(models),
        flush=True,
    )

    # Models run in small parallel batches; the shared semaphore caps total
    # in-flight requests, so per-model progress interleaves without ever
    # exceeding the account's concurrency budget.
    for i in range(0, len(models), model_parallelism):
        batch = models[i : i + model_parallelism]
        await asyncio.gather(*(survey.run_model(llm) for llm in batch))

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
