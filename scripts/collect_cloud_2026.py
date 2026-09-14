"""Phase-3 driver: administer the survey to every Ollama Cloud model.

Reads OLLAMA_API_KEY (and optionally OLLAMA_HOST) from the environment or a
repo-root .env file. Run from the repo root:

    uv run python scripts/collect_cloud_2026.py [model ...]

With no arguments, surveys every model the cloud account can see; pass
--language=zh for the Chinese arm and --prompt-variant=nosys for the
no-persona baseline (flags may come in either order, before the model
list). Raw records append to data/collection_2026/<model>[__zh].jsonl for
the persona protocol and to data/collection_2026_nosys/ for the baseline
(resumable — re-running skips completed calls). The baseline gets its own
directory because the 2026 analysis scripts key the arm on ``language``
alone and would silently pool a third arm into ``en``. Analysis reads the
JSONL directly via app.llm_bootstrap.load_responses_2026.
"""

import asyncio
import os
import sys
from pathlib import Path

RAW_DIRS = {
    "persona": Path("data/collection_2026"),
    "nosys": Path("data/collection_2026_nosys"),
}
RAW_DIR = RAW_DIRS["persona"]
# Outer re-sweeps of trials deferred by transport failures, values unchanged from
# the original collection. The backoff lets transient throttling clear between
# sweeps; the cap stops a persistently failing provider holding a run open.
# Each sweep gives a deferred trial up to MAX_ATTEMPTS fresh calls, so one
# invocation makes at most MAX_ATTEMPTS x MAX_SWEEPS calls per trial (15);
# re-running the collector has no bound across invocations.
MAX_SWEEPS = 5
SWEEP_BACKOFF_S = 120


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
            print(
                f"startup list failed (attempt {attempt + 1}): "
                f"{survey._redact(str(exc))[:80]} — retrying in 60s",
                flush=True,
            )
            await asyncio.sleep(60)
    raise RuntimeError("could not list cloud models after 60 attempts")


async def main() -> int:
    # CloudSurvey reads OLLAMA_API_KEY when constructed, so .env must load first.
    from app.cloud_survey import CloudSurvey, load_dotenv

    load_dotenv()

    args = sys.argv[1:]
    language = "en"
    prompt_variant = "persona"
    while args and args[0].startswith("--"):
        flag, _, value = args.pop(0).partition("=")
        if flag == "--language" and value in ("en", "zh"):
            language = value
        elif flag == "--prompt-variant" and value in RAW_DIRS:
            prompt_variant = value
        else:
            print(f"unknown flag {flag}={value}", file=sys.stderr)
            return 2

    concurrency = int(os.environ.get("SURVEY_CONCURRENCY", "18"))
    model_parallelism = int(os.environ.get("SURVEY_MODEL_PARALLELISM", "3"))
    timeout_s = float(os.environ.get("SURVEY_TIMEOUT_S", "300"))
    survey = CloudSurvey(
        out_dir=RAW_DIRS[prompt_variant],
        concurrency=concurrency,
        language=language,
        timeout_s=timeout_s,
        prompt_variant=prompt_variant,
    )
    models = args or await list_cloud_models(survey)
    print(
        f"Surveying {len(models)} models [{language}, {prompt_variant}] "
        f"(in-flight cap {concurrency}, {model_parallelism} models at a time): "
        + ", ".join(models),
        flush=True,
    )

    # Models run in parallel batches; the shared semaphore caps total
    # in-flight requests. The outer loop re-sweeps deferred transport
    # failures until every cell converges (refusals are persisted and count
    # as complete), so a single invocation reaches 100% or reports why not.
    for sweep in range(1, MAX_SWEEPS + 1):
        deferred_total = 0
        for i in range(0, len(models), model_parallelism):
            batch = models[i : i + model_parallelism]
            counts = await asyncio.gather(*(survey.run_model(llm) for llm in batch))
            deferred_total += sum(c["deferred"] for c in counts)
        if deferred_total == 0:
            print(f"converged after sweep {sweep}", flush=True)
            return 0
        print(
            f"sweep {sweep}: {deferred_total} deferred rows remain; "
            f"re-sweeping in {SWEEP_BACKOFF_S}s",
            flush=True,
        )
        await asyncio.sleep(SWEEP_BACKOFF_S)

    print(
        f"WARNING: deferred rows remain after {MAX_SWEEPS} sweeps — provider persistently failing",
        flush=True,
    )
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
