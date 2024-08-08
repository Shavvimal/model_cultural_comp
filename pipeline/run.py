import asyncio
import os
import sys

from graphrag.index import run_pipeline, run_pipeline_with_config
from graphrag.index.config import PipelineCSVInputConfig, PipelineWorkflowReference
from graphrag.index.input import load_input

import asyncio
import json
import logging
import platform
import sys
import time
import warnings
from pathlib import Path

from graphrag.config import (
    create_graphrag_config,
)
from graphrag.index import PipelineConfig, create_pipeline_config
from graphrag.index.cache import NoopPipelineCache
from graphrag.index.progress import (
    NullProgressReporter,
    PrintProgressReporter,
    ProgressReporter,
)
from graphrag.index.progress.rich import RichProgressReporter
from graphrag.index.run import run_pipeline_with_config
from graphrag.index.emit import TableEmitterType


'''Append the package root to sys.path'''
'''Now equivalent to running this script within graphrag with a runner.py script'''
PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'graphrag'))
sys.path.append(PACKAGE_ROOT)

data_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../bin/data"
)

shared_dataset = asyncio.run(
    load_input(
        PipelineCSVInputConfig(
            file_pattern=".*\\.csv$",
            base_dir=data_dir,
            source_column="uuid",
            text_column="text",
            timestamp_column="date",
            timestamp_format="%Y-%m-%d %H:%M:%S",
            title_column="text",
        ),
    )
)


async def run_with_config():
    """Run a pipeline with a config file"""
    # We're cheap, and this is an example, lets just do 10
    dataset = shared_dataset.head(10)

    print(dataset)

    # load pipeline.yml in this directory
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "./pipeline.yml"
    )

    # Grab the last result from the pipeline, should be our entity extraction
    tables = []
    async for table in run_pipeline_with_config(
        config_or_path=config_path, dataset=dataset
    ):
        tables.append(table)
    pipeline_result = tables[-1]

    # Print the entities.  This will be a row for each text unit, each with a list of entities,
    # This should look pretty close to the python version, but since we're using an LLM
    # it will be a little different depending on how it feels about the text
    if pipeline_result.result is not None:
        print(pipeline_result.result["entities"].to_list())
    else:
        print("No results!")


def run(
    root: str,
    init: bool,
    verbose: bool,
    resume: str | None,
    memprofile: bool,
    nocache: bool,
    reporter: str | None,
    config: str | None,
    emit: str | None,
    dryrun: bool,
    overlay_defaults: bool,
    cli: bool = False,
):
    """Run the pipeline with the given config."""
    run_id = resume or time.strftime("%Y%m%d-%H%M%S")
    _enable_logging(root, run_id, verbose)
    progress_reporter = _get_progress_reporter(reporter)
    if init:
        _initialize_project_at(root, progress_reporter)
        sys.exit(0)
    if overlay_defaults:
        pipeline_config: str | PipelineConfig = _create_default_config(
            root, config, verbose, dryrun or False, progress_reporter
        )
    else:
        pipeline_config: str | PipelineConfig = config or _create_default_config(
            root, None, verbose, dryrun or False, progress_reporter
        )
    cache = NoopPipelineCache() if nocache else None
    pipeline_emit = emit.split(",") if emit else None
    encountered_errors = False

    def _run_workflow_async() -> None:
        import signal

        def handle_signal(signum, _):
            # Handle the signal here
            progress_reporter.info(f"Received signal {signum}, exiting...")
            progress_reporter.dispose()
            for task in asyncio.all_tasks():
                task.cancel()
            progress_reporter.info("All tasks cancelled. Exiting...")

        # Register signal handlers for SIGINT and SIGHUP
        signal.signal(signal.SIGINT, handle_signal)

        if sys.platform != "win32":
            signal.signal(signal.SIGHUP, handle_signal)

        async def execute():
            nonlocal encountered_errors
            async for output in run_pipeline_with_config(
                pipeline_config,
                run_id=run_id,
                memory_profile=memprofile,
                cache=cache,
                progress_reporter=progress_reporter,
                emit=(
                    [TableEmitterType(e) for e in pipeline_emit]
                    if pipeline_emit
                    else None
                ),
                is_resume_run=bool(resume),
            ):
                if output.errors and len(output.errors) > 0:
                    encountered_errors = True
                    progress_reporter.error(output.workflow)
                else:
                    progress_reporter.success(output.workflow)

                progress_reporter.info(str(output.result))

        if platform.system() == "Windows":
            import nest_asyncio  # type: ignore Ignoring because out of windows this will cause an error

            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            loop.run_until_complete(execute())
        elif sys.version_info >= (3, 11):
            import uvloop  # type: ignore Ignoring because on windows this will cause an error

            with asyncio.Runner(loop_factory=uvloop.new_event_loop) as runner:  # type: ignore Ignoring because minor versions this will throw an error
                runner.run(execute())
        else:
            import uvloop  # type: ignore Ignoring because on windows this will cause an error

            uvloop.install()
            asyncio.run(execute())

    _run_workflow_async()
    progress_reporter.stop()
    if encountered_errors:
        progress_reporter.error(
            "Errors occurred during the pipeline run, see logs for more details."
        )
    else:
        progress_reporter.success("All workflows completed successfully.")

    if cli:
        sys.exit(1 if encountered_errors else 0)

if __name__ == "__main__":
    from graphrag.index.cli import index_cli
    import asyncio
    import time

    # loop = asyncio.new_event_loop()
    # asyncio.set_event_loop(loop)
    #
    # asyncio.run(
    #     index_cli(
    #         root="../bin",
    #         verbose= False,
    #         resume= None,
    #         memprofile= False,
    #         nocache= False,
    #         reporter=None,
    #         config= None,
    #         emit= None,
    #         dryrun= False,
    #         init= False,
    #         overlay_defaults= False,
    #         cli=False,
    #     )
    # )

    print(time.strftime("%Y%m%d-%H%M%S"))

