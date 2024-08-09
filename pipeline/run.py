import os
import sys
import asyncio
from graphrag.index.config import PipelineCSVInputConfig
from graphrag.index.input import load_input
from graphrag.index.run import run_pipeline_with_config

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
        os.path.dirname(os.path.abspath(__file__)), "../bin/settings.yml"
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


if __name__ == "__main__":
    from graphrag.index.cli import index_cli
    import asyncio
    import time

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    asyncio.run(
        index_cli(
            root="../bin",
            verbose= False,
            resume= None,
            memprofile= False,
            nocache= False,
            reporter=None,
            config= None,
            emit= None,
            dryrun= False,
            init= False,
            overlay_defaults= False,
            cli=False,
        )
    )

    print(time.strftime("%Y%m%d-%H%M%S"))

