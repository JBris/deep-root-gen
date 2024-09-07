#!/usr/bin/env python

######################################
# Imports
######################################

import asyncio

from prefect import context, flow, task
from prefect.client import get_client
from prefect.runtime import flow_run
from prefect.task_runners import SequentialTaskRunner

######################################
# Main
######################################


@task
async def hello() -> None:
    print(context.get_run_context().task_run.flow_run_id)
    print("Hello")


@task
async def add_tags(id: str) -> None:
    client = get_client()
    current_flow_run_id = flow_run.id
    tags = flow_run.tags
    tags.append(id)
    await client.update_flow_run(current_flow_run_id, tags=tags)


task_runner = SequentialTaskRunner()


@flow(
    name="Hello",
    description="Hello description.",
    task_runner=task_runner,
)
async def hello_flow() -> None:
    await hello.submit()
    await add_tags("http://localhost:4200")


async def main() -> None:
    await hello_flow()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
