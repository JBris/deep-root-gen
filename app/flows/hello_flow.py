#!/usr/bin/env python

######################################
# Imports
######################################

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner

######################################
# Main
######################################


@task
def hello() -> None:
    print("Hello")


task_runner = SequentialTaskRunner()


@flow(
    name="Hello",
    description="Hello description.",
    task_runner=task_runner,
)
def hello_flow() -> None:
    hello.submit()


def main() -> None:
    hello_flow()


if __name__ == "__main__":
    main()
