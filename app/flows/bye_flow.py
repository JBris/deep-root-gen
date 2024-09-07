#!/usr/bin/env python

######################################
# Imports
######################################

from prefect import context, flow, task
from prefect.task_runners import SequentialTaskRunner

######################################
# Main
######################################


@task
def bye() -> None:
    print(context.get_run_context().task_run.flow_run_id)
    print("Bye")


task_runner = SequentialTaskRunner()


@flow(
    name="Bye",
    description="Bye description.",
    task_runner=task_runner,
)
def bye_flow() -> None:
    bye.submit()


def main() -> None:
    bye_flow()


if __name__ == "__main__":
    main()
