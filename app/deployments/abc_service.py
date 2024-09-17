#!/usr/bin/env python

######################################
# Imports
######################################

import bentoml
import pandas as pd
from bentoml.io import JSON, PandasDataFrame
from pydantic import BaseModel

######################################
# Constants
######################################

TASK = "abc"

######################################
# Main
######################################

runner = bentoml.mlflow.get(f"{TASK}:latest").to_runner()

svc = bentoml.Service(TASK, runners=[runner])


class AbcFeatures(BaseModel):
    """The Approximate Bayesian Computation features data model."""

    t: list[int]


input_spec = JSON(pydantic_model=AbcFeatures)


@svc.api(input=input_spec, output=PandasDataFrame())
def predict(inputs: AbcFeatures) -> dict:
    """Get Approximate Bayesian Computation sampling data.

    Args:
        inputs (AbcFeatures):
            The Approximate Bayesian Computation request data.

    Returns:
        dict:
            The Approximate Bayesian Computation sampling data.
    """
    input_list = []
    for t in inputs.t:
        input_list.append({"t": t})

    if len(input_list) == 1:
        index = [0]
    else:
        index = None
    input_df = pd.DataFrame(input_list, index=index)
    result = runner.predict.run(input_df)
    return result
