#!/usr/bin/env python

######################################
# Imports
######################################

import bentoml
import numpy as np
import pandas as pd
from bentoml.io import JSON, PandasDataFrame
from pydantic import BaseModel

######################################
# Constants
######################################

TASK = "optimisation"

######################################
# Main
######################################

runner = bentoml.mlflow.get(f"{TASK}:latest").to_runner()

svc = bentoml.Service(TASK, runners=[runner])


class OptimisationFeatures(BaseModel):
    """The optimisation features data model."""

    n_trials: int


input_spec = JSON(pydantic_model=OptimisationFeatures)


@svc.api(input=input_spec, output=PandasDataFrame())
def predict(inputs: OptimisationFeatures) -> np.ndarray:
    """Get optimisation trial data.

    Args:
        inputs (OptimisationFeatures):
            The optimisation request data.

    Returns:
        np.ndarray:
            The optimisation trial history.
    """
    input_dict = inputs.dict()
    if len(input_dict) == 1:
        index = [0]
    else:
        index = None
    input_df = pd.DataFrame(input_dict, index=index)
    result = runner.predict.run(input_df)
    return result
