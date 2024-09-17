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

TASK = "sensitivity_analysis"

######################################
# Main
######################################

runner = bentoml.mlflow.get(f"{TASK}:latest").to_runner()

svc = bentoml.Service(TASK, runners=[runner])


class SensitivityAnalysisFeatures(BaseModel):
    """The sensitivity analysis features data model."""

    names: list[str]


input_spec = JSON(pydantic_model=SensitivityAnalysisFeatures)


@svc.api(input=input_spec, output=PandasDataFrame())
def predict(inputs: SensitivityAnalysisFeatures) -> dict:
    """Get sensitivity analysis indices data.

    Args:
        inputs (SensitivityAnalysisFeatures):
            The sensitivity analysis request data.

    Returns:
        dict:
            The sensitivity analysis indices.
    """
    input_list = []
    for name in inputs.names:
        input_list.append({"name": name})

    if len(input_list) == 1:
        index = [0]
    else:
        index = None
    input_df = pd.DataFrame(input_list, index=index)
    result = runner.predict.run(input_df)
    return result
