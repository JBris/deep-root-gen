#!/usr/bin/env python

######################################
# Imports
######################################

from typing import List

import bentoml
import pandas as pd
from bentoml.io import JSON, PandasDataFrame
from pydantic import BaseModel

from deeprootgen.data_model import SummaryStatisticsModel

######################################
# Constants
######################################

TASK = "snpe"

######################################
# Main
######################################

runner = bentoml.mlflow.get(f"{TASK}:latest").to_runner()

svc = bentoml.Service(TASK, runners=[runner])


class SnpeFeatures(BaseModel):
    """The Sequential Neural Posterior Estimation features data model."""

    summary_statistics: List[SummaryStatisticsModel]


input_spec = JSON(pydantic_model=SnpeFeatures)


@svc.api(input=input_spec, output=PandasDataFrame())
def predict(inputs: SnpeFeatures) -> dict:
    """Get Sequential Neural Posterior Estimation sampling data.

    Args:
        inputs (SnpeFeatures):
            The Sequential Neural Posterior Estimation request data.

    Returns:
        dict:
            The Sequential Neural Posterior Estimation sampling data.
    """
    if len(inputs.summary_statistics) > 0:
        input_list = [statistic.dict() for statistic in inputs.summary_statistics]

    if len(input_list) == 1:
        index = [0]
    else:
        index = None
    input_df = pd.DataFrame(input_list, index=index)
    result = runner.predict.run(input_df)
    return result
