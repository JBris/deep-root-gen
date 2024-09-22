#!/usr/bin/env python

######################################
# Imports
######################################

from typing import List, Optional

import bentoml
import pandas as pd
from bentoml.io import JSON, PandasDataFrame
from pydantic import BaseModel

######################################
# Constants
######################################

TASK = "surrogate"

######################################
# Main
######################################

runner = bentoml.mlflow.get(f"{TASK}:latest").to_runner()

svc = bentoml.Service(TASK, runners=[runner])


class SimulationFeatures(BaseModel):
    """The simulation features data model."""

    random_seed: Optional[int] = 100
    max_order: Optional[int] = 3
    root_ratio: Optional[float] = 0.5
    fine_root_threshold: Optional[float] = 0.06
    outer_root_num: Optional[int] = 10
    inner_root_num: Optional[int] = 8
    min_primary_length: Optional[float] = 20
    max_primary_length: Optional[float] = 30
    base_diameter: Optional[float] = 0.11
    diameter_reduction: Optional[float] = 0.2
    apex_diameter: Optional[float] = 0.02
    min_sec_root_num: Optional[int] = 1
    max_sec_root_num: Optional[int] = 3
    growth_sec_root: Optional[float] = 0.2
    min_sec_root_length: Optional[float] = 100
    max_sec_root_length: Optional[float] = 220
    segments_per_root: Optional[int] = 50
    length_reduction: Optional[float] = 0.5
    root_vary: Optional[float] = 30
    interbranch_distance: Optional[float] = 0.0078
    mechanical_constraints: Optional[float] = 0.5
    root_tissue_density: Optional[float] = 0.05
    gravitropism: Optional[float] = 7.5
    origin_min: Optional[float] = 1e-3
    origin_max: Optional[float] = 1e-2
    enable_soil: Optional[bool] = False
    soil_layer_height: Optional[float] = 0
    soil_layer_width: Optional[float] = 0
    soil_n_layers: Optional[int] = 0
    soil_n_cols: Optional[int] = 0
    max_val_attempts: Optional[int] = 50
    simulation_tag: Optional[str] = "default"
    no_root_zone: Optional[float] = 1e-4
    floor_threshold: Optional[float] = 0.4
    ceiling_threshold: Optional[float] = 0.9


class SurrogateFeatures(BaseModel):
    """The Surrogate features data model."""

    data: List[SimulationFeatures]


input_spec = JSON(pydantic_model=SurrogateFeatures)


@svc.api(input=input_spec, output=PandasDataFrame())
def predict(inputs: SurrogateFeatures) -> dict:
    """Get the surrogate model predictions.

    Args:
        inputs (SurrogateFeatures):
            The simulation parameter data.

    Returns:
        dict:
            The surrogate model predictions.
    """
    if len(inputs.data) > 0:
        input_list = [simulation.dict() for simulation in inputs.data]

    if len(input_list) == 1:
        index = [0]
    else:
        index = None
    input_df = pd.DataFrame(input_list, index=index)
    result = runner.predict.run(input_df)
    return result
