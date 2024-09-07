"""Contains Pydantic data models for the simulation

Several Pydantic data models are defined for different
root system architecture simulation procedures.

"""

from pydantic import BaseModel


class RootSimulationModel(BaseModel):
    """
    The root system architecture simulation data model.

    Args:
        BaseModel (BaseModel):
            The Pydantic Base model class.
    """

    random_seed: int | None
    max_order: int
    root_ratio: float
    fine_root_threshold: float
    outer_root_num: int
    inner_root_num: int
    min_primary_length: float
    max_primary_length: float
    diam_primary_root: float
    min_sec_root_num: int
    max_sec_root_num: int
    growth_sec_root: float
    min_sec_root_length: float
    max_sec_root_length: float
    segments_per_root: int
    root_length_reduction: float
    root_vary: float
    origin_min: float
    origin_max: float
    enable_soil: bool
    soil_layer_height: float
    soil_layer_width: float
    soil_n_layers: int
    soil_n_cols: int
