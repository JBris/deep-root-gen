"""Contains Pydantic data models for the simulation

Several Pydantic data models are defined for different
root system architecture simulation procedures.

"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class Config:
    arbitrary_types_allowed = True


class RootType(Enum):
    STRUCTURAL = "structural"
    FINE = "fine"
    PRIMARY = "primary"
    SECONDARY = "secondary"
    OUTER = "outer"
    INNER = "inner"


class RootTypeModel(BaseModel):
    """The root type data model for classifying roots.

    Args:
        BaseModel (BaseModel):
            The Pydantic Base model class.
    """

    root_type: str
    order_type: str
    position_type: str


class RootNodeModel(BaseModel):
    """The node data model for the hierarchical graph representation of the root system.

    Args:
        BaseModel (BaseModel):
            The Pydantic Base model class.
    """

    node_id: Optional[int] = 0
    plant_id: Optional[int] = 1
    parent_id: Optional[int] = -1
    organ_id: Optional[int] = 0
    x: Optional[float] = 0.0
    y: Optional[float] = 0.0
    z: Optional[float] = 0.0
    order: Optional[int] = 0
    segment_rank: Optional[int] = 0
    diameter: Optional[float] = 0.0
    length: Optional[float] = 0.0
    root_type: Optional[str] = "base"
    order_type: Optional[str] = "base"
    position_type: Optional[str] = "base"
    simulation_tag: Optional[str] = "default"
    invalid_root: Optional[bool] = False


class RootEdgeModel(BaseModel):
    """The edge data model for the hierarchical graph representation of the root system.

    Args:
        BaseModel (BaseModel):
            The Pydantic Base model class.
    """

    parent_id: int
    child_id: int


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
    base_diameter: float
    diameter_reduction: float
    apex_diameter: float
    min_sec_root_num: int
    max_sec_root_num: int
    growth_sec_root: float
    min_sec_root_length: float
    max_sec_root_length: float
    segments_per_root: int
    length_reduction: float
    root_vary: float
    origin_min: Optional[float] = 1e-3
    origin_max: Optional[float] = 1e-2
    enable_soil: bool
    soil_layer_height: float
    soil_layer_width: float
    soil_n_layers: int
    soil_n_cols: int
    max_val_attempts: Optional[int] = 50
    simulation_tag: Optional[str] = "default"
    no_root_zone: Optional[float] = 1e-4
    floor_threshold: Optional[float] = 0.4
    ceiling_threshold: Optional[float] = 0.9
