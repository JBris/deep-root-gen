"""Contains Pydantic Hydra configuration data models.

Several Pydantic composable configuration data models have been
defined within this module for integration with Hydra
Core and OmegaConf.

"""

from pydantic import BaseModel


class ExperimentModel(BaseModel):
    """
    The experiment data model.

    Args:
        BaseModel (BaseModel):
            The Pydantic Base model class.
    """

    name_prefix: str
    tracking_uri: str
    insecure_tls: str
    enabled: bool


class ObjectStorageModel(BaseModel):
    """
    The object storage data model.

    Args:
        BaseModel (BaseModel):
            The Pydantic Base model class.
    """

    s3_ignore_tls: str
    s3_endpoint_url: str
    aws_access_key_id: str
    aws_secret_access_key: str
    enabled: bool


class OrchestrationModel(BaseModel):
    """
    The orchestration data model.

    Args:
        BaseModel (BaseModel):
            The Pydantic Base model class.
    """

    api_url: str
    api_enable_http2: str
    api_tls_insecure_skip_verify: str
    enabled: bool
