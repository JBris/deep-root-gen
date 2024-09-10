"""Contains utilities for managing simulation data.

This module defines utility functions for managing inputs and outputs for simulation data.

"""

import os

import boto3
from adbnx_adapter import ADBNX_Adapter
from arango import ArangoClient

from ..model import RootSystemSimulation


def s3_upload_file(file_name: str, object_name: str, bucket_name: str = "data") -> bool:
    """Upload a file to an S3 bucket.

    Args:
        file_name (str):
            The file to upload.
        object_name (str):
            The S3 object name.
        bucket_name (str, optional):
            The bucket name. Defaults to "data".

    Returns:
        bool:
            The client response.
    """
    s3_client = boto3.client(
        "s3",
        use_ssl=False,
        endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )

    response = s3_client.upload_file(file_name, bucket_name, object_name)
    return response


def save_graph_to_db(
    simulation: RootSystemSimulation, task: str, simulation_uuid: str
) -> None:
    G = simulation.G.as_networkx()
    collection = f"{task}-{simulation_uuid}"

    db_host = os.environ["ARANGO_HOST_URL"]
    db_name = os.environ["ARANGO_DB"]
    db_username = os.environ["ARANGO_ROOT_USER"]
    db_password = os.environ["ARANGO_ROOT_PASSWORD"]

    sys_db = ArangoClient(hosts=db_host).db(
        "_system", username=db_username, password=db_password
    )
    if not sys_db.has_database(db_name):
        sys_db.create_database(db_name)
    db = ArangoClient(hosts=db_host).db(
        db_name, username=db_username, password=db_password
    )

    edges_collection = f"{collection}_edges"
    for db_collection in [collection, edges_collection]:
        if db.has_collection(db_collection):
            db.delete_collection(db_collection)

    if db.has_graph(collection):
        db.delete_graph(collection)

    graph_definitions = [
        {
            "edge_collection": edges_collection,
            "from_vertex_collections": [collection],
            "to_vertex_collections": [collection],
        }
    ]

    adapter = ADBNX_Adapter(db)
    adapter.networkx_to_arangodb(collection, G, graph_definitions)