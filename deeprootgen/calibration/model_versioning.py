"""Contains MLflow compatible models for versioning and deployment.

This module defines MLflow compatible models for versioning and deployment as microservices.
"""

import bentoml
import mlflow
import numpy as np
import pandas as pd
from mlflow.client import MlflowClient
from mlflow.models import infer_signature
from mlflow.pyfunc.context import Context

from ..data_model import RootCalibrationModel


def log_model(
    task: str,
    input_parameters: RootCalibrationModel,
    calibration_model: mlflow.pyfunc.PythonModel,
    artifacts: dict,
    simulation_uuid: str,
    signature_x: pd.DataFrame | np.ndarray | list | None = None,
    signature_y: pd.DataFrame | np.ndarray | list | None = None,
) -> None:
    """Log the calibrator model to the registry.

    Args:
        task (str):
            The name of the current task for the experiment.
        input_parameters (RootCalibrationModel):
            The root calibration data model.
        calibration_model (mlflow.pyfunc.PythonModel):
            The calibrator to log.
        artifacts (dict):
            Experiment artifacts to log.
        simulation_uuid (str):
            The simulation uuid.
        signature_x (pd.DataFrame | np.ndarray | list | None, optional):
            The signature for data inputs. Defaults to None.
        signature_y (pd.DataFrame | np.ndarray | list | None, optional):
            The signature for data outputs. Defaults to None.. Defaults to None.
    """
    if signature_x is None and signature_y is None:
        signature = None
    else:
        signature = infer_signature(signature_x, signature_y)

    logged_model = mlflow.pyfunc.log_model(
        python_model=calibration_model,
        artifact_path=task,
        artifacts=artifacts,
        signature=signature,
    )
    model_uri = logged_model.model_uri
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=task,
        tags=dict(
            task=task,
            simulation_uuid=simulation_uuid,
            simulation_tag=input_parameters.simulation_tag,
        ),
    )

    client = MlflowClient(mlflow.get_tracking_uri())
    client.update_model_version(
        name=task,
        version=model_version.version,
        description=f"A root model calibrator for performing the following task: {task}",
    )

    bentoml.mlflow.import_model(
        task,
        model_uri,
        labels=mlflow.active_run().data.tags,
        metadata={
            "metrics": mlflow.active_run().data.metrics,
            "params": mlflow.active_run().data.params,
        },
    )


class OptimisationModel(mlflow.pyfunc.PythonModel):
    """An optimisation calibration model."""

    def __init__(self) -> None:
        """The OptimisationModel constructor."""
        self.task = "optimisation"
        self.calibrator = None

    def load_context(self, context: Context) -> None:
        """Load the model context.

        Args:
            context (Context):
                The model context.
        """
        import joblib

        calibrator_data = context.artifacts["calibrator"]
        self.calibrator = joblib.load(calibrator_data)

    def predict(
        self, context: Context, model_input: pd.DataFrame, params: dict | None = None
    ) -> pd.DataFrame:
        """Make a model prediction.

        Args:
            context (Context):
                The model context.
            model_input (pd.DataFrame):
                The model input data.
            params (dict, optional):
                Optional model parameters. Defaults to None.

        Raises:
            ValueError:
                Error raised when the calibrator has not been loaded.

        Returns:
            pd.DataFrame:
                The model prediction.
        """
        if self.calibrator is None:
            raise ValueError(f"The {self.task} calibrator has not been loaded.")

        n_trials = model_input["n_trials"].item()
        trials_df: pd.DataFrame = self.calibrator.trials_dataframe().sort_values(
            "value", ascending=True
        )
        return trials_df.head(n_trials)


class SensitivityAnalysisModel(mlflow.pyfunc.PythonModel):
    """A sensitivity analysis calibration model."""

    def __init__(self) -> None:
        """The SensitivityAnalysisModel constructor."""
        self.task = "sensitivity_analysis"
        self.calibrator = None

    def load_context(self, context: Context) -> None:
        """Load the model context.

        Args:
            context (Context):
                The model context.
        """
        import joblib

        calibrator_data = context.artifacts["calibrator"]
        self.calibrator = joblib.load(calibrator_data)

    def predict(
        self, context: Context, model_input: pd.DataFrame, params: dict | None = None
    ) -> pd.DataFrame:
        """Make a model prediction.

        Args:
            context (Context):
                The model context.
            model_input (pd.DataFrame):
                The model input data.
            params (dict, optional):
                Optional model parameters. Defaults to None.

        Raises:
            ValueError:
                Error raised when the calibrator has not been loaded.

        Returns:
            pd.DataFrame:
                The model prediction.
        """
        if self.calibrator is None:
            raise ValueError(f"The {self.task} calibrator has not been loaded.")

        names = model_input["name"].values  # noqa: F841
        si_df = self.calibrator.total_si_df
        return si_df.query("name in @names")


class AbcModel(mlflow.pyfunc.PythonModel):
    """An Approximate Bayesian Computation calibration model."""

    def __init__(self) -> None:
        """The AbcModel constructor."""
        self.task = "abc"
        self.calibrator = None

    def load_context(self, context: Context) -> None:
        """Load the model context.

        Args:
            context (Context):
                The model context.
        """
        import joblib

        calibrator_data = context.artifacts["calibrator"]
        self.calibrator = joblib.load(calibrator_data)

    def predict(
        self, context: Context, model_input: pd.DataFrame, params: dict | None = None
    ) -> pd.DataFrame:
        """Make a model prediction.

        Args:
            context (Context):
                The model context.
            model_input (pd.DataFrame):
                The model input data.
            params (dict, optional):
                Optional model parameters. Defaults to None.

        Raises:
            ValueError:
                Error raised when the calibrator has not been loaded.

        Returns:
            pd.DataFrame:
                The model prediction.
        """
        if self.calibrator is None:
            raise ValueError(f"The {self.task} calibrator has not been loaded.")

        t: list[int] = model_input["t"].values
        sampling_df = self.calibrator

        if len(t) == 0 or t[0] == -1:
            return sampling_df
        else:
            return sampling_df.query("t in @t")
