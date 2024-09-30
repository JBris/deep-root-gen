"""Contains MLflow compatible models for versioning and deployment.

This module defines MLflow compatible models for versioning and deployment as microservices.
"""

from typing import Any

import bentoml
import gpytorch
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import zuko
from gpytorch.models import ApproximateGP
from lampe.inference import NPE
from mlflow.client import MlflowClient
from mlflow.models import infer_signature
from mlflow.pyfunc.context import Context
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.pool import global_max_pool

from ..data_model import RootCalibrationModel
from .surrogates import SingleTaskVariationalGPModel


def log_model(
    task: str,
    input_parameters: RootCalibrationModel,
    calibration_model: mlflow.pyfunc.PythonModel,
    artifacts: dict,
    simulation_uuid: str,
    signature_x: pd.DataFrame | np.ndarray | list | None = None,
    signature_y: pd.DataFrame | np.ndarray | list | None = None,
    model_config: dict = None,  # type: ignore [assignment]
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
            The signature for data outputs. Defaults to None.
        model_config (dict, optional):
            The model configuration. Defaults to None.
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
        model_config=model_config,
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


class SnpeModel(mlflow.pyfunc.PythonModel):
    """A Sequential neural posterior estimation calibration model."""

    def __init__(self) -> None:
        """The SnpeModel constructor."""
        self.task = "snpe"
        self.inference = None
        self.posterior = None
        self.parameter_intervals = None
        self.statistics_df = None

    def load_context(self, context: Context) -> None:
        """Load the model context.

        Args:
            context (Context):
                The model context.
        """
        import joblib

        def load_data(k: str) -> Any:
            artifact = context.artifacts[k]
            return joblib.load(artifact)

        self.inference = load_data("inference")
        self.posterior = load_data("posterior")
        self.parameter_intervals = load_data("parameter_intervals")
        statistics_list = load_data("statistics_list")
        self.statistics_df = pd.DataFrame(statistics_list)

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
        for prop in [
            self.inference,
            self.posterior,
            self.parameter_intervals,
            self.statistics_df,
        ]:
            if prop is None:
                raise ValueError(f"The {self.task} calibrator has not been loaded.")

        if context.model_config["inference_type"] == "summary_statistics":
            statistic_names = self.statistics_df.statistic_name.unique()
            filtered_inputs = model_input.query("statistic_name in @statistic_names")
            if len(filtered_inputs) == 0:
                return filtered_inputs
            filtered_inputs = filtered_inputs.set_index("statistic_name")
            filtered_inputs = filtered_inputs.loc[statistic_names]
            observed_values = filtered_inputs["statistic_value"].values
            posterior_samples = self.posterior.sample((100,), x=observed_values)

            names = []
            for name in self.parameter_intervals:
                names.append(name)
        else:
            raise NotImplementedError("Inference for outputs unsupported.")

        df = pd.DataFrame(posterior_samples, columns=names)
        return df


class GraphFlowFeatureExtractor(torch.nn.Module):
    """A graph feature extractor for density estimation."""

    def __init__(
        self,
        theta_dim: int,
        organ_keys: list[str],
        hidden_features: tuple[float],
        transforms: tuple[float],
        organ_features: list[str],
    ) -> None:
        """GraphFlowFeatureExtractor constructor.

        Args:
            organ_keys (list[str]):
                The list of organ keys for grouping organ features.
            hidden_features (tuple[float]):
                The number of hidden features for the normalising flow.
            transforms (tuple[float]):
                The number of autoregressive transformations.
            organ_features (list[str]):
                The list of organ feature names.
        """
        super().__init__()
        self.organ_keys = organ_keys
        self.organ_features = organ_features

        num_organ_features = len(organ_features)
        self.num_organ_features = num_organ_features
        self.conv1 = SAGEConv(
            num_organ_features,
            num_organ_features * 4,
            aggr="mean",
            normalize=True,
            bias=True,
        )
        self.conv2 = SAGEConv(
            num_organ_features * 4,
            num_organ_features * 2,
            aggr="mean",
            normalize=True,
            bias=True,
        )

        self.fc = torch.nn.Linear(num_organ_features * 2, num_organ_features)
        self.pool = global_max_pool
        self.activation = F.elu

        self.npe = NPE(
            theta_dim=theta_dim,
            x_dim=num_organ_features,
            build=zuko.flows.NSF,
            hidden_features=[hidden_features] * 4,
            transforms=transforms,
            activation=nn.ELU,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Construct graph embeddings from nodes and edges.

        Args:
            x (torch.Tensor):
                The node features.

        Returns:
            torch.Tensor:
                The graph embeddings.
        """
        x, edge_index = x
        batch_index = torch.Tensor(np.repeat(0, x.shape[0])).type(torch.int64)

        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.conv2(x, edge_index)
        x = self.activation(x)
        x = self.pool(x, batch_index)
        x = self.activation(x)
        x = self.fc(x)
        return x.squeeze()

    def forward(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """The forward pass.

        Args:
            theta (torch.Tensor):
                The batch of parameter vectors.
            x (torch.Tensor):
                The batch tensor data.

        Returns:
            torch.Tensor:
                The normalising flow.
        """
        x = self.encode(x)
        x = self.npe(theta, x)
        return x

    def flow(self, x: tuple[torch.Tensor]) -> dist.Distribution:
        """Evaluate the normalising flow.

        Args:
            x (torch.Tensor):
                The node and edge data.

        Returns:
            dist.Distribution:
                The normalising flow.
        """
        x = self.encode(x)
        x = self.npe.flow(x)
        return x


class SurrogateModel(mlflow.pyfunc.PythonModel):
    """A surrogate calibration model."""

    def __init__(self) -> None:
        """The SurrogateModel constructor."""
        self.task = "surrogate"
        self.state_dict = None
        self.X_scaler = None
        self.Y_scaler = None
        self.model = None
        self.likelihood = None
        self.column_names = None

    def load_context(self, context: Context) -> None:
        """Load the model context.

        Args:
            context (Context):
                The model context.
        """
        import joblib

        def load_data(k: str) -> Any:
            artifact = context.artifacts[k]
            return joblib.load(artifact)

        state_dict_path = context.artifacts["state_dict"]
        self.state_dict = torch.load(state_dict_path)

        if context.model_config["surrogate_type"] == "cost_emulator":
            inducing_points_path = context.artifacts["inducing_points"]
            inducing_points = torch.load(inducing_points_path).double()
            self.model = SingleTaskVariationalGPModel(inducing_points).double()
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood().double()

        self.model.load_state_dict(self.state_dict)
        self.model.eval()
        self.X_scaler = load_data("X_scaler")
        self.Y_scaler = load_data("Y_scaler")
        self.column_names = load_data("column_names")

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
        for prop in [
            self.state_dict,
            self.X_scaler,
            self.Y_scaler,
            self.model,
            self.likelihood,
            self.column_names,
        ]:
            if prop is None:
                raise ValueError(f"The {self.task} calibrator has not been loaded.")

        filtered_df = model_input[self.column_names]
        X = self.X_scaler.transform(filtered_df.values)
        X = torch.Tensor(X).double()
        predictions = self.likelihood(self.model(X))

        mean = predictions.mean.detach().cpu().numpy()
        lower, upper = predictions.confidence_region()
        lower, upper = lower.detach().cpu().numpy(), upper.detach().cpu().numpy()

        if context.model_config["surrogate_type"] == "cost_emulator":
            mean = self.Y_scaler.inverse_transform(mean.reshape(-1, 1)).flatten()
            lower = self.Y_scaler.inverse_transform(lower.reshape(-1, 1)).flatten()
            upper = self.Y_scaler.inverse_transform(upper.reshape(-1, 1)).flatten()

            df = pd.DataFrame(
                {"discrepancy": mean, "lower_bound": lower, "upper_bound": upper}
            )

            return df
