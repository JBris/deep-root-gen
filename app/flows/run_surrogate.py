#!/usr/bin/env python

######################################
# Imports
######################################

# mypy: ignore-errors

# isort: off

# This is for compatibility with Prefect.

import multiprocessing

# isort: on

import os.path as osp

import gpytorch
import mlflow
import numpy as np
import pandas as pd
import torch
from joblib import dump as calibrator_dump
from matplotlib import pyplot as plt
from prefect import flow, task
from prefect.artifacts import create_table_artifact
from prefect.task_runners import SequentialTaskRunner
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from deeprootgen.calibration import (
    AbcModel,
    EarlyStopper,
    calculate_summary_statistic_discrepancy,
    get_calibration_summary_stats,
    lhc_sample,
    log_model,
    run_calibration_simulation,
)
from deeprootgen.data_model import (
    RootCalibrationModel,
    StatisticsComparisonModel,
    SummaryStatisticsModel,
)
from deeprootgen.io import save_graph_to_db
from deeprootgen.pipeline import (
    begin_experiment,
    get_datetime_now,
    get_outdir,
    log_config,
    log_experiment_details,
    log_simulation,
)
from deeprootgen.statistics import DistanceMetricBase

######################################
# Settings
######################################

multiprocessing.set_start_method("spawn", force=True)

######################################
# Constants
######################################

TASK = "surrogate"

######################################
# Main
######################################


@task
def prepare_task(input_parameters: RootCalibrationModel) -> tuple:
    """Prepare the surrogate modelling procedure.

    Args:
        input_parameters (RootCalibrationModel):
            The root calibration data model.

    Returns:
        tuple:
            The Quasi-Monte Carlo specification.
    """

    sample_df = lhc_sample(input_parameters)
    distances, statistics_list = get_calibration_summary_stats(input_parameters)

    return sample_df, distances, statistics_list


class GPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MultitaskVariationalGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, n_features, num_latents, num_tasks):
        inducing_points = torch.rand(num_latents, n_features, n_features)

        variational_distribution = gpytorch.variational.NaturalVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        )

        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True,
            ),
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1,
        )

        super().__init__(variational_strategy)
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=num_tasks
        )
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(0, 1)

        self.mean_module = gpytorch.means.ZeroMean(
            batch_shape=torch.Size([num_latents])
        )
        self.covar_module = gpytorch.kernels.MaternKernel(
            nu=2.5, batch_shape=torch.Size([num_latents]), ard_num_dims=n_features
        )

    def forward(self, x):
        x = self.scale_to_bounds(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


@task
def perform_summary_stat_task(
    input_parameters: RootCalibrationModel,
    sample_df: pd.DataFrame,
    distances: list[DistanceMetricBase],
    statistics_list: list[SummaryStatisticsModel],
) -> None:
    discrepancies = []
    parameter_intervals = input_parameters.parameter_intervals.dict()
    for sample_row in sample_df.to_dict("records"):
        parameter_specs = {}
        for parameter in parameter_intervals:
            parameter_specs[parameter] = sample_row[parameter]

        discrepancy = calculate_summary_statistic_discrepancy(
            parameter_specs, input_parameters, statistics_list, distances
        )
        discrepancies.append(discrepancy)

    sample_df["discrepancy"] = discrepancies

    training_df = sample_df.loc[:, sample_df.nunique() != 1]
    splits = []
    for i in range(3):
        split_df = training_df.query(f"training_data_split == {i}").drop(
            columns=["training_data_split", "sim_ids"]
        )
        X = split_df.drop(columns="discrepancy").values
        y = split_df["discrepancy"].values.reshape(-1, 1)
        splits.append((X, y))

    train, val, test = splits
    X_train, y_train = train
    X_scaler = MinMaxScaler()
    X_scaler.fit(X_train)
    y_scaler = MinMaxScaler()
    y_scaler.fit(y_train)

    calibration_parameters = input_parameters.calibration_parameters
    num_inducing_points = calibration_parameters["num_inducing_points"]

    def prepare_data(split: tuple, shuffle: bool = True) -> tuple:
        X, y = split
        X = torch.Tensor(X_scaler.transform(X)).double()
        y = torch.Tensor(y).double()
        y = torch.Tensor(y_scaler.transform(y)).double()
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=num_inducing_points, shuffle=shuffle)
        return loader

    train_loader = prepare_data(train)
    val_loader = prepare_data(val)
    test_loader = prepare_data(test, shuffle=False)
    early_stopper = EarlyStopper(patience=5, min_delta=0)

    n_epochs = calibration_parameters["n_epochs"]
    lr = calibration_parameters["lr"]

    # model = MultitaskVariationalGPModel(
    #     n_features= X_train.shape[-1],
    #     num_latents = num_inducing_points,
    #     num_tasks = y_train.shape[-1]
    # ).train().double()

    for X_batch, _ in train_loader:
        model = GPModel(inducing_points=X_batch).double()
        break

    mll = gpytorch.mlls.VariationalELBO(
        model.likelihood, model, num_data=y_train.shape[0]
    )

    # variational_ngd_optimizer = gpytorch.optim.NGD(
    #     model.variational_parameters(),
    #     num_data=y_train.shape[0],
    #     lr = 0.1
    # )
    hyperparameter_optimizer = torch.optim.Adam(
        [
            {"params": model.parameters()},
            # {"params": model.hyperparameters()},
        ],
        lr=lr,
    )

    scheduler = StepLR(
        hyperparameter_optimizer,
        step_size=int(n_epochs * 0.7),
        gamma=0.1,  # type: ignore[operator]
    )

    validation_check: int = np.ceil(n_epochs * 0.05).astype("int")  # type: ignore[operator]
    for epoch in range(n_epochs):  # type: ignore
        model.train()
        for X_batch, y_batch in train_loader:
            # variational_ngd_optimizer.zero_grad()
            hyperparameter_optimizer.zero_grad()
            out = model(X_batch)
            loss = -mll(out, y_batch).mean()
            loss.backward()
            # variational_ngd_optimizer.step()
            hyperparameter_optimizer.step()
            scheduler.step()

        if (epoch % validation_check) == 0:
            model.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                validation_loss = 0
                for X_batch, y_batch in val_loader:
                    out = model(X_batch)
                    validation_loss += -mll(out, y_batch).mean().item()
                if early_stopper.early_stop(validation_loss):
                    break
            print(
                f"Epoch: {epoch}, Training loss: {loss.item()} Validation loss: {validation_loss}",
            )

    model.eval()
    X_test = []
    y_test = []
    for X_batch, y_batch in test_loader:
        X_test.extend(X_batch)
        y_test.extend(y_batch)
    X_test = torch.stack(X_test)
    y_test = torch.stack(y_test)

    time_now = get_datetime_now()
    outdir = get_outdir()

    outfile = osp.join(outdir, f"{time_now}-{TASK}_samples.csv")
    sample_df.to_csv(outfile, index=False)
    mlflow.log_artifact(outfile)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = model.likelihood(model(X_test))
        mean = predictions.mean.cpu().numpy()
        lower, upper = predictions.confidence_region()
        lower, upper = lower.cpu().numpy(), upper.cpu().numpy()

    metrics = []
    for metric_func in [
        gpytorch.metrics.mean_standardized_log_loss,
        gpytorch.metrics.mean_squared_error,
        gpytorch.metrics.mean_absolute_error,
        gpytorch.metrics.quantile_coverage_error,
        gpytorch.metrics.negative_log_predictive_density,
    ]:
        metric_name = metric_func.__name__
        metric_score = metric_func(predictions, y_test).cpu().detach().numpy().item()
        mlflow.log_metric(metric_name, metric_score)
        metrics.append({"metric_name": metric_name, "metric_score": metric_score})

    metric_df = pd.DataFrame(metrics)
    outfile = osp.join(outdir, f"{time_now}-{TASK}_metrics.csv")
    metric_df.to_csv(outfile, index=False)
    mlflow.log_artifact(outfile)

    create_table_artifact(
        key="surrogate-metrics",
        table=metric_df.to_dict(orient="records"),
        description="# Surrogate metrics.",
    )

    mean = y_scaler.inverse_transform(mean).flatten()
    actual = y_scaler.inverse_transform(y_test.cpu().numpy()).flatten()
    lower = y_scaler.inverse_transform(lower).flatten()
    upper = y_scaler.inverse_transform(upper).flatten()
    index = np.arange(len(mean))
    out_df = pd.DataFrame(
        {"test_predicted": mean, "test_actual": actual, "index": index}
    )
    out_df.plot.scatter("test_actual", "test_predicted")
    outfile = osp.join(outdir, f"{time_now}-{TASK}_actual_predicted.png")
    plt.tight_layout()
    plt.savefig(outfile)
    mlflow.log_artifact(outfile)


@task
def run_surrogate(input_parameters: RootCalibrationModel, simulation_uuid: str) -> None:
    """Running a surrogate model on the root model.

    Args:
        input_parameters (RootCalibrationModel):
            The root calibration data model.
        simulation_uuid (str):
            The simulation uuid.
    """
    begin_experiment(TASK, simulation_uuid, input_parameters.simulation_tag)
    log_experiment_details(simulation_uuid)

    sample_df, distances, statistics_list = prepare_task(input_parameters)

    use_summary_statistics: bool = (
        input_parameters.statistics_comparison.use_summary_statistics  # type: ignore
    )
    if use_summary_statistics:
        perform_summary_stat_task(
            input_parameters, sample_df, distances, statistics_list
        )

    config = input_parameters.dict()
    log_config(config, TASK)
    mlflow.end_run()


@flow(
    name="surrogate",
    description="Train a surrogate model for the root model.",
    task_runner=SequentialTaskRunner(),
)
def run_surrogate_flow(
    input_parameters: RootCalibrationModel, simulation_uuid: str
) -> None:
    """Flow for training a surrogate model on the root model.

    Args:
        input_parameters (RootCalibrationModel):
            The root calibration data model.
        simulation_uuid (str):
            The simulation uuid.
    """
    run_surrogate.submit(input_parameters, simulation_uuid)
