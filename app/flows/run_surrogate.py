#!/usr/bin/env python

######################################
# Imports
######################################

# isort: off

# This is for compatibility with Prefect.

import multiprocessing

# isort: on


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


class VariationalSingleTaskModel(gpytorch.models.ApproximateGP):
    def __init__(
        self,
        inducing_points: torch.Tensor,
        grid_bounds: tuple = (0.0, 1.0),
    ):

        variational_distribution = gpytorch.variational.NaturalVariationalDistribution(
            inducing_points.size(-2)
        )

        variational_strategy = gpytorch.variational.CiqVariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )

        super().__init__(variational_strategy)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.MaternKernel(nu=2.5)

        self.grid_bounds = grid_bounds
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(
            grid_bounds[0], grid_bounds[1]
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        # x = self.scale_to_bounds(x)
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

    dfs = []
    for i in range(3):
        split_df = sample_df.query(f"training_data_split == {i}").drop(
            columns=["training_data_split", "sim_ids"]
        )
        dfs.append(split_df)
    train_df, val_df, test_df = dfs

    splits = []
    for df in [train_df, val_df, test_df]:
        X = df.drop(columns="discrepancy").values
        y = train_df["discrepancy"].values  # .reshape(-1, 1)
        splits.append((X, y))

    train, val, test = splits
    X_train, y_train = train
    X_scaler = MinMaxScaler()
    X_scaler.fit(X_train)
    y_scaler = StandardScaler()
    y_scaler.fit(y_train)

    def prepare_data(split: tuple) -> tuple:
        X, y = split
        X = torch.Tensor(X_scaler.transform(X)).double()
        y = torch.Tensor(y).double()
        return X, y

    X_train, y_train = prepare_data(train)
    X_val, y_val = prepare_data(val)
    X_test, y_test = prepare_data(test)
    print(y_train.shape)
    early_stopper = EarlyStopper(patience=5, min_delta=0)

    calibration_parameters = input_parameters.calibration_parameters
    n_epochs = calibration_parameters["n_epochs"]
    lr = calibration_parameters["lr"]
    num_inducing_points = calibration_parameters["num_inducing_points"]

    inducing_points = X_train[torch.randperm(X_train.size(0))[:num_inducing_points]]
    print(inducing_points.shape)
    model = VariationalSingleTaskModel(inducing_points).train().double()

    mll = gpytorch.mlls.VariationalELBO(
        model.likelihood, model, num_data=y_train.size(0)
    )

    # variational_ngd_optimizer = gpytorch.optim.NGD(
    #     model.variational_parameters(),
    #     num_data=y_train.size(0), lr = 0.1
    # )
    hyperparameter_optimizer = torch.optim.Adam(
        [
            {"params": model.hyperparameters()},
        ],
        lr=lr,
    )

    scheduler = StepLR(
        hyperparameter_optimizer, step_size=int(n_epochs * 0.7), gamma=0.1  # type: ignore[operator]
    )

    validation_check = np.ceil(n_epochs * 0.05).astype("int")  # type: ignore[operator]
    for epoch in range(n_epochs):  # type: ignore
        model.train()
        # variational_ngd_optimizer.zero_grad()
        hyperparameter_optimizer.zero_grad()
        print(X_train.shape)
        out = model(X_train)
        print(out)
        loss = -mll(out, y_train)
        loss.backward()
        # variational_ngd_optimizer.step()
        hyperparameter_optimizer.step()
        scheduler.step()

        if (epoch % validation_check) == 0:
            model.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                out = model(X_val)
                validation_loss = -mll(out, y_val).item()
                if early_stopper.early_stop(validation_loss):
                    break
            print(
                f"Epoch: {epoch}, Training loss: {loss.item()} Validation loss: {validation_loss}",
            )

    model.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = model.likelihood(model(X_test))
        mean = predictions.mean.cpu().numpy()
        lower, upper = predictions.confidence_region()
        lower, upper = lower.cpu().numpy(), upper.cpu().numpy()

    mslls = (
        gpytorch.metrics.mean_standardized_log_loss(predictions, y_test)
        .cpu()
        .detach()
        .numpy()
        .item()
    )
    mses = gpytorch.metrics.mean_squared_error(predictions, y_test).cpu().numpy().item()
    maes = (
        gpytorch.metrics.mean_absolute_error(predictions, y_test).cpu().numpy().item()
    )
    coverage_errors = (
        gpytorch.metrics.quantile_coverage_error(predictions, y_test)
        .cpu()
        .numpy()
        .item()
    )

    print(
        f"""
    mslls: {mslls}
    mses: {mses}
    maes: {maes}
    coverage_errors: {coverage_errors}
    """
    )

    # get_datetime_now()
    # get_outdir()

    print(mean.shape)
    print(y_test.shape)
    # out_df = pd.DataFrame({
    #     "predicted": mean.flatten(),
    #     "actual": y_test.cpu().numpy().flatten(),
    # })
    # out_df.plot.scatter("actual", "predicted")
    # outfile = osp.join(outdir, f"{time_now}-{TASK}_actual_predicted.png")
    # plt.tight_layout()
    # plt.savefig(outfile)
    # mlflow.log_artifact(outfile)


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
