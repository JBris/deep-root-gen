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
from torch.optim.lr_scheduler import StepLR

from deeprootgen.calibration import (
    EarlyStopper,
    SingleTaskVariationalGPModel,
    SurrogateModel,
    calculate_summary_statistic_discrepancy,
    get_calibration_summary_stats,
    lhc_sample,
    log_model,
    prepare_surrogate_data,
    run_calibration_simulation,
    training_loop,
)
from deeprootgen.data_model import RootCalibrationModel, SummaryStatisticsModel
from deeprootgen.pipeline import (
    begin_experiment,
    get_datetime_now,
    get_outdir,
    log_config,
    log_experiment_details,
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
) -> tuple:
    """Train a surrogate model using cost emulation.

    Args:
        input_parameters (RootCalibrationModel):
            The root calibration data model.
        sample_df (pd.DataFrame):
            The data produced using Latin Hypercube sampling.
        distances (list[DistanceMetricBase]):
            The list of distance metrics.
        statistics_list (list[SummaryStatisticsModel]):
            The list of summary statistics.

    Returns:
        tuple:
            The trained surrogate.
    """
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

    loaders, scalers, training_df, num_data = prepare_surrogate_data(sample_df)
    train_loader, val_loader, _ = loaders

    calibration_parameters = input_parameters.calibration_parameters
    num_inducing_points = calibration_parameters["num_inducing_points"]
    for inducing_points, _ in train_loader:
        inducing_points = inducing_points[:num_inducing_points, :]
        break
    model = SingleTaskVariationalGPModel(inducing_points).double()
    likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
    model.train()
    likelihood.train()

    variational_ngd_optimizer = gpytorch.optim.NGD(
        model.variational_parameters(), num_data=num_data, lr=0.1
    )
    lr = calibration_parameters["lr"]
    optimizer = torch.optim.Adam(
        [
            {"params": model.hyperparameters()},
            {"params": likelihood.parameters()},
        ],
        lr=lr,
    )

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=num_data)

    n_epochs = calibration_parameters["n_epochs"]
    scheduler = StepLR(
        optimizer,
        step_size=int(n_epochs * 0.7),
        gamma=0.1,  # type: ignore[operator]
    )

    patience = calibration_parameters["early_stopping_patience"]
    early_stopper = EarlyStopper(patience=patience, min_delta=0)
    training_loop(
        model,
        train_loader,
        val_loader,
        n_epochs,
        mll,
        optimizer,
        variational_ngd_optimizer,
        scheduler,
        early_stopper,
    )

    return model, likelihood, sample_df, loaders, scalers, training_df, inducing_points


@task
def log_summary_stat_task(
    input_parameters: RootCalibrationModel,
    model: gpytorch.models.ApproximateGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    sample_df: pd.DataFrame,
    loaders: tuple,
    scalers: tuple,
    training_df: pd.DataFrame,
    inducing_points: torch.Tensor,
    simulation_uuid: str,
) -> None:
    """Log the surrogate model results.

    Args:
        input_parameters (RootCalibrationModel):
            The root calibration data model.
        model (gpytorch.models.ApproximateGP):
            The surrogate model.
        likelihood (gpytorch.likelihoods.GaussianLikelihood):
            The surrogate model likelihood.
        sample_df (pd.DataFrame):
            The sample simulation data.
        loaders (tuple):
            The data loaders.
        scalers (tuple):
            The data scalers.
        training_df (pd.DataFrame):
            The model training data.
        simulation_uuid (str):
            The simulation UUID.
    """
    time_now = get_datetime_now()
    outdir = get_outdir()

    outfile = osp.join(outdir, f"{time_now}-{TASK}_samples.csv")
    sample_df.to_csv(outfile, index=False)
    mlflow.log_artifact(outfile)

    train_loader, val_loader, test_loader = loaders
    X_scaler, y_scaler = scalers

    def log_performance(loader: torch.utils.data.DataLoader, split_name: str) -> None:
        model.eval()
        X_data = []
        y_data = []

        for X_batch, y_batch in loader:
            X_data.extend(X_batch)
            y_data.extend(y_batch)

        X_data = torch.stack(X_data)
        y_data = torch.stack(y_data)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = likelihood(model(X_data))
            mean = predictions.mean.detach().cpu().numpy()
            lower, upper = predictions.confidence_region()
            lower, upper = lower.detach().cpu().numpy(), upper.detach().cpu().numpy()

        metrics = []
        for metric_func in [
            gpytorch.metrics.mean_standardized_log_loss,
            gpytorch.metrics.mean_squared_error,
            gpytorch.metrics.mean_absolute_error,
            gpytorch.metrics.quantile_coverage_error,
            gpytorch.metrics.negative_log_predictive_density,
        ]:
            metric_name = metric_func.__name__
            metric_score = (
                metric_func(predictions, y_data).cpu().detach().numpy().item()
            )
            mlflow.log_metric(f"{split_name}_{metric_name}", metric_score)
            metrics.append({"metric_name": metric_name, "metric_score": metric_score})

        metric_df = pd.DataFrame(metrics)
        outfile = osp.join(outdir, f"{time_now}-{TASK}_{split_name}_metrics.csv")
        metric_df.to_csv(outfile, index=False)
        mlflow.log_artifact(outfile)

        create_table_artifact(
            key=f"surrogate-metrics-{split_name}-data",
            table=metric_df.to_dict(orient="records"),
            description=f"# Surrogate metrics on {split_name} data.",
        )

        y_data = y_data.detach().cpu().numpy().reshape(-1, 1)
        mean = y_scaler.inverse_transform(mean.reshape(-1, 1)).flatten()
        actual = y_scaler.inverse_transform(y_data).flatten()
        lower = y_scaler.inverse_transform(lower.reshape(-1, 1)).flatten()
        upper = y_scaler.inverse_transform(upper.reshape(-1, 1)).flatten()
        index = np.arange(len(mean))

        first_col = sample_df.columns[0]
        X_data = X_scaler.inverse_transform(X_data.detach().cpu().numpy())
        x = X_data[:, 0]
        out_df = pd.DataFrame(
            {
                f"{split_name}_predicted": mean,
                f"{split_name}_actual": actual,
                f"{split_name}_{first_col}": x,
                "index": index,
            }
        )
        out_df.plot.scatter(f"{split_name}_actual", f"{split_name}_predicted")
        outfile = osp.join(
            outdir, f"{time_now}-{TASK}_{split_name}_actual_predicted.png"
        )
        plt.tight_layout()
        plt.savefig(outfile)
        mlflow.log_artifact(outfile)

        out_df.plot.scatter(
            f"{split_name}_{first_col}",
            f"{split_name}_actual",
            title=f"{split_name}_actual",
        )
        outfile = osp.join(outdir, f"{time_now}-{TASK}_{split_name}_actual.png")
        plt.tight_layout()
        plt.savefig(outfile)
        mlflow.log_artifact(outfile)

        fig = out_df.plot.scatter(
            f"{split_name}_{first_col}",
            f"{split_name}_predicted",
            title=f"{split_name}_predicted",
        )
        fig.fill_between(mean, lower, upper, alpha=0.5)
        outfile = osp.join(outdir, f"{time_now}-{TASK}_{split_name}_predicted.png")
        plt.tight_layout()
        plt.savefig(outfile)
        mlflow.log_artifact(outfile)

        outfile = osp.join(
            outdir, f"{time_now}-{TASK}_{split_name}_actual_predicted.csv"
        )
        out_df.to_csv(outfile, index=False)
        mlflow.log_artifact(outfile)

        return mean, lower, upper

    log_performance(val_loader, "validation")
    log_performance(train_loader, "train")
    mean, lower, upper = log_performance(test_loader, "test")

    mlflow.set_tag("surrogate_type", "cost emulation")
    artifacts = {}
    for obj, name in [
        (model.state_dict(), "state_dict"),
        (inducing_points, "inducing_points"),
    ]:
        outfile = osp.join(outdir, f"{time_now}-{TASK}_{name}.pth")
        artifacts[name] = outfile
        torch.save(obj, outfile)

    drop_columns = ["training_data_split", "sim_ids", "discrepancy"]
    column_names = training_df.drop(columns=drop_columns).columns

    for obj, name in [
        (X_scaler, "X_scaler"),
        (y_scaler, "Y_scaler"),
        (column_names, "column_names"),
    ]:
        outfile = osp.join(outdir, f"{time_now}-{TASK}_{name}.pkl")
        artifacts[name] = outfile
        calibrator_dump(obj, outfile)

    signature_x = training_df.drop(columns=drop_columns).head(5).to_dict("records")
    signature_y = pd.DataFrame(
        {"discrepancy": mean, "lower_bound": lower, "upper_bound": upper}
    )

    calibration_model = SurrogateModel()

    log_model(
        TASK,
        input_parameters,
        calibration_model,
        artifacts,
        simulation_uuid,
        signature_x,
        signature_y,
        {"surrogate_type": "cost_emulator", "n_tasks": 1},
    )


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
        model, likelihood, sample_df, loaders, scalers, training_df, inducing_points = (
            perform_summary_stat_task(
                input_parameters, sample_df, distances, statistics_list
            )
        )

        log_summary_stat_task(
            input_parameters,
            model,
            likelihood,
            sample_df,
            loaders,
            scalers,
            training_df,
            inducing_points,
            simulation_uuid,
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
