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
from itertools import islice
from typing import Callable

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.optim as optim
import torch_geometric.transforms as T
from joblib import dump as calibrator_dump
from lampe.data import JointLoader
from lampe.diagnostics import expected_coverage_mc
from lampe.inference import NPELoss
from lampe.plots import coverage_plot
from lampe.utils import GDStep
from matplotlib import pyplot as plt
from prefect import flow, task
from prefect.artifacts import create_table_artifact
from prefect.task_runners import SequentialTaskRunner
from sbi import analysis as analysis
from sbi import utils as utils
from sbi.inference import (
    SNPE,
    DirectPosterior,
    NeuralInference,
    prepare_for_sbi,
    simulate_for_sbi,
)

from deeprootgen.calibration import (
    GraphFlowFeatureExtractor,
    PriorCollection,
    SnpeModel,
    calculate_summary_statistics,
    get_calibration_summary_stats,
    log_model,
    run_calibration_simulation,
)
from deeprootgen.data_model import RootCalibrationModel, SummaryStatisticsModel
from deeprootgen.io import save_graph_to_db
from deeprootgen.model import RootSystemGraph, process_graph
from deeprootgen.pipeline import (
    begin_experiment,
    get_datetime_now,
    get_outdir,
    log_config,
    log_experiment_details,
    log_simulation,
)

######################################
# Settings
######################################

multiprocessing.set_start_method("spawn", force=True)

######################################
# Constants
######################################

TASK = "snpe"

######################################
# Main
######################################


@task
def prepare_task(input_parameters: RootCalibrationModel) -> tuple:
    """Prepare the Bayesian SNPE procedure.

    Args:
        input_parameters (RootCalibrationModel):
            The root calibration data model.

    Raises:
        ValueError:
            Error thrown when the summary statistic list is empty.

    Returns:
        tuple:
            The Bayesian model specification.
    """
    names = []
    priors = []
    limits = []
    parameter_intervals = input_parameters.parameter_intervals.dict()

    for name, v in parameter_intervals.items():
        names.append(name)
        lower_bound = v["lower_bound"]
        upper_bound = v["upper_bound"]
        limits.append((lower_bound, upper_bound))

        data_type = v["data_type"]
        if data_type == "discrete":
            lower_bound, upper_bound = int(lower_bound), int(upper_bound)
            replicates = np.floor(upper_bound - lower_bound).astype("int")
            probabilities = torch.tensor([1 / replicates])
            probabilities = probabilities.repeat(replicates)
            base_distribution = dist.Categorical(probabilities)
            transforms = [
                dist.AffineTransform(
                    loc=torch.Tensor([lower_bound]), scale=torch.Tensor([1])
                )
            ]
            prior = dist.TransformedDistribution(base_distribution, transforms)
        else:
            prior = dist.Uniform(
                torch.Tensor([lower_bound]), torch.Tensor([upper_bound])
            )

        priors.append(prior)

    _, statistics_list = get_calibration_summary_stats(input_parameters)
    return names, priors, limits, statistics_list


@task
def perform_summary_stat_task(
    input_parameters: RootCalibrationModel,
    priors: list[dist.Distribution],
    names: list[str],
    statistics_list: list[SummaryStatisticsModel],
) -> tuple:
    """Perform the Bayesian SNPE procedure on summary statistics.

    Args:
        parameter_intervals (RootCalibrationIntervals):
            The simulation parameter intervals.
        priors (list[dist.Distribution]):
            The model prior specification.
        names (list[str]):
            The parameter names.
        statistics_list (list[SummaryStatisticsModel]):
            The list of summary statistics.

    Returns:
        tuple:
            The trained model and samples.
    """

    def simulator_func(theta: np.ndarray) -> np.ndarray:
        theta = theta.detach().cpu().numpy()
        parameter_specs = {}
        for i, name in enumerate(names):
            parameter_specs[name] = theta[i]
        simulated, _ = calculate_summary_statistics(
            parameter_specs, input_parameters, statistics_list
        )
        return simulated

    embedding_net = nn.Identity()
    calibration_parameters = input_parameters.calibration_parameters
    simulator, prior = prepare_for_sbi(simulator_func, priors)
    neural_posterior = utils.posterior_nn(
        model="nsf",
        embedding_net=embedding_net,
        hidden_features=calibration_parameters["nn_num_hidden_features"],
        num_transforms=calibration_parameters["nn_num_transforms"],
    )

    inference = SNPE(prior=prior, density_estimator=neural_posterior)
    theta, x = simulate_for_sbi(
        simulator,
        proposal=prior,
        num_simulations=calibration_parameters["n_simulations"],
    )

    inference._device = "cpu"
    inference = inference.append_simulations(theta, x, data_device="cpu")
    density_estimator = inference.train()
    posterior = inference.build_posterior(density_estimator)
    calibration_parameters = input_parameters.calibration_parameters
    n_draws = calibration_parameters["pp_samples"]
    observed_values = []
    for statistic in statistics_list:
        observed_values.append(statistic.statistic_value)
    posterior.set_default_x(observed_values)
    posterior_samples = posterior.sample((n_draws,), x=observed_values)
    observed_values = [statistic.dict() for statistic in statistics_list]

    return inference, simulator, prior, posterior, posterior_samples, observed_values


@task
def perform_data_task(
    input_parameters: RootCalibrationModel,
    priors: list[dist.Distribution],
    names: list[str],
) -> tuple:
    """Perform the Bayesian SNPE procedure on graph data.

    Args:
        parameter_intervals (RootCalibrationIntervals):
            The simulation parameter intervals.
        priors (list[dist.Distribution]):
            The model prior specification.
        names (list[str]):
            The parameter names.

    Returns:
        tuple:
            The trained model and samples.
    """

    priors = PriorCollection(priors)

    G = RootSystemGraph()
    organ_keys = ["organ_coordinates", "organ_hierarchy", "organ_size"]
    organ_features = []
    for k in organ_keys:
        organ_features.extend(G.organ_columns[k])

    transform = T.Compose([T.NormalizeFeatures(organ_keys)])

    def simulator_func(theta: np.ndarray) -> np.ndarray:
        theta = theta.detach().cpu().numpy().T
        parameter_specs = {}
        for i, name in enumerate(names):
            parameter_specs[name] = theta[i]
        simulation, _ = run_calibration_simulation(parameter_specs, input_parameters)
        G = simulation.G.as_torch(drop=True)
        simulated = process_graph(G, organ_keys, transform)
        return simulated

    loader = JointLoader(priors, simulator_func, batch_size=1, vectorized=True)

    calibration_parameters = input_parameters.calibration_parameters
    lr = calibration_parameters["lr"]

    estimator = GraphFlowFeatureExtractor(
        len(names),
        organ_keys,
        calibration_parameters["nn_num_hidden_features"],
        calibration_parameters["nn_num_transforms"],
        organ_features,
    )
    loss = NPELoss(estimator)
    optimizer = optim.Adam(estimator.parameters(), lr=lr)
    step = GDStep(optimizer, clip=0.0)
    estimator.train()

    n_epochs = calibration_parameters["n_epochs"]
    n_simulations = calibration_parameters["n_simulations"]
    for _ in range(n_epochs):
        for theta, x in islice(loader, n_simulations):
            neg_log_p = loss(theta, x)
            step(neg_log_p)

    observed_data_content = input_parameters.observed_data_content
    raw_edge_content = input_parameters.raw_edge_content
    node_df, edge_df = G.from_content_string(observed_data_content, raw_edge_content)
    G = G.as_torch(node_df, edge_df, drop=True)
    x_star = process_graph(G, organ_keys, transform)

    n_draws = calibration_parameters["pp_samples"]
    estimator.eval()
    with torch.no_grad():
        samples = estimator.flow(x_star).sample((n_draws,))

    return estimator, samples, node_df, edge_df, loader


@task
def log_data_task(
    input_parameters: RootCalibrationModel,
    estimator: GraphFlowFeatureExtractor,
    posterior_samples: torch.Tensor,
    node_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    loader: JointLoader,
    statistics_list: list[SummaryStatisticsModel],
    simulation_uuid: str,
) -> tuple:
    """Log the Bayesian SNPE model for output data.

    Args:
        input_parameters (RootCalibrationModel):
            The root calibration data model.
        estimator (GraphFlowFeatureExtractor):
            The normalising flow model.
        posterior_samples (torch.Tensor):
            Samples from the posterior.
        node_df (pd.DataFrame):
            The node dataframe.
        edge_df (pd.DataFrame):
            The edge dataframe.
        loader (JointLoader):
            The data loader.
        statistics_list (list[SummaryStatisticsModel]):
            The list of summary statistics.
        simulation_uuid (str):
            The simulation UUID.

    Returns:
        tuple:
            The simulation and its parameters.
    """
    time_now = get_datetime_now()
    outdir = get_outdir()

    parameter_intervals = input_parameters.parameter_intervals.dict()
    names = []
    lower_bounds = []
    upper_bounds = []
    for name, v in parameter_intervals.items():
        names.append(name)
        lower_bound = v["lower_bound"]
        lower_bounds.append(lower_bound)
        upper_bound = v["upper_bound"]
        upper_bounds.append(upper_bound)

    for plot_func in [analysis.pairplot]:
        outfile = osp.join(outdir, f"{time_now}-{plot_func.__name__}.png")
        plt.rcParams.update({"font.size": 8})
        fig, _ = plot_func(posterior_samples, figsize=(24, 24), labels=names)
        fig.savefig(outfile)
        mlflow.log_artifact(outfile)

    calibration_parameters = input_parameters.calibration_parameters
    n_simulations = calibration_parameters["n_simulations"]
    levels, coverages = expected_coverage_mc(
        posterior=estimator.flow,
        pairs=((theta, x) for theta, x in islice(loader, n_simulations)),
    )

    fig = coverage_plot(levels, coverages, legend="NPE")
    outfile = osp.join(outdir, f"{time_now}-{coverage_plot.__name__}.png")
    fig.savefig(outfile)
    mlflow.log_artifact(outfile)

    parameter_specs = {}
    posterior_means = (
        torch.mean(posterior_samples, dim=0).flatten().detach().cpu().numpy()
    )

    i = 0
    parameter_intervals = input_parameters.parameter_intervals.dict()
    for name, v in parameter_intervals.items():
        posterior_mean = abs(posterior_means[i])
        if v["data_type"] == "discrete":
            posterior_mean = int(posterior_mean)
            if posterior_mean == 0:
                posterior_mean += 1
        parameter_specs[name] = posterior_mean
        i += 1

    simulation, simulation_parameters = run_calibration_simulation(
        parameter_specs, input_parameters
    )

    statistics_list = [statistic.dict() for statistic in statistics_list]
    parameter_intervals["inference_type"] = "data"
    mlflow.set_tag("inference_type", parameter_intervals["inference_type"])
    artifacts = {}
    for obj, name in [
        (estimator, "inference"),
        (posterior_samples, "posterior"),
        (parameter_intervals, "parameter_intervals"),
        (statistics_list, "statistics_list"),
    ]:
        outfile = osp.join(outdir, f"{time_now}-{TASK}_{name}.pkl")
        artifacts[name] = outfile
        calibrator_dump(obj, outfile)

    signature_x = {
        "node_df": node_df.head(5).to_dict("records"),
        "edge_df": edge_df.head(5).to_dict("records"),
    }
    signature_y = pd.DataFrame(posterior_samples, columns=names)
    calibration_model = SnpeModel()

    log_model(
        TASK,
        input_parameters,
        calibration_model,
        artifacts,
        simulation_uuid,
        signature_x,
        signature_y,
    )

    return simulation, simulation_parameters


@task
def log_summary_stat_task(
    inference: NeuralInference,
    simulator: Callable,
    prior: dist.Distribution,
    posterior: DirectPosterior,
    posterior_samples: torch.Tensor,
    input_parameters: RootCalibrationModel,
    observed_values: list,
    statistics_list: list[SummaryStatisticsModel],
    names: list[str],
    limits: list[tuple],
    simulation_uuid: str,
) -> tuple:
    """Log the Bayesian SNPE model for summary statistics.

    Args:
        inference (NeuralInference):
            The neural inference object.
        simulator (Callable):
            The simulation function.
        prior (dist.Distribution):
            The prior specification.
        posterior (DirectPosterior):
            The trained posterior.
        posterior_samples (torch.Tensor):
            Samples from the posterior.
        input_parameters (RootCalibrationModel):
            The root calibration data model.
        observed_values (list):
            The list of observed_values.
        statistics_list (list[SummaryStatisticsModel]):
            The list of summary statistics.
        names (list[str]):
            The parameter names.
        limits (list[tuple]):
            The parameter lower and upper bounds.
        simulation_uuid (str):
            The simulation uuid.

    Returns:
        tuple:
            The simulation and its parameters.
    """
    time_now = get_datetime_now()
    outdir = get_outdir()

    calibration_parameters = input_parameters.calibration_parameters
    n_draws = calibration_parameters["pp_samples"]

    for plot_func in [analysis.pairplot]:
        outfile = osp.join(outdir, f"{time_now}-{plot_func.__name__}.png")
        plt.rcParams.update({"font.size": 8})
        fig, _ = plot_func(posterior_samples, figsize=(24, 24), labels=names)
        fig.savefig(outfile)
        mlflow.log_artifact(outfile)

    for plot_func in [analysis.conditional_pairplot]:
        outfile = osp.join(outdir, f"{time_now}-{plot_func.__name__}.png")
        plt.rcParams.update({"font.size": 8})
        fig, _ = plot_func(
            density=posterior,
            condition=posterior.sample((1,)),
            figsize=(24, 24),
            labels=names,
            limits=limits,
        )
        fig.savefig(outfile)
        mlflow.log_artifact(outfile)

    sbc_draws = calibration_parameters["sbc_draws"]
    thetas = prior.sample((sbc_draws,))
    xs = simulator(thetas)

    ranks, dap_samples = analysis.run_sbc(
        thetas, xs, posterior, num_posterior_samples=n_draws
    )

    check_stats = analysis.check_sbc(
        ranks, thetas, dap_samples, num_posterior_samples=n_draws
    )

    check_stats_list = []
    for metric in check_stats:
        metric_dict = {"metric": metric}
        check_stats_list.append(metric_dict)
        scores = check_stats[metric].detach().cpu().numpy()
        for i, score in enumerate(scores):
            col_name = names[i]
            metric_dict[col_name] = score
            mlflow.log_metric(f"{metric}_{col_name}", score)

    check_stats_df = pd.DataFrame(check_stats_list)
    outfile = osp.join(outdir, f"{time_now}-{TASK}_diagnostics.csv")
    check_stats_df.to_csv(outfile, index=False)
    mlflow.log_artifact(outfile)

    create_table_artifact(
        key="sbc-metrics",
        table=check_stats_df.to_dict(orient="records"),
        description="# Simulation-based calibration metrics.",
    )

    num_bins = None
    if sbc_draws <= 20:  # type: ignore
        num_bins = sbc_draws

    for plot_type in ["hist", "cdf"]:
        outfile = osp.join(
            outdir, f"{time_now}-{analysis.sbc_rank_plot.__name__}_{plot_type}.png"
        )
        plt.rcParams.update({"font.size": 8})

        fig, _ = analysis.sbc_rank_plot(
            ranks=ranks,
            num_bins=num_bins,
            num_posterior_samples=n_draws,
            plot_type=plot_type,
            parameter_labels=names,
        )
        fig.savefig(outfile)
        mlflow.log_artifact(outfile)

    parameter_specs = {}
    posterior_means = (
        torch.mean(posterior_samples, dim=0).flatten().detach().cpu().numpy()
    )

    i = 0
    parameter_intervals = input_parameters.parameter_intervals.dict()
    for name, v in parameter_intervals.items():
        posterior_mean = posterior_means[i]
        if v["data_type"] == "discrete":
            posterior_mean = int(posterior_mean)
        parameter_specs[name] = posterior_mean
        i += 1

    simulation, simulation_parameters = run_calibration_simulation(
        parameter_specs, input_parameters
    )

    statistics_list = [statistic.dict() for statistic in statistics_list]
    parameter_intervals["inference_type"] = "summary_statistics"
    mlflow.set_tag("inference_type", parameter_intervals["inference_type"])
    artifacts = {}
    for obj, name in [
        (inference, "inference"),
        (posterior, "posterior"),
        (parameter_intervals, "parameter_intervals"),
        (statistics_list, "statistics_list"),
    ]:
        outfile = osp.join(outdir, f"{time_now}-{TASK}_{name}.pkl")
        artifacts[name] = outfile
        calibrator_dump(obj, outfile)

    signature_x = pd.DataFrame(observed_values)
    signature_y = pd.DataFrame(posterior_samples, columns=names)
    calibration_model = SnpeModel()

    log_model(
        TASK,
        input_parameters,
        calibration_model,
        artifacts,
        simulation_uuid,
        signature_x,
        signature_y,
    )

    return simulation, simulation_parameters


@task
def run_snpe(input_parameters: RootCalibrationModel, simulation_uuid: str) -> None:
    """Running Sequential Neural Posterior Estimation.

    Args:
        input_parameters (RootCalibrationModel):
            The root calibration data model.
        simulation_uuid (str):
            The simulation uuid.
    """
    begin_experiment(TASK, simulation_uuid, input_parameters.simulation_tag)
    log_experiment_details(simulation_uuid)

    names, priors, limits, statistics_list = prepare_task(input_parameters)

    use_summary_statistics: bool = (
        input_parameters.statistics_comparison.use_summary_statistics
    )
    if use_summary_statistics:
        inference, simulator, prior, posterior, posterior_samples, observed_values = (
            perform_summary_stat_task(input_parameters, priors, names, statistics_list)
        )
        simulation, simulation_parameters = log_summary_stat_task(
            inference,
            simulator,
            prior,
            posterior,
            posterior_samples,
            input_parameters,
            observed_values,
            statistics_list,
            names,
            limits,
            simulation_uuid,
        )
    else:
        estimator, samples, node_df, edge_df, loader = perform_data_task(
            input_parameters, priors, names
        )

        simulation, simulation_parameters = log_data_task(
            input_parameters,
            estimator,
            samples,
            node_df,
            edge_df,
            loader,
            statistics_list,
            simulation_uuid,
        )

    config = input_parameters.dict()
    log_config(config, TASK)
    log_simulation(simulation_parameters, simulation, TASK)
    save_graph_to_db(simulation, TASK, simulation_uuid)
    mlflow.end_run()


@flow(
    name="snpe",
    description="Perform Bayesian parameter estimation for the root model using Sequential Neural Posterior Estimation.",
    task_runner=SequentialTaskRunner(),
)
def run_snpe_flow(input_parameters: RootCalibrationModel, simulation_uuid: str) -> None:
    """Flow for running Sequential Neural Posterior Estimation.

    Args:
        input_parameters (RootCalibrationModel):
            The root calibration data model.
        simulation_uuid (str):
            The simulation uuid.
    """
    run_snpe.submit(input_parameters, simulation_uuid)
