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
from random import choices
from typing import Callable

import mlflow
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from joblib import dump as calibrator_dump
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
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.pool import global_max_pool

from deeprootgen.calibration import (
    SnpeModel,
    calculate_summary_statistics,
    get_calibration_summary_stats,
    log_model,
    run_calibration_simulation,
)
from deeprootgen.data_model import RootCalibrationModel, SummaryStatisticsModel
from deeprootgen.io import save_graph_to_db
from deeprootgen.model import RootSystemGraph
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


# @TODO this is a hack for compatibility with the sbi API,
# and should be replaced with a GNN feature extractor surrogate.
# i.e. we train a separate feature extractor to provide graph embeddings,
# then use that feature extractor instead of this embedding net.
class GraphFeatureExtractor(torch.nn.Module):
    """A graph feature extractor for density estimation."""

    def __init__(self, organ_columns: list[str]) -> None:
        """GraphFeatureExtractor constructor.

        Args:
            organ_columns (list[str]):
                The list of organ columns for grouping organ features.
        """
        super().__init__()
        self.organ_columns = organ_columns

        self.transform = T.Compose([T.NormalizeFeatures(organ_columns)])

        G = RootSystemGraph()
        organ_features = []
        for organ_column in organ_columns:
            organ_features.extend(G.organ_columns[organ_column])
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

        self.G_list: list = []

    def process_graph(self, G: nx.Graph) -> tuple:
        """Process a new NetworkX graph.

        Args:
            G (nx.Graph):
                The NetworkX graph.

        Returns:
            tuple:
                The node and edge features.
        """
        for column in self.organ_columns:
            G[column] = torch.Tensor(pd.DataFrame(G[column]).values).double()

        train_data = self.transform(G)
        organ_features = []
        for column in self.organ_columns:
            organ_features.append(train_data[column])

        x = torch.Tensor(np.hstack(organ_features))
        edge_index = train_data.edge_index
        return x, edge_index

    def add_graph(self, G: nx.Graph) -> int:
        """Add a graph to the graph list.

        Args:
            G (nx.Graph):
                The NetworkX graph.

        Returns:
            int:
                The list index.
        """
        x, edge_index = self.process_graph(G)

        self.G_list.append((x, edge_index))
        return len(self.G_list) - 1

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Construct graph embeddings from node and edges.

        Args:
            x (torch.Tensor):
                The node features.
            edge_index (torch.Tensor):
                The edge index.

        Returns:
            torch.Tensor:
                The graph embeddings.
        """
        batch_index = torch.Tensor(np.repeat(0, x.shape[0])).type(torch.int64)

        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.conv2(x, edge_index)
        x = self.activation(x)
        x = self.pool(x, batch_index)
        x = self.activation(x)
        x = self.fc(x)
        x = x.view(-1)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward pass.

        Args:
            x (torch.Tensor):
                The batch tensor.

        Returns:
            torch.Tensor:
                The graph embedding.
        """
        if x.shape[1] > 1:
            return x

        batch_size = x.shape[0]
        indices = np.array(range(batch_size))

        batches = []
        batch = choices(self.G_list, k=batch_size)
        for i in indices:
            x, edge_index = batch[i]
            x = self.encode(x, edge_index)
            batches.append(x)
        x = torch.stack(batches)
        return x


@task
def perform_task(
    input_parameters: RootCalibrationModel,
    priors: list[dist.Distribution],
    names: list[str],
    statistics_list: list[SummaryStatisticsModel],
) -> tuple:
    """Perform the Bayesian SNPE procedure.

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
    use_summary_statistics: bool = (
        input_parameters.statistics_comparison.use_summary_statistics
    )
    if use_summary_statistics:
        embedding_net = nn.Identity()
    else:
        organ_columns = ["organ_coordinates", "organ_hierarchy", "organ_size"]
        embedding_net = GraphFeatureExtractor(organ_columns)

    def simulator_func(theta: np.ndarray) -> np.ndarray:
        theta = theta.detach().cpu().numpy()
        parameter_specs = {}
        for i, name in enumerate(names):
            parameter_specs[name] = theta[i]

        if use_summary_statistics:
            simulated, _ = calculate_summary_statistics(
                parameter_specs, input_parameters, statistics_list
            )
        else:
            simulation, _ = run_calibration_simulation(
                parameter_specs, input_parameters
            )

            G = simulation.G.as_torch(drop=True)
            indx = embedding_net.add_graph(G)
            simulated = np.array([indx]).astype("int")

        return simulated

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

    if use_summary_statistics:
        observed_values = []
        for statistic in statistics_list:
            observed_values.append(statistic.statistic_value)
        posterior.set_default_x(observed_values)
        posterior_samples = posterior.sample((n_draws,), x=observed_values)
        observed_values = [statistic.dict() for statistic in statistics_list]
    else:
        root_g = RootSystemGraph()
        observed_data_content = input_parameters.observed_data_content
        raw_edge_content = input_parameters.raw_edge_content
        node_df, edge_df = root_g.from_content_string(
            observed_data_content, raw_edge_content
        )
        G = root_g.as_torch(node_df, edge_df, drop=True)
        x, edge_index = embedding_net.process_graph(G)

        with torch.no_grad():
            observed_values = embedding_net.encode(x, edge_index)

        posterior.set_default_x(observed_values)
        posterior_samples = posterior.sample((n_draws,), x=observed_values)
        embedding_net.G_list = []
        observed_values = (node_df, edge_df)

    return inference, simulator, prior, posterior, posterior_samples, observed_values


@task
def log_task(
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
    """Log the Bayesian SNPE model.

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
    # use_summary_statistics: bool = (
    #     input_parameters.statistics_comparison.use_summary_statistics
    # )
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
    inference, simulator, prior, posterior, posterior_samples, observed_values = (
        perform_task(input_parameters, priors, names, statistics_list)
    )
    simulation, simulation_parameters = log_task(
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
