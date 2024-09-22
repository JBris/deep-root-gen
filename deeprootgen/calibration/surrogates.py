"""Contains methods for implementing surrogate models for the root generator.

This module defines various utlities and methods for working with surrogate models.
"""

import gpytorch
import numpy as np
import pandas as pd
import torch
from gpytorch.models import ApproximateGP
from gpytorch.variational import NaturalVariationalDistribution, VariationalStrategy
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset


class EarlyStopper:
    """Early stopping for training."""

    def __init__(self, patience: int = 1, min_delta: float = 0):
        """EarlyStopper constructor.

        Args:
            patience (int, optional):
                The number of iterations before performing early stopping. Defaults to 1.
            min_delta (float, optional):
                The minimum difference for the validation loss. Defaults to 0.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter: int = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss: float) -> bool:
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class SingleTaskVariationalGPModel(ApproximateGP):
    """A variational Gaussian process for single task regression."""

    def __init__(self, inducing_points: torch.Tensor) -> None:
        """SingleTaskVariationalGPModel constructor.

        Args:
            inducing_points (torch.Tensor):
                The inducing points for training.
        """
        variational_distribution = NaturalVariationalDistribution(
            inducing_points.size(0)
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )

        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """The forward pass.

        Args:
            x (torch.Tensor):
                The input data.

        Returns:
            gpytorch.distributions.MultivariateNormal:
                A multivariate Gaussian distribution.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def prepare_surrogate_data(
    sample_df: pd.DataFrame, output_names: list[str] = None, batch_size: int = 1024  # type: ignore [assignment]
) -> tuple:
    """Prepare data for training the surrogate model.

    Args:
        sample_df (pd.DataFrame):
            The dataframe of sample simulation data.
        output_names (list[str], optional):
            The list of output names. Defaults to None.
        batch_size (int, optional):
            The tensor data batch size. Defaults to 1024.

    Returns:
        tuple:
            The processed data for the surrogate.
    """
    training_df = sample_df.loc[:, sample_df.nunique() != 1]

    if output_names is None:
        output_names = ["discrepancy"]

    splits = []
    for i in range(3):
        split_df = training_df.query(f"training_data_split == {i}").drop(
            columns=["training_data_split", "sim_ids"]
        )
        X = split_df.drop(columns=output_names).values
        y = split_df[output_names].values
        if y.shape[-1] == 1:
            y = y.reshape(-1, 1)

        splits.append((X, y))

    train, val, test = splits
    X_train, y_train = train
    X_scaler = MinMaxScaler()
    X_scaler.fit(X_train)
    y_scaler = MinMaxScaler()
    y_scaler.fit(y_train)

    def prepare_data(split: tuple) -> tuple:
        X, y = split
        X = torch.Tensor(X_scaler.transform(X)).double()
        y = torch.Tensor(y_scaler.transform(y)).double()
        if y.shape[-1] == 1:
            y = y.squeeze()

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return loader

    train_loader = prepare_data(train)
    val_loader = prepare_data(val)
    test_loader = prepare_data(test)

    num_data = y_train.shape[0]
    return (
        (train_loader, val_loader, test_loader),
        (X_scaler, y_scaler),
        training_df,
        num_data,
    )


def training_loop(
    model: ApproximateGP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int,
    mll: gpytorch.mlls.VariationalELBO,
    optimizer: torch.optim.Adam,
    variational_ngd_optimizer: gpytorch.optim.NGD = None,
    scheduler: StepLR = None,
    early_stopper: EarlyStopper = None,  # type: ignore [assignment]
    silent: bool = False,
) -> ApproximateGP:
    validation_check: int = np.ceil(n_epochs * 0.05).astype("int")
    for epoch in range(n_epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            if variational_ngd_optimizer is not None:
                variational_ngd_optimizer.zero_grad()

            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()

            optimizer.step()
            if variational_ngd_optimizer is not None:
                variational_ngd_optimizer.step()

            if scheduler is not None:
                scheduler.step()

        if (epoch % validation_check) == 0:
            model.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                validation_loss = 0
                for X_batch, y_batch in val_loader:
                    out = model(X_batch)
                    validation_loss += -mll(out, y_batch).mean().item()

                if early_stopper is not None:
                    if early_stopper.early_stop(validation_loss):
                        return model

            if not silent:
                print(
                    f"""
                    Epoch: {epoch}
                    Training loss: {loss.item()}
                    Validation loss: {validation_loss}
                    """
                )

    return model
