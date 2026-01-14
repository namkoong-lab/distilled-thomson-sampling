#!/usr/bin/env python3

from typing import Optional

import torch
from gpytorch.utils.cholesky import psd_safe_cholesky
from torch import Tensor
from torch.distributions import Gamma
from torch.nn import Module


class ParametricBayesianLinearRegression(Module):
    """
    Class implementing parametric Bayesian Linear Regression (weight-space view).
    The model assumes:
        beta, sigma^2 ~ NIG(mu, precision, a, b)
        sigma_2 ~ IG(a, b)
        beta | sigma^2 ~ N(mu, sigma^2 * precision^{-1})

    The posterior uses efficient online updates.

    Qian, Hang. Big Data Bayesian Linear Regression and Variable Selection
        by Normal-Inverse-Gamma Summation. Bayesian Anal. 13 (2018), no. 4,
        1011--1035. https://projecteuclid.org/euclid.ba/1510110046

    Note: the internal `dtype` of this model is set to torch.double for numerical
        stability. Changing the dtype to torch.float (e.g. via `.float()` may
        lead to numerical issues.
    """

    def __init__(
        self,
        d: int,
        dtype: torch.dtype,
        device: torch.device,
        a0: Optional[float] = None,
        b0: float = 0.0,
        lambda_prior: float = 1.0,
    ) -> None:
        super().__init__()
        self.original_dtype = dtype
        self.register_buffer("mu", torch.zeros(d, dtype=torch.double, device=device))
        self.register_buffer(
            "precision", torch.eye(d, dtype=torch.double, device=device) * lambda_prior
        )
        # pyre-ignore [16]
        self.register_buffer("precision_root", self.precision.sqrt())
        self.register_buffer(
            "a",
            torch.tensor(
                a0 if a0 is not None else -d / 2, dtype=torch.double, device=device
            ),
        )
        self.register_buffer("b", torch.tensor(b0, dtype=torch.double, device=device))

    def update(self, X: Tensor, Y: Tensor) -> None:
        """
        Perform online posterior updates using new batch of observations.

        Args:
            X: A `n x d`-dim Tensor of inputs
            Y: `n`-dim tensor of targets
        """
        # pyre-ignore [16]
        X = X.to(self.precision.dtype)
        Y = Y.to(self.precision.dtype)
        new_precision = self.precision + X.t() @ X
        x_t_y = X.t() @ Y
        # We want to solve:
        # new_precision @ new_mu = self.precision @ self.mu + X.t() @ y
        # For better numerical stability, we instead solve:
        # L @ z = self.precision @ self.mu + X.t() @ y
        # and L.t() @ new_mu = z
        # torch.cholesky_solve solves both in one pass given L.
        self.precision_root = psd_safe_cholesky(new_precision)  # pyre-ignore [16]
        new_mu = torch.cholesky_solve(
            (self.precision @ self.mu + x_t_y).unsqueeze(-1), self.precision_root
        ).squeeze(-1)
        # pyre-ignore [16]
        self.a += Y.shape[0] / 2
        # pyre-ignore [16]
        self.b += 0.5 * (
            Y.t() @ Y
            + self.mu.t() @ self.precision @ self.mu
            - new_mu.t() @ new_precision @ new_mu
        )
        # pyre-ignore [16]
        self.mu = new_mu
        self.precision = new_precision

    def posterior_samples(self, n_samples: int) -> Tensor:
        """
        Sample from the posterior over the parameters.

        Args:
            n_samples: number of samples

        Returns:
            Tensor: `n_samples x d`-dim Tensor of samples
        """
        # sample sigma^2 ~ IG(a, b)
        # pyre-ignore [16]
        sigma2_samples = 1.0 / Gamma(self.a, self.b).sample(torch.Size([n_samples]))
        # sample beta | sigma^2 ~ N(mu, sigma^2 * cov)
        # sample from multivariate normal using precision parameterization
        # and use a linear solve to avoid inverting the precision matrix
        z = torch.randn(
            (n_samples, self.precision.shape[0]),  # pyre-ignore [16]
            dtype=self.a.dtype,
            device=self.a.device,
        )
        X = torch.triangular_solve(
            z.unsqueeze(-1),
            1.0
            / sigma2_samples.sqrt().unsqueeze(-1).unsqueeze(-1)  # pyre-ignore [16]
            * self.precision_root.t().unsqueeze(0),  # pyre-ignore [16]
            upper=True,
        ).solution
        return (self.mu + X.squeeze(-1)).to(self.original_dtype)  # pyre-ignore [16]
