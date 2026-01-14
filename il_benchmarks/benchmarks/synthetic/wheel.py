#!/usr/bin/env python3

from typing import Optional, Tuple

import numpy as np
import torch
from .problem import (
    DiscreteActionBenchmarkProblem,
)
from ...core.hyperparameters import DatasetParams
from torch import Tensor


class Wheel(DiscreteActionBenchmarkProblem):
    def __init__(self, n_examples: int, seed: Optional[int] = None) -> None:
        self._data_config = DatasetParams(context_dim=2, num_actions=5)
        if seed is not None:
            np.random.seed(seed)

        dataset, opt_vals = sample_wheel_data(
            n_samples=n_examples,
            dtype=self._data_config.dtype,
            mean_safe=1.2,
            mean_nonsafe_bad=1.0,
            mean_nonsafe_good=50.0,
            stdev=0.01,
            prob_bad=0.95 ** 2,  # prob_bad (=delta^2)
        )
        self._contexts = dataset[:, : self._data_config.context_dim]
        # normalize contexts to [0,1]^2
        self._contexts = (self._contexts + 1) / 2
        self._rewards = dataset[:, self._data_config.context_dim :]
        self._optimal_rewards = opt_vals[:, 0:1]
        self._optimal_actions = opt_vals[:, 1:]


def sample_wheel_data(
    n_samples: int,
    dtype: torch.dtype,
    mean_safe: float = 1.5,
    mean_nonsafe_bad: float = 1.0,
    mean_nonsafe_good: float = 5.0,
    stdev: float = 1.0,
    prob_bad: float = 0.8,
) -> Tuple[Tensor, Tensor]:

    """Sample from Wheel bandit example (see the paper for details
    https://arxiv.org/abs/1802.09127). Their public release on github
    https://github.com/tensorflow/models/blob/master/research/deep_contextual_bandits/bandits/data/synthetic_data_sampler.py#L114
    seems somewhat flawed as they don't seem to actually sample from the unit
    norm uniformly at random. They do rejection sampling, but with an incorrect ratio.

    Args:
        n_samples: number of points to sample, i.e. (context, action rewards).
        dtype: dtype
        mean_safe: mean return when playing the safe action 0
        mean_nonsafe_bad: mean return for nonsafe action 1-4 that is not optimal
        mean_nonsafe_good: mean return for nonsafe action 1-4 that is optimal
        stdev: standard deviation for all returns in all scenarios
        prob_bad: conditional on nonsafe action 1-4, prob. of being optimal
    Returns:
        Tensor: A `n x (d + num_actions)`-dim Tensor, where each row contains
            the context and corresponding reward for taking each action.
        Tensor: A `n x 2`-dim tensor containing expected optimal (reward, action)
            for each context.
    """

    if mean_nonsafe_good < mean_nonsafe_bad or mean_nonsafe_good < mean_safe:
        raise ValueError(
            "Mean returns should satisfy mean_nonsafe_g > mean_safe and mean_nonsafe_good > mean_nonsafe_bad"
        )

    context_dim = 2
    num_actions = 5

    # sample contexts
    # sample in polar coordinates, then convert to euclidean
    theta = 2.0 * np.pi * np.random.uniform(0.0, 1.0, n_samples)
    radius = 1.0 * np.sqrt(np.random.uniform(0.0, 1.0, n_samples))
    # pyre-ignore [6]
    contexts = np.dstack((radius * np.cos(theta), radius * np.sin(theta))).reshape(
        (n_samples, context_dim)
    )

    # sample rewards
    r_safe = np.random.normal(mean_safe, stdev, n_samples).reshape(n_samples, 1)
    r_nonsafe = np.random.normal(mean_nonsafe_bad, stdev, (n_samples, num_actions - 1))
    r_nonsafe_good = np.random.normal(mean_nonsafe_good, stdev, n_samples)

    opt_vec = np.empty((n_samples, 2))  # reward, action

    # falls outside of the delta-radius region
    if_good = np.linalg.norm(contexts, axis=1) > np.sqrt(prob_bad)

    opt_vec[:, 0] = r_safe.reshape((-1,))
    opt_vec[if_good, 0] = r_nonsafe_good[if_good]  # good reward is optimal

    opt_vec[:, 1] = 0

    if_opt = (contexts[:, 0] > 0) * (contexts[:, 1] > 0) * if_good
    r_nonsafe[if_opt, 0] = r_nonsafe_good[if_opt]
    opt_vec[if_opt, 1] = 1

    if_opt = (contexts[:, 0] > 0) * (contexts[:, 1] < 0) * if_good
    r_nonsafe[if_opt, 1] = r_nonsafe_good[if_opt]
    opt_vec[if_opt, 1] = 2

    if_opt = (contexts[:, 0] < 0) * (contexts[:, 1] < 0) * if_good
    r_nonsafe[if_opt, 2] = r_nonsafe_good[if_opt]
    opt_vec[if_opt, 1] = 3

    if_opt = (contexts[:, 0] < 0) * (contexts[:, 1] > 0) * if_good
    r_nonsafe[if_opt, 3] = r_nonsafe_good[if_opt]
    opt_vec[if_opt, 1] = 4

    dataset = torch.from_numpy(np.hstack((contexts, r_safe, r_nonsafe))).to(dtype=dtype)

    opt_vec = torch.from_numpy(opt_vec).to(dtype=dtype)

    return dataset, opt_vec
