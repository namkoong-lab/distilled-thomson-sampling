#!/usr/bin/env python3

"""Functions to create bandit problems from datasets."""

from __future__ import absolute_import, division, print_function
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import os
import torch
from .problem import (
    DiscreteActionBenchmarkProblem,
)
from ...core.hyperparameters import DatasetParams
from ...utils import one_hot_encode_df
from torch import Tensor


class Mushroom(DiscreteActionBenchmarkProblem):
    def __init__(self, n_examples: int, seed: Optional[int] = None) -> None:
        self._data_config = DatasetParams(context_dim=117, num_actions=2)
        if seed is not None:
            np.random.seed(seed)

        df = load_mushroom_data()
        dataset, opt_vals = sample_mushroom_data(
            data=df, n_samples=n_examples, dtype=self._data_config.dtype
        )
        self._contexts = dataset[:, : self._data_config.context_dim]
        self._rewards = dataset[:, self._data_config.context_dim :]
        self._optimal_rewards = opt_vals[:, 0:1]
        self._optimal_actions = opt_vals[:, 1:]


def load_fbpkg_data(fbpkg_name: str, dataset: Optional[str] = None) -> pd.DataFrame:
    with TemporaryDirectory() as dst:
        directory = Path(fetch(fbpkg_name, dst=dst))
        if dataset is not None:
            return pd.read_csv(directory / dataset, header=None, engine="python")
        else:
            return pd.read_csv(directory, header=None, engine="python")


def reformat_mushroom_data(df: pd.DataFrame) -> pd.DataFrame:
    column_swap = {"0": "y"}
    for i in range(1, len(df.columns)):
        column_swap[f"{i}"] = f"x_{i}"
    df = df.rename(columns=column_swap)

    # One hot encode the string features.
    del column_swap["0"]
    df = one_hot_encode_df(df, list(column_swap.values()))

    # Get the labels as 0, 1.
    df.loc[df["y"] == "p", "y"] = 0
    df.loc[df["y"] == "e", "y"] = 1
    return df


def load_mushroom_data() -> pd.DataFrame:
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'mushroom_data.csv')
    df = pd.read_csv(filename)
    return reformat_mushroom_data(df)


def sample_mushroom_data(
    data: pd.DataFrame,
    n_samples: int,
    dtype: torch.dtype,
    r_noeat: float = 0.0,
    r_eat_safe: float = 5.0,
    r_eat_poison_bad: float = -35.0,
    r_eat_poison_good: float = 5.0,
    prob_poison_bad: float = 0.5,
) -> Tuple[Tensor, Tensor]:
    """Samples bandit game from Mushroom UCI Dataset.

    Adds a reward for choosing not to eat, as well as a probability of sustaining
    harm from a poisonous mushroom.
    We assume r_eat_safe > r_noeat, and r_eat_poison_good > r_eat_poison_bad.

    Args:
        data: Mushroom UCI dataset in df.
        n_samples: Number of points to sample, i.e. (context, action rewards).
        dtype: dtype
        r_noeat: Reward for not eating a mushroom.
        r_eat_safe: Reward for eating a non-poisonous mushroom.
        r_eat_poison_bad: Reward for eating a poisonous mushroom if harmed.
        r_eat_poison_good: Reward for eating a poisonous mushroom if not harmed.
        prob_poison_bad: Probability of being harmed by eating a poisonous mushroom.
     Returns:
        Tensor: A `n x (d + num_actions)`-dim Tensor, where each row contains
            the context and corresponding reward for taking each action.
        Tensor: A `n x 2`-dim tensor containing expected optimal (reward, action)
            for each context.

    """
    # Draw samples.
    ind = np.random.choice(range(data.shape[0]), n_samples, replace=True)

    # Set up indices. In the future, this may be in a generic data transfomer.
    LABEL_COL = "y"
    POISON = 0
    EDIBLE = 1
    sample = data.iloc[ind]
    labels: np.ndarray = sample[LABEL_COL].values
    # Use booleans as 0/1 ints throughout.
    poisonous = labels == POISON
    edible = labels == EDIBLE

    # Replace with drop label column.
    no_eat_reward = r_noeat * np.ones((n_samples, 1))
    random_poison = np.random.choice(
        [r_eat_poison_bad, r_eat_poison_good],
        p=[prob_poison_bad, 1 - prob_poison_bad],
        size=n_samples,
    )
    # 1*r_eat_safe if edible, 0 else
    eat_reward: np.ndarray = r_eat_safe * edible  # pyre-ignore[9]: doesn't understand float * array
    eat_reward += np.multiply(random_poison, poisonous)
    eat_reward = eat_reward.reshape((n_samples, 1))

    # Compute optimal expected reward and optimal actions.
    exp_eat_poison_reward = r_eat_poison_bad * prob_poison_bad
    exp_eat_poison_reward += r_eat_poison_good * (1 - prob_poison_bad)
    opt_exp_reward: np.ndarray = (  # pyre-ignore[9]: doesn't understand float * array
        r_eat_safe * edible + max(r_noeat, exp_eat_poison_reward) * edible
    )

    if r_noeat > exp_eat_poison_reward:
        # Actions: no eat = 0; eat = 1
        opt_actions = edible
    else:
        # Should always eat (higher expected reward)
        opt_actions = np.ones((n_samples, 1))

    # Convert back from booleans to original action ints.
    opt_vals = torch.from_numpy(
        np.hstack(
            [opt_exp_reward.reshape(-1, 1), opt_actions.astype(int).reshape(-1, 1)]
        )
    ).to(dtype=dtype)
    contexts = sample.drop([LABEL_COL], axis=1)
    torch_dataset = torch.from_numpy(
        np.hstack((contexts, no_eat_reward, eat_reward)).astype(np.float32)
    ).to(dtype=dtype)
    return torch_dataset, opt_vals
