#!/usr/bin/env python3

from typing import Optional

import os
import pandas as pd
import torch
from .problem import (
    DiscreteActionBenchmarkProblem,
)
from ...core.hyperparameters import DatasetParams
from botorch.utils.sampling import manual_seed


class Warfarin(DiscreteActionBenchmarkProblem):
    """
    Warfarin CB Problem.

    Given contextual features (17-dim):
        - demographic features
        - one-hot-encoded features for certain genetic markers
    The objective is to choose the optimal initial dosage.

    The reward is the negative L2 distance between the optimal dosage and the
    chosen dosage.

    We use a discrete action space.
    """

    def __init__(
        self, n_examples: int, seed: Optional[int] = None, num_actions: int = 5
    ) -> None:
        self._data_config = DatasetParams(context_dim=17, num_actions=num_actions)
        self.seed = seed
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'warfarin_data.csv')
        df = pd.read_csv(filename)
        # scale numeric features to [0,1]
        for col in ["age", "height", "weight"]:
            col_min = df[col].min()
            df[col] = (df[col] - col_min) / (df[col].max() - df[col].min())
        dataset = torch.tensor(df.values, dtype=self._data_config.dtype)
        with manual_seed(seed):
            shuffle_indices = torch.randperm(dataset.shape[0])
        # index=1 is an intercept, which is not necessary.
        self._contexts = dataset[:, 2:][shuffle_indices]
        # dosage is first column
        raw_optimal_actions = dataset[:, 0:1][shuffle_indices].pow(2)
        # normalize to [0, 1]
        self._raw_min_dose = raw_optimal_actions.min().item()
        self._raw_max_dose = raw_optimal_actions.max().item()
        self._optimal_actions = (raw_optimal_actions - self._raw_min_dose) / (
            self._raw_max_dose - self._raw_min_dose
        )
        # TODO: determine reasonable bounds (we may want to make this wider)
        min_dose = self._optimal_actions.min().item()
        max_dose = self._optimal_actions.max().item()
        possible_actions = torch.linspace(
            start=min_dose, end=max_dose, steps=num_actions, dtype=dataset.dtype
        ).unsqueeze(0)
        # Compute rewards for given contexts and actions. This is the negative
        # L2 distance between the action and the optimal action.
        self._rewards = -(possible_actions - self._optimal_actions).pow(2).sqrt()
        # Shift by maximum L2 distance to make the rewards non-negative.
        self._rewards -= self._rewards.min().item()
        self._optimal_rewards = self._rewards.max(dim=1)[0].view(-1, 1)
        self._all_contexts = dataset[:, 2:]
