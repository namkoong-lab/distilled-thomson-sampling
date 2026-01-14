#!/usr/bin/env python3

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional

from ...core.hyperparameters import DatasetParams
from torch import Tensor


class BenchmarkProblem(ABC):
    """Abstract class for representing a benchmark problem."""

    _data_config: DatasetParams
    _contexts: Tensor
    _optimal_rewards: Optional[Tensor]
    _optimal_actions: Optional[Tensor]

    @property
    def name(self) -> str:
        """Return name of benchmark problem."""
        return self.__class__.__name__

    @property
    def contexts(self) -> Tensor:
        """Return `n x d`-dim tensor of contexts"""
        return self._contexts

    @property
    def optimal_rewards(self) -> Optional[Tensor]:
        """Return optimal expected rewards

        Returns:
            Tensor: A `n x 1`-dim tensor containing expected optimal reward
                for each context.
        """
        return self._optimal_rewards

    @property
    def optimal_actions(self) -> Optional[Tensor]:
        """Return optimal expected actions.

        Returns:
            Tensor: A `n x raw_encoded_dim`-dim tensor containing expected optimal
                action for each context.
        """
        return self._optimal_actions

    @abstractmethod
    def reward_func(
        self, contexts: Tensor, actions: Tensor, timesteps: Optional[Tensor] = None
    ) -> Tensor:
        """Compute rewards for given contexts and actions.

        Args:
            contexts: A `n x d`-dim tensor of contexts
            actions: A `n x raw_action_dim'-dim tensor of actions
            timesteps: A `n`-dim tensor containing the time step corresponding to
                each context. This allows passing the same context multiple times
                with different actions.
        Returns:
            Tensor: A `n x 1`-dim tensor of rewards
        """
        ...

    def get_data_config(self) -> DatasetParams:
        """Get the data config for the benchmark problem."""
        return deepcopy(self._data_config)


class DiscreteActionBenchmarkProblem(BenchmarkProblem, ABC):
    """Class for benchmark problems with discrete action spaces"""

    _rewards: Tensor

    def reward_func(
        self, contexts: Tensor, actions: Tensor, timesteps: Optional[Tensor] = None
    ) -> Tensor:
        """Compute rewards for given contexts and actions.

        Args:
            contexts: A `n x d`-dim tensor of contexts
            actions: A `n x raw_action_dim'-dim tensor of actions
            timesteps: A `n`-dim tensor containing the time step corresponding to
                each context. This allows passing the same context multiple times
                with different actions.
        Returns:
            Tensor: A `n x 1`-dim tensor of rewards
        """
        if timesteps is None:
            raise ValueError(f"timesteps is required for {self.__class__.__name__}")
        return self._rewards[timesteps].gather(dim=-1, index=actions)


class ObservedDataBenchmarkProblem(DiscreteActionBenchmarkProblem, ABC):
    """Class for benchmark problems based on logged data with discrete action spaces.

    The logged data consists of observation tuples (context, action, reward).

    Online (or online-batch) learning is performed by rejecting observation tuples
    when the action selected by the policy does not match the observed action.
    """

    _rewards: Tensor
    _actions: Tensor

    def reward_func(
        self, contexts: Tensor, actions: Tensor, timesteps: Optional[Tensor] = None
    ) -> Tensor:
        """Compute rewards for given contexts and actions.

        Args:
            contexts: A `n x d`-dim tensor of contexts
            actions: A `n x raw_action_dim'-dim tensor of actions
            timesteps: A `n`-dim tensor containing the time step corresponding to
                each context. This allows passing the same context multiple times
                with different actions.
        Returns:
            Tensor: A `n x 1`-dim tensor of rewards. A NaN indicates that the chosen
                action does not match the logged action.
        """
        if timesteps is None:
            raise ValueError(f"timesteps is required for {self.__class__.__name__}")
        observed_actions = self._actions[timesteps]
        rewards = self._rewards[timesteps].clone()
        same_action_mask = (observed_actions == actions).all(dim=1)
        # if chosen action does not match logged action, set reward to NaN
        rewards[~same_action_mask] = float("nan")
        return rewards
