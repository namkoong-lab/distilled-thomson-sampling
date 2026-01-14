#!/usr/bin/env python3
from abc import ABC
from enum import Enum
from typing import List, Optional

import torch
from torch import Tensor
from torch.nn import Module


class RewardFunctionType(Enum):
    LINEAR: str = "linear"


class RewardFunction(ABC, Module):
    """Abstract Reward Function Class"""

    reward_parameters: Tensor

    def __init__(self, reward_parameters: Tensor, **kwargs) -> None:
        super().__init__()
        self.register_buffer("reward_parameters", reward_parameters)

    def forward(
        self,
        Y: Tensor,
        extra_outcomes: Optional[Tensor] = None,
        extra_outcome_names: Optional[List[str]] = None,
    ) -> Tensor:
        r"""Compute rewards using provided outcomes.

        Args:
            Y: A `(batch_shape) x n x num_actions x 1`-dim Tensor of outcomes
            extra_outcomes: A `n x (num_actions) x m-1`-dim Tensor of outcomes
            extra_outcome_names: A list of names of the outcomes

        Returns:
            A `(batch_shape) x n x num_actions`-dim Tensor of rewards.

        """
        pass

    @staticmethod
    def _expand_extra_outcomes(Y: Tensor, extra_outcomes: Tensor) -> Tensor:
        """Expand extra_outcomes to include an dimension for num_actions (if missing).

        Args:
            Y: A `(batch_shape) x n x num_actions x 1`-dim Tensor of outcomes
            extra_outcomes: A dictionary mapping outcome names to
                `n x (num_actions) x 1`-dim Tensors of outcomes
        """
        if extra_outcomes.ndim == 3:  # pyre-ignore [16]
            # extra outcomes has all dimensions
            if Y.shape[-3:-1] != extra_outcomes.shape[:-1]:
                raise ValueError(
                    "If extra_outcomes has the same number of dimensions as Y, "
                    "then their shapes must match in all but the last dimension. "
                    f"Y.shape: {Y.shape}, extra_outcomes.shape: {extra_outcomes.shape}."
                )
        elif extra_outcomes.ndim == 2:
            # extra outcomes is missing a dimension
            if Y.shape[-3] != extra_outcomes.shape[0]:
                # missing dimension is not actions dimension
                raise NotImplementedError("Outcomes must be provided for all contexts.")
            else:
                # extra outcomes is actions
                # pyre-ignore [6]
                expanded_shape = Y.shape[-3:-1] + torch.Size([extra_outcomes.shape[-1]])
                extra_outcomes = extra_outcomes.unsqueeze(-2).expand(expanded_shape)
        else:
            raise ValueError(
                "Unsupported shapes: "
                f"Y.shape: {Y.shape}, extra_outcomes.shape: {extra_outcomes.shape}."
            )
        return extra_outcomes


# pyre-fixme [13]: `register_buffer` initializes attribute
class LinearRewardFunction(RewardFunction):
    def __init__(
        self, reward_parameters: Tensor, num_actions: int, num_outcomes: int, **kwargs
    ) -> None:
        self.num_actions = num_actions
        self.num_outcomes = num_outcomes
        expected_shape = torch.Size([self.num_actions, self.num_outcomes])
        if len(reward_parameters.shape) == 1:
            reward_parameters = reward_parameters.unsqueeze(dim=-1)
        if reward_parameters.shape != expected_shape:
            raise ValueError(
                f"{type(self).__name__} requires reward_parameters with shape:"
                f" {expected_shape}"
            )
        super().__init__(reward_parameters=reward_parameters)

    def forward(
        self,
        Y: Tensor,
        extra_outcomes: Optional[Tensor] = None,
        extra_outcome_names: Optional[List[str]] = None,
    ) -> Tensor:
        r"""Compute linear rewards using provided outcomes.

        Args:
            Y: A `(batch_shape) x n x num_actions x 1`-dim Tensor of outcomes
            extra_outcomes: A dictionary mapping outcome names to
                `n x (num_actions) x 1`-dim Tensors of outcomes
            extra_outcome_names: A list of names of the outcomes

        Returns:
            A `(batch_shape) x n x num_actions`-dim Tensor of rewards.
        """
        # compute linear terms from Y
        rewards = torch.einsum(
            "...km, km -> ...k", [Y, self.reward_parameters[:, 0:1].to(Y)]
        )
        if self.num_outcomes > 1:
            if extra_outcomes is None:
                raise ValueError(
                    f"{type(self).__name__}.forward requires extra_outcomes"
                    " when num_outcomes > 1, but got None."
                )
            extra_outcomes = self._expand_extra_outcomes(
                Y=Y, extra_outcomes=extra_outcomes
            )
            # compute linear terms from extra_outcomes
            # this avoids expanding the extra_outcomes to Y's batch_shape
            rewards += torch.einsum(
                "...km, km -> ...k",
                [extra_outcomes, self.reward_parameters[:, 1:].to(Y)],
            )
        return rewards


def get_reward_function(
    reward_function_type: RewardFunctionType,
    reward_parameters: Tensor,
    num_actions: int,
    num_outcomes: int,
) -> RewardFunction:
    """Factory function for creating RewardFunctions.

    Args:
        reward_function_type: type of reward function
        reward_parameters: A tensor of parameters for the reward function

    Returns:
        A RewardFunction.
    """

    if reward_function_type == RewardFunctionType.LINEAR:
        return LinearRewardFunction(
            reward_parameters=reward_parameters,
            num_actions=num_actions,
            num_outcomes=num_outcomes,
        )
    raise ValueError(
        f"Invalid reward_function_type specified. Got {reward_function_type}"
    )
