#!/usr/bin/env python3

from enum import IntEnum
from typing import Callable, Tuple

import torch
from torch import Tensor


class ActionSelectionStrategy(IntEnum):
    GREEDY = 0
    PROBABILISTIC = 1


TActionCallable = Callable[[Tensor, int], Tensor]


def action_strategy_factory(
    strategy: ActionSelectionStrategy
) -> Tuple[TActionCallable, Callable[[Tensor], Tensor]]:
    """
    Factory method returns a callable for selecting actions based on the specified
        strategy.
    Args:
        strategy: An int representing the strategy type based on ActionSelectionStrategy
    Returns:
        1. A callable for selecting actions.
        2. A callable for obtaining propensity scores, based on the first callable.

    """
    if strategy == ActionSelectionStrategy.GREEDY:
        return greedy_action, greedy_prop_scores
    elif strategy == ActionSelectionStrategy.PROBABILISTIC:
        return probabilistic_action, normalize_values
    else:
        raise ValueError("Provided strategy is not currently supported.")


def greedy_action(values: Tensor, n_samples: int = 1) -> Tensor:
    """
    Select action (index) greedily using provided values.
    Args:
        values: `n x num_actions` tensor of values for each action
    Returns:
        Tensor: `n` tensor of actions
    """
    action = torch.max(values, dim=-1)[1]
    return action.type_as(values)


def greedy_prop_scores(values: Tensor) -> Tensor:
    """Obtain propensity scores under greedy action selection.

    With greedy selection, the prop scores should simply be 1.0 for the greedy
    action and 0.0 for all others.

    Args:
        values: `n x num_actions` tensor of values for each action
    Returns:
        Tensor: `n x num_actions` tensor of propensity scores for each action.
            Binary tensor, one for greedy action, zero elsewhere.
    """
    prop_scores = torch.zeros_like(values)
    argmax_indices = torch.argmax(values, dim=1)
    row_indices = torch.arange(argmax_indices.shape[0])
    prop_scores[row_indices, argmax_indices] = 1.0
    return prop_scores


def probabilistic_action(values: Tensor, n_samples: int = 1) -> Tensor:
    """
    Select action (index) from multinomial distribution
    Args:
        values: `n x num_actions` tensor of probabilities for each action. Each
            row must specify a valid multinomial probability distribution.
        n_samples: number of samples
    Returns:
        Tensor: `(n_samples) x n` tensor of actions
    """
    actions = (
        torch.multinomial(input=values, num_samples=n_samples, replacement=True)
        .transpose(0, 1)
        .contiguous()
        .to(values)
    )
    if n_samples == 1:
        return actions.squeeze(0)
    return actions


def normalize_values(values: Tensor) -> Tensor:
    """Normalize input values.

    The normalized values can then be used to obtain propensity scores under
    probabilistic action selection. The final propensity scores will depend on
    the model used.

    Args:
        values: `n x num_actions` tensor of expected rewards for each action
    Returns:
        Tensor: `n x num_actions` tensor of normalized expected rewards for each action.
    """
    return values / values.sum(dim=1, keepdim=True)
