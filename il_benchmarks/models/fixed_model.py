#!/usr/bin/env python3

from typing import Callable

from .action_selection import (
    ActionSelectionStrategy,
    TActionCallable,
    action_strategy_factory,
)
from .bandit_model import BanditModel
from .common import PROP_SCORE_SAMPLES
from torch import Tensor


class FixedModel(BanditModel):
    """Defines a baseline model; returns a fixed posterior distribution."""

    def __init__(self, p: Tensor, num_actions: int) -> None:
        """Creates a FixedPolicySampling object.
        Args:
            p: Vector of normalized probabilities corresponding to sampling each arm.
            num_actions: number of actions.

        Raises:
            ValueError: when p dimension does not match the number of actions.
        """
        if len(p) != num_actions:
            raise ValueError(
                "Length of FixedPolicy probabilities must be the same as the "
                "number of actions"
            )

        self.p = p
        self.num_actions = num_actions
        self.is_trained = True
        self.strategy: ActionSelectionStrategy = ActionSelectionStrategy.PROBABILISTIC
        action_selection_methods = action_strategy_factory(strategy=self.strategy)
        self.select_action: TActionCallable = action_selection_methods[0]
        self.strategy_prop_scores: Callable[
            [Tensor], Tensor
        ] = action_selection_methods[1]

    def predict(self, X: Tensor, n_samples: int = 1) -> Tensor:
        """Return an n x num_actions vector of predicted distribution."""
        if n_samples != 1:
            raise ValueError("FixedModel only supports n_sample = 1.")
        return self.p.clone().detach().type_as(X).repeat(X.shape[0], 1)

    def prop_scores(self, X: Tensor, n_samples: int = PROP_SCORE_SAMPLES) -> Tensor:
        return self.predict(X=X)

    def sample(self, X: Tensor, n_samples: int = 1, **kwargs) -> Tensor:
        """Return a `(n_samples) x n` tensor of actions."""
        probs = self.predict(X=X, n_samples=1)
        return self.select_action(probs, n_samples)
