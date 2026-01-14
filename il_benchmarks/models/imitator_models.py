#!/usr/bin/env python3

from abc import ABC, abstractmethod
from typing import Callable, Tuple

import torch
from torch import Tensor

from ..core.hyperparameters import DatasetParams, Hyperparameters, ImitatorHyperparams
from ..utils import should_update
from .action_selection import (
    ActionSelectionStrategy,
    TActionCallable,
    action_strategy_factory,
)
from .bandit_model import BanditModel
from .base_network import FCClassifier
from .neural_bandit_models import NeuralBanditMultinomialImitator


class ImitatorModel(ABC):
    """
    Imitates an existing stochastic Bandit model.
    """

    def __init__(
        self, data_config: DatasetParams, imitator_hparams: ImitatorHyperparams
    ) -> None:
        self.data_config = data_config
        self.is_trained = False
        self.hparams = imitator_hparams
        self.strategy: ActionSelectionStrategy = ActionSelectionStrategy.PROBABILISTIC
        action_selection_methods = action_strategy_factory(strategy=self.strategy)
        self.select_action: TActionCallable = action_selection_methods[0]
        self.strategy_prop_scores: Callable[
            [Tensor], Tensor
        ] = action_selection_methods[1]

    @abstractmethod
    def imitate(self, X: Tensor, target_model: BanditModel) -> None:
        pass

    @abstractmethod
    def _imitation_data(
        self, X: Tensor, target_model: BanditModel
    ) -> Tuple[Tensor, Tensor]:
        pass

    def prop_scores(self, X: Tensor) -> torch.Tensor:
        """Return prop scores corresponding to action selection strategy.

        Args:
            X: `n x d` tensor of contexts
        Returns:
            Tensor: `n x num_actions` tensor w/ probs for each action
        """
        preds = self.predict(X)
        return self.strategy_prop_scores(preds)

    def predict(self, X: Tensor) -> Tensor:
        pass


class NNMultinomialImitator(ImitatorModel):
    """NN Imitator for target bandit models with discrete actions."""

    def __init__(
        self,
        data_config: DatasetParams,
        imitator_hparams: ImitatorHyperparams,
        hparams: Hyperparameters,
    ) -> None:
        super().__init__(data_config=data_config, imitator_hparams=imitator_hparams)
        nn_hparams = hparams.nn_hparams
        if nn_hparams is None:
            raise ValueError(
                "Hyperparameters.nn_hparams is required for NeuralBanditModels."
            )
        self.imitator = NeuralBanditMultinomialImitator(
            data_config=data_config,
            network=FCClassifier(data_config=data_config, hparams=nn_hparams),
            hparams=hparams,
        )
        self.training_freq: int = nn_hparams.training_freq

    def imitate(self, X: torch.Tensor, target_model: BanditModel) -> None:
        if should_update(
            t=X.shape[0],
            training_freq=self.training_freq,
            initial_pulls=self.hparams.initial_pulls,
            num_actions=self.data_config.num_actions,
        ):
            self.is_trained = True
            X_, probs = self._imitation_data(X=X, target_model=target_model)
            self.imitator.fit(X=X_, targets=probs)
            if self.imitator.strategy != target_model.strategy:
                self.imitator.strategy = target_model.strategy
                action_selection_methods = action_strategy_factory(
                    strategy=self.imitator.strategy
                )
                self.imitator.select_action = action_selection_methods[0]
                self.imitator.strategy_prop_scores = action_selection_methods[1]

    def _imitation_data(
        self, X: Tensor, target_model: BanditModel
    ) -> Tuple[Tensor, Tensor]:
        """Obtain data for imitation model.

        First, get the action distribution from the target model.
        Sample from this distribution, and repeat contexts as necessary.

        Args:
            X: Tensor of contexts.
            target_model: BanditModel to imitate.

        Returns:
            sampled_X: `n x d` tensor of contexts used as input
            probs: `n x num_actions` tensor of probs
        """
        # Get multinomial targets inducing by target model policy
        multinomial_targets = target_model.prop_scores(
            X=X, n_samples=self.hparams.num_mc_samples
        )
        return X, multinomial_targets

    def predict(self, X: torch.Tensor, n_samples: int = 1) -> Tensor:
        return self.imitator.predict(X=X, n_samples=n_samples)

    def sample(self, X: torch.Tensor, n_samples: int = 1) -> Tensor:
        return self.imitator.sample(X=X, n_samples=n_samples)
