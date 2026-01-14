#!/usr/bin/env python3

from typing import List, Optional

import torch
from ..core.hyperparameters import (
    DatasetParams,
    Hyperparameters,
    ParametricBLRHyperparameters,
)
from .action_selection import ActionSelectionStrategy
from .base_network import FCNetwork
from .neural_bandit_models import NeuralBanditRegressor
from .parametric_linear_model import ParametricLinearModel
from .utils import mini_batch_evaluate
from torch import Tensor, nn


class ParametricNeuralLinearModel(ParametricLinearModel, NeuralBanditRegressor):
    """A parametric Neural Linear Model.

    If action_encoder_params is None, then an independent BLR is used to model
        the reward for each action. Otherwise, the action is part of the input
        and a single BLR is used.
    """

    def __init__(
        self, data_config: DatasetParams, network: FCNetwork, hparams: Hyperparameters
    ) -> None:
        NeuralBanditRegressor.__init__(
            self,
            data_config=data_config,
            network=network,
            hparams=hparams,
            strategy=ActionSelectionStrategy.PROBABILISTIC,
        )
        bayesian_hparams = hparams.bayesian_hparams
        if bayesian_hparams is None or not isinstance(
            bayesian_hparams, ParametricBLRHyperparameters
        ):
            raise ValueError(
                f"{type(self).__name__} requires Hyperparameters.bayesian_hparams"
            )
        self.bayesian_hparams = bayesian_hparams
        self.is_trained_blr = False
        self._instantiated_blr_target_transforms = nn.ModuleList([])

    def fit(
        self,
        X: Tensor,
        targets: Tensor,
        actions: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        force_update: bool = False,
    ) -> None:
        """Trains the network and the GP.

        Args:
            X: `n x d` Tensor of contexts
            actions: `n x 1` tensor of actions
            targets: `n x 1` Tensor of targets
            weights: `n x 1` Tensor of example weights
            force_update: A boolean indicating whether to ignore the training_freq
                and update the model
        """
        if actions is None:
            raise ValueError("actions are required for fitting {type(self).__name__}")
        # Train network
        NeuralBanditRegressor.fit(
            self,
            X=X,
            targets=targets,
            actions=actions,
            weights=weights,
            force_update=force_update,
        )
        # Fit BLR
        inputs = self._get_inputs(X=X, actions=actions)
        with torch.no_grad():
            latent_X = self.network.forward_latent_features(
                inputs.to(device=self.data_config.device)
            ).cpu()
        self.fit_blr(
            X=latent_X, targets=targets, actions=actions, force_update=force_update
        )
        # is_trained is set in super().train(), but only for the network
        # is_trained should only be true when both the network and the blr have
        # been trained.
        self.is_trained = self.is_trained and self.is_trained_blr

    def predict(
        self,
        X: Tensor,
        n_samples: int = 1,
        extra_outcomes: Optional[Tensor] = None,
        extra_outcome_names: Optional[List[str]] = None,
        **kwargs,
    ) -> Tensor:
        """
        Get the posterior over the predicted reward for each action.
        Args:
            X: `n x d` tensor of contexts
            n_samples: number of posterior draws
            extra_outcomes: A `n x (m - 1)`-dim tensor of outcomes to use in the reward
                function, where `m` is the total number of outcomes
            extra_outcome_names: A list of names of the outcomes

        Returns:
            Tensor: `(n_samples) x n x num_actions` tensor with the model predicted
             value for each action
        """
        inputs = self._get_inputs(X)
        self._sample_betas(n_samples=n_samples)
        return self._predict(
            X=inputs,
            n_samples=n_samples,
            extra_outcomes=extra_outcomes,
            extra_outcome_names=extra_outcome_names,
        )

    @mini_batch_evaluate
    def _predict(
        self,
        X: Tensor,
        n_samples: int,
        extra_outcomes: Optional[Tensor] = None,
        extra_outcome_names: Optional[List[str]] = None,
    ) -> Tensor:
        with torch.no_grad():
            inputs = self.network.forward_latent_features(
                X.to(device=self.data_config.device)
            )
        return super()._predict(
            X=inputs,
            n_samples=n_samples,
            extra_outcomes=extra_outcomes,
            extra_outcome_names=extra_outcome_names,
        )
