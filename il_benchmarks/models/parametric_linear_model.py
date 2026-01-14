#!/usr/bin/env python3

from typing import Callable, List, Optional

import torch
from ..core.hyperparameters import (
    DatasetParams,
    Hyperparameters,
    ParametricBLRHyperparameters,
)
from .action_selection import (
    ActionSelectionStrategy,
    TActionCallable,
    action_strategy_factory,
)
from .bandit_model import BanditModel
from .common import PROP_SCORE_SAMPLES
from .parametric_bayesian_models import (
    ParametricBayesianLinearRegression,
)
from .utils import mini_batch_evaluate
from ..transforms.utils import apply_transforms
from ..utils import should_update
from torch import Tensor, nn


class ParametricLinearModel(BanditModel):
    def __init__(self, data_config: DatasetParams, hparams: Hyperparameters) -> None:
        super().__init__(
            data_config=data_config,
            hparams=hparams,
            action_encoder_params=hparams.action_encoder_params,
        )
        bayesian_hparams = hparams.bayesian_hparams
        if bayesian_hparams is None or not isinstance(
            bayesian_hparams, ParametricBLRHyperparameters
        ):
            raise ValueError(
                f"{type(self).__name__} requires Hyperparameters.bayesian_hparams"
            )
        self.bayesian_hparams: ParametricBLRHyperparameters = bayesian_hparams
        self.is_trained_blr = False
        self._instantiated_blr_target_transforms = nn.ModuleList([])
        self.blrs = nn.ModuleList([])
        self.strategy: ActionSelectionStrategy = ActionSelectionStrategy.PROBABILISTIC
        action_selection_methods = action_strategy_factory(strategy=self.strategy)
        self.select_action: TActionCallable = action_selection_methods[0]
        self.strategy_prop_scores: Callable[
            [Tensor], Tensor
        ] = action_selection_methods[1]
        # track number of previously seen observations for online learning
        self._t = 0

    def _concat_bias_column(self, X: Tensor) -> Tensor:
        """Concatenate a columns of ones to X.
        """
        return torch.cat([X, torch.ones_like(X[..., 0:1])], dim=-1)

    def reset_blr(self, d: int) -> None:
        """Re-initialize the BLR layer(s).

        If action_encoder_params is None, then an independent BLR is used to model
            the reward for each action. Otherwise, the action is part of the input
            and a single BLR is used.
        """
        if self.action_encoder_params is None:
            num_models = self.data_config.num_actions
        else:
            num_models = 1
        self.blrs = nn.ModuleList(
            [
                ParametricBayesianLinearRegression(
                    d=d,
                    dtype=self.data_config.dtype,
                    device=self.data_config.device,
                    a0=self.bayesian_hparams.a0,
                    b0=self.bayesian_hparams.b0,
                    lambda_prior=self.bayesian_hparams.lambda_prior,
                )
                for _ in range(num_models)
            ]
        )
        self._t = 0

    def prop_scores(self, X: Tensor, n_samples: int = PROP_SCORE_SAMPLES) -> Tensor:
        """Get Thompson Sampling propensity scores.

        The propensity scores for the linear model are the probabilities that each
        action is optimal. To obtain this, we sample actions (maximize expected reward),
        then count action frequencies. These values are normalized by the
        `strategy_prop_scores` function. Note: this method draws `n_samples` samples
        of the parameters, and uses those samples for estimating the propensity scores
        for all contexts. This is much faster than resampling parameters for each
        context.
        TODO: Add ability to sample betas for each context.

        Args:
            X: `n x d` tensor of contexts
            n_samples: number of MC samples to use to estimate the prop scores

        Returns:
            Tensor: `n x num_actions` tensor of propensity scores for each context.
        """
        inputs = self._get_inputs(X)
        # fix beta samples
        self._sample_betas(n_samples=n_samples)
        return self._prop_scores(X=inputs, n_samples=n_samples)

    @mini_batch_evaluate
    def _prop_scores(self, X: Tensor, n_samples: int = PROP_SCORE_SAMPLES) -> Tensor:
        values = self._predict(X=X, n_samples=n_samples)  # pyre-fixme: X is an argument
        action_samples = torch.max(values, dim=-1)[1].long()
        counts = torch.eye(
            self.data_config.num_actions, dtype=X.dtype, device=X.device
        )[action_samples].sum(dim=0)
        return self.strategy_prop_scores(counts.float())

    def fit_blr(
        self, X: Tensor, targets: Tensor, actions: Tensor, force_update: bool = False
    ) -> None:
        if (
            should_update(
                t=X.shape[0],
                training_freq=self.bayesian_hparams.training_freq,
                num_actions=self.data_config.num_actions,
                initial_pulls=self.hparams.initial_pulls,
            )
            or force_update
        ):
            self.is_trained_blr = True
            X = self._concat_bias_column(X=X)
            if self.bayesian_hparams.online_learning:
                # only update with new data
                X = X[self._t :]
                targets = targets[self._t :]
                actions = actions[self._t :]
            if (not self.bayesian_hparams.online_learning) or (
                self.bayesian_hparams.online_learning and len(self.blrs) == 0
            ):
                self.reset_blr(d=X.shape[1])
            train_Y = targets
            blr_target_transforms = self.bayesian_hparams.blr_target_transforms
            if blr_target_transforms is not None:
                if self.bayesian_hparams.online_learning:
                    raise NotImplementedError(
                        "Target transforms are not supported with online learning."
                    )
                self._instantiated_blr_target_transforms = nn.ModuleList(
                    [t.__class__(ndim=train_Y.shape[1]) for t in blr_target_transforms]
                )
                train_Y = apply_transforms(
                    X=train_Y, transforms=self._instantiated_blr_target_transforms
                )
            for a, model in enumerate(self.blrs):
                if self.action_encoder_params is None:
                    action_mask = (actions == a).view(-1)
                    model_inputs = X[action_mask]
                    model_targets = train_Y[action_mask]
                else:
                    model_inputs = X
                    model_targets = train_Y
                model.update(
                    X=model_inputs.to(device=self.data_config.device),
                    Y=model_targets.to(device=self.data_config.device).view(-1),
                )
            self._t += X.shape[0]

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
            targets: `n x 1` Tensor of targets
            actions: `n x 1` tensor of actions
            weights: `n x 1` Tensor of example weights
            force_update: A boolean indicating whether to ignore the training_freq
                and update the model
        """
        if actions is None:
            raise ValueError("actions are required for fitting {type(self).__name__}")
        # Fit BLR
        inputs = self._get_inputs(X=X, actions=actions)
        self.fit_blr(
            X=inputs, targets=targets, actions=actions, force_update=force_update
        )
        # is_trained is set in super().train(), but only for the network
        # is_trained should only be true when both the network and the blr have
        # been trained.
        self.is_trained = self.is_trained_blr

    @mini_batch_evaluate
    def _predict(
        self,
        X: Tensor,
        n_samples: int,
        extra_outcomes: Optional[Tensor] = None,
        extra_outcome_names: Optional[List[str]] = None,
    ) -> Tensor:
        """
        Get the posterior over the predicted reward for each action.
        Args:
            X: `n x (num_actions) x d_input` tensor of inputs (possibly with
                a concatenated action)
            n_samples: number of posterior draws
            extra_outcomes: A `n x m - 1`-dim tensor of outcomes to use in the reward
                function, where `m` is the total number of outcomes
            extra_outcome_names: A list of names of the outcomes

        Returns:
            Tensor: `(n_samples) x n x num_actions` tensor with the model predicted
             value for each action
        """
        inputs = self._concat_bias_column(X=X).to(device=self.data_config.device)
        if inputs.ndim == 2:  # pyre-ignore [16]
            inputs = inputs.unsqueeze(1)  # unsqueeze action_dim
        # n_samples x n x num_actions
        samples = (
            inputs.permute(1, 0, 2)
            @ self.beta_samples.permute(1, 2, 0)  # pyre-ignore [16]
        ).permute(2, 1, 0)
        blr_target_transforms = self.bayesian_hparams.blr_target_transforms
        if blr_target_transforms is not None:
            samples = apply_transforms(
                X=samples,
                transforms=self._instantiated_blr_target_transforms,
                reverse=True,
            )
        expected_rewards = self.reward_function(
            Y=samples.unsqueeze(-1),
            extra_outcomes=extra_outcomes,
            extra_outcome_names=extra_outcome_names,
        )
        if n_samples == 1:
            expected_rewards = expected_rewards.squeeze(0)
        return expected_rewards.cpu()

    def _sample_betas(self, n_samples: int) -> None:
        beta_samples_list = []
        for model in self.blrs:
            beta_samples_list.append(model.posterior_samples(n_samples=n_samples))
        if len(beta_samples_list) > 1:
            # pyre-ignore [16]
            self.beta_samples = torch.stack(beta_samples_list, dim=1)
        else:
            self.beta_samples = beta_samples_list[0].unsqueeze(1)

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
            extra_outcomes: A `n x m - 1`-dim tensor of outcomes to use in the reward
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

    def sample(
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
            extra_outcomes: A `n x m - 1`-dim tensor of outcomes to use in the reward
                function, where `m` is the total number of outcomes
            extra_outcome_names: A list of names of the outcomes

        Returns:
            Tensor: `(n_samples) x n` tensor action for each context
        """
        values = self.predict(
            X=X,
            n_samples=n_samples,
            extra_outcomes=extra_outcomes,
            extra_outcome_names=extra_outcome_names,
        )
        return torch.max(values, dim=-1)[1].type_as(X)
