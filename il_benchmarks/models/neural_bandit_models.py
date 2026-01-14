#!/usr/bin/env python3
from typing import Callable, List, Optional, Tuple

import torch
from ..core.hyperparameters import DatasetParams, Hyperparameters
from .action_selection import (
    ActionSelectionStrategy,
    TActionCallable,
    action_strategy_factory,
)
from .bandit_model import BanditModel
from .base_network import FCClassifier, FCNetwork
from .fit import evaluate_loss, fit_model
from .utils import (
    mini_batch_evaluate,
    uniform_weight_initializer,
)
from ..utils import one_hot_encode_action, should_update
from torch import Tensor
from torch.nn import BCELoss, KLDivLoss, MSELoss


class NeuralBanditModel(BanditModel, torch.nn.Module):
    """Implements a neural network for bandit problems.

    If action_encoder_params is None, then the NN has an output for each action.
        Otherwise, the action is part of the input and the NN has a single
        output.
    """

    loss_module: torch.nn.Module

    def __init__(
        self,
        data_config: DatasetParams,
        network: FCNetwork,
        hparams: Hyperparameters,
        strategy: ActionSelectionStrategy = ActionSelectionStrategy.GREEDY,
    ) -> None:
        """Saves hyper-params and builds the Torch network."""
        super().__init__(
            data_config=data_config,
            hparams=hparams,
            action_encoder_params=hparams.action_encoder_params,
        )
        nn_hparams = hparams.nn_hparams
        if nn_hparams is None:
            raise ValueError(
                "Hyperparameters.nn_hparams is required for NeuralBanditModels."
            )
        self.network = network
        self.network.apply(
            lambda m: uniform_weight_initializer(
                m, -nn_hparams.init_scale, nn_hparams.init_scale
            )
        )
        self.lr: float = nn_hparams.initial_lr
        self.strategy: ActionSelectionStrategy = strategy
        action_selection_methods = action_strategy_factory(strategy=self.strategy)
        self.select_action: TActionCallable = action_selection_methods[0]
        self.strategy_prop_scores: Callable[
            [Tensor], Tensor
        ] = action_selection_methods[1]
        self.loss_module = MSELoss(reduction="none")

    @mini_batch_evaluate
    def predict(self, X: Tensor, n_samples: int = 1, **kwargs) -> Tensor:
        """
        Get the predicted reward for each action.
        Args:
            X: `n x d` tensor of contexts
            n_samples: number of samples
        Returns:
            Tensor: `(n_samples) x n x num_actions` tensor with the model predicted
             value for each action
        Raises:
            ValueError: If n_samples != 1.
        """
        if n_samples != 1:
            raise ValueError(
                "NeuralNetworkModel `predict` only supports n_samples = 1."
            )
        with torch.no_grad():
            return self.network(X.to(device=self.data_config.device)).cpu()

    @mini_batch_evaluate
    def prop_scores(self, X: Tensor, n_samples: int = 1) -> Tensor:
        """Get prob scores corresponding to on action selection strategy.

        Args:
            X: `n x d` tensor of contexts
            n_samples: number of MC samples to use to estimate the prop scores.
            For deterministic policies, this is ignored.

        Returns:
            Tensor: `n x num_actions` tensor of propensity scores for each action.
        """
        values = self.predict(X=X, n_samples=1)
        return self.strategy_prop_scores(values)

    def sample(
        self,
        X: Tensor,
        n_samples: int = 1,
        extra_outcomes: Optional[Tensor] = None,
        extra_outcome_names: Optional[List[str]] = None,
        **kwargs,
    ) -> Tensor:
        """Sample action using on action selection strategy.

        Args:
            X: `n x d` tensor of contexts
            n_samples: number of samples
            extra_outcomes: A `n x (m - 1)`-dim tensor of outcomes to use in the reward
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
            **kwargs,
        )
        return self.select_action(values, n_samples)

    def fit(
        self,
        X: Tensor,
        targets: Tensor,
        actions: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        force_update: bool = False,
    ) -> None:
        """Trains the network for using the provided data.

        Args:
            X: `n x d` Tensor of contexts
            targets: `n x 1` Tensor of targets
            actions: `n x 1` tensor of actions
            weights: `n x 1` Tensor of example weights
            force_update: A boolean indicating whether to ignore the training_freq
                and update the model
        """
        raise NotImplementedError

    def _fit(
        self,
        X: Tensor,
        targets: Tensor,
        actions: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        output_mask_numeric: Optional[Tensor] = None,
        force_update: bool = False,
    ) -> None:
        """Trains the network for using the provided data.

        Args:
            X: `n x d` Tensor of contexts
            targets: `n x target_dim` Tensor of targets
            actions: `n x 1` tensor of actions
            weights: `n x 1` Tensor of example weights
        """
        nn_hparams = self.hparams.nn_hparams
        assert nn_hparams is not None  # for pyre
        if (
            should_update(
                t=X.shape[0],
                training_freq=nn_hparams.training_freq,
                initial_pulls=self.hparams.initial_pulls,
                num_actions=self.data_config.num_actions,
            )
            or force_update
        ):
            self.is_trained = True
            new_lr = fit_model(
                model=self.network,
                inputs=self._get_inputs(X=X, actions=actions),
                targets=targets,
                criterion=self.loss_module,
                model_hparams=nn_hparams,
                lr=self.lr,
                verbose=self.hparams.verbose,
                weights=weights,
                output_mask_numeric=output_mask_numeric,
            )
            if not nn_hparams.reset_lr:
                self.lr = new_lr

    def compute_loss(
        self,
        X: Tensor,
        targets: Tensor,
        actions: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
    ) -> float:
        """Compute loss on provided data.

        Args:
            X: `n x d` tensor of contexts
            targets: `n x 1` Tensor of targets
            actions: `n x 1` tensor of actions
            weights: `n x 1` Tensor of example weights
        Returns:
            float: loss
        """
        raise NotImplementedError


# pyre-fixme [13]: inherited attributes are initialized
class NeuralBanditRegressor(NeuralBanditModel):
    """Implements a neural network regressor for bandit problems."""

    def __init__(
        self,
        data_config: DatasetParams,
        network: FCNetwork,
        hparams: Hyperparameters,
        strategy: ActionSelectionStrategy = ActionSelectionStrategy.GREEDY,
    ) -> None:
        super().__init__(
            data_config=data_config, network=network, hparams=hparams, strategy=strategy
        )

    def fit(
        self,
        X: Tensor,
        targets: Tensor,
        actions: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        force_update: bool = False,
    ) -> None:
        """Trains the network for num_steps, using the provided data.

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
        # make targets multioutput when using a multioutput model
        targets = (
            targets
            * one_hot_encode_action(X=actions, num_actions=self.data_config.num_actions)
            if self.action_encoder_params is None
            else targets
        )
        self._fit(
            X=X,
            targets=targets,
            actions=actions,
            weights=weights,
            output_mask_numeric=actions if self.action_encoder_params is None else None,
            force_update=force_update,
        )

    @mini_batch_evaluate
    def predict(
        self,
        X: Tensor,
        n_samples: int = 1,
        extra_outcomes: Optional[Tensor] = None,
        extra_outcome_names: Optional[List[str]] = None,
        **kwargs,
    ) -> Tensor:
        """
        Get the predicted reward for each action.
        Args:
            X: `n x d` tensor of contexts
            n_samples: number of samples
            extra_outcomes: A `n x (m - 1)`-dim tensor of outcomes to use in the reward
                function, where `m` is the total number of outcomes
            extra_outcome_names: A list of names of the outcomes

        Returns:
            Tensor: `(n_samples) x n x num_actions` tensor with the model predicted
             value for each action
        Raises:
            ValueError: If n_samples != 1.
        """

        if n_samples != 1:
            raise ValueError("NeuralBanditModel only n_samples = 1.")
        inputs = self._get_inputs(X=X)
        with torch.no_grad():
            output = self.network(inputs.to(device=self.data_config.device)).cpu()
        if self.action_encoder_params is None:
            # make output is n x num_actions x 1
            output = output.unsqueeze(-1)
        return self.reward_function(
            Y=output,
            extra_outcomes=extra_outcomes,
            extra_outcome_names=extra_outcome_names,
        )

    def compute_loss(
        self,
        X: Tensor,
        targets: Tensor,
        actions: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
    ) -> float:
        if actions is None:
            raise ValueError(
                "actions are required for evaluating loss of {type(self).__name__}"
            )
        # make targets multioutput when using a multioutput model
        targets = (
            targets
            * one_hot_encode_action(X=actions, num_actions=self.data_config.num_actions)
            if self.action_encoder_params is None
            else targets
        )
        return evaluate_loss(
            model=self.network,
            inputs=self._get_inputs(X=X, actions=actions),
            targets=targets,
            criterion=self.loss_module,
            weights=weights,
            output_mask_numeric=actions if self.action_encoder_params is None else None,
        )


class NeuralBanditClassifier(NeuralBanditModel):
    """Implements a neural network classifier for bandit problems."""

    def __init__(
        self,
        data_config: DatasetParams,
        network: FCClassifier,
        hparams: Hyperparameters,
        strategy: ActionSelectionStrategy = ActionSelectionStrategy.GREEDY,
    ) -> None:
        if hparams.action_encoder_params is None:
            raise ValueError(
                "NeuralBanditClassifier requires that the action is embedded"
                "with the input."
            )
        super().__init__(
            data_config=data_config, network=network, hparams=hparams, strategy=strategy
        )
        self.loss_module = BCELoss(reduction="none")
        self.ts_asc_alpha = hparams.ts_asc_alpha

    def predict(
        self,
        X: Tensor,
        n_samples: int = 1,
        max_actions: Optional[Tensor] = None,
        status_quo_actions: Optional[Tensor] = None,
        extra_outcomes: Optional[Tensor] = None,
        extra_outcome_names: Optional[List[str]] = None,
        **kwargs,
    ) -> Tensor:
        """
        Get the predicted reward for each action.
        Args:
            X: `n x d` tensor of contexts
            n_samples: number of posterior draws
            max_actions: `n x 1` tensor containing the maximum allowed action
                for each context. This type of constraint is useful in cases where
                actions have an inate ordering.
            status_quo_actions: `n x 1` tensor of actions taken under the status
                quo policy used for evaluating constraints of the form:
                `A_feas = {P(success|a) >= (1-alpha) * P(success | a_status_quo)}`
            extra_outcomes: A `n x (m - 1)`-dim tensor of outcomes to use in the reward
                function, where `m` is the total number of outcomes
            extra_outcome_names: A list of names of the outcomes

        Returns:
            Tensor: `(n_samples) x n x num_actions` tensor with the model predicted
             value for each action
        """
        samples, feasibility = self.sample_p(
            X=X,
            n_samples=n_samples,
            max_actions=max_actions,
            status_quo_actions=status_quo_actions,
        )
        expected_rewards = self.reward_function(
            Y=samples.unsqueeze(-1),
            extra_outcomes=extra_outcomes,
            extra_outcome_names=extra_outcome_names,
        )
        expected_rewards[~feasibility] = float("-inf")  # pyre-ignore [16]
        return expected_rewards

    def fit(
        self,
        X: Tensor,
        targets: Tensor,
        actions: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        force_update: bool = False,
    ) -> None:
        """Trains the network for num_steps, using the provided data.

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
        binary_classes = (targets > 0).type_as(targets)
        self._fit(
            X=X,
            targets=binary_classes,
            actions=actions,
            weights=weights,
            force_update=force_update,
        )

    def compute_loss(
        self,
        X: Tensor,
        targets: Tensor,
        actions: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
    ) -> float:
        if targets.shape[-1] != 1:
            raise ValueError(
                "{type(self).__name__} does not support multi-dimensional targets."
            )
        elif actions is None:
            raise ValueError(
                "actions are required for evaluating loss of {type(self).__name__}"
            )

        binary_classes = (targets > 0).type_as(targets)
        return evaluate_loss(
            model=self.network,
            inputs=self._get_inputs(X=X, actions=actions),
            targets=binary_classes,
            criterion=self.loss_module,
            weights=weights,
        )

    @mini_batch_evaluate
    def _sample_p(self, X: Tensor, n_samples: int = 1) -> Tensor:
        """Sample the probability of success from the posterior.
        Args:
            X: `n x d` tensor of contexts
            n_samples: number of posterior draws
        Returns:
            Tensor: `n x n_actions` tensor with sampled probabilities
                of success for each action
        """
        if n_samples != 1:
            raise ValueError(
                "NeuralNetworkModel `predict` only supports n_samples = 1."
            )
        inputs = self._get_inputs(X=X)
        with torch.no_grad():
            probs = self.network(inputs.to(self.data_config.device)).cpu()
        return probs.squeeze(-1)

    def sample_p(
        self,
        X: Tensor,
        n_samples: int = 1,
        max_actions: Optional[Tensor] = None,
        status_quo_actions: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Sample the probability of success from the posterior and compute the
            feasibility of each action.
        Args:
            X: `n x d` tensor of contexts
            n_samples: number of posterior draws
            max_actions: `n x 1` tensor containing the maximum allowed action
                for each context. This type of constraint is useful in cases where
                actions have an inate ordering.
            status_quo_actions: `n x num_actions` tensor of one-hot encoded actions
                taken under the status quo policy used for evaluating constraints
                of the form:
                `A_feas = {P(success|a) >= P(success | a_status_quo) * eps}`
        Returns:
            Tensor: `(n_samples) x n x n_actions` tensor with sampled probabilities
                of success for each action
            Tensor: `(n_samples) x n x n_actions` byte-tensor with binary feasibilities
                for each action
        """
        samples = self._sample_p(X=X, n_samples=n_samples)
        # apply constraints
        feasible_mask = torch.ones(samples.shape, dtype=torch.bool)
        # apply constraints based on action from status quo policy
        if status_quo_actions is not None and self.ts_asc_alpha is not None:
            status_quo_actions = one_hot_encode_action(
                X=status_quo_actions, num_actions=self.data_config.num_actions
            )
            if status_quo_actions.dim() < samples.dim():
                # add batch dim for n_samples
                status_quo_actions = status_quo_actions.unsqueeze(0).expand(
                    samples.shape
                )
            samples_for_status_quo_action = samples[status_quo_actions == 1].view(
                status_quo_actions.shape[:-1]
            )
            samples_for_status_quo_action = samples_for_status_quo_action.unsqueeze(
                -1
            ).expand(samples.shape)
            feasible_mask *= samples >= (
                samples_for_status_quo_action * (1 - self.ts_asc_alpha)
            )
        # apply max action constraints
        if max_actions is not None:
            max_action_feas = torch.ones(
                (X.shape[0], self.data_config.num_actions), dtype=torch.bool
            )
            for i in range(self.data_config.num_actions):
                max_action_feas[:, i] = max_actions >= i
            feasible_mask *= max_action_feas
        return samples, feasible_mask


class NeuralBanditMultinomialImitator(NeuralBanditModel):
    """Implements a neural network that outputs the parameters of a multinomial."""

    def __init__(
        self,
        network: FCClassifier,
        data_config: DatasetParams,
        hparams: Hyperparameters,
        strategy: ActionSelectionStrategy = ActionSelectionStrategy.GREEDY,
    ) -> None:
        super().__init__(
            network=network, data_config=data_config, hparams=hparams, strategy=strategy
        )
        self.loss_module = KLDivLoss(reduction="none")

    @mini_batch_evaluate
    def predict(self, X: Tensor, n_samples: int = 1, **kwargs) -> Tensor:
        """
        Get the probabilities of each action.
        Args:
            X: `n x d` tensor of contexts
            n_samples: number of samples
        Returns:
            Tensor: `(n_samples) x n x num_actions` tensor with the model predicted
             value for each action
        Raises:
            ValueError: If n_samples != 1.
        """
        if n_samples != 1:
            raise ValueError(
                "NeuralNetworkModel `predict` only supports n_samples = 1."
            )
        with torch.no_grad():
            return torch.exp(self.network(X.to(device=self.data_config.device))).cpu()

    def fit(
        self,
        X: Tensor,
        targets: Tensor,
        actions: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        force_update: bool = False,
    ) -> None:
        """Trains the network for num_steps, using the provided data.

        Args:
            X: `n x d` Tensor of contexts
            actions: `n x 1` tensor of actions
            targets: `n x num_actions` Tensor of targets
            weights: `n x 1` Tensor of example weights
            force_update: A boolean indicating whether to ignore the training_freq
                and update the model
        """
        if targets.shape[-1] != self.data_config.num_actions:
            raise ValueError(
                f"{type(self).__name__} requires multinomial targets of shape"
                f" `n x {self.data_config.num_actions}`"
            )
        self._fit(
            X=X,
            targets=targets,
            actions=actions,
            weights=weights,
            force_update=force_update,
        )

    def compute_loss(
        self,
        X: Tensor,
        targets: Tensor,
        actions: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
    ) -> float:
        if targets.shape[-1] != self.data_config.num_actions:
            raise ValueError(
                f"{type(self).__name__} requires multinomial targets of shape"
                f" `n x {self.data_config.num_actions}`"
            )
        return evaluate_loss(
            model=self.network,
            inputs=X,
            targets=targets,
            criterion=self.loss_module,
            weights=weights,
        )
