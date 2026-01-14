#!/usr/bin/env python3

from typing import Callable, Optional

import torch
from ..core.hyperparameters import (
    BootstrapHyperparams,
    DatasetParams,
    Hyperparameters,
)
from .action_selection import (
    ActionSelectionStrategy,
    TActionCallable,
    action_strategy_factory,
)
from .bandit_model import BanditModel
from .common import PROP_SCORE_SAMPLES
from .model_config import ModelType, get_model
from ..utils import should_update
from torch import Tensor
from torch.distributions import Bernoulli


# pyre-fixme [13]: inherited attributes are initialized
class BootstrapTSModel(BanditModel):
    """Bootstrapped Thompson Sampling Bandit Model.

    This work implements the online bootstrap of:

    Eckles, D., & Kaptein, M. (2014). Thompson sampling with the
    online bootstrap (arXiv:1410.4009). Retrieved from https://
    arxiv.org/abs/1410.4009

    When a new observation is received, the new observation will be
    used by each bootstrap replicate model with probability `p`.

    The underlying model can be any BanditModel.
    """

    def __init__(
        self,
        base_model_type: ModelType,
        hparams: Hyperparameters,
        data_config: DatasetParams,
        bootstrap_hparams: BootstrapHyperparams,
    ) -> None:
        """Initialize the Bootstrap TS policy.

        Args:
            base_model_type: The ModelType of each underlying bootstrap replicate model
            hparams: The hyperparameters that should be passed to each bootstrap
                replicate model.
            data_config: Data specifications
            bootstrap_hparams: The BootstrapHyperparams for the bootstrap policy.
        """
        super().__init__(
            data_config=data_config,
            hparams=hparams,
            action_encoder_params=hparams.action_encoder_params,
        )
        self.bootstrap_hparams = bootstrap_hparams
        self.models = []
        self.data_indices = []
        for _ in range(bootstrap_hparams.n_replicates):
            self.models.append(
                get_model(
                    model_type=base_model_type, hparams=hparams, data_config=data_config
                )
            )
            self.data_indices.append(torch.empty((0,), dtype=torch.long))
        self.n_obs = 0
        self.bernoulli = Bernoulli(probs=self.bootstrap_hparams.p)
        self.strategy: ActionSelectionStrategy = ActionSelectionStrategy.PROBABILISTIC
        action_selection_methods = action_strategy_factory(strategy=self.strategy)
        self.select_action: TActionCallable = action_selection_methods[0]
        self.strategy_prop_scores: Callable[
            [Tensor], Tensor
        ] = action_selection_methods[1]

    def fit(
        self,
        X: Tensor,
        targets: Tensor,
        actions: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        force_update: bool = False,
    ) -> None:
        """Trains the model using the provided data.

        New observations are added to each bootstrap's dataset with probability
        `p` (set in the BootstrapHyperparams). All bootstrap models are updated
        according to the BootstrapTSModel's  training_freq.

        Args:
            X: `n x d` tensor of contexts
            targets: `n x 1` tensor of rewards
            actions: `n x 1` tensor of actions
            weights: `n x 1` Tensor of example weights
            force_update: A boolean indicating whether to ignore the training_freq
                and update the model
        """
        num_new_obs = X.shape[0] - self.n_obs
        # for new observations, sample bernoulli mask indicating
        # whether the observation should be included in each replicate's
        # training set
        new_mask = self.bernoulli.sample(
            torch.Size([self.bootstrap_hparams.n_replicates, num_new_obs])
        )
        for i in range(self.bootstrap_hparams.n_replicates):
            new_indices = new_mask[i].nonzero().view(-1) + self.n_obs
            self.data_indices[i] = torch.cat([self.data_indices[i], new_indices], dim=0)
        self.n_obs += num_new_obs
        nn_hparams = self.hparams.nn_hparams
        if (
            should_update(
                t=X.shape[0],
                training_freq=nn_hparams.training_freq,
                num_actions=self.data_config.num_actions,
                initial_pulls=self.hparams.initial_pulls,
            )
            or force_update
        ):
            self.is_trained = True
            for i in range(self.bootstrap_hparams.n_replicates):
                data_indices = self.data_indices[i]
                self.models[i].fit(
                    X=X[data_indices],
                    targets=targets[data_indices],
                    actions=actions[data_indices] if actions is not None else None,
                    weights=weights[data_indices] if weights is not None else None,
                    force_update=True,
                )

    def predict(self, X: Tensor, n_samples: int = 1, **kwargs) -> Tensor:
        """Get the predicted reward for each action.

        For each of `n_samples` samples, a model is sampled from the set of bootstraps
        and used for prediction.

        Args:
            X: `n x d` tensor of contexts
            n_samples: number of samples
        Returns:
            Tensor: `(n_samples) x n x num_actions` tensor with the model predicted
             value for each action
        """
        model_indices = torch.randint(
            0, self.bootstrap_hparams.n_replicates, (n_samples,)
        )
        preds = torch.stack(
            [self.models[i].predict(X=X) for i in model_indices.tolist()], dim=0
        )
        if n_samples == 1:
            preds = preds.squeeze(0)
        return preds

    def prop_scores(self, X: Tensor, n_samples: int = PROP_SCORE_SAMPLES) -> Tensor:
        """Get the predicted probability of taking each action.

        Args:
            X: `n x d` tensor of contexts
            n_samples: number of MC samples to use to estimate the prop scores
        Returns:
            Tensor: `n x num_actions` tensor with the modeled
             probability for each action
        """
        action_samples = self.sample(X=X, n_samples=n_samples).long()
        counts = torch.eye(
            self.data_config.num_actions, dtype=X.dtype, device=X.device
        )[action_samples].sum(dim=0)
        return self.strategy_prop_scores(counts.float())

    def sample(self, X: Tensor, n_samples: int = 1, **kwargs) -> Tensor:
        """Draw one action per context from the predicted reward for each action.

        For each of `n_samples` samples, a model is sampled from the set of bootstraps
        and the best action under that model is returned.

        Args:
            X: `n x d` tensor of contexts
            n_samples: number of samples
        Returns:
            Tensor: `(n_samples) x n ` tensor with action for each context
        """
        preds = self.predict(X=X, n_samples=n_samples)
        return preds.argmax(dim=-1).to(X)  # pyre-ignore [16]

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
            targets: `n x 1` tensor of targets
            actions: `n x 1` tensor of actions
            weights: `n x 1` tensor of weights
        Returns:
            float: loss
        """
        raise NotImplementedError
