#!/usr/bin/env python3

from abc import ABC, abstractmethod
from typing import Optional

import torch
from .core.hyperparameters import DatasetParams
from .models.bandit_model import BanditModel
from .models.imitator_models import ImitatorModel
from torch import Tensor


class BaseBandit(ABC):
    """
    An abstract benchmark runner for contextual bandits.
        1. Choose an action given a context.
    """

    model: Optional[BanditModel] = None
    name: str

    def __init__(self, name: str, data_config: DatasetParams) -> None:
        self.name = name
        self.data_config = data_config

    def action(self, contexts: Tensor, t: Optional[int] = None) -> Tensor:
        """
        Get actions for contexts.
        Args:
            context: `n x d` tensor of contexts
        Returns:
            Tensor: `n`-dim tensor of actions
        """
        model = self.model
        if model is None:
            raise ValueError("Model must be initialized before requesting actions")
        if model.is_trained:
            return model.sample(contexts)
        elif (
            t is not None
            and contexts.shape[0] == 1
            and model.hparams.initial_pulls is not None
        ):
            # round robin
            return torch.tensor(
                [t % self.data_config.num_actions], dtype=self.data_config.dtype
            )
        return torch.randint(
            high=self.data_config.num_actions,
            size=(contexts.shape[0],),
            dtype=self.data_config.dtype,
        )

    def prop_scores(self, contexts: Tensor) -> Tensor:
        """Get propensity scores for actions, given a set of contexts.

        Args:
            context: `n x d` tensor of contexts
        Returns:
            Tensor: `n x n_actions` tensor of action propensities
        """
        model = self.model
        if model is None:
            raise ValueError("Model must be initialized before requesting actions")
        if model.is_trained:
            return model.prop_scores(contexts)
        # Uniform random if model is not trained.
        return (
            torch.ones(
                (contexts.shape[0], self.data_config.num_actions),
                dtype=self.data_config.dtype,
            )
            / self.data_config.num_actions
        )

    @abstractmethod
    def update(self, contexts: Tensor, actions: Tensor, rewards: Tensor) -> None:
        """
        Update model with provided data model.
        Args:
            contexts: `n x context_dim` tensor of contexts
            actions: `n x 1` tensor of actions
            rewards: `n x 1` tensor of rewards

        Returns:
            None
        """
        pass


class FixedPolicyBandit(BaseBandit):
    """A baseline runner returns an action at random with probs given by p."""

    def __init__(self, name: str, bandit_model: BanditModel) -> None:
        self.name = name
        self.model: BanditModel = bandit_model

    def update(self, contexts: Tensor, actions: Tensor, rewards: Tensor) -> None:
        """Fixed policy bandit ignores updates.

        Args:
            contexts: `n x context_dim` tensor of contexts
            actions: `n x 1` tensor of actions
            rewards: `n x 1` tensor of rewards

        Returns:
            None
        """
        pass


class Bandit(BaseBandit):
    def __init__(
        self, name: str, data_config: DatasetParams, bandit_model: BanditModel
    ) -> None:
        super().__init__(name=name, data_config=data_config)
        self.model: BanditModel = bandit_model

    def update(self, contexts: Tensor, actions: Tensor, rewards: Tensor) -> None:
        """
        Store new observations and update model.
        Args:
            contexts: `n x context_dim` tensor of contexts
            actions: `n x 1` tensor of actions
            rewards: `n x 1` tensor of rewards

        Returns:
            None
        """
        # Update model
        self.model.fit(X=contexts, actions=actions, targets=rewards)


class ImitationBandit(BaseBandit):
    """A policy defined by two models: a base and an imitator.

    The base model is trained, using both actions and rewards, to learn a stochastic
    policy--the model predicts a distribution over actions.

    The imitator, on the other hand, is simply trained using actions chosen via
    sampling from the distribution returned by the base model.
    E.g. X = Contexts, Y = Sample_Action(Base_Model(X)).

    """

    def __init__(
        self,
        name: str,
        data_config: DatasetParams,
        base_model: BanditModel,
        imitator_model: ImitatorModel,
    ) -> None:
        """Creates an ImitatorPolicy object.

        Args:
            name: Name of the algorithm.
            base_model: Instance of the base model.
            imitator_model: Instance of the imitator model.
        """
        super().__init__(name=name, data_config=data_config)
        self.base_model = base_model
        self.model: ImitatorModel = imitator_model

    def update(self, contexts: Tensor, actions: Tensor, rewards: Tensor) -> None:
        """
        Store new observations and update model.
        Args:
            contexts: `n x context_dim` tensor of contexts
            actions: `n x 1` tensor of actions
            rewards: `n x 1` tensor of rewards
        Returns:
            None
        """
        # Update base model
        self.base_model.fit(X=contexts, actions=actions, targets=rewards)
        # Update imitator
        self.model.imitate(X=contexts, target_model=self.base_model)
