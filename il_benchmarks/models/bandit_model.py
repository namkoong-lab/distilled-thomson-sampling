#!/usr/bin/env python3

from abc import ABC
from typing import Optional

import torch
from ..core.hyperparameters import (
    ActionEncoderParams,
    DatasetParams,
    Hyperparameters,
)
from .action_selection import ActionSelectionStrategy
from .common import PROP_SCORE_SAMPLES
from ..reward_functions import (
    RewardFunction,
    RewardFunctionType,
    get_reward_function,
)
from torch import Tensor


class BanditModel(torch.nn.Module, ABC):
    """Abstract class for Contextual Bandit Models."""

    data_config: DatasetParams
    hparams: Hyperparameters
    is_trained: bool
    strategy: ActionSelectionStrategy
    reward_function: RewardFunction
    _reward_parameters: Tensor

    def __init__(
        self,
        data_config: DatasetParams,
        hparams: Hyperparameters,
        action_encoder_params: Optional[ActionEncoderParams] = None,
    ) -> None:
        super().__init__()
        self.data_config = data_config
        self.hparams = hparams
        self.is_trained = False
        self.action_encoder_params = action_encoder_params
        if action_encoder_params is not None:
            self.ordinal_action_space = torch.arange(
                data_config.num_actions, dtype=data_config.dtype
            ).view(-1, 1)
        reward_parameters = hparams.reward_parameters
        self.reward_function_type = hparams.reward_function_type
        if (
            self.reward_function_type is not None
            and not isinstance(self.reward_function_type, RewardFunctionType)
            and isinstance(self.reward_function_type, str)
        ):
            self.reward_function_type = RewardFunctionType(self.reward_function_type)
        if reward_parameters is None:
            if hparams.reward_function_type == RewardFunctionType.LINEAR:
                reward_parameters = torch.ones(
                    (data_config.num_actions, data_config.num_outcomes),
                    dtype=data_config.dtype,
                )
            else:
                raise ValueError(
                    "reward_parameters are required for reward_function_type: "
                    f"{self.reward_function_type.name}."
                )
        self.reward_parameters = torch.tensor(
            reward_parameters, dtype=data_config.dtype
        )

    @property
    def reward_parameters(self) -> Tensor:
        return self._reward_parameters

    @reward_parameters.setter
    def reward_parameters(self, reward_parameters: Tensor) -> None:
        r"""Set reward parameters and update RewardFunction.

        The purpose of this setter is to update the RewardFunction,
        whenever the bandit model reward parameters change.
        """
        self._reward_parameters = reward_parameters
        self.reward_function = get_reward_function(
            reward_function_type=self.reward_function_type,
            reward_parameters=self.reward_parameters,
            num_actions=self.data_config.num_actions,
            num_outcomes=self.data_config.num_outcomes,
        )

    def fit(
        self,
        X: Tensor,
        targets: Tensor,
        actions: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        force_update: bool = False,
    ) -> None:
        """Trains the model using the provided data.

        Args:
            X: `n x d` tensor of contexts
            targets: `n x 1` tensor of rewards
            actions: `n x 1` tensor of actions
            weights: `n x 1` Tensor of example weights
            force_update: A boolean indicating whether to ignore the training_freq
                and update the model
        """
        pass

    def predict(self, X: Tensor, n_samples: int = 1, **kwargs) -> Tensor:
        """Get the predicted reward for each action.

        Args:
            X: `n x d` tensor of contexts
            n_samples: number of samples
        Returns:
            Tensor: `(n_samples) x n x num_actions` tensor with the model predicted
             value for each action
        """
        pass

    def prop_scores(self, X: Tensor, n_samples: int = PROP_SCORE_SAMPLES) -> Tensor:
        """Get the predicted probability of taking each action.

        Args:
            X: `n x d` tensor of contexts
            n_samples: number of MC samples to use to estimate the prop scores
        Returns:
            Tensor: `n x num_actions` tensor with the modeled
             probability for each action
        """
        pass

    def _get_inputs(self, X: Tensor, actions: Optional[Tensor] = None) -> Tensor:
        """
        Get the inputs. If actions are to be embedded with inputs:
            - if actions are provided, encode and embed with input
            - if actions are not provided, encode all actions and embed with input
                by adding an extra batch dimension.
        If actions are not to be embedded with the inputs, this just returns X
        Args:
            X: `n x d` tensor of contexts
            actions: `n x 1` tensor of actions
        Returns:
            Tensor: `n x d` tensor (X) if actions are not encoded, otherwise a
                `n x num_actions x d + encoded_dim` tensor of augmented inputs.
        """
        action_encoder_params = self.action_encoder_params
        if action_encoder_params is None:
            return X
        elif actions is None:
            actions = self.ordinal_action_space.unsqueeze(0).expand(
                X.shape[0], self.data_config.num_actions, 1
            )
            X = X.unsqueeze(1).expand(
                X.shape[0], self.ordinal_action_space.shape[0], X.shape[1]
            )
        encoded_actions = action_encoder_params.action_encoder(
            actions, **action_encoder_params.kwargs
        )
        return torch.cat([X, encoded_actions], dim=-1)

    def sample(self, X: Tensor, n_samples: int = 1, **kwargs) -> Tensor:
        """Draw one action per context from the predicted reward for each action.

        Args:
            X: `n x d` tensor of contexts
            n_samples: number of samples
        Returns:
            Tensor: `(n_samples) x n ` tensor with action for each context
        """
        pass

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
        pass
