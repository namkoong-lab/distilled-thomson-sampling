#!/usr/bin/env python3

from typing import Any, Callable, Dict, List, NamedTuple, Optional, Union

import torch
from ..action_encoders import one_hot_action_encoder
from ..core.setup import DEVICE, DTYPE
from ..models.common import PROP_SCORE_SAMPLES
from ..reward_functions import RewardFunctionType
from ..transforms.modules import Transform
from torch import Tensor, nn
from torch.nn.modules.activation import Tanh
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer
from torch.optim.rmsprop import RMSprop


TRewardParameters = Union[List[float], List[List[float]]]


class DatasetParams(NamedTuple):
    """
    Container for storing attributes of dataset necessary for models/policies.
    """

    context_dim: int
    num_actions: int
    num_outcomes: int = 1
    device: torch.device = DEVICE
    dtype: torch.dtype = DTYPE


class ActionEncoderParams(NamedTuple):
    """
    Container for storing configuration for embedding action with input.
    """

    encoded_dim: int
    action_encoder: Callable[[Tensor, Any], Tensor] = one_hot_action_encoder
    kwargs: Dict[str, Any] = {}


class NNHyperparameters(NamedTuple):  # pyre-ignore [9]
    """
    Container for storing neural network hyperparameters.
    """

    hidden_layer_sizes: List[int] = [50]
    initial_lr: float = 0.1
    reset_lr: bool = True
    batch_size: int = 512
    init_scale: float = 0.3
    activation_function: Callable[[], nn.Module] = Tanh
    optimizer_cls: Callable[..., Optimizer] = RMSprop
    layer_norm: bool = False
    lr_decay_rate: float = 0.5
    max_grad_norm: Optional[float] = 5.0
    training_freq: int = 500
    num_epochs: int = 1
    num_mini_batches: Optional[int] = None
    weight_decay: float = 0.0
    scheduler_step_interval: Optional[int] = None
    shuffle: bool = True


class ParametricBLRHyperparameters(NamedTuple):
    a0: Optional[float] = None
    b0: float = 0.0
    lambda_prior: float = 1.0
    training_freq: int = 500
    blr_target_transforms: Optional[List[Transform]] = None
    online_learning: bool = False


class Hyperparameters(NamedTuple):
    """
    Container for storing Hyperparameters.
    """

    nn_hparams: Optional[NNHyperparameters] = None
    verbose: bool = False
    initial_pulls: Optional[int] = None
    bayesian_hparams: Optional[
        Union[
            ParametricBLRHyperparameters,
        ]
    ] = None
    action_encoder_params: Optional[ActionEncoderParams] = None
    ts_asc_alpha: Optional[float] = None
    reward_function_type: RewardFunctionType = RewardFunctionType.LINEAR
    # reward parameters can either be a nested list of num_actions x num_outcomes
    # or a list of num_outcomes, where the same params are used for all actions
    reward_parameters: Optional[TRewardParameters] = None


class ImitatorHyperparams(NamedTuple):
    """
    Container for storing Imitator Hyperparams.
    """

    num_mc_samples: int = PROP_SCORE_SAMPLES
    initial_pulls: Optional[int] = None


class BootstrapHyperparams(NamedTuple):
    """
    Container for storing Imitator Hyperparams.
    """

    n_replicates: int = 10
    p: float = 1.0
