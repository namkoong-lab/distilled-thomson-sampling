#!/usr/bin/env python3

from enum import IntEnum

from ..core.hyperparameters import DatasetParams, Hyperparameters
from .bandit_model import BanditModel
from .base_network import FCClassifier, FCNetwork
from .neural_bandit_models import (
    NeuralBanditClassifier,
    NeuralBanditMultinomialImitator,
    NeuralBanditRegressor,
)
from .parametric_neural_linear_models import (
    ParametricNeuralLinearModel,
)


class ModelType(IntEnum):
    NN_REGRESSOR = 0  # Basic
    NN_CLASSIFIER = 1
    NN_MULTINOMIAL_IMITATOR = 2
    PARAMETRIC_NEURAL_LINEAR = 3


def get_model(
    model_type: ModelType, data_config: DatasetParams, hparams: Hyperparameters
) -> BanditModel:
    nn_hparams = hparams.nn_hparams
    if nn_hparams is None:
        raise ValueError("Hyperparameters.nn_hparams is required for neural models.")
    if model_type == ModelType.NN_REGRESSOR:
        return NeuralBanditRegressor(
            network=FCNetwork(
                data_config=data_config,
                hparams=nn_hparams,
                action_encoder_params=hparams.action_encoder_params,
            ),
            data_config=data_config,
            hparams=hparams,
        )
    elif model_type == ModelType.PARAMETRIC_NEURAL_LINEAR:
        return ParametricNeuralLinearModel(
            network=FCNetwork(
                data_config=data_config,
                hparams=nn_hparams,
                action_encoder_params=hparams.action_encoder_params,
            ),
            data_config=data_config,
            hparams=hparams,
        )
    elif model_type == ModelType.NN_CLASSIFIER:
        return NeuralBanditClassifier(
            network=FCClassifier(
                data_config=data_config,
                hparams=nn_hparams,
                action_encoder_params=hparams.action_encoder_params,
            ),
            data_config=data_config,
            hparams=hparams,
        )
    elif model_type == ModelType.NN_MULTINOMIAL_IMITATOR:
        return NeuralBanditMultinomialImitator(
            network=FCClassifier(data_config=data_config, hparams=nn_hparams),
            data_config=data_config,
            hparams=hparams,
        )
    else:
        raise ValueError("Invalid model type specified.")
