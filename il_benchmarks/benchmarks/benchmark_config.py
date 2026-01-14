#!/usr/bin/env python3

from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import torch
from ..bandits import (
    Bandit,
    BaseBandit,
    FixedPolicyBandit,
    ImitationBandit,
)
from .synthetic.problem import BenchmarkProblem
from ..core.hyperparameters import (
    BootstrapHyperparams,
    Hyperparameters,
    ImitatorHyperparams,
    NNHyperparameters,
    ParametricBLRHyperparameters,
)
from ..models.base_network import FCNetwork
from ..models.bootstrapped_model import BootstrapTSModel
from ..models.common import PROP_SCORE_SAMPLES
from ..models.fixed_model import FixedModel
from ..models.imitator_models import NNMultinomialImitator
from ..models.model_config import ModelType
from ..models.neural_bandit_models import NeuralBanditRegressor
from ..models.parametric_linear_model import ParametricLinearModel
from ..models.parametric_neural_linear_models import (
    ParametricNeuralLinearModel,
)


class AlgoCollectionType(IntEnum):
    CUSTOM = 0
    REGRESSION = 1
    REGRESSION_IMITATION = 2
    ALL = 3


class AlgoType(IntEnum):
    UNIFORM_RANDOM = 0
    NEURAL_GREEDY = 1
    LINEAR_TS = 2
    NEURAL_LINEAR_TS = 3
    LINEAR_TS_IL = 4
    NEURAL_LINEAR_TS_IL = 5
    BOOTSTRAP_NN_TS = 6
    BOOTSTRAP_NN_TS_IL = 7


ALGO_COLLECTIONS: Dict[AlgoCollectionType, List[AlgoType]] = {
    AlgoCollectionType.REGRESSION: [
        AlgoType.UNIFORM_RANDOM,
        AlgoType.NEURAL_GREEDY,
        AlgoType.LINEAR_TS,
        AlgoType.NEURAL_LINEAR_TS,
        AlgoType.BOOTSTRAP_NN_TS,
    ]
}

ALGO_COLLECTIONS[AlgoCollectionType.REGRESSION_IMITATION] = [
    AlgoType.UNIFORM_RANDOM,
    AlgoType.LINEAR_TS_IL,
    AlgoType.NEURAL_LINEAR_TS_IL,
    AlgoType.BOOTSTRAP_NN_TS_IL,
]

ALGO_COLLECTIONS[AlgoCollectionType.ALL] = (
    ALGO_COLLECTIONS[AlgoCollectionType.REGRESSION]
    + ALGO_COLLECTIONS[AlgoCollectionType.REGRESSION_IMITATION][1:]
)


def get_algo(
    algo_type: AlgoType,
    problem_spec: BenchmarkProblem,
    prop_score_mc_samples: int = PROP_SCORE_SAMPLES,
    training_freq: int = 1000,
    imitator_num_mini_batches: int = 500,
    initial_pulls: Optional[int] = None,
    n_bootstrap_replicates: int = 10,
) -> BaseBandit:
    data_config = problem_spec.get_data_config()
    tkwargs = {"dtype": data_config.dtype, "device": data_config.device}
    # use RMS2 hparams from Deep Bayesian Bandits Paper
    nn_hparams = NNHyperparameters(
        hidden_layer_sizes=[100, 100],
        training_freq=training_freq,
        num_mini_batches=100,
        initial_lr=0.01,
        shuffle=True,
        lr_decay_rate=0.55,
        scheduler_step_interval=1,
        activation_function=torch.nn.ReLU,
    )
    imitator_hparams = ImitatorHyperparams(
        num_mc_samples=prop_score_mc_samples, initial_pulls=initial_pulls
    )
    nn_imitator_hparams = NNHyperparameters(
        num_mini_batches=imitator_num_mini_batches,
        lr_decay_rate=0.05,
        scheduler_step_interval=100,
        hidden_layer_sizes=[100, 100],
        training_freq=training_freq,
        initial_lr=0.001,
        shuffle=True,
    )
    parametric_linear_hparams = Hyperparameters(
        verbose=False,
        bayesian_hparams=ParametricBLRHyperparameters(
            training_freq=training_freq,
            a0=6.0,
            b0=6.0,
            lambda_prior=0.25,
            online_learning=True,
        ),
        initial_pulls=initial_pulls,
    )
    parametric_nl_hparams = Hyperparameters(
        nn_hparams=nn_hparams,
        verbose=False,
        bayesian_hparams=ParametricBLRHyperparameters(
            training_freq=training_freq, a0=3.0, b0=3.0, lambda_prior=0.25
        ),
        initial_pulls=initial_pulls,
    )
    parametric_linear_imitation_hparams = Hyperparameters(
        verbose=False,
        bayesian_hparams=ParametricBLRHyperparameters(
            training_freq=nn_imitator_hparams.training_freq,
            a0=6.0,
            b0=6.0,
            lambda_prior=0.25,
            online_learning=True,
        ),
        initial_pulls=initial_pulls,
    )
    parametric_nl_imitation_hparams = Hyperparameters(
        nn_hparams=nn_hparams,
        verbose=False,
        bayesian_hparams=ParametricBLRHyperparameters(
            training_freq=nn_imitator_hparams.training_freq,
            a0=3.0,
            b0=3.0,
            lambda_prior=0.25,
        ),
        initial_pulls=initial_pulls,
    )
    bootstrap_hparams = BootstrapHyperparams(n_replicates=n_bootstrap_replicates, p=1.0)
    if algo_type == AlgoType.UNIFORM_RANDOM:
        return FixedPolicyBandit(
            name="UniformRandom",
            bandit_model=FixedModel(
                p=torch.ones(data_config.num_actions, **tkwargs).div(
                    data_config.num_actions
                ),
                num_actions=data_config.num_actions,
            ),
        )
    elif algo_type == AlgoType.NEURAL_GREEDY:
        return Bandit(
            name="Neural-Greedy",
            data_config=data_config,
            bandit_model=NeuralBanditRegressor(
                network=FCNetwork(data_config=data_config, hparams=nn_hparams),
                data_config=data_config,
                hparams=Hyperparameters(
                    verbose=False, nn_hparams=nn_hparams, initial_pulls=initial_pulls
                ),
            ),
        )
    elif algo_type == AlgoType.LINEAR_TS:
        return Bandit(
            name="Linear-TS",
            data_config=data_config,
            bandit_model=ParametricLinearModel(
                data_config=data_config, hparams=parametric_linear_hparams
            ),
        )
    elif algo_type == AlgoType.NEURAL_LINEAR_TS:
        return Bandit(
            name="NeuralLinear-TS",
            data_config=data_config,
            bandit_model=ParametricNeuralLinearModel(
                network=FCNetwork(
                    data_config=data_config,
                    # pyre-ignore [6]
                    hparams=parametric_nl_hparams.nn_hparams,
                ),
                data_config=data_config,
                hparams=parametric_nl_hparams,
            ),
        )
    elif algo_type == AlgoType.BOOTSTRAP_NN_TS:
        return Bandit(
            name="Bootstrap-NN-TS",
            data_config=data_config,
            bandit_model=BootstrapTSModel(
                base_model_type=ModelType.NN_REGRESSOR,
                data_config=data_config,
                hparams=Hyperparameters(
                    verbose=False, nn_hparams=nn_hparams, initial_pulls=initial_pulls
                ),
                bootstrap_hparams=bootstrap_hparams,
            ),
        )
    elif algo_type == AlgoType.LINEAR_TS_IL:
        return ImitationBandit(
            name="Linear-TS-IL",
            data_config=data_config,
            base_model=ParametricLinearModel(
                data_config=data_config, hparams=parametric_linear_imitation_hparams
            ),
            imitator_model=NNMultinomialImitator(
                data_config=data_config,
                imitator_hparams=imitator_hparams,
                hparams=Hyperparameters(
                    nn_hparams=nn_imitator_hparams,
                    verbose=False,
                    initial_pulls=initial_pulls,
                ),
            ),
        )
    elif algo_type == AlgoType.BOOTSTRAP_NN_TS_IL:
        return ImitationBandit(
            name="Bootstrap-NN-TS-IL",
            data_config=data_config,
            base_model=BootstrapTSModel(
                base_model_type=ModelType.NN_REGRESSOR,
                data_config=data_config,
                hparams=Hyperparameters(
                    verbose=False, nn_hparams=nn_hparams, initial_pulls=initial_pulls
                ),
                bootstrap_hparams=bootstrap_hparams,
            ),
            imitator_model=NNMultinomialImitator(
                data_config=data_config,
                imitator_hparams=imitator_hparams,
                hparams=Hyperparameters(
                    nn_hparams=nn_imitator_hparams,
                    verbose=False,
                    initial_pulls=initial_pulls,
                ),
            ),
        )
    elif algo_type == AlgoType.NEURAL_LINEAR_TS_IL:
        return ImitationBandit(
            name="NeuralLinear-TS-IL",
            data_config=data_config,
            base_model=ParametricNeuralLinearModel(
                network=FCNetwork(
                    data_config=data_config,
                    # pyre-ignore [6]
                    hparams=parametric_nl_imitation_hparams.nn_hparams,
                ),
                data_config=data_config,
                hparams=parametric_nl_imitation_hparams,
            ),
            imitator_model=NNMultinomialImitator(
                data_config=data_config,
                imitator_hparams=imitator_hparams,
                hparams=Hyperparameters(
                    nn_hparams=nn_imitator_hparams,
                    verbose=False,
                    initial_pulls=initial_pulls,
                ),
            ),
        )
    else:
        raise ValueError(f"Unsupported algo_type: {algo_type.name}")


def get_algo_collection(
    algo_collection_type: AlgoCollectionType,
    problem_spec: BenchmarkProblem,
    prop_score_mc_samples: int = PROP_SCORE_SAMPLES,
    training_freq: int = 1000,
    imitator_num_mini_batches: int = 500,
    initial_pulls: Optional[int] = None,
) -> Tuple[List[BaseBandit], str]:
    algo_types = ALGO_COLLECTIONS[algo_collection_type]
    bandits = [
        get_algo(
            algo_type=algo_type,
            problem_spec=problem_spec,
            prop_score_mc_samples=prop_score_mc_samples,
            training_freq=training_freq,
            imitator_num_mini_batches=imitator_num_mini_batches,
            initial_pulls=initial_pulls,
        )
        for algo_type in algo_types
    ]
    return (bandits, algo_collection_type.name)
