#!/usr/bin/env python3

from typing import Optional

import torch
from torch import Tensor, nn

from ..core.hyperparameters import ActionEncoderParams, DatasetParams, NNHyperparameters


class FCNetwork(nn.Module):
    """
    Base fully-connected neural network.
    """

    def __init__(
        self,
        data_config: DatasetParams,
        hparams: NNHyperparameters,
        action_encoder_params: Optional[ActionEncoderParams] = None,
    ) -> None:
        super().__init__()
        hidden_layer_sizes = hparams.hidden_layer_sizes
        activation_function = hparams.activation_function
        input_dim = data_config.context_dim
        if action_encoder_params is not None:
            input_dim += action_encoder_params.encoded_dim
            self.output_dim = 1
        else:
            self.output_dim: int = data_config.num_actions
        layer_sizes = zip([input_dim] + hidden_layer_sizes[:-1], hidden_layer_sizes)

        # features contains the network up to the last activation layer
        self.features = nn.Sequential()
        layer_idx = 1
        for layer_size in layer_sizes:
            # Linear
            self.features.add_module(
                f"linear{layer_idx}", nn.Linear(layer_size[0], layer_size[1])
            )
            # Activation
            self.features.add_module(f"activation{layer_idx}", activation_function())

            if hparams.layer_norm:
                # Normalize
                self.features.add_module(
                    f"norm{layer_idx}", nn.LayerNorm(layer_size[1])
                )

            layer_idx += 1
        # Linear layer final activation to num_actions
        self.linear_output = nn.Linear(hidden_layer_sizes[-1], self.output_dim)
        self.to(dtype=data_config.dtype, device=data_config.device)  # pyre-ignore [28]

    def forward(self, X: Tensor) -> Tensor:
        features = self.features(X)
        return self.linear_output(features)

    def forward_latent_features(self, X: Tensor) -> Tensor:
        """
        Forward pass returning features for last latent layer before output layer.
        """
        return self.features(X)


class FCClassifier(FCNetwork):
    def forward(self, X: Tensor) -> Tensor:
        features = self.features(X)
        logits = self.linear_output(features)
        if self.output_dim == 1:
            return torch.sigmoid(logits)
        # distribution imitator requires log softmax
        return nn.functional.log_softmax(logits, dim=-1)
