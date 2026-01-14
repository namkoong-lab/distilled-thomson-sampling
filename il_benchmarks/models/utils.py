#!/usr/bin/env python3

from functools import wraps
from typing import Any, Callable

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


PREDICT_BATCH_SIZE = 1024


def uniform_weight_initializer(module: nn.Module, lower: float, upper: float) -> None:
    """
    Initialize weights from Unif(lower, upper). Initialize biases as zero.
    Initialization is performed in-place.
    """
    if isinstance(module, nn.Linear):
        nn.init.uniform_(module.weight, lower, upper)
        bias = module.bias
        if bias is not None:
            nn.init.zeros_(bias)


def calibrate_at_p(p: float, p_samples: Tensor, labels: Tensor) -> float:
    """
    Find and return threshold such that `p_samples` >= `threshold` corresponds to
        positive labels proportion `p` of the time.

    Args:
        p: desired threshold probability
        p_samples: a `n_samples x n` tensor of sampled probabilities
        labels: a `n x 1` tensor of labels in [0, 1]
    Returns:
        float: the threshold
    """
    n_thresholds = 101
    n_samples = p_samples.shape[0]
    n = p_samples.shape[1]
    # initialize range of threshold
    base_t = torch.linspace(
        0, 1, n_thresholds, dtype=p_samples.dtype, device=p_samples.device
    )
    t = base_t.repeat(n_samples, n, 1)
    # compute positive samples under t
    positive_under_t = p_samples.unsqueeze(-1).expand(-1, -1, n_thresholds) >= t
    # gather negative observed labels
    observed_negative = (
        (1 - labels.view(-1))
        .repeat(n_samples, 1)
        .unsqueeze(-1)
        .expand(-1, -1, n_thresholds)
    )
    # gather thresholds s.t. P(label = 1 | p_sample >= t) >= p
    acceptable_threshold_mask = (
        1.0 - (positive_under_t & observed_negative)  # pyre-ignore [6]
    ).float().mean(dim=0).mean(dim=0) >= p
    # select minimum acceptable threshold
    min_t_idx = acceptable_threshold_mask.nonzero().min()  # pyre-ignore [16]
    return base_t[min_t_idx].item()


def mini_batch_evaluate(
    method: Callable[[Any, Tensor, int, Any], Tensor]
) -> Callable[[Any, Tensor, int, Any], Tensor]:
    """Decorator for evaluating a function in mini-batches.

    This decorator works on methods like `BanditModel.predict`, which taking a tensor
    `X` as the argument, an integer `n_samples` and optionally keyword arguments.
    The decorator splits `X` into minibatches of the specified size and makes
    predictions for each mini-batch.

    Args:
        method: the `predict` method to be decorated

    Returns:
        decorated method
    """

    @wraps(method)
    def decorated(cls: Any, X: Tensor, **kwargs: Any) -> Tensor:
        if X.shape[-2] <= PREDICT_BATCH_SIZE:
            return method(cls, X, **kwargs)
        kwargs = kwargs or {}
        tensor_kwarg_names = []
        tensor_kwarg_vals = []
        for k, v in kwargs.items():
            if torch.is_tensor(v):
                tensor_kwarg_names.append(k)
                tensor_kwarg_vals.append(v)
        non_tensor_kwargs = {k: v for k, v in kwargs.items() if not torch.is_tensor(v)}
        data_loader = DataLoader(
            TensorDataset(X, *tensor_kwarg_vals), batch_size=PREDICT_BATCH_SIZE
        )
        preds = []
        for tensor_batch in data_loader:
            X_batch = tensor_batch[0]
            tensor_batch_kwargs = dict(zip(tensor_kwarg_names, tensor_batch[1:]))
            pred_batch = method(
                cls, X_batch, **tensor_batch_kwargs, **non_tensor_kwargs
            )
            preds.append(pred_batch)
        return torch.cat(preds, dim=-2)

    return decorated  # pyre-ignore [7]
