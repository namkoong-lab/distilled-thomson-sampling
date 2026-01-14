#!/usr/bin/env python3

"""
Utility functions for encoding actions as input.
"""

from .utils import one_hot_encode_action
from torch import Tensor


def one_hot_action_encoder(actions: Tensor, **kwargs) -> Tensor:
    """Takes discrete integer actions and returns one-hot encoded actions.

    Args:
        actions: A `(batch_shape) x n x 1`-dim tensor of one-hot actions
    Returns:
        Tensor: A `(batch_shape) x n x num_actions`-dim tensor of one-hot actions
    """
    if "num_actions" not in kwargs:
        raise ValueError("num_actions is required for one_hot_action_encoder.")
    return one_hot_encode_action(actions, num_actions=kwargs["num_actions"])


def identity_action_encoder(actions: Tensor, **kwargs) -> Tensor:
    """Takes actions and returns them as is. (identity function)
    Args:
        actions: A `(batch_shape) x n x 1`-dim tensor of ordinal actions
    Returns:
        Tensor: A `(batch_shape) x n x 1` tensor ordinal actions
    """
    return actions
