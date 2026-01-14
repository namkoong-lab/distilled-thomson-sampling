#!/usr/bin/env python3

from datetime import timedelta
from typing import List, Optional

import pandas as pd
import torch
from pandas import DataFrame
from torch import Tensor


def one_hot_encode_df(df: DataFrame, cols: List[str]) -> pd.DataFrame:
    """Returns one-hot encoding of DataFrame df including columns in cols."""
    for col in cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(col, axis=1)
    return df


def one_hot_encode_action(X: Tensor, num_actions: int) -> Tensor:
    """One hot encode integer actions.

    Args:
        X: `(batch_shape) x n x 1` tensor of integer actions
        num_actions: number of actions
    Returns:
        Tensor: a `(batch_shape) x n x num_actions` tensor of one-hot encoded actions
    """
    return torch.eye(num_actions).to(X)[X.squeeze(-1).long()]


def should_update(
    t: int, training_freq: int, num_actions: int, initial_pulls: Optional[int] = None
) -> bool:
    """Helper function that returns a boolean indicating whether the model should
        be updated.
    """
    if initial_pulls is not None:
        # update after initial pulls * num_actions
        if t == initial_pulls * num_actions:
            return True
        # only update after initial pulls * num_actions
        is_update_period = t >= initial_pulls * num_actions
    else:
        is_update_period = True
    is_update_period = t % training_freq == 0 and is_update_period and t > 0
    return is_update_period


def get_retention_period(days: int) -> timedelta:
    """
    Helper function returns the timedelta for the retention period. If days=0,
        then the minimum timedelta is returned. This is useful for testing
        fblearner workflows.
    """
    return timedelta(days=days) if days > 0 else timedelta(minutes=60)
