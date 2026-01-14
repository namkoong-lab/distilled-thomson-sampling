#!/usr/bin/env python3

from typing import List

from .modules import Transform
from torch import Tensor


def apply_transforms(
    X: Tensor, transforms: List[Transform], reverse: bool = False
) -> Tensor:
    """
    Apply a list of transforms to X in order. If reverse is True, untransforms
        are applied in reverse order.

    Args:
        X: data
        transforms: list of Transforms
        reverse: bool indicating whether to transform or untransform
    Return:
        Tensor: transformed data
    """

    if reverse:
        for t in reversed(transforms):
            X = t(X=X, reverse=True)
    else:
        for t in transforms:
            X = t(X=X)
    return X
