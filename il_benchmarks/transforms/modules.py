#!/usr/bin/env python3

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor, nn


class Transform(nn.Module, ABC):
    """Base interface for tensor transforms"""

    def __init__(self, ndim: Optional[int] = None) -> None:
        super().__init__()

    def forward(self, X: Tensor, reverse: bool = False) -> Tensor:
        """
        Transform each column of X if reverse is False, otherwise untransform X.

        Args:
            X: A `... x n x d` tensor of data
            reverse: A bool indicating whether to transform (True) or untransform
                (False)
        Returns:
            Tensor: transformed (or untransformed) X
        """

        if reverse:
            return self._untransform(X)
        return self._transform(X)

    @abstractmethod
    def _transform(self, X: Tensor) -> Tensor:
        ...

    @abstractmethod
    def _untransform(self, X: Tensor) -> Tensor:
        ...


class SqrtTransform(Transform):
    """Square root transform"""

    def _transform(self, X: Tensor) -> Tensor:
        return X.sqrt()

    def _untransform(self, X: Tensor) -> Tensor:
        return X ** 2


class StandardizeTransform(Transform):
    def __init__(self, ndim: Optional[int] = None) -> None:
        if ndim is None:
            raise ValueError("ndim is required for StandardizeTransform")
        super().__init__()
        self.register_buffer("mean_x", torch.empty(ndim))
        self.register_buffer("sd_x", torch.empty(ndim))
        self.register_buffer("has_params", torch.zeros(1))

    def _transform(self, X: Tensor) -> Tensor:
        """
        Transform each column of X to N(0, 1).

        Args:
            X: A `n x d` tensor of data
        Returns:
            Tensor: standardized X
        """
        # pyre-ignore [16]
        if not self.has_params:
            self.mean_x = X.mean(dim=-2)  # pyre-ignore [8]
            self.sd_x = torch.max(  # pyre-ignore [8]
                X.std(dim=-2), torch.tensor(1e-8, dtype=X.dtype, device=X.device)
            )
            self.has_params[0] = 1
        return (X - self.mean_x) / self.sd_x

    def _untransform(self, X: Tensor) -> Tensor:
        """
        Unstandarize each column of X from N(0, 1) using the provided means and standard
            deviations.

        Args:
            X: A `n x d` tensor of data
            params: mean and sd of each column of X
        Returns:
            Tensor: unstandardized X
        """
        # pyre-ignore [16]
        if not self.has_params:
            raise RuntimeError("You must standardize before unstandardizing.")
        # pyre-ignore [16]
        return X * self.sd_x + self.mean_x


class MinMaxTransform(Transform):
    def __init__(self, ndim: Optional[int] = None) -> None:
        if ndim is None:
            raise ValueError("ndim is required for MinMaxTransform")
        super().__init__()
        self.register_buffer("min_x", torch.empty(ndim))
        self.register_buffer("range_x", torch.empty(ndim))
        self.register_buffer("has_params", torch.zeros(1))

    def _transform(self, X: Tensor) -> Tensor:
        """
        Transform each column of X to [0, 1].

        Args:
            X: A `n x d` tensor of data
        Returns:
            Tensor: scaled X
        """
        # pyre-ignore [16]
        if not self.has_params:
            # pyre-ignore [16]
            self.min_x = X.min(dim=-2)[0]
            max = X.max(dim=-2)[0]
            # pyre-ignore [16]
            self.range_x = max - self.min_x
            self.range_x[self.range_x < 1e-8] = 1
            self.has_params[0] = 1
        return (X - self.min_x) / self.range_x

    def _untransform(self, X: Tensor) -> Tensor:
        """
        Unscale the columns of X from [0, 1] back to the original values.

        Args:
            X: A `n x d` tensor of data
        Returns:
            Tensor: unscaled X
        """
        # pyre-ignore [16]
        if not self.has_params:
            raise RuntimeError("You must transform before untransforming.")
        # pyre-ignore [16]
        return X * self.range_x + self.min_x
