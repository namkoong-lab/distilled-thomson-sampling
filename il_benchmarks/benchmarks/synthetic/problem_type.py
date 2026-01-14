#!/usr/bin/env python3

from enum import IntEnum
from typing import Optional

from .mushroom import Mushroom
from .problem import BenchmarkProblem
from .warfarin import Warfarin
from .wheel import Wheel


class BenchmarkProblemType(IntEnum):
    """Enum for representing different benchmark problems"""

    MUSHROOM = 0
    WHEEL = 1
    WARFARIN = 2


def get_problem_spec(
    problem_type: BenchmarkProblemType,
    n_examples: int,
    seed: Optional[int] = None,
    num_actions: Optional[int] = None,
) -> BenchmarkProblem:
    """
    Args:
        problem_type: the problem type
        n_examples: number of examples (time steps)
        seed: the random seed to the benchmark problem
        num_actions: number of actions
    Returns:
        BenchmarkProblem: the problem instance
    """
    if problem_type == BenchmarkProblemType.MUSHROOM:
        return Mushroom(n_examples=n_examples, seed=seed)
    elif problem_type == BenchmarkProblemType.WHEEL:
        return Wheel(n_examples=n_examples, seed=seed)
    elif problem_type == BenchmarkProblemType.WARFARIN:
        kwargs = {"num_actions": num_actions} if num_actions is not None else {}
        return Warfarin(n_examples=n_examples, seed=seed, **kwargs)
    else:
        raise ValueError("Invalid problem_type specified.")
