#!/usr/bin/env python3

from il_benchmarks.benchmarks.benchmark_config import (
    AlgoCollectionType,
    get_algo_collection,
)
from il_benchmarks.benchmarks.synthetic.problem_type import (
    get_problem_spec,
    BenchmarkProblemType,
)
from il_benchmarks.benchmarks.run_benchmarks import benchmark_cb


def example_main():
    seed = 0
    kwargs = {"prop_score_mc_samples": 2048, "imitator_num_mini_batches": 2000}
    results = {}
    for problem_type in [BenchmarkProblemType.WHEEL, BenchmarkProblemType.MUSHROOM, BenchmarkProblemType.WARFARIN]:
        if problem_type == BenchmarkProblemType.WARFARIN:
            kwargs["training_freq"] = 100
            n_examples = 499
            num_actions = 20
        else:
            kwargs["training_freq"] = 1000
            n_examples = 4999
            num_actions = None
        problem_spec = get_problem_spec(
            problem_type=problem_type,
            n_examples=n_examples,
            seed=seed,
            num_actions=num_actions,
        )
        bandits, _ = get_algo_collection(
            algo_collection_type=AlgoCollectionType.ALL,
            problem_spec=problem_spec,
            **kwargs
        )
        results[problem_type] = benchmark_cb(
            problem_spec=problem_spec,
            algo_collection_type=AlgoCollectionType.CUSTOM,
            custom_algorithms=bandits,
            display=True,
        )



if __name__ == '__main__':
    example_main()
