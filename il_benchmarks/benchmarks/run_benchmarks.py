#!/usr/bin/env python3

import time
from typing import List, NamedTuple, Optional

import torch
from ..bandits import BaseBandit
from .benchmark_config import (
    AlgoCollectionType,
    get_algo_collection,
)
from .synthetic.problem import BenchmarkProblem
from torch import Tensor


MIN_EXAMPLES = 20


class BenchmarkResults(NamedTuple):
    # `n_a` is the number of observations for algo `a`
    algos: List[str]
    opt_rewards: Optional[Tensor]  # `n x 1`-dim tensor of optimal expected rewards
    opt_actions: Optional[Tensor]  # `n x 1`-dim tensor of optimal expected actions
    # List of `n_a x 1`-dim tensor of actions
    algo_actions: List[Tensor]
    # List of `n_a x 1`-dim tensor of rewards where
    algo_rewards: List[Tensor]
    # List of `n_a x num_actions`-dim tensor of rewards
    algo_prop_scores: Optional[List[Tensor]]
    # each element is a `n x num_actions`-dim tensor of predicted rewards
    algo_model_rewards: List[Optional[Tensor]]
    benchmark_name: str
    # List of `n_a x d`-dim tensor of contexts
    algo_contexts: List[Tensor]
    # List of `n_a`-dim tensor of wall times for action generation
    action_gen_wall_times: List[List[float]]
    # List of `n_a`-dim tensor of wall times for action generation
    policy_update_wall_times: List[List[float]]


def run_contextual_bandit(
    problem_spec: BenchmarkProblem,
    algos: List[BaseBandit],
    show_progress: bool = True,
    store_prop_scores: bool = True,
) -> BenchmarkResults:
    """Run a contextual bandit problem on a set of algorithms.

    Args:
        problem_spec: benchmark problem
        algos: List of algorithms to use in the contextual bandit instance.
        show_progress: a boolean indicating whether to print out progress and
            cumulative regret.
        store_prop_scores: a boolean indicating whether to store prop_scores

    Returns:
        BenchmarkResults: results
    """
    contexts = problem_spec.contexts
    if contexts.shape[0] < MIN_EXAMPLES:
        raise ValueError(f"Too few samples: {contexts.shape[0]} < {MIN_EXAMPLES}")
    observed_contexts = [
        torch.empty((0, contexts.shape[1]), dtype=contexts.dtype)
        for _ in algos  # `n x d`
    ]
    observed_actions = [
        torch.empty((0, 1), dtype=contexts.dtype) for _ in algos  # `n x 1`
    ]
    observed_rewards = [torch.empty((0, 1), dtype=contexts.dtype) for _ in algos]  # `n`
    action_gen_wall_times = [[] for _ in algos]
    policy_update_wall_times = [[] for _ in algos]
    if store_prop_scores:
        observed_prop_scores = [
            torch.empty(  # `n x num_actions`
                (0, problem_spec._data_config.num_actions), dtype=contexts.dtype
            )
            for _ in algos
        ]
    else:
        observed_prop_scores = None
    model_rewards: List[Optional[Tensor]] = []
    # Run the contextual bandit process
    for t in range(contexts.shape[0]):
        context = contexts[t].view(1, contexts.shape[-1])
        algo_actions = []
        for j, a in enumerate(algos):
            t_algo = observed_contexts[j].shape[0]
            action_gen_start_time = time.time()
            algo_action = a.action(contexts=context, t=t_algo).view(1, 1)
            action_gen_wall_times[j].append(time.time() - action_gen_start_time)
            algo_actions.append(algo_action)

        actions = torch.cat(algo_actions, dim=0)
        rewards = problem_spec.reward_func(
            contexts=context.expand(len(algos), -1),
            actions=actions.long(),
            timesteps=torch.full(size=(len(algos),), fill_value=t, dtype=torch.long),
        )
        new_obs_for_algo = [False for _ in range(len(algos))]
        for j, a in enumerate(algos):
            if not torch.isnan(rewards[j]).item():
                new_obs_for_algo[j] = True
                observed_contexts[j] = torch.cat([observed_contexts[j], context], dim=0)
                observed_actions[j] = torch.cat(
                    [observed_actions[j], actions[j].unsqueeze(0)], dim=0
                )
                observed_rewards[j] = torch.cat(
                    (observed_rewards[j], rewards[j].unsqueeze(0)), dim=0
                )
                if store_prop_scores:
                    prop_scores = a.prop_scores(contexts=context).view(
                        1, problem_spec._data_config.num_actions
                    )
                    observed_prop_scores[j] = torch.cat(
                        (observed_prop_scores[j], prop_scores), dim=0
                    )
        for j, a in enumerate(algos):
            policy_update_start_time = time.time()
            # only update when there is a new observation
            if new_obs_for_algo[j]:
                a.update(
                    contexts=observed_contexts[j],
                    actions=observed_actions[j],
                    rewards=observed_rewards[j],
                )
            policy_update_wall_times[j].append(time.time() - policy_update_start_time)
        if show_progress and t % int(contexts.shape[0] / MIN_EXAMPLES) == 0:
            print(f"Finished iteration {t}")
            optimal_rewards = problem_spec.optimal_rewards
            if optimal_rewards is not None:
                # Compute regret
                metric_name = "Regret"
                cumulative_metric = [
                    (optimal_rewards[: (t + 1)] - observed_rewards[j]).sum().item()
                    for j in range(len(algos))
                ]
            else:
                # Use reward
                metric_name = "Reward"
                cumulative_metric = [
                    observed_rewards[j].sum().item() for j in range(len(algos))
                ]
            cumulative_metric_by_algo = list(
                zip([a.name for a in algos], cumulative_metric)
            )

            print(f"Cumulative {metric_name}: {cumulative_metric_by_algo}")

    for a in algos:
        if a.name in (
            "greedy_neural_bandit_regressor",
            "gaussian_nl_bandit",
        ):
            model_rewards.append(a.model.predict(contexts).cpu())  # pyre-ignore
        else:
            model_rewards.append(None)

    return BenchmarkResults(
        algos=[a.name for a in algos],
        opt_rewards=problem_spec.optimal_rewards,
        opt_actions=problem_spec.optimal_actions,
        algo_actions=observed_actions,
        algo_rewards=observed_rewards,
        algo_prop_scores=observed_prop_scores,
        algo_model_rewards=model_rewards,
        benchmark_name=problem_spec.name,
        algo_contexts=observed_contexts,
        action_gen_wall_times=action_gen_wall_times,
        policy_update_wall_times=policy_update_wall_times,
    )


def display_results(
    benchmark_results: BenchmarkResults, total_wall_time: float
) -> None:

    print("---------------------------------------------------")
    print("---------------------------------------------------")
    print(
        f"{benchmark_results.benchmark_name} bandit completed after "
        f"{total_wall_time} seconds."
    )
    print("---------------------------------------------------")
    performance_pairs = []
    for j, algo_name in enumerate(benchmark_results.algos):
        algo_rewards = benchmark_results.algo_rewards[j]
        valid_mask = ~(torch.isnan(algo_rewards.view(-1)))
        total_reward = algo_rewards[valid_mask].sum().item()
        performance_pairs.append((algo_name, total_reward))
        performance_pairs = sorted(
            performance_pairs, key=lambda elt: elt[1], reverse=True
        )
    for i, (name, reward) in enumerate(performance_pairs):
        print("{:3}) {:20}| \t \t total reward = {:10}.".format(i, name, reward))
        optimal_rewards = benchmark_results.opt_rewards
        optimal_actions = benchmark_results.opt_actions
        if optimal_rewards is not None:
            print("---------------------------------------------------")
            print("Optimal total reward = {}.".format(optimal_rewards.sum().item()))
        if optimal_actions is not None:
            print("Frequency of optimal actions (action, frequency):")
            opt_action_list = optimal_actions.view(-1).tolist()
            print([[elt, opt_action_list.count(elt)] for elt in set(opt_action_list)])
        print("---------------------------------------------------")
        print("---------------------------------------------------")


def benchmark_cb(
    problem_spec: BenchmarkProblem,
    algo_collection_type: AlgoCollectionType = AlgoCollectionType.REGRESSION,
    custom_algorithms: Optional[List[BaseBandit]] = None,
    display: bool = True,
    store_prop_scores: bool = False,
) -> BenchmarkResults:
    # Problem parameters
    if not isinstance(algo_collection_type, AlgoCollectionType):
        raise ValueError("Invalid benchmark suite specified.")
    elif algo_collection_type == AlgoCollectionType.CUSTOM:
        benchmark_name = "Mushroom Benchmark: CUSTOM"
        if custom_algorithms is None:
            raise ValueError(
                "When algo_type is CUSTOM, custom_algorithms must be provided."
            )
        algos = custom_algorithms
    else:
        algos, benchmark_name = get_algo_collection(
            algo_collection_type=algo_collection_type, problem_spec=problem_spec
        )
    start_time = time.time()
    results = run_contextual_bandit(
        problem_spec=problem_spec,
        algos=algos,
        show_progress=display,
        store_prop_scores=store_prop_scores,
    )
    if display:
        # Display results
        display_results(
            benchmark_results=results, total_wall_time=time.time() - start_time
        )
    return results
