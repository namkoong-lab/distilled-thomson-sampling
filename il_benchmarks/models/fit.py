#!/usr/bin/env python3

from math import ceil
from typing import Optional, Union

import torch
from torch import Tensor, nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset

from ..core.hyperparameters import NNHyperparameters
from ..utils import one_hot_encode_action


def fit_model(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    criterion: nn.Module,
    model_hparams:  NNHyperparameters,
    lr: float,
    verbose: bool,
    weights: Optional[Tensor] = None,
    minimize: bool = True,
    output_mask_numeric: Optional[Tensor] = None,
) -> float:
    """Fit a pytorch model using the specified optimizer.

    Args:
        model: A torch.nn.module
        inputs: `n x d` tensor of inputs
        targets: `n x num_targets`-dim tensor of targets
        criterion: loss function
        nn_hparams: hyperparameters for the NN training
        lr: the learning rate
        verbose: bool indicating whether to use verbose printing
        weights: `n x 1`-dim tensor of example level weights
        minimize: bool indicating whether the criterion should be minimized.
        output_mask_numeric: `n x 1`-dim tensor that contains the output index that
            corresponds the observed target.
    Returns:
        float: the new (possibly decayed) learning rate
    """
    model.train()
    # set up dataset
    tensors = [inputs, targets]
    if weights is not None:
        tensors.append(weights)
    if output_mask_numeric is not None:
        tensors.append(output_mask_numeric)
    data_loader = DataLoader(
        dataset=TensorDataset(*tensors),
        batch_size=model_hparams.batch_size,
        shuffle=model_hparams.shuffle,
    )
    # set up optimizer
    optimizer = model_hparams.optimizer_cls(
        params=[{"params": model.parameters()}],
        lr=lr,
        weight_decay=model_hparams.weight_decay,
    )
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=[lambda epoch: 1.0 / (1 + model_hparams.lr_decay_rate * epoch)],
    )
    batch_idx = 0
    num_mini_batches = model_hparams.num_mini_batches
    if num_mini_batches is not None:
        if verbose:
            print(f"Training model for {num_mini_batches} mini-batches.")
    else:
        num_mini_batches = model_hparams.num_epochs * ceil(
            len(data_loader.dataset) / model_hparams.batch_size
        )
        if verbose:
            print(
                f"Training model for {model_hparams.num_epochs} epochs"
                f" ({num_mini_batches} mini-batches)."
            )
    device = next(model.parameters()).device
    while batch_idx < num_mini_batches:
        for batch_tensors in data_loader:
            if batch_idx >= num_mini_batches:
                break
            if verbose and batch_idx % max(int(num_mini_batches / 20), 1) == 0:
                print(f"Batch {batch_idx}/{num_mini_batches}")
            input_batch = batch_tensors[0].to(device=device)
            target_batch = batch_tensors[1].to(device=device)
            output_batch = model(input_batch)
            # this for multi-output outcome models, where we model:
            # \hat{f}(x) -> [\hat{f}(x|a_0), ...\hat{f}(x|a_k)]
            # We want to compute the loss between the observed outcome y|a and
            # the corresponding prediction \hat{f}(x|a).
            # So we apply a mask to set predictions for unobserved actions to 0.
            if output_mask_numeric is not None:
                output_mask_numeric_batch = batch_tensors[-1].to(device=device)
                output_mask_batch = one_hot_encode_action(
                    output_mask_numeric_batch, num_actions=target_batch.shape[-1]
                )
                output_batch *= output_mask_batch
            loss = criterion(output_batch, target_batch).sum(dim=-1)
            if weights is not None:
                weights_batch = batch_tensors[2].to(device=device)
                loss *= weights_batch.view(-1)
            # Zero the gradients before running the backward pass.
            model.zero_grad()
            if not minimize:
                loss *= -1
            loss.mean().backward()
            # clip gradients
            if model_hparams.max_grad_norm is not None:
                clip_grad_norm_(model.parameters(), model_hparams.max_grad_norm)
            optimizer.step()  # pyre-ignore [20]
            # decay learning rate
            if (
                model_hparams.scheduler_step_interval is not None
                # pyre-ignore [6]
                and (batch_idx + 1) % model_hparams.scheduler_step_interval == 0
            ):
                scheduler.step()
            batch_idx += 1
            if verbose and batch_idx % max(int(num_mini_batches / 20), 1) == 0:
                # Evaluate in-sample loss
                in_sample_loss = evaluate_loss(
                    model=model,
                    inputs=inputs,
                    targets=targets,
                    criterion=criterion,
                    weights=weights,
                    flip_sign=not minimize,
                )
                model.train()
                print(
                    f"Mean in-sample loss {type(criterion).__name__}: "
                    f"{round(in_sample_loss, 3)}"
                )

    model.eval()
    return optimizer.param_groups[0]["lr"]


def evaluate_loss(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    criterion: nn.Module,
    weights: Optional[Tensor] = None,
    flip_sign: bool = False,
    output_mask_numeric: Optional[Tensor] = None,
) -> float:
    """Evaluate loss on given data.

    Args:
        network: A torch.nn.module
        inputs: `n x d` tensor of inputs
        targets: `n x num_targets`-dim tensor of targets
        criterion: loss function
        weights: `n x 1`-dim tensor of example level weights
        flip_sign: bool indicating whether the loss criterion should be multiplied
            by -1.
        output_mask_numeric: `n x 1`-dim tensor that contains the output index that
            corresponds the observed target.
    Returns:
        float: the mean loss
    """
    model.eval()
    tensors = [inputs, targets]
    if weights is not None:
        tensors.append(weights)
    data_loader = DataLoader(dataset=TensorDataset(*tensors), batch_size=1024)
    device = next(model.parameters()).device
    in_sample_loss = 0.0
    for batch_tensors in data_loader:
        input_batch = batch_tensors[0].to(device=device)
        target_batch = batch_tensors[1].to(device=device)
        with torch.no_grad():
            output_batch = model(input_batch)
            if output_mask_numeric is not None:
                output_mask_numeric_batch = batch_tensors[-1].to(device=device)
                output_mask_batch = one_hot_encode_action(
                    output_mask_numeric_batch, num_actions=target_batch.shape[-1]
                )
                output_batch *= output_mask_batch
            batch_loss = criterion(output_batch, target_batch).sum(dim=-1)
        if weights is not None:
            weights_batch = batch_tensors[2].to(device=device)
            batch_loss *= weights_batch.view(-1)
        in_sample_loss += batch_loss.sum().item()
    if flip_sign:
        in_sample_loss *= -1
    return in_sample_loss / len(data_loader.dataset)
