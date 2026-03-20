"""Differentiable inner training loop utilities for bilevel optimization."""

from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..core.hyperparameters import DifferentiableHyperparameters
from ..core.lbfgs import classify_param_group, should_freeze_in_inner_loop

try:
    from torch.func import functional_call as torch_functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call as torch_functional_call


def functional_call_model(
    model: nn.Module,
    params_override: OrderedDict[str, torch.Tensor],
    *args,
    **kwargs,
):
    """Run a model forward pass against a virtual parameter mapping."""
    return torch_functional_call(model, params_override, args=args, kwargs=kwargs)


def _infer_layer_index(param_name: str, max_layers: int) -> int:
    parts = param_name.split(".")
    for idx, part in enumerate(parts[:-1]):
        if part in {"layers", "h", "blocks"} and parts[idx + 1].isdigit():
            return min(int(parts[idx + 1]), max_layers - 1)
    for part in parts:
        if part.isdigit():
            return min(int(part), max_layers - 1)
    return 0


def inner_train(
    model: nn.Module,
    train_data,
    hyperparams: DifferentiableHyperparameters,
    loss_fn: Callable[[OrderedDict[str, torch.Tensor]], torch.Tensor],
    steps: int = 10,
    create_graph: bool = True,
    initial_params: Optional[OrderedDict[str, torch.Tensor]] = None,
    l2_inner_reg: float = 1e-4,
    enable_sensitivity_debug: bool = False,
    mismatch_threshold: float = 0.5,
    disconnect_ratio: float = 0.1,
) -> Tuple[torch.Tensor, OrderedDict[str, torch.Tensor], Optional[Dict[str, object]]]:
    """
    Run a differentiable inner loop using virtual parameter updates.

    The returned parameter mapping stays connected to the computation graph,
    allowing outer-loop hypergradients to depend on inner optimization steps.
    """
    if initial_params is None:
        adapted_params: OrderedDict[str, torch.Tensor] = OrderedDict(
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad
        )
    else:
        adapted_params = OrderedDict(initial_params)

    final_loss: Optional[torch.Tensor] = None
    sensitivity_debug: Optional[Dict[str, object]] = None

    if enable_sensitivity_debug:
        sensitivity_debug = {
            "steps": [],
            "first_suspected_disconnect_step": None,
        }
        hp_probe = next(hyperparams.parameters())

        def _estimate_actual_sensitivity_norm(
            params_map: OrderedDict[str, torch.Tensor],
        ) -> torch.Tensor:
            # Jacobian-vector probes provide a cheap approximation to ||d w / d lambda||.
            sq_sum = torch.zeros((), device=hp_probe.device, dtype=hp_probe.dtype)
            for p in params_map.values():
                probe = torch.ones_like(p)
                dot = (p * probe).sum()
                sens = torch.autograd.grad(
                    dot,
                    hp_probe,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )[0]
                if sens is None:
                    continue
                sq_sum = sq_sum + sens.reshape(-1).pow(2).sum()
            return torch.sqrt(sq_sum + 1e-18)

        def _predict_sensitivity(history: List[float]) -> float:
            if len(history) == 0:
                return 0.0
            if len(history) == 1:
                return history[-1]
            if len(history) == 2:
                v = history[-1] - history[-2]
                return history[-1] + v

            v_t = history[-1] - history[-2]
            v_tm1 = history[-2] - history[-3]
            a_t = v_t - v_tm1
            return history[-1] + v_t + 0.5 * a_t

        actual_history: List[float] = []

    for step_idx in range(steps):
        # ADDS: explicit L2 term in inner objective to support mu-strong-convexity assumptions.
        # REMOVES: raw unregularized inner loss assignment in this function.
        base_loss = loss_fn(adapted_params)
        reg_loss = l2_inner_reg * sum(p.norm() ** 2 for p in adapted_params.values())
        final_loss = base_loss + reg_loss
        grads = torch.autograd.grad(
            final_loss,
            tuple(adapted_params.values()),
            create_graph=create_graph,
            retain_graph=create_graph,
            allow_unused=True,
        )

        updated_params: OrderedDict[str, torch.Tensor] = OrderedDict()
        for (name, param), grad in zip(adapted_params.items(), grads):
            # ADDS: freeze guard for rope/embedding groups during inner-loop updates.
            # REMOVES: unconditional update attempts over all parameter groups.
            group = classify_param_group(name, param, model)
            if should_freeze_in_inner_loop(group):
                updated_params[name] = param
                continue

            if grad is None:
                updated_params[name] = param
                continue

            layer_idx = _infer_layer_index(name, hyperparams.n_layers)
            lr = hyperparams.get_layer_lr(layer_idx)
            wd = hyperparams.get_layer_wd(layer_idx)
            update = grad + wd * param
            updated_params[name] = param - lr * update

        adapted_params = updated_params

        if sensitivity_debug is not None:
            predicted = _predict_sensitivity(actual_history)
            actual = _estimate_actual_sensitivity_norm(adapted_params).item()
            denom = abs(predicted) + 1e-12
            rel_mismatch = abs(actual - predicted) / denom if denom > 0.0 else 0.0
            suspected = (rel_mismatch > mismatch_threshold) and (abs(actual) < disconnect_ratio * max(abs(predicted), 1e-12))
            if suspected and sensitivity_debug["first_suspected_disconnect_step"] is None:
                sensitivity_debug["first_suspected_disconnect_step"] = step_idx

            sensitivity_debug["steps"].append(
                {
                    "step": step_idx,
                    "predicted_sensitivity_norm": float(predicted),
                    "actual_sensitivity_norm": float(actual),
                    "relative_mismatch": float(rel_mismatch),
                    "suspected_disconnect": bool(suspected),
                }
            )
            actual_history.append(actual)

    if final_loss is None:
        raise ValueError("inner_train requires steps >= 1")

    return final_loss, adapted_params, sensitivity_debug


def inner_train_synthetic(
    weights: torch.Tensor,
    train_fn: Callable,
    hyperparams: DifferentiableHyperparameters,
    steps: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simplified inner loop for synthetic optimization problems.

    Works with raw weight tensors instead of nn.Module for testing.

    Args:
        weights: Parameter tensor to optimize.
        train_fn: Function (weights) → scalar training loss.
        hyperparams: Differentiable hyperparameters.
        steps: Number of inner steps.

    Returns:
        (final_loss, final_weights).
    """
    lr = hyperparams.lr
    wd = hyperparams.wd
    w = weights.clone().requires_grad_(True)

    for _ in range(steps):
        loss = train_fn(w) + wd * (w * w).sum()
        grad = torch.autograd.grad(loss, w, create_graph=True)[0]
        w = w - lr * grad

    final_loss = train_fn(w) + wd * (w * w).sum()
    return final_loss, w
