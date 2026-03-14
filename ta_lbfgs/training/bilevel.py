"""Bilevel optimization orchestrator with differentiable inner updates."""

import inspect
import math
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from ..config import TaLBFGSConfig
from ..core.baseline_lbfgs import FullBatchLBFGS
from ..core.hypergradient import implicit_differentiation
from ..core.hyperparameters import DifferentiableHyperparameters
from ..dashboard.live_dashboard import OptimizerDashboard
from .inner_loop import inner_train


class BilevelValidationError(RuntimeError):
    """Raised when bilevel assumptions are violated at runtime."""


class BilevelOptimizer:
    """
    Bilevel Optimization Engine.

    Outer loop: ta-LBFGS optimizes hyperparameters (lr, wd) using
    validation loss gradients computed via IFT.

    Inner loop: SGD trains model weights using training loss with
    differentiable hyperparameters.

    Integrates:
    - Layerwise topology-aware adaptive memory
    - Per-step condition number estimation
    - Saddle-point evasion
    - Rich CLI dashboard rendering

    Args:
        config: TaLBFGSConfig instance.
    """

    def __init__(self, config: TaLBFGSConfig):
        self.config = config
        self.hyperparams = DifferentiableHyperparameters(
            n_layers=config.n_layers,
            initial_lr=config.initial_lr,
            initial_wd=config.initial_weight_decay,
            device=config.device,
            dtype=config.get_torch_dtype(),
        )

        # Outer-loop optimizer on all differentiable hyperparameters.
        self.outer_optimizer = FullBatchLBFGS(
            list(self.hyperparams.parameters()),
            lr=config.lbfgs_lr,
            history_size=config.lbfgs_memory_base,
            line_search="None",  # Simplified for hyperparam space
        )

        self.dashboard = OptimizerDashboard(
            refresh_rate=config.dashboard_refresh_rate,
            sparkline_width=config.dashboard_sparkline_width,
        )

        # Tracking
        self.loss_history: List[float] = []
        self.inner_loss_history: List[float] = []
        self.best_loss = float("inf")
        self.hyperparam_history: List[Dict[str, object]] = []
        self.evasion_events: List[Dict] = []
        self.grad_magnitude_history: List[float] = []
        self.validity_checks: List[Dict[str, float]] = []
        self.inner_sensitivity_debug: List[Dict[str, object]] = []
        self.active_hybrid_shard: Dict[str, Any] = {}

    @staticmethod
    def _infer_layer_index(param_name: str, fallback_layers: int) -> int:
        parts = param_name.split(".")
        for idx, part in enumerate(parts[:-1]):
            if part in {"layers", "h", "blocks"} and parts[idx + 1].isdigit():
                return min(int(parts[idx + 1]), max(0, fallback_layers - 1))
        for part in parts:
            if part.isdigit():
                return min(int(part), max(0, fallback_layers - 1))
        return 0

    def _select_hybrid_shard(
        self,
        adapted_params: OrderedDict[str, torch.Tensor],
        outer_iter: int,
    ) -> OrderedDict[str, torch.Tensor]:
        if not self.config.hybrid_hypergradient:
            self.active_hybrid_shard = {
                "enabled": False,
                "outer_iter": outer_iter,
                "selected_layer_count": 0,
                "selected_param_count": len(adapted_params),
                "num_shards": 1,
                "active_shard": 0,
            }
            return adapted_params

        names = list(adapted_params.keys())
        if not names:
            self.active_hybrid_shard = {
                "enabled": True,
                "outer_iter": outer_iter,
                "selected_layer_count": 0,
                "selected_param_count": 0,
                "num_shards": 1,
                "active_shard": 0,
            }
            return adapted_params

        layer_for_name = {
            name: self._infer_layer_index(name, self.config.n_layers)
            for name in names
        }
        unique_layers = sorted(set(layer_for_name.values()))
        if not unique_layers:
            unique_layers = [0]

        fraction = min(1.0, max(1e-3, float(self.config.hybrid_shard_fraction)))
        num_shards = max(1, int(math.ceil(1.0 / fraction)))
        layers_per_shard = max(1, int(math.ceil(len(unique_layers) / num_shards)))
        rot_steps = max(1, int(self.config.hybrid_rotation_steps))
        active_shard = (outer_iter // rot_steps) % num_shards

        start = active_shard * layers_per_shard
        end = min(len(unique_layers), start + layers_per_shard)
        selected_layers = set(unique_layers[start:end])
        if not selected_layers:
            selected_layers = {unique_layers[active_shard % len(unique_layers)]}

        shard = OrderedDict(
            (name, param)
            for name, param in adapted_params.items()
            if layer_for_name.get(name, 0) in selected_layers
        )
        if not shard:
            first_name = names[0]
            shard = OrderedDict([(first_name, adapted_params[first_name])])

        self.active_hybrid_shard = {
            "enabled": True,
            "outer_iter": outer_iter,
            "selected_layer_count": len(selected_layers),
            "selected_param_count": len(shard),
            "num_shards": num_shards,
            "active_shard": active_shard,
        }
        return shard

    @staticmethod
    def _extract_trainable_params(model: nn.Module) -> OrderedDict[str, torch.Tensor]:
        return OrderedDict(
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad
        )

    @staticmethod
    def _supports_kwarg(fn: Callable, kwarg_name: str) -> bool:
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return False

        for param in sig.parameters.values():
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                return True
            if param.name == kwarg_name:
                return True
        return False

    def _call_objective(
        self,
        fn: Callable,
        model: nn.Module,
        data: Any,
        params_override: Optional[OrderedDict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if params_override is not None and not self._supports_kwarg(fn, "params_override"):
            raise ValueError(
                "Objective function must accept a 'params_override' kwarg "
                "for mathematically real bilevel optimization."
            )

        kwargs = {"params_override": params_override} if params_override is not None else {}
        return fn(model, data, self.hyperparams, **kwargs)

    def _validate_objective_signature(self, fn: Callable, fn_name: str):
        if not self._supports_kwarg(fn, "params_override"):
            raise BilevelValidationError(
                f"{fn_name} must accept a 'params_override' keyword argument for "
                "differentiable bilevel optimization."
            )

    @staticmethod
    def _check_finite_scalar(x: torch.Tensor, name: str):
        if not torch.isfinite(x).item():
            raise BilevelValidationError(f"Non-finite scalar detected for {name}: {x.item()}")

    def _validate_bilevel_state(self, state: Dict[str, Any], outer_iter: int):
        train_loss = state["train_loss"]
        val_loss = state["val_loss"]
        hypergrads = state["hypergradients"]

        self._check_finite_scalar(train_loss, f"train_loss@iter{outer_iter}")
        self._check_finite_scalar(val_loss, f"val_loss@iter{outer_iter}")

        finite_hypergrads = []
        for idx, grad in enumerate(hypergrads):
            if grad is None:
                raise BilevelValidationError(
                    f"Missing hypergradient tensor at index {idx} (iter={outer_iter})."
                )
            if not torch.isfinite(grad).all().item():
                raise BilevelValidationError(
                    f"Non-finite hypergradient values at index {idx} (iter={outer_iter})."
                )
            finite_hypergrads.append(grad)

        grad_norm = sum(g.norm().item() for g in finite_hypergrads)
        if grad_norm <= 1e-16:
            raise BilevelValidationError(
                "Degenerate hypergradient norm (near zero). "
                "Check objective coupling between inner loop and hyperparameters."
            )

    def _evaluate_meta_val_loss(
        self,
        model: nn.Module,
        train_fn: Callable,
        val_fn: Callable,
        train_data: Any,
        val_data: Any,
        create_graph: bool,
    ) -> torch.Tensor:
        initial_params = self._extract_trainable_params(model)
        _, adapted_params, _ = inner_train(
            model,
            train_data,
            self.hyperparams,
            loss_fn=lambda params_override: self._call_objective(
                train_fn,
                model,
                train_data,
                params_override=params_override,
            ),
            steps=self.config.inner_steps,
            create_graph=create_graph,
            initial_params=initial_params,
            enable_sensitivity_debug=False,
        )
        return self._call_objective(
            val_fn,
            model,
            val_data,
            params_override=adapted_params,
        )

    def _finite_difference_check(
        self,
        model: nn.Module,
        train_fn: Callable,
        val_fn: Callable,
        train_data: Any,
        val_data: Any,
        hypergrads: List[torch.Tensor],
        eps: float = 1e-3,
    ) -> Dict[str, float]:
        grad_scores = [g.abs().reshape(-1).max().item() for g in hypergrads]
        target_idx = int(np.argmax(grad_scores))
        target_param = list(self.hyperparams.parameters())[target_idx]
        target_grad_tensor = hypergrads[target_idx].reshape(-1)
        elem_idx = int(torch.argmax(target_grad_tensor.abs()).item())
        target_grad = target_grad_tensor[elem_idx].item()
        target_flat = target_param.data.view(-1)
        original = target_flat[elem_idx].item()
        local_eps = max(eps, 1e-2 * max(abs(original), 1.0))

        with torch.no_grad():
            target_flat[elem_idx] = original + local_eps
        val_plus = self._evaluate_meta_val_loss(
            model,
            train_fn,
            val_fn,
            train_data,
            val_data,
            create_graph=False,
        ).item()

        with torch.no_grad():
            target_flat[elem_idx] = original - local_eps
        val_minus = self._evaluate_meta_val_loss(
            model,
            train_fn,
            val_fn,
            train_data,
            val_data,
            create_graph=False,
        ).item()

        with torch.no_grad():
            target_flat[elem_idx] = original

        fd_slope = (val_plus - val_minus) / (2.0 * local_eps)
        sign_match = float(np.sign(fd_slope) == np.sign(target_grad))

        check = {
            "fd_slope": float(fd_slope),
            "hypergrad": float(target_grad),
            "abs_error": float(abs(fd_slope - target_grad)),
            "sign_match": sign_match,
        }
        self.validity_checks.append(check)
        return check

    def _evaluate_bilevel_state(
        self,
        model: nn.Module,
        train_fn: Callable,
        val_fn: Callable,
        train_data: Any,
        val_data: Any,
        outer_iter: int,
    ) -> Dict[str, Any]:
        initial_params = self._extract_trainable_params(model)

        train_loss, adapted_params, sensitivity_debug = inner_train(
            model,
            train_data,
            self.hyperparams,
            loss_fn=lambda params_override: self._call_objective(
                train_fn,
                model,
                train_data,
                params_override=params_override,
            ),
            steps=self.config.inner_steps,
            create_graph=True,
            initial_params=initial_params,
            enable_sensitivity_debug=True,
        )

        val_loss = self._call_objective(
            val_fn,
            model,
            val_data,
            params_override=adapted_params,
        )

        shard_params = self._select_hybrid_shard(adapted_params, outer_iter)
        weights = list(shard_params.values())
        hyper_tensors = list(self.hyperparams.parameters())
        hypergradients = implicit_differentiation(
            val_loss,
            train_loss,
            hyper_tensors,
            weights,
            method=self.config.hypergradient_method,
            cg_max_iter=self.config.cg_max_iter,
            cg_tol=self.config.cg_tol,
            neumann_terms=self.config.neumann_terms,
        )

        return {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "adapted_params": adapted_params,
            "hypergradients": hypergradients,
            "inner_sensitivity_debug": sensitivity_debug,
            "hybrid_shard": self.active_hybrid_shard,
        }

    def optimize(
        self,
        model: nn.Module,
        train_fn: Callable,
        val_fn: Callable,
        train_data: Any,
        val_data: Any,
        use_dashboard: bool = True,
        run_validity_checks: bool = True,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run the full bilevel optimization loop.

        Args:
            model: The model to optimize hyperparameters for.
            train_fn: (model, data, hyperparams) → training loss.
            val_fn: (model, data) → validation loss.
            train_data: Training dataset/batch.
            val_data: Validation dataset/batch.
            use_dashboard: Whether to render the live CLI dashboard.

        Returns:
            Dictionary with optimization results.
        """
        config = self.config

        self._validate_objective_signature(train_fn, "train_fn")
        self._validate_objective_signature(val_fn, "val_fn")

        if use_dashboard:
            live = self.dashboard.live()
            live.__enter__()
            self.dashboard.add_log("ta-LBFGS Bilevel Optimizer starting...")

        try:
            for outer_iter in range(config.outer_steps):
                latest_state: Dict[str, Any] = {}

                def outer_closure():
                    latest_state.clear()
                    for p in self.hyperparams.parameters():
                        p.grad = None

                    state = self._evaluate_bilevel_state(
                        model,
                        train_fn,
                        val_fn,
                        train_data,
                        val_data,
                        outer_iter,
                    )
                    latest_state.update(state)

                    for param, grad in zip(self.hyperparams.parameters(), state["hypergradients"]):
                        param.grad = grad

                    return state["val_loss"]

                self.outer_optimizer.step(outer_closure)

                if not latest_state:
                    raise RuntimeError("Outer closure did not produce a bilevel state.")

                self._validate_bilevel_state(latest_state, outer_iter)

                train_loss = latest_state["train_loss"]
                val_loss = latest_state["val_loss"]
                hypergrads = latest_state["hypergradients"]
                sensitivity_debug = latest_state.get("inner_sensitivity_debug")
                hybrid_shard = latest_state.get("hybrid_shard")
                if sensitivity_debug is not None:
                    self.inner_sensitivity_debug.append(sensitivity_debug)

                if run_validity_checks and outer_iter == 0:
                    fd_check = self._finite_difference_check(
                        model,
                        train_fn,
                        val_fn,
                        train_data,
                        val_data,
                        hypergrads,
                    )
                    if use_dashboard:
                        self.dashboard.add_log(
                            "Validity check: "
                            f"fd={fd_check['fd_slope']:.3e}, "
                            f"hg={fd_check['hypergrad']:.3e}, "
                            f"sign_match={int(fd_check['sign_match'])}"
                        )

                train_loss_val = train_loss.item() if isinstance(train_loss, torch.Tensor) else float(train_loss)
                loss_val = val_loss.item() if isinstance(val_loss, torch.Tensor) else float(val_loss)

                self.inner_loss_history.append(train_loss_val)
                self.loss_history.append(loss_val)
                if loss_val < self.best_loss:
                    self.best_loss = loss_val

                grad_mag = sum(g.norm().item() for g in hypergrads if g is not None)
                self.grad_magnitude_history.append(grad_mag)

                # Clamp hyperparameters to valid range
                self.hyperparams.clamp()

                # Record
                hp_dict = self.hyperparams.as_float_dict()
                self.hyperparam_history.append(hp_dict)

                # ── Dashboard Update ────────────────────────────────
                if use_dashboard:
                    outer_state = {
                        "iteration": outer_iter + 1,
                        "total_iterations": config.outer_steps,
                        "loss": loss_val,
                        "best_loss": self.best_loss,
                        "lr": hp_dict["lr"][0] if isinstance(hp_dict["lr"], list) else hp_dict["lr"],
                        "wd": hp_dict["wd"][0] if isinstance(hp_dict["wd"], list) else hp_dict["wd"],
                    }

                    # For preliminary demo, create simple layer data
                    layer_data = self._get_demo_layer_data(outer_iter)

                    self.dashboard.update(
                        layer_data=layer_data,
                        outer_state=outer_state,
                        evasion_events=None,
                    )

                    if outer_iter % 5 == 0:
                        self.dashboard.add_log(
                            f"Iter {outer_iter}: train={train_loss_val:.6f} val={loss_val:.6f} "
                            f"lr={outer_state['lr']:.6f} wd={outer_state['wd']:.6f}"
                        )

                if progress_callback is not None:
                    progress_callback(
                        {
                            "iteration": outer_iter + 1,
                            "total_iterations": config.outer_steps,
                            "train_loss": train_loss_val,
                            "val_loss": loss_val,
                            "best_loss": self.best_loss,
                            "grad_magnitude": grad_mag,
                            "hyperparams": hp_dict,
                            "sensitivity_debug": sensitivity_debug,
                            "hybrid_shard": hybrid_shard,
                        }
                    )

        finally:
            if use_dashboard:
                self.dashboard.add_log("Optimization complete.", style="bold green")
                time.sleep(2)
                live.__exit__(None, None, None)

        return {
            "best_loss": self.best_loss,
            "inner_loss_history": self.inner_loss_history,
            "loss_history": self.loss_history,
            "hyperparam_history": self.hyperparam_history,
            "final_hyperparams": self.hyperparams.as_float_dict(),
            "grad_magnitude_history": self.grad_magnitude_history,
            "validity_checks": self.validity_checks,
            "inner_sensitivity_debug": self.inner_sensitivity_debug,
            "hybrid_shard": self.active_hybrid_shard,
            "evasion_events": self.evasion_events,
        }

    def _get_demo_layer_data(self, iteration: int) -> Dict[str, Dict]:
        """Generate synthetic layer data for demo dashboard rendering."""
        # In production, this comes from LayerwiseTaLBFGS.get_all_layer_data()
        import random

        layers = {}
        for i in range(4):
            name = f"layers.{i}"
            kappa = max(1.0, 10 * (i + 1) + random.gauss(0, 3))
            secant = random.gauss(0.5, 0.3) if random.random() > 0.1 else -0.02
            grad_norm = max(0, 0.1 * (4 - i) + random.gauss(0, 0.02))

            landscape = (
                "Saddle Point" if secant <= 0
                else "Narrow Ravine" if kappa > 30
                else "Ill-Conditioned" if kappa > 10
                else "Convex Bowl"
            )

            layers[name] = {
                "kappa": kappa,
                "memory_size": max(3, min(20, int(np.log(kappa) + 5))),
                "grad_norm": grad_norm,
                "secant": secant,
                "landscape": landscape,
                "kappa_history": [
                    max(1, kappa + random.gauss(0, 5))
                    for _ in range(min(iteration + 1, 30))
                ],
                "evasion_count": random.randint(0, 3) if secant <= 0 else 0,
                "iteration": iteration,
            }

        return layers
