"""Bilevel optimization orchestrator with differentiable inner updates."""

import inspect
import math
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..config import TaLBFGSConfig
from ..core.baseline_lbfgs import FullBatchLBFGS
from ..core.hypergradient import implicit_differentiation, outer_precondition
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
            damping=config.lbfgs_damping,
            damping_eps=config.lbfgs_damping_eps,
            curvature_threshold=config.curvature_threshold,
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
        self._ema_deltas: List[Optional[torch.Tensor]] = []
        self._hp_delta_window: List[float] = []
        self._hp_updates_frozen: bool = False
        self.hutch_trace_history: List[float] = []
        self._sach_probe_cache: Optional[torch.Tensor] = None
        self._sach_cache_age: int = 0
        self._sach_cached_residual_diag: Optional[torch.Tensor] = None
        self._sach_cached_residual_trace: Optional[float] = None
        self._sach_refresh_count: int = 0
        self._sach_total_calls: int = 0
        self._sach_last_drift: float = 0.0

    @staticmethod
    def _flatten_tensor_list(tensors: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat([t.reshape(-1) for t in tensors], dim=0)

    @staticmethod
    def _unflatten_tensor_list(flat: torch.Tensor, templates: List[torch.Tensor]) -> List[torch.Tensor]:
        out: List[torch.Tensor] = []
        offset = 0
        for t in templates:
            n = t.numel()
            out.append(flat[offset: offset + n].view_as(t))
            offset += n
        return out

    def _get_outer_secant_pairs(self, device: torch.device, dtype: torch.dtype) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        state = self.outer_optimizer.state.get("global_state", {})
        old_stps = state.get("old_stps", [])
        old_dirs = state.get("old_dirs", [])
        pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for s_vec, y_vec in zip(old_stps, old_dirs):
            if s_vec is None or y_vec is None:
                continue
            s = s_vec.detach().to(device=device, dtype=dtype).reshape(-1)
            y = y_vec.detach().to(device=device, dtype=dtype).reshape(-1)
            if s.numel() != y.numel() or s.numel() == 0:
                continue
            pairs.append((s, y))
        return pairs

    def _sach_should_refresh(self, topology_drift: float) -> bool:
        if self._sach_probe_cache is None:
            return True
        if self._sach_cached_residual_diag is None:
            return True
        if self._sach_cache_age >= max(1, int(self.config.outer_sachpp_refresh_interval)):
            return True
        if topology_drift > float(self.config.outer_sachpp_drift_threshold):
            return True
        return False

    def _sach_refresh_probes(self, dim: int, device: torch.device, dtype: torch.dtype) -> None:
        m = max(1, int(self.config.outer_sachpp_probe_count))
        raw = torch.randn((dim, m), device=device, dtype=dtype)
        if bool(self.config.outer_sachpp_use_qr_probes) and m > 1:
            q, _ = torch.linalg.qr(raw)
            probes = q.transpose(0, 1).contiguous()
        else:
            denom = raw.norm(dim=0, keepdim=True).clamp_min(1e-8)
            probes = (raw / denom).transpose(0, 1).contiguous()
        self._sach_probe_cache = probes
        self._sach_cache_age = 0
        self._sach_refresh_count += 1

    def _sach_lowrank_terms(
        self,
        secant_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
        dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, float]:
        eps = float(self.config.outer_sachpp_epsilon)
        diag = torch.zeros(dim, device=device, dtype=dtype)
        tr = 0.0
        for s_k, y_k in secant_pairs:
            curv = float(torch.dot(s_k, y_k).item())
            if curv <= eps:
                continue
            diag = diag + (y_k * y_k) / curv
            tr += float(torch.dot(y_k, y_k).item() / curv)
        return diag, tr

    def _sach_lowrank_hvp(
        self,
        z_flat: torch.Tensor,
        secant_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        eps = float(self.config.outer_sachpp_epsilon)
        out = torch.zeros_like(z_flat)
        for s_k, y_k in secant_pairs:
            curv = float(torch.dot(s_k, y_k).item())
            if curv <= eps:
                continue
            out = out + y_k * (torch.dot(y_k, z_flat) / curv)
        return out

    def _precondition_hypergradients_sach(
        self,
        val_loss: torch.Tensor,
        hypergrads: List[torch.Tensor],
        params: List[torch.Tensor],
        topology_drift: float = 0.0,
    ) -> Tuple[List[torch.Tensor], float]:
        self._sach_total_calls += 1
        self._sach_last_drift = float(topology_drift)

        hvp_fn = self._build_hvp_fn(val_loss, params)
        flat_grads = self._flatten_tensor_list(hypergrads)
        dim = int(flat_grads.numel())

        def flat_hvp(z_flat: torch.Tensor) -> torch.Tensor:
            z_list = self._unflatten_tensor_list(z_flat, params)
            hz_list = hvp_fn(z_list)
            return self._flatten_tensor_list(hz_list)

        secant_pairs = self._get_outer_secant_pairs(device=flat_grads.device, dtype=flat_grads.dtype)
        diag_lowrank, tr_lowrank = self._sach_lowrank_terms(
            secant_pairs,
            dim=dim,
            device=flat_grads.device,
            dtype=flat_grads.dtype,
        )

        if self._sach_should_refresh(topology_drift):
            self._sach_refresh_probes(dim=dim, device=flat_grads.device, dtype=flat_grads.dtype)
            probes = self._sach_probe_cache
            assert probes is not None
            residual_diag = torch.zeros(dim, device=flat_grads.device, dtype=flat_grads.dtype)
            residual_trace = 0.0
            for i in range(probes.size(0)):
                z = probes[i]
                hz_full = flat_hvp(z)
                hz_low = self._sach_lowrank_hvp(z, secant_pairs)
                hz_res = hz_full - hz_low
                residual_diag = residual_diag + z * hz_res
                residual_trace += float(torch.dot(z, hz_res).item())
            m = float(max(1, probes.size(0)))
            self._sach_cached_residual_diag = residual_diag / m
            self._sach_cached_residual_trace = residual_trace / m
        self._sach_cache_age += 1

        residual_diag = self._sach_cached_residual_diag
        if residual_diag is None:
            residual_diag = torch.zeros(dim, device=flat_grads.device, dtype=flat_grads.dtype)

        diag_full = diag_lowrank + residual_diag
        denom = torch.abs(diag_full) + float(self.config.outer_sachpp_epsilon)
        precond_flat = torch.nan_to_num(flat_grads / denom, nan=0.0, posinf=1.0, neginf=-1.0)
        precond_grads = self._unflatten_tensor_list(precond_flat, hypergrads)
        grad_norm = float(sum(g.norm().item() for g in precond_grads if g is not None))

        if bool(self.config.outer_hutchpp_trace_enabled):
            tr_res = float(self._sach_cached_residual_trace or 0.0)
            self.hutch_trace_history.append(float(tr_lowrank + tr_res))

        return precond_grads, grad_norm

    @staticmethod
    def _build_hvp_fn(
        loss: torch.Tensor,
        params: List[torch.Tensor],
    ) -> Callable[[List[torch.Tensor]], List[torch.Tensor]]:
        """Create an HVP callable that returns H @ v without materializing dense Hessian."""
        first_grads = torch.autograd.grad(
            loss,
            params,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )

        def hvp_fn(v_list: List[torch.Tensor]) -> List[torch.Tensor]:
            dot = torch.zeros((), device=loss.device, dtype=loss.dtype)
            for g, v in zip(first_grads, v_list):
                if g is None:
                    continue
                dot = dot + torch.sum(g * v)

            hvp_raw = torch.autograd.grad(
                dot,
                params,
                retain_graph=True,
                allow_unused=True,
            )
            out: List[torch.Tensor] = []
            for hvi, pi in zip(hvp_raw, params):
                if hvi is None:
                    out.append(torch.zeros_like(pi))
                else:
                    out.append(hvi)
            return out

        return hvp_fn

    def _set_outer_lr_for_step(self, outer_iter: int):
        """Apply optional warm-up scaling to the outer optimizer learning rate."""
        base_lr = float(self.config.lbfgs_lr)
        scale = 1.0
        if self.config.outer_lr_warmup_enabled:
            warmup_steps = max(1, int(self.config.outer_lr_warmup_steps))
            start_scale = float(self.config.outer_lr_warmup_start_scale)
            start_scale = max(0.0, min(1.0, start_scale))
            progress = min(1.0, float(outer_iter + 1) / float(warmup_steps))
            scale = start_scale + (1.0 - start_scale) * progress

        for group in self.outer_optimizer.param_groups:
            group["lr"] = base_lr * scale

    def _clip_hypergradients(
        self,
        hypergrads: List[torch.Tensor],
    ) -> (List[torch.Tensor], float):
        """Optionally clip hypergradient global norm without changing direction."""
        grads = [g.clone() for g in hypergrads]
        grad_norm = float(sum(g.norm().item() for g in grads if g is not None))

        if not self.config.outer_grad_clip_enabled:
            return grads, grad_norm

        max_norm = float(self.config.outer_grad_clip_max_norm)
        if max_norm <= 0.0:
            return grads, grad_norm

        stacked_norm = torch.sqrt(sum(torch.sum(g * g) for g in grads if g is not None))
        if stacked_norm.item() > max_norm:
            scale = max_norm / (stacked_norm.item() + 1e-12)
            grads = [g * scale for g in grads]
        clipped_norm = float(sum(g.norm().item() for g in grads if g is not None))
        return grads, clipped_norm

    @staticmethod
    def _sanitize_hypergradients(
        hypergrads: List[torch.Tensor],
        fallback: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[List[torch.Tensor], bool]:
        """Replace non-finite gradient values with safe finite tensors."""
        sanitized: List[torch.Tensor] = []
        recovered = False

        for idx, grad in enumerate(hypergrads):
            if grad is None:
                recovered = True
                if fallback is not None and idx < len(fallback) and fallback[idx] is not None:
                    safe_grad = torch.nan_to_num(fallback[idx].detach(), nan=0.0, posinf=1.0, neginf=-1.0)
                else:
                    raise BilevelValidationError(
                        f"Missing hypergradient tensor at index {idx} with no fallback available."
                    )
                sanitized.append(safe_grad)
                continue

            if torch.isfinite(grad).all().item():
                sanitized.append(grad)
                continue

            recovered = True
            if fallback is not None and idx < len(fallback) and fallback[idx] is not None and torch.isfinite(fallback[idx]).all().item():
                safe_grad = fallback[idx].detach().clone()
            else:
                safe_grad = torch.nan_to_num(grad, nan=0.0, posinf=1.0, neginf=-1.0)
            sanitized.append(safe_grad)

        return sanitized, recovered

    def _apply_hp_ema_damping(self, before_params: List[torch.Tensor]):
        """Smooth raw outer updates with an EMA over parameter deltas."""
        if not self.config.outer_hp_ema_enabled:
            return

        beta = float(self.config.outer_hp_ema_beta)
        beta = max(0.0, min(0.9999, beta))
        params = list(self.hyperparams.parameters())
        if not self._ema_deltas or len(self._ema_deltas) != len(params):
            self._ema_deltas = [None] * len(params)

        with torch.no_grad():
            for i, (p, p_before) in enumerate(zip(params, before_params)):
                delta = p.data - p_before
                prev = self._ema_deltas[i]
                if prev is None:
                    ema_delta = delta.clone()
                else:
                    ema_delta = beta * prev + (1.0 - beta) * delta
                p.data.copy_(p_before + ema_delta)
                self._ema_deltas[i] = ema_delta.detach().clone()

    def _update_plateau_state(self, before_params: List[torch.Tensor]) -> float:
        """Track HP update magnitude and optionally freeze future outer updates."""
        with torch.no_grad():
            sq_sum = 0.0
            for p, p_before in zip(self.hyperparams.parameters(), before_params):
                d = p.data - p_before
                sq_sum += float(torch.sum(d * d).item())
        delta_norm = float(math.sqrt(max(0.0, sq_sum)))

        if not self.config.outer_plateau_detection_enabled:
            return delta_norm

        self._hp_delta_window.append(delta_norm)
        patience = max(1, int(self.config.outer_plateau_patience))
        eps = float(self.config.outer_plateau_delta_epsilon)
        eps = max(0.0, eps)
        if len(self._hp_delta_window) >= patience:
            tail = self._hp_delta_window[-patience:]
            if all(v <= eps for v in tail):
                self._hp_updates_frozen = True
        return delta_norm

    def _precondition_hypergradients_hutchinson(
        self,
        val_loss: torch.Tensor,
        hypergrads: List[torch.Tensor],
        params: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], float]:
        """Apply diagonal Hutchinson preconditioning via a flat-vector operator."""
        hvp_fn = self._build_hvp_fn(val_loss, params)

        if not self.config.outer_hutchpp_diagonal_precondition_enabled:
            grad_norm = float(sum(g.norm().item() for g in hypergrads if g is not None))
            return hypergrads, grad_norm

        # ADDS: single canonical diagonal preconditioner through core.outer_precondition.
        # REMOVES: local scalar-trace helper and per-parameter diagonal estimator methods.
        flat_grads = self._flatten_tensor_list(hypergrads)
        dim = int(flat_grads.numel())

        def flat_h_fn(v_flat: torch.Tensor) -> torch.Tensor:
            v_list = self._unflatten_tensor_list(v_flat, params)
            hv_list = hvp_fn(v_list)
            return self._flatten_tensor_list(hv_list)

        precond_flat = outer_precondition(
            hypgrad=flat_grads,
            H_fn=flat_h_fn,
            dim=dim,
            n_probes=max(1, int(self.config.outer_hutchpp_samples)),
            eps=float(self.config.outer_hutchpp_eps),
        )
        precond_flat = torch.nan_to_num(precond_flat, nan=0.0, posinf=1.0, neginf=-1.0)
        precond = self._unflatten_tensor_list(precond_flat, hypergrads)
        grad_norm = float(sum(g.norm().item() for g in precond if g is not None))
        return precond, grad_norm

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
            # ADDS: explicit inner-loop L2 regularization coefficient from config.
            # REMOVES: implicit no-regularization call path for inner objective.
            l2_inner_reg=self.config.l2_inner_reg,
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
            # ADDS: explicit inner-loop L2 regularization coefficient from config.
            # REMOVES: implicit no-regularization call path for inner objective.
            l2_inner_reg=self.config.l2_inner_reg,
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
        hypergradients, recovered = self._sanitize_hypergradients(hypergradients)

        return {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "adapted_params": adapted_params,
            "hypergradients": hypergradients,
            "inner_sensitivity_debug": sensitivity_debug,
            "hybrid_shard": self.active_hybrid_shard,
            "nonfinite_hypergrad_recovered": recovered,
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
        progress_callback: Optional[Callable[[Dict[str, Any]], Optional[bool]]] = None,
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
                hp_before = [p.detach().clone() for p in self.hyperparams.parameters()]

                if config.outer_hutchpp_precondition_enabled or config.outer_sachpp_enabled:
                    state = self._evaluate_bilevel_state(
                        model,
                        train_fn,
                        val_fn,
                        train_data,
                        val_data,
                        outer_iter,
                    )
                    params = list(self.hyperparams.parameters())
                    eta = float(self.config.lbfgs_lr)
                    topology_drift = 0.0
                    curr_val_loss = float(state["val_loss"].detach().item())
                    if self.loss_history:
                        prev_val_loss = float(self.loss_history[-1])
                        topology_drift = abs(curr_val_loss - prev_val_loss) / (abs(prev_val_loss) + 1e-8)

                    if bool(config.outer_sachpp_enabled):
                        precond_grads, _ = self._precondition_hypergradients_sach(
                            state["val_loss"],
                            state["hypergradients"],
                            params,
                            topology_drift=topology_drift,
                        )
                    else:
                        precond_grads, _ = self._precondition_hypergradients_hutchinson(
                            state["val_loss"],
                            state["hypergradients"],
                            params,
                        )
                    sanitized_grads, recovered = self._sanitize_hypergradients(
                        precond_grads,
                        fallback=state["hypergradients"],
                    )
                    clipped_grads, _ = self._clip_hypergradients(sanitized_grads)
                    state["hypergradients"] = clipped_grads
                    if recovered:
                        state["nonfinite_hypergrad_recovered"] = True
                    if self.hutch_trace_history:
                        curr_trace = self.hutch_trace_history[-1]
                        state["hutch_trace"] = curr_trace
                        if len(self.hutch_trace_history) > 1:
                            prev_trace = self.hutch_trace_history[-2]
                            rel_drift = abs(curr_trace - prev_trace) / (abs(prev_trace) + 1e-8)
                            state["hutch_trace_drift_triggered"] = rel_drift > 0.2
                    with torch.no_grad():
                        for p, g in zip(params, state["hypergradients"]):
                            p.data.add_(g, alpha=-eta)
                    latest_state.update(state)
                else:
                    self._set_outer_lr_for_step(outer_iter)

                    if not config.lbfgs_reuse_history_across_outer:
                        self.outer_optimizer.clear_history()

                    def outer_closure():
                        latest_state.clear()
                        for p in self.hyperparams.parameters():
                            p.grad = None

                        local_state = self._evaluate_bilevel_state(
                            model,
                            train_fn,
                            val_fn,
                            train_data,
                            val_data,
                            outer_iter,
                        )
                        latest_state.update(local_state)

                        grads_to_apply, _ = self._clip_hypergradients(local_state["hypergradients"])
                        latest_state["hypergradients"] = grads_to_apply
                        for param, grad in zip(self.hyperparams.parameters(), grads_to_apply):
                            param.grad = grad

                        return local_state["val_loss"]

                    if self._hp_updates_frozen:
                        frozen_state = self._evaluate_bilevel_state(
                            model,
                            train_fn,
                            val_fn,
                            train_data,
                            val_data,
                            outer_iter,
                        )
                        clipped_hg, _ = self._clip_hypergradients(frozen_state["hypergradients"])
                        frozen_state["hypergradients"] = clipped_hg
                        latest_state.update(frozen_state)
                    else:
                        self.outer_optimizer.step(outer_closure)
                        self._apply_hp_ema_damping(hp_before)

                hp_delta_norm = self._update_plateau_state(hp_before)

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
                            f"lr={outer_state['lr']:.6f} wd={outer_state['wd']:.6f} "
                            f"|hg|={grad_mag:.3e} dHP={hp_delta_norm:.3e} frozen={int(self._hp_updates_frozen)}"
                        )

                if progress_callback is not None:
                    stop_requested = bool(progress_callback(
                        {
                            "iteration": outer_iter + 1,
                            "total_iterations": config.outer_steps,
                            "train_loss": train_loss_val,
                            "val_loss": loss_val,
                            "best_loss": self.best_loss,
                            "grad_magnitude": grad_mag,
                            "hp_delta_norm": hp_delta_norm,
                            "hp_updates_frozen": self._hp_updates_frozen,
                            "hyperparams": hp_dict,
                            "sensitivity_debug": sensitivity_debug,
                            "hybrid_shard": hybrid_shard,
                        }
                    ))
                    if stop_requested:
                        break

                # If plateau logic froze HP updates, terminate outer loop early.
                if self._hp_updates_frozen:
                    break

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
            "hp_delta_history": self._hp_delta_window,
            "hp_updates_frozen": self._hp_updates_frozen,
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
