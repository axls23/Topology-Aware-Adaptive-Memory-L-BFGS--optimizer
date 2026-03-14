"""
Layerwise Topology-Aware Adaptive-Memory L-BFGS (ta-LBFGS).

Wraps the baseline FullBatchLBFGS to provide:
- Block-diagonal Hessian approximation (independent per-layer state)
- Adaptive memory sizing via condition number (κ → m_l)
- Saddle-point evasion via secant condition monitoring
- Integration hooks for the Rich CLI dashboard
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from collections import defaultdict

from .baseline_lbfgs import FullBatchLBFGS, is_legal
from ..config import TaLBFGSConfig


@dataclass
class LayerState:
    """Per-layer optimizer state tracked for dashboard visualization."""

    name: str
    kappa: float = 1.0                # condition number
    memory_size: int = 5              # current m_l
    grad_norm: float = 0.0            # ||∇||
    secant_value: float = 1.0         # y_k^T s_k
    landscape_status: str = "Unknown" # Convex Bowl / Narrow Ravine / Saddle Point
    kappa_history: List[float] = field(default_factory=list)
    grad_norm_history: List[float] = field(default_factory=list)
    secant_history: List[float] = field(default_factory=list)
    loss_history: List[float] = field(default_factory=list)
    evasion_count: int = 0
    iteration: int = 0


class LayerwiseTaLBFGS:
    """
    Layerwise Topology-Aware Adaptive-Memory L-BFGS Optimizer.

    Manages independent L-BFGS instances for each model layer block,
    implementing the block-diagonal Hessian approximation. Each layer's
    memory window dynamically expands/contracts based on its local
    condition number.

    This is NOT a torch.optim.Optimizer subclass — it orchestrates
    multiple FullBatchLBFGS instances and integrates topology analysis.

    Args:
        config: TaLBFGSConfig instance.
    """

    def __init__(self, config: TaLBFGSConfig):
        self.config = config
        self.layer_optimizers: Dict[str, FullBatchLBFGS] = {}
        self.layer_states: Dict[str, LayerState] = {}
        self.evasion_log: List[Dict[str, Any]] = []
        self._callbacks: List[Callable] = []

    def register_layer(self, name: str, params: List[nn.Parameter]):
        """
        Register a model layer block for independent optimization.

        Creates a dedicated FullBatchLBFGS instance with the configured
        base memory size.

        Args:
            name: Layer identifier (e.g., 'layers.0', 'layers.1').
            params: List of nn.Parameter tensors for this layer.
        """
        optimizer = FullBatchLBFGS(
            params,
            lr=self.config.lbfgs_lr,
            history_size=self.config.lbfgs_memory_base,
            line_search=self.config.lbfgs_line_search,
        )
        self.layer_optimizers[name] = optimizer
        self.layer_states[name] = LayerState(
            name=name,
            memory_size=self.config.lbfgs_memory_base,
        )

    def register_callback(self, callback: Callable):
        """Register a callback invoked after each layer step (for dashboard)."""
        self._callbacks.append(callback)

    def step_layer(
        self,
        layer_name: str,
        closure: Callable,
        kappa: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Perform one optimization step for a specific layer.

        Integrates adaptive memory resizing and saddle detection.

        Args:
            layer_name: Name of the layer to step.
            closure: Closure that computes and returns the loss.
            kappa: Pre-computed condition number (if available).

        Returns:
            Dictionary with step results for dashboard consumption.
        """
        if layer_name not in self.layer_optimizers:
            raise KeyError(f"Layer '{layer_name}' not registered.")

        opt = self.layer_optimizers[layer_name]
        state = self.layer_states[layer_name]

        # ── Adaptive Memory Sizing ──────────────────────────────────
        if kappa is not None and self.config.adaptive_memory_enabled:
            from ..topology.adaptive_memory import compute_memory_size

            new_m = compute_memory_size(
                kappa,
                self.config.lbfgs_memory_base,
                self.config.lbfgs_memory_min,
                self.config.lbfgs_memory_max,
            )
            if new_m != state.memory_size:
                opt.resize_history(new_m)
                state.memory_size = new_m

            state.kappa = kappa
            state.kappa_history.append(kappa)

        # ── L-BFGS Step ─────────────────────────────────────────────
        loss = opt.step(closure)

        # ── Collect Metrics ─────────────────────────────────────────
        flat_grad = opt._gather_flat_grad()
        grad_norm = flat_grad.norm().item()
        state.grad_norm = grad_norm
        state.grad_norm_history.append(grad_norm)
        state.iteration += 1

        # ── Secant Condition Check ──────────────────────────────────
        global_state = opt.state["global_state"]
        ys = 0.0
        if len(global_state["old_dirs"]) > 0 and len(global_state["old_stps"]) > 0:
            y = global_state["old_dirs"][-1]
            s = global_state["old_stps"][-1]
            ys = y.dot(s).item()

        state.secant_value = ys
        state.secant_history.append(ys)

        # ── Landscape Classification ────────────────────────────────
        state.landscape_status = self._classify_landscape(state)

        # ── Saddle Evasion ──────────────────────────────────────────
        evasion_triggered = False
        if ys <= self.config.secant_threshold and state.iteration > 1:
            evasion_triggered = True
            self._inject_perturbation(opt, flat_grad)
            state.evasion_count += 1
            self.evasion_log.append({
                "iteration": state.iteration,
                "layer": layer_name,
                "ys": ys,
                "grad_norm": grad_norm,
                "kappa": state.kappa,
            })

        if loss is not None:
            state.loss_history.append(
                loss.item() if isinstance(loss, torch.Tensor) else loss
            )

        # ── Fire Callbacks ──────────────────────────────────────────
        result = {
            "layer": layer_name,
            "loss": loss.item() if isinstance(loss, torch.Tensor) else loss,
            "grad_norm": grad_norm,
            "kappa": state.kappa,
            "memory_size": state.memory_size,
            "secant": ys,
            "landscape": state.landscape_status,
            "evasion": evasion_triggered,
        }
        for cb in self._callbacks:
            cb(result)

        return result

    def step_all_layers(
        self,
        closures: Dict[str, Callable],
        kappas: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Step all registered layers.

        Args:
            closures: Dict mapping layer_name → closure.
            kappas: Dict mapping layer_name → condition number.

        Returns:
            Dict mapping layer_name → step results.
        """
        results = {}
        for name in self.layer_optimizers:
            kappa = kappas.get(name) if kappas else None
            closure = closures.get(name)
            if closure is not None:
                results[name] = self.step_layer(name, closure, kappa)
        return results

    def _classify_landscape(self, state: LayerState) -> str:
        """Classify the local landscape topology based on metrics."""
        kappa = state.kappa
        ys = state.secant_value
        grad_norm = state.grad_norm

        if ys <= 0:
            return "Saddle Point"
        elif kappa > 100:
            return "Narrow Ravine"
        elif kappa > 10:
            return "Ill-Conditioned"
        elif grad_norm < 1e-6:
            return "Converged"
        else:
            return "Convex Bowl"

    def _inject_perturbation(self, opt: FullBatchLBFGS, flat_grad: torch.Tensor):
        """
        Inject an orthogonal perturbation to escape saddle points.

        Generates a random vector orthogonal to the current gradient
        and applies a scaled perturbation to the parameters.
        """
        from ..topology.saddle import generate_orthogonal_perturbation

        perturbation = generate_orthogonal_perturbation(
            flat_grad, self.config.perturbation_scale
        )
        opt._add_update(1.0, perturbation)

    def get_all_layer_data(self) -> Dict[str, Dict]:
        """Get current state of all layers (for dashboard rendering)."""
        data = {}
        for name, state in self.layer_states.items():
            data[name] = {
                "kappa": state.kappa,
                "memory_size": state.memory_size,
                "grad_norm": state.grad_norm,
                "secant": state.secant_value,
                "landscape": state.landscape_status,
                "kappa_history": state.kappa_history,
                "grad_norm_history": state.grad_norm_history,
                "evasion_count": state.evasion_count,
                "iteration": state.iteration,
            }
        return data

    @property
    def total_iterations(self) -> int:
        """Total iterations across all layers."""
        return sum(s.iteration for s in self.layer_states.values())
