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
    in_warmup: bool = True
    warmup_target_steps: int = 0
    warmup_seen_steps: int = 0
    warmup_refresh_remaining: int = 0
    topology_edges: int = 0


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
        self._topology_state: Dict[str, Dict[str, Any]] = {}

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
            damping=self.config.lbfgs_damping,
            damping_eps=self.config.lbfgs_damping_eps,
            curvature_threshold=self.config.curvature_threshold,
            secant_topology_enabled=self.config.inner_secant_topology_enabled,
            secant_topology_warmup_steps=self.config.inner_secant_warmup_steps,
            secant_topology_top_k=self.config.inner_secant_top_k,
            secant_topology_percentile=self.config.inner_secant_percentile,
            secant_symmetrize_enabled=self.config.inner_secant_symmetrize_enabled,
            secant_symmetry_assert_enabled=self.config.inner_secant_symmetry_assert_enabled,
            use_spectral_scaler=self.config.lbfgs_use_spectral_scaler,
            spectral_mu=self.config.lbfgs_spectral_mu,
        )
        legacy_topology_enabled = (
            self.config.auto_topology_enabled and not self.config.inner_secant_topology_enabled
        )
        self.layer_optimizers[name] = optimizer
        self.layer_states[name] = LayerState(
            name=name,
            memory_size=self.config.lbfgs_memory_base,
            in_warmup=legacy_topology_enabled,
            warmup_target_steps=self.config.auto_topology_warmup_steps,
        )
        self._topology_state[name] = {
            "current_step": 0,
            "warmup_count": 0,
            "running_mean": None,
            "running_m2": None,
            "edge_scores": {},
            "edge_weights": {},
            "sketch": None,
            "num_params": sum(p.numel() for p in params),
        }

    def register_callback(self, callback: Callable):
        """Register a callback invoked after each layer step (for dashboard)."""
        self._callbacks.append(callback)

    def _init_sparse_sketch(self, layer_name: str, device: torch.device):
        topo = self._topology_state[layer_name]
        if topo["sketch"] is not None:
            return

        n = topo["num_params"]
        d = max(4, min(self.config.auto_topology_sketch_dim, n))
        nnz_per_row = max(1, min(self.config.auto_topology_nnz_per_row, n))

        rows = []
        cols = []
        vals = []
        for r in range(d):
            idx = torch.randperm(n)[:nnz_per_row]
            rows.append(torch.full((nnz_per_row,), r, dtype=torch.long))
            cols.append(idx.long())
            vals.append(torch.randn(nnz_per_row) / max(1.0, float(nnz_per_row) ** 0.5))

        row_idx = torch.cat(rows)
        col_idx = torch.cat(cols)
        values = torch.cat(vals).to(torch.float32)
        indices = torch.stack([row_idx, col_idx], dim=0)
        topo["sketch"] = torch.sparse_coo_tensor(indices, values, size=(d, n), device=device).coalesce()

    def _update_topology_from_grad(self, layer_name: str, flat_grad: torch.Tensor):
        topo = self._topology_state[layer_name]
        if topo.get("num_params", 0) != int(flat_grad.numel()):
            topo["num_params"] = int(flat_grad.numel())
            topo["sketch"] = None
        self._init_sparse_sketch(layer_name, flat_grad.device)
        S = topo["sketch"]

        if S.size(1) != flat_grad.numel():
            topo["num_params"] = int(flat_grad.numel())
            topo["sketch"] = None
            self._init_sparse_sketch(layer_name, flat_grad.device)
            S = topo["sketch"]

        g_hat = torch.sparse.mm(S, flat_grad.view(-1, 1)).view(-1)
        count = topo["warmup_count"] + 1
        topo["warmup_count"] = count

        mean = topo["running_mean"]
        m2 = topo["running_m2"]
        if mean is None:
            mean = torch.zeros_like(g_hat)
            m2 = torch.zeros((g_hat.numel(), g_hat.numel()), device=g_hat.device, dtype=g_hat.dtype)

        delta = g_hat - mean
        mean = mean + delta / count
        delta2 = g_hat - mean
        m2 = m2 + torch.outer(delta, delta2)

        topo["running_mean"] = mean
        topo["running_m2"] = m2

        active = max(4, min(self.config.auto_topology_active_coords, flat_grad.numel()))
        abs_g = flat_grad.abs()
        idx = torch.topk(abs_g, k=active).indices.tolist()
        score = topo["edge_scores"]
        for i in range(len(idx)):
            ii = int(idx[i])
            vi = float(abs_g[ii].item())
            for j in range(i + 1, len(idx)):
                jj = int(idx[j])
                vj = float(abs_g[jj].item())
                a, b = (ii, jj) if ii < jj else (jj, ii)
                score[(a, b)] = score.get((a, b), 0.0) + (vi * vj)

    def _finalize_topology_mask(self, layer_name: str, beta: Optional[float] = None):
        topo = self._topology_state[layer_name]
        state = self.layer_states[layer_name]
        opt = self.layer_optimizers[layer_name]

        edges_scores = topo["edge_scores"]
        if not edges_scores:
            opt.set_topology_mask(None, topo["num_params"])
            state.topology_edges = 0
            return

        keys = list(edges_scores.keys())
        vals = torch.tensor([edges_scores[k] for k in keys], dtype=torch.float32)

        if topo["warmup_count"] > 1 and topo["running_m2"] is not None:
            cov = topo["running_m2"] / max(1, topo["warmup_count"] - 1)
            cov_vals = cov.abs().flatten()
            tau = torch.quantile(
                cov_vals,
                q=max(0.0, min(1.0, self.config.auto_topology_edge_top_percentile / 100.0)),
            )
            scale = float(torch.clamp(tau, min=1e-8).item())
            vals = vals / scale

        prev = topo["edge_weights"]
        if beta is not None and prev:
            merged = dict(prev)
            for k, v in zip(keys, vals.tolist()):
                merged[k] = beta * merged.get(k, 0.0) + (1.0 - beta) * float(v)
            edge_weights = merged
        else:
            edge_weights = {k: float(v.item()) for k, v in zip(keys, vals)}

        sparse_threshold = self.config.edrt_sparse_threshold if beta is not None else 0.0
        edge_weights = {k: v for k, v in edge_weights.items() if v >= sparse_threshold}

        if self.config.auto_topology_edge_budget is not None and len(edge_weights) > self.config.auto_topology_edge_budget:
            sorted_items = sorted(edge_weights.items(), key=lambda item: item[1], reverse=True)
            edge_weights = dict(sorted_items[: self.config.auto_topology_edge_budget])

        topo["edge_weights"] = edge_weights
        topo["edge_scores"] = {}
        topo["warmup_count"] = 0
        topo["running_mean"] = None
        topo["running_m2"] = None

        if not edge_weights:
            opt.set_topology_mask(None, topo["num_params"])
            state.topology_edges = 0
            return

        edge_list = list(edge_weights.keys())
        edge_idx = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        opt.set_topology_mask(edge_idx, topo["num_params"])
        state.topology_edges = edge_idx.size(1)

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
        topo = self._topology_state[layer_name]

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

        # ── Autonomous Discovery / EDRT Collection Phase ───────────
        if state.in_warmup:
            loss = closure()
            flat_grad = opt._gather_flat_grad()
            self._update_topology_from_grad(layer_name, flat_grad)

            # First-order warmup update without adding L-BFGS curvature history.
            opt._add_update(-self.config.lbfgs_lr, flat_grad)

            topo["current_step"] += 1
            state.warmup_seen_steps += 1

            warmup_done = False
            if state.warmup_refresh_remaining > 0:
                state.warmup_refresh_remaining -= 1
                warmup_done = state.warmup_refresh_remaining == 0
            else:
                warmup_done = state.warmup_seen_steps >= max(1, state.warmup_target_steps)

            if warmup_done:
                beta = self.config.edrt_beta if state.iteration > 0 else None
                self._finalize_topology_mask(layer_name, beta=beta)
                state.in_warmup = False
                state.warmup_seen_steps = 0

            grad_norm = flat_grad.norm().item()
            state.grad_norm = grad_norm
            state.grad_norm_history.append(grad_norm)
            state.iteration += 1

            ys = 0.0
            state.secant_value = ys
            state.secant_history.append(ys)
            state.landscape_status = "Topology Warmup"

            if loss is not None:
                state.loss_history.append(
                    loss.item() if isinstance(loss, torch.Tensor) else loss
                )

            result = {
                "layer": layer_name,
                "loss": loss.item() if isinstance(loss, torch.Tensor) else loss,
                "grad_norm": grad_norm,
                "kappa": state.kappa,
                "memory_size": state.memory_size,
                "secant": ys,
                "landscape": state.landscape_status,
                "evasion": False,
                "topology_edges": state.topology_edges,
            }
            for cb in self._callbacks:
                cb(result)

            return result

        # ── L-BFGS Step ─────────────────────────────────────────────
        loss = opt.step(closure)
        topo["current_step"] += 1

        # Trigger EDRT mini-warmup periodically.
        if (
            self.config.edrt_enabled
            and self.config.auto_topology_enabled
            and topo["current_step"] % max(1, self.config.edrt_refresh_interval) == 0
        ):
            state.in_warmup = True
            state.warmup_refresh_remaining = max(1, self.config.edrt_mini_warmup)

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
                "topology_edges": state.topology_edges,
            }
        return data

    @property
    def total_iterations(self) -> int:
        """Total iterations across all layers."""
        return sum(s.iteration for s in self.layer_states.values())
