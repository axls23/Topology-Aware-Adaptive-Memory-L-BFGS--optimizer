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
from typing import Dict, List, Optional, Callable, Any, Tuple
from collections import defaultdict
from torch import Tensor

from .baseline_lbfgs import FullBatchLBFGS, is_legal
from ..config import TaLBFGSConfig
from ..topology.attention_topo import AttentionTopologyBuilder
from ..topology.chain_topo import ChainTopologyController
from ..topology.moe_topo import MoETopologyBuilder
from ..topology.residual_topo import ResidualTopologyBuilder
from ..utils.kfac import KFACEmbedding


PARAM_GROUP_TYPES = ("rope", "embedding", "layernorm", "lora", "standard")
FROZEN_GROUPS = ("rope", "embedding")


def classify_param_group(name: str, param: Tensor, model) -> str:
    """Route each parameter to the correct curvature handler."""
    n = name.lower()
    if any(x in n for x in ("rope", "rotary", "freq")):
        return "rope"
    if any(x in n for x in ("embed", "lm_head", "wte")):
        return "embedding"
    if any(x in n for x in ("norm", "layernorm", "ln_")):
        return "layernorm"
    if any(x in n for x in ("lora_a", "lora_b", "lora_")):
        return "lora"
    return "standard"


def should_freeze_in_inner_loop(param_group_type: str) -> bool:
    """Rope and embedding are outer-loop only. Never update in inner loop."""
    return param_group_type in FROZEN_GROUPS


class AdamDiagPreconditioner:
    """For LayerNorm gamma/beta: exact diagonal Hessian proxy with EMA moments."""

    def __init__(self, eps: float = 1e-8, beta2: float = 0.999):
        self.v: Optional[Tensor] = None
        self.eps = eps
        self.beta2 = beta2

    def step(self, grad: Tensor) -> Tensor:
        if self.v is None:
            self.v = torch.zeros_like(grad)
        self.v.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
        return grad / (self.v.sqrt() + self.eps)


# ADDS: module-level validity gate for tests and shared pair checking semantics.
# REMOVES: dependence on simple y^T s > 0 acceptance in curvature pairing logic.
def _is_valid_pair(s: torch.Tensor, y: torch.Tensor, eps_rel: float = 0.01) -> bool:
    dot = (y @ s).item()
    return dot > eps_rel * y.norm().item() * s.norm().item()


@dataclass
class LayerState:
    """Per-layer optimizer state tracked for dashboard visualization."""

    name: str
    kappa: float = 1.0                # condition number
    memory_size: int = 5              # current m_l
    grad_norm: float = 0.0            # ||∇||
    secant_value: float = 1.0         # y_k^T s_k
    landscape_status: str = "Unknown" # Convex Bowl / Narrow Ravine / Saddle Point
    morse_index: int = 0              # number of negative eigenvalues
    topology_euler_char: int = 0      # Betti number / Euler connection
    kappa_history: List[float] = field(default_factory=list)
    morse_history: List[int] = field(default_factory=list)
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
    topology_segment: str = "reasoning"
    topology_hessian_strategy: str = "block_diag"


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

    def __init__(self, config: TaLBFGSConfig, model: Optional[nn.Module] = None):
        self.config = config
        self.model = model
        self.layer_optimizers: Dict[str, FullBatchLBFGS] = {}
        self.layer_states: Dict[str, LayerState] = {}
        self.evasion_log: List[Dict[str, Any]] = []
        self._callbacks: List[Callable] = []
        self._topology_state: Dict[str, Dict[str, Any]] = {}
        self._layer_group_type: Dict[str, str] = {}
        self._kfac_embed: Dict[str, KFACEmbedding] = {}
        self._layer_index: Dict[str, int] = {}
        self._moe_topology: Dict[str, MoETopologyBuilder] = {}
        self.attention_topology = AttentionTopologyBuilder(
            model=model,
            window_size=max(1, int(getattr(config, "gradient_window_size", 10))),
            warmup_steps=max(1, int(getattr(config, "auto_topology_warmup_steps", 50))),
        )
        self.chain_topology = ChainTopologyController(
            pivot_sigma=float(getattr(config, "diagnostics_spike_threshold", 3.0))
        )
        self.residual_topology = ResidualTopologyBuilder(
            n_layers=max(1, int(getattr(config, "n_layers", 1))),
            threshold=float(getattr(config, "distance_threshold", 0.05)),
        )
        self._pending_topology_snapshot: Optional[Dict[str, Any]] = None
        self._last_grad_norm: float = 0.0

    @staticmethod
    def _infer_layer_index(name: str) -> int:
        parts = name.split(".")
        for i, part in enumerate(parts[:-1]):
            if part in {"layers", "h", "blocks"} and parts[i + 1].isdigit():
                return int(parts[i + 1])
        for part in parts:
            if part.isdigit():
                return int(part)
        return 0

    def register_layer(self, name: str, params: List[nn.Parameter]):
        """
        Register a model layer block for independent optimization.

        Creates a dedicated FullBatchLBFGS instance with the configured
        base memory size.

        Args:
            name: Layer identifier (e.g., 'layers.0', 'layers.1').
            params: List of nn.Parameter tensors for this layer.
        """
        group_type = classify_param_group(name, params[0].data if params else torch.empty(0), self.model)
        self._layer_group_type[name] = group_type
        self._layer_index[name] = self._infer_layer_index(name)
        history_size = self.config.lbfgs_memory_max if group_type == "embedding" else self.config.lbfgs_memory_base

        optimizer = FullBatchLBFGS(
            params,
            lr=self.config.lbfgs_lr,
            history_size=history_size,
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
        if group_type == "embedding":
            legacy_topology_enabled = False
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

        if group_type == "embedding" and params:
            p0 = params[0].data
            if p0.dim() == 2:
                self._kfac_embed[name] = KFACEmbedding(vocab_size=p0.shape[0], embed_dim=p0.shape[1])

    def _embedding_kfac_step(self, layer_name: str, closure: Callable) -> Dict[str, Any]:
        opt = self.layer_optimizers[layer_name]
        state = self.layer_states[layer_name]

        for p in opt._params:
            if p.grad is not None:
                p.grad.zero_()
        loss = closure()

        grad_norm = 0.0
        for p in opt._params:
            if p.grad is not None:
                grad_norm += float(p.grad.norm().item())

        precond_norm = grad_norm
        kfac = self._kfac_embed.get(layer_name)
        if kfac is not None and opt._params and opt._params[0].grad is not None:
            p0 = opt._params[0]
            if p0.data.dim() == 2 and p0.grad.dim() == 2:
                embed_in = p0.data
                grad_out = p0.grad
                kfac.update(embed_in, grad_out)
                pre = kfac.inverse_precondition(grad_out)
                p0.data.add_(pre, alpha=-self.config.lbfgs_lr)
                precond_norm = float(pre.norm().item())
                for p in opt._params[1:]:
                    if p.grad is not None:
                        p.data.add_(p.grad, alpha=-self.config.lbfgs_lr)

        state.grad_norm = precond_norm
        state.grad_norm_history.append(precond_norm)
        state.iteration += 1
        state.secant_value = 0.0
        state.secant_history.append(0.0)
        state.landscape_status = "Embedding-KFAC"
        topo_meta = self._update_axis_topologies(
            layer_name=layer_name,
            opt=opt,
            state=state,
            loss_value=loss.item() if isinstance(loss, torch.Tensor) else None,
        )

        result = {
            "layer": layer_name,
            "loss": loss.item() if isinstance(loss, torch.Tensor) else loss,
            "grad_norm": precond_norm,
            "kappa": state.kappa,
            "memory_size": state.memory_size,
            "secant": 0.0,
            "landscape": state.landscape_status,
            "evasion": False,
            "topology": topo_meta,
        }
        for cb in self._callbacks:
            cb(result)
        return result

    def register_callback(self, callback: Callable):
        """Register a callback invoked after each layer step (for dashboard)."""
        self._callbacks.append(callback)

    def ingest_topology_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Queue the latest HF-derived topology snapshot for next layer update."""
        self._pending_topology_snapshot = snapshot

    def _update_topology_from_snapshot(self, snap: Dict[str, Any], default_grad_norm: float) -> None:
        """Single topology entry point fed by one intercepted model output."""
        attn_weights = snap.get("attn_weights")
        if attn_weights is not None:
            for layer_idx, layer_attn in enumerate(attn_weights):
                if layer_attn is None:
                    continue
                attn = layer_attn.detach()
                if attn.dim() == 4:
                    mean_attn = attn.mean(dim=0)
                elif attn.dim() == 3:
                    mean_attn = attn
                else:
                    continue

                for head_idx in range(int(mean_attn.shape[0])):
                    head_mat = mean_attn[head_idx]
                    self.attention_topology.classify_head(layer_idx, head_idx, head_mat)
                    col = head_mat.sum(dim=0)
                    std = head_mat.std(dim=0)
                    self.attention_topology.accumulate_secant(layer_idx, head_idx, "attn", col, std)
        else:
            kv_key_norms = snap.get("kv_key_norms")
            if kv_key_norms is not None:
                for layer_idx, head_norms in enumerate(kv_key_norms):
                    for head_idx, norm in enumerate(head_norms):
                        self.attention_topology.update_kappa_proxy(layer_idx, head_idx, float(norm))

        layer_name_by_idx = {idx: name for name, idx in self._layer_index.items()}
        expert_topk_indices = snap.get("expert_topk_indices")
        if expert_topk_indices is not None:
            for layer_idx, idx_tensor in enumerate(expert_topk_indices):
                layer_name = layer_name_by_idx.get(layer_idx)
                if layer_name is None:
                    continue
                if layer_name not in self._moe_topology:
                    self._moe_topology[layer_name] = MoETopologyBuilder(
                        n_experts=max(1, int(getattr(self.config, "n_experts", 8))),
                        top_k=max(1, int(getattr(self.config, "moe_top_k", 2))),
                        m_max=int(self.config.lbfgs_memory_max),
                        ttl_expire=max(1, int(getattr(self.config, "edrt_refresh_interval", 50))),
                    )
                moe = self._moe_topology[layer_name]
                active = idx_tensor.detach().reshape(-1).unique().tolist()
                active = [int(v) for v in active if 0 <= int(v) < moe.n_experts]
                moe.on_forward(active)
                moe.expire_stale()

        drift = snap.get("layer_norm_drift")
        if drift is not None:
            self.residual_topology.update_from_drift([float(v) for v in drift])

        grad_norm = snap.get("grad_norm")
        if grad_norm is None:
            grad_norm = default_grad_norm if default_grad_norm > 0 else self._last_grad_norm
        grad_norm = float(grad_norm)
        self._last_grad_norm = grad_norm
        self.chain_topology.on_outer_step_with_snapshot(snap, grad_norm)

    def _latest_secant_pair(self, opt: FullBatchLBFGS) -> Optional[Tuple[Tensor, Tensor]]:
        state = opt.state.get("global_state", {})
        old_stps = state.get("old_stps", [])
        old_dirs = state.get("old_dirs", [])
        if not old_stps or not old_dirs:
            return None
        s = old_stps[-1].detach().reshape(-1)
        y = old_dirs[-1].detach().reshape(-1)
        if s.numel() == 0 or y.numel() == 0 or s.numel() != y.numel():
            return None
        return s, y

    def _extract_active_experts(self, layer_name: str) -> List[int]:
        if self.model is None:
            return []

        layer_idx = self._layer_index.get(layer_name, 0)
        candidates = [
            getattr(self.model, "active_experts", None),
            getattr(self.model, "last_active_experts", None),
            getattr(self.model, "moe_active_experts", None),
        ]
        for cand in candidates:
            if cand is None:
                continue
            if isinstance(cand, dict):
                vals = cand.get(layer_name, cand.get(layer_idx, []))
            else:
                vals = cand
            if isinstance(vals, (list, tuple)):
                return [int(v) for v in vals if isinstance(v, (int, float))]
        return []

    def _apply_chain_memory_scale(self, state: LayerState, opt: FullBatchLBFGS) -> None:
        scale = float(self.chain_topology.window_scale())
        if scale <= 0:
            return
        new_m = int(round(state.memory_size * scale))
        new_m = max(int(self.config.lbfgs_memory_min), min(int(self.config.lbfgs_memory_max), new_m))
        if new_m != state.memory_size:
            opt.resize_history(new_m)
            state.memory_size = new_m

    def _update_axis_topologies(
        self,
        layer_name: str,
        opt: FullBatchLBFGS,
        state: LayerState,
        loss_value: Optional[float],
    ) -> Dict[str, Any]:
        from ..topology.condition import estimate_condition_and_subspace
        from ..topology.saddle import detect_topology_break
        
        layer_idx = self._layer_index.get(layer_name, 0)
        secant_pair = self._latest_secant_pair(opt)

        global_state = opt.state.get("global_state", {})
        old_dirs = global_state.get("old_dirs", [])
        if len(old_dirs) >= 4:
            # Compute True TDA: Euler characteristic on the PCA active subspace
            matrix = torch.stack(old_dirs).view(len(old_dirs), -1)
            sketch_dim = int(getattr(self.config, "auto_topology_sketch_dim", 8))
            kappa, active_subspace = estimate_condition_and_subspace(matrix, n_components=sketch_dim)
            if active_subspace is not None:
                topo_break = detect_topology_break(
                    recent_gradients=matrix,
                    distance_threshold=float(getattr(self.config, "distance_threshold", 0.05)),
                    prev_chi=state.topology_euler_char,
                    active_subspace=active_subspace,
                )
                state.topology_euler_char = topo_break.get("euler_characteristic", state.topology_euler_char)

        if self._pending_topology_snapshot is not None:
            self._update_topology_from_snapshot(self._pending_topology_snapshot, float(state.grad_norm))
            self._pending_topology_snapshot = None

        if secant_pair is not None:
            s, y = secant_pair
            self.attention_topology.accumulate_secant(layer_idx, 0, "q", s, y)
            if self.attention_topology.should_rederive(val_loss=loss_value):
                mask = self.attention_topology.derive_mask(layer_idx, 0, "q")
                if mask is not None:
                    density = float(mask.float().mean().item())
                    self.attention_topology.head_type[(layer_idx, 0)] = "local" if density < 0.2 else "global"

        attention_strategy = self.attention_topology.hessian_strategy(layer_idx, 0)

        self._last_grad_norm = float(state.grad_norm)
        self.chain_topology.on_outer_step(float(state.grad_norm), prm_score=None)
        state.topology_segment = self.chain_topology.current_segment
        self._apply_chain_memory_scale(state, opt)

        residual_strategy = self.residual_topology.hessian_strategy(layer_idx)
        state.topology_hessian_strategy = residual_strategy

        active_experts = self._extract_active_experts(layer_name)
        if active_experts or "moe" in layer_name.lower() or "expert" in layer_name.lower():
            if layer_name not in self._moe_topology:
                self._moe_topology[layer_name] = MoETopologyBuilder(
                    n_experts=max(1, int(getattr(self.config, "n_experts", 8))),
                    top_k=max(1, int(getattr(self.config, "moe_top_k", 2))),
                    m_max=int(self.config.lbfgs_memory_max),
                    ttl_expire=max(1, int(getattr(self.config, "edrt_refresh_interval", 50))),
                )
            moe = self._moe_topology[layer_name]
            moe.on_forward(active_experts)
            if secant_pair is not None:
                s, y = secant_pair
                for e in active_experts:
                    if 0 <= int(e) < moe.n_experts:
                        moe.add_pair(int(e), s, y)
            moe.expire_stale()

        return {
            "attention_strategy": attention_strategy,
            "chain_segment": self.chain_topology.current_segment,
            "residual_strategy": residual_strategy,
            "moe_active_experts": active_experts,
        }

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
        if self._layer_group_type.get(layer_name) == "embedding":
            return self._embedding_kfac_step(layer_name, closure)

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
            result["topology"] = self._update_axis_topologies(
                layer_name=layer_name,
                opt=opt,
                state=state,
                loss_value=loss.item() if isinstance(loss, torch.Tensor) else None,
            )
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
        result["topology"] = self._update_axis_topologies(
            layer_name=layer_name,
            opt=opt,
            state=state,
            loss_value=loss.item() if isinstance(loss, torch.Tensor) else None,
        )
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

        if results:
            layer_outputs = [
                torch.tensor([float(self.layer_states[name].grad_norm)], dtype=torch.float32)
                for name in results
            ]
            self.residual_topology.probe_jacobian_norms(
                model=self.model,
                x_sample=None,
                layer_outputs=layer_outputs,
            )
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
        Inject an eigenvector-directed perturbation to escape saddle points.
        """
        from ..topology.saddle import escape_saddle, is_saddle_point

        # ADDS: Lanczos probe over two-loop recursion before perturbing parameters.
        # REMOVES: unconditional random orthogonal perturbation update path.
        two_loop_fn = lambda v: opt.two_loop_recursion(v)
        saddle, min_eigvec, morse_index = is_saddle_point(
            two_loop_fn,
            dim=int(flat_grad.numel()),
            eps=self.config.secant_threshold,
        )
        opt.state.setdefault("global_state", {})["morse_index"] = morse_index
        if not saddle:
            return
        grad_norm = float(flat_grad.norm().item())
        escape_saddle(
            params=list(opt._params),
            grad_norm=grad_norm,
            min_eigvec=min_eigvec.to(device=flat_grad.device, dtype=flat_grad.dtype),
            scale=self.config.perturbation_scale,
        )

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
                "morse_history": state.morse_history,
                "evasion_count": state.evasion_count,
                "iteration": state.iteration,
                "topology_edges": state.topology_edges,
                "topology_euler_char": state.topology_euler_char,
                "topology_segment": state.topology_segment,
                "topology_hessian_strategy": state.topology_hessian_strategy,
            }
        return data

    @property
    def total_iterations(self) -> int:
        """Total iterations across all layers."""
        return sum(s.iteration for s in self.layer_states.values())

class TaLBFGS(FullBatchLBFGS):
    """
    Topology-Aware L-BFGS Optimizer (Single Group).
    
    A standard PyTorch Optimizer API that enables topological 
    features out-of-the-box for a flat parameter list.
    """
    def __init__(self, params, lr=1.0, **kwargs):
        kwargs.setdefault("history_size", 20)
        kwargs.setdefault("secant_topology_enabled", True)
        kwargs.setdefault("damping", True)
        super().__init__(params, lr=lr, **kwargs)
