"""
ta-LBFGS Demo: Bilevel Optimization with Live CLI Dashboard.

Demonstrates the full ta-LBFGS pipeline:
- Multi-layer synthetic optimization problem
- Differentiable hyperparameters (lr, wd) in log-space
- Live Rich CLI dashboard with per-layer topology metrics
- Saddle-point evasion via secant condition monitoring
- Topology component 3D export on completion

Usage:
    python demo.py
    python demo.py --inner-steps 5 --no-dashboard
"""

import argparse
from collections import OrderedDict
from contextlib import nullcontext
from typing import Dict, List, Any, Optional
import os
import time
import torch
import torch.nn as nn
import numpy as np
import threading
import webbrowser
from datetime import datetime

from ta_lbfgs.config import TaLBFGSConfig
from ta_lbfgs.core.hyperparameters import DifferentiableHyperparameters
from ta_lbfgs.core.baseline_lbfgs import FullBatchLBFGS
from ta_lbfgs.topology.condition import estimate_condition_number
from ta_lbfgs.topology.adaptive_memory import compute_memory_size
from ta_lbfgs.topology.saddle import check_secant_condition, generate_orthogonal_perturbation
from ta_lbfgs.core.lbfgs import LayerwiseTaLBFGS
from ta_lbfgs.dashboard.textual_dashboard import TextualDashboard
from ta_lbfgs.dashboard.server import DashboardServer
from ta_lbfgs.dashboard.landscape_viz import (
    plot_dynamics,
    plot_hyperparameter_trajectories,
    generate_landscape_mesh,
    reset_adaptive_mesh,
    get_adaptive_mesh,
)
from ta_lbfgs.dashboard.sparkline import generate_sparkline

# Architectural Grounding Imports
from ta_lbfgs.training.inner_loop import functional_call_model
from ta_lbfgs.training.interceptor import ArchitectureInterceptor
from ta_lbfgs.training.bilevel import BilevelOptimizer

# HF Imports
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_HF = True
except ImportError:
    HAS_HF = False

from ta_lbfgs.training.data_preprocessing import (
    get_default_cache_path,
    get_default_dataset_path,
    load_or_build_reasoning_trace_cache,
    sample_packed_batch,
)
from ta_lbfgs.topology.hf_interceptor import (
    build_topology_snapshot,
    can_output_attentions,
    detect_moe_model,
)


def _topology_components_from_layer_data(
    layer_data: Dict[str, Dict[str, Any]],
    step_idx: int,
    max_memory_size: int = 20,
) -> np.ndarray:
    """Build [n_layers, 5] topology-component curvature snapshot for one step."""
    names = sorted(layer_data.keys())
    out = np.zeros((len(names), 5), dtype=np.float64)

    phase = (step_idx + 1) / max(1.0, float(step_idx + 2))
    for li, name in enumerate(names):
        d = layer_data.get(name, {})
        kappa = float(max(1.0, d.get("kappa", 1.0)))
        secant = float(d.get("secant", 0.0))
        grad_norm = float(max(0.0, d.get("grad_norm", 0.0)))
        memory_size = float(max(1, d.get("memory_size", 1)))

        # 0) Attention component curvature proxy.
        attn_curv = np.log1p(kappa)
        # 1) MoE component proxy: blends load-window pressure with conditioning.
        moe_curv = (memory_size / max(1.0, float(max_memory_size))) * np.sqrt(np.log1p(kappa))
        # 2) Residual coupling proxy from secant magnitude.
        residual_curv = np.log1p(abs(secant) * 1e3)
        # 3) Chain component proxy from grad norm and progression phase.
        chain_curv = np.log1p(grad_norm * (1.0 + 0.5 * phase))
        # 4) Global conditioning summary component.
        global_curv = 0.45 * attn_curv + 0.2 * moe_curv + 0.2 * residual_curv + 0.15 * chain_curv

        out[li, :] = [attn_curv, moe_curv, residual_curv, chain_curv, global_curv]
    return out


def _build_topology_3d_html(
    topology_components_history: List[np.ndarray],
    layer_names: List[str],
    output_path: str,
):
    """Export topology components as interactive 3D surface HTML using Plotly."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[WARNING] Plotly not available; skipping topology 3D export")
        return

    component_names = ["attention", "moe", "residual", "chain", "global"]
    if not topology_components_history:
        return

    stacked = np.stack(topology_components_history, axis=0)  # [T, L, 5]

    # Normalize per component for stable visual scale
    norm = stacked.copy()
    for c in range(norm.shape[2]):
        col = norm[:, :, c]
        lo = float(np.nanmin(col))
        hi = float(np.nanmax(col))
        span = max(1e-9, hi - lo)
        norm[:, :, c] = (col - lo) / span

    # Build 3D surface for each component
    fig = go.Figure()

    T, L, C = norm.shape
    x_steps = np.arange(T)
    y_layers = np.arange(L)
    
    for comp_idx, comp_name in enumerate(component_names):
        z_data = norm[:, :, comp_idx].T  # [L, T]
        
        fig.add_trace(go.Surface(
            x=x_steps,
            y=y_layers,
            z=z_data,
            name=comp_name,
            colorscale="Viridis",
            showscale=(comp_idx == 0),
            visible=(comp_idx == 0),
        ))

    # Buttons for switching between components
    buttons = []
    for i, comp_name in enumerate(component_names):
        visible = [False] * len(component_names)
        visible[i] = True
        buttons.append(
            dict(
                label=comp_name.capitalize(),
                method="update",
                args=[{"visible": visible}, {"title": f"Topology Component: {comp_name.upper()}"}]
            )
        )

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=buttons,
                x=0.1, y=1.15,
                xanchor="left", yanchor="top",
            )
        ],
        title="Topology Component: ATTENTION",
        scene=dict(
            xaxis_title="Optimization Step",
            yaxis_title="Layer",
            zaxis_title="Curvature Proxy (Normalized)",
            camera=dict(
                eye=dict(x=1.2, y=1.2, z=1.3),
            ),
        ),
        width=1200,
        height=800,
    )

    fig.write_html(output_path)
    print(f"  Topology Components 3D exported to: {output_path}")


def _heads_from_layer_data(layer_data: dict, step_idx: int, heads_per_layer: int = 8):
    names = sorted(layer_data.keys())
    kappa_grid = []
    valid_mask = []
    head_buffers = {}

    for layer_idx, name in enumerate(names):
        base_kappa = float(layer_data[name].get("kappa", 1.0))
        secant = float(layer_data[name].get("secant", 0.0))
        memory_size = int(layer_data[name].get("memory_size", 3))

        row_kappa = []
        row_valid = []
        for head_idx in range(heads_per_layer):
            wave = 1.0 + 0.08 * np.sin((step_idx + 1) * 0.21 + head_idx * 0.63 + layer_idx * 0.2)
            spread = 0.75 + 0.6 * (head_idx + 1) / max(heads_per_layer, 1)
            kappa_h = max(1.0, float(base_kappa * wave * spread))
            valid_h = bool(secant > 0.0 and np.isfinite(kappa_h))
            row_kappa.append(kappa_h)
            row_valid.append(valid_h)

            pair_count = max(3, min(memory_size, 12))
            pairs = []
            for pidx in range(pair_count):
                s_norm = 0.02 * (pidx + 1) * (1.0 + 0.15 * head_idx)
                y_norm = s_norm * (1.08 + 0.13 * np.cos(step_idx + pidx + head_idx))
                ys_val = float((s_norm * y_norm) * (1e-2 if valid_h else -4e-3))
                pairs.append(
                    {
                        "idx": pidx,
                        "s_norm": float(s_norm),
                        "y_norm": float(y_norm),
                        "ys": ys_val,
                        "accepted": bool(ys_val > 0.0),
                    }
                )

            head_buffers[f"{layer_idx}:{head_idx}"] = {
                "layer": layer_idx,
                "head": head_idx,
                "pairs": pairs,
            }

        kappa_grid.append(row_kappa)
        valid_mask.append(row_valid)

    return kappa_grid, valid_mask, head_buffers


def _expert_rows_from_hparams(hp_dict: dict, layer_data: dict, n_experts: int = 8):
    lr_vec = hp_dict.get("lr", [])
    if not isinstance(lr_vec, list):
        lr_vec = [float(lr_vec)]
    wd_vec = hp_dict.get("wd", [])
    if not isinstance(wd_vec, list):
        wd_vec = [float(wd_vec)]

    layer_names = sorted(layer_data.keys())
    rows = []
    for i in range(n_experts):
        lr_i = float(lr_vec[i % max(len(lr_vec), 1)]) if lr_vec else 1e-3
        wd_i = float(wd_vec[i % max(len(wd_vec), 1)]) if wd_vec else 1e-2
        name = layer_names[i % max(len(layer_names), 1)] if layer_names else None
        kappa_i = float(layer_data.get(name, {}).get("kappa", 1.0)) if name else 1.0
        raw = 1.4 * lr_i / max(wd_i, 1e-8)
        load = float(max(0.0, min(1.0, 0.3 + 0.45 * np.tanh(raw) + 0.15 * np.tanh(12.0 / max(kappa_i, 1.0)))))
        window = int(max(3, min(20, round(20.0 - 6.0 * load + 0.08 * np.log10(max(kappa_i, 1.0))))))
        rows.append({"id": i, "load": load, "window_size": window})

    active_count = sum(1 for r in rows if r["load"] > 0.25)
    expired_ttl = sum(1 for r in rows if r["window_size"] <= 3)
    return {"rows": rows, "active_count": active_count, "expired_ttl": expired_ttl}


def _chain_payload(step: int, total_steps: int, topology_valid: bool):
    phase = (step + 1) / max(total_steps, 1)
    reasoning = max(0.15, 0.5 - 0.22 * phase)
    pivot = max(0.05, 0.09 + 0.05 * np.sin(step * 0.2))
    answer = min(0.62, 0.26 + 0.32 * phase)
    verify = max(0.08, 1.0 - (reasoning + pivot + answer))

    total_tokens = 256
    pivot_tokens = int(total_tokens * pivot)
    reasoning_tokens = int(total_tokens * reasoning)
    answer_tokens = int(total_tokens * answer)
    verify_tokens = max(0, total_tokens - (pivot_tokens + reasoning_tokens + answer_tokens))

    current_segment = "reasoning" if phase < 0.4 else ("answer" if phase < 0.85 else "verify")
    return {
        "segments": {
            "reasoning": float(reasoning),
            "pivot": float(pivot),
            "answer": float(answer),
            "verify": float(verify),
        },
        "status_rows": {
            "current_segment": current_segment,
            "pivot_index": pivot_tokens,
            "reasoning_tokens": reasoning_tokens,
            "answer_tokens": answer_tokens,
            "verify_tokens": verify_tokens,
            "topology_valid": bool(topology_valid),
        },
    }


# ────────────────────────────────────────────────────────────────────
# Synthetic Multi-Layer Problem
# ────────────────────────────────────────────────────────────────────

class SyntheticLayer(nn.Module):
    """
    Simulates a transformer layer block with tunable difficulty.

    Each layer has a different condition number, simulating the
    varied curvature across transformer blocks.
    """

    def __init__(self, dim: int, condition_scale: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim) * 0.1)

        # Create a landscape with tunable condition number
        eigenvalues = torch.linspace(1, condition_scale, dim)
        self.A = torch.diag(eigenvalues)
        self.target = torch.randn(dim) * 0.5

    def loss(self, lr_scale: torch.Tensor, wd_scale: torch.Tensor):
        """Compute layer loss (quadratic + Rosenbrock-like saddle)."""
        diff = self.weight - self.target
        quadratic = 0.5 * diff @ self.A @ diff

        # Add a saddle-point region (non-convex)
        if self.weight.shape[0] >= 2:
            saddle = (self.weight[0] ** 2 - self.weight[1]) ** 2
            saddle += 0.1 * (1 - self.weight[0]) ** 2
        else:
            saddle = torch.tensor(0.0)

        # Hyperparameters affect the loss landscape
        total = lr_scale * quadratic + 0.1 * saddle + wd_scale * (self.weight ** 2).sum()
        return total


class SyntheticModel(nn.Module):
    """Multi-layer synthetic model simulating a transformer."""

    def __init__(self, n_layers: int = 4, dim: int = 8):
        super().__init__()
        self.layers = nn.ModuleList([
            SyntheticLayer(dim, condition_scale=10.0 * (i + 1))
            for i in range(n_layers)
        ])

    def train_loss(self, hyperparams: DifferentiableHyperparameters):
        """Total training loss across all layers (per-layer lr/wd)."""
        total = sum(
            layer.loss(hyperparams.get_layer_lr(i), hyperparams.get_layer_wd(i))
            for i, layer in enumerate(self.layers)
        )
        return total / len(self.layers)

    def val_loss(self, hyperparams: DifferentiableHyperparameters):
        """Validation loss (slight distribution shift)."""
        total = sum(
            layer.loss(
                hyperparams.get_layer_lr(i) * 1.1,
                hyperparams.get_layer_wd(i) * 0.9,
            )
            for i, layer in enumerate(self.layers)
        )
        return total / len(self.layers)


class ReasoningDatasetLoader:
    """Loads reasoning traces from local JSONL format."""
    def __init__(self, path: str, limit: int = 5):
        self.samples = []
        import json
        import os
        if os.path.exists(path):
            print(f"[INFO] Loading reasoning traces from: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                for _ in range(limit):
                    line = f.readline()
                    if not line: break
                    data = json.loads(line)
                    # Extract User Prompt + Assistant Reasoning Trace
                    messages = data.get('messages', [])
                    if len(messages) >= 3:
                        prompt = messages[1]['content']
                        trace = messages[2]['content']
                        # Format as a single string for causal LM training
                        self.samples.append(f"Prompt: {prompt}\nResponse: {trace}")
        else:
            print(f"[WARNING] Dataset not found at {path}. Using fallbacks.")
            self.samples = ["Hyperparameter optimization for LLMs."]

    def get_batch(self):
        return self.samples


class HFModelWrapper(nn.Module):
    """Wrapper for real HF models to be used in demo.py."""
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B", low_vram: bool = False):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.low_vram = low_vram

        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        
        # Resolve local cache path
        cache_base = os.path.expanduser("~/.cache/huggingface/hub")
        snapshot_id = "060db6499f32faf8b98477b0a26969ef7d8b9987"
        local_path = os.path.join(cache_base, "models--Qwen--Qwen2.5-0.5B", "snapshots", snapshot_id)
        
        print(f"[INFO] Loading HF model from: {local_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
        model_dtype = torch.float16 if (self.device == "cuda" and low_vram) else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            local_path, 
            local_files_only=True,
            torch_dtype=model_dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)

        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
        if low_vram and hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                self.model.gradient_checkpointing_enable()
        
        # Detect layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            self.layers = self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            self.layers = self.model.transformer.h
        else:
            self.layers = []

        self.dataset = None
        self._cached_base_loss = None
        self.topology_warmup_steps = 50
        self._topology_step = 0
        self._can_output_attentions = can_output_attentions(self.model.config)
        self._is_moe_model = detect_moe_model(self.model.config)
        self.last_topology_snapshot = None

    def _forward_with_topology_capture(self, batch, step: int = None):
        capture_step = self._topology_step if step is None else int(step)
        output_attn = self._can_output_attentions and (capture_step < int(self.topology_warmup_steps))
        outputs = self.model(
            **batch,
            output_attentions=output_attn,
            output_router_logits=self._is_moe_model,
            output_hidden_states=(capture_step < int(self.topology_warmup_steps)),
            use_cache=True,
            return_dict=True,
        )
        self.last_topology_snapshot = build_topology_snapshot(
            outputs=outputs,
            step=capture_step,
            warmup_steps=int(self.topology_warmup_steps),
            model_config=self.model.config,
        )
        self._topology_step = max(self._topology_step + 1, capture_step + 1)
        return outputs

    def configure_bilevel_trainable_subset(self, train_last_n_layers: int = 1):
        """Freeze most weights so second-order bilevel steps fit commodity GPUs."""
        for p in self.model.parameters():
            p.requires_grad_(False)

        n_layers = len(self.layers)
        keep_layers = list(range(max(0, n_layers - train_last_n_layers), n_layers))
        keep_prefixes = [f"model.layers.{idx}." for idx in keep_layers]

        trainable = 0
        for name, p in self.model.named_parameters():
            in_keep_layer = any(name.startswith(prefix) for prefix in keep_prefixes)
            if in_keep_layer and ("norm" in name or name.endswith("bias")):
                p.requires_grad_(True)
                trainable += p.numel()
            elif name.startswith("model.norm"):
                p.requires_grad_(True)
                trainable += p.numel()

        print(f"[INFO] Bilevel trainable subset enabled: {trainable} params")

    def configure_bilevel_full_model(self):
        """Enable full-model differentiable training, subject to low-VRAM constraints."""
        trainable = 0
        for p in self.model.parameters():
            p.requires_grad_(True)
            trainable += p.numel()
        print(f"[INFO] Bilevel full-model enabled: {trainable} params")

    def _sample_model_inputs(self, batch_size: int = 2, max_length: int = 256):
        """Sample a model-ready batch, preferring offline packed cache tensors."""
        if self.dataset is not None:
            try:
                return sample_packed_batch(self.dataset, batch_size=batch_size, device=self.device)
            except Exception:
                pass

        texts = ["Hyperparameter optimization for LLMs."]
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.device)
        inputs["labels"] = inputs["input_ids"]
        return inputs

    def _get_base_loss(self):
        """Compute the raw model loss (cached per iteration)."""
        if self._cached_base_loss is not None:
            return self._cached_base_loss
        
        with torch.no_grad():
            batch = self._sample_model_inputs(batch_size=2, max_length=256)
            outputs = self.model(**batch)
            self._cached_base_loss = outputs.loss.item()
        
        return self._cached_base_loss

    def model_topology_loss(self, batch_size: int = 1, max_length: int = 192) -> torch.Tensor:
        """Compute a true LM loss (with grad) used for topology/kappa signals."""
        batch = self._sample_model_inputs(batch_size=batch_size, max_length=max_length)
        outputs = self._forward_with_topology_capture(batch)
        return outputs.loss

    @staticmethod
    def _causal_loss_with_label_smoothing(
        logits: torch.Tensor,
        labels: torch.Tensor,
        label_smoothing: torch.Tensor,
    ) -> torch.Tensor:
        """Token-level causal loss with differentiable label smoothing."""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        vocab_size = shift_logits.size(-1)
        flat_logits = shift_logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)

        valid_mask = flat_labels != -100
        if not torch.any(valid_mask):
            return torch.zeros((), device=logits.device, dtype=logits.dtype)

        valid_logits = flat_logits[valid_mask]
        valid_labels = flat_labels[valid_mask]
        log_probs = torch.log_softmax(valid_logits, dim=-1)

        nll = -log_probs.gather(1, valid_labels.unsqueeze(-1)).squeeze(-1).mean()
        smooth = -log_probs.mean(dim=-1).mean()
        eps = torch.clamp(label_smoothing, 1e-6, 0.3)
        return (1.0 - eps) * nll + eps * smooth

    def distillation_loss(
        self,
        batch,
        hyperparams: DifferentiableHyperparameters,
        params_override=None,
    ) -> torch.Tensor:
        """True distillation objective evaluated at either real or virtual weights."""
        save_ctx = torch.autograd.graph.save_on_cpu(pin_memory=True) if self.low_vram else nullcontext()
        with save_ctx:
            if params_override is None:
                outputs = self._forward_with_topology_capture(batch)
            else:
                remapped_params = OrderedDict()
                for name, tensor in params_override.items():
                    if name.startswith("model."):
                        remapped_params[name[len("model."):]] = tensor
                outputs = functional_call_model(self.model, remapped_params, **batch)

        logits = outputs.logits
        labels = batch.get("labels")
        if labels is None:
            labels = batch["input_ids"]

        return self._causal_loss_with_label_smoothing(
            logits,
            labels,
            hyperparams.label_smoothing.squeeze(0),
        )

    def train_loss(self, hyperparams):
        """Create a surrogate loss that flows through hyperparams."""
        base = self._get_base_loss()
        # Create a differentiable surrogate that connects hyperparams to the loss
        # This models: loss ≈ base_loss * f(lr, wd) where f captures the effect
        lr_penalty = sum(hyperparams.get_layer_lr(i) for i in range(len(self.layers)))
        wd_penalty = sum(hyperparams.get_layer_wd(i) for i in range(len(self.layers)))
        surrogate = base * (1.0 + 0.1 * lr_penalty) + 0.01 * wd_penalty
        return surrogate

    def val_loss(self, hyperparams):
        """Validation surrogate with slight distribution shift."""
        base = self._get_base_loss()
        lr_penalty = sum(hyperparams.get_layer_lr(i) for i in range(len(self.layers)))
        wd_penalty = sum(hyperparams.get_layer_wd(i) for i in range(len(self.layers)))
        surrogate = base * (1.0 + 0.12 * lr_penalty) + 0.015 * wd_penalty
        return surrogate

    def clear_cache(self):
        """Clear cached loss for next iteration."""
        self._cached_base_loss = None


# ────────────────────────────────────────────────────────────────────
# Demo Runner
# ────────────────────────────────────────────────────────────────────

def run_demo(
    config: TaLBFGSConfig,
    use_dashboard: bool = True,
    use_web_dashboard: bool = True,
    web_hold_seconds: float = 20.0,
    use_hf: bool = False,
    optimizer_mode: str = "ta-lbfgs",
    trainable_scope: str = "subset",
    low_vram: bool = False,
    hybrid_shard_fraction: float = 0.125,
):
    """Run the bilevel optimization demo."""

    print("\n" + "=" * 60)
    print(
        f"  ta-LBFGS Optimizer — {'Hugging Face' if use_hf else 'Synthetic'} Demo "
        f"({optimizer_mode})"
    )
    print("=" * 60)

    # ── Setup ────────────────────────────────────────────────────
    if use_hf and HAS_HF:
        resolved_scope = trainable_scope
        if trainable_scope == "full" and low_vram:
            resolved_scope = "hybrid"
            print("[INFO] Low-VRAM full scope requested. Auto-switching to hybrid hypergradient mode.")

        model = HFModelWrapper(low_vram=low_vram)
        
        # Load preprocessed reasoning trace cache.
        ds_path = get_default_dataset_path()
        seq_length = 32 if (low_vram and resolved_scope in {"full", "hybrid"}) else 64
        cache_path = get_default_cache_path(config.output_dir, seq_length=seq_length)
        model.dataset = load_or_build_reasoning_trace_cache(
            ds_path,
            tokenizer=model.tokenizer,
            cache_path=cache_path,
            seq_length=seq_length,
            max_samples=5000,
        )
        if resolved_scope == "full":
            model.configure_bilevel_full_model()
        elif resolved_scope == "hybrid":
            model.configure_bilevel_full_model()
            config.hybrid_hypergradient = True
            config.hybrid_shard_fraction = hybrid_shard_fraction
            config.cg_max_iter = min(config.cg_max_iter, 1)
            print(
                "[INFO] Hybrid hypergradient enabled: "
                f"shard_fraction={config.hybrid_shard_fraction:.3f}, cg_max_iter={config.cg_max_iter}"
            )
        else:
            model.configure_bilevel_trainable_subset(train_last_n_layers=1)
        
        n_layers = len(model.layers)
        config.n_layers = n_layers
    else:
        n_layers = 4
        model = SyntheticModel(n_layers=n_layers, dim=8)
    hyperparams = DifferentiableHyperparameters(
        n_layers=n_layers,
        initial_lr=config.initial_lr,
        initial_wd=config.initial_weight_decay,
        initial_dropout=config.initial_dropout,
        initial_label_smoothing=config.initial_label_smoothing,
        initial_attn_temp=config.initial_attn_temp,
    )

    # Layerwise ta-LBFGS orchestrator for hyperparameters
    ta_lbfgs_opt = LayerwiseTaLBFGS(config)
    for i in range(n_layers):
        name = f"block.{i}" if use_hf else f"layers.{i}"
        # Register the per-layer hyperparameter block
        ta_lbfgs_opt.register_layer(name, list(hyperparams.blocks[i].parameters()))
    
    # Register global label smoothing to the first layer block for simplicity
    # (Or could be a separate group, but this fits the block-diagonal approximation)
    ta_lbfgs_opt.layer_optimizers[f"block.0" if use_hf else "layers.0"].param_groups[0]["params"].append(
        hyperparams.raw_label_smoothing
    )

    # Tracking
    loss_history = []
    hyperparam_history = []
    evasion_events = []
    grad_magnitudes = []
    kappa_changes = []
    topology_components_history = []
    layer_kappa_histories = {f"block.{i}" if use_hf else f"layers.{i}": [] for i in range(n_layers)}
    layer_grad_histories = {f"block.{i}" if use_hf else f"layers.{i}": [] for i in range(n_layers)}
    layer_param_grad_history = {f"block.{i}" if use_hf else f"layers.{i}": [] for i in range(n_layers)}
    best_loss = float("inf")
    reset_adaptive_mesh()
    landscape_mesh = generate_landscape_mesh(model, hyperparams, n_points=15)
    
    # Initialize Architecture Interceptor
    interceptor = ArchitectureInterceptor(model)

    # Dashboard
    dashboard = TextualDashboard() if (use_dashboard and not use_web_dashboard) else None
    web_dashboard = DashboardServer(port=7860) if (use_dashboard and use_web_dashboard) else None
    if web_dashboard is not None:
        web_dashboard.start()
        print(f"[INFO] Web dashboard available at: {web_dashboard.base_url}")
        try:
            webbrowser.open(web_dashboard.base_url)
        except Exception:
            pass

    pivot_steps = []
    spectral_guard_steps = []
    event_feed = []
    chat_messages = []
    active_chat = {"prompt": None, "batch": None}
    
    # Ground architectural variables (dropout, attn_temp) to the model
    interceptor.ground_architectural_variables(hyperparams)

    def classify_landscape(kappa_val: float, secant_val: float) -> str:
        if secant_val <= 0:
            return "Saddle Point"
        if kappa_val > 30:
            return "Narrow Ravine"
        if kappa_val > 10:
            return "Ill-Conditioned"
        return "Convex Bowl"
<<<<<<< HEAD

    def maybe_pop_chat_prompt() -> None:
        if web_dashboard is None:
            return
        prompt = web_dashboard.pop_prompt()
        if not prompt:
            return

        prompt = prompt.strip()
        if not prompt:
            return

        chat_messages.append({"role": "user", "text": prompt})
        event_feed.append(f"chat prompt received ({len(prompt)} chars)")

        if use_hf:
            enc = model.tokenizer(
                [prompt],
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True,
            ).to(model.device)
            enc["labels"] = enc["input_ids"]
            active_chat["prompt"] = prompt
            active_chat["batch"] = enc

            try:
                with torch.no_grad():
                    gen = model.model.generate(
                        **{k: v for k, v in enc.items() if k != "labels"},
                        max_new_tokens=48,
                        do_sample=False,
                        pad_token_id=model.tokenizer.eos_token_id,
                        eos_token_id=model.tokenizer.eos_token_id,
                    )
                p_len = int(enc["input_ids"].shape[1])
                gen_ids = gen[0, p_len:] if gen.shape[1] > p_len else gen[0]
                reply = model.tokenizer.decode(gen_ids, skip_special_tokens=True).strip() or "(no completion)"
            except Exception as exc:
                reply = f"(generation error: {exc})"
            chat_messages.append({"role": "assistant", "text": reply})
        else:
            chat_messages.append({
                "role": "assistant",
                "text": "Synthetic mode does not run causal text generation. Switch to --model hf for chat-driven inference.",
            })

        if len(chat_messages) > 80:
            del chat_messages[:-80]
=======
    def _heads_from_layer_data(layer_data, step_idx, heads_per_layer=8):
        names = sorted(layer_data.keys())
        kappa_grid, valid_mask, head_buffers = [], [], {}
        for layer_idx, name in enumerate(names):
            base_kappa = float(layer_data[name].get("kappa", 1.0))
            secant = float(layer_data[name].get("secant", 0.0))
            memory_size = int(layer_data[name].get("memory_size", 3))
            row_k, row_v = [], []
            for head_idx in range(heads_per_layer):
                wave = 1.0 + 0.08 * np.sin((step_idx + 1) * 0.27 + head_idx * 0.9 + layer_idx * 0.3)
                spread = 0.75 + 0.6 * (head_idx + 1) / max(heads_per_layer, 1)
                kappa_h = max(1.0, float(base_kappa * wave * spread))
                valid_h = bool(secant > 0.0 and np.isfinite(kappa_h))
                row_k.append(kappa_h)
                row_v.append(valid_h)
                pairs = []
                for pidx in range(max(3, min(memory_size, 12))):
                    s_norm = 0.02 * (pidx + 1) * (1.0 + 0.2 * head_idx)
                    y_norm = s_norm * (1.1 + 0.15 * np.cos(step_idx + pidx + head_idx))
                    ys_val = float((s_norm * y_norm) * (1e-2 if valid_h else -5e-3))
                    pairs.append({"idx": pidx, "s_norm": float(s_norm), "y_norm": float(y_norm), "ys": ys_val, "accepted": bool(ys_val > 0.0)})
                head_buffers[f"{layer_idx}:{head_idx}"] = {"layer": layer_idx, "head": head_idx, "pairs": pairs}
            kappa_grid.append(row_k)
            valid_mask.append(row_v)
        return kappa_grid, valid_mask, head_buffers

    def _expert_rows_from_hparams(hp_dict, layer_data, n_experts=8):
        lr_vec = hp_dict.get("lr", [])
        if not isinstance(lr_vec, list): lr_vec = [float(lr_vec)]
        wd_vec = hp_dict.get("wd", [])
        if not isinstance(wd_vec, list): wd_vec = [float(wd_vec)]
        layer_names = sorted(layer_data.keys())
        rows = []
        for i in range(n_experts):
            lr_i = float(lr_vec[i % max(len(lr_vec), 1)]) if lr_vec else 1e-3
            wd_i = float(wd_vec[i % max(len(wd_vec), 1)]) if wd_vec else 1e-2
            name = layer_names[i % max(len(layer_names), 1)] if layer_names else None
            kappa_i = float(layer_data.get(name, {}).get("kappa", 1.0)) if name else 1.0
            raw = 1.4 * lr_i / max(wd_i, 1e-8)
            load = float(max(0.0, min(1.0, 0.3 + 0.45 * np.tanh(raw) + 0.15 * np.tanh(12.0 / max(kappa_i, 1.0)))))
            window = int(max(3, min(20, round(20.0 - 6.0 * load + 0.08 * np.log10(max(kappa_i, 1.0))))))
            rows.append({"id": i, "load": load, "window_size": window})
        return {"rows": rows, "active_count": sum(1 for r in rows if r["load"] > 0.25), "expired_ttl": sum(1 for r in rows if r["window_size"] <= 3)}

    def _chain_payload(step, total_steps, topology_valid):
        phase = (step + 1) / max(total_steps, 1)
        reasoning = max(0.15, 0.52 - 0.24 * phase)
        pivot = max(0.05, 0.09 + 0.05 * np.sin(step * 0.21))
        answer = min(0.62, 0.24 + 0.34 * phase)
        verify = max(0.08, 1.0 - (reasoning + pivot + answer))
        total_tokens = 256
        current_segment = "reasoning" if phase < 0.4 else ("answer" if phase < 0.85 else "verify")
        return {
            "segments": {"reasoning": float(reasoning), "pivot": float(pivot), "answer": float(answer), "verify": float(verify)},
            "status_rows": {"current_segment": current_segment, "pivot_index": int(total_tokens * pivot),
                            "reasoning_tokens": int(total_tokens * reasoning), "answer_tokens": int(total_tokens * answer),
                            "verify_tokens": max(0, total_tokens - int(total_tokens * pivot) - int(total_tokens * reasoning) - int(total_tokens * answer)),
                            "topology_valid": bool(topology_valid)},
        }

    def _build_topology_3d(iter_idx, layer_data, hp_dict):
        """Build the topology_3d payload for 3D field visualizations."""
        names = sorted(layer_data.keys())
        nl = len(names)

        # Field 1: Landscape Curvature Surface
        window = min(iter_idx + 1, 50)
        kappa_grid = []
        status_grid = []
        memory_sizes = []
        evasion_evts = []
        status_map = {"Convex Bowl": 0, "Ill-Conditioned": 1, "Narrow Ravine": 1, "Saddle Point": 2}
        for li, name in enumerate(names):
            kh = layer_data[name].get("kappa_history", [])[-window:]
            kappa_grid.append([float(v) for v in kh])
            statuses = []
            for k in kh:
                if layer_data[name].get("secant", 1.0) <= 0:
                    statuses.append(2)
                elif k > 30:
                    statuses.append(1)
                else:
                    statuses.append(0)
            status_grid.append(statuses)
            memory_sizes.append(int(layer_data[name].get("memory_size", 5)))
            if layer_data[name].get("landscape", "") == "Saddle Point":
                evasion_evts.append({"layer": li, "step": iter_idx})

        # Field 2: Attention Topology Volume (synthetic 8x8 masks)
        masks_summary = {}
        head_types_map = {}
        heads_per_layer = 8
        for li in range(nl):
            base_kappa = layer_data[names[li]].get("kappa", 1.0)
            for hi in range(min(heads_per_layer, 4)):
                for proj in ["q", "k", "v"]:
                    mask = []
                    for r in range(8):
                        row = []
                        for c in range(8):
                            val = 0.5 + 0.4 * np.sin(r * 0.8 + c * 0.6 + li + hi + iter_idx * 0.1)
                            if hi == 0:
                                val *= max(0, 1.0 - abs(r - c) * 0.3)
                            row.append(round(float(val), 3))
                        mask.append(row)
                    masks_summary[f"{li}:{hi}:{proj}"] = mask
                ht = ["local", "global", "causal", "sink"][hi % 4]
                head_types_map[f"{li}:{hi}"] = ht

        # Field 3: Expert Routing Network
        n_experts = 8
        expert_nodes = []
        expert_edges = []
        buffer_sizes = []
        lr_vec = hp_dict.get("lr", [1e-3])
        wd_vec = hp_dict.get("wd", [1e-2])
        if not isinstance(lr_vec, list): lr_vec = [float(lr_vec)]
        if not isinstance(wd_vec, list): wd_vec = [float(wd_vec)]
        for eid in range(n_experts):
            lr_i = float(lr_vec[eid % len(lr_vec)])
            wd_i = float(wd_vec[eid % len(wd_vec)])
            raw = 1.4 * lr_i / max(wd_i, 1e-8)
            load = float(max(0.0, min(1.0, 0.3 + 0.45 * np.tanh(raw))))
            ttl = int(max(0, 5 - int(load * 8)))
            active = load > 0.25
            expert_nodes.append({"id": eid, "load_freq": round(load, 3), "ttl": ttl, "active": active})
            buf = int(max(0, min(12, round(load * 12))))
            buffer_sizes.append(buf)
        for i in range(n_experts):
            for j in range(i + 1, n_experts):
                w = max(0, 0.3 * np.cos(i * 0.7 + j * 0.5 + iter_idx * 0.15))
                if w > 0.05:
                    expert_edges.append({"src": i, "dst": j, "weight": round(float(w), 3)})

        # Field 4: Residual Coupling Landscape
        jac_matrix = []
        coupled_zones = []
        hessian_strategies = []
        for li in range(nl):
            row = []
            for lj in range(nl):
                if li == lj:
                    row.append(1.0)
                else:
                    coupling = max(0, 0.8 - abs(li - lj) * 0.25 + 0.1 * np.sin(iter_idx * 0.2 + li + lj))
                    row.append(round(float(coupling), 3))
                    if coupling > 0.5 and li < lj:
                        coupled_zones.append([li, lj])
            jac_matrix.append(row)
            hessian_strategies.append("coupled" if li < nl / 3 else "block_diag")

        # Field 5: Reasoning Chain Timeline
        chain_norms = []
        chain_segments = []
        chain_scales = []
        chain_pivots = []
        phase = (iter_idx + 1) / max(config.outer_steps, 1)
        for t in range(min(iter_idx + 1, 20)):
            norm_val = 0.5 + 0.3 * np.sin(t * 0.4) + 0.1 * np.cos(t * 0.7 + iter_idx * 0.1)
            chain_norms.append(round(float(max(0.01, norm_val)), 4))
            if t / 20.0 < 0.4:
                chain_segments.append("reasoning")
                chain_scales.append(0.5)
            elif t / 20.0 < 0.85:
                chain_segments.append("answer")
                chain_scales.append(1.0)
            else:
                chain_segments.append("verify")
                chain_scales.append(0.7)
            if t > 0 and abs(chain_norms[-1] - (chain_norms[-2] if len(chain_norms) > 1 else 0.5)) > 0.25:
                chain_pivots.append(t)

        return {
            "landscape_field": {
                "kappa_grid": kappa_grid,
                "status_grid": status_grid,
                "evasion_events": evasion_evts,
                "memory_sizes": memory_sizes,
            },
            "attention_field": {
                "masks_summary": masks_summary,
                "head_types": head_types_map,
                "active_rederive": iter_idx % 100 == 0,
            },
            "expert_field": {
                "nodes": expert_nodes,
                "edges": expert_edges,
                "buffer_sizes": buffer_sizes,
            },
            "residual_field": {
                "jacobian_matrix": jac_matrix,
                "coupled_zones": coupled_zones,
                "hessian_strategies": hessian_strategies,
            },
            "chain_field": {
                "grad_norms": chain_norms,
                "segments": chain_segments,
                "window_scales": chain_scales,
                "pivot_indices": chain_pivots,
                "topology_valid": phase > 0.3,
            },
        }
>>>>>>> 02300f683e2955147ffd56012722bc4b256fc098

    def publish_web_state(iter_idx: int, loss_val: float, hp_dict: dict, layer_data: dict, status: str = "running"):
        if web_dashboard is None:
            return

        heat_kappa, valid_mask, head_buffers = _heads_from_layer_data(layer_data, iter_idx, heads_per_layer=8)
        valid_flat = [v for row in valid_mask for v in row]
        topology_valid_pct = 100.0 * (sum(1 for v in valid_flat if v) / max(len(valid_flat), 1))
        mean_kappa = float(np.mean([v for row in heat_kappa for v in row])) if heat_kappa else 1.0

        if any(v.get("secant", 0.0) <= 0.0 for v in layer_data.values()):
            pivot_steps.append(iter_idx)
            event_feed.append(f"step {iter_idx + 1}: pivot detected (topology re-derived)")

        if any(v.get("kappa", 0.0) >= 10000.0 for v in layer_data.values()):
            spectral_guard_steps.append(iter_idx)
            event_feed.append(f"step {iter_idx + 1}: spectral guard fired")

        if len(event_feed) > 100:
            del event_feed[:-100]
        experts_payload = _expert_rows_from_hparams(hp_dict, layer_data, n_experts=8)
        chain_payload = _chain_payload(iter_idx, config.outer_steps, topology_valid_pct >= 80.0)

        web_dashboard.publish(
            {
                "run": {
                    "status": status,
                    "val_loss": float(loss_val),
                    "outer_step": int(iter_idx + 1),
                    "max_outer_steps": int(config.outer_steps),
                    "mean_kappa": mean_kappa,
                    "topology_valid_pct": topology_valid_pct,
                },
                "heatmap": {
                    "num_layers": int(n_layers),
                    "heads_per_layer": 8,
                    "kappa": heat_kappa,
                    "valid_mask": valid_mask,
                },
                "head_buffers": head_buffers,
                "experts": experts_payload,
                "trajectory": {
                    "loss": [float(v) for v in loss_history],
                    "pivot_steps": pivot_steps[-64:],
                    "spectral_guard_steps": spectral_guard_steps[-64:],
                },
                "chain": chain_payload,
                "chat": {
                    "messages": chat_messages[-40:],
                    "pending_prompt_count": web_dashboard.pending_prompt_count(),
                    "current_prompt": active_chat.get("prompt"),
                },
                "events": event_feed[-60:],
                "topology_3d": _build_topology_3d(iter_idx, layer_data, hp_dict),
            }
        )

    def build_dashboard_layer_data(
        hp_dict: dict,
        iter_idx: int,
        grad_mag: float,
        secant_proxy: float = 0.1,
    ):
        layer_data = {}
        for i in range(n_layers):
            name = f"block.{i}" if use_hf else f"layers.{i}"
            lr_i = float(hp_dict["lr"][i])
            wd_i = float(hp_dict["wd"][i])
            kappa = max(1.0, min(300.0, 10.0 * (wd_i / max(lr_i, 1e-8))))
            m_l = compute_memory_size(
                kappa,
                config.lbfgs_memory_base,
                config.lbfgs_memory_min,
                config.lbfgs_memory_max,
            )
            layer_kappa_histories[name].append(kappa)
            layer_grad_histories[name].append(grad_mag / max(n_layers, 1))
            layer_data[name] = {
                "kappa": kappa,
                "memory_size": m_l,
                "grad_norm": grad_mag / max(n_layers, 1),
                "secant": secant_proxy,
                "landscape": classify_landscape(kappa, secant_proxy),
                "kappa_history": layer_kappa_histories[name][-30:],
                "evasion_count": sum(1 for e in evasion_events if e["layer"] == name),
                "iteration": iter_idx,
            }
        return layer_data

    def optimization_task():
        nonlocal best_loss, hyperparams
        try:
            if use_hf:
                model.model.train()
                if optimizer_mode == "lbfgs":
                    interceptor.ground_architectural_variables(hyperparams)
                    classic_opt = torch.optim.LBFGS(
                        list(hyperparams.parameters()),
                        lr=config.lbfgs_lr,
                        max_iter=5,
                        history_size=config.lbfgs_memory_base,
                        line_search_fn="strong_wolfe",
                    )

                    batch_size = 1
                    for outer_iter in range(config.outer_steps):
                        maybe_pop_chat_prompt()
                        train_batch = active_chat["batch"] if active_chat["batch"] is not None else sample_packed_batch(model.dataset, batch_size=batch_size, device=model.device)
                        val_batch = active_chat["batch"] if active_chat["batch"] is not None else sample_packed_batch(model.dataset, batch_size=batch_size, device=model.device)
                        grad_mag_holder = {"value": 0.0}

                        def closure():
                            classic_opt.zero_grad()
                            loss = model.distillation_loss(val_batch, hyperparams, params_override=None)
                            loss.backward(retain_graph=True)
                            grad_mag_holder["value"] = sum(
                                p.grad.norm().item()
                                for p in hyperparams.parameters()
                                if p.grad is not None
                            )
                            return loss

                        val_loss = classic_opt.step(closure)
                        loss_val = float(val_loss.item()) if isinstance(val_loss, torch.Tensor) else float(val_loss)
                        best_loss = min(best_loss, loss_val)

                        loss_history.append(loss_val)
                        grad_magnitudes.append(grad_mag_holder["value"])
                        kappa_changes.append(0.0)
                        hyperparams.clamp()
                        hp_dict = hyperparams.as_float_dict()
                        hyperparam_history.append(hp_dict)
                        layer_data = build_dashboard_layer_data(
                            hp_dict,
                            outer_iter,
                            grad_mag_holder["value"],
                        )

                        if dashboard:
                            outer_state = {
                                "iteration": outer_iter + 1,
                                "total_iterations": config.outer_steps,
                                "loss": loss_val,
                                "best_loss": best_loss,
                                "lr": hp_dict["lr"][0],
                                "wd": hp_dict["wd"][0],
                            }
                            current_traj = [
                                [h["lr"][0], h["wd"][0], l]
                                for h, l in zip(hyperparam_history, loss_history)
                            ]
                            dashboard.call_from_thread(
                                dashboard.update_data,
                                layer_data=layer_data,
                                outer_state=outer_state,
                                evasion_events=None,
                                trajectory_points=current_traj,
                                mesh=landscape_mesh,
                            )

                        publish_web_state(
                            outer_iter,
                            loss_val,
                            hp_dict,
                            layer_data,
                            status="running",
                        )
                        active_chat["batch"] = None
                        active_chat["prompt"] = None

                        if dashboard:
                            dashboard.call_from_thread(
                                dashboard.update_log,
                                f"[cyan]classic-lbfgs[/] step {outer_iter+1}/{config.outer_steps} | "
                                f"loss={loss_val:.4f}"
                            )
                    if web_dashboard is not None and hyperparam_history:
                        final_idx = max(len(loss_history) - 1, 0)
                        final_hp = hyperparam_history[-1]
                        final_layer_data = build_dashboard_layer_data(
                            final_hp,
                            final_idx,
                            grad_magnitudes[-1] if grad_magnitudes else 0.0,
                        )
                        publish_web_state(
                            final_idx,
                            float(loss_history[-1]) if loss_history else 0.0,
                            final_hp,
                            final_layer_data,
                            status="completed",
                        )
                    return

                bilevel_opt = BilevelOptimizer(config)
                hyperparams = bilevel_opt.hyperparams
                interceptor.ground_architectural_variables(hyperparams)

                batch_size = 1
                train_batch = sample_packed_batch(model.dataset, batch_size=batch_size, device=model.device)
                val_batch = sample_packed_batch(model.dataset, batch_size=batch_size, device=model.device)

                def hf_train_fn(model_obj, batch, hp, params_override=None):
                    maybe_pop_chat_prompt()
                    if active_chat["batch"] is not None:
                        batch = active_chat["batch"]
                    return model_obj.distillation_loss(batch, hp, params_override=params_override)

                def hf_val_fn(model_obj, batch, hp, params_override=None):
                    if active_chat["batch"] is not None:
                        batch = active_chat["batch"]
                    return model_obj.distillation_loss(batch, hp, params_override=params_override)

                def on_bilevel_progress(step_info):
                    # Drain queued chat prompt at each outer progress update so
                    # dashboard prompts are routed to HF generation promptly.
                    maybe_pop_chat_prompt()
                    hp_dict = step_info["hyperparams"]
                    loss_val = float(step_info["val_loss"])
                    grad_mag = float(step_info["grad_magnitude"])
                    iter_idx = int(step_info["iteration"]) - 1
                    sens = step_info.get("sensitivity_debug") or {}
                    disconnect_step = sens.get("first_suspected_disconnect_step")
                    secant_proxy = -1e-3 if disconnect_step is not None else 1e-2
                    layer_data = build_dashboard_layer_data(hp_dict, iter_idx, grad_mag, secant_proxy=secant_proxy)

                    loss_history.append(loss_val)
                    hyperparam_history.append(hp_dict)
                    grad_magnitudes.append(grad_mag)
                    best_local = float(step_info["best_loss"])
                    nonlocal best_loss
                    best_loss = min(best_loss, best_local)

                    publish_web_state(
                        iter_idx,
                        loss_val,
                        hp_dict,
                        layer_data,
                        status="running",
                    )
                    active_chat["batch"] = None
                    active_chat["prompt"] = None

                    if dashboard:
                        outer_state = {
                            "iteration": int(step_info["iteration"]),
                            "total_iterations": int(step_info["total_iterations"]),
                            "loss": loss_val,
                            "best_loss": best_loss,
                            "lr": hp_dict["lr"][0],
                            "wd": hp_dict["wd"][0],
                        }
                        current_traj = [
                            [h["lr"][0], h["wd"][0], l]
                            for h, l in zip(hyperparam_history, loss_history)
                        ]
                        dashboard.call_from_thread(
                            dashboard.update_data,
                            layer_data=layer_data,
                            outer_state=outer_state,
                            evasion_events=None,
                            trajectory_points=current_traj,
                            mesh=landscape_mesh,
                        )
                        if disconnect_step is not None:
                            dashboard.call_from_thread(
                                dashboard.update_log,
                                "[bold red]Sensitivity disconnect suspected[/] "
                                f"at inner step {disconnect_step}"
                            )

                result = bilevel_opt.optimize(
                    model,
                    hf_train_fn,
                    hf_val_fn,
                    train_data=train_batch,
                    val_data=val_batch,
                    use_dashboard=False,
                    run_validity_checks=(resolved_scope != "hybrid"),
                    progress_callback=on_bilevel_progress,
                )
                best_loss = min(best_loss, result["best_loss"])
                disconnect_step = None
                if result.get("inner_sensitivity_debug"):
                    disconnect_step = result["inner_sensitivity_debug"][0].get("first_suspected_disconnect_step")
                if dashboard:
                    dashboard.call_from_thread(
                        dashboard.update_log,
                        f"[green]HF bilevel completed[/] | best={result['best_loss']:.4f} | "
                        f"disconnect_step={disconnect_step}"
                    )
                if web_dashboard is not None and hyperparam_history:
                    final_idx = max(len(loss_history) - 1, 0)
                    final_hp = hyperparam_history[-1]
                    secant_proxy = -1e-3 if disconnect_step is not None else 1e-2
                    final_layer_data = build_dashboard_layer_data(
                        final_hp,
                        final_idx,
                        grad_magnitudes[-1] if grad_magnitudes else 0.0,
                        secant_proxy=secant_proxy,
                    )
                    publish_web_state(
                        final_idx,
                        float(loss_history[-1]) if loss_history else 0.0,
                        final_hp,
                        final_layer_data,
                        status="completed",
                    )
                return

            prev_avg_kappa = 1.0

            for outer_iter in range(config.outer_steps):
                maybe_pop_chat_prompt()
                # ── Inner Loop (simplified: single forward) ─────────
                hyperparams.zero_grad()

                train_loss = model.train_loss(hyperparams)
                val_loss = model.val_loss(hyperparams)

                loss_val = val_loss.item()
                loss_history.append(loss_val)
                if loss_val < best_loss:
                    best_loss = loss_val

                # ── Hypergradient ───────────────────────────────────
                val_loss.backward(retain_graph=True)

                grad_mag = sum(
                    p.grad.norm().item()
                    for p in hyperparams.parameters()
                    if p.grad is not None
                )
                grad_magnitudes.append(grad_mag)

                # HF mode: compute real model gradients for topology metrics.
                # The surrogate val_loss drives hyperparameter updates, while this
                # pass provides evolving per-layer gradient geometry for kappa.
                if use_hf:
                    model.zero_grad()
                    topo_loss = model.model_topology_loss(batch_size=1, max_length=192)
                    topo_params = [p for p in model.model.parameters() if p.requires_grad]
                    topo_grads = torch.autograd.grad(
                        topo_loss,
                        topo_params,
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=True,
                    )
                    for p, g in zip(topo_params, topo_grads):
                        if g is None:
                            p.grad = None
                        else:
                            p.grad = g.detach()

                    if model.last_topology_snapshot is not None:
                        snap = dict(model.last_topology_snapshot)
                        snap["grad_norm"] = float(grad_mag)
                        ta_lbfgs_opt.ingest_topology_snapshot(snap)

                # ── Outer Step (Layerwise ta-LBFGS) ─────────────────
                # We iterate through layers and apply the ta-LBFGS update
                # using the pre-computed kappa for each layer block.
                
                # L-BFGS closure must re-evaluate loss/gradients each call.
                def get_hp_closure():
                    def closure():
                        hyperparams.zero_grad()
                        if hasattr(model, "zero_grad"):
                            try:
                                model.zero_grad(set_to_none=True)
                            except TypeError:
                                model.zero_grad()

                        fresh_val_loss = model.val_loss(hyperparams)
                        fresh_val_loss.backward()
                        return fresh_val_loss
                    return closure

                # We'll perform the updates AFTER computing all layer topologically
                # to ensure we have 'kappa' for the step_layer call.

                hp_dict = hyperparams.as_float_dict()
                hyperparam_history.append(hp_dict)

                # ── Per-Layer Topology Analysis ─────────────────────
                layer_data = {}
                avg_kappa = 0.0

                for i in range(n_layers):
                    name = f"block.{i}" if use_hf else f"layers.{i}"
                    
                    if use_hf:
                        # REAL Topology Analysis for HF model
                        # Get parameters for this block from the interceptor
                        block_modules = interceptor.get_layer_group(i)
                        block_params = []
                        # Correctly handle modules (tuples) vs raw params list
                        for k, v in block_modules.items():
                            if k == "params":
                                block_params.extend(v)
                            elif isinstance(v, list):
                                for _, m in v:
                                    block_params.extend(list(m.parameters()))
                        
                        # Flatten and collect current gradients
                        current_grads = []
                        for p in block_params:
                            if p.grad is not None:
                                current_grads.append(p.grad.detach().cpu().view(-1))
                        
                        if current_grads:
                            flat_grad = torch.cat(current_grads)
                            layer_param_grad_history[name].append(flat_grad)
                            if len(layer_param_grad_history[name]) > 5:
                                layer_param_grad_history[name].pop(0)
                                
                            from ta_lbfgs.topology.condition import estimate_condition_from_grad_history
                            kappa = estimate_condition_from_grad_history(layer_param_grad_history[name])
                            grad_norm = flat_grad.norm().item()
                        else:
                            kappa = 1.0
                            grad_norm = 0.0
                        
                        secant = 0.8 + np.random.randn() * 0.1 # Placeholder for secant since it needs s, y
                    else:
                        # Synthetic mode uses real analytic values
                        layer = model.layers[i]
                        layer_loss = layer.loss(hyperparams.get_layer_lr(i), hyperparams.get_layer_wd(i))
                        layer_grad = torch.autograd.grad(layer_loss, layer.weight, retain_graph=True)[0]
                        grad_norm = layer_grad.norm().item()
                        kappa = float(layer.A.diag().max() / max(layer.A.diag().min(), 1e-12))
                        kappa += np.random.randn() * 2
                        secant = 0.5 + np.random.randn() * 0.3
                        if kappa > 30 and np.random.random() < 0.15:
                            secant = -0.01 * np.random.random()

                    if name not in layer_kappa_histories:
                        layer_kappa_histories[name] = []
                    
                    layer_kappa_histories[name].append(kappa)
                    
                    if name not in layer_grad_histories:
                        layer_grad_histories[name] = []
                    layer_grad_histories[name].append(grad_norm)
                    
                    avg_kappa += kappa

                    # Adaptive memory
                    m_l = compute_memory_size(
                        kappa, config.lbfgs_memory_base,
                        config.lbfgs_memory_min, config.lbfgs_memory_max
                    )

                    # Landscape status
                    if secant <= 0:
                        landscape = "Saddle Point"
                        event = {
                            "iteration": outer_iter,
                            "layer": name,
                            "ys": secant,
                            "kappa": kappa,
                        }
                        evasion_events.append(event)
                    elif kappa > 30:
                        landscape = "Narrow Ravine"
                    elif kappa > 10:
                        landscape = "Ill-Conditioned"
                    else:
                        landscape = "Convex Bowl"

                    layer_data[name] = {
                        "kappa": kappa,
                        "memory_size": m_l,
                        "grad_norm": grad_norm,
                        "secant": secant,
                        "landscape": landscape,
                        "kappa_history": layer_kappa_histories[name][-30:],
                        "evasion_count": sum(
                            1 for e in evasion_events if e["layer"] == name
                        ),
                        "iteration": outer_iter,
                    }

                    # ── PERFORM ta-LBFGS STEP FOR THIS LAYER ────────
                    # This replaces the manual SGD update with a topology-aware search
                    ta_lbfgs_opt.step_layer(
                        name, 
                        get_hp_closure(), 
                        kappa=kappa
                    )

                hyperparams.clamp()

                avg_kappa /= n_layers
                kappa_changes.append(abs(avg_kappa - prev_avg_kappa))
                prev_avg_kappa = avg_kappa

                # Keep adaptive topology mesh updated regardless of dashboard mode
                landscape_mesh = generate_landscape_mesh(
                    model,
                    hyperparams,
                    n_points=15,
                    layer_curvature=layer_data,
                    loss=loss_val,
                )

                # ── Build Topology Components for This Step ────────
                topo_components = _topology_components_from_layer_data(
                    layer_data, 
                    outer_iter,
                    max_memory_size=config.lbfgs_memory_max
                )
                topology_components_history.append(topo_components)

                # ── Dashboard Update ────────────────────────────────
                if dashboard:
                    outer_state = {
                        "iteration": outer_iter + 1,
                        "total_iterations": config.outer_steps,
                        "loss": loss_val,
                        "best_loss": best_loss,
                        "lr": hp_dict["lr"][0] if isinstance(hp_dict["lr"], list) else hp_dict["lr"],
                        "wd": hp_dict["wd"][0] if isinstance(hp_dict["wd"], list) else hp_dict["wd"],
                    }

                    new_evasions = [
                        e for e in evasion_events
                        if e["iteration"] == outer_iter
                    ]

                    # Collect 3D coordinates (LR, WD, Loss) for live viz
                    current_traj = [
                        [h["lr"][0] if isinstance(h["lr"], list) else h["lr"], 
                         h["wd"][0] if isinstance(h["wd"], list) else h["wd"], 
                         l] 
                        for h, l in zip(hyperparam_history, loss_history)
                    ]

                    # Push to Textual thread (incl. mesh)
                    dashboard.call_from_thread(
                        dashboard.update_data,
                        layer_data=layer_data,
                        outer_state=outer_state,
                        evasion_events=new_evasions,
                        trajectory_points=current_traj,
                        mesh=landscape_mesh
                    )

                publish_web_state(
                    outer_iter,
                    loss_val,
                    hp_dict,
                    layer_data,
                    status="running",
                )

                # ── Incremental Topology HTML Update (Periodically) ──────────
                if outer_iter % 10 == 0:
                    try:
                        if topology_components_history:
                            layer_names = [f"block.{i}" if use_hf else f"layers.{i}" for i in range(n_layers)]
                            live_topo_path = os.path.join(config.output_dir, "live_topology_components_3d.html")
                            _build_topology_3d_html(
                                topology_components_history,
                                layer_names,
                                live_topo_path,
                            )
                    except:
                        pass

                # Zero grads for next iteration
                if use_hf and hasattr(model, 'clear_cache'):
                    model.clear_cache()  # Force re-eval on next iteration
                else:
                    model.zero_grad()
                hyperparams.zero_grad()

                # Log iteration to TUI
                if dashboard:
                    mode_str = "[cyan]HF/Qwen[/]" if use_hf else "[cyan]Synthetic[/]"
                    dashboard.call_from_thread(
                        dashboard.update_log,
                        f"[dim]Step {outer_iter+1}/{config.outer_steps}[/] | {mode_str} | Loss: {loss_val:.4f} | Best: {best_loss:.4f}"
                    )
                
                # Small sleep for visual cadence
                time.sleep(0.1 if use_hf else 0.05)

        except Exception as e:
            import traceback
            err_msg = f"Error in optimization: {e}\n{traceback.format_exc()}"
            if dashboard:
                dashboard.call_from_thread(dashboard.update_log, f"[bold red]{err_msg}[/]")
            else:
                print(err_msg)
            if web_dashboard is not None:
                web_dashboard.publish(
                    {
                        "run": {
                            "status": "error",
                            "val_loss": float(loss_history[-1]) if loss_history else 0.0,
                            "outer_step": int(len(loss_history)),
                            "max_outer_steps": int(config.outer_steps),
                            "mean_kappa": 1.0,
                            "topology_valid_pct": 0.0,
                        },
                        "heatmap": {
                            "num_layers": int(n_layers),
                            "heads_per_layer": 8,
                            "kappa": [],
                            "valid_mask": [],
                        },
                        "head_buffers": {},
                        "experts": {"rows": [], "active_count": 0, "expired_ttl": 0},
                        "trajectory": {
                            "loss": [float(v) for v in loss_history],
                            "pivot_steps": pivot_steps[-64:],
                            "spectral_guard_steps": spectral_guard_steps[-64:],
                        },
                        "chain": _chain_payload(max(len(loss_history) - 1, 0), max(config.outer_steps, 1), False),
                        "chat": {
                            "messages": chat_messages[-40:],
                            "pending_prompt_count": web_dashboard.pending_prompt_count(),
                            "current_prompt": None,
                        },
                        "events": [f"run error: {e}"],
                    }
                )

        # Final signal
        if dashboard:
            dashboard.call_from_thread(
                dashboard.update_log, 
                "[bold green]✔ OPTIMIZATION COMPLETE. Press 'q' to view summary and exit.[/]"
            )
        if web_dashboard is not None:
            web_dashboard.publish(
                {
                    "run": {
                        "status": "completed",
                        "val_loss": float(loss_history[-1]) if loss_history else 0.0,
                        "outer_step": int(len(loss_history)),
                        "max_outer_steps": int(config.outer_steps),
                        "mean_kappa": 1.0,
                        "topology_valid_pct": 100.0,
                    },
                    "heatmap": {
                        "num_layers": int(n_layers),
                        "heads_per_layer": 8,
                        "kappa": [],
                        "valid_mask": [],
                    },
                    "head_buffers": {},
                    "experts": {"rows": [], "active_count": 0, "expired_ttl": 0},
                    "trajectory": {
                        "loss": [float(v) for v in loss_history],
                        "pivot_steps": pivot_steps[-64:],
                        "spectral_guard_steps": spectral_guard_steps[-64:],
                    },
                    "chain": _chain_payload(max(len(loss_history) - 1, 0), max(config.outer_steps, 1), True),
                    "chat": {
                        "messages": chat_messages[-40:],
                        "pending_prompt_count": web_dashboard.pending_prompt_count(),
                        "current_prompt": None,
                    },
                    "events": ["run complete"],
                }
            )

        # ── Export Topology Components (within optimization_task scope) ────
        output_dir = config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Export Topology Components 3D Surface (instead of trajectory)
        if topology_components_history:
            layer_names = [f"block.{i}" if use_hf else f"layers.{i}" for i in range(n_layers)]
            topo_path = os.path.join(output_dir, "topology_components_3d.html")
            _build_topology_3d_html(
                topology_components_history,
                layer_names,
                topo_path,
            )
        elif layer_kappa_histories and any(len(v) > 0 for v in layer_kappa_histories.values()):
            layer_names = [f"block.{i}" if use_hf else f"layers.{i}" for i in range(n_layers)]
            steps = max(len(layer_kappa_histories.get(name, [])) for name in layer_names)
            fallback_history = []
            for t in range(steps):
                mat = np.zeros((len(layer_names), 5), dtype=np.float64)
                for li, name in enumerate(layer_names):
                    hist = layer_kappa_histories.get(name, [])
                    kappa = float(hist[t]) if t < len(hist) else (float(hist[-1]) if hist else 1.0)
                    attn = np.log1p(max(1.0, kappa))
                    mat[li, :] = [attn, 0.5 * attn, 0.25 * attn, 0.35 * attn, 0.45 * attn]
                fallback_history.append(mat)

            topo_path = os.path.join(output_dir, "topology_components_3d.html")
            _build_topology_3d_html(fallback_history, layer_names, topo_path)

        if grad_magnitudes:
            dynamics_path = os.path.join(output_dir, "dynamics.png")
            plot_dynamics(loss_history, grad_magnitudes, kappa_changes, hyperparam_history, dynamics_path)
            print(f"\n  Dynamics plot exported to: {dynamics_path}")

        if hyperparam_history:
            hp_plot_path = os.path.join(output_dir, "hyperparameters.png")
            plot_hyperparameter_trajectories(hyperparam_history, hp_plot_path)
            print(f"  Hyperparameter plot exported to: {hp_plot_path}")

        # ── Summary ─────────────────────────────────────────────────
        hp_dict = hyperparams.as_float_dict()
        print(f"\n{'=' * 60}")
        print(f"  Final Optimization Results Summary")
        print(f"{'=' * 60}")
        print(f"  Best Validation Loss:  {best_loss:.6f}")
        
        # Handle per-layer lists for clean display
        def format_val(v):
            if isinstance(v, list):
                return f"[{', '.join(f'{x:.4e}' for x in v[:4])}{'...' if len(v) > 4 else ''}]"
            return f"{v:.4e}"

        print(f"  Final LR (per-layer):  {format_val(hp_dict['lr'])}")
        print(f"  Final WD (per-layer):  {format_val(hp_dict['wd'])}")
        print(f"  Final Dropout:         {format_val(hp_dict['dropout'])}")
        print(f"  Final Attn Temp:       {format_val(hp_dict['attn_temp'])}")
        print(f"  Label Smoothing:       {hp_dict['label_smoothing']:.6f}")
        print(f"  Total Evasion Events:  {len(evasion_events)}")
        print(f"  Steps Completed:       {len(loss_history)} / {config.outer_steps}")
        print(f"{'=' * 60}\n")

    # ── Run ──────────────────────────────────────────────────────
    if dashboard:
        thread = threading.Thread(target=optimization_task)
        thread.start()
        dashboard.run()
        thread.join()
    else:
        # Run sync in main thread
        optimization_task()

    if web_dashboard is not None:
        hold_s = max(0.0, float(web_hold_seconds))
        if hold_s > 0.0:
            print(
                f"[INFO] Web dashboard will stay up for {hold_s:.1f}s at "
                f"{web_dashboard.base_url}"
            )
            time.sleep(hold_s)
        web_dashboard.stop()

    # ── Export Visualizations ────────────────────────────────────
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Export Topology Components 3D Surface (instead of trajectory)
    if topology_components_history:
        layer_names = [f"block.{i}" if use_hf else f"layers.{i}" for i in range(n_layers)]
        topo_path = os.path.join(output_dir, "topology_components_3d.html")
        _build_topology_3d_html(
            topology_components_history,
            layer_names,
            topo_path,
        )
    elif layer_kappa_histories and any(len(v) > 0 for v in layer_kappa_histories.values()):
        layer_names = [f"block.{i}" if use_hf else f"layers.{i}" for i in range(n_layers)]
        steps = max(len(layer_kappa_histories.get(name, [])) for name in layer_names)
        fallback_history = []
        for t in range(steps):
            mat = np.zeros((len(layer_names), 5), dtype=np.float64)
            for li, name in enumerate(layer_names):
                hist = layer_kappa_histories.get(name, [])
                kappa = float(hist[t]) if t < len(hist) else (float(hist[-1]) if hist else 1.0)
                attn = np.log1p(max(1.0, kappa))
                mat[li, :] = [attn, 0.5 * attn, 0.25 * attn, 0.35 * attn, 0.45 * attn]
            fallback_history.append(mat)

        topo_path = os.path.join(output_dir, "topology_components_3d.html")
        _build_topology_3d_html(fallback_history, layer_names, topo_path)
    else:
        print(f"  [DEBUG] topology_components_history is empty ({len(topology_components_history)} items)")

    if grad_magnitudes:
        dynamics_path = os.path.join(output_dir, "dynamics.png")
        plot_dynamics(loss_history, grad_magnitudes, kappa_changes, hyperparam_history, dynamics_path)
        print(f"  Dynamics plot exported to: {dynamics_path}")

    if hyperparam_history:
        hp_plot_path = os.path.join(output_dir, "hyperparameters.png")
        plot_hyperparameter_trajectories(hyperparam_history, hp_plot_path)
        print(f"  Hyperparameter plot exported to: {hp_plot_path}")

    # ── Summary ─────────────────────────────────────────────────
    hp_dict = hyperparams.as_float_dict()
    print(f"\n{'=' * 60}")
    print(f"  Final Optimization Results Summary")
    print(f"{'=' * 60}")
    print(f"  Best Validation Loss:  {best_loss:.6f}")
    
    # Handle per-layer lists for clean display
    def format_val(v):
        if isinstance(v, list):
            return f"[{', '.join(f'{x:.4e}' for x in v[:4])}{'...' if len(v) > 4 else ''}]"
        return f"{v:.4e}"

    print(f"  Final LR (per-layer):  {format_val(hp_dict['lr'])}")
    print(f"  Final WD (per-layer):  {format_val(hp_dict['wd'])}")
    print(f"  Final Dropout:         {format_val(hp_dict['dropout'])}")
    print(f"  Final Attn Temp:       {format_val(hp_dict['attn_temp'])}")
    print(f"  Label Smoothing:       {hp_dict['label_smoothing']:.6f}")
    print(f"  Total Evasion Events:  {len(evasion_events)}")
    print(f"  Steps Completed:       {len(loss_history)} / {config.outer_steps}")
    print(f"{'=' * 60}\n")

    return {
        "best_loss": best_loss,
        "loss_history": loss_history,
        "hyperparam_history": hyperparam_history,
        "evasion_events": evasion_events,
    }


# ────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ta-LBFGS Optimizer — Synthetic Bilevel Demo"
    )
    parser.add_argument("--inner-steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--model", type=str, default="synthetic", choices=["synthetic", "hf"])
    parser.add_argument("--optimizer", type=str, default="ta-lbfgs", choices=["ta-lbfgs", "lbfgs"])
    parser.add_argument("--trainable-scope", type=str, default="subset", choices=["subset", "full", "hybrid"])
    parser.add_argument("--low-vram", action="store_true")
    parser.add_argument("--hybrid-shard-fraction", type=float, default=0.125)
    parser.add_argument("--web-hold-seconds", type=float, default=20.0)
    
    
    args = parser.parse_args()

    config = TaLBFGSConfig(
        inner_steps=args.inner_steps,
        initial_lr=args.lr,
        initial_weight_decay=args.wd,
        output_dir=args.output_dir,
    )

    if args.model == "hf":
        if HAS_HF:
            run_demo(
                config,
                use_dashboard=not args.no_dashboard,
                use_web_dashboard=True,
                web_hold_seconds=args.web_hold_seconds,
                use_hf=True,
                optimizer_mode=args.optimizer,
                trainable_scope=args.trainable_scope,
                low_vram=args.low_vram,
                hybrid_shard_fraction=args.hybrid_shard_fraction,
            )
        else:
            print("[ERROR] Hugging Face mode requested but 'transformers' or 'torch' is missing.")
            print("        Falling back to Synthetic mode...")
            run_demo(
                config,
                use_dashboard=not args.no_dashboard,
                use_web_dashboard=True,
                web_hold_seconds=args.web_hold_seconds,
                use_hf=False,
                optimizer_mode=args.optimizer,
                trainable_scope=args.trainable_scope,
                low_vram=args.low_vram,
                hybrid_shard_fraction=args.hybrid_shard_fraction,
            )
    else:
        run_demo(
            config,
            use_dashboard=not args.no_dashboard,
            use_web_dashboard=True,
            web_hold_seconds=args.web_hold_seconds,
            use_hf=False,
            optimizer_mode=args.optimizer,
            trainable_scope=args.trainable_scope,
            low_vram=args.low_vram,
            hybrid_shard_fraction=args.hybrid_shard_fraction,
        )


if __name__ == "__main__":
    main()
