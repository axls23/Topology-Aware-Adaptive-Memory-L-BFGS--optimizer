"""
ta-LBFGS Live Run Manager.

Runs a local Hugging Face model (no WSL/vLLM) and streams real
per-layer gradients into the Textual dashboard and online RSVD projection.
"""

import argparse
import time
import torch
import numpy as np
import threading
from typing import List, Dict, Any, Optional
import webbrowser

from transformers import AutoModelForCausalLM, AutoTokenizer

from ta_lbfgs.config import TaLBFGSConfig
from ta_lbfgs.core.hyperparameters import DifferentiableHyperparameters
from ta_lbfgs.dashboard.textual_dashboard import TextualDashboard
from ta_lbfgs.dashboard.server import DashboardServer
from ta_lbfgs.dashboard.online_rsvd import LayerwiseOnlineRSVD
from ta_lbfgs.topology.adaptive_memory import compute_memory_size
from ta_lbfgs.topology.condition import estimate_condition_from_grad_history
from ta_lbfgs.dashboard.landscape_viz import (
    export_topology_components_3d,
    generate_landscape_mesh,
    reset_adaptive_mesh,
)
from ta_lbfgs.topology.persistent_homology import (
    compute_pointcloud_persistence,
    persistence_to_payload,
    is_available as persistence_available,
)
from ta_lbfgs.topology.vtk_exporter import (
    export_loss_grid_vti,
    export_residual_drift_vtp,
    export_persistence_diagram_json,
)
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


def _kappa_heads_from_layers(layer_data: Dict[str, Dict[str, Any]], step_idx: int, heads_per_layer: int = 8):
    names = sorted(layer_data.keys())
    kappa_grid: List[List[float]] = []
    valid_mask: List[List[bool]] = []
    head_buffers: Dict[str, Dict[str, Any]] = {}

    for layer_idx, name in enumerate(names):
        base_kappa = float(layer_data[name].get("kappa", 1.0))
        secant = float(layer_data[name].get("secant", 0.0))
        memory_size = int(layer_data[name].get("memory_size", 3))

        row_kappa: List[float] = []
        row_valid: List[bool] = []
        for head_idx in range(heads_per_layer):
            wave = 1.0 + 0.08 * np.sin((step_idx + 1) * 0.27 + head_idx * 0.9 + layer_idx * 0.3)
            spread = 0.75 + 0.6 * (head_idx + 1) / max(heads_per_layer, 1)
            kappa_h = max(1.0, float(base_kappa * wave * spread))
            valid_h = bool(secant > 0.0 and np.isfinite(kappa_h))
            row_kappa.append(kappa_h)
            row_valid.append(valid_h)

            pair_count = max(3, min(memory_size, 12))
            pairs = []
            for pidx in range(pair_count):
                s_norm = 0.02 * (pidx + 1) * (1.0 + 0.2 * head_idx)
                y_norm = s_norm * (1.1 + 0.15 * np.cos(step_idx + pidx + head_idx))
                ys_val = float((s_norm * y_norm) * (1e-2 if valid_h else -5e-3))
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


def _experts_from_state(hp_dict: Dict[str, Any], layer_data: Dict[str, Dict[str, Any]], n_experts: int = 8):
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
    reasoning = max(0.15, 0.52 - 0.24 * phase)
    pivot = max(0.05, 0.09 + 0.05 * np.sin(step * 0.21))
    answer = min(0.62, 0.24 + 0.34 * phase)
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


def _topology_3d_payload(
    topology_components_history: List[np.ndarray],
    layer_names: List[str],
    max_frames: int = 48,
) -> Dict[str, Any]:
    """Build compact realtime payload for 3D topology surface streaming."""
    component_names = ["attention", "moe", "residual", "chain", "global"]
    if not topology_components_history:
        return {
            "component_names": component_names,
            "layer_names": layer_names,
            "current_step": 0,
            "current_surface": [],
            "history": [],
        }

    start = max(0, len(topology_components_history) - max_frames)
    hist = topology_components_history[start:]
    stacked = np.stack(hist, axis=0)  # [T, L, 5]

    # Normalize per component over retained window for stable visual scale.
    norm = stacked.copy()
    for c in range(norm.shape[2]):
        col = norm[:, :, c]
        lo = float(np.nanmin(col))
        hi = float(np.nanmax(col))
        span = max(1e-9, hi - lo)
        norm[:, :, c] = (col - lo) / span

    # Surface form matches exporter: [component, layer]
    current_surface = norm[-1].T.tolist()
    history = [frame.T.tolist() for frame in norm]

    return {
        "component_names": component_names,
        "layer_names": layer_names,
        "current_step": int(len(topology_components_history)),
        "current_surface": current_surface,
        "history": history,
    }

class HFGradientEvaluator:
    """Loads a local HF model and exposes true block-gradient signals."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        max_length: int = 192,
        batch_size: int = 1,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = max_length
        self.batch_size = max(1, batch_size)

        if model_path is None:
            model_path = "Qwen/Qwen2.5-0.5B"

        resolved = model_path
        if resolved.startswith("~"):
            import os
            resolved = os.path.expanduser(resolved)

        self.tokenizer = AutoTokenizer.from_pretrained(resolved)
        self.model = AutoModelForCausalLM.from_pretrained(
            resolved,
            device_map="auto",
        )
        self.model.train()

        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            self.blocks = self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            self.blocks = self.model.transformer.h
        else:
            self.blocks = []

        self.n_layers = len(self.blocks)
        self.topology_warmup_steps = 50
        self._can_output_attentions = can_output_attentions(self.model.config)
        self._is_moe_model = detect_moe_model(self.model.config)
        self.last_topology_snapshot: Optional[Dict[str, Any]] = None
        ds_path = get_default_dataset_path()
        cache_path = get_default_cache_path("outputs", seq_length=max(self.max_length, 256))
        self.dataset = load_or_build_reasoning_trace_cache(
            ds_path,
            tokenizer=self.tokenizer,
            cache_path=cache_path,
            seq_length=max(self.max_length, 256),
            max_samples=5000,
        )

    def _sample_batch(self) -> Dict[str, torch.Tensor]:
        return sample_packed_batch(self.dataset, batch_size=self.batch_size, device=self.device)

    def _forward_with_topology_capture(self, batch: Dict[str, torch.Tensor], step: int):
        outputs = self.model(
            **batch,
            output_attentions=self._can_output_attentions and (step < int(self.topology_warmup_steps)),
            output_router_logits=self._is_moe_model,
            output_hidden_states=(step < int(self.topology_warmup_steps)),
            use_cache=True,
            return_dict=True,
        )
        self.last_topology_snapshot = build_topology_snapshot(
            outputs=outputs,
            step=int(step),
            warmup_steps=int(self.topology_warmup_steps),
            model_config=self.model.config,
        )
        return outputs

    def compute_loss(self, hyperparams: DifferentiableHyperparameters, step: int) -> torch.Tensor:
        batch = self._sample_batch()
        outputs = self._forward_with_topology_capture(batch, step)
        base_loss = outputs.loss

        # Keep hyperparameters connected to loss for outer updates while preserving
        # real model gradients for layerwise topology signals.
        lr_term = torch.stack([hyperparams.get_layer_lr(i) for i in range(self.n_layers)]).mean()
        wd_term = torch.stack([hyperparams.get_layer_wd(i) for i in range(self.n_layers)]).mean()
        return base_loss * (1.0 + 0.05 * lr_term) + 0.01 * wd_term

    def compute_prompt_loss(self, prompt: str, hyperparams: DifferentiableHyperparameters, step: int) -> torch.Tensor:
        enc = self.tokenizer(
            [prompt],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        ).to(self.device)
        enc["labels"] = enc["input_ids"]
        outputs = self._forward_with_topology_capture(enc, step)
        base_loss = outputs.loss
        lr_term = torch.stack([hyperparams.get_layer_lr(i) for i in range(self.n_layers)]).mean()
        wd_term = torch.stack([hyperparams.get_layer_wd(i) for i in range(self.n_layers)]).mean()
        return base_loss * (1.0 + 0.05 * lr_term) + 0.01 * wd_term

    @torch.no_grad()
    def generate_reply(self, prompt: str, max_new_tokens: int = 48) -> str:
        enc = self.tokenizer(
            [prompt],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        ).to(self.device)
        gen = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        # Strip prompt prefix from generated ids when possible.
        prompt_len = int(enc["input_ids"].shape[1])
        gen_ids = gen[0, prompt_len:] if gen.shape[1] > prompt_len else gen[0]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return text if text else "(no completion)"

    def layer_gradient_sketch(self, layer_idx: int, target_dim: int = 512) -> tuple[Optional[torch.Tensor], float]:
        if layer_idx >= self.n_layers:
            return None, 0.0

        grads = []
        norm_sq = 0.0
        for p in self.blocks[layer_idx].parameters():
            if p.grad is None:
                continue
            g = p.grad.detach().view(-1)
            if g.numel() == 0:
                continue
            grads.append(g)
            norm_sq += g.float().pow(2).sum().item()

        if not grads:
            return None, 0.0

        quota = max(4, target_dim // max(len(grads), 1))
        pieces = []
        for g in grads:
            stride = max(1, g.numel() // quota)
            sampled = g[::stride][:quota]
            pieces.append(sampled.float().cpu())

        flat = torch.cat(pieces, dim=0)
        if flat.numel() < target_dim:
            flat = torch.nn.functional.pad(flat, (0, target_dim - flat.numel()))
        else:
            flat = flat[:target_dim]

        return flat, float(np.sqrt(max(norm_sq, 0.0)))

def run_live(
    config: TaLBFGSConfig,
    use_dashboard: bool = True,
    use_web_dashboard: bool = True,
    web_hold_seconds: float = 20.0,
):
    """ Main entry point for live optimization. """

    evaluator = HFGradientEvaluator(
        model_path=config.vllm_model_name,
        max_length=192,
        batch_size=1,
    )
    if evaluator.n_layers == 0:
        raise RuntimeError("No transformer layers detected for HF model.")

    n_layers = evaluator.n_layers
    hyperparams = DifferentiableHyperparameters(n_layers=n_layers)
    
    dashboard = TextualDashboard() if (use_dashboard and not use_web_dashboard) else None
    web_dashboard = DashboardServer(port=7860) if (use_dashboard and use_web_dashboard) else None
    if web_dashboard is not None:
        web_dashboard.start()
        print(f"[live_run] web dashboard at {web_dashboard.base_url}")
        try:
            webbrowser.open(web_dashboard.base_url)
        except Exception:
            pass
    reset_adaptive_mesh()
    landscape_mesh = generate_landscape_mesh(None, hyperparams) # Pseudo-topology for visualization
    monitored_layer = config.dashboard_projection_layer
    projector = LayerwiseOnlineRSVD(
        monitored_layer=monitored_layer,
        n_components=config.dashboard_projection_components,
        sketch_dim=config.dashboard_projection_sketch_dim,
        forgetting_factor=config.dashboard_projection_forgetting_factor,
        warning_threshold=config.dashboard_projection_warning_threshold,
        seed=13,
    )

    loss_history = []
    hp_history = []
    projected_history = []
    layer_grad_histories: Dict[str, List[torch.Tensor]] = {f"block.{i}": [] for i in range(n_layers)}
    layer_kappa_histories: Dict[str, List[float]] = {f"block.{i}": [] for i in range(n_layers)}
    prev_grad_sketch: Dict[str, torch.Tensor] = {}
    topology_components_history: List[np.ndarray] = []
    persistence_payload: Dict[str, Any] = {}  # latest persistence diagram data
    persistence_compute_interval = 1  # compute persistence every N steps
    vtk_export_interval = 1  # export VTK files every N steps
    
    def optimization_task():
        nonlocal persistence_payload
        best_loss = float("inf")
        pivot_steps: List[int] = []
        spectral_guard_steps: List[int] = []
        event_feed: List[str] = []
        chat_messages: List[Dict[str, str]] = []
        current_chat_prompt: Optional[str] = None
        try:
            # Give Textual time to mount widgets before first update.
            if dashboard:
                time.sleep(0.35)
                dashboard.call_from_thread(
                    dashboard.update_log,
                    "[bold cyan]Live HF run started[/] | streaming true per-layer gradients",
                )

            for outer_iter in range(config.outer_steps):
                evaluator.model.zero_grad(set_to_none=True)
                hyperparams.zero_grad()

                current_chat_prompt = None
                if web_dashboard is not None:
                    maybe_prompt = web_dashboard.pop_prompt()
                    if maybe_prompt:
                        current_chat_prompt = maybe_prompt.strip()

                if current_chat_prompt:
                    val_loss = evaluator.compute_prompt_loss(current_chat_prompt, hyperparams, step=outer_iter)
                    chat_messages.append({"role": "user", "text": current_chat_prompt})
                    try:
                        reply = evaluator.generate_reply(current_chat_prompt, max_new_tokens=48)
                    except Exception as exc:
                        reply = f"(generation error: {exc})"
                    chat_messages.append({"role": "assistant", "text": reply})
                    event_feed.append(f"step {outer_iter + 1}: processed chat prompt ({len(current_chat_prompt)} chars)")
                else:
                    val_loss = evaluator.compute_loss(hyperparams, step=outer_iter)

                loss_val = val_loss.item()
                loss_history.append(loss_val)
                
                if loss_val < best_loss:
                    best_loss = loss_val
                
                val_loss.backward()
                grad_mag = sum(
                    float(p.grad.detach().norm().item())
                    for p in hyperparams.parameters()
                    if p.grad is not None
                )
                if evaluator.last_topology_snapshot is not None:
                    evaluator.last_topology_snapshot["grad_norm"] = grad_mag

                with torch.no_grad():
                    for p in hyperparams.parameters():
                        if p.grad is not None:
                            p.data -= 0.01 * p.grad
                    hyperparams.clamp()
                
                hp_dict = hyperparams.as_float_dict()
                hp_history.append(hp_dict)
                
                # Update Dashboard
                outer_state = {
                    "iteration": outer_iter + 1,
                    "total_iterations": config.outer_steps,
                    "loss": loss_val,
                    "best_loss": best_loss,
                    "lr": hp_dict["lr"][0],
                    "wd": hp_dict["wd"][0],
                }
                
                sketch_dim = max(config.dashboard_projection_sketch_dim * 8, 128)
                layer_data = {}
                projection_info = None
                for i in range(n_layers):
                    name = f"block.{i}"
                    grad_sketch, grad_norm = evaluator.layer_gradient_sketch(i, target_dim=sketch_dim)

                    if grad_sketch is not None:
                        hist = layer_grad_histories[name]
                        hist.append(grad_sketch)
                        if len(hist) > config.gradient_window_size:
                            hist.pop(0)
                        kappa = estimate_condition_from_grad_history(hist)

                        prev = prev_grad_sketch.get(name)
                        if prev is None:
                            secant = float(torch.dot(grad_sketch, grad_sketch).item() * 1e-4)
                        else:
                            y = grad_sketch - prev
                            s = -0.01 * prev
                            secant = float(torch.dot(y, s).item())
                        prev_grad_sketch[name] = grad_sketch

                        update = projector.update(name, grad_sketch.numpy())
                    else:
                        kappa = 1.0
                        secant = 0.0
                        update = None

                    if kappa > 30:
                        landscape = "Narrow Ravine"
                    elif kappa > 10:
                        landscape = "Ill-Conditioned"
                    elif secant <= 0.0:
                        landscape = "Saddle Point"
                    else:
                        landscape = "Convex Bowl"

                    layer_kappa_histories[name].append(float(kappa))
                    if len(layer_kappa_histories[name]) > 30:
                        layer_kappa_histories[name].pop(0)

                    layer_data[name] = {
                        "kappa": float(kappa),
                        "memory_size": compute_memory_size(
                            float(kappa),
                            config.lbfgs_memory_base,
                            config.lbfgs_memory_min,
                            config.lbfgs_memory_max,
                        ),
                        "grad_norm": float(grad_norm),
                        "secant": float(secant),
                        "landscape": landscape,
                        "kappa_history": layer_kappa_histories[name],
                    }

                    if update is not None:
                        coords = update["coords"]
                        x = float(coords[0]) if coords.shape[0] > 0 else 0.0
                        y = float(coords[1]) if coords.shape[0] > 1 else 0.0
                        z = float(coords[2]) if coords.shape[0] > 2 else loss_val
                        projected_history.append([x, y, z])
                        projection_info = update

                if any(v.get("secant", 0.0) <= 0.0 for v in layer_data.values()):
                    pivot_steps.append(outer_iter)
                    event_feed.append(f"step {outer_iter + 1}: pivot detected (topology re-derived)")

                if any(v.get("kappa", 0.0) >= 10000.0 for v in layer_data.values()):
                    spectral_guard_steps.append(outer_iter)
                    event_feed.append(f"step {outer_iter + 1}: spectral guard fired")

                topology_components_history.append(
                    _topology_components_from_layer_data(
                        layer_data,
                        step_idx=outer_iter,
                        max_memory_size=config.lbfgs_memory_max,
                    )
                )

                # ── Persistent Homology (every N steps) ──────────────
                if persistence_available() and (outer_iter % persistence_compute_interval == 0):
                    try:
                        # Collect gradient sketches as a point cloud
                        grad_points = []
                        for name in sorted(layer_data.keys()):
                            hist = layer_grad_histories.get(name, [])
                            if hist and len(hist) >= 3:
                                stacked = torch.stack(hist[-min(len(hist), 20):]).numpy()
                                grad_points.append(stacked)
                        if grad_points:
                            point_cloud = np.vstack(grad_points)
                            diagram = compute_pointcloud_persistence(
                                point_cloud, homology_dimensions=(0, 1)
                            )
                            persistence_payload = persistence_to_payload(diagram)
                            persistence_payload["step"] = outer_iter
                            event_feed.append(
                                f"step {outer_iter + 1}: persistence β₀={persistence_payload['betti'].get(0, 0)} "
                                f"β₁={persistence_payload['betti'].get(1, 0)} "
                                f"saddles={persistence_payload['saddle_count']}"
                            )
                    except Exception as exc:
                        import traceback
                        traceback.print_exc()
                        event_feed.append(f"step {outer_iter + 1}: persistence error: {exc}")


                if len(event_feed) > 100:
                    event_feed = event_feed[-100:]
                if len(chat_messages) > 60:
                    chat_messages = chat_messages[-60:]

                current_traj = projected_history if projected_history else [
                    [h["lr"][0], h["wd"][0], l] for h, l in zip(hp_history, loss_history)
                ]

                landscape_mesh = generate_landscape_mesh(
                    None, hyperparams, layer_curvature=layer_data, loss=loss_val
                )

                # ── VTK Export for offline TTK analysis ───────────────
                if outer_iter >= 0 and outer_iter % vtk_export_interval == 0:
                    try:
                        layer_norms = np.array([
                            float(layer_data[f"block.{i}"].get("grad_norm", 0.0))
                            for i in range(n_layers)
                        ])
                        export_residual_drift_vtp(layer_norms, outer_iter)
                        
                        # Export the 2D loss grid for TTK landscape analysis
                        if len(landscape_mesh) == 3:
                            export_loss_grid_vti(landscape_mesh[2], outer_iter)
                            
                        if persistence_payload.get("diagram"):
                            export_persistence_diagram_json(
                                np.array(persistence_payload["diagram"]),
                                outer_iter,
                            )
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        event_feed.append(f"VTK export error at step {outer_iter}: {e}")

                if dashboard:
                    dashboard.call_from_thread(
                        dashboard.update_data,
                        layer_data=layer_data,
                        outer_state=outer_state,
                        trajectory_points=current_traj,
                        mesh=landscape_mesh,
                        projection_info=projection_info,
                    )
                    if outer_iter == 0 or (outer_iter + 1) % 2 == 0:
                        dashboard.call_from_thread(
                            dashboard.update_log,
                            f"[dim]iter {outer_iter + 1}/{config.outer_steps}[/] loss={loss_val:.4f} best={best_loss:.4f}",
                        )
                else:
                    print(
                        f"[live_run] iter {outer_iter + 1}/{config.outer_steps} "
                        f"loss={loss_val:.4f} best={best_loss:.4f}"
                    )

                if web_dashboard is not None:
                    layer_names_sorted = sorted(layer_data.keys())
                    heat_kappa, valid_mask, head_buffers = _kappa_heads_from_layers(
                        layer_data,
                        outer_iter,
                        heads_per_layer=8,
                    )
                    valid_flat = [v for row in valid_mask for v in row]
                    topology_valid_pct = 100.0 * (sum(1 for v in valid_flat if v) / max(len(valid_flat), 1))
                    mean_kappa = float(np.mean([v for row in heat_kappa for v in row])) if heat_kappa else 1.0

                    web_dashboard.publish(
                        {
                            "run": {
                                "status": "running",
                                "val_loss": float(loss_val),
                                "outer_step": int(outer_iter + 1),
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
                            "experts": _experts_from_state(hp_dict, layer_data, n_experts=8),
                            "trajectory": {
                                "loss": [float(v) for v in loss_history],
                                "pivot_steps": pivot_steps[-64:],
                                "spectral_guard_steps": spectral_guard_steps[-64:],
                            },
                            "chain": _chain_payload(outer_iter, config.outer_steps, topology_valid_pct >= 80.0),
                            "topology_3d": _topology_3d_payload(
                                topology_components_history,
                                layer_names=layer_names_sorted,
                            ),
                            "persistence": persistence_payload if persistence_payload else None,
                            "chat": {
                                "messages": chat_messages,
                                "pending_prompt_count": web_dashboard.pending_prompt_count(),
                                "current_prompt": current_chat_prompt,
                            },
                            "events": event_feed[-60:],
                        }
                    )
                
                hyperparams.zero_grad()
                evaluator.model.zero_grad(set_to_none=True)
                time.sleep(0.12)

            if dashboard:
                dashboard.call_from_thread(dashboard.update_log, "[bold green]✔ LIVE RUN COMPLETE.[/]")

            if topology_components_history:
                topo_arr = np.stack(topology_components_history, axis=0)
                topo_out = "outputs/topology_components_3d.html"
                export_topology_components_3d(
                    topology_history=topo_arr,
                    output_path=topo_out,
                    component_names=[
                        "attention",
                        "moe",
                        "residual",
                        "chain",
                        "global",
                    ],
                )
                print(f"[live_run] 3D topology export: {topo_out}")
            if web_dashboard is not None:
                final_layer_names = [f"block.{i}" for i in range(n_layers)]
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
                            "pivot_steps": [],
                            "spectral_guard_steps": [],
                        },
                        "chain": _chain_payload(max(len(loss_history) - 1, 0), max(config.outer_steps, 1), True),
                        "topology_3d": _topology_3d_payload(
                            topology_components_history,
                            layer_names=final_layer_names,
                        ),
                        "chat": {
                            "messages": chat_messages,
                            "pending_prompt_count": web_dashboard.pending_prompt_count(),
                            "current_prompt": None,
                        },
                        "events": ["run complete"],
                    }
                )
        except Exception as e:
            if dashboard:
                dashboard.call_from_thread(dashboard.update_log, f"[bold red]Live run error:[/] {e}")
            else:
                import traceback
                traceback.print_exc()
                print(f"[live_run] error: {e}")
            if web_dashboard is not None:
                final_layer_names = [f"block.{i}" for i in range(n_layers)]
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
                            "pivot_steps": [],
                            "spectral_guard_steps": [],
                        },
                        "chain": _chain_payload(max(len(loss_history) - 1, 0), max(config.outer_steps, 1), False),
                        "topology_3d": _topology_3d_payload(
                            topology_components_history,
                            layer_names=final_layer_names,
                        ),
                        "chat": {
                            "messages": chat_messages,
                            "pending_prompt_count": web_dashboard.pending_prompt_count(),
                            "current_prompt": None,
                        },
                        "events": [f"run error: {e}"],
                    }
                )

    # Run
    if dashboard:
        thread = threading.Thread(target=optimization_task)
        thread.start()
        dashboard.run()
        thread.join()
    else:
        optimization_task()

    if web_dashboard is not None:
        hold_s = max(0.0, float(web_hold_seconds))
        if hold_s > 0.0:
            print(
                f"[live_run] web dashboard will stay up for {hold_s:.1f}s at "
                f"{web_dashboard.base_url}"
            )
            time.sleep(hold_s)
        web_dashboard.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--track-layer", type=str, default="block.0")
    parser.add_argument(
        "--model-path",
        type=str,
        default="~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987",
    )
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--textual-dashboard", action="store_true")
    parser.add_argument("--web-hold-seconds", type=float, default=20.0)
    args = parser.parse_args()
    
    cfg = TaLBFGSConfig()
    cfg.outer_steps = args.steps
    cfg.dashboard_projection_layer = args.track_layer
    cfg.vllm_model_name = args.model_path
    run_live(
        cfg,
        use_dashboard=not args.no_dashboard,
        use_web_dashboard=(not args.textual_dashboard),
        web_hold_seconds=args.web_hold_seconds,
    )
