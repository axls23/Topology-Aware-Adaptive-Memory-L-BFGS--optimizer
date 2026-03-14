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

from transformers import AutoModelForCausalLM, AutoTokenizer

from ta_lbfgs.config import TaLBFGSConfig
from ta_lbfgs.core.hyperparameters import DifferentiableHyperparameters
from ta_lbfgs.dashboard.textual_dashboard import TextualDashboard
from ta_lbfgs.dashboard.online_rsvd import LayerwiseOnlineRSVD
from ta_lbfgs.topology.adaptive_memory import compute_memory_size
from ta_lbfgs.topology.condition import estimate_condition_from_grad_history
from ta_lbfgs.dashboard.landscape_viz import generate_landscape_mesh, reset_adaptive_mesh
from ta_lbfgs.training.data_preprocessing import (
    get_default_cache_path,
    get_default_dataset_path,
    load_or_build_reasoning_trace_cache,
    sample_packed_batch,
)

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
            model_path = (
                "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/"
                "snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
            )

        resolved = model_path
        if resolved.startswith("~"):
            import os
            resolved = os.path.expanduser(resolved)

        self.tokenizer = AutoTokenizer.from_pretrained(resolved, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            resolved,
            local_files_only=True,
            torch_dtype=torch.float32,
        ).to(self.device)
        self.model.train()

        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            self.blocks = self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            self.blocks = self.model.transformer.h
        else:
            self.blocks = []

        self.n_layers = len(self.blocks)
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

    def compute_loss(self, hyperparams: DifferentiableHyperparameters) -> torch.Tensor:
        batch = self._sample_batch()
        outputs = self.model(**batch)
        base_loss = outputs.loss

        # Keep hyperparameters connected to loss for outer updates while preserving
        # real model gradients for layerwise topology signals.
        lr_term = torch.stack([hyperparams.get_layer_lr(i) for i in range(self.n_layers)]).mean()
        wd_term = torch.stack([hyperparams.get_layer_wd(i) for i in range(self.n_layers)]).mean()
        return base_loss * (1.0 + 0.05 * lr_term) + 0.01 * wd_term

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

def run_live(config: TaLBFGSConfig, use_dashboard: bool = True):
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
    
    dashboard = TextualDashboard() if use_dashboard else None
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
    
    def optimization_task():
        best_loss = float("inf")
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

                val_loss = evaluator.compute_loss(hyperparams)
                loss_val = val_loss.item()
                loss_history.append(loss_val)
                
                if loss_val < best_loss:
                    best_loss = loss_val
                
                val_loss.backward()
                
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

                current_traj = projected_history if projected_history else [
                    [h["lr"][0], h["wd"][0], l] for h, l in zip(hp_history, loss_history)
                ]

                landscape_mesh = generate_landscape_mesh(
                    None, hyperparams, layer_curvature=layer_data, loss=loss_val
                )

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
                
                hyperparams.zero_grad()
                evaluator.model.zero_grad(set_to_none=True)
                time.sleep(0.12)

            if dashboard:
                dashboard.call_from_thread(dashboard.update_log, "[bold green]✔ LIVE RUN COMPLETE.[/]")
        except Exception as e:
            if dashboard:
                dashboard.call_from_thread(dashboard.update_log, f"[bold red]Live run error:[/] {e}")
            else:
                print(f"[live_run] error: {e}")

    # Run
    if dashboard:
        thread = threading.Thread(target=optimization_task)
        thread.start()
        dashboard.run()
        thread.join()
    else:
        optimization_task()

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
    args = parser.parse_args()
    
    cfg = TaLBFGSConfig()
    cfg.outer_steps = args.steps
    cfg.dashboard_projection_layer = args.track_layer
    cfg.vllm_model_name = args.model_path
    run_live(cfg, use_dashboard=not args.no_dashboard)
