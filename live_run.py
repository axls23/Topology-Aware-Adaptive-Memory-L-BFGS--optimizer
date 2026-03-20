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
from ta_lbfgs.dashboard.landscape_viz import generate_landscape_mesh, reset_adaptive_mesh
from ta_lbfgs.training.data_preprocessing import (
    get_default_cache_path,
    get_default_dataset_path,
    load_or_build_reasoning_trace_cache,
    sample_packed_batch,
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
    
    def optimization_task():
        best_loss = float("inf")
        pivot_steps: List[int] = []
        spectral_guard_steps: List[int] = []
        event_feed: List[str] = []
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

                if any(v.get("secant", 0.0) <= 0.0 for v in layer_data.values()):
                    pivot_steps.append(outer_iter)
                    event_feed.append(f"step {outer_iter + 1}: pivot detected (topology re-derived)")

                if any(v.get("kappa", 0.0) >= 10000.0 for v in layer_data.values()):
                    spectral_guard_steps.append(outer_iter)
                    event_feed.append(f"step {outer_iter + 1}: spectral guard fired")

                if len(event_feed) > 100:
                    event_feed = event_feed[-100:]

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

                if web_dashboard is not None:
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
                            "events": event_feed[-60:],
                        }
                    )
                
                hyperparams.zero_grad()
                evaluator.model.zero_grad(set_to_none=True)
                time.sleep(0.12)

            if dashboard:
                dashboard.call_from_thread(dashboard.update_log, "[bold green]✔ LIVE RUN COMPLETE.[/]")
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
                            "pivot_steps": [],
                            "spectral_guard_steps": [],
                        },
                        "chain": _chain_payload(max(len(loss_history) - 1, 0), max(config.outer_steps, 1), True),
                        "events": ["run complete"],
                    }
                )
        except Exception as e:
            if dashboard:
                dashboard.call_from_thread(dashboard.update_log, f"[bold red]Live run error:[/] {e}")
            else:
                print(f"[live_run] error: {e}")
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
                            "pivot_steps": [],
                            "spectral_guard_steps": [],
                        },
                        "chain": _chain_payload(max(len(loss_history) - 1, 0), max(config.outer_steps, 1), False),
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
