"""
ta-LBFGS Configuration.

Dataclass-based configuration for the Topology-Aware Adaptive-Memory
L-BFGS optimizer. Adapted from Chronoscope's config pattern.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class TaLBFGSConfig:
    """Configuration for the ta-LBFGS optimizer and its subsystems."""

    # ── L-BFGS Core ──────────────────────────────────────────────────
    lbfgs_memory_base: int = 5
    lbfgs_memory_min: int = 3
    lbfgs_memory_max: int = 20
    lbfgs_lr: float = 1.0
    lbfgs_line_search: str = "Wolfe"  # 'Wolfe', 'Armijo', 'None'
    lbfgs_damping: bool = True
    lbfgs_damping_eps: float = 0.2

    # ── Hyperparameter Search Space ──────────────────────────────────
    n_layers: int = 4                  # number of transformer layers
    initial_lr: float = 1e-4           # initial per-layer learning rate
    initial_weight_decay: float = 1e-2 # initial per-layer weight decay
    initial_dropout: float = 0.1       # initial per-layer dropout rate
    initial_attn_temp: float = 1.0     # initial per-layer attention temperature
    initial_label_smoothing: float = 0.1  # global label smoothing ε
    enable_moe_routing: bool = False   # enable MoE routing temperature tuning
    hyperparams_to_optimize: List[str] = field(
        default_factory=lambda: [
            "lr", "wd", "dropout", "attn_temp", "label_smoothing"
        ]
    )

    # ── Bilevel Optimization ─────────────────────────────────────────
    inner_steps: int = 10
    outer_steps: int = 50
    inner_optimizer: str = "SGD"  # inner loop optimizer for model weights
    inner_lr: float = 1e-3

    # ── Topology / Saddle Detection ──────────────────────────────────
    condition_svd_components: int = 8
    secant_threshold: float = 0.0
    perturbation_scale: float = 0.01
    distance_threshold: float = 0.5  # Euler characteristic adjacency
    gradient_window_size: int = 10   # sliding window for topology analysis

    # ── Adaptive Memory ──────────────────────────────────────────────
    adaptive_memory_enabled: bool = True

    # ── Hypergradient Computation ────────────────────────────────────
    hypergradient_method: str = "CG"  # 'CG', 'Neumann', 'direct'
    cg_max_iter: int = 10
    cg_tol: float = 1e-5
    neumann_terms: int = 5
    # Hybrid mode: exact IFT on rotating parameter shard + approximation elsewhere.
    hybrid_hypergradient: bool = False
    hybrid_shard_fraction: float = 0.125
    hybrid_rotation_steps: int = 1

    # ── vLLM Inference Engine ────────────────────────────────────────
    vllm_api_base: str = "http://localhost:8000/v1"
    vllm_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    vllm_max_tokens: int = 512
    vllm_temperature: float = 0.0

    # ── Optuna Baseline ──────────────────────────────────────────────
    optuna_n_trials: int = 50
    optuna_sampler: str = "TPE"  # 'TPE', 'CMA-ES'

    # ── Dashboard ────────────────────────────────────────────────────
    dashboard_refresh_rate: float = 0.25  # seconds
    dashboard_sparkline_width: int = 30
    dashboard_projection_layer: str = "block.0"
    dashboard_projection_components: int = 3
    dashboard_projection_sketch_dim: int = 24
    dashboard_projection_forgetting_factor: float = 0.98
    dashboard_projection_warning_threshold: float = 0.80

    # ── Device / Precision ───────────────────────────────────────────
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float32"

    # ── Output ───────────────────────────────────────────────────────
    output_dir: str = "outputs"
    export_trajectory_3d: bool = True

    def get_torch_dtype(self) -> torch.dtype:
        """Convert string dtype to torch dtype."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "float64": torch.float64,
        }
        return dtype_map.get(self.dtype, torch.float32)
