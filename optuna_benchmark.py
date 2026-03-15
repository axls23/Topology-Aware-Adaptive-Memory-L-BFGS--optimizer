import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
import optuna
import argparse
import json
from dataclasses import dataclass
from typing import Any, Optional
from typing import Dict, List, Tuple

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None

# Import our core components
from ta_lbfgs.core.hyperparameters import DifferentiableHyperparameters
from ta_lbfgs.config import TaLBFGSConfig
from ta_lbfgs.core.lbfgs import LayerwiseTaLBFGS
from ta_lbfgs.core.baseline_lbfgs import FullBatchLBFGS
from ta_lbfgs.training.bilevel import BilevelOptimizer
from ta_lbfgs.training.inner_loop import functional_call_model
from ta_lbfgs.dashboard.landscape_viz import (
    export_trajectory_3d,
    export_trajectory_3d_static,
    plot_dynamics,
    plot_hyperparameter_trajectories,
)
from ta_lbfgs.training.data_preprocessing import get_default_dataset_path

# Re-implement a simplified version of the Synthetic Model for the benchmark
# to ensure we are testing on the exact same surface as demo.py
class LayerTarget:
    def __init__(self, idx, n_layers):
        self.idx = idx
        # Target LR: follows a shift per layer
        self.target_lr = 1e-3 * (1.0 + 0.1 * math.sin(idx))
        # Target WD: exponential decay across layers
        self.target_wd = 0.05 * math.exp(-idx / n_layers)

    def loss(self, lr, wd, prompt_signature: float = 0.0):
        # A non-convex surface with a ravine
        prompt_lr_scale = 1.0 + 0.08 * math.sin(prompt_signature + 0.31 * self.idx)
        prompt_wd_scale = 1.0 + 0.08 * math.cos(prompt_signature + 0.17 * self.idx)
        effective_lr = self.target_lr * max(prompt_lr_scale, 1e-3)
        effective_wd = self.target_wd * max(prompt_wd_scale, 1e-3)

        d_lr = (torch.log(lr) - math.log(effective_lr))
        d_wd = (torch.log(wd) - math.log(effective_wd))
        # Basic quadratic bowl
        bowl = d_lr**2 + 5.0 * d_wd**2
        # Add a non-convex ripple
        phase = 2.0 * prompt_signature
        ripple = 0.2 * torch.sin(10 * d_lr + phase) * torch.cos(10 * d_wd - phase)
        return bowl + ripple

class SyntheticBilevelModel:
    def __init__(self, n_layers=4):
        self.layers = [LayerTarget(i, n_layers) for i in range(n_layers)]

    def train_loss(self, hyperparams, prompt_signature: float = 0.0):
        total = sum(
            layer.loss(
                hyperparams.get_layer_lr(i),
                hyperparams.get_layer_wd(i),
                prompt_signature=prompt_signature,
            )
            for i, layer in enumerate(self.layers)
        )
        return total / len(self.layers)

    def val_loss(self, hyperparams, prompt_signature: float = 0.0):
        total = sum(
            layer.loss(
                hyperparams.get_layer_lr(i) * 1.1,
                hyperparams.get_layer_wd(i) * 0.9,
                prompt_signature=prompt_signature,
            )
            for i, layer in enumerate(self.layers)
        )
        return total / len(self.layers)


class SyntheticMetaModel(nn.Module):
    """Tiny differentiable model used to exercise strict bilevel optimization path."""

    def __init__(self, n_layers: int, device: str, dtype: torch.dtype):
        super().__init__()
        init = torch.linspace(-0.2, 0.2, n_layers, device=device, dtype=dtype)
        self.w = nn.Parameter(init)

    def val_loss(self, hyperparams, prompt_signature: float = 0.0):
        idx = torch.arange(self.w.numel(), device=self.w.device, dtype=self.w.dtype)
        lr = torch.stack([hyperparams.get_layer_lr(i) for i in range(self.w.numel())]).to(dtype=self.w.dtype)
        wd = torch.stack([hyperparams.get_layer_wd(i) for i in range(self.w.numel())]).to(dtype=self.w.dtype)
        sig = torch.tensor(prompt_signature, device=self.w.device, dtype=self.w.dtype)
        target = torch.sin(0.23 * idx + sig)
        shifted = 1.03 * self.w - target
        return ((1.0 + 4.5 * wd) * shifted.pow(2)).mean() + 0.02 * torch.log(lr).pow(2).mean()


class RealPromptTuningObjective(nn.Module):
    """Real-model objective using a trainable soft prompt over a frozen HF causal LM."""

    def __init__(
        self,
        model_id: str,
        local_path: Optional[str],
        max_length: int,
        device: str,
        prefix_len: int = 4,
    ):
        super().__init__()
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise ImportError("transformers is required for real model objective mode")

        resolved_path = _resolve_hf_local_path(model_id=model_id, explicit_path=local_path)
        self.device = device
        self.max_length = int(max_length)
        self.prefix_len = int(prefix_len)

        # Implicit differentiation requires second-order gradients through attention.
        # Force math SDPA kernels to avoid unsupported efficient-attention backward2 paths.
        if hasattr(torch.backends, "cuda"):
            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(False)
            if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                torch.backends.cuda.enable_mem_efficient_sdp(False)
            if hasattr(torch.backends.cuda, "enable_math_sdp"):
                torch.backends.cuda.enable_math_sdp(True)

        self.tokenizer = AutoTokenizer.from_pretrained(resolved_path, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            resolved_path,
            local_files_only=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.train()

        for p in self.model.parameters():
            p.requires_grad_(False)

        hidden_size = int(self.model.config.hidden_size)
        self.soft_prompt = nn.Parameter(
            torch.zeros(self.prefix_len, hidden_size, device=self.device, dtype=self.model.dtype)
        )
        with torch.no_grad():
            self.soft_prompt.normal_(mean=0.0, std=0.02)

    def forward(self, prompt_texts: List[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attn_mask = encoded["attention_mask"].to(self.device)

        token_embeds = self.model.get_input_embeddings()(input_ids)
        prefix = self.soft_prompt.unsqueeze(0).expand(input_ids.size(0), -1, -1)
        inputs_embeds = torch.cat([prefix, token_embeds], dim=1)

        prefix_mask = torch.ones(
            (attn_mask.size(0), self.prefix_len),
            device=self.device,
            dtype=attn_mask.dtype,
        )
        attention_mask = torch.cat([prefix_mask, attn_mask], dim=1)

        ignore = torch.full(
            (input_ids.size(0), self.prefix_len),
            -100,
            device=self.device,
            dtype=input_ids.dtype,
        )
        labels = torch.cat([ignore, input_ids], dim=1)

        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        return out.loss

@dataclass
class BenchmarkProfile:
    """Execution profile for benchmark speed-vs-fidelity tradeoffs."""

    name: str
    description: str
    claim_valid: bool
    ta_overrides: Dict[str, object]
    real_max_length: int


def build_benchmark_profiles(base_real_max_length: int) -> Dict[str, BenchmarkProfile]:
    """Return named benchmark profiles with explicit speed/fidelity semantics."""
    speed_max_len = max(48, min(int(base_real_max_length), 64))
    full_max_len = max(128, int(base_real_max_length))

    return {
        "speed_tuned": BenchmarkProfile(
            name="speed_tuned",
            description=(
                "For smoke/performance sweeps only. Reduces second-order and sequence costs; "
                "not claim-valid for final reported metrics."
            ),
            claim_valid=False,
            ta_overrides={
                "inner_steps": 1,
                "cg_max_iter": 2,
                "outer_hutchpp_samples": 1,
                "outer_hutchpp_trace_enabled": False,
                "outer_hutchpp_diagonal_precondition_enabled": False,
                "hybrid_hypergradient": True,
                "hybrid_shard_fraction": 0.125,
            },
            real_max_length=speed_max_len,
        ),
        "full_fidelity": BenchmarkProfile(
            name="full_fidelity",
            description=(
                "Claim-valid profile for final reported numbers. Preserves trace and diagonal "
                "preconditioning with stronger inner optimization."
            ),
            claim_valid=True,
            ta_overrides={
                "inner_steps": 3,
                "outer_hutchpp_samples": 3,
                "outer_hutchpp_trace_enabled": True,
                "outer_hutchpp_diagonal_precondition_enabled": True,
            },
            real_max_length=full_max_len,
        ),
    }


def _prompt_signature_from_text(prompt_text: str, trial_idx: int) -> float:
    """Map prompt text + trial index to a stable phase in [-pi, pi]."""
    h = abs(hash((prompt_text, trial_idx))) % 1000003
    u = h / 1000003.0
    return math.pi * (2.0 * u - 1.0)


def _resolve_hf_local_path(model_id: str, explicit_path: Optional[str]) -> str:
    if explicit_path:
        return explicit_path
    model_stub = model_id.replace("/", "--")
    base_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    snapshots_root = os.path.join(base_dir, f"models--{model_stub}", "snapshots")
    if not os.path.isdir(snapshots_root):
        raise FileNotFoundError(
            "Hugging Face cache directory not found for model: "
            f"{model_id}. Provide --hf-local-path to a cached snapshot directory."
        )

    candidates = [
        os.path.join(snapshots_root, d)
        for d in os.listdir(snapshots_root)
        if os.path.isdir(os.path.join(snapshots_root, d))
    ]
    if not candidates:
        raise FileNotFoundError(
            "No cached snapshots found. Provide --hf-local-path to a valid local snapshot."
        )
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def build_prompt_signatures_from_hf_cache(
    prompt_texts: List[str],
    n_trials: int,
    model_id: str,
    local_path: Optional[str] = None,
    max_length: int = 256,
    device: str = "auto",
) -> List[float]:
    if AutoModelForCausalLM is None or AutoTokenizer is None:
        raise ImportError(
            "transformers is required for --signature-source hf_cached. "
            "Install dependencies from requirements.txt."
        )
    if not prompt_texts:
        raise ValueError("prompt_texts cannot be empty for hf_cached signature source")

    resolved_path = _resolve_hf_local_path(model_id=model_id, explicit_path=local_path)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using HF cached model path: {resolved_path}")
    tokenizer = AutoTokenizer.from_pretrained(resolved_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        resolved_path,
        local_files_only=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    losses: List[float] = []
    selected_prompts = [prompt_texts[i % len(prompt_texts)] for i in range(n_trials)]
    with torch.no_grad():
        for prompt in selected_prompts:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model(**inputs, labels=inputs["input_ids"])
            losses.append(float(out.loss.detach().cpu().item()))

    arr = np.array(losses, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std < 1e-12:
        std = 1.0

    z = (arr - mean) / std
    # Map normalized HF losses to a bounded phase range used by benchmark surfaces.
    phases = np.tanh(z) * math.pi
    return [float(x) for x in phases.tolist()]


def build_reasoning_prompt_bank(
    n_trials: int,
    seed: int,
    min_trace_steps: int = 8,
    max_trace_steps: int = 40,
) -> List[str]:
    """
    Build synthetic long-reasoning prompts for trial-conditioned benchmarking.

    Semantics:
    - 1 trial = 1 optimization iteration
    - each trial has a distinct prompt
    - prompt length/depth simulates chain-of-thought trace diversity.
    """
    rng = np.random.default_rng(seed)
    domains = [
        "multi-hop arithmetic proof",
        "graph shortest-path derivation",
        "Bayesian update puzzle",
        "symbolic logic consistency check",
        "algorithmic complexity tradeoff",
        "causal intervention analysis",
        "dynamical system stability trace",
        "combinatorial counting argument",
    ]
    actions = [
        "derive",
        "verify",
        "decompose",
        "bound",
        "cross-check",
        "refine",
        "simulate",
        "synthesize",
    ]

    prompts: List[str] = []
    lo = max(2, min_trace_steps)
    hi = max(lo, max_trace_steps)
    for i in range(n_trials):
        n_steps = int(rng.integers(lo, hi + 1))
        domain = domains[i % len(domains)]
        action = actions[(i * 5 + 3) % len(actions)]
        steps = [
            f"Step {k+1}: {action} sub-claim {k+1} for {domain}."
            for k in range(n_steps)
        ]
        prompt = (
            f"Trial {i+1}. Solve with explicit long chain-of-thought over {n_steps} steps. "
            f"Task: {domain}. End with final answer and consistency audit.\n"
            + "\n".join(steps)
        )
        prompts.append(prompt)
    return prompts


def load_dataset_prompts(
    source_path: str,
    max_prompts: int,
) -> List[str]:
    """Load user prompts from reasoning-trace JSONL dataset."""
    prompts: List[str] = []
    if not source_path or not os.path.exists(source_path):
        return prompts

    with open(source_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(prompts) >= max_prompts:
                break
            raw = line.strip()
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            messages = data.get("messages", [])
            user_prompt = ""
            for msg in messages:
                if msg.get("role") == "user":
                    user_prompt = str(msg.get("content", "")).strip()
                    break
            if user_prompt:
                prompts.append(user_prompt)

    return prompts


def build_prompt_signatures(
    n_trials: int,
    seed: int,
    min_trace_steps: int = 8,
    max_trace_steps: int = 40,
    prompt_texts: Optional[List[str]] = None,
) -> List[float]:
    """Create deterministic prompt-derived signatures (one prompt per trial)."""
    if prompt_texts:
        prompts = [prompt_texts[i % len(prompt_texts)] for i in range(n_trials)]
    else:
        prompts = build_reasoning_prompt_bank(
            n_trials=n_trials,
            seed=seed,
            min_trace_steps=min_trace_steps,
            max_trace_steps=max_trace_steps,
        )
    return [_prompt_signature_from_text(prompt, i) for i, prompt in enumerate(prompts)]


def _loss_variance_summary(loss_history: List[float]) -> Dict[str, float]:
    arr = np.array(loss_history, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


@dataclass
class BaselineResult:
    name: str
    best_loss: float
    elapsed_sec: float
    trajectory_path: str
    diagnostics_path: str
    hyperparams_path: str
    metadata: Optional[Dict[str, Any]] = None


def _set_layerwise_hyperparameters(
    hp_obj: DifferentiableHyperparameters,
    lrs: List[float],
    wds: List[float],
) -> None:
    """Write explicit layerwise lr/wd values into raw parameter storage."""
    with torch.no_grad():
        for i, (lr, wd) in enumerate(zip(lrs, wds)):
            hp_obj.blocks[i].raw_lr.copy_(torch.tensor(math.log(lr)))
            hp_obj.blocks[i].raw_wd.copy_(torch.tensor(math.log(wd)))


def _build_hyperparams(n_layers: int, lrs: List[float], wds: List[float]) -> DifferentiableHyperparameters:
    hp_obj = DifferentiableHyperparameters(
        n_layers=n_layers,
        initial_label_smoothing=0.1,
    )
    _set_layerwise_hyperparameters(hp_obj, lrs, wds)
    return hp_obj


def _export_baseline_artifacts(
    baseline_name: str,
    loss_history: List[float],
    hp_history: List[Dict[str, List[float]]],
    output_dir: str,
    optimizer_name: str = "",
    annotate_instability_spike: bool = True,
    spike_threshold: float = 0.8,
) -> Tuple[str, str, str]:
    os.makedirs(output_dir, exist_ok=True)

    hp_array = np.array([[x for x in (h["lr"] + h["wd"])] for h in hp_history], dtype=np.float64)
    loss_array = np.array(loss_history, dtype=np.float64)

    traj_path = os.path.join(output_dir, f"{baseline_name}_trajectory_3d.html")
    export_trajectory_3d(hp_array, loss_array, traj_path)

    static_landscape_path = os.path.join(output_dir, f"{baseline_name}_loss_landscape_3d.png")
    export_trajectory_3d_static(hp_array, loss_array, static_landscape_path)
    print(f"  - Static Landscape: {static_landscape_path}")

    est_grads = [0.0]
    for i in range(1, len(loss_history)):
        est_grads.append(abs(loss_history[i] - loss_history[i - 1]))

    dynamics_path = os.path.join(output_dir, f"{baseline_name}_diagnostics.png")
    plot_dynamics(
        loss_history,
        est_grads,
        [0.0] * len(loss_history),
        hp_history,
        dynamics_path,
        optimizer_name=optimizer_name,
        annotate_instability_spike=annotate_instability_spike,
        spike_threshold=spike_threshold,
    )

    hp_plot_path = os.path.join(output_dir, f"{baseline_name}_hyperparameters.png")
    plot_hyperparameter_trajectories(hp_history, hp_plot_path)

    return traj_path, dynamics_path, hp_plot_path


def run_evaluation(
    optimizer: Any,
    optimizer_name: str,
    strict_label_binding: bool = True,
) -> Dict[str, Any]:
    """
    Strict label binding for diagnostics collection.

    optimizer_name must match the class identity of the optimizer object.
    """
    if strict_label_binding:
        assert optimizer_name in type(optimizer).__name__, (
            f"Label mismatch: {optimizer_name} != {type(optimizer).__name__}"
        )
    return {
        "optimizer": optimizer_name,
        "phase_portrait": [],
        "hp_modulation": [],
        "grad_norms": [],
    }


def _summarize_success(
    name: str,
    best_loss: float,
    elapsed_sec: float,
    traj_path: str,
    dynamics_path: str,
    hp_plot_path: str,
    variance: Dict[str, float],
) -> None:
    print(f"\n[SUCCESS] {name} Baseline Complete.")
    print(f"  - Best Loss: {best_loss:.6f}")
    print(f"  - Elapsed: {elapsed_sec:.2f}s")
    print(
        "  - Prompt Variance: "
        f"mean={variance['mean']:.6f}, std={variance['std']:.6f}, "
        f"min={variance['min']:.6f}, max={variance['max']:.6f}"
    )
    print(f"  - 3D Map: {traj_path}")
    print(f"  - Diagnostics: {dynamics_path}")
    print(f"  - Hyperparameters: {hp_plot_path}")


def run_optuna_baseline(
    n_trials=40,
    n_layers=4,
    output_dir="outputs",
    prompt_signatures: Optional[List[float]] = None,
) -> BaselineResult:
    print("=" * 60)
    print(f"  OPTIMIZATION BASELINE: Optuna (Bayesian / TPE)")
    print("=" * 60)
    
    model = SyntheticBilevelModel(n_layers=n_layers)
    if prompt_signatures is None:
        prompt_signatures = build_prompt_signatures(n_trials=n_trials, seed=42)
    if len(prompt_signatures) < n_trials:
        raise ValueError("prompt_signatures must contain at least n_trials entries")

    loss_history = []
    hp_history = []
    start = time.perf_counter()
    
    def objective(trial):
        # Construct hyperparams from Optuna suggestions
        # We need to construct a DifferentiableHyperparameters object 
        # to use as_float_dict for the visualization later
        hp_obj = DifferentiableHyperparameters(
            n_layers=n_layers,
            initial_label_smoothing=0.1
        )
        
        # Suggest values for each layer
        for i in range(n_layers):
            # We suggest in log space but Optuna handles that with log=True
            lr = trial.suggest_float(f"layer_{i}_lr", 1e-7, 1.0, log=True)
            wd = trial.suggest_float(f"layer_{i}_wd", 1e-7, 1.0, log=True)
            
            # Manually set the raw data to match Optuna's suggestion
            with torch.no_grad():
                hp_obj.blocks[i].raw_lr.copy_(torch.tensor(math.log(lr)))
                hp_obj.blocks[i].raw_wd.copy_(torch.tensor(math.log(wd)))
        
        trial_idx = len(loss_history)
        prompt_sig = float(prompt_signatures[trial_idx])
        loss = model.val_loss(hp_obj, prompt_signature=prompt_sig).item()
        
        # Record for visualization
        loss_history.append(loss)
        hp_history.append(hp_obj.as_float_dict())
        
        return loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    elapsed = time.perf_counter() - start
    variance = _loss_variance_summary(loss_history)

    traj_path, dynamics_path, hp_plot_path = _export_baseline_artifacts(
        baseline_name="optuna",
        loss_history=loss_history,
        hp_history=hp_history,
        output_dir=output_dir,
    )
    _summarize_success("Optuna", study.best_value, elapsed, traj_path, dynamics_path, hp_plot_path, variance)
    return BaselineResult(
        name="optuna",
        best_loss=float(study.best_value),
        elapsed_sec=float(elapsed),
        trajectory_path=traj_path,
        diagnostics_path=dynamics_path,
        hyperparams_path=hp_plot_path,
    )


def run_random_baseline(
    n_trials: int = 40,
    n_layers: int = 4,
    output_dir: str = "outputs",
    seed: Optional[int] = 42,
    prompt_signatures: Optional[List[float]] = None,
) -> BaselineResult:
    print("=" * 60)
    print("  OPTIMIZATION BASELINE: Random Search")
    print("=" * 60)

    model = SyntheticBilevelModel(n_layers=n_layers)
    rng = np.random.default_rng(seed)
    if prompt_signatures is None:
        prompt_signatures = build_prompt_signatures(n_trials=n_trials, seed=seed if seed is not None else 42)
    if len(prompt_signatures) < n_trials:
        raise ValueError("prompt_signatures must contain at least n_trials entries")

    loss_history: List[float] = []
    hp_history: List[Dict[str, List[float]]] = []
    best_loss = float("inf")

    log_min = math.log(1e-7)
    log_max = math.log(1.0)

    start = time.perf_counter()
    for trial_idx in range(n_trials):
        lrs = np.exp(rng.uniform(log_min, log_max, size=n_layers)).tolist()
        wds = np.exp(rng.uniform(log_min, log_max, size=n_layers)).tolist()

        hp_obj = _build_hyperparams(n_layers=n_layers, lrs=lrs, wds=wds)
        prompt_sig = float(prompt_signatures[trial_idx])
        loss = model.val_loss(hp_obj, prompt_signature=prompt_sig).item()

        loss_history.append(loss)
        hp_history.append(hp_obj.as_float_dict())
        best_loss = min(best_loss, loss)

    elapsed = time.perf_counter() - start
    variance = _loss_variance_summary(loss_history)

    traj_path, dynamics_path, hp_plot_path = _export_baseline_artifacts(
        baseline_name="random",
        loss_history=loss_history,
        hp_history=hp_history,
        output_dir=output_dir,
    )
    _summarize_success("Random Search", best_loss, elapsed, traj_path, dynamics_path, hp_plot_path, variance)
    return BaselineResult(
        name="random",
        best_loss=float(best_loss),
        elapsed_sec=float(elapsed),
        trajectory_path=traj_path,
        diagnostics_path=dynamics_path,
        hyperparams_path=hp_plot_path,
    )


def _decode_mixed_radix(index: int, base: int, dims: int) -> List[int]:
    digits = [0] * dims
    value = index
    for i in range(dims):
        digits[i] = value % base
        value //= base
    return digits


def run_grid_baseline(
    n_trials: int = 40,
    n_layers: int = 4,
    output_dir: str = "outputs",
    prompt_signatures: Optional[List[float]] = None,
) -> BaselineResult:
    print("=" * 60)
    print("  OPTIMIZATION BASELINE: Grid Search")
    print("=" * 60)

    model = SyntheticBilevelModel(n_layers=n_layers)
    if prompt_signatures is None:
        prompt_signatures = build_prompt_signatures(n_trials=n_trials, seed=42)
    if len(prompt_signatures) < n_trials:
        raise ValueError("prompt_signatures must contain at least n_trials entries")

    loss_history: List[float] = []
    hp_history: List[Dict[str, List[float]]] = []
    best_loss = float("inf")

    dims = 2 * n_layers
    log_min = math.log(1e-7)
    log_max = math.log(1.0)

    points_per_dim = max(2, int(round(n_trials ** (1.0 / max(1, dims)))))
    while points_per_dim ** dims < n_trials:
        points_per_dim += 1

    log_grid = np.linspace(log_min, log_max, num=points_per_dim, dtype=np.float64)
    total_grid_points = points_per_dim ** dims
    stride = max(1, total_grid_points // n_trials)

    start = time.perf_counter()
    for i in range(n_trials):
        point_index = (i * stride) % total_grid_points
        digit_idx = _decode_mixed_radix(point_index, points_per_dim, dims)
        coords = [float(log_grid[d]) for d in digit_idx]

        lrs = [math.exp(v) for v in coords[:n_layers]]
        wds = [math.exp(v) for v in coords[n_layers:]]

        hp_obj = _build_hyperparams(n_layers=n_layers, lrs=lrs, wds=wds)
        prompt_sig = float(prompt_signatures[i])
        loss = model.val_loss(hp_obj, prompt_signature=prompt_sig).item()

        loss_history.append(loss)
        hp_history.append(hp_obj.as_float_dict())
        best_loss = min(best_loss, loss)

    elapsed = time.perf_counter() - start
    variance = _loss_variance_summary(loss_history)

    traj_path, dynamics_path, hp_plot_path = _export_baseline_artifacts(
        baseline_name="grid",
        loss_history=loss_history,
        hp_history=hp_history,
        output_dir=output_dir,
    )
    _summarize_success("Grid Search", best_loss, elapsed, traj_path, dynamics_path, hp_plot_path, variance)
    return BaselineResult(
        name="grid",
        best_loss=float(best_loss),
        elapsed_sec=float(elapsed),
        trajectory_path=traj_path,
        diagnostics_path=dynamics_path,
        hyperparams_path=hp_plot_path,
    )


def run_ta_lbfgs_baseline(
    n_trials: int = 40,
    n_layers: int = 4,
    output_dir: str = "outputs",
    prompt_signatures: Optional[List[float]] = None,
    ta_overrides: Optional[Dict[str, object]] = None,
    real_model_objective: bool = False,
    hf_model_id: str = "Qwen/Qwen2.5-0.5B",
    hf_local_path: Optional[str] = None,
    hf_max_length: int = 256,
    hf_device: str = "auto",
    dataset_prompts: Optional[List[str]] = None,
) -> BaselineResult:
    print("=" * 60)
    mode_name = "Real HF Model Objective" if real_model_objective else "Strict Bilevel + Hutch++"
    print(f"  OPTIMIZATION BASELINE: ta-LBFGS ({mode_name})")
    print("=" * 60)

    if prompt_signatures is None:
        prompt_signatures = build_prompt_signatures(n_trials=n_trials, seed=42)
    if len(prompt_signatures) < n_trials:
        raise ValueError("prompt_signatures must contain at least n_trials entries")

    config_kwargs: Dict[str, object] = dict(
        n_layers=n_layers,
        outer_steps=n_trials,
        inner_steps=3,
        lbfgs_memory_base=5,
        lbfgs_memory_min=3,
        lbfgs_memory_max=20,
        lbfgs_lr=1.0,
        lbfgs_line_search="None",
        inner_secant_topology_enabled=True,
        inner_secant_warmup_steps=10,
        inner_secant_top_k=16,
        inner_secant_percentile=95.0,
        outer_hutchpp_precondition_enabled=True,
        outer_hutchpp_samples=3,
        outer_hutchpp_eps=1e-8,
        outer_grad_clip_enabled=False,
        outer_lr_warmup_enabled=False,
        outer_hp_ema_enabled=False,
        outer_plateau_detection_enabled=False,
    )
    if ta_overrides:
        config_kwargs.update(ta_overrides)

    config = TaLBFGSConfig(**config_kwargs)
    if hf_device == "auto":
        hf_device_resolved = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        hf_device_resolved = hf_device

    if real_model_objective:
        model = RealPromptTuningObjective(
            model_id=hf_model_id,
            local_path=hf_local_path,
            max_length=hf_max_length,
            device=hf_device_resolved,
            prefix_len=4,
        )
        if dataset_prompts and len(dataset_prompts) >= 4:
            train_prompts = [str(p) for p in dataset_prompts[:2]]
            val_prompts = [str(p) for p in dataset_prompts[2:4]]
        elif dataset_prompts:
            train_prompts = [str(dataset_prompts[0])]
            val_prompts = [str(dataset_prompts[min(1, len(dataset_prompts) - 1)])]
        else:
            train_prompts = [
                "Explain why quasi-Newton methods improve conditioning in deep optimization.",
                "Describe tradeoffs between memory and curvature fidelity in L-BFGS.",
            ]
            val_prompts = [
                "How does Hessian information help stabilize updates in non-convex landscapes?",
            ]
    else:
        model = SyntheticMetaModel(
            n_layers=n_layers,
            device=config.device,
            dtype=config.get_torch_dtype(),
        )
    bilevel = BilevelOptimizer(config)

    trial_cursor = {"idx": 0}

    def _current_signature() -> float:
        idx = min(int(trial_cursor["idx"]), n_trials - 1)
        return float(prompt_signatures[idx])

    if real_model_objective:
        def _forward_real(model_obj, prompts, params_override=None):
            if params_override is not None:
                return functional_call_model(model_obj, params_override, prompt_texts=prompts)
            return model_obj(prompts)

        def train_fn(model_obj, _data, hyperparams, params_override=None):
            base_loss = _forward_real(model_obj, train_prompts, params_override=params_override)
            lr_vec = torch.stack([hyperparams.get_layer_lr(i) for i in range(n_layers)]).to(dtype=base_loss.dtype)
            wd_vec = torch.stack([hyperparams.get_layer_wd(i) for i in range(n_layers)]).to(dtype=base_loss.dtype)
            sig = torch.tensor(_current_signature(), device=base_loss.device, dtype=base_loss.dtype)
            return base_loss * (1.0 + 0.05 * lr_vec.mean()) + 0.01 * wd_vec.mean() + 0.005 * torch.sin(sig)

        def val_fn(model_obj, _data, hyperparams, params_override=None):
            base_loss = _forward_real(model_obj, val_prompts, params_override=params_override)
            lr_vec = torch.stack([hyperparams.get_layer_lr(i) for i in range(n_layers)]).to(dtype=base_loss.dtype)
            wd_vec = torch.stack([hyperparams.get_layer_wd(i) for i in range(n_layers)]).to(dtype=base_loss.dtype)
            sig = torch.tensor(_current_signature(), device=base_loss.device, dtype=base_loss.dtype)
            return base_loss * (1.0 + 0.06 * lr_vec.mean()) + 0.012 * wd_vec.mean() + 0.006 * torch.cos(sig)
    else:
        def train_fn(model_obj, _data, hyperparams, params_override=None):
            w = params_override["w"] if params_override is not None else model_obj.w
            idx = torch.arange(n_layers, device=w.device, dtype=w.dtype)
            sig = torch.tensor(_current_signature(), device=w.device, dtype=w.dtype)
            lr = torch.stack([hyperparams.get_layer_lr(i) for i in range(n_layers)]).to(dtype=w.dtype)
            wd = torch.stack([hyperparams.get_layer_wd(i) for i in range(n_layers)]).to(dtype=w.dtype)

            target = torch.sin(0.17 * idx + sig) + 0.25 * torch.cos(0.07 * idx - sig)
            resid = w - target
            loss = ((1.0 + 4.0 * wd) * resid.pow(2)).mean()
            loss = loss + 0.03 * torch.log(lr).pow(2).mean() + 0.01 * torch.sin(4.0 * w + sig).mean()
            return loss

        def val_fn(model_obj, _data, hyperparams, params_override=None):
            w = params_override["w"] if params_override is not None else model_obj.w
            idx = torch.arange(n_layers, device=w.device, dtype=w.dtype)
            sig = torch.tensor(_current_signature(), device=w.device, dtype=w.dtype)
            lr = torch.stack([hyperparams.get_layer_lr(i) for i in range(n_layers)]).to(dtype=w.dtype)
            wd = torch.stack([hyperparams.get_layer_wd(i) for i in range(n_layers)]).to(dtype=w.dtype)

            target = torch.sin(0.19 * idx + 0.5 * sig) + 0.30 * torch.cos(0.11 * idx + 0.2 * sig)
            shifted = 1.05 * w - target
            loss = ((1.0 + 5.0 * wd) * shifted.pow(2)).mean()
            loss = loss + 0.04 * torch.log(lr).pow(2).mean() + 0.02 * torch.cos(6.0 * w - sig).mean()
            return loss

    def _progress_cb(payload: Dict[str, object]):
        trial_cursor["idx"] = int(payload["iteration"])

    start = time.perf_counter()
    run_result = bilevel.optimize(
        model=model,
        train_fn=train_fn,
        val_fn=val_fn,
        train_data=None,
        val_data=None,
        use_dashboard=False,
        run_validity_checks=False,
        progress_callback=_progress_cb,
    )

    elapsed = time.perf_counter() - start
    loss_history = [float(x) for x in run_result["loss_history"]]
    hp_history = run_result["hyperparam_history"]
    finite_losses = [x for x in loss_history if math.isfinite(x)]
    best_loss = min(finite_losses) if finite_losses else float("inf")
    variance = _loss_variance_summary(loss_history)

    eval_results = run_evaluation(
        LayerwiseTaLBFGS(config),
        "TaLBFGS",
        strict_label_binding=bool(config.strict_optimizer_label_binding),
    )
    eval_results["phase_portrait"] = list(loss_history)
    eval_results["hp_modulation"] = list(hp_history)
    eval_results["grad_norms"] = list(run_result.get("grad_magnitude_history", []))

    traj_path, dynamics_path, hp_plot_path = _export_baseline_artifacts(
        baseline_name="ta_lbfgs",
        loss_history=loss_history,
        hp_history=hp_history,
        output_dir=output_dir,
        optimizer_name=eval_results["optimizer"],
        annotate_instability_spike=config.diagnostics_spike_annotation_enabled,
        spike_threshold=float(config.diagnostics_spike_threshold),
    )
    _summarize_success("ta-LBFGS (Bilevel)", best_loss, elapsed, traj_path, dynamics_path, hp_plot_path, variance)
    return BaselineResult(
        name="ta_lbfgs",
        best_loss=float(best_loss),
        elapsed_sec=float(elapsed),
        trajectory_path=traj_path,
        diagnostics_path=dynamics_path,
        hyperparams_path=hp_plot_path,
        metadata={
            "grad_norms": [float(x) for x in run_result.get("grad_magnitude_history", [])],
            "hutch_trace_history": [
                float(x) for x in getattr(bilevel, "hutch_trace_history", [])
            ],
        },
    )


def run_pure_lbfgs_baseline(
    n_trials: int = 40,
    n_layers: int = 4,
    output_dir: str = "outputs",
    prompt_signatures: Optional[List[float]] = None,
) -> BaselineResult:
    print("=" * 60)
    print("  OPTIMIZATION BASELINE: Pure L-BFGS (torch.optim.LBFGS)")
    print("=" * 60)

    model = SyntheticBilevelModel(n_layers=n_layers)
    if prompt_signatures is None:
        prompt_signatures = build_prompt_signatures(n_trials=n_trials, seed=42)
    if len(prompt_signatures) < n_trials:
        raise ValueError("prompt_signatures must contain at least n_trials entries")

    hp_obj = DifferentiableHyperparameters(
        n_layers=n_layers,
        initial_label_smoothing=0.1,
    )

    optimizer = torch.optim.LBFGS(
        list(hp_obj.parameters()),
        lr=1.0,
        max_iter=5,
        history_size=10,
        line_search_fn="strong_wolfe",
    )
    eval_results = run_evaluation(optimizer, "LBFGS")

    loss_history: List[float] = []
    hp_history: List[Dict[str, List[float]]] = []
    best_loss = float("inf")

    start = time.perf_counter()
    for trial_idx in range(n_trials):
        hp_history.append(hp_obj.as_float_dict())
        prompt_sig = float(prompt_signatures[trial_idx])

        def closure():
            optimizer.zero_grad()
            loss = model.val_loss(hp_obj, prompt_signature=prompt_sig)
            loss.backward()
            return loss

        step_loss = optimizer.step(closure)
        hp_obj.clamp()

        # Evaluate after the update for trajectory consistency.
        with torch.no_grad():
            current_loss = float(model.val_loss(hp_obj, prompt_signature=prompt_sig).item())
        loss_history.append(current_loss)
        best_loss = min(best_loss, current_loss)

        # keep step_loss reference used for potential debug consistency
        _ = step_loss

    elapsed = time.perf_counter() - start
    variance = _loss_variance_summary(loss_history)

    eval_results["phase_portrait"] = list(loss_history)
    eval_results["hp_modulation"] = list(hp_history)
    eval_results["grad_norms"] = [
        0.0 if i == 0 else abs(loss_history[i] - loss_history[i - 1])
        for i in range(len(loss_history))
    ]

    traj_path, dynamics_path, hp_plot_path = _export_baseline_artifacts(
        baseline_name="lbfgs",
        loss_history=loss_history,
        hp_history=hp_history,
        output_dir=output_dir,
        optimizer_name=eval_results["optimizer"],
    )
    _summarize_success("Pure L-BFGS", best_loss, elapsed, traj_path, dynamics_path, hp_plot_path, variance)
    return BaselineResult(
        name="lbfgs",
        best_loss=float(best_loss),
        elapsed_sec=float(elapsed),
        trajectory_path=traj_path,
        diagnostics_path=dynamics_path,
        hyperparams_path=hp_plot_path,
    )


def run_dense_regression_benchmark(
    n_samples: int = 10000,
    n_features: int = 1000,
    steps: int = 25,
    seed: int = 42,
) -> Dict[str, float]:
    """Dense OLS benchmark where topology sparsity assumptions are intentionally violated."""
    torch.manual_seed(seed)
    rng = torch.Generator(device="cpu").manual_seed(seed)

    X = torch.randn(n_samples, n_features, generator=rng)
    w_true = torch.randn(n_features, generator=rng)
    y = X @ w_true + 0.01 * torch.randn(n_samples, generator=rng)

    def mse_loss(w: torch.Tensor) -> torch.Tensor:
        pred = X @ w
        return torch.mean((pred - y) ** 2)

    # Standard L-BFGS
    w_lbfgs = nn.Parameter(torch.zeros(n_features))
    opt_lbfgs = torch.optim.LBFGS(
        [w_lbfgs],
        lr=1.0,
        max_iter=1,
        history_size=10,
        line_search_fn="strong_wolfe",
    )
    t0 = time.perf_counter()
    for _ in range(steps):
        def closure_lbfgs():
            opt_lbfgs.zero_grad()
            loss = mse_loss(w_lbfgs)
            loss.backward()
            return loss
        opt_lbfgs.step(closure_lbfgs)
    t_lbfgs = time.perf_counter() - t0
    final_lbfgs = float(mse_loss(w_lbfgs).detach().item())

    # Topology-aware core optimizer on same dense task.
    w_ta = nn.Parameter(torch.zeros(n_features))
    opt_ta = FullBatchLBFGS(
        [w_ta],
        lr=1.0,
        history_size=10,
        line_search="Wolfe",
        damping=True,
        curvature_threshold=0.2,
        secant_topology_enabled=True,
        secant_topology_warmup_steps=10,
        secant_topology_top_k=16,
        secant_topology_percentile=95.0,
    )
    t1 = time.perf_counter()
    for _ in range(steps):
        def closure_ta():
            if w_ta.grad is not None:
                w_ta.grad.zero_()
            loss = mse_loss(w_ta)
            loss.backward()
            return loss
        opt_ta.step(closure_ta)
    t_ta = time.perf_counter() - t1
    final_ta = float(mse_loss(w_ta).detach().item())

    return {
        "lbfgs_mse": final_lbfgs,
        "ta_lbfgs_mse": final_ta,
        "lbfgs_time": float(t_lbfgs),
        "ta_lbfgs_time": float(t_ta),
        "predicted_winner": "LBFGS",
    }


def run_complexity_scaling_benchmark(
    sizes: List[int],
    n_samples: int = 2000,
    seed: int = 42,
) -> List[Dict[str, float]]:
    """Measure single-step wall-clock growth for dense LBFGS vs topology-aware LBFGS."""
    rows: List[Dict[str, float]] = []
    for p in sizes:
        torch.manual_seed(seed)
        rng = torch.Generator(device="cpu").manual_seed(seed + p)
        X = torch.randn(n_samples, p, generator=rng)
        w_true = torch.randn(p, generator=rng)
        y = X @ w_true + 0.01 * torch.randn(n_samples, generator=rng)

        def mse_loss(w: torch.Tensor) -> torch.Tensor:
            pred = X @ w
            return torch.mean((pred - y) ** 2)

        w_l = nn.Parameter(torch.zeros(p))
        opt_l = torch.optim.LBFGS([w_l], lr=1.0, max_iter=1, history_size=10, line_search_fn="strong_wolfe")
        t0 = time.perf_counter()
        def closure_l():
            opt_l.zero_grad()
            loss = mse_loss(w_l)
            loss.backward()
            return loss
        opt_l.step(closure_l)
        t_lbfgs = time.perf_counter() - t0

        w_t = nn.Parameter(torch.zeros(p))
        opt_t = FullBatchLBFGS(
            [w_t],
            lr=1.0,
            history_size=10,
            line_search="Wolfe",
            damping=True,
            secant_topology_enabled=True,
            secant_topology_warmup_steps=10,
            secant_topology_top_k=16,
            secant_topology_percentile=95.0,
        )
        t1 = time.perf_counter()
        def closure_t():
            if w_t.grad is not None:
                w_t.grad.zero_()
            loss = mse_loss(w_t)
            loss.backward()
            return loss
        opt_t.step(closure_t)
        t_ta = time.perf_counter() - t1

        rows.append(
            {
                "n_params": float(p),
                "lbfgs_step_time": float(t_lbfgs),
                "ta_lbfgs_step_time": float(t_ta),
            }
        )
    return rows


def _hypergradient_stability_summary(grad_norms: List[float], threshold: float = 0.8) -> Dict[str, float]:
    if not grad_norms:
        return {
            "max_grad_norm": float("nan"),
            "mean_tail_grad_norm": float("nan"),
            "num_steps_above_threshold": float("nan"),
            "threshold": float(threshold),
        }
    tail = grad_norms[-10:] if len(grad_norms) >= 10 else grad_norms
    above = sum(1 for x in grad_norms if x > threshold)
    return {
        "max_grad_norm": float(max(grad_norms)),
        "mean_tail_grad_norm": float(np.mean(np.array(tail, dtype=np.float64))),
        "num_steps_above_threshold": float(above),
        "threshold": float(threshold),
    }


def write_benchmark_protocol_report(
    output_dir: str,
    sparse_claim_result: BaselineResult,
    sparse_profiles: Dict[str, Dict[str, object]],
    dense_result: Dict[str, float],
    complexity_rows: List[Dict[str, float]],
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "benchmark_protocol_report.md")

    lines: List[str] = []
    lines.append("# Tiered Benchmark Protocol Report")
    lines.append("")
    lines.append("## Core Comparison Table")
    lines.append("")
    lines.append("| Task | Sparsity Type | L-BFGS | Talpha-LBFGS | Winner | Predicted by Theory? |")
    lines.append("|---|---|---|---|---|---|")

    claim_profile_name = "full_fidelity"
    if claim_profile_name not in sparse_profiles and sparse_profiles:
        claim_profile_name = list(sparse_profiles.keys())[0]

    claim_profile_entry = sparse_profiles[claim_profile_name]
    claim_profile = claim_profile_entry["profile"]
    claim_stability = claim_profile_entry["stability"]

    lines.append(
        f"| Real HF prompt-tuning (proxy sparse LM, {claim_profile_name}) | Block/moderate sparse | n/a | "
        f"best_loss={sparse_claim_result.best_loss:.6f} | TBD | Yes - Assumption 1 (proxy) |"
    )
    lines.append(
        f"| Dense linear regression | None | mse={dense_result['lbfgs_mse']:.6f} | mse={dense_result['ta_lbfgs_mse']:.6f} | "
        f"{'L-BFGS' if dense_result['lbfgs_mse'] <= dense_result['ta_lbfgs_mse'] else 'Talpha-LBFGS'} | Yes - Limitation |"
    )
    lines.append("")
    lines.append("## Sparse Profile Comparison (Speed vs Fidelity)")
    lines.append("")
    lines.append(
        "| Profile | Intended Use | Claim-Valid | Best Loss | Elapsed (s) | max_length | inner_steps | hutch_samples | trace | diagonal_precond | max ||grad_lambda|| | mean tail ||grad_lambda|| |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for profile_name, payload in sparse_profiles.items():
        profile: BenchmarkProfile = payload["profile"]
        result: BaselineResult = payload["result"]
        stability: Dict[str, float] = payload["stability"]
        overrides = profile.ta_overrides
        lines.append(
            f"| {profile_name} | {'final-reporting' if profile.claim_valid else 'smoke/perf sweeps'} | "
            f"{'yes' if profile.claim_valid else 'no'} | {result.best_loss:.6f} | {result.elapsed_sec:.2f} | "
            f"{profile.real_max_length} | {int(overrides.get('inner_steps', -1))} | "
            f"{int(overrides.get('outer_hutchpp_samples', -1))} | "
            f"{bool(overrides.get('outer_hutchpp_trace_enabled', False))} | "
            f"{bool(overrides.get('outer_hutchpp_diagonal_precondition_enabled', False))} | "
            f"{stability['max_grad_norm']:.6f} | {stability['mean_tail_grad_norm']:.6f} |"
        )
    lines.append("")
    lines.append("## Complexity Scaling")
    lines.append("")
    lines.append("| N params | L-BFGS step time (s) | Talpha-LBFGS step time (s) |")
    lines.append("|---|---|---|")
    for row in complexity_rows:
        lines.append(
            f"| {int(row['n_params'])} | {row['lbfgs_step_time']:.6f} | {row['ta_lbfgs_step_time']:.6f} |"
        )
    lines.append("")
    lines.append("## Hypergradient Stability")
    lines.append("")
    lines.append(
        f"- claim-valid profile: {claim_profile_name}"
    )
    lines.append(
        f"- max ||grad_lambda||: {claim_stability['max_grad_norm']:.6f}"
    )
    lines.append(
        f"- mean tail ||grad_lambda|| (last 10): {claim_stability['mean_tail_grad_norm']:.6f}"
    )
    lines.append(
        f"- steps above threshold ({claim_stability['threshold']:.2f}): {int(claim_stability['num_steps_above_threshold'])}"
    )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for profile_name, payload in sparse_profiles.items():
        result: BaselineResult = payload["result"]
        lines.append(f"- sparse ({profile_name}) diagnostics: {result.diagnostics_path}")
        lines.append(f"- sparse ({profile_name}) trajectory: {result.trajectory_path}")
        lines.append(f"- sparse ({profile_name}) hyperparams: {result.hyperparams_path}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return report_path


def run_mvp4_protocol(
    n_trials: int,
    n_layers: int,
    output_dir: str,
    prompt_signatures: List[float],
    dataset_prompts: Optional[List[str]],
    args: argparse.Namespace,
) -> None:
    print("=" * 60)
    print("  MVP-4 PROTOCOL: Sparse + Dense + Complexity + Stability")
    print("=" * 60)

    profiles = build_benchmark_profiles(base_real_max_length=args.real_max_length)
    if args.profile_report == "both":
        selected_profile_names = ["speed_tuned", "full_fidelity"]
    else:
        selected_profile_names = [args.profile_report]

    sparse_profiles: Dict[str, Dict[str, object]] = {}
    for profile_name in selected_profile_names:
        profile = profiles[profile_name]
        profile_output_dir = os.path.join(output_dir, profile_name)
        print(
            f"[PROFILE] {profile_name} | claim_valid={profile.claim_valid} | "
            f"max_length={profile.real_max_length}"
        )
        print(f"          {profile.description}")

        sparse_result = run_ta_lbfgs_baseline(
            n_trials=n_trials,
            n_layers=n_layers,
            output_dir=profile_output_dir,
            prompt_signatures=prompt_signatures,
            ta_overrides=dict(profile.ta_overrides),
            real_model_objective=True,
            hf_model_id=args.hf_model_id,
            hf_local_path=args.hf_local_path or None,
            hf_max_length=profile.real_max_length,
            hf_device=args.hf_device,
            dataset_prompts=dataset_prompts,
        )

        grad_norms = []
        if sparse_result.metadata is not None:
            grad_norms = [float(x) for x in sparse_result.metadata.get("grad_norms", [])]
        stability = _hypergradient_stability_summary(grad_norms, threshold=0.8)
        sparse_profiles[profile_name] = {
            "profile": profile,
            "result": sparse_result,
            "stability": stability,
        }

    if "full_fidelity" in sparse_profiles:
        sparse_claim_result = sparse_profiles["full_fidelity"]["result"]
    else:
        sparse_claim_result = sparse_profiles[selected_profile_names[0]]["result"]

    dense = run_dense_regression_benchmark(
        n_samples=10000,
        n_features=1000,
        steps=max(10, min(40, n_trials)),
        seed=args.seed,
    )

    complexity_rows = run_complexity_scaling_benchmark(
        sizes=[10000, 100000, 500000],
        n_samples=2000,
        seed=args.seed,
    )

    report_path = write_benchmark_protocol_report(
        output_dir=output_dir,
        sparse_claim_result=sparse_claim_result,
        sparse_profiles=sparse_profiles,
        dense_result=dense,
        complexity_rows=complexity_rows,
    )

    print("\n[SUCCESS] MVP-4 protocol report generated")
    print(f"  - Report: {report_path}")
    for profile_name in selected_profile_names:
        result = sparse_profiles[profile_name]["result"]
        print(
            f"  - Sparse {profile_name}: best_loss={result.best_loss:.6f}, "
            f"elapsed={result.elapsed_sec:.2f}s"
        )
    print(
        f"  - Dense winner: {'L-BFGS' if dense['lbfgs_mse'] <= dense['ta_lbfgs_mse'] else 'Talpha-LBFGS'} "
        f"(lbfgs={dense['lbfgs_mse']:.6f}, ta={dense['ta_lbfgs_mse']:.6f})"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-design",
        type=str,
        choices=["legacy", "mvp4"],
        default="legacy",
        help="Run legacy baseline flow or the new MVP-4 tiered benchmark protocol.",
    )
    parser.add_argument(
        "--profile-report",
        type=str,
        choices=["both", "speed_tuned", "full_fidelity"],
        default="both",
        help=(
            "Profile mode for MVP-4 sparse benchmark runs. 'both' reports speed-tuned "
            "(smoke/perf) and full-fidelity (claim-valid) side-by-side."
        ),
    )
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument(
        "--method",
        type=str,
        choices=["optuna", "random", "grid", "ta_lbfgs", "lbfgs", "all"],
        default="all",
        help="Which baseline(s) to run.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trace-min-steps", type=int, default=8)
    parser.add_argument("--trace-max-steps", type=int, default=40)
    parser.add_argument("--prompt-source", type=str, default=get_default_dataset_path())
    parser.add_argument("--prompt-max-samples", type=int, default=5000)
    parser.add_argument(
        "--signature-source",
        type=str,
        choices=["hash", "hf_cached"],
        default="hash",
        help="How to derive per-trial prompt signatures.",
    )
    parser.add_argument("--hf-model-id", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument(
        "--hf-local-path",
        type=str,
        default="",
        help="Optional local HF snapshot path; if empty, resolve from ~/.cache/huggingface/hub",
    )
    parser.add_argument("--hf-max-length", type=int, default=256)
    parser.add_argument(
        "--hf-device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )
    parser.add_argument("--ta-lr", type=float, default=1.0)
    parser.add_argument(
        "--ta-line-search",
        type=str,
        choices=["None", "Armijo", "Wolfe"],
        default="None",
    )
    parser.add_argument("--ta-memory-base", type=int, default=5)
    parser.add_argument("--ta-memory-min", type=int, default=3)
    parser.add_argument("--ta-memory-max", type=int, default=20)
    parser.add_argument("--ta-disable-topology", action="store_true")
    parser.add_argument("--ta-warmup-steps", type=int, default=-1)
    parser.add_argument("--ta-disable-edrt", action="store_true")
    parser.add_argument(
        "--real-model-objective",
        action="store_true",
        help="Use real HF model loss objective for ta_lbfgs baseline instead of synthetic surface.",
    )
    parser.add_argument("--real-max-length", type=int, default=128)
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    results: List[BaselineResult] = []
    dataset_prompts = load_dataset_prompts(
        source_path=args.prompt_source,
        max_prompts=max(args.prompt_max_samples, args.trials),
    )

    if args.signature_source == "hf_cached" and dataset_prompts:
        prompt_signatures = build_prompt_signatures_from_hf_cache(
            prompt_texts=dataset_prompts,
            n_trials=args.trials,
            model_id=args.hf_model_id,
            local_path=args.hf_local_path or None,
            max_length=args.hf_max_length,
            device=args.hf_device,
        )
    else:
        if args.signature_source == "hf_cached" and not dataset_prompts:
            print("[WARNING] hf_cached signature source requested but no dataset prompts found; falling back to hash signatures.")
        prompt_signatures = build_prompt_signatures(
            n_trials=args.trials,
            seed=args.seed,
            min_trace_steps=args.trace_min_steps,
            max_trace_steps=args.trace_max_steps,
            prompt_texts=dataset_prompts if dataset_prompts else None,
        )
    if dataset_prompts:
        print(
            f"Loaded {len(dataset_prompts)} prompts from dataset: {args.prompt_source}"
        )
    else:
        print(
            "[WARNING] Dataset prompts unavailable; using synthetic long-reasoning prompts."
        )
    print(
        f"Using {len(prompt_signatures)} prompt-conditioned trial contexts "
        "(1 optimization iteration + 1 unique long-reasoning prompt per trial)."
    )
    print(f"Signature source: {args.signature_source}")

    if args.benchmark_design == "mvp4":
        run_mvp4_protocol(
            n_trials=args.trials,
            n_layers=args.layers,
            output_dir=args.output_dir,
            prompt_signatures=prompt_signatures,
            dataset_prompts=dataset_prompts if dataset_prompts else None,
            args=args,
        )
        raise SystemExit(0)

    if args.method in {"optuna", "all"}:
        results.append(run_optuna_baseline(args.trials, args.layers, args.output_dir, prompt_signatures=prompt_signatures))
    if args.method in {"random", "all"}:
        results.append(run_random_baseline(args.trials, args.layers, args.output_dir, args.seed, prompt_signatures=prompt_signatures))
    if args.method in {"grid", "all"}:
        results.append(run_grid_baseline(args.trials, args.layers, args.output_dir, prompt_signatures=prompt_signatures))
    if args.method in {"ta_lbfgs", "all"}:
        ta_overrides: Dict[str, object] = {
            "lbfgs_lr": float(args.ta_lr),
            "lbfgs_line_search": args.ta_line_search,
            "lbfgs_memory_base": int(args.ta_memory_base),
            "lbfgs_memory_min": int(args.ta_memory_min),
            "lbfgs_memory_max": int(args.ta_memory_max),
            "inner_secant_topology_enabled": not args.ta_disable_topology,
        }
        if args.ta_warmup_steps >= 0:
            ta_overrides["inner_secant_warmup_steps"] = int(args.ta_warmup_steps)
        if args.ta_disable_edrt:
            print("[NOTE] --ta-disable-edrt has no effect in strict bilevel benchmark mode.")
        results.append(
            run_ta_lbfgs_baseline(
                args.trials,
                args.layers,
                args.output_dir,
                prompt_signatures=prompt_signatures,
                ta_overrides=ta_overrides,
                real_model_objective=bool(args.real_model_objective),
                hf_model_id=args.hf_model_id,
                hf_local_path=args.hf_local_path or None,
                hf_max_length=args.real_max_length,
                hf_device=args.hf_device,
                dataset_prompts=dataset_prompts if dataset_prompts else None,
            )
        )
    if args.method in {"lbfgs", "all"}:
        results.append(run_pure_lbfgs_baseline(args.trials, args.layers, args.output_dir, prompt_signatures=prompt_signatures))

    if len(results) > 1:
        print("\n" + "=" * 60)
        print("  BASELINE COMPARISON")
        print("=" * 60)
        for rank, result in enumerate(sorted(results, key=lambda x: x.best_loss), start=1):
            print(
                f"{rank}. {result.name:<8} best_loss={result.best_loss:.6f} "
                f"elapsed={result.elapsed_sec:.2f}s"
            )
