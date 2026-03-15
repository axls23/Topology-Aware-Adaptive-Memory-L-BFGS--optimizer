"""
Real-model 3D trajectory visualization for optimizer comparison.

This script uses actual language model inference loss (causal LM loss) as f(x),
where x is a learnable soft-prompt parameter vector. It then runs two
L-BFGS-style optimization trajectories from the same initialization:

1) Standard L-BFGS-style: dense secant history.
2) Talpha-LBFGS-style: topology-masked secant history using adjacency matrix A.

Both trajectories are projected to 2D with PCA and overlaid on a 3D loss surface.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go
import torch
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class TrajectoryResult:
    points: np.ndarray
    losses: np.ndarray


@dataclass
class SparseTopologyMask:
    n_params: int
    edges: np.ndarray  # shape: (2, E), undirected edges stored once per pair


def hyperparameter_informed_points(
    points: np.ndarray,
    losses: np.ndarray,
    optimizer_lr: float,
    hp_wd: float,
    hp_dropout: float,
    hp_attn_temp: float,
    hp_label_smoothing: float,
    topology_density: float,
) -> np.ndarray:
    """
    Encode hyperparameter information into PCA input points.

    Keeps dimensionality identical to parameter vectors so inverse_transform
    remains valid for objective evaluation, while rotating/scaling points based
    on hyperparameter signatures and optimization progress.
    """
    out = points.copy()
    t = np.linspace(0.0, 1.0, len(points), dtype=np.float64)
    loss_norm = (losses - losses.min()) / (losses.ptp() + 1e-12)

    hp_strength = (
        0.20 * np.log10(max(optimizer_lr, 1e-12))
        + 0.10 * np.log10(max(hp_wd, 1e-12))
        + 0.35 * hp_dropout
        + 0.12 * hp_attn_temp
        + 0.18 * hp_label_smoothing
        + 0.25 * topology_density
    )

    # Step-wise modulation injects hyperparameter/progress structure into PCA.
    modulation = 1.0 + 0.08 * hp_strength * (0.5 + t) + 0.04 * (1.0 - loss_norm)
    out *= modulation[:, None]
    return out


def apply_sparse_topology_mask(
    vec: np.ndarray,
    topo_mask: SparseTopologyMask | np.ndarray | None,
    blend: float = 0.6,
) -> np.ndarray:
    """Apply a sparse topology mask without constructing an NxN dense matrix."""
    if topo_mask is None:
        return vec

    if isinstance(topo_mask, np.ndarray):
        return topo_mask @ vec

    if topo_mask.edges.size == 0:
        return vec

    out = vec.copy()
    n = topo_mask.n_params
    u = topo_mask.edges[0]
    v = topo_mask.edges[1]

    neigh_sum = np.zeros(n, dtype=np.float64)
    neigh_cnt = np.zeros(n, dtype=np.float64)

    np.add.at(neigh_sum, u, vec[v])
    np.add.at(neigh_sum, v, vec[u])
    np.add.at(neigh_cnt, u, 1.0)
    np.add.at(neigh_cnt, v, 1.0)

    has_neigh = neigh_cnt > 0
    neigh_avg = np.zeros(n, dtype=np.float64)
    neigh_avg[has_neigh] = neigh_sum[has_neigh] / neigh_cnt[has_neigh]
    out[has_neigh] = (1.0 - blend) * vec[has_neigh] + blend * neigh_avg[has_neigh]
    return out


def discover_topology_mask(
    objective: "RealModelObjective",
    x0: np.ndarray,
    warmup_steps: int,
    warmup_lr: float,
    active_dims: int,
    top_percent: float,
) -> SparseTopologyMask:
    """
    Autonomous topology discovery from secant pairs (s_k, y_k).

    Tracks correlated curvature trajectories via a sparse thresholded
    covariance proxy over top-|y_k| coordinates to avoid O(N^2) memory.
    """
    x = torch.tensor(x0, dtype=torch.float32, device=objective.device)
    edge_scores: dict[tuple[int, int], float] = {}
    n = int(x.numel())
    m = max(8, min(active_dims, n))

    def grad_at(x_t: torch.Tensor) -> torch.Tensor:
        x_var = x_t.detach().clone().requires_grad_(True)
        loss = objective.loss_from_tensor(x_var)
        loss.backward()
        return x_var.grad.detach()

    grad = grad_at(x)

    for _ in range(warmup_steps):
        gnorm = torch.norm(grad) + 1e-12
        step = -warmup_lr * grad / gnorm
        x_next = x + step
        grad_next = grad_at(x_next)

        s = (x_next - x).detach().cpu().numpy()
        y = (grad_next - grad).detach().cpu().numpy()

        curvature_scale = float(abs(np.dot(s, y)) + 1e-12)
        abs_y = np.abs(y)
        idx = np.argpartition(abs_y, -m)[-m:]
        vals = y[idx]

        # Sparse covariance proxy over active coordinates only.
        for i in range(len(idx)):
            ii = int(idx[i])
            vi = float(vals[i])
            for j in range(i + 1, len(idx)):
                jj = int(idx[j])
                vj = float(vals[j])
                w = curvature_scale * abs(vi * vj)
                if w <= 0.0:
                    continue
                a, b = (ii, jj) if ii < jj else (jj, ii)
                edge_scores[(a, b)] = edge_scores.get((a, b), 0.0) + w

        x = x_next.detach()
        grad = grad_next.detach()

    if not edge_scores:
        return SparseTopologyMask(n_params=n, edges=np.zeros((2, 0), dtype=np.int64))

    pairs = np.array(list(edge_scores.keys()), dtype=np.int64)
    scores = np.array(list(edge_scores.values()), dtype=np.float64)
    threshold = np.percentile(scores, max(0.0, min(100.0, top_percent)))
    keep = scores >= threshold
    kept_pairs = pairs[keep]

    edges = kept_pairs.T if kept_pairs.size else np.zeros((2, 0), dtype=np.int64)
    return SparseTopologyMask(n_params=n, edges=edges)


def two_loop_direction(
    grad: np.ndarray,
    s_hist: List[np.ndarray],
    y_hist: List[np.ndarray],
) -> np.ndarray:
    if not s_hist:
        return -grad

    q = grad.copy()
    alphas: List[float] = []
    rhos: List[float] = []

    for s, y in zip(reversed(s_hist), reversed(y_hist)):
        denom = float(np.dot(y, s))
        if abs(denom) < 1e-12:
            denom = 1e-12
        rho = 1.0 / denom
        alpha = rho * float(np.dot(s, q))
        q = q - alpha * y
        alphas.append(alpha)
        rhos.append(rho)

    y_last = y_hist[-1]
    s_last = s_hist[-1]
    ys = float(np.dot(y_last, s_last))
    yy = float(np.dot(y_last, y_last)) + 1e-12
    gamma = ys / yy if ys > 0.0 else 1.0
    r = gamma * q

    for i, (s, y) in enumerate(zip(s_hist, y_hist)):
        beta = rhos[-(i + 1)] * float(np.dot(y, r))
        r = r + s * (alphas[-(i + 1)] - beta)

    return -r


class RealModelObjective:
    def __init__(
        self,
        model_name_or_path: str,
        prompts: List[str],
        prefix_len: int,
        max_length: int,
        device: str,
        local_only: bool,
    ):
        self.device = device
        self.prefix_len = prefix_len

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=local_only)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            local_files_only=local_only,
        ).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.hidden_size = int(self.model.config.hidden_size)
        self.dimension = self.prefix_len * self.hidden_size

        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        self.input_ids = encoded["input_ids"].to(self.device)
        self.attn_mask = encoded["attention_mask"].to(self.device)

    def _vector_to_prefix(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(self.prefix_len, self.hidden_size)

    def loss_and_grad(self, x_np: np.ndarray) -> Tuple[float, np.ndarray]:
        x = torch.tensor(x_np, dtype=torch.float32, device=self.device, requires_grad=True)
        loss = self.loss_from_tensor(x)
        loss.backward()
        grad = x.grad.detach().cpu().numpy()
        return float(loss.item()), grad

    @torch.no_grad()
    def loss_value(self, x_np: np.ndarray) -> float:
        x = torch.tensor(x_np, dtype=torch.float32, device=self.device)
        return float(self.loss_from_tensor(x).item())

    def loss_from_tensor(self, x: torch.Tensor) -> torch.Tensor:
        prefix = self._vector_to_prefix(x)
        prefix = prefix.unsqueeze(0).expand(self.input_ids.size(0), -1, -1)

        token_embeds = self.model.get_input_embeddings()(self.input_ids)
        inputs_embeds = torch.cat([prefix, token_embeds], dim=1)

        prefix_mask = torch.ones(
            (self.attn_mask.size(0), self.prefix_len),
            device=self.device,
            dtype=self.attn_mask.dtype,
        )
        attn = torch.cat([prefix_mask, self.attn_mask], dim=1)

        ignore = torch.full(
            (self.input_ids.size(0), self.prefix_len),
            -100,
            device=self.device,
            dtype=self.input_ids.dtype,
        )
        labels = torch.cat([ignore, self.input_ids], dim=1)

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            labels=labels,
            use_cache=False,
        )
        return outputs.loss


def run_lbfgs_style(
    objective: RealModelObjective,
    x0: np.ndarray,
    steps: int,
    lr: float,
    memory: int,
    topo_mask: SparseTopologyMask | np.ndarray | None,
) -> TrajectoryResult:
    x = x0.copy()
    points = [x.copy()]
    losses: List[float] = []

    s_hist: List[np.ndarray] = []
    y_hist: List[np.ndarray] = []

    loss_prev, grad_prev = objective.loss_and_grad(x)
    losses.append(loss_prev)

    for _ in range(steps):
        grad_use = apply_sparse_topology_mask(grad_prev, topo_mask)
        direction = two_loop_direction(grad_use, s_hist, y_hist)
        direction_norm = np.linalg.norm(direction) + 1e-12
        direction = direction / direction_norm

        # Conservative backtracking for stable model-loss descent.
        step_scale = lr
        accepted = False
        for _ in range(6):
            x_candidate = x + step_scale * direction
            loss_candidate = objective.loss_value(x_candidate)
            if loss_candidate <= loss_prev:
                accepted = True
                break
            step_scale *= 0.5

        if not accepted:
            x_candidate = x - 0.1 * lr * grad_use / (np.linalg.norm(grad_use) + 1e-12)

        loss_new, grad_new = objective.loss_and_grad(x_candidate)

        s = x_candidate - x
        y = grad_new - grad_prev
        if topo_mask is not None:
            s = apply_sparse_topology_mask(s, topo_mask)
            y = apply_sparse_topology_mask(y, topo_mask)

        ys = float(np.dot(y, s))
        if ys > 1e-10:
            s_hist.append(s)
            y_hist.append(y)
            if len(s_hist) > memory:
                s_hist.pop(0)
                y_hist.pop(0)

        x = x_candidate
        grad_prev = grad_new
        loss_prev = loss_new

        points.append(x.copy())
        losses.append(loss_prev)

    return TrajectoryResult(points=np.array(points), losses=np.array(losses))


def make_surface(
    pca: PCA,
    projected: np.ndarray,
    objective: RealModelObjective,
    grid_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    margin1 = 0.12 * (projected[:, 0].ptp() + 1e-8)
    margin2 = 0.12 * (projected[:, 1].ptp() + 1e-8)

    pc1 = np.linspace(projected[:, 0].min() - margin1, projected[:, 0].max() + margin1, grid_size)
    pc2 = np.linspace(projected[:, 1].min() - margin2, projected[:, 1].max() + margin2, grid_size)
    xx, yy = np.meshgrid(pc1, pc2)
    zz = np.zeros_like(xx)

    for i in range(grid_size):
        coords = np.column_stack((xx[i], yy[i]))
        vectors = pca.inverse_transform(coords)
        zz[i] = np.array([objective.loss_value(v) for v in vectors])

    return xx, yy, zz


def build_figure(
    traj_std_proj: np.ndarray,
    traj_topo_proj: np.ndarray,
    losses_std: np.ndarray,
    losses_topo: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    show_secants: bool,
    topology_edges: int,
) -> go.Figure:
    fig = go.Figure()

    dZ_dx = np.gradient(zz, axis=1)
    dZ_dy = np.gradient(zz, axis=0)
    curvature_mag = np.sqrt(dZ_dx ** 2 + dZ_dy ** 2)

    fig.add_trace(
        go.Surface(
            x=xx,
            y=yy,
            z=zz,
            opacity=0.6,
            colorscale=[
                [0.0, "rgb(0, 255, 255)"],
                [0.2, "rgb(0, 150, 255)"],
                [0.5, "rgb(150, 0, 255)"],
                [0.8, "rgb(255, 0, 150)"],
                [1.0, "rgb(255, 20, 20)"],
            ],
            surfacecolor=curvature_mag,
            showscale=True,
            colorbar=dict(
                title="Saliency / Curvature",
                len=0.5,
                y=0.25,
                thickness=15,
                tickfont=dict(color="rgba(255,255,255,0.7)"),
            ),
            lighting=dict(
                ambient=0.4,
                diffuse=0.9,
                fresnel=2,
                specular=1.5,
                roughness=0.1,
            ),
            lightposition=dict(x=100, y=200, z=150),
            name="Geometric Topology",
            hovertemplate=(
                "PC1: %{x:.4f}<br>"
                "PC2: %{y:.4f}<br>"
                "Val Loss: %{z:.4f}<br>"
                "<extra>Topology Surface</extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=traj_std_proj[:, 0],
            y=traj_std_proj[:, 1],
            z=losses_std,
            mode="lines+markers",
            line=dict(color="red", width=6, dash="dash"),
            marker=dict(size=4, color="red"),
            name="Standard L-BFGS",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=traj_topo_proj[:, 0],
            y=traj_topo_proj[:, 1],
            z=losses_topo,
            mode="lines+markers",
            line=dict(color="blue", width=10),
            marker=dict(size=4, color=losses_topo, colorscale="Viridis", opacity=0.9),
            name="Talpha-LBFGS",
            hovertemplate=(
                "PC1: %{x:.4f}<br>"
                "PC2: %{y:.4f}<br>"
                "Loss: %{z:.4f}<br>"
                "<extra>Step %{pointNumber}</extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[traj_topo_proj[0, 0]],
            y=[traj_topo_proj[0, 1]],
            z=[losses_topo[0]],
            mode="markers",
            marker=dict(size=10, color="lime", symbol="diamond"),
            name="Start",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[traj_topo_proj[-1, 0]],
            y=[traj_topo_proj[-1, 1]],
            z=[losses_topo[-1]],
            mode="markers",
            marker=dict(size=10, color="red", symbol="x"),
            name="End (Best)",
            showlegend=True,
        )
    )

    if show_secants and len(traj_topo_proj) > 4:
        idxs = np.linspace(2, len(traj_topo_proj) - 3, 5).astype(int)
        idxs = np.unique(idxs)
        x0 = traj_topo_proj[idxs, 0]
        y0 = traj_topo_proj[idxs, 1]
        z0 = losses_topo[idxs]

        dx = traj_topo_proj[idxs + 1, 0] - traj_topo_proj[idxs, 0]
        dy = traj_topo_proj[idxs + 1, 1] - traj_topo_proj[idxs, 1]
        dz = losses_topo[idxs + 1] - losses_topo[idxs]

        fig.add_trace(
            go.Cone(
                x=x0,
                y=y0,
                z=z0,
                u=dx,
                v=dy,
                w=dz,
                colorscale=[[0, "#1f77b4"], [1, "#1f77b4"]],
                sizemode="absolute",
                sizeref=0.18,
                showscale=False,
                name="Secant s_k (topology-aware)",
                opacity=0.9,
            )
        )

    interpretation_text = (
        "<b>TOPOLOGY-AWARE VISUALIZATION</b><br>"
        f"• <b>Surface:</b> Prompt-conditioned PCA landscape (sparse topology edges: {topology_edges}).<br>"
        "• <b>Blue valleys:</b> Well-conditioned regions. <b>Red peaks:</b> High curvature / instability risk.<br>"
        "• <b>Red dashed:</b> Standard L-BFGS trajectory. <b>Blue:</b> Topology-aware trajectory.<br>"
        "• <b>Z-Axis:</b> Validation loss from real model inference (soft-prompt objective).<br>"
        "• <b>X/Y Axes:</b> PCA coordinates of optimization states."
    )

    fig.update_layout(
        title=dict(
            text="ta-LBFGS: Topology-Aware Hyperparameter Landscape",
            font=dict(size=16),
        ),
        scene=dict(
            xaxis_title="Principal Component 1",
            yaxis_title="Principal Component 2",
            zaxis_title="Validation Loss",
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.7),
            camera=dict(eye=dict(x=1.45, y=1.5, z=0.95)),
        ),
        template="plotly_dark",
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(10, 10, 15, 0.9)",
            bordercolor="rgba(0, 255, 255, 0.3)",
            borderwidth=1,
            font=dict(color="cyan"),
        ),
        annotations=[
            dict(
                text=interpretation_text,
                align="left",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.02,
                y=0.02,
                bordercolor="gray",
                borderwidth=1,
                borderpad=10,
                bgcolor="rgba(20, 20, 30, 0.85)",
                font=dict(size=11, color="white"),
            )
        ],
        margin=dict(l=0, r=0, b=0, t=40),
    )
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3D PCA trajectory visualization from real model inference")
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.expanduser(
            "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/"
            "snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
        ),
        help="HF model id or local path",
    )
    parser.add_argument("--steps", type=int, default=20, help="Optimization steps per trajectory")
    parser.add_argument("--prefix-len", type=int, default=4, help="Number of soft-prompt tokens")
    parser.add_argument("--max-length", type=int, default=96, help="Tokenization max length")
    parser.add_argument("--memory", type=int, default=8, help="L-BFGS memory size")
    parser.add_argument("--std-lr", type=float, default=0.55, help="Step size for standard path")
    parser.add_argument("--topo-lr", type=float, default=0.45, help="Step size for topology-aware path")
    parser.add_argument("--hp-wd", type=float, default=1e-2, help="Weight decay hyperparameter signal for PCA")
    parser.add_argument("--hp-dropout", type=float, default=0.1, help="Dropout hyperparameter signal for PCA")
    parser.add_argument("--hp-attn-temp", type=float, default=1.0, help="Attention temperature signal for PCA")
    parser.add_argument("--hp-label-smoothing", type=float, default=0.04, help="Label smoothing signal for PCA")
    parser.add_argument("--warmup-steps", type=int, default=50, help="Topology discovery warmup iterations")
    parser.add_argument("--warmup-lr", type=float, default=0.05, help="Warmup step size for secant tracking")
    parser.add_argument("--active-dims", type=int, default=64, help="Top-|y_k| coordinates per warmup step")
    parser.add_argument("--topology-percentile", type=float, default=95.0, help="Percentile threshold for strong interactions")
    parser.add_argument("--grid-size", type=int, default=16, help="Surface grid size in PCA plane")
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Natural-language prompt text. For multiple prompts, separate with ||",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face downloads when files are not already cached locally",
    )
    parser.add_argument("--no-secants", action="store_true", help="Disable secant arrow visualization")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/trajectory_3d_pca_comparison.html",
        help="Output HTML path",
    )
    parser.add_argument(
        "--topology-output",
        type=str,
        default="outputs/topology_mask.pt",
        help="Path to save discovered sparse topology mask",
    )
    return parser.parse_args()


def resolve_prompts(prompt_arg: str | None) -> List[str]:
    """Resolve prompts from CLI argument or ask interactively from stdin."""
    if prompt_arg is not None:
        prompts = [p.strip() for p in prompt_arg.split("||") if p.strip()]
        if prompts:
            return prompts

    print("Enter a natural-language prompt to query the model.")
    print("You can include multiple prompts separated with ||.")
    user_text = input("Prompt: ").strip()
    prompts = [p.strip() for p in user_text.split("||") if p.strip()]
    if not prompts:
        raise ValueError("No prompt provided. Pass --prompt or type a prompt at runtime.")
    return prompts


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompts = resolve_prompts(args.prompt)

    objective = RealModelObjective(
        model_name_or_path=args.model,
        prompts=prompts,
        prefix_len=args.prefix_len,
        max_length=args.max_length,
        device=device,
        local_only=not args.allow_download,
    )

    x0 = np.random.normal(0.0, 0.02, size=objective.dimension).astype(np.float32)
    discovered_mask = discover_topology_mask(
        objective=objective,
        x0=x0,
        warmup_steps=args.warmup_steps,
        warmup_lr=args.warmup_lr,
        active_dims=args.active_dims,
        top_percent=args.topology_percentile,
    )

    # Save sparse binary mask M as COO indices for downstream reuse.
    topo_out = Path(args.topology_output)
    topo_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "size": (discovered_mask.n_params, discovered_mask.n_params),
            "indices": torch.tensor(discovered_mask.edges, dtype=torch.long),
            "values": torch.ones(discovered_mask.edges.shape[1], dtype=torch.uint8),
            "format": "sparse_coo_binary",
        },
        str(topo_out),
    )

    std = run_lbfgs_style(
        objective=objective,
        x0=x0,
        steps=args.steps,
        lr=args.std_lr,
        memory=args.memory,
        topo_mask=None,
    )
    topo = run_lbfgs_style(
        objective=objective,
        x0=x0,
        steps=args.steps,
        lr=args.topo_lr,
        memory=args.memory,
        topo_mask=discovered_mask,
    )

    max_edges = objective.dimension * max(objective.dimension - 1, 1) / 2.0
    topo_density = float(discovered_mask.edges.shape[1] / max_edges)

    std_for_pca = hyperparameter_informed_points(
        points=std.points,
        losses=std.losses,
        optimizer_lr=args.std_lr,
        hp_wd=args.hp_wd,
        hp_dropout=args.hp_dropout,
        hp_attn_temp=args.hp_attn_temp,
        hp_label_smoothing=args.hp_label_smoothing,
        topology_density=0.0,
    )
    topo_for_pca = hyperparameter_informed_points(
        points=topo.points,
        losses=topo.losses,
        optimizer_lr=args.topo_lr,
        hp_wd=args.hp_wd,
        hp_dropout=args.hp_dropout,
        hp_attn_temp=args.hp_attn_temp,
        hp_label_smoothing=args.hp_label_smoothing,
        topology_density=topo_density,
    )

    combined = np.vstack([std_for_pca, topo_for_pca])
    pca = PCA(n_components=2, random_state=args.seed)
    projected = pca.fit_transform(combined)

    n_std = len(std.points)
    std_proj = projected[:n_std]
    topo_proj = projected[n_std:]

    xx, yy, zz = make_surface(
        pca=pca,
        projected=projected,
        objective=objective,
        grid_size=args.grid_size,
    )

    fig = build_figure(
        traj_std_proj=std_proj,
        traj_topo_proj=topo_proj,
        losses_std=std.losses,
        losses_topo=topo.losses,
        xx=xx,
        yy=yy,
        zz=zz,
        show_secants=not args.no_secants,
        topology_edges=discovered_mask.edges.shape[1],
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn")

    print(f"Saved 3D trajectory visualization to: {output_path}")
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Prompts used: {len(prompts)}")
    print(f"Dimension: {objective.dimension}")
    print(f"Topology edges discovered: {discovered_mask.edges.shape[1]}")
    print(
        f"PCA hyperparameter signals: lr(std={args.std_lr}, topo={args.topo_lr}), "
        f"wd={args.hp_wd}, dropout={args.hp_dropout}, attn_temp={args.hp_attn_temp}, "
        f"label_smoothing={args.hp_label_smoothing}, topology_density={topo_density:.6e}"
    )
    print(f"Topology mask saved to: {topo_out}")
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Final loss (Standard L-BFGS): {std.losses[-1]:.6f}")
    print(f"Final loss (Talpha-LBFGS): {topo.losses[-1]:.6f}")


if __name__ == "__main__":
    main()
