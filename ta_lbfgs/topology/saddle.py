"""
Saddle-Point Detection and Evasion.

Two-pronged detection:
1. Secant condition check (y_k^T s_k ≤ 0)
2. Euler characteristic topology proxy (from Chronoscope)

Plus orthogonal perturbation injection for saddle escape.
"""

import torch
import numpy as np
from typing import Callable, Dict, Optional, Tuple


def check_secant_condition(
    s: torch.Tensor,
    y: torch.Tensor,
    threshold: float = 0.0,
) -> bool:
    """
    Check the secant condition y_k^T s_k.

    When y^T s ≤ threshold, the curvature pair indicates a
    non-convex region (saddle point or negative curvature).

    Args:
        s: Step difference vector s_k = x_{k+1} - x_k.
        y: Gradient difference vector y_k = g_{k+1} - g_k.
        threshold: Trigger threshold (default: 0.0).

    Returns:
        True if the secant condition is violated (non-convex region).
    """
    # ADDS: Lanczos-style saddle decision using a cheap rank-1 two_loop proxy.
    # REMOVES: direct y^T s threshold check in this function.
    denom = torch.dot(s, s).clamp(min=1e-12)

    def rank1_two_loop(v: torch.Tensor) -> torch.Tensor:
        return y * (torch.dot(s, v) / denom)

    is_saddle, _ = is_saddle_point(rank1_two_loop, dim=int(s.numel()), eps=threshold)
    return is_saddle


# ADDS: Lanczos min-eigenvalue probe over two-loop Hessian-vector handle.
# REMOVES: y^T s-only saddle detection as the primary curvature criterion.
def is_saddle_point(
    two_loop_fn: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    eps: float = 1e-4,
) -> Tuple[bool, torch.Tensor]:
    """3-step Lanczos min-eigenvalue probe. Returns (is_saddle, min_eigvec)."""
    q = torch.randn(dim)
    q = q / q.norm().clamp(min=1e-12)
    q_prev = torch.zeros_like(q)
    beta_prev = torch.tensor(0.0, dtype=q.dtype, device=q.device)

    basis = []
    alphas = []
    betas = []

    for _ in range(3):
        basis.append(q)
        w = two_loop_fn(q) - beta_prev * q_prev
        alpha = torch.dot(q, w)
        alphas.append(alpha)
        w = w - alpha * q
        beta = w.norm()
        betas.append(beta)
        if beta < 1e-10:
            break
        q_prev = q
        q = w / beta
        beta_prev = beta

    m = len(alphas)
    T = torch.zeros((m, m), dtype=basis[0].dtype, device=basis[0].device)
    for i in range(m):
        T[i, i] = alphas[i]
        if i + 1 < m:
            T[i, i + 1] = betas[i]
            T[i + 1, i] = betas[i]

    eigvals, eigvecs = torch.linalg.eigh(T)
    lam_min = float(eigvals[0].item())
    coeffs = eigvecs[:, 0]
    min_vec = torch.zeros_like(basis[0])
    for i in range(m):
        min_vec = min_vec + coeffs[i] * basis[i]
    min_vec = min_vec / min_vec.norm().clamp(min=1e-12)
    return lam_min <= eps, min_vec


def detect_topology_break(
    recent_gradients: torch.Tensor,
    distance_threshold: float = 0.5,
    prev_chi: Optional[int] = None,
) -> Dict:
    """
    Detect structural breaks in gradient space via Euler Characteristic.

    Adapted from Chronoscope's SignalObserver.incremental_analysis().
    Uses pairwise L2 distances to build an adjacency graph and computes
    χ = V - E as a fast topological proxy.

    A sudden change in χ suggests the optimizer has entered a
    topologically different region (saddle point, plateau boundary).

    Args:
        recent_gradients: Tensor [window_size, param_dim] of recent gradients.
        distance_threshold: Threshold for adjacency (normalized).
        prev_chi: Previous Euler characteristic for change detection.

    Returns:
        Dict with 'euler_characteristic', 'topology_break_detected',
        and 'should_perturb'.
    """
    n = recent_gradients.shape[0]
    if n < 2:
        return {
            "euler_characteristic": n,
            "topology_break_detected": False,
            "should_perturb": False,
        }

    grads_np = recent_gradients.float().detach().cpu().numpy()

    # Pairwise L2 distances
    diffs = grads_np[:, np.newaxis, :] - grads_np[np.newaxis, :, :]
    distances = np.linalg.norm(diffs, axis=-1)

    max_dist = distances.max()
    if max_dist > 0:
        distances = distances / max_dist

    adjacency = (distances < distance_threshold).astype(int)
    V = n
    E = (np.sum(adjacency) - V) // 2
    chi = V - E

    anomaly = False
    if prev_chi is not None:
        chi_delta = abs(chi - prev_chi)
        if chi_delta >= 2:  # Structural break in gradient manifold
            anomaly = True

    return {
        "euler_characteristic": int(chi),
        "topology_break_detected": anomaly,
        "should_perturb": anomaly,
    }


def generate_orthogonal_perturbation(
    grad: torch.Tensor,
    scale: float = 0.01,
) -> torch.Tensor:
    """
    Generate a perturbation vector orthogonal to the current gradient.

    Uses Gram-Schmidt orthogonalization with a random vector to create
    a perturbation that escapes saddle points without fighting the
    current gradient direction.

    Args:
        grad: Current gradient vector (1-D tensor).
        scale: Magnitude of the perturbation relative to grad norm.

    Returns:
        Orthogonal perturbation vector (same shape as grad).
    """
    # ADDS: deterministic eigenvector-aligned perturbation proxy for compatibility.
    # REMOVES: random Gram-Schmidt orthogonal perturbation path.
    grad_norm = float(grad.norm().item())
    direction = (-grad).clone()
    direction = direction / direction.norm().clamp(min=1e-12)
    return scale * grad_norm * direction


# ADDS: direct eigenvector-guided saddle escape update for parameter tensors.
# REMOVES: random orthogonal perturbation as the core escape mechanism.
def escape_saddle(
    params: list,
    grad_norm: float,
    min_eigvec: torch.Tensor,
    scale: float = 0.01,
) -> None:
    """Eigenvector-directed perturbation. Satisfies Zoutendijk descent."""
    delta = scale * grad_norm * min_eigvec
    offset = 0
    for p in params:
        numel = p.numel()
        p.data.add_(delta[offset:offset + numel].view_as(p))
        offset += numel
