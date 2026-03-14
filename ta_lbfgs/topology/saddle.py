"""
Saddle-Point Detection and Evasion.

Two-pronged detection:
1. Secant condition check (y_k^T s_k ≤ 0)
2. Euler characteristic topology proxy (from Chronoscope)

Plus orthogonal perturbation injection for saddle escape.
"""

import torch
import numpy as np
from typing import Dict, Optional


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
    ys = y.dot(s).item()
    return ys <= threshold


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
    grad_norm = grad.norm()
    if grad_norm < 1e-12:
        # Gradient is near zero — use random direction
        perturbation = torch.randn_like(grad)
        return perturbation * scale

    # Generate random vector
    random_vec = torch.randn_like(grad)

    # Gram-Schmidt: project out the gradient component
    # v_perp = random - (random · grad_hat) * grad_hat
    grad_hat = grad / grad_norm
    projection = random_vec.dot(grad_hat)
    v_perp = random_vec - projection * grad_hat

    # Normalize and scale
    v_perp_norm = v_perp.norm()
    if v_perp_norm < 1e-12:
        # Extremely unlikely: random_vec parallel to grad
        perturbation = torch.randn_like(grad)
        return perturbation * scale

    perturbation = v_perp / v_perp_norm * grad_norm * scale

    return perturbation
