"""
Condition Number Estimation via SVD.

Adapted from Chronoscope's SignalObserver.svd_compress() pipeline.
Estimates the condition number κ of a layer's gradient/Hessian
approximation using truncated SVD with NaN sanitization.
"""

import numpy as np
import torch
from typing import Optional


def estimate_condition_number(
    matrix: torch.Tensor,
    n_components: int = 8,
) -> float:
    """
    Estimate condition number of a matrix via SVD.

    Computes κ = σ_max / σ_min from the top singular values.
    Handles degenerate cases (zero-variance columns, NaN, Inf).

    Adapted from Chronoscope's observer.py SVD pipeline.

    Args:
        matrix: 2D tensor [rows, cols] (e.g., gradient outer product
                or stacked gradient history).
        n_components: Number of singular values to retain.

    Returns:
        Condition number κ (float). Returns 1.0 for degenerate cases.
    """
    X = matrix.detach().cpu().numpy()
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Filter zero-variance columns
    active_cols = np.where(X.std(axis=0) > 1e-6)[0]
    if len(active_cols) == 0:
        return 1.0  # Perfectly conditioned (degenerate)

    X_active = X[:, active_cols]
    X_centered = X_active - X_active.mean(axis=0)

    try:
        _, S, _ = np.linalg.svd(X_centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return 1.0

    k = min(n_components, len(S))
    S_top = S[:k]
    sigma_min = S_top[-1] if S_top[-1] > 1e-12 else 1e-12
    kappa = float(S_top[0] / sigma_min)

    return kappa


def estimate_condition_from_grad_history(
    grad_history: list,
    n_components: int = 8,
) -> float:
    """
    Estimate condition number from a history of gradient vectors.

    Stacks gradient vectors into a matrix and computes κ via SVD.

    Args:
        grad_history: List of 1-D gradient tensors.
        n_components: SVD truncation.

    Returns:
        Condition number κ.
    """
    if len(grad_history) < 2:
        return 1.0

    # Stack into [n_steps, param_dim] matrix
    matrix = torch.stack(grad_history)
    return estimate_condition_number(matrix, n_components)
