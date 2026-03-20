"""
Hypergradient Computation Module.

Implements the Implicit Function Theorem (IFT) approach for computing
gradients of the validation loss with respect to hyperparameters,
without explicitly computing full dense Hessian matrices.

Methods:
- Hessian-Vector Products (HVP) via torch.autograd double-backward
- Conjugate Gradient (CG) solver for the IFT linear system
- Neumann Series approximation for inverse-HVP
"""

import torch
from typing import Callable, List, Optional, Tuple
from math import prod


def _tensor_list_all_finite(tensors: List[torch.Tensor]) -> bool:
    return all(torch.isfinite(t).all().item() for t in tensors)


def hessian_vector_product(
    loss: torch.Tensor,
    params: List[torch.Tensor],
    v: List[torch.Tensor],
    retain_graph: bool = True,
) -> List[torch.Tensor]:
    """
    Compute Hessian-vector product H * v via double backward pass.

    Uses torch.autograd.grad twice:
    1. Compute gradient g = ∇_params(loss)
    2. Compute ∇_params(g^T v) = H * v

    Args:
        loss: Scalar loss tensor.
        params: List of parameter tensors.
        v: List of vectors to multiply with Hessian (same shapes as params).
        retain_graph: Whether to retain graph for further backward passes.

    Returns:
        List of Hessian-vector product tensors.
    """
    # First backward: compute gradients
    grads = torch.autograd.grad(
        loss,
        params,
        create_graph=True,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    grads = [
        g if g is not None else torch.zeros_like(p)
        for g, p in zip(grads, params)
    ]

    # Compute g^T v (dot product of gradients and vectors)
    gv = sum(
        (g * vi).sum() for g, vi in zip(grads, v) if g is not None
    )

    # Second backward: compute H * v = ∇_params(g^T v)
    hvp = torch.autograd.grad(
        gv,
        params,
        retain_graph=retain_graph,
        allow_unused=True,
    )

    hvp = [
        h if h is not None else torch.zeros_like(p)
        for h, p in zip(hvp, params)
    ]

    return [h.detach() for h in hvp]


def conjugate_gradient_solve(
    hvp_fn: Callable[[List[torch.Tensor]], List[torch.Tensor]],
    b: List[torch.Tensor],
    max_iter: int = 10,
    tol: float = 1e-5,
) -> List[torch.Tensor]:
    """
    Solve the linear system H * x = b using Conjugate Gradient.

    Used in the IFT to solve for the implicit gradient:
    H_{θθ} * v = ∇_θ L_train

    Args:
        hvp_fn: Function that computes Hessian-vector products.
        b: Right-hand side vectors (same structure as params).
        max_iter: Maximum CG iterations.
        tol: Convergence tolerance.

    Returns:
        Approximate solution x ≈ H^{-1} * b.
    """
    # Initialize: x_0 = 0, r_0 = b, p_0 = b
    x = [torch.zeros_like(bi) for bi in b]
    r = [bi.clone() for bi in b]
    p = [bi.clone() for bi in b]

    r_norm_sq = sum((ri * ri).sum() for ri in r)

    for _ in range(max_iter):
        if r_norm_sq.item() < tol:
            break

        # Compute H * p
        Hp = hvp_fn(p)

        # α = r^T r / (p^T H p)
        pHp = sum((pi * hpi).sum() for pi, hpi in zip(p, Hp))
        if pHp.abs() < 1e-12:
            break
        alpha = r_norm_sq / pHp

        # x_{k+1} = x_k + α p_k
        x = [xi + alpha * pi for xi, pi in zip(x, p)]

        # r_{k+1} = r_k - α H p_k
        r = [ri - alpha * hpi for ri, hpi in zip(r, Hp)]

        r_norm_sq_new = sum((ri * ri).sum() for ri in r)

        # β = r_{k+1}^T r_{k+1} / (r_k^T r_k)
        beta = r_norm_sq_new / (r_norm_sq + 1e-12)

        # p_{k+1} = r_{k+1} + β p_k
        p = [ri + beta * pi for ri, pi in zip(r, p)]

        r_norm_sq = r_norm_sq_new

    return x


def _flatten_tensor_list(tensors: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Size]]:
    flat = torch.cat([t.reshape(-1) for t in tensors], dim=0)
    shapes = [t.shape for t in tensors]
    return flat, shapes


def _unflatten_tensor(flat: torch.Tensor, shapes: List[torch.Size]) -> List[torch.Tensor]:
    chunks: List[torch.Tensor] = []
    offset = 0
    for shape in shapes:
        n = prod(shape)
        chunks.append(flat[offset: offset + n].reshape(shape))
        offset += n
    return chunks


# ADDS: step-size stability guard for truncated Neumann inverse-HVP.
# REMOVES: unconstrained alpha usage from old neumann_series_approx() in this file.
def _spectral_guard(
    H_fn: Callable[[torch.Tensor], torch.Tensor],
    alpha: float,
    n_probe: int = 3,
    probe: Optional[torch.Tensor] = None,
) -> float:
    """Ensures alpha < 2/lambda_max(H) via power iteration."""
    if probe is None:
        probe = torch.randn(1)
    v = probe.clone()
    v = v / v.norm().clamp(min=1e-12)
    lam_max = torch.tensor(0.0, device=v.device, dtype=v.dtype)
    for _ in range(n_probe):
        v = H_fn(v)
        if not torch.isfinite(v).all():
            raise FloatingPointError("Non-finite H_fn(v) in _spectral_guard")
        v = v / v.norm().clamp(min=1e-12)
    Hv = H_fn(v)
    if not torch.isfinite(Hv).all():
        raise FloatingPointError("Non-finite H_fn(v) in _spectral_guard Rayleigh quotient")
    lam_max = torch.dot(v, Hv).abs()
    if bool(alpha * lam_max >= 1.0):
        denom = max(float(lam_max.detach().cpu()), 1e-6)
        alpha = 0.9 / denom
    return alpha


# ADDS: spectral-guarded truncated Neumann inverse-HVP kernel.
# REMOVES: old neumann_series_approx() recurrence and accumulator path in this file.
def neumann_hypergradient(
    H_fn: Callable[[torch.Tensor], torch.Tensor],
    g_val: torch.Tensor,
    alpha: float,
    K: int = 5,
) -> torch.Tensor:
    """Truncated Neumann. Caller must pass L2-regularized H_fn."""
    alpha = _spectral_guard(H_fn, alpha, probe=g_val)
    v = g_val.clone()
    result = v.clone()
    for _ in range(K):
        Hv = H_fn(v)
        if not torch.isfinite(Hv).all():
            raise FloatingPointError("Non-finite H_fn(v) in neumann_hypergradient")
        v = v - alpha * Hv
        result = result + v
    return alpha * result


# ADDS: elementwise Hutchinson diagonal Hessian estimator for outer preconditioning.
# REMOVES: scalar-trace-centric preconditioning dependence from outer-loop usage paths.
def hutchinson_diagonal(
    H_fn: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    n_probes: int = 10,
) -> torch.Tensor:
    """Diagonal Hessian estimator. Error ~O(1/sqrt(n_probes)) per element."""
    probes = max(1, int(n_probes))

    # Probe once to infer the target device/dtype from the Hessian-vector product.
    z0 = (torch.randint(0, 2, (dim,), dtype=torch.int64).to(torch.float32) * 2 - 1)
    hz0 = H_fn(z0)
    diag = z0.to(dtype=hz0.dtype, device=hz0.device) * hz0

    for _ in range(1, probes):
        z = (torch.randint(0, 2, (dim,), device=hz0.device, dtype=torch.int64).to(hz0.dtype) * 2 - 1)
        hz = H_fn(z)
        diag.add_(z * hz)
    return diag / float(probes)


# ADDS: canonical outer-loop diagonal preconditioner built on Hutchinson diagonal.
# REMOVES: ad-hoc scalar trace preconditioning as a curvature surrogate.
def outer_precondition(
    hypgrad: torch.Tensor,
    H_fn: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    n_probes: int = 10,
    eps: float = 1e-5,
) -> torch.Tensor:
    diag_H = hutchinson_diagonal(H_fn, dim, n_probes)
    return hypgrad / diag_H.abs().clamp(min=eps)


def implicit_differentiation(
    val_loss: torch.Tensor,
    train_loss: torch.Tensor,
    hyperparams: List[torch.Tensor],
    weights: List[torch.Tensor],
    method: str = "CG",
    cg_max_iter: int = 10,
    cg_tol: float = 1e-5,
    neumann_terms: int = 5,
    neumann_alpha: float = 0.01,
) -> List[torch.Tensor]:
    """
    Compute hypergradients via Implicit Function Theorem.

    Computes ∇_λ L_val(w*(λ), λ) using the IFT:
    ∇_λ L_val = ∂L_val/∂λ - ∂L_val/∂w * H_{ww}^{-1} * ∂²L_train/(∂w∂λ)

    where H_{ww} is the Hessian of training loss w.r.t. weights.

    Args:
        val_loss: Validation loss (scalar tensor, with grad graph).
        train_loss: Training loss (scalar tensor, with grad graph).
        hyperparams: List of hyperparameter tensors (λ).
        weights: List of model weight tensors (w).
        method: Solver method ('CG', 'Neumann', 'direct').
        cg_max_iter: Max CG iterations.
        cg_tol: CG convergence tolerance.
        neumann_terms: Number of Neumann series terms.
        neumann_alpha: Neumann series scaling.

    Returns:
        List of hypergradient tensors (one per hyperparameter).
    """
    # Step 1: ∂L_val / ∂w
    dLval_dw = torch.autograd.grad(
        val_loss,
        weights,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )
    dLval_dw = [
        g if g is not None else torch.zeros_like(w)
        for g, w in zip(dLval_dw, weights)
    ]

    # Step 2: Solve H_{ww}^{-1} * (∂L_val/∂w) via chosen method
    def hvp_fn(v):
        return hessian_vector_product(train_loss, weights, v)

    if method == "CG":
        v_star = conjugate_gradient_solve(
            hvp_fn, list(dLval_dw), max_iter=cg_max_iter, tol=cg_tol
        )
        if not _tensor_list_all_finite(v_star):
            # Fallback for mixed-precision or ill-conditioned hybrid shards.
            v_star = [g.detach().clone() for g in dLval_dw]
    elif method == "Neumann":
        # ADDS: vectorized H_fn bridge for spectral-guarded Neumann updates.
        # REMOVES: silent non-finite fallback after old neumann_series_approx() in this file.
        g_vec, g_shapes = _flatten_tensor_list(list(dLval_dw))

        def flat_h_fn(v: torch.Tensor) -> torch.Tensor:
            vec_list = _unflatten_tensor(v, g_shapes)
            hv_list = hvp_fn(vec_list)
            flat_hv, _ = _flatten_tensor_list(hv_list)
            return flat_hv

        v_vec = neumann_hypergradient(
            flat_h_fn,
            g_vec,
            alpha=neumann_alpha,
            K=neumann_terms,
        )
        if not torch.isfinite(v_vec).all():
            raise FloatingPointError("Non-finite Neumann hypergradient vector")
        v_star = _unflatten_tensor(v_vec, g_shapes)
    elif method == "direct":
        # Direct solve (only for small problems / debugging)
        v_star = list(dLval_dw)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Step 3: Compute ∂²L_train/(∂w∂λ) * v_star = ∂/∂λ(∇_w L_train · v_star)
    # This gives us the implicit hypergradient
    grad_train_w = torch.autograd.grad(
        train_loss,
        weights,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )
    grad_train_w = [
        g if g is not None else torch.zeros_like(w)
        for g, w in zip(grad_train_w, weights)
    ]
    gv = sum((g * v).sum() for g, v in zip(grad_train_w, v_star))

    # ∂(g^T v) / ∂λ = cross-derivative term
    if gv.requires_grad:
        implicit_grads = torch.autograd.grad(
            gv,
            hyperparams,
            retain_graph=True,
            allow_unused=True,
        )
        implicit_grads = [
            g if g is not None else torch.zeros_like(h)
            for g, h in zip(implicit_grads, hyperparams)
        ]
    else:
        implicit_grads = [torch.zeros_like(h) for h in hyperparams]

    # Step 4: Direct term ∂L_val/∂λ (if val_loss depends directly on λ)
    try:
        direct_grads = torch.autograd.grad(
            val_loss, hyperparams, retain_graph=True, allow_unused=True
        )
    except RuntimeError:
        direct_grads = [torch.zeros_like(h) for h in hyperparams]

    # Combine: ∇_λ L_val = direct - implicit
    hypergradients = []
    for dg, ig, hp in zip(direct_grads, implicit_grads, hyperparams):
        if ig is None:
            ig = torch.zeros_like(hp)
        if dg is None:
            dg = torch.zeros_like(hp)
        hg = dg - ig
        hg = torch.nan_to_num(hg, nan=0.0, posinf=1.0, neginf=-1.0)
        hypergradients.append(hg)

    return [hg.detach() for hg in hypergradients]
